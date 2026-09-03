"""Export the step-by-step benchmark ground-truth document.

Writes ``docs/benchmark_ground_truth.md``: a human-readable, step-by-step
derivation of every benchmark item, from the source corpora through seed
selection and capability extraction to the four controlled modality variants
and their gold labels. The document is generated from the same code path that
builds the benchmark (``eval_utils.source_statement``,
``eval_utils.build_benchmark_items``, ``eval_utils.WEAK_MODALITY_PROBE_TEMPLATES``)
and from the shipped processed tables, so it cannot drift from the pipeline.

Run:

    .venv/bin/python scripts/export_benchmark_ground_truth.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import eval_utils as eu
except ModuleNotFoundError:  # pragma: no cover
    from scripts import eval_utils as eu


DOC_HEADER = """# Benchmark Ground Truth

This document derives every benchmark item step by step. It is *generated* by
`scripts/export_benchmark_ground_truth.py` from the same template code that
builds `data/processed/benchmark_items*.csv`, so what you read here is what the
models received — not a hand-written paraphrase of it. Regenerate with:

```bash
.venv/bin/python scripts/export_benchmark_ground_truth.py
```

Each cell of the design (dataset x mandatory keyword) holds 180 reviewed seeds
x 4 modality conditions = 720 items; there are four cells (`nice`/`mlm_tapt`
datasets x `MUST`/`SHALL` keywords).
"""




def _seed_row(rows: list[dict[str, Any]], seed_id: str) -> dict[str, Any]:
    for row in rows:
        if str(row.get("seed_id")) == seed_id:
            return row
    raise ValueError(f"Seed {seed_id!r} not found in the selected-seeds table.")


def _dataset_section(
    title: str,
    seeds_path: Path,
    items_paths: dict[str, Path],
    worked_seed_id: str | None,
    review_rows_path: Path,
) -> str:
    seeds = eu.read_csv_rows(seeds_path)
    review_rows = len(eu.read_csv_rows(review_rows_path)) if review_rows_path.exists() else 0
    items = eu.read_csv_rows(items_paths["must"])
    shall_items = eu.read_csv_rows(items_paths.get("shall")) if items_paths.get("shall") else []
    worked_seed_id = worked_seed_id or str(seeds[0]["seed_id"])
    seed = _seed_row(seeds, worked_seed_id)
    worked = [row for row in items if str(row.get("seed_id")) == worked_seed_id]
    worked.sort(key=lambda row: eu.ORDINAL_STRENGTH[str(row["source_modality"])])
    shall_row = next(
        (
            row
            for row in shall_items
            if str(row.get("seed_id")) == worked_seed_id
            and str(row.get("item_id", "")).endswith("_mandatory")
        ),
        None,
    )

    lines = [f"## {title}", ""]
    lines.append(
        f"Selected seeds: **{len(seeds)}** (screened from {review_rows} corpus rows in "
        f"`{review_rows_path.name}`; the same review table records the inclusion/exclusion "
        f"decision for every candidate seed). Items per `MUST` cell: **{len(items)}** "
        f"({len(items) // len(seeds)} conditions per seed)."
    )
    lines.append("")

    # Step 1: the source requirement and the extracted capability.
    lines.append(f"### Step 1 — Source requirement and capability ({worked_seed_id})")
    lines.append("")
    lines.append("The seed row in `data/processed/seeds_selected*.csv` records the original "
                 "corpus requirement and the reviewed capability clause:")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| `seed_id` | `{worked_seed_id}` |")
    lines.append(f"| `source_dataset` | `{seed.get('source_dataset', '')}` |")
    lines.append("| `original_requirement` | " + _inline(str(seed.get("original_requirement", ""))) + " |")
    lines.append("| `capability_text_final` | " + _inline(str(seed.get("capability_text_final", ""))) + " |")
    lines.append("")
    lines.append("The capability clause is the *content* that must survive every transformation "
                 "unchanged; only its modal force varies.")
    lines.append("")

    # Step 2: the four fixed templates applied to that capability.
    lines.append("### Step 2 — The four controlled variants")
    lines.append("")
    lines.append("Every seed is rendered through the four fixed templates below (verbatim from "
                 "`eval_utils.source_statement`); nothing else in the sentence changes:")
    lines.append("")
    template_rows = eu.main_modality_template_rows(str(seed.get("capability_text_final", "")))
    lines.append(eu.markdown_table(template_rows, eu.MAIN_MODALITY_TEMPLATE_INVENTORY_FIELDS))
    lines.append("")

    # Step 3: the resulting items with gold labels.
    lines.append("### Step 3 — The resulting items and their gold labels")
    lines.append("")
    lines.append("| `item_id` | Source statement (what the model sees) | Task 1 gold | Task 2 gold | Ordinal |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in worked:
        lines.append(
            "| `{item_id}` | {statement} | `{t1}` (yes={yes}) | `{t2}` | {ordinal} |".format(
                item_id=row["item_id"],
                statement=_inline(row["source_statement"]),
                t1=row["task1_gold_decision"],
                yes=row["task1_gold_yes"],
                t2=row["task2_gold_modality"],
                ordinal=row["ordinal_strength"],
            )
        )
    if shall_row:
        lines.append(
            "| `{item_id}` | {statement} | `{t1}` (yes={yes}) | `{t2}` | {ordinal} |".format(
                item_id=f"{shall_row['item_id']} (SHALL cell)",
                statement=_inline(shall_row["source_statement"]),
                t1=shall_row["task1_gold_decision"],
                yes=shall_row["task1_gold_yes"],
                t2=shall_row["task2_gold_modality"],
                ordinal=shall_row["ordinal_strength"],
            )
        )
    lines.append("")
    lines.append("Gold labels are **structural**, not judged per item: the gold modality is the "
                 "condition the template encodes, Task 1's gold entailment decision is `yes` "
                 "iff the condition is `mandatory`, and the ordinal strength is fixed per "
                 "condition (`mandatory`=3, `recommended`=2, `optional`=1, `nice_to_have`=0). "
                 "The `SHALL` row above is the robustness variant: identical to the mandatory "
                 "condition except the keyword `MUST` is swapped for `SHALL`.")
    lines.append("")

    # Step 4: weak-intent phrasing probes for the same seed.
    lines.append("### Step 4 — Weak-intent phrasing probes (same seed)")
    lines.append("")
    lines.append("The benchmark's weak condition uses the `useful_if` template. To check that "
                 "results are not tied to that single surface form, the same capability is also "
                 "rendered through three alternative weak phrasings "
                 "(`eval_utils.WEAK_MODALITY_PROBE_TEMPLATES`); all keep the gold label "
                 "`nice_to_have`:")
    lines.append("")
    capability = eu.capability_clause(str(seed.get("capability_text_final", "")))
    probe_rows = [
        {
            "template_id": f"probe_{template['template_id']}",
            "source statement": template["source_template"].format(capability=capability),
            "gold modality": "nice_to_have",
        }
        for template in eu.WEAK_MODALITY_PROBE_TEMPLATES
    ]
    lines.append(eu.markdown_table(probe_rows, ["template_id", "source statement", "gold modality"]))
    lines.append("")
    return "\n".join(lines)


def _inline(text: str) -> str:
    cleaned = " ".join(str(text).split())
    # The seed tables wrap some cells in single quotes; strip one pair.
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "'\"":
        cleaned = cleaned[1:-1].strip()
    return "`" + cleaned + "`" if cleaned else ""


def _doc_link(root: Path, path: Path) -> str:
    relative = path.relative_to(root)
    return f"[`{relative}`](../{relative})"


def build_document(root: Path, nice_seed: str | None, mlm_seed: str | None) -> str:
    processed = root / "data/processed"
    sections = [
        DOC_HEADER,
        _dataset_section(
            "Dataset `nice` (NICE/PROMISE seeds, `MUST` + `SHALL`)",
            processed / "seeds_selected.csv",
            {
                "must": processed / "benchmark_items.csv",
                "shall": processed / "benchmark_items_shall.csv",
            },
            nice_seed,
            processed / "seeds_review.csv",
        ),
        _dataset_section(
            "Dataset `mlm_tapt` (`limsc/mlm-tapt-requirements` seeds, `MUST` + `SHALL`)",
            processed / "seeds_selected_mlm_tapt.csv",
            {
                "must": processed / "benchmark_items_mlm_tapt.csv",
                "shall": processed / "benchmark_items_mlm_tapt_shall.csv",
            },
            mlm_seed,
            processed / "seeds_review_mlm_tapt.csv",
        ),
        _validation_section(root),
    ]
    return "\n\n".join(section.rstrip("\n") for section in sections) + "\n"


def _validation_section(root: Path) -> str:
    docs = root / "docs"
    construct_rows = eu.read_csv_rows(docs / "weak_modality_construct_review.csv")
    reviewer_roles = sorted({str(row.get("reviewer_role", "")) for row in construct_rows})
    seed_reviews = [
        ("`nice`", root / "data/processed/seeds_review.csv"),
        ("`mlm_tapt`", root / "data/processed/seeds_review_mlm_tapt.csv"),
    ]
    statement_reviews = [
        ("`nice`", "`MUST`", root / "outputs/benchmark_statements_review.csv"),
        ("`nice`", "`SHALL`", root / "outputs/benchmark_statements_review_shall.csv"),
        ("`mlm_tapt`", "`MUST`", root / "outputs/benchmark_statements_review_mlm_tapt.csv"),
        (
            "`mlm_tapt`",
            "`SHALL`",
            root / "outputs/benchmark_statements_review_mlm_tapt_shall.csv",
        ),
    ]
    manifests = [
        ("`nice`", root / "outputs/benchmark_manifest.json"),
        ("`mlm_tapt`", root / "outputs/benchmark_manifest_mlm_tapt.json"),
    ]
    lines = [
        "## Validation trail",
        "",
        "Every derivation step above is backed by a tracked review record:",
        "",
        "1. **Seed review.** Both source datasets have per-candidate inclusion/exclusion "
        "decisions and final capability clauses:",
        "",
        "| Dataset | Review record | Candidate rows |",
        "| --- | --- | ---: |",
    ]
    for dataset, path in seed_reviews:
        lines.append(
            f"| {dataset} | {_doc_link(root, path)} | {len(eu.read_csv_rows(path))} |"
        )
    lines.extend(
        [
            "",
            "2. **Statement review.** Every dataset/mandatory-keyword cell has its own "
            "review table:",
            "",
            "| Dataset | Mandatory keyword | Review record | Reviewed statement rows |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for dataset, keyword, path in statement_reviews:
        lines.append(
            f"| {dataset} | {keyword} | {_doc_link(root, path)} | "
            f"{len(eu.read_csv_rows(path))} |"
        )
    lines.extend(
        [
            "",
            f"3. **Weak-template construct review.** `docs/weak_modality_construct_review.csv` — "
            f"{len(construct_rows)} rows over the four weak templates; current reviewer roles: "
            f"{', '.join(f'`{role}`' for role in reviewer_roles) or 'none'}. Until the pending "
            "human sign-off ([`TODO.md`](../TODO.md) section D), weak-intent claims carry that "
            "caveat.",
            "",
            "4. **File integrity.** Each dataset manifest records sha256 digests and row "
            "counts for its seed and benchmark tables:",
            "",
            "| Dataset | Integrity manifest | Artifact entries |",
            "| --- | --- | ---: |",
        ]
    )
    for dataset, path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        lines.append(
            f"| {dataset} | {_doc_link(root, path)} | "
            f"{len(payload.get('artifacts', []))} |"
        )
    lines.extend(
        [
            "",
            "The benchmark construction itself is a pure function of the reviewed seed table: "
            "`eval_utils.build_benchmark_items` renders each seed through "
            "`eval_utils.source_statement`; no manual edits happen between the two.",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description="Export the step-by-step benchmark ground-truth document."
    )
    parser.add_argument("--output", type=Path, default=Path("docs/benchmark_ground_truth.md"))
    parser.add_argument("--nice-seed", default="S0001", help="Worked-example seed for the nice dataset.")
    parser.add_argument("--mlm-seed", default=None, help="Worked-example seed for mlm_tapt (default: first).")
    args = parser.parse_args(argv)

    root = eu.project_root()
    document = build_document(root, args.nice_seed, args.mlm_seed)
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    try:
        location = str(output.relative_to(root))
    except ValueError:  # output written outside the repository
        location = str(output)
    print(f"wrote {location} ({len(document.splitlines())} lines)")
    return output


if __name__ == "__main__":
    main()
