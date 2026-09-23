"""Check the paper's result tables against the archived per-item scores, without MLX.

The full refresh (``rerun_all.py --only analysis --refresh-analysis``) rebuilds
every score and needs Apple Silicon for the embeddings. This check does not
re-embed anything. It reads the per-item score rows of the selected runs
(``outputs/evaluation_*/uq_scores.csv``, shipped in the Zenodo raw archive),
recomputes the counts behind Tables 4-6 per model and pooled with its own code,
and compares them with the tracked ``paper_per_model_rq_table.csv`` and
``paper_per_model_modality_pooled.csv``:

    .venv/bin/python scripts/verify_paper_numbers.py
    .venv/bin/python scripts/verify_paper_numbers.py --raw-check

``--raw-check`` also re-applies the wording rules to the models' raw Task 2
answers (``data/processed/model_outputs_raw*``) and confirms that they reproduce
the strict and broad verdicts in the score rows. Confidence intervals and the
embedding classifier are not checked; they need the full refresh.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

try:
    import eval_utils as eu
    import export_paper_tables
except ModuleNotFoundError:  # pragma: no cover
    from scripts import eval_utils as eu, export_paper_tables

HIGH_CONFIDENCE = 0.90
ESCALATION_MODALITIES = {"recommended", "mandatory"}
AUROC_TOLERANCE = 1e-9
ALL = "all"
# Column prefixes of the rates in Tables 4-5; each has `_n` and `_denominator`.
RQ_RATES = (
    "task1_unsupported_acceptance_90",
    "task2_no_cue",
    "task2_strict_strengthening",
    "task2_broad_strengthening",
    "task2_weak_strict_strengthening",
    "task2_weak_strict_escalation",
    "task2_weak_strict_frame_only",
    "task2_strict_high_conf_90",
    "task2_strict_agreement",
    "task3_strict_flagged",
    "task3_strict_called_preserved",
)
RQ_AUROCS = ("task2_meaning_variation_auroc", "task2_verbalized_confidence_auroc")


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true"}


def number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def auroc(labels: list[int], scores: list[float]) -> float | None:
    """Mann-Whitney AUROC; tied scores count half. None without both classes."""
    positives = [s for label, s in zip(labels, scores, strict=True) if label]
    negatives = [s for label, s in zip(labels, scores, strict=True) if not label]
    if not positives or not negatives:
        return None
    ranked = sorted(scores)
    ranks: dict[float, float] = {}
    index = 0
    while index < len(ranked):
        end = index
        while end + 1 < len(ranked) and ranked[end + 1] == ranked[index]:
            end += 1
        ranks[ranked[index]] = (index + end) / 2 + 1
        index = end + 1
    rank_sum = sum(ranks[s] for s in positives)
    n_pos, n_neg = len(positives), len(negatives)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def load_score_rows(manifest: Path, root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for entry in eu.read_csv_rows(manifest):
        path = root / entry["analysis_dir"] / "uq_scores.csv"
        if not path.is_file():
            missing.append(str(path.relative_to(root)))
            continue
        for row in eu.read_csv_rows(path):
            # A Task 3 row carries its own run id, so rows are joined on the cell.
            row["model"] = entry["model"]
            row["cell"] = f"{entry['dataset_id']}/{entry['benchmark_variant']}"
            rows.append(row)
    if missing:
        raise SystemExit(
            f"missing per-item scores for {len(missing)} run(s), e.g. {missing[0]}. "
            "Unpack the Zenodo raw archive first (README, reproduction tier 2)."
        )
    return rows


def rq_counts(rows: Iterable[Mapping[str, str]]) -> dict[str, Any]:
    """Numerators, denominators and AUROCs of Tables 4-5 for one group of rows."""
    rows = list(rows)

    def select(task: str, method: str) -> list[Mapping[str, str]]:
        return [r for r in rows if r["task"] == task and r["uq_method"] == method]

    task1 = select("task1", "verbalized_confidence")
    task2 = select("task2", "verbalized_confidence")
    readable = [r for r in task2 if r["text_modality_parse_status"] == "ok"]
    strict = [r for r in readable if truthy(r["strict_text_overcommit"])]
    weak = [r for r in readable if r["source_modality"] == "nice_to_have"]
    weak_strict = [r for r in weak if truthy(r["strict_text_overcommit"])]
    key = lambda r: (r["model"], r["cell"], r["item_id"])  # noqa: E731
    consistency = {key(r): r for r in select("task2", "modality_consistency")}
    variation = {key(r): r for r in select("task2", eu.ACSE_PROXY_METHOD)}
    strict_keys = {key(r) for r in strict}
    task3 = [
        r
        for r in select("task3", "verbalized_confidence")
        if (r["model"], r["cell"], r["source_item_id"]) in strict_keys
    ]
    unsupported = [r for r in task1 if r["y_true"] in {"0", "0.0"}]
    complete = [
        consistency[key(r)]
        for r in strict
        if key(r) in consistency and truthy(consistency[key(r)]["stochastic_complete"])
    ]
    scored = [
        (r, number(variation[key(r)]["uncertainty_score"]))
        for r in readable
        if key(r) in variation
        and number(variation[key(r)]["uncertainty_score"]) is not None
    ]
    labels = [int(truthy(r["strict_text_overcommit"])) for r, _ in scored]

    def high(r: Mapping[str, str]) -> bool:
        return (number(r["confidence"]) or 0.0) >= HIGH_CONFIDENCE

    return {
        "task1_unsupported_acceptance_90": (
            sum(1 for r in unsupported if r["y_pred"] in {"1", "1.0"} and high(r)),
            len(unsupported),
        ),
        "task2_no_cue": (len(task2) - len(readable), len(task2)),
        "task2_strict_strengthening": (len(strict), len(readable)),
        "task2_broad_strengthening": (
            sum(1 for r in readable if truthy(r["text_overcommit"])),
            len(readable),
        ),
        "task2_weak_strict_strengthening": (len(weak_strict), len(weak)),
        "task2_weak_strict_escalation": (
            sum(1 for r in weak_strict if r["text_modality"] in ESCALATION_MODALITIES),
            len(weak),
        ),
        "task2_weak_strict_frame_only": (
            sum(1 for r in weak_strict if r["text_modality"] == "optional"),
            len(weak),
        ),
        "task2_strict_high_conf_90": (sum(1 for r in strict if high(r)), len(strict)),
        "task2_strict_agreement": (
            sum(1 for r in complete if number(r["uncertainty_score"]) == 0.0),
            len(complete),
        ),
        "task3_strict_flagged": (
            sum(1 for r in task3 if r["pred_relation"] == "strengthens"),
            len(task3),
        ),
        "task3_strict_called_preserved": (
            sum(1 for r in task3 if r["pred_relation"] == "preserves"),
            len(task3),
        ),
        "task2_meaning_variation_auroc": auroc(labels, [s for _, s in scored]),
        "task2_verbalized_confidence_auroc": auroc(
            labels, [number(r["uncertainty_score"]) or 0.0 for r, _ in scored]
        ),
    }


def modality_counts(rows: Iterable[Mapping[str, str]]) -> dict[str, dict[str, int]]:
    """Strict and broad n/N per source modality (Table 6)."""
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"strict": 0, "broad": 0, "denominator": 0}
    )
    for r in rows:
        if r["task"] != "task2" or r["uq_method"] != "verbalized_confidence":
            continue
        if r["text_modality_parse_status"] != "ok":
            continue
        cell = counts[r["source_modality"]]
        cell["denominator"] += 1
        cell["strict"] += truthy(r["strict_text_overcommit"])
        cell["broad"] += truthy(r["text_overcommit"])
    return counts


def compare(
    rows: list[dict[str, str]], rq_table: Path, modality_table: Path
) -> list[tuple[str, str, str, str, bool]]:
    """One (group, metric, recomputed, tracked, match) line per checked number."""
    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    groups = {**by_model, ALL: rows}
    tracked_rq = {
        r["model"]: r
        for r in eu.read_csv_rows(rq_table)
        if r["dataset"] == ALL and r["variant"] == ALL
    }
    tracked_modality = {
        (r["model"], r["source_modality"]): r
        for r in eu.read_csv_rows(modality_table)
        if r["dataset"] == ALL and r["variant"] == ALL
    }
    lines = []
    for group, group_rows in sorted(groups.items()):
        counts = rq_counts(group_rows)
        tracked = tracked_rq.get(group)
        if tracked is None:
            lines.append((group, "row", "present", "absent", False))
            continue
        for name in RQ_RATES:
            n, d = counts[name]
            want = f"{int(float(tracked[f'{name}_n']))}/{int(float(tracked[f'{name}_denominator']))}"
            lines.append((group, name, f"{n}/{d}", want, f"{n}/{d}" == want))
        for name in RQ_AUROCS:
            got, want = counts[name], number(tracked[name])
            ok = (got is None and want is None) or (
                got is not None
                and want is not None
                and abs(got - want) <= AUROC_TOLERANCE
            )
            lines.append(
                (
                    group,
                    name,
                    "n/a" if got is None else f"{got:.6f}",
                    "n/a" if want is None else f"{want:.6f}",
                    ok,
                )
            )
        for modality, cell in sorted(modality_counts(group_rows).items()):
            tracked_cell = tracked_modality.get((group, modality))
            for check in ("strict", "broad"):
                got = f"{cell[check]}/{cell['denominator']}"
                want = (
                    "absent"
                    if tracked_cell is None
                    else f"{int(float(tracked_cell[f'{check}_strengthening_n']))}/"
                    f"{int(float(tracked_cell[f'{check}_strengthening_denominator']))}"
                )
                lines.append((group, f"{check}:{modality}", got, want, got == want))
    return lines


def raw_check(
    rows: list[dict[str, str]], manifest: Path, root: Path
) -> tuple[int, list[str]]:
    """Re-apply the wording rules to the raw Task 2 answers; return checks, mismatches."""
    scored = {
        (r["run_id"], r["item_id"]): r
        for r in rows
        if r["task"] == "task2" and r["uq_method"] == "verbalized_confidence"
    }
    checked, mismatches = 0, []
    for entry in eu.read_csv_rows(manifest):
        path = eu.model_outputs_raw_path(
            root, entry["dataset_id"], entry["benchmark_variant"]
        )
        raw = eu.dedupe_raw_rows(
            export_paper_tables.stream_raw_rows(path, {entry["run_id"]})
        )
        for record in raw:
            if (
                record.get("task") != "task2"
                or record.get("sample_kind") != "deterministic"
            ):
                continue
            score = scored.get((record["run_id"], record["item_id"]))
            parsed = record.get("parsed_json") or {}
            if score is None or record.get("parse_status") != "ok":
                continue
            fields = eu.text_modality_fields(
                parsed.get("requirement", ""),
                record["source_modality"],
                str(parsed.get("modality", "")),
                float(parsed.get("confidence") or 0.0),
            )
            checked += 1
            for field in ("strict_text_overcommit", "text_overcommit"):
                if bool(fields[field]) != truthy(score[field]):
                    mismatches.append(f"{record['run_id']} {record['item_id']} {field}")
    return checked, mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/rerun/acse_selected_manifest.csv"),
    )
    parser.add_argument(
        "--rq-table", type=Path, default=Path("outputs/paper_per_model_rq_table.csv")
    )
    parser.add_argument(
        "--modality-table",
        type=Path,
        default=Path("outputs/paper_per_model_modality_pooled.csv"),
    )
    parser.add_argument("--raw-check", action="store_true")
    args = parser.parse_args()
    root = eu.project_root()
    resolve = lambda path: path if path.is_absolute() else root / path  # noqa: E731

    rows = load_score_rows(resolve(args.manifest), root)
    lines = compare(rows, resolve(args.rq_table), resolve(args.modality_table))
    failed = [line for line in lines if not line[4]]
    for group, metric, got, want, ok in lines:
        if not ok:
            print(
                f"MISMATCH {group:18} {metric:36} recomputed {got:>14}  tracked {want:>14}"
            )
    print(
        f"{len(lines) - len(failed)} of {len(lines)} table numbers match "
        f"({len({line[0] for line in lines})} groups: 9 models and the pooled row)."
    )
    if args.raw_check:
        checked, mismatches = raw_check(rows, resolve(args.manifest), root)
        for mismatch in mismatches[:20]:
            print(f"MISMATCH raw {mismatch}")
        print(
            f"raw check: {checked} Task 2 answers re-scored from the raw store, "
            f"{len(mismatches)} verdict mismatch(es)."
        )
        failed += mismatches
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
