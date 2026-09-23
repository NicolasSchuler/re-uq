"""Build the `pure` document-context ablation dataset (docs/context_ablation.md).

Two stages, mirroring what notebooks 00 and 01 do for the paper datasets but
without touching them:

    candidates  PURE XML -> data/processed/seeds_review_pure.csv
                (all eligible optional-marked requirements plus a deterministic
                sample of mandatory-marked ones; the reviewer edits `include`
                and `capability_text_final` in place)
    benchmark   reviewed seeds -> data/processed/benchmark_items_pure.csv,
                outputs/benchmark_statements_review_pure.{csv,md} and
                outputs/benchmark_manifest_pure.json

Only the MUST cell is built; the ablation never uses the SHALL variant.
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections import Counter
from pathlib import Path

try:
    import eval_utils as eu
except ModuleNotFoundError:  # pragma: no cover
    from scripts import eval_utils as eu


DATASET_ID = eu.DATASET_PURE
MANIFEST_PROMPTS = [
    "prompts/modality_extraction.txt",
    "prompts/modality_extraction_context.txt",
]


def seeds_review_path(root: Path) -> Path:
    return eu.artifact_path(root / "data/processed/seeds_review.csv", DATASET_ID)


def seeds_selected_path(root: Path) -> Path:
    return eu.artifact_path(root / "data/processed/seeds_selected.csv", DATASET_ID)


def benchmark_path(root: Path) -> Path:
    return eu.artifact_path(root / "data/processed/benchmark_items.csv", DATASET_ID)


def manifest_path(root: Path) -> Path:
    return eu.artifact_path(root / "outputs/benchmark_manifest.json", DATASET_ID)


def build_candidates(root: Path, *, overwrite: bool = False) -> dict[str, object]:
    config = eu.load_config()
    rows = eu.load_pure_requirement_rows(config)
    target_count = eu.dataset_target_seed_count(config, DATASET_ID)
    candidates = eu.make_pure_seed_candidates(
        rows, target_count=target_count, seed=int(config["project"]["seed"])
    )
    result = eu.write_csv_rows_if_changed(
        seeds_review_path(root),
        candidates,
        fieldnames=eu.seed_review_fields(DATASET_ID),
        overwrite=overwrite,
    )
    selected = [row for row in candidates if row["include"] == "yes"]
    markers = Counter(row["context_marker"] for row in selected)
    print(
        f"{result['status']}: {result['path']} ({len(candidates)} candidates, "
        f"{len(selected)} selected; markers {dict(sorted(markers.items()))})"
    )
    if result["candidate_path"]:
        print(
            f"existing review kept; differing candidate at {result['candidate_path']}"
        )
    return result


def capability_review_issues(seeds: list[dict[str, str]]) -> list[dict[str, str]]:
    """Mechanical red flags, not a substitute for source-based semantic review."""
    counts = Counter(
        eu.strip_final_punctuation(s["capability_text_final"]).lower() for s in seeds
    )
    issues = []
    for seed in seeds:
        capability = eu.strip_final_punctuation(seed["capability_text_final"])
        reasons = []
        if not capability:
            reasons.append("missing explicitly reviewed capability")
        if re.search(
            r"\b(?:shall|must|should|may|will|can|could|(?:is|are)\s+(?:required|recommended|expected)\s+to)\b",
            capability,
            re.I,
        ):
            reasons.append("residual modality")
        if counts[capability.lower()] > 1:
            reasons.append("duplicate capability; check distinct source content")
        if capability == eu.auto_capability_text(
            seed["original_requirement"]
        ) and capability != eu.auto_capability_text(
            seed["original_requirement"], preserve_subject=True
        ):
            reasons.append(
                "automatic extraction discarded subject or leading condition"
            )
        if reasons:
            issues.append({"seed_id": seed["seed_id"], "issues": "; ".join(reasons)})
    return issues


def require_reviewed_capabilities(
    root: Path, target_count: int
) -> list[dict[str, str]]:
    # Do not accept load_reviewed_seeds' generic fallback to auto suggestions.
    selected = [
        row
        for row in eu.read_csv_rows(seeds_review_path(root))
        if eu.is_truthy(row.get("include"))
    ]
    issues = capability_review_issues(selected)
    if issues:
        raise ValueError(
            "PURE capabilities require source-based review before building: "
            + "; ".join(f"{r['seed_id']}: {r['issues']}" for r in issues)
        )
    return eu.load_reviewed_seeds(
        seeds_review_path(root), target_count=target_count, strict=True
    )


def write_capability_review(root: Path) -> dict[str, Path]:
    """Join proposed revisions to intact source records; never approve them."""
    seeds = [
        r
        for r in eu.read_csv_rows(seeds_review_path(root))
        if eu.is_truthy(r.get("include"))
    ]
    proposals = eu.read_csv_rows(root / "docs/pure_capability_revisions.csv")
    by_id = {r["seed_id"]: r["proposed_capability"] for r in proposals}
    if len(by_id) != len(proposals) or set(by_id) != {r["seed_id"] for r in seeds}:
        raise ValueError(
            "Proposed PURE revisions must match selected seed IDs exactly once"
        )
    issues = {r["seed_id"]: r["issues"] for r in capability_review_issues(seeds)}
    proposed_issues = {
        r["seed_id"]: r["issues"]
        for r in capability_review_issues(
            [{**r, "capability_text_final": by_id[r["seed_id"]]} for r in seeds]
        )
    }
    rows = [
        {
            "seed_id": r["seed_id"],
            "source_corpus": r["source_corpus"],
            "context_requirement_id": r["context_requirement_id"],
            "original_requirement": r["original_requirement"],
            "current_capability": r["capability_text_final"],
            "proposed_capability": by_id[r["seed_id"]],
            "current_issues": issues.get(r["seed_id"], ""),
            "proposed_issues": proposed_issues.get(r["seed_id"], ""),
            "context_section": r["context_section"],
            "context_before": r["context_before"],
            "context_after": r["context_after"],
            "author_decision": "pending",
            "author_notes": "",
        }
        for r in seeds
    ]
    path = root / "outputs/pure_capability_revision_review.csv"
    # Never overwrite judgments entered into a previous review export.
    result = eu.write_csv_rows_if_changed(path, rows, overwrite=False)
    actual_path = Path(result.get("candidate_path") or result["path"])
    print(
        f"{result['status']}: {actual_path} ({len(rows)} proposals; author review pending)"
    )
    return {"review": actual_path}


def validate_seed_items(seeds: list[dict], target_count: int) -> list[dict]:
    issues = capability_review_issues(seeds)
    if issues:
        raise ValueError(f"PURE capability review issues: {issues}")
    items = eu.build_benchmark_items(seeds, passthrough_fields=eu.PURE_CONTEXT_FIELDS)
    expected = target_count * len(eu.MODALITIES)
    if len(items) != expected or len({row["item_id"] for row in items}) != expected:
        raise ValueError(
            f"Expected {expected} unique benchmark items, got {len(items)}."
        )
    missing_marker = [
        row["item_id"] for row in items if row["context_marker"] not in {"M", "O"}
    ]
    if missing_marker:
        raise ValueError(f"Items without an M/O marker: {missing_marker[:5]}")
    mismatches = [
        row["item_id"]
        for row in items
        if eu.requirement_text_modality_diagnostic(row["source_statement"])[
            "text_modality"
        ]
        != row["source_modality"]
    ]
    if mismatches:
        raise ValueError(f"Source-copy modality conflicts: {mismatches}")
    return items


def validated_inputs(root: Path) -> tuple[list[dict], list[dict]]:
    config = eu.load_config()
    target_count = eu.dataset_target_seed_count(config, DATASET_ID)
    seeds = require_reviewed_capabilities(root, target_count)
    return seeds, validate_seed_items(seeds, target_count)


def apply_capability_revisions(root: Path) -> dict[str, object]:
    """Apply the reviewed capability clauses, preserving old benchmark inputs.

    Every proposal needs an explicit ``accepted`` review decision. Original
    source/context fields and every raw response remain unchanged.
    """
    path = seeds_review_path(root)
    rows = eu.read_csv_rows(path)
    selected = [r for r in rows if eu.is_truthy(r.get("include"))]
    proposals = eu.read_csv_rows(root / "docs/pure_capability_revisions.csv")
    by_id = {r["seed_id"]: r for r in proposals}
    if len(by_id) != len(proposals) or set(by_id) != {r["seed_id"] for r in selected}:
        raise ValueError("Reviewed PURE revisions must match selected IDs exactly once")
    if any(r.get("review_decision") != "accepted" for r in proposals):
        raise ValueError(
            "Every PURE proposal needs an explicit accepted decision"
        )
    revised = [
        {**r, "capability_text_final": by_id[r["seed_id"]]["proposed_capability"]}
        if eu.is_truthy(r.get("include"))
        else r
        for r in rows
    ]
    config = eu.load_config()
    validate_seed_items(
        [r for r in revised if eu.is_truthy(r.get("include"))],
        eu.dataset_target_seed_count(config, DATASET_ID),
    )
    if revised == rows:
        print("PURE revisions already applied; no inputs changed")
        return {"status": "unchanged", "path": path}
    backup = root / "outputs/pure_before_capability_review"
    targets = [
        path,
        seeds_selected_path(root),
        benchmark_path(root),
        manifest_path(root),
        root / "outputs/benchmark_statements_review_pure.csv",
        root / "outputs/benchmark_statements_review_pure.md",
    ]
    # Validate all backup destinations before writing anything. Never replace
    # a different archive with later inputs under the same name.
    for source in targets:
        dest = backup / source.relative_to(root)
        if (
            source.exists()
            and dest.exists()
            and source.read_bytes() != dest.read_bytes()
        ):
            raise ValueError(f"Existing PURE backup differs: {dest}")
    for source in targets:
        if source.exists():
            dest = backup / source.relative_to(root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copy2(source, dest)
    eu.write_csv_rows(path, revised, fieldnames=list(rows[0]))
    print(
        f"Applied {len(proposals)} reviewed PURE capabilities; originals kept at {backup}"
    )
    return {"status": "applied", "path": path, "backup": backup}


def validate_existing_benchmark(root: Path) -> None:
    """Reject invalid or stale PURE inputs before any model request."""
    _, expected = validated_inputs(root)
    actual = eu.read_csv_rows(benchmark_path(root))
    by_id = {row["item_id"]: row for row in actual}
    if (
        len(actual) != len(expected)
        or len(by_id) != len(actual)
        or any(
            any(
                str(by_id.get(row["item_id"], {}).get(key, "")) != str(value)
                for key, value in row.items()
            )
            for row in expected
        )
    ):
        raise ValueError(
            "PURE benchmark differs from reviewed seeds; regenerate it before running"
        )


def build_benchmark(root: Path, *, overwrite: bool = False) -> dict[str, object]:
    seeds, items = validated_inputs(root)
    target_count = len(seeds)
    selected_result = eu.write_csv_rows_if_changed(
        seeds_selected_path(root),
        seeds,
        fieldnames=eu.seed_review_fields(DATASET_ID),
        overwrite=overwrite,
    )
    benchmark_result = eu.write_csv_rows_if_changed(
        benchmark_path(root), items, overwrite=overwrite
    )
    review_paths = eu.write_benchmark_statement_review(
        items, root / "outputs", suffix=eu.dataset_suffix(DATASET_ID)
    )
    marker_counts = Counter(row["context_marker"] for row in seeds)
    document_counts = Counter(row["source_corpus"] for row in seeds)
    manifest = eu.write_benchmark_manifest(
        [
            seeds_review_path(root),
            seeds_selected_path(root),
            benchmark_path(root),
            *(root / prompt for prompt in MANIFEST_PROMPTS),
        ],
        manifest_path(root),
        root=root,
        metadata={
            "main_benchmark": "MUST",
            "robustness_benchmark": "",
            "dataset_id": DATASET_ID,
            "seed_count": target_count,
            "source_modalities": eu.MODALITIES,
            "purpose": "document-context ablation (docs/context_ablation.md)",
            "documents": dict(sorted(document_counts.items())),
            "marker_counts": dict(sorted(marker_counts.items())),
        },
    )
    print(f"{selected_result['status']}: {selected_result['path']}")
    print(
        f"{benchmark_result['status']}: {benchmark_result['path']} ({len(items)} items)"
    )
    print(f"wrote {review_paths['markdown']} and {review_paths['csv']}")
    print(f"wrote {manifest_path(root)} ({len(manifest['artifacts'])} artifacts)")
    return {"benchmark": benchmark_result, "manifest": manifest}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stage",
        choices=["candidates", "review", "apply-revisions", "benchmark"],
        required=True,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing artifact instead of writing a *_candidate file.",
    )
    args = parser.parse_args(argv)
    root = eu.project_root()
    if args.stage == "candidates":
        build_candidates(root, overwrite=args.overwrite)
    elif args.stage == "review":
        write_capability_review(root)
    elif args.stage == "apply-revisions":
        apply_capability_revisions(root)
    else:
        build_benchmark(root, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
