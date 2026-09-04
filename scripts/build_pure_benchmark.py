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


def build_benchmark(root: Path, *, overwrite: bool = False) -> dict[str, object]:
    config = eu.load_config()
    target_count = eu.dataset_target_seed_count(config, DATASET_ID)
    seeds = eu.load_reviewed_seeds(
        seeds_review_path(root), target_count=target_count, strict=True
    )
    selected_result = eu.write_csv_rows_if_changed(
        seeds_selected_path(root),
        seeds,
        fieldnames=eu.seed_review_fields(DATASET_ID),
        overwrite=overwrite,
    )
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
    parser.add_argument("--stage", choices=["candidates", "benchmark"], required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing artifact instead of writing a *_candidate file.",
    )
    args = parser.parse_args(argv)
    root = eu.project_root()
    if args.stage == "candidates":
        build_candidates(root, overwrite=args.overwrite)
    else:
        build_benchmark(root, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
