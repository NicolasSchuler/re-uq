"""Prepare NICE or MLM-TAPT seed candidates and build reviewed modality benchmarks.

The candidates stage preserves an existing manual review table. Inspect its
``include`` and ``capability_text_final`` columns before running the build stage.
Changed frozen benchmarks are written as candidates unless --overwrite is given.
Use --root with an isolated checkout to verify regeneration without changing the
published inputs. This script never calls an LLM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import eval_utils as eu
except ModuleNotFoundError:  # pragma: no cover - invocation-path fallback
    from scripts import eval_utils as eu


MANIFEST_PROMPTS = (
    "mandatory_entailment.txt",
    "mandatory_entailment_strict.txt",
    "modality_extraction.txt",
    "modality_extraction_labels_only.txt",
    "modality_verification.txt",
    "modality_verification_declared.txt",
)


def prepare_candidates(root: Path, dataset_id: str, config: dict[str, Any]) -> Path:
    """Write automatic suggestions separately when a review table already exists."""
    target_count = eu.dataset_target_seed_count(config, dataset_id)
    if dataset_id == "nice":
        source = root / config["datasets"]["nice_local_path"]
        if not source.exists():
            print(f"Downloading NICE source to {source}")
            eu.download_file(
                config["datasets"]["nice_url"],
                source,
                timeout_s=int(config["llm"]["timeout_s"]),
            )
        candidates = eu.make_seed_candidates(
            eu.read_csv_rows(source), target_count=target_count
        )
    else:
        candidates = eu.make_mlm_tapt_seed_candidates(
            eu.load_mlm_tapt_rows(config),
            target_count=target_count,
            seed=int(config["project"]["seed"]),
            exclude_source_regex=config["datasets"]["mlm_tapt_exclude_source_regex"],
            source_cap=30,
        )
    review_path = eu.artifact_path(root / "data/processed/seeds_review.csv", dataset_id)
    destination = (
        eu.auto_candidates_path(review_path) if review_path.exists() else review_path
    )
    eu.write_csv_rows(
        destination, candidates, fieldnames=eu.seed_review_fields(dataset_id)
    )
    print(f"Wrote {len(candidates)} automatic candidates: {destination}")
    if destination != review_path:
        print(f"Preserved existing manual review: {review_path}")
    print(
        f"Review include and capability_text_final in {review_path}; "
        f"the build stage requires exactly {target_count} included capabilities."
    )
    return destination


def validate_items(items: list[dict[str, Any]], target_count: int) -> None:
    """Keep the original benchmark shape, unique-item and gold-label checks."""
    expected = target_count * len(eu.MODALITIES)
    if len(items) != expected or len({r["item_id"] for r in items}) != expected:
        raise ValueError(f"Expected {expected} unique benchmark items.")
    if any(
        row["task1_gold_decision"]
        != ("yes" if row["source_modality"] == "mandatory" else "no")
        or row["task2_gold_modality"] != row["source_modality"]
        for row in items
    ):
        raise ValueError("Benchmark gold labels do not match source modality.")


def build_benchmark(
    root: Path,
    dataset_id: str,
    config: dict[str, Any],
    *,
    variant: str = "both",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build from reviewed seeds without accepting changed frozen inputs silently."""
    target_count = eu.dataset_target_seed_count(config, dataset_id)
    review_path = eu.artifact_path(root / "data/processed/seeds_review.csv", dataset_id)
    seeds = eu.load_reviewed_seeds(review_path, target_count=target_count, strict=True)
    variants = ("must", "shall") if variant == "both" else (variant,)
    benchmarks = {
        name: eu.build_benchmark_items(seeds, mandatory_keyword=name.upper())
        for name in ("must", "shall")
    }
    paths = {
        name: eu.artifact_path(
            root / "data/processed/benchmark_items.csv", dataset_id, name
        )
        for name in benchmarks
    }
    for name, items in benchmarks.items():
        validate_items(items, target_count)
        # A shared manifest must not certify a stale, unselected sibling variant.
        if name not in variants and paths[name].exists():
            expected = [
                {key: str(value) for key, value in row.items()} for row in items
            ]
            if eu.read_csv_rows(paths[name]) != expected:
                raise ValueError(
                    f"{paths[name]} differs from the reviewed seeds; "
                    "use --variant both to review or rebuild both variants."
                )
    prompt_paths = [root / "prompts" / name for name in MANIFEST_PROMPTS]
    missing = [str(path) for path in prompt_paths if not path.is_file()]
    if missing:
        raise ValueError("Missing benchmark prompt inputs: " + ", ".join(missing))

    selected_path = eu.artifact_path(
        root / "data/processed/seeds_selected.csv", dataset_id
    )
    results = [
        eu.write_csv_rows_if_changed(selected_path, seeds, overwrite=overwrite),
        *[
            eu.write_csv_rows_if_changed(
                paths[name], benchmarks[name], overwrite=overwrite
            )
            for name in variants
        ],
    ]
    for result in results:
        print(f"{result['status']}: {result['path']}")
        if result["candidate_path"]:
            print(f"Review regenerated candidate: {result['candidate_path']}")
    if any(result["status"] == "candidate_written" for result in results):
        raise ValueError(
            "Frozen inputs were preserved. Review the candidate files, then use "
            "--overwrite to accept them. The manifest and review exports were not updated."
        )

    eu.write_included_capability_review(
        review_path, root / "outputs", suffix=eu.dataset_suffix(dataset_id)
    )
    for name in variants:
        eu.write_benchmark_statement_review(
            benchmarks[name],
            root / "outputs",
            suffix=eu.dataset_variant_suffix(dataset_id, name),
        )
    manifest_path = eu.artifact_path(
        root / "outputs/benchmark_manifest.json", dataset_id
    )
    metadata = (
        json.loads(manifest_path.read_text(encoding="utf-8")).get("metadata", {})
        if manifest_path.exists()
        else {}
    )
    metadata.update(
        main_benchmark="MUST" if paths["must"].exists() else "",
        robustness_benchmark="SHALL" if paths["shall"].exists() else "",
        dataset_id=dataset_id,
        seed_count=target_count,
        source_modalities=eu.MODALITIES,
    )
    manifest = eu.write_benchmark_manifest(
        [
            review_path,
            selected_path,
            *[p for p in paths.values() if p.exists()],
            *prompt_paths,
        ],
        manifest_path,
        root=root,
        metadata=metadata,
    )
    print(f"Wrote benchmark review exports and manifest: {manifest_path}")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=("candidates", "build"), required=True)
    parser.add_argument("--dataset", choices=("nice", "mlm_tapt"), required=True)
    parser.add_argument("--variant", choices=("must", "shall", "both"), default="both")
    parser.add_argument(
        "--root",
        type=Path,
        default=eu.project_root(),
        help="Root containing inputs and outputs.",
    )
    parser.add_argument(
        "--config", type=Path, help="Config path, relative to --root unless absolute."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Build only: accept changed frozen inputs.",
    )
    args = parser.parse_args(argv)
    if args.stage == "candidates" and args.overwrite:
        parser.error(
            "--overwrite applies only to build; manual reviews are always preserved"
        )
    root = args.root.resolve()
    config_path = args.config or (
        Path("config.json")
        if (root / "config.json").exists()
        else Path("config.example.json")
    )
    config_path = root / config_path
    if not config_path.is_file():
        parser.error(f"Config file not found: {config_path}")
    config = eu.load_config(config_path)
    try:
        if args.stage == "candidates":
            prepare_candidates(root, args.dataset, config)
        else:
            build_benchmark(
                root,
                args.dataset,
                config,
                variant=args.variant,
                overwrite=args.overwrite,
            )
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
