"""Which Task 2 run each Task 3 audit should read.

A Task 3 run audits one model's own Task 2 answers, so it needs a
``--source-run-id``. Picking it by hand does not scale past a cell or two, and
picking it wrongly is invisible afterwards: the audit runs happily against a
superseded generation and its verdicts then join to nothing.

The resolver lived only inside a heredoc in ``scripts/enqueue_task3_runs.sh``.
It is a module now so the shell queue and any driver resolve sources the same
way -- newest **compatible complete** Task 2 run per (dataset, variant, profile,
model), where compatible means `eval_utils.registry_row_compatibility_issues`
reports nothing against the run group, benchmark size, sampling plan, and
batching of the config that is about to be used.

CLI:

    .venv/bin/python scripts/task3_sources.py --config run_configs/current_run.json

prints one ``dataset<TAB>variant<TAB>profile<TAB>model<TAB>source_run_id`` line
per resolvable cell and a ``skip:`` note per cell that has none.

Note that the archived 2026-05 runs are *not* resolvable: their registry rows
predate the ``batch_order`` column and this check does not waive it (the shell
script did not either). Every run written since does carry it, so this only
ever affects re-queueing the archive, which is not a thing the reruns do.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, NamedTuple

try:
    import eval_utils as eu
except ModuleNotFoundError:  # pragma: no cover - invocation-path fallback
    from scripts import eval_utils as eu


class Task3Source(NamedTuple):
    """One resolvable audit: the cell, the auditor, and what it reads."""

    dataset_id: str
    variant: str
    profile_id: str
    model: str
    source_run_id: str

    def as_line(self) -> str:
        return "\t".join(self)


class Task3SourceGap(NamedTuple):
    """A cell with no compatible Task 2 run, and why it was looked for."""

    dataset_id: str
    variant: str
    profile_id: str
    model: str
    reason: str

    def as_line(self) -> str:
        return (
            f"skip: {self.dataset_id}/{self.variant} "
            f"{self.profile_id}/{self.model}: {self.reason}"
        )


def source_run_prefix(variant: str) -> str:
    """Run-id prefix of the Task 2 runs an official audit may read."""
    return "full" if eu.normalize_benchmark_variant(variant) == "must" else "full-shall"


def compatible_source_rows(
    registry_rows: list[dict[str, Any]],
    *,
    profile: dict[str, Any],
    model: str,
    variant: str,
    run_group_id: str,
    benchmark_item_count: int,
    expected_stochastic_samples: int,
    default_batch_order: str = eu.DEFAULT_BATCH_ORDER,
) -> list[dict[str, Any]]:
    """Registry rows this audit could read, oldest first."""
    prefix = source_run_prefix(variant)
    return [
        row
        for row in registry_rows
        if str(row.get("profile_id", "")) == str(profile["profile_id"])
        and str(row.get("model", "")) == model
        and eu.run_id_matches_prefix(row.get("run_id", ""), prefix)
        and not eu.registry_row_compatibility_issues(
            row,
            run_group_id=run_group_id,
            benchmark_item_count=benchmark_item_count,
            expected_stochastic_samples=expected_stochastic_samples,
            required_tasks=("task2",),
            expected_batch_order=profile.get("batch_order", default_batch_order),
            expected_batch_size=int(profile["batch_size"]),
        )
    ]


def resolve_task3_sources(
    run_config: dict[str, Any],
    root: Path | None = None,
    *,
    skip_profiles: set[str] | None = None,
) -> tuple[list[Task3Source], list[Task3SourceGap]]:
    """Newest compatible Task 2 run per cell, plus the cells that have none."""
    root = Path(root or eu.project_root())
    skip = skip_profiles or set()
    sources: list[Task3Source] = []
    gaps: list[Task3SourceGap] = []
    for dataset_id in run_config["datasets"]:
        for variant in run_config["benchmark_variants"]:
            benchmark = eu.read_csv_rows(
                eu.artifact_path(
                    root / "data/processed/benchmark_items.csv", dataset_id, variant
                )
            )
            registry_path = eu.run_registry_path(root, dataset_id, variant)
            registry_rows = (
                eu.read_csv_rows(registry_path) if registry_path.exists() else []
            )
            for profile in run_config["profiles"]:
                if profile["profile_id"] in skip:
                    continue
                for model in profile["models"]:
                    candidates = compatible_source_rows(
                        registry_rows,
                        profile=profile,
                        model=model,
                        variant=variant,
                        run_group_id=str(run_config["run_group_id"]),
                        benchmark_item_count=len(benchmark),
                        expected_stochastic_samples=int(
                            run_config["stochastic"]["samples"]
                        ),
                        default_batch_order=str(
                            run_config.get("batch_order", eu.DEFAULT_BATCH_ORDER)
                        ),
                    )
                    if not candidates:
                        gaps.append(
                            Task3SourceGap(
                                dataset_id,
                                variant,
                                str(profile["profile_id"]),
                                model,
                                f"no complete {source_run_prefix(variant)}-* Task 2 "
                                f"run in {registry_path}",
                            )
                        )
                        continue
                    newest = max(
                        candidates, key=lambda row: str(row.get("started_at_utc", ""))
                    )
                    sources.append(
                        Task3Source(
                            dataset_id,
                            variant,
                            str(profile["profile_id"]),
                            model,
                            str(newest["run_id"]),
                        )
                    )
    return sources, gaps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--skip-profile",
        action="append",
        default=[],
        help="Profile id to leave out of the queue. Repeatable.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_config = eu.load_run_config(args.config)
    sources, gaps = resolve_task3_sources(
        run_config, skip_profiles=set(args.skip_profile)
    )
    for gap in gaps:
        print(gap.as_line(), file=sys.stderr)
    for source in sources:
        print(source.as_line())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
