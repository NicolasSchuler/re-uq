"""Which Task 2 run each Task 3 audit should read.

A Task 3 run audits one model's own Task 2 answers, so it needs a
`--source-run-id`. Picking it by hand does not scale past a cell or two, and
picking it wrongly is invisible afterwards: the audit runs happily against a
superseded generation and its verdicts then join to nothing.

The resolver lived only inside a heredoc in `scripts/enqueue_task3_runs.sh`.
It is a module now so the shell queue and any driver resolve sources the same
way -- newest **compatible complete** Task 2 run per (dataset, variant, profile,
model), where compatible means `eval_utils.registry_row_compatibility_issues`
reports nothing against the run group, benchmark size, sampling plan, and
batching of the config that is about to be used.

CLI:

    .venv/bin/python scripts/task3_sources.py --config run_configs/current_run.json

prints one `dataset<TAB>variant<TAB>profile<TAB>model<TAB>source_run_id` line
per resolvable cell and a `skip:` note per cell that has none.

Note that the archived 2026-05 runs are *not* resolvable: their registry rows
predate the `batch_order` column and this check does not waive it (the shell
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
    """One resolvable audit: the cell, the auditor, and what it reads.

    Attributes:
        dataset_id: Dataset of the cell.
        variant: Benchmark variant of the cell.
        profile_id: Run-config profile of the auditing model.
        model: Model that audits its own Task 2 answers.
        source_run_id: Task 2 run the audit reads, passed as `--source-run-id`.
    """

    dataset_id: str
    variant: str
    profile_id: str
    model: str
    source_run_id: str

    def as_line(self) -> str:
        """Return the source as one tab-separated line for the shell queue.

        Returns:
            The fields joined by tabs, in declaration order.
        """
        return "\t".join(self)


class Task3SourceGap(NamedTuple):
    """A cell with no compatible Task 2 run, and why it was looked for.

    Attributes:
        dataset_id: Dataset of the cell.
        variant: Benchmark variant of the cell.
        profile_id: Run-config profile of the would-be auditor.
        model: Model whose own Task 2 run is missing.
        reason: Why no source was found, naming the expected run-id prefix and
            the registry that was searched.
    """

    dataset_id: str
    variant: str
    profile_id: str
    model: str
    reason: str

    def as_line(self) -> str:
        """Return the gap as a `skip:` note line.

        Returns:
            `skip: dataset/variant profile/model: reason`.
        """
        return (
            f"skip: {self.dataset_id}/{self.variant} "
            f"{self.profile_id}/{self.model}: {self.reason}"
        )


def source_run_prefix(variant: str, mode: str = "full") -> str:
    """Return the run-id prefix of the Task 2 runs an audit of this mode may read.

    Mirrors `run_task3_verification_from_config.source_run_prefixes`: a
    paper-facing audit reads only `full-*` runs, and a smoke audit reads the
    smoke runs of the same variant.

    Args:
        variant: Benchmark variant; `must` maps to `full` and every other
            variant to `full-shall`.
        mode: `full` for a paper-facing audit; `smoke` swaps the leading `full`
            for `smoke`.

    Returns:
        `full`, `full-shall`, `smoke` or `smoke-shall`.
    """
    base = "full" if eu.normalize_benchmark_variant(variant) == "must" else "full-shall"
    return base.replace("full", "smoke", 1) if mode == "smoke" else base


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
    mode: str = "full",
) -> list[dict[str, Any]]:
    """Return the registry rows this audit could read, in registry order.

    Args:
        registry_rows: Run-registry rows of one dataset and variant.
        profile: Run-config profile of the auditor; a source must match its
            `profile_id` and `batch_size`, and its `batch_order` when set.
        model: Model whose own Task 2 runs are wanted.
        variant: Benchmark variant, which fixes the run-id prefix.
        run_group_id: Run group the source must belong to.
        benchmark_item_count: Number of benchmark items a full source must
            cover.
        expected_stochastic_samples: Stochastic samples per item the source must
            have planned.
        default_batch_order: Batch order expected when the profile sets none.
        mode: `full` requires a complete `full-*` run; `smoke` reads smoke runs
            and accepts a truncated benchmark.

    Returns:
        The rows that pass every check, in the order the registry lists them.
    """
    prefix = source_run_prefix(variant, mode)
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
            # A smoke run answers a truncated benchmark on purpose; a
            # paper-facing one must cover all of it.
            allow_partial_benchmark=mode == "smoke",
        )
    ]


def resolve_task3_sources(
    run_config: dict[str, Any],
    root: Path | None = None,
    *,
    skip_profiles: set[str] | None = None,
    mode: str = "full",
) -> tuple[list[Task3Source], list[Task3SourceGap]]:
    """Return the newest compatible Task 2 run per cell, plus the cells that have none.

    Args:
        run_config: Run config naming the `datasets`, `benchmark_variants`,
            `profiles` (each with `profile_id`, `models` and batching),
            `run_group_id`, `stochastic.samples` and optionally `batch_order`.
        root: Project root; None uses this checkout.
        skip_profiles: Profile ids to leave out entirely.
        mode: `full` or `smoke`, as for `compatible_source_rows`; `smoke` also
            reads the smoke run registry.

    Returns:
        One `Task3Source` per (dataset, variant, profile, model) with a
        compatible run, choosing the latest `started_at_utc`, and one
        `Task3SourceGap` per combination without one.
    """
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
            registry_path = eu.run_registry_path(
                root, dataset_id, variant, smoke=mode == "smoke"
            )
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
                        mode=mode,
                    )
                    if not candidates:
                        gaps.append(
                            Task3SourceGap(
                                dataset_id,
                                variant,
                                str(profile["profile_id"]),
                                model,
                                f"no complete {source_run_prefix(variant, mode)}-* "
                                "Task 2 "
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
    """Build the command-line parser.

    Returns:
        A parser with the required `--config` run config and the repeatable
        `--skip-profile`.
    """
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
    """Print the resolved sources to stdout and the unresolvable cells to stderr.

    Args:
        argv: Command-line arguments; None reads them from `sys.argv`.

    Returns:
        Exit status 0; unresolvable cells do not fail the run.
    """
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
