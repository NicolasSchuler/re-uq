"""Read-only progress reporter for provider-aware benchmark runs."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import eval_utils as eu
except ModuleNotFoundError:  # pragma: no cover
    from scripts import eval_utils as eu


def selected_registry_row(
    rows: list[dict[str, str]],
    run_id: str,
    *,
    model: str | None = None,
    profile_id: str | None = None,
) -> dict[str, str]:
    """Return the registry row of `run_id`, narrowed by model and profile.

    Args:
        rows: Rows of the run registry CSV.
        run_id: Run to look up.
        model: Model the row must carry; None accepts any model.
        profile_id: Profile the row must carry; None accepts any profile.

    Returns:
        The single matching row, or {} when no row matches.

    Raises:
        ValueError: If more than one row matches, so `--model` or `--profile`
            is needed to choose.
    """
    matches = [
        row
        for row in rows
        if str(row.get("run_id", "")) == str(run_id)
        and (model is None or str(row.get("model", "")) == str(model))
        and (profile_id is None or str(row.get("profile_id", "")) == str(profile_id))
    ]
    if len(matches) > 1:
        options = ", ".join(
            f"{row.get('profile_id', '')}/{row.get('model', '')}" for row in matches[:5]
        )
        raise ValueError(
            f"Run ID {run_id!r} matches multiple registry rows ({options}); "
            "pass --model and/or --profile."
        )
    return matches[0] if matches else {}


def expected_stochastic_samples(
    registry_row: dict[str, Any], benchmark_count: int
) -> int:
    """Infer the stochastic samples per item from a registry row's plan.

    Each item and task is planned as one deterministic record plus the
    stochastic samples, so the count is `expected_records` divided by items
    times tasks, minus one.

    Args:
        registry_row: Registry row with `expected_records` and `tasks`; an
            empty row falls back to the default.
        benchmark_count: Number of items the plan is divided by.

    Returns:
        The inferred sample count, or 5 when `expected_records` is missing,
        unreadable, or not a whole multiple of items times tasks.
    """
    try:
        expected_records = int(registry_row.get("expected_records", 0))
    except (TypeError, ValueError):
        return 5
    tasks = [
        task
        for task in str(registry_row.get("tasks", "task1,task2")).split(",")
        if task
    ]
    denominator = max(1, benchmark_count * max(1, len(tasks)))
    if expected_records and expected_records % denominator == 0:
        return max(0, expected_records // denominator - 1)
    return 5


def raw_rows_for_selection(
    rows: list[dict[str, Any]],
    run_id: str,
    registry_row: dict[str, Any],
    *,
    model: str | None = None,
    profile_id: str | None = None,
) -> list[dict[str, Any]]:
    """Select the raw rows of `run_id` for one model and profile.

    The model and profile come from the arguments or, when not given, from the
    registry row. Raw rows without a `profile_id` (written before it was
    recorded) are kept under any profile.

    Args:
        rows: Raw rows of the dataset and variant.
        run_id: Run whose rows are selected.
        registry_row: Selected registry row, used for the model and profile
            when the arguments leave them unset; may be empty.
        model: Model to keep; None falls back to the registry row's.
        profile_id: Profile to keep; None falls back to the registry row's.

    Returns:
        The matching raw rows, in file order.
    """
    selected_model = model or str(registry_row.get("model", "") or "")
    selected_profile_id = profile_id or str(registry_row.get("profile_id", "") or "")
    return [
        row
        for row in rows
        if str(row.get("run_id", "")) == run_id
        and (not selected_model or str(row.get("model", "")) == selected_model)
        and (
            not selected_profile_id
            or not row.get("profile_id")
            or str(row.get("profile_id", "")) == selected_profile_id
        )
    ]


def planned_benchmark_rows(
    benchmark: list[dict[str, Any]],
    registry_row: dict[str, Any],
    raw_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the benchmark rows this run actually planned, not the whole benchmark.

    Smoke runs plan only the first `--smoke-items` benchmark rows, so scoring
    their per-task progress against the full benchmark reported nonsense such as
    `records=12/4320`. Prefer the planned item count implied by the registry
    row's own `expected_records`; otherwise fall back to the item ids the run
    actually produced.

    Args:
        benchmark: Every benchmark row of the dataset and variant.
        registry_row: Selected registry row; may be empty.
        raw_rows: Raw rows of the run, used for the observed item ids.

    Returns:
        The first planned rows of `benchmark`, else the rows whose item ids
        the run produced when those are fewer than all, else the whole
        benchmark.
    """
    if not benchmark:
        return benchmark
    tasks = [
        task for task in str(registry_row.get("tasks", "") or "").split(",") if task
    ]
    try:
        expected_records = int(registry_row.get("expected_records", 0) or 0)
    except (TypeError, ValueError):
        expected_records = 0
    samples = expected_stochastic_samples(registry_row, len(benchmark)) + 1
    denominator = max(1, len(tasks)) * max(1, samples)
    planned_items = expected_records // denominator
    if (
        expected_records
        and planned_items * denominator == expected_records
        and 0 < planned_items <= len(benchmark)
    ):
        return benchmark[:planned_items]
    observed_ids = {str(row.get("item_id", "")) for row in raw_rows}
    scoped = [row for row in benchmark if str(row.get("item_id", "")) in observed_ids]
    if scoped and len(scoped) < len(benchmark):
        return scoped
    return benchmark


def print_progress(
    root: Path,
    dataset_id: str,
    variant: str,
    run_id: str,
    *,
    model: str | None = None,
    profile_id: str | None = None,
    reader: RawRowReader | None = None,
) -> None:
    """Print a progress snapshot of one run without writing anything.

    Prints the live progress line, the registry row, the logical-versus-raw
    attempt counts, parse status and run quality over every attempt, per-task
    progress over the latest logical observations, and the artifact paths.

    Args:
        root: Project root the artifacts are resolved under.
        dataset_id: Normalized dataset of the run.
        variant: Normalized benchmark variant of the run.
        run_id: Run to report; a smoke run id reads the smoke tree.
        model: Model to report when the run id is reused across models.
        profile_id: Profile to report when the run id is reused across
            profiles.
        reader: Incremental reader kept across `--watch` ticks; None reads the
            whole raw store each time.
    """
    benchmark_path = eu.artifact_path(
        root / "data/processed/benchmark_items.csv", dataset_id, variant
    )
    # Smoke run ids resolve to the parallel data/processed/smoke/ tree.
    raw_path = eu.model_outputs_raw_path(root, dataset_id, variant, run_id=run_id)
    registry_path = eu.run_registry_path(root, dataset_id, variant, run_id=run_id)
    live_progress_path = eu.run_progress_live_path(
        root, dataset_id, variant, run_id=run_id
    )
    events_path = eu.run_events_path(root, dataset_id, variant, run_id=run_id)

    benchmark = eu.read_csv_rows(benchmark_path)
    registry_rows = eu.read_csv_rows(registry_path) if registry_path.exists() else []
    registry_row = selected_registry_row(
        registry_rows, run_id, model=model, profile_id=profile_id
    )
    all_raw_rows = (
        reader.read(raw_path) if reader is not None else eu.read_jsonl(raw_path)
    )
    raw_rows = raw_rows_for_selection(
        all_raw_rows,
        run_id,
        registry_row,
        model=model,
        profile_id=profile_id,
    )
    # A raw file records physical attempts; a resumed run holds a failed
    # attempt and the retry that superseded it. Read both views by name so each
    # line below says which one it is reporting.
    ledger = eu.AttemptLedger.from_raw_rows(raw_rows)
    # Without a registry row there is no plan to compare against, so fall back
    # to the observations themselves -- logical ones, to match the counter.
    observed = len(ledger.latest_logical_observations)
    expected_records = int(registry_row.get("expected_records", observed) or observed)
    expected_api_calls = int(registry_row.get("expected_api_calls", 0) or 0)
    counters = eu.live_run_counters(
        ledger,
        expected_records=expected_records,
        expected_api_calls=expected_api_calls,
    )
    progress = eu.run_progress_summary(
        planned_benchmark_rows(benchmark, registry_row, ledger.all_attempts),
        ledger,
        expected_stochastic_samples=expected_stochastic_samples(
            registry_row, len(benchmark)
        ),
        view=eu.ATTEMPT_VIEW_LATEST,
    )

    print(eu.format_live_progress_line(run_id, counters))
    if registry_row:
        print(
            "registry:",
            {
                "status": registry_row.get("status", ""),
                "profile_id": registry_row.get("profile_id", ""),
                "model": registry_row.get("model", ""),
                "parse_success_rate": registry_row.get("parse_success_rate", ""),
                "observed_records": registry_row.get("observed_records", ""),
                "expected_records": registry_row.get("expected_records", ""),
                "observed_api_calls": registry_row.get("observed_api_calls", ""),
                "expected_api_calls": registry_row.get("expected_api_calls", ""),
            },
        )
    print(
        "attempts:",
        {
            "logical_observations": len(ledger.latest_logical_observations),
            "raw_attempts": len(ledger.all_attempts),
        },
    )
    # Per attempt: what the provider actually returned, retries included.
    print(
        "parse_status (all_attempts):",
        dict(Counter(str(row.get("parse_status", "")) for row in ledger.all_attempts)),
    )
    quality = eu.run_quality_counters(ledger.all_attempts)
    print("run_quality (all_attempts):", eu.format_run_quality_line(run_id, quality))
    if progress:
        # Per planned observation: a retried request counts once.
        print("task_progress (latest_logical_observations):")
        for row in progress:
            print(
                " ",
                row["task"],
                f"records={row['observed_records']}/{row['expected_records']}",
                f"parse={float(row['parse_success_rate']):.3f}",
                f"det_cov={float(row['deterministic_item_coverage']):.3f}",
                f"stoch_complete={float(row['stochastic_complete_item_rate']):.3f}",
            )
    print(
        "files:",
        {
            "raw": str(raw_path),
            "registry": str(registry_path),
            "live_progress": str(live_progress_path),
            "events": str(events_path),
        },
    )


class RawRowReader:
    """Incremental raw-store reader for `--watch`.

    Loads the compacted Parquet rows once, then remembers the JSONL byte offset
    already consumed so each tick parses only rows appended since the previous
    tick. Restarts from the Parquet rows if the tail shrinks or is replaced.
    """

    def __init__(self) -> None:
        self._offset = 0
        self._rows: list[dict[str, Any]] = []
        self._path: Path | None = None

    def read(self, path: Path) -> list[dict[str, Any]]:
        """Return every raw row of `path`, parsing only lines new since last call.

        A partial trailing line is left for the next call.

        Args:
            path: Raw JSONL file; a different path from the previous call
                starts over from its Parquet rows.

        Returns:
            A copy of all rows read so far, Parquet rows first.
        """
        path = Path(path)
        if path != self._path:
            self._path, self._offset = path, 0
            self._rows = eu.read_raw_store(path)
        if not path.exists():
            return list(self._rows)
        size = path.stat().st_size
        if size < self._offset:
            # The tail was compacted into the Parquet sibling: reload it.
            self._offset, self._rows = 0, eu.read_raw_store(path)
        if size > self._offset:
            with path.open("r", encoding="utf-8") as handle:
                handle.seek(self._offset)
                for line in handle:
                    if not line.endswith("\n"):
                        # Partial trailing line: leave it for the next tick.
                        break
                    self._offset += len(line.encode("utf-8"))
                    stripped = line.strip()
                    if stripped:
                        self._rows.append(json.loads(stripped))
        return list(self._rows)


def main() -> None:
    """Print one progress snapshot, or refresh it every `--watch` seconds."""
    parser = argparse.ArgumentParser(
        description="Show read-only progress for a provider-aware benchmark run."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--variant", default="must")
    parser.add_argument(
        "--model", help="Disambiguate when a run ID is reused across models."
    )
    parser.add_argument(
        "--profile",
        dest="profile_id",
        help="Disambiguate when a run ID is reused across profiles.",
    )
    parser.add_argument(
        "--watch",
        type=int,
        default=0,
        help="Refresh interval in seconds. Omit or set 0 for a single snapshot.",
    )
    args = parser.parse_args()

    root = eu.project_root()
    dataset_id = eu.normalize_dataset_id(args.dataset)
    variant = eu.normalize_benchmark_variant(args.variant)
    reader = RawRowReader() if args.watch > 0 else None
    while True:
        print_progress(
            root,
            dataset_id,
            variant,
            args.run_id,
            model=args.model,
            profile_id=args.profile_id,
            reader=reader,
        )
        if args.watch <= 0:
            break
        print()
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
