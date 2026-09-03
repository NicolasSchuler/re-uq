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
    """Benchmark rows this run actually planned, not the whole benchmark.

    Smoke runs plan only the first ``--smoke-items`` benchmark rows, so scoring
    their per-task progress against the full benchmark reported nonsense such as
    ``records=12/4320``. Prefer the planned item count implied by the registry
    row's own ``expected_records``; otherwise fall back to the item ids the run
    actually produced.
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
    reader: "RawRowReader | None" = None,
) -> None:
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
    expected_records = int(
        registry_row.get("expected_records", len(raw_rows)) or len(raw_rows)
    )
    expected_api_calls = int(registry_row.get("expected_api_calls", 0) or 0)
    counters = eu.live_run_counters(
        raw_rows,
        expected_records=expected_records,
        expected_api_calls=expected_api_calls,
    )
    progress = eu.run_progress_summary(
        planned_benchmark_rows(benchmark, registry_row, raw_rows),
        raw_rows,
        expected_stochastic_samples=expected_stochastic_samples(
            registry_row, len(benchmark)
        ),
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
        "parse_status:",
        dict(Counter(str(row.get("parse_status", "")) for row in raw_rows)),
    )
    quality = eu.run_quality_counters(raw_rows)
    print("run_quality:", eu.format_run_quality_line(run_id, quality))
    if progress:
        print("task_progress:")
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
    """Incremental JSONL reader for ``--watch``.

    Remembers the byte offset already consumed so each tick parses only rows
    appended since the previous tick instead of re-reading a multi-hundred-MB
    raw file. Restarts from scratch if the file shrinks or is replaced.
    """

    def __init__(self) -> None:
        self._offset = 0
        self._rows: list[dict[str, Any]] = []
        self._path: Path | None = None

    def read(self, path: Path) -> list[dict[str, Any]]:
        path = Path(path)
        if path != self._path:
            self._path, self._offset, self._rows = path, 0, []
        if not path.exists():
            return list(self._rows)
        size = path.stat().st_size
        if size < self._offset:
            self._offset, self._rows = 0, []
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
