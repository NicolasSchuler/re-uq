from __future__ import annotations

import argparse
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


def expected_stochastic_samples(registry_row: dict[str, Any], benchmark_count: int) -> int:
    try:
        expected_records = int(registry_row.get("expected_records", 0))
    except (TypeError, ValueError):
        return 5
    tasks = [task for task in str(registry_row.get("tasks", "task1,task2")).split(",") if task]
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


def print_progress(
    root: Path,
    dataset_id: str,
    variant: str,
    run_id: str,
    *,
    model: str | None = None,
    profile_id: str | None = None,
) -> None:
    benchmark_path = eu.artifact_path(root / "data/processed/benchmark_items.csv", dataset_id, variant)
    raw_path = eu.artifact_path(root / "data/processed/model_outputs_raw.jsonl", dataset_id, variant)
    registry_path = eu.run_registry_path(root, dataset_id, variant)
    live_progress_path = eu.run_progress_live_path(root, dataset_id, variant)
    events_path = eu.run_events_path(root, dataset_id, variant)

    benchmark = eu.read_csv_rows(benchmark_path)
    registry_rows = eu.read_csv_rows(registry_path) if registry_path.exists() else []
    registry_row = selected_registry_row(registry_rows, run_id, model=model, profile_id=profile_id)
    raw_rows = raw_rows_for_selection(
        eu.read_jsonl(raw_path),
        run_id,
        registry_row,
        model=model,
        profile_id=profile_id,
    )
    expected_records = int(registry_row.get("expected_records", len(raw_rows)) or len(raw_rows))
    expected_api_calls = int(registry_row.get("expected_api_calls", 0) or 0)
    counters = eu.live_run_counters(raw_rows, expected_records=expected_records, expected_api_calls=expected_api_calls)
    progress = eu.run_progress_summary(
        benchmark,
        raw_rows,
        expected_stochastic_samples=expected_stochastic_samples(registry_row, len(benchmark)),
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
    print("parse_status:", dict(Counter(str(row.get("parse_status", "")) for row in raw_rows)))
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
    print("files:", {"raw": str(raw_path), "registry": str(registry_path), "live_progress": str(live_progress_path), "events": str(events_path)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Show read-only progress for a provider-aware benchmark run.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--variant", default="must")
    parser.add_argument("--model", help="Disambiguate when a run ID is reused across models.")
    parser.add_argument("--profile", dest="profile_id", help="Disambiguate when a run ID is reused across profiles.")
    parser.add_argument("--watch", type=int, default=0, help="Refresh interval in seconds. Omit or set 0 for a single snapshot.")
    args = parser.parse_args()

    root = eu.project_root()
    dataset_id = eu.normalize_dataset_id(args.dataset)
    variant = eu.normalize_benchmark_variant(args.variant)
    while True:
        print_progress(root, dataset_id, variant, args.run_id, model=args.model, profile_id=args.profile_id)
        if args.watch <= 0:
            break
        print()
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
