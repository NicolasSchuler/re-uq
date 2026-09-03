"""Run the Task 3 modality-preservation diagnostic from a provider config.

Task 3 is deliberately downstream of Task 2: it builds audit items from a
complete deterministic Task 2 run, then asks the same provider/model whether
each extracted requirement preserved, strengthened, weakened, or changed the
source statement. The default audit is blind to the Task 2 declared modality;
declared-modality modes are anchoring ablations. Task 3 never rewrites or
corrects the Task 2 raw outputs.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import eval_utils as eu
    import run_provenance as rp
    from runner_args import Task3RunnerArgs
except ModuleNotFoundError:  # pragma: no cover
    from scripts import eval_utils as eu, run_provenance as rp
    from scripts.runner_args import Task3RunnerArgs


def source_run_prefix(variant: str) -> str:
    """Canonical (paper-facing) Task 2 source-run prefix for a benchmark variant."""
    return "full" if variant == "must" else f"full-{variant}"


def smoke_source_run_prefix(variant: str) -> str:
    return "smoke" if variant == "must" else f"smoke-{variant}"


def source_run_prefixes(variant: str, mode: str = "full") -> list[str]:
    """Accepted Task 2 source-run prefixes.

    Official Task 3 runs always audit a `full-*` Task 2 run. `--mode smoke` also
    accepts a `smoke-*` source so the no-credentials fake-completion chain can
    run end to end without paper-facing data.
    """
    prefixes = [source_run_prefix(variant)]
    if mode == "smoke":
        prefixes.append(smoke_source_run_prefix(variant))
    return prefixes


def task3_run_prefix(
    mode: str, variant: str, audit_mode: str = eu.OFFICIAL_TASK3_AUDIT_MODE
) -> str:
    audit_mode = eu.normalize_task3_audit_mode(audit_mode)
    audit_suffix = (
        ""
        if audit_mode == eu.OFFICIAL_TASK3_AUDIT_MODE
        else f"-{audit_mode.replace('_', '-')}"
    )
    base = (
        f"task3{audit_suffix}"
        if variant == "must"
        else f"task3{audit_suffix}-{variant}"
    )
    return f"{base}-smoke" if mode == "smoke" else base


def task3_prompt_path(root: Path, audit_mode: str) -> Path:
    audit_mode = eu.normalize_task3_audit_mode(audit_mode)
    if audit_mode == eu.OFFICIAL_TASK3_AUDIT_MODE:
        return root / "prompts/modality_verification.txt"
    return root / "prompts/modality_verification_declared.txt"


def task3_declared_modality_for_prompt(item: Mapping[str, Any], audit_mode: str) -> str:
    audit_mode = eu.normalize_task3_audit_mode(audit_mode)
    if audit_mode == "declared_text":
        return str(item.get("task2_text_modality", ""))
    if audit_mode == "declared_source":
        return str(item.get("source_modality", ""))
    return ""


def task3_prompt_for(
    template: str,
    item: Mapping[str, Any],
    audit_mode: str = eu.OFFICIAL_TASK3_AUDIT_MODE,
) -> str:
    audit_mode = eu.normalize_task3_audit_mode(audit_mode)
    values = {
        "source_statement": item["source_statement"],
        "extracted_requirement": item["task2_requirement"],
    }
    if audit_mode != eu.OFFICIAL_TASK3_AUDIT_MODE:
        values["declared_extracted_modality"] = task3_declared_modality_for_prompt(
            item, audit_mode
        )
    return eu.render_prompt(template, **values)


def fake_completion(**kwargs: Any) -> dict[str, Any]:
    prompt = str(kwargs.get("prompt", ""))
    if "Items:\n" in prompt:
        items = json.loads(prompt.split("Items:\n", 1)[1])
        raw_text = json.dumps(
            {
                "results": [
                    {
                        "request_index": int(item["request_index"]),
                        "relation": "preserves",
                        "confidence": 0.9,
                        "evidence_phrase": str(item.get("source_statement", ""))[:80],
                        "brief_reason": "fake smoke",
                    }
                    for item in items
                ]
            }
        )
    elif "decision" in prompt:
        raw_text = '{"decision":"yes","confidence":0.9,"brief_reason":"fake smoke"}'
    else:
        raw_text = (
            '{"relation":"preserves","confidence":0.9,'
            '"evidence_phrase":"fake source phrase","brief_reason":"fake smoke"}'
        )
    return {
        "ok": True,
        "raw_text": raw_text,
        "response_json": {"fake": True, "model": kwargs.get("model", "")},
        "latency_s": 0.0,
        "error": "",
    }


def source_rows_for_model(
    source_rows: list[dict[str, Any]], model: str, profile_id: str
) -> list[dict[str, Any]]:
    model_rows = [row for row in source_rows if str(row.get("model", "")) == str(model)]
    profile_rows = [
        row for row in model_rows if str(row.get("profile_id", "")) in {"", profile_id}
    ]
    return profile_rows or model_rows


def require_complete_task2_source(
    benchmark: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    source_run_id: str,
    prefix: str | list[str],
    expected_stochastic_samples: int,
) -> None:
    progress = eu.run_progress_summary(
        benchmark,
        source_rows,
        expected_stochastic_samples=expected_stochastic_samples,
    )
    complete = eu.complete_run_ids_from_progress(
        progress, expected_tasks=["task2"], prefix=prefix
    )
    if source_run_id not in complete:
        raise ValueError(
            "Task 3 requires a complete full Task 2 source run for the selected model. "
            f"Run {source_run_id!r} is not complete for task2."
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run Task 3 blind text audit from a provider-aware run config."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--profile")
    parser.add_argument("--model")
    parser.add_argument("--dataset")
    parser.add_argument("--variant")
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--mode", choices=["smoke", "full", "resume"], default="smoke")
    parser.add_argument(
        "--audit-mode",
        choices=eu.TASK3_AUDIT_MODES,
        default=eu.OFFICIAL_TASK3_AUDIT_MODE,
    )
    parser.add_argument("--run-id")
    parser.add_argument("--smoke-items", type=int, default=2)
    parser.add_argument("--fake-completion", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-partial-source", action="store_true")
    parser.add_argument("--progress-every-records", type=int)
    parser.add_argument("--progress-every-seconds", type=int)
    parser.add_argument("--warn-after-records", type=int)
    parser.add_argument("--warn-parse-failure-rate", type=float)
    parser.add_argument("--warn-request-error-rate", type=float)
    parser.add_argument("--no-progress-artifacts", action="store_true")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level for the re_uq logger (default: INFO).",
    )
    args = parser.parse_args(argv)
    eu.configure_run_logging(args.log_level)
    run_from_config(eu.load_run_config(args.config), args)


@dataclass(frozen=True, slots=True)
class Task3Run:
    """State shared by every cell of one Task 3 audit run."""

    run_config: dict[str, Any]
    args: Task3RunnerArgs
    root: Path
    audit_mode: str
    logging_config: dict[str, Any]
    completion_fn: Callable[..., dict[str, Any]]
    task3_template: str
    smoke_tree: bool
    dry_run: bool

    @property
    def run_group_id(self) -> str:
        return str(self.run_config["run_group_id"])

    @property
    def expected_stochastic_samples(self) -> int:
        return int(self.run_config["stochastic"]["samples"])


@dataclass(frozen=True, slots=True)
class Task3Cell:
    """One (profile, model, dataset, variant) Task 3 cell, fully planned."""

    profile: dict[str, Any]
    model: str
    dataset_id: str
    variant: str
    run_id: str
    source_run_id: str
    started_at: str
    seed: int
    send_seed: bool
    max_retries: int
    batch_order: str
    preflight: dict[str, Any]
    items: list[dict[str, Any]]
    jobs: list[dict[str, Any]]
    existing_rows: list[dict[str, Any]]
    pending_jobs: list[dict[str, Any]]
    planned_api_calls: int
    pending_api_calls: int
    output_path: Path
    registry_path: Path
    progress_path: Path
    events_path: Path
    items_path: Path

    @property
    def profile_id(self) -> str:
        return str(self.profile["profile_id"])

    @property
    def provider_id(self) -> str:
        return str(self.profile["provider_id"])

    @property
    def batch_size(self) -> int:
        return int(self.profile["batch_size"])

    def identity(self) -> dict[str, Any]:
        """The fields that identify this cell in logs and warning events."""
        return {
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "benchmark_variant": self.variant,
            "provider_id": self.provider_id,
            "profile_id": self.profile_id,
            "model": self.model,
        }


def cell_context(matrix: Task3Run, cell: Task3Cell) -> dict[str, Any]:
    """Identity fields stamped onto every run event and warning event."""
    return {**cell.identity(), "run_group_id": matrix.run_group_id}


def resolve_source_rows(
    matrix: Task3Run,
    profile: Mapping[str, Any],
    model: str,
    dataset_id: str,
    variant: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Locate the complete Task 2 run this cell audits.

    Raises if no source run matches, if it holds no rows for `model`, or if it
    is incomplete and `--allow-partial-source` was not passed.
    """
    args = matrix.args
    # A smoke-* source run lives in the parallel smoke tree.
    source_raw_path = eu.model_outputs_raw_path(
        matrix.root, dataset_id, variant, run_id=args.source_run_id
    )
    accepted_prefixes = source_run_prefixes(variant, args.mode)
    source_run_id, source_rows = eu.select_run_rows(
        eu.read_jsonl(source_raw_path),
        run_id=args.source_run_id,
        prefix=accepted_prefixes,
    )
    if not source_run_id or not source_rows:
        raise ValueError(
            f"No source rows found for Task 2 run {args.source_run_id!r}. "
            f"Accepted run-id prefixes for --mode {args.mode} and variant {variant!r}: "
            f"{', '.join(f'{prefix}-*' for prefix in accepted_prefixes)}. "
            f"Searched {source_raw_path}."
        )
    model_source_rows = source_rows_for_model(
        source_rows, model, str(profile["profile_id"])
    )
    if not model_source_rows:
        raise ValueError(
            f"Source run {source_run_id!r} has no rows for model {model!r}; "
            "Task 3 should audit the same model's Task 2 outputs."
        )
    if not args.allow_partial_source:
        require_complete_task2_source(
            eu.read_csv_rows(
                eu.artifact_path(
                    matrix.root / "data/processed/benchmark_items.csv",
                    dataset_id,
                    variant,
                )
            ),
            model_source_rows,
            str(source_run_id),
            accepted_prefixes,
            expected_stochastic_samples=matrix.expected_stochastic_samples,
        )
    return str(source_run_id), model_source_rows


def plan_cell(
    matrix: Task3Run,
    profile: dict[str, Any],
    model: str,
    dataset_id: str,
    variant: str,
    *,
    run_id: str,
    preflight: dict[str, Any],
    seed: int,
    send_seed: bool,
    max_retries: int,
    batch_order: str,
) -> Task3Cell:
    """Resolve one Task 3 cell: its source run, audit items, and job plan."""
    args = matrix.args
    audit_mode = matrix.audit_mode
    source_run_id, model_source_rows = resolve_source_rows(
        matrix, profile, model, dataset_id, variant
    )
    benchmark = eu.read_csv_rows(
        eu.artifact_path(
            matrix.root / "data/processed/benchmark_items.csv", dataset_id, variant
        )
    )
    all_items = eu.build_task3_verification_items(
        benchmark, model_source_rows, audit_mode=audit_mode
    )
    if not all_items:
        raise ValueError(
            "No Task 3 text-audit items were built from valid deterministic Task 2 rows."
        )
    items_path = eu.task3_verification_items_path(
        matrix.root,
        dataset_id,
        variant,
        source_run_id,
        model,
        audit_mode,
        smoke=matrix.smoke_tree,
    )
    if not matrix.dry_run:
        eu.write_csv_rows(
            items_path, all_items, fieldnames=eu.TASK3_VERIFICATION_FIELDS
        )
    items = (
        all_items[: max(1, args.smoke_items)] if args.mode == "smoke" else all_items
    )
    jobs = eu.planned_completion_jobs_for_items(
        items,
        prompt_fn=lambda item: task3_prompt_for(
            matrix.task3_template, item, audit_mode=audit_mode
        ),
        prompt_version=f"{matrix.run_config['prompt_version']}:task3:{audit_mode}",
        model=model,
        host=profile["base_url"],
        run_id=run_id,
        deterministic=matrix.run_config["deterministic"],
        stochastic=matrix.run_config["stochastic"],
        max_tokens=int(profile["max_tokens"]),
        timeout_s=int(profile["timeout_s"]),
        api_key_env=profile["api_key_env"],
        provider_id=profile["provider_id"],
        profile_id=profile["profile_id"],
        run_group_id=matrix.run_group_id,
        json_mode=bool(profile["json_mode"]),
        structured_output=str(profile.get("structured_output", "none")),
        response_format=profile.get("response_format"),
        extra_body=profile.get("extra_body"),
        instructor_mode=str(profile.get("instructor_mode", "json")),
        validation_retries=int(profile.get("validation_retries", 2)),
        fallback_batch_size=int(profile.get("fallback_batch_size", 1)),
        seed=seed,
        send_seed=send_seed,
        max_retries=max_retries,
        batch_order=batch_order,
        batch_size=int(profile["batch_size"]),
        server_model_probe=preflight,
    )
    output_path = eu.task3_raw_path(
        matrix.root, dataset_id, variant, smoke=matrix.smoke_tree
    )
    existing_rows = eu.read_jsonl(output_path)
    pending_jobs = eu.pending_completion_jobs(jobs, existing_rows, run_id)
    batch_size = int(profile["batch_size"])
    return Task3Cell(
        profile=profile,
        model=model,
        dataset_id=dataset_id,
        variant=variant,
        run_id=run_id,
        source_run_id=source_run_id,
        started_at=eu.utc_now_iso(),
        seed=seed,
        send_seed=send_seed,
        max_retries=max_retries,
        batch_order=batch_order,
        preflight=preflight,
        items=items,
        jobs=jobs,
        existing_rows=existing_rows,
        pending_jobs=pending_jobs,
        planned_api_calls=len(eu.completion_job_batches(jobs, batch_size)),
        # `planned_jobs` batches over the full plan and then keeps the pending
        # slice, so a resumed run never re-shuffles the batches.
        pending_api_calls=len(
            eu.completion_job_batches(pending_jobs, batch_size, planned_jobs=jobs)
        ),
        output_path=output_path,
        registry_path=eu.task3_registry_path(
            matrix.root, dataset_id, variant, smoke=matrix.smoke_tree
        ),
        progress_path=eu.task3_progress_path(
            matrix.root, dataset_id, variant, smoke=matrix.smoke_tree
        ),
        events_path=eu.task3_events_path(
            matrix.root, dataset_id, variant, smoke=matrix.smoke_tree
        ),
        items_path=items_path,
    )


def registry_row(
    matrix: Task3Run,
    cell: Task3Cell,
    raw_rows: list[dict[str, Any]],
    *,
    status: str | None = None,
    finished_at_utc: str = "",
) -> dict[str, Any]:
    """Summarize `raw_rows` as this cell's Task 3 registry row."""
    profile = cell.profile
    return eu.run_registry_summary(
        cell.items,
        raw_rows,
        run_id=cell.run_id,
        run_group_id=matrix.run_group_id,
        provider_id=cell.provider_id,
        profile_id=cell.profile_id,
        model=cell.model,
        dataset_id=cell.dataset_id,
        variant=cell.variant,
        tasks=["task3"],
        expected_stochastic_samples=matrix.expected_stochastic_samples,
        started_at_utc=cell.started_at,
        finished_at_utc=finished_at_utc,
        status=status,
        base_url=profile["base_url"],
        api_key_env=profile["api_key_env"],
        concurrency=profile["concurrency"],
        batch_size=profile["batch_size"],
        timeout_s=profile["timeout_s"],
        json_mode=bool(profile["json_mode"]),
        structured_output=str(profile.get("structured_output", "none")),
        request_extra_body=profile.get("extra_body"),
        server_model_probe=cell.preflight,
        batch_order=cell.batch_order,
        notes=rp.run_notes(
            matrix.args,
            f"audit_mode={matrix.audit_mode}; source_run_id={cell.source_run_id}",
        ),
    )


def log_planned_cell(matrix: Task3Run, cell: Task3Cell) -> None:
    """Report a dry-run cell plan without contacting a provider."""
    eu.logger.info(
        "%s",
        {
            "dry_run": True,
            "run_id": cell.run_id,
            "source_run_id": cell.source_run_id,
            "dataset_id": cell.dataset_id,
            "variant": cell.variant,
            "profile": cell.profile_id,
            "model": cell.model,
            "audit_mode": matrix.audit_mode,
            "task3_items": len(cell.items),
            "planned_jobs": len(cell.jobs),
            "pending_jobs": len(cell.pending_jobs),
            "planned_batches": cell.planned_api_calls,
            "pending_api_calls": cell.pending_api_calls,
            "batch_size": cell.profile["batch_size"],
            "batch_order": cell.batch_order,
            "output_path": str(cell.output_path),
            "task3_items_path": str(cell.items_path),
        },
    )


def execute_cell(matrix: Task3Run, cell: Task3Cell) -> None:
    """Run one Task 3 cell to completion, streaming progress artifacts."""
    logging_config = matrix.logging_config
    current_rows = list(cell.existing_rows)
    started_monotonic = time.monotonic()
    emitted_warning_types: set[str] = set()

    def refresh_live_progress(
        event_type: str, finished_at_utc: str = "", print_line: bool = True
    ) -> dict[str, Any]:
        run_rows = eu.select_model_run_rows(
            current_rows, cell.run_id, cell.model, ("task3",)
        )
        pending_jobs_now = eu.pending_completion_jobs(
            cell.jobs, current_rows, cell.run_id
        )
        pending_api_calls_now = len(
            eu.completion_job_batches(
                pending_jobs_now, cell.batch_size, planned_jobs=cell.jobs
            )
        )
        status = (
            "running"
            if event_type in {"start", "progress"} and pending_jobs_now
            else None
        )
        eu.upsert_run_registry_row(
            cell.registry_path,
            registry_row(
                matrix,
                cell,
                current_rows,
                status=status,
                finished_at_utc=finished_at_utc,
            ),
        )
        if logging_config["write_progress_csv"]:
            eu.write_live_progress_csv(
                cell.progress_path,
                cell.items,
                run_rows,
                expected_stochastic_samples=matrix.expected_stochastic_samples,
            )
        counters = eu.live_run_counters(
            run_rows,
            expected_records=len(cell.jobs),
            expected_api_calls=cell.planned_api_calls,
            started_monotonic=started_monotonic,
        )
        event = {
            "event_type": event_type,
            **cell_context(matrix, cell),
            "source_run_id": cell.source_run_id,
            "tasks": ["task3"],
            "mode": matrix.args.mode,
            "audit_mode": matrix.audit_mode,
            "output_path": str(cell.output_path),
            "task3_items_path": str(cell.items_path),
            "registry_path": str(cell.registry_path),
            "progress_path": str(cell.progress_path),
            "planned_jobs": len(cell.jobs),
            "pending_jobs": len(pending_jobs_now),
            "planned_api_calls": cell.planned_api_calls,
            "pending_api_calls": pending_api_calls_now,
            **counters,
        }
        if logging_config["write_event_jsonl"]:
            eu.append_run_event(cell.events_path, event)
        if print_line and event_type in {"progress", "finish"}:
            eu.logger.info("%s", eu.format_live_progress_line(cell.run_id, counters))
        if event_type == "finish":
            eu.logger.info(
                "%s",
                eu.format_run_quality_line(
                    cell.run_id, eu.run_quality_counters(run_rows)
                ),
            )
        return counters

    eu.upsert_run_registry_row(
        cell.registry_path,
        registry_row(
            matrix,
            cell,
            current_rows,
            status="running" if cell.pending_jobs else None,
        ),
    )
    refresh_live_progress("start", print_line=False)
    eu.logger.info(
        "%s",
        {
            "run_id": cell.run_id,
            "source_run_id": cell.source_run_id,
            "dataset_id": cell.dataset_id,
            "variant": cell.variant,
            "profile": cell.profile_id,
            "model": cell.model,
            "audit_mode": matrix.audit_mode,
            "task3_items": len(cell.items),
            "planned_jobs": len(cell.jobs),
            "pending_jobs": len(cell.pending_jobs),
            "batch_size": cell.profile["batch_size"],
            "batch_order": cell.batch_order,
            "seed": cell.seed,
            "send_seed": cell.send_seed,
            "max_retries": cell.max_retries,
            "output_path": str(cell.output_path),
            "task3_items_path": str(cell.items_path),
            "registry_path": str(cell.registry_path),
        },
    )

    progress_every_records = int(logging_config["progress_every_records"])
    progress_every_seconds = int(logging_config["progress_every_seconds"])
    last_progress_record_index = 0
    last_progress_monotonic = time.monotonic()
    for index, record in enumerate(
        eu.run_completion_jobs(
            cell.pending_jobs,
            max_workers=int(cell.profile["concurrency"]),
            completion_fn=matrix.completion_fn,
            batch_size=cell.batch_size,
            planned_jobs=cell.jobs,
        ),
        start=1,
    ):
        eu.append_jsonl(cell.output_path, record)
        current_rows.append(record)
        now_monotonic = time.monotonic()
        records_due = index - last_progress_record_index >= progress_every_records
        seconds_due = (
            progress_every_seconds > 0
            and now_monotonic - last_progress_monotonic >= progress_every_seconds
        )
        if records_due or seconds_due or index == len(cell.pending_jobs):
            counters = refresh_live_progress("progress")
            last_progress_record_index = index
            last_progress_monotonic = now_monotonic
            eu.emit_warning_events(
                counters,
                logging_config=logging_config,
                emitted_warning_types=emitted_warning_types,
                context=cell_context(matrix, cell),
                events_path=cell.events_path,
            )

    finish_row = registry_row(
        matrix, cell, current_rows, finished_at_utc=eu.utc_now_iso()
    )
    eu.upsert_run_registry_row(cell.registry_path, finish_row)
    refresh_live_progress("finish", finished_at_utc=str(finish_row["finished_at_utc"]))
    eu.logger.info(
        "Task 3 registry status: %s at %s", finish_row["status"], cell.registry_path
    )


def run_from_config(run_config: dict[str, Any], args: Task3RunnerArgs) -> None:
    """Execute a normalized run config.

    Shared by the argparse CLI above and by the Hydra entry point in
    `scripts/run.py`, which composes the same dictionary from `conf/`.
    """
    eu.configure_run_logging(args.log_level)
    eu.apply_acse_embedding_env(run_config)
    if args.mode == "resume" and not args.run_id:
        raise ValueError("--mode resume requires --run-id.")

    root = eu.project_root()
    audit_mode = eu.normalize_task3_audit_mode(args.audit_mode)
    profiles = eu.filter_run_profiles(
        run_config, profile_id=args.profile, model=args.model
    )
    datasets = eu.selected_values(list(run_config["datasets"]), args.dataset, "dataset")
    variants = eu.selected_values(
        list(run_config["benchmark_variants"]), args.variant, "variant"
    )
    matrix = Task3Run(
        run_config=run_config,
        args=args,
        root=root,
        audit_mode=audit_mode,
        logging_config=eu.logging_config_from_args(run_config, args),
        completion_fn=fake_completion if args.fake_completion else eu.chat_completion,
        task3_template=eu.load_prompt(task3_prompt_path(root, audit_mode)),
        # Smoke/fake Task 3 runs are isolated from the paper-facing artifact tree.
        smoke_tree=bool(args.fake_completion)
        or args.mode == "smoke"
        or eu.is_smoke_run_id(args.run_id),
        dry_run=bool(getattr(args, "dry_run", False)),
    )

    for profile in profiles:
        eu.validate_manual_server_profile(profile)
        seed = int(profile.get("seed", run_config.get("seed", eu.DEFAULT_REQUEST_SEED)))
        send_seed = bool(profile.get("send_seed", True))
        max_retries = int(profile.get("max_retries", eu.DEFAULT_MAX_RETRIES))
        batch_order = eu.normalize_batch_order(
            profile.get(
                "batch_order", run_config.get("batch_order", eu.DEFAULT_BATCH_ORDER)
            )
        )
        for model in profile["models"]:
            preflight = (
                {"dry_run": True}
                if matrix.dry_run
                else eu.preflight_profile(
                    profile,
                    model=model,
                    prompt_version=run_config["prompt_version"],
                    completion_fn=matrix.completion_fn,
                )
            )
            for dataset_id in datasets:
                for variant in variants:
                    run_id = args.run_id or eu.new_run_id(
                        task3_run_prefix(args.mode, variant, audit_mode)
                    )
                    if not matrix.dry_run:
                        # Configured before planning so planning-time warnings
                        # land in this run's log file.
                        eu.configure_run_logging(
                            args.log_level, log_path=eu.run_log_path(root, run_id)
                        )
                        # No-op unless the run was composed by scripts/run.py.
                        rp.write_resolved_config(
                            root, run_id, getattr(args, "resolved_config_yaml", "")
                        )
                    cell = plan_cell(
                        matrix,
                        profile,
                        model,
                        dataset_id,
                        variant,
                        run_id=run_id,
                        preflight=preflight,
                        seed=seed,
                        send_seed=send_seed,
                        max_retries=max_retries,
                        batch_order=batch_order,
                    )
                    if matrix.dry_run:
                        log_planned_cell(matrix, cell)
                    else:
                        execute_cell(matrix, cell)


if __name__ == "__main__":
    main()
