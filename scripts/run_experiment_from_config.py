"""Run Task 1 and Task 2 provider-matrix experiments from JSON config.

The runner is the canonical reproduction entry point for primary benchmark
calls. It writes one raw JSONL record per item/sample and records run status in
local registry/progress artifacts so incomplete runs can be resumed safely.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import eval_utils as eu
    import run_provenance as rp
    from runner_args import RunnerArgs
except ModuleNotFoundError:  # pragma: no cover
    from scripts import eval_utils as eu, run_provenance as rp
    from scripts.runner_args import RunnerArgs


def run_prefix(mode: str, variant: str) -> str:
    if mode == "smoke":
        return "smoke" if variant == "must" else f"smoke-{variant}"
    return "full" if variant == "must" else f"full-{variant}"


def source_from_prompt(prompt: str) -> str:
    match = re.search(r'Source:\s*\n?"([^"]+)"', prompt, flags=re.S)
    return match.group(1) if match else prompt


FAKE_CONFIDENCE = 0.9
FAKE_REASON = "fake smoke"


def fake_modality(source_statement: str) -> str:
    """Classify a statement by modal keyword, as the smoke fixture expects."""
    source = source_statement.lower()
    if "must " in source or "shall " in source:
        return "mandatory"
    if "should " in source:
        return "recommended"
    if "may " in source:
        return "optional"
    return "nice_to_have"


def fake_batch_results(prompt: str) -> list[dict[str, Any]]:
    """Answer every item of a batched prompt, one result per request index."""
    results: list[dict[str, Any]] = []
    for item in json.loads(prompt.split("Items:\n", 1)[1]):
        request_index = int(item["request_index"])
        if "candidate_requirement" in item:
            results.append(
                {
                    "request_index": request_index,
                    "decision": "yes",
                    "confidence": FAKE_CONFIDENCE,
                    "brief_reason": FAKE_REASON,
                }
            )
        else:
            statement = str(item["source_statement"])
            results.append(
                {
                    "request_index": request_index,
                    "requirement": statement,
                    "modality": fake_modality(statement),
                    "confidence": FAKE_CONFIDENCE,
                }
            )
    return results


def fake_completion(**kwargs: Any) -> dict[str, Any]:
    """Answer a prompt offline, so smoke runs never contact a provider.

    The reply shape is chosen from the prompt: batched wrapper, Task 1
    entailment decision, or Task 2 modality extraction.
    """
    prompt = str(kwargs.get("prompt", ""))
    if '"results"' in prompt and "Items:\n" in prompt:
        payload: dict[str, Any] = {"results": fake_batch_results(prompt)}
    elif "Candidate requirement:" in prompt or '"decision"' in prompt:
        payload = {
            "decision": "yes",
            "confidence": FAKE_CONFIDENCE,
            "brief_reason": FAKE_REASON,
        }
    else:
        statement = source_from_prompt(prompt)
        payload = {
            "requirement": statement,
            "modality": fake_modality(statement),
            "confidence": FAKE_CONFIDENCE,
        }
    return {
        "ok": True,
        "raw_text": json.dumps(payload),
        "response_json": {"fake": True, "model": kwargs.get("model", "")},
        "latency_s": 0.0,
        "error": "",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run provider-aware benchmark experiments from a run config."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--profile")
    parser.add_argument("--model")
    parser.add_argument("--dataset")
    parser.add_argument("--variant")
    parser.add_argument("--task", choices=["task1", "task2", "both"], default=None)
    parser.add_argument("--mode", choices=["smoke", "full", "resume"], default="smoke")
    parser.add_argument("--run-id")
    parser.add_argument("--smoke-items", type=int, default=2)
    parser.add_argument("--fake-completion", action="store_true")
    parser.add_argument("--progress-every-records", type=int)
    parser.add_argument("--progress-every-seconds", type=int)
    parser.add_argument("--warn-after-records", type=int)
    parser.add_argument("--warn-parse-failure-rate", type=float)
    parser.add_argument("--warn-request-error-rate", type=float)
    parser.add_argument("--no-progress-artifacts", action="store_true")
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Iterate every model of the selected profile(s) sequentially (ignores --model).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned job/batch/API-call counts and exit without contacting a provider.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level for the re_uq logger (default: INFO).",
    )
    args = parser.parse_args(argv)
    eu.configure_run_logging(args.log_level)
    run_from_config(eu.load_run_config(args.config), args)


@dataclass(frozen=True, slots=True)
class MatrixRun:
    """State shared by every cell of one provider-matrix run."""

    run_config: dict[str, Any]
    args: RunnerArgs
    root: Path
    tasks: list[str]
    logging_config: dict[str, Any]
    completion_fn: Callable[..., dict[str, Any]]
    task1_template: str
    task2_template: str
    task2_context_template: str
    item_context: str
    semantic_embedding_backend: str
    smoke_tree: bool

    @property
    def run_group_id(self) -> str:
        return str(self.run_config["run_group_id"])

    @property
    def expected_stochastic_samples(self) -> int:
        return int(self.run_config["stochastic"]["samples"])


@dataclass(frozen=True, slots=True)
class RunCell:
    """One (profile, model, dataset, variant) cell, fully planned.

    Everything here is resolved before any provider call happens, so the dry-run
    path and the execution path report identical plans.
    """

    profile: dict[str, Any]
    model: str
    dataset_id: str
    variant: str
    run_id: str
    started_at: str
    seed: int
    send_seed: bool
    max_retries: int
    batch_order: str
    preflight: dict[str, Any]
    benchmark_rows: list[dict[str, str]]
    jobs: list[dict[str, Any]]
    existing_rows: list[dict[str, Any]]
    pending_jobs: list[dict[str, Any]]
    planned_api_calls: int
    pending_api_calls: int
    output_path: Path
    registry_path: Path
    progress_path: Path
    events_path: Path

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


def cell_context(matrix: MatrixRun, cell: RunCell) -> dict[str, Any]:
    """Identity fields stamped onto every run event and warning event."""
    return {**cell.identity(), "run_group_id": matrix.run_group_id}


def plan_cell(
    matrix: MatrixRun,
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
) -> RunCell:
    """Resolve one matrix cell: its artifact paths and job plan."""
    args = matrix.args
    smoke_run = matrix.smoke_tree or eu.is_smoke_run_id(run_id)
    benchmark = eu.read_csv_rows(
        eu.artifact_path(
            matrix.root / "data/processed/benchmark_items.csv", dataset_id, variant
        )
    )
    benchmark_rows = (
        benchmark[: max(1, args.smoke_items)] if args.mode == "smoke" else benchmark
    )
    jobs = eu.planned_completion_jobs(
        benchmark_rows,
        tasks=matrix.tasks,
        model=model,
        host=profile["base_url"],
        run_id=run_id,
        prompt_version=matrix.run_config["prompt_version"],
        task1_template=matrix.task1_template,
        task2_template=matrix.task2_template,
        task2_context_template=matrix.task2_context_template,
        item_context=matrix.item_context,
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
    output_path = eu.model_outputs_raw_path(
        matrix.root, dataset_id, variant, smoke=smoke_run
    )
    existing_rows = eu.read_jsonl(output_path)
    pending_jobs = eu.pending_completion_jobs(jobs, existing_rows, run_id)
    batch_size = int(profile["batch_size"])
    return RunCell(
        profile=profile,
        model=model,
        dataset_id=dataset_id,
        variant=variant,
        run_id=run_id,
        started_at=eu.utc_now_iso(),
        seed=seed,
        send_seed=send_seed,
        max_retries=max_retries,
        batch_order=batch_order,
        preflight=preflight,
        benchmark_rows=benchmark_rows,
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
        registry_path=eu.run_registry_path(
            matrix.root, dataset_id, variant, smoke=smoke_run
        ),
        progress_path=eu.run_progress_live_path(
            matrix.root, dataset_id, variant, smoke=smoke_run
        ),
        events_path=eu.run_events_path(
            matrix.root, dataset_id, variant, smoke=smoke_run
        ),
    )


def registry_row(
    matrix: MatrixRun,
    cell: RunCell,
    raw_rows: list[dict[str, Any]],
    *,
    status: str | None = None,
    finished_at_utc: str = "",
) -> dict[str, Any]:
    """Summarize `raw_rows` as this cell's registry row.

    The start, progress, and finish writes differ only in `status` and
    `finished_at_utc`; every other field is a property of the cell.
    """
    profile = cell.profile
    return eu.run_registry_summary(
        cell.benchmark_rows,
        raw_rows,
        run_id=cell.run_id,
        run_group_id=matrix.run_group_id,
        provider_id=cell.provider_id,
        profile_id=cell.profile_id,
        model=cell.model,
        dataset_id=cell.dataset_id,
        variant=cell.variant,
        tasks=matrix.tasks,
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
        item_context=matrix.item_context,
        notes=rp.run_notes(matrix.args),
    )


def log_planned_cell(matrix: MatrixRun, cell: RunCell) -> None:
    """Report a dry-run cell plan without contacting a provider."""
    eu.logger.info(
        "%s",
        {
            "dry_run": True,
            "dataset_id": cell.dataset_id,
            "variant": cell.variant,
            "profile": cell.profile_id,
            "model": cell.model,
            "planned_jobs": len(cell.jobs),
            "planned_batches": cell.planned_api_calls,
            "batch_size": cell.profile["batch_size"],
            "batch_order": cell.batch_order,
            "item_context": matrix.item_context,
            "seed": cell.seed,
            "send_seed": cell.send_seed,
            "max_retries": cell.max_retries,
            "estimated_api_calls": cell.planned_api_calls,
            "pending_api_calls": cell.pending_api_calls,
            "output_path": str(cell.output_path),
        },
    )


def execute_cell(matrix: MatrixRun, cell: RunCell) -> None:
    """Run one matrix cell to completion, streaming progress artifacts."""
    logging_config = matrix.logging_config
    current_rows = list(cell.existing_rows)
    started_monotonic = time.monotonic()
    emitted_warning_types: set[str] = set()

    def refresh_live_progress(
        event_type: str, finished_at_utc: str = "", print_line: bool = True
    ) -> dict[str, Any]:
        run_rows = eu.select_model_run_rows(
            current_rows, cell.run_id, cell.model, matrix.tasks
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
                cell.benchmark_rows,
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
            "tasks": matrix.tasks,
            "mode": matrix.args.mode,
            "output_path": str(cell.output_path),
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
            "dataset_id": cell.dataset_id,
            "variant": cell.variant,
            "profile": cell.profile_id,
            "model": cell.model,
            "planned_jobs": len(cell.jobs),
            "pending_jobs": len(cell.pending_jobs),
            "batch_size": cell.profile["batch_size"],
            "batch_order": cell.batch_order,
            "item_context": matrix.item_context,
            "seed": cell.seed,
            "send_seed": cell.send_seed,
            "max_retries": cell.max_retries,
            "planned_api_calls": cell.planned_api_calls,
            "pending_api_calls": cell.pending_api_calls,
            "events_path": str(cell.events_path),
            "progress_path": str(cell.progress_path),
            "output_path": str(cell.output_path),
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
        # Persist the resolved choice on the run artifact. The analysis process
        # cannot inherit this process's env.
        record["semantic_embedding_backend"] = matrix.semantic_embedding_backend
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
        "Registry status: %s at %s", finish_row["status"], cell.registry_path
    )


def run_from_config(run_config: dict[str, Any], args: RunnerArgs) -> None:
    """Execute a normalized run config.

    Shared by the argparse CLI above and by the Hydra entry point in
    `scripts/run.py`, which composes the same dictionary from `conf/`.
    """
    eu.configure_run_logging(args.log_level)
    if args.mode == "resume" and not args.run_id:
        raise ValueError("--mode resume requires --run-id.")

    semantic_embedding_backend, _ = eu.semantic_embedding_backend_label(
        run_config.get("acse_embedding_backend"),
        run_config.get("acse_embedding_mlx_model"),
    )
    root = eu.project_root()
    model_filter = None if args.all_models else args.model
    profiles = eu.filter_run_profiles(
        run_config, profile_id=args.profile, model=model_filter
    )
    datasets = eu.selected_values(list(run_config["datasets"]), args.dataset, "dataset")
    variants = eu.selected_values(
        list(run_config["benchmark_variants"]), args.variant, "variant"
    )
    tasks = eu.normalize_task_filter(args.task or run_config["tasks"])
    item_context = eu.normalize_item_context(run_config.get("item_context"))
    if item_context != eu.DEFAULT_ITEM_CONTEXT and tasks != ["task2"]:
        # The context prompt exists for Task 2 only; a mixed run would silently
        # send bare Task 1 items under a `document` label.
        raise ValueError(
            f"item_context={item_context!r} requires task=task2, got {tasks}."
        )
    matrix = MatrixRun(
        run_config=run_config,
        args=args,
        root=root,
        tasks=tasks,
        logging_config=eu.logging_config_from_args(run_config, args),
        completion_fn=fake_completion if args.fake_completion else eu.chat_completion,
        task1_template=eu.load_prompt(root / "prompts/mandatory_entailment.txt"),
        task2_template=eu.load_prompt(root / "prompts/modality_extraction.txt"),
        # The context prompt is loaded only for the `document` arm, so bare
        # runs keep exactly the two frozen prompt inputs of the paper.
        task2_context_template=(
            eu.load_prompt(root / "prompts/modality_extraction_context.txt")
            if item_context != eu.DEFAULT_ITEM_CONTEXT
            else ""
        ),
        item_context=item_context,
        semantic_embedding_backend=semantic_embedding_backend,
        # Fake completions and smoke run ids never touch the paper-facing
        # artifacts; they are written to the parallel data/processed/smoke/ tree.
        smoke_tree=bool(args.fake_completion)
        or eu.is_smoke_run_id(args.run_id)
        or args.mode == "smoke",
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
        if args.all_models:
            eu.logger.info(
                "profile %s: running %d model(s) sequentially: %s",
                profile["profile_id"],
                len(profile["models"]),
                ", ".join(profile["models"]),
            )
        for model in profile["models"]:
            preflight = (
                {"dry_run": True}
                if args.dry_run
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
                        run_prefix(args.mode, variant)
                    )
                    if not args.dry_run:
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
                    if args.dry_run:
                        log_planned_cell(matrix, cell)
                    else:
                        execute_cell(matrix, cell)


if __name__ == "__main__":
    main()
