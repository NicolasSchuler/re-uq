"""Run Task 1 and Task 2 provider-matrix experiments from JSON config.

The runner is the canonical reproduction entry point for primary benchmark
calls. It writes one raw JSONL record per item/sample and records run status in
local registry/progress artifacts so incomplete runs can be resumed safely.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import eval_utils as eu
    import run_provenance as rp
    import run_transcripts as rt
    import runner_lifecycle as rl
    from runner_args import RunnerArgs
except ModuleNotFoundError:  # pragma: no cover
    from scripts import (
        eval_utils as eu,
        run_provenance as rp,
        run_transcripts as rt,
        runner_lifecycle as rl,
    )
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
    # The synthesized request is reported like a real one so a fake run
    # exercises the transcript sidecar end to end.
    request_kwargs = {
        "model": kwargs.get("model", ""),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": kwargs.get("temperature"),
        "top_p": kwargs.get("top_p"),
        "max_tokens": kwargs.get("max_tokens"),
    }
    if kwargs.get("seed") is not None:
        request_kwargs["seed"] = kwargs["seed"]
    return {
        "ok": True,
        "raw_text": json.dumps(payload),
        "response_json": {"fake": True, "model": kwargs.get("model", "")},
        "latency_s": 0.0,
        "error": "",
        "request_payload": eu.transcript_request_payload(request_kwargs),
        "request_payload_sha": eu.request_payload_sha(request_kwargs),
        "request_seed": kwargs.get("seed"),
        "attempt_errors": [],
    }


def build_parser() -> argparse.ArgumentParser:
    """CLI for this runner: the shared options plus the Task 1/2 ones."""
    parser = argparse.ArgumentParser(
        description="Run provider-aware benchmark experiments from a run config.",
        parents=[rl.common_runner_parser()],
    )
    parser.add_argument("--task", choices=["task1", "task2", "both"], default=None)
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
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
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

    def smoke_for(self, run_id: str) -> bool:
        """Whether `run_id` writes into the parallel smoke artifact tree."""
        return self.smoke_tree or eu.is_smoke_run_id(run_id)


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
    #: Set only under `--mode resume`; carries the first attempt's identity so
    #: this cell reports the original start time and provenance notes.
    resume: rl.ResumeState | None = None

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
    resume: rl.ResumeState | None = None,
) -> RunCell:
    """Resolve one matrix cell: its artifact paths and job plan."""
    args = matrix.args
    smoke_run = matrix.smoke_for(run_id)
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
        # A resumed cell keeps the start time its first attempt recorded.
        started_at=resume.started_at_utc if resume else eu.utc_now_iso(),
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
        resume=resume,
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
        # A resumed cell keeps the notes (and resolved-config digest) of the run
        # it continues instead of restamping them with `mode=resume`.
        notes=cell.resume.notes if cell.resume else rp.run_notes(matrix.args),
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


def cell_execution(matrix: MatrixRun, cell: RunCell) -> rl.CellExecution:
    """Bind this cell's Task 1/2 specifics to the shared lifecycle."""

    def stamp_record(record: dict[str, Any]) -> None:
        # Persist the resolved choice on the run artifact. The analysis process
        # cannot inherit this process's env.
        record["semantic_embedding_backend"] = matrix.semantic_embedding_backend

    return rl.CellExecution(
        cell=cell,
        tasks=matrix.tasks,
        progress_items=cell.benchmark_rows,
        expected_stochastic_samples=matrix.expected_stochastic_samples,
        logging_config=matrix.logging_config,
        completion_fn=matrix.completion_fn,
        context=cell_context(matrix, cell),
        mode=matrix.args.mode,
        registry_row=lambda raw_rows, **kwargs: registry_row(
            matrix, cell, raw_rows, **kwargs
        ),
        start_log={
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
        stamp_record=stamp_record,
        registry_label="Registry status",
        transcript=rt.TranscriptWriter.for_run(
            matrix.root,
            cell.run_id,
            enabled=bool(matrix.logging_config["write_request_transcripts"]),
        ),
        lease=rl.CellLease.for_cell(
            matrix.root,
            run_id=cell.run_id,
            profile_id=cell.profile_id,
            model=cell.model,
            dataset_id=cell.dataset_id,
            variant=cell.variant,
        ),
    )


def execute_cell(matrix: MatrixRun, cell: RunCell) -> None:
    """Run one matrix cell to completion, streaming progress artifacts."""
    rl.execute_cell(cell_execution(matrix, cell))


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
                    resume = (
                        # Fails here, before this run id opens a log file or
                        # touches the registry, if it is not a real run of this
                        # exact cell.
                        rl.resolve_resume(
                            root,
                            eu.run_registry_path(
                                root,
                                dataset_id,
                                variant,
                                smoke=matrix.smoke_for(run_id),
                            ),
                            run_id=run_id,
                            provider_id=profile["provider_id"],
                            profile_id=profile["profile_id"],
                            model=model,
                            dataset_id=dataset_id,
                            variant=variant,
                            tasks=tasks,
                            record=not args.dry_run,
                        )
                        if args.mode == "resume"
                        else None
                    )
                    if not args.dry_run:
                        rl.prepare_run_directory(root, run_id, args, resume=resume)
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
                        resume=resume,
                    )
                    if args.dry_run:
                        log_planned_cell(matrix, cell)
                    else:
                        execute_cell(matrix, cell)


if __name__ == "__main__":
    main()
