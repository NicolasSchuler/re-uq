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
from pathlib import Path
from typing import Any

try:
    import eval_utils as eu
    import run_provenance as rp
except ModuleNotFoundError:  # pragma: no cover
    from scripts import eval_utils as eu, run_provenance as rp


def run_prefix(mode: str, variant: str) -> str:
    if mode == "smoke":
        return "smoke" if variant == "must" else f"smoke-{variant}"
    return "full" if variant == "must" else f"full-{variant}"


def source_from_prompt(prompt: str) -> str:
    match = re.search(r'Source:\s*\n?"([^"]+)"', prompt, flags=re.S)
    return match.group(1) if match else prompt


def fake_completion(**kwargs: Any) -> dict[str, Any]:
    prompt = str(kwargs.get("prompt", ""))
    if '"results"' in prompt and "Items:\n" in prompt:
        items = json.loads(prompt.split("Items:\n", 1)[1])
        results: list[dict[str, Any]] = []
        for item in items:
            request_index = int(item["request_index"])
            if "candidate_requirement" in item:
                results.append(
                    {
                        "request_index": request_index,
                        "decision": "yes",
                        "confidence": 0.9,
                        "brief_reason": "fake smoke",
                    }
                )
            else:
                source = str(item["source_statement"]).lower()
                if "must " in source or "shall " in source:
                    modality = "mandatory"
                elif "should " in source:
                    modality = "recommended"
                elif "may " in source:
                    modality = "optional"
                else:
                    modality = "nice_to_have"
                results.append(
                    {
                        "request_index": request_index,
                        "requirement": item["source_statement"],
                        "modality": modality,
                        "confidence": 0.9,
                    }
                )
        raw_text = json.dumps({"results": results})
    elif "Candidate requirement:" in prompt or '"decision"' in prompt:
        raw_text = '{"decision":"yes","confidence":0.9,"brief_reason":"fake smoke"}'
    else:
        source = source_from_prompt(prompt).lower()
        if "must " in source or "shall " in source:
            modality = "mandatory"
        elif "should " in source:
            modality = "recommended"
        elif "may " in source:
            modality = "optional"
        else:
            modality = "nice_to_have"
        raw_text = (
            '{"requirement":"'
            + source_from_prompt(prompt).replace("\\", "\\\\").replace('"', '\\"')
            + f'","modality":"{modality}","confidence":0.9}}'
        )
    return {
        "ok": True,
        "raw_text": raw_text,
        "response_json": {"fake": True, "model": kwargs.get("model", "")},
        "latency_s": 0.0,
        "error": "",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run provider-aware benchmark experiments from a run config.")
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
    parser.add_argument("--log-level", default="INFO", help="Logging level for the re_uq logger (default: INFO).")
    args = parser.parse_args(argv)
    eu.configure_run_logging(args.log_level)
    run_from_config(eu.load_run_config(args.config), args)


def run_from_config(run_config: dict[str, Any], args: Any) -> None:
    """Execute a normalized run config.

    Shared by the argparse CLI above and by the Hydra entry point in
    `scripts/run.py`, which composes the same dictionary from `conf/`.
    """
    eu.configure_run_logging(args.log_level)
    semantic_embedding_backend, _ = eu.semantic_embedding_backend_label(
        run_config.get("acse_embedding_backend"),
        run_config.get("acse_embedding_mlx_model"),
    )
    root = eu.project_root()
    model_filter = None if args.all_models else args.model
    profiles = eu.filter_run_profiles(run_config, profile_id=args.profile, model=model_filter)
    datasets = eu.selected_values(list(run_config["datasets"]), args.dataset, "dataset")
    variants = eu.selected_values(list(run_config["benchmark_variants"]), args.variant, "variant")
    tasks = eu.normalize_task_filter(args.task or run_config["tasks"])
    logging_config = eu.logging_config_from_args(run_config, args)

    if args.mode == "resume" and not args.run_id:
        raise ValueError("--mode resume requires --run-id.")

    task1_template = eu.load_prompt(root / "prompts/mandatory_entailment.txt")
    task2_template = eu.load_prompt(root / "prompts/modality_extraction.txt")
    completion_fn = fake_completion if args.fake_completion else eu.chat_completion
    # Fake completions and smoke run ids never touch the paper-facing artifacts;
    # they are written to the parallel data/processed/smoke/ tree instead.
    smoke_tree = bool(args.fake_completion) or eu.is_smoke_run_id(args.run_id) or args.mode == "smoke"

    for profile in profiles:
        eu.validate_manual_server_profile(profile)
        seed = int(profile.get("seed", run_config.get("seed", eu.DEFAULT_REQUEST_SEED)))
        send_seed = bool(profile.get("send_seed", True))
        max_retries = int(profile.get("max_retries", eu.DEFAULT_MAX_RETRIES))
        batch_order = eu.normalize_batch_order(
            profile.get("batch_order", run_config.get("batch_order", eu.DEFAULT_BATCH_ORDER))
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
                    completion_fn=completion_fn,
                )
            )

            for dataset_id in datasets:
                for variant in variants:
                    prefix = run_prefix(args.mode, variant)
                    run_id = args.run_id or eu.new_run_id(prefix)
                    started_at = eu.utc_now_iso()
                    benchmark_path = eu.artifact_path(root / "data/processed/benchmark_items.csv", dataset_id, variant)
                    smoke_run = smoke_tree or eu.is_smoke_run_id(run_id)
                    output_path = eu.model_outputs_raw_path(root, dataset_id, variant, smoke=smoke_run)
                    registry_path = eu.run_registry_path(root, dataset_id, variant, smoke=smoke_run)
                    progress_path = eu.run_progress_live_path(root, dataset_id, variant, smoke=smoke_run)
                    events_path = eu.run_events_path(root, dataset_id, variant, smoke=smoke_run)
                    if not args.dry_run:
                        eu.configure_run_logging(args.log_level, log_path=eu.run_log_path(root, run_id))
                        # No-op unless the run was composed by scripts/run.py.
                        rp.write_resolved_config(root, run_id, getattr(args, "resolved_config_yaml", ""))
                    benchmark = eu.read_csv_rows(benchmark_path)
                    benchmark_for_run = benchmark[: max(1, args.smoke_items)] if args.mode == "smoke" else benchmark

                    jobs = eu.planned_completion_jobs(
                        benchmark_for_run,
                        tasks=tasks,
                        model=model,
                        host=profile["base_url"],
                        run_id=run_id,
                        prompt_version=run_config["prompt_version"],
                        task1_template=task1_template,
                        task2_template=task2_template,
                        deterministic=run_config["deterministic"],
                        stochastic=run_config["stochastic"],
                        max_tokens=int(profile["max_tokens"]),
                        timeout_s=int(profile["timeout_s"]),
                        api_key_env=profile["api_key_env"],
                        provider_id=profile["provider_id"],
                        profile_id=profile["profile_id"],
                        run_group_id=run_config["run_group_id"],
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
                    existing_rows = eu.read_jsonl(output_path)
                    pending_jobs = eu.pending_completion_jobs(jobs, existing_rows, run_id)
                    total_api_calls = len(eu.completion_job_batches(jobs, int(profile["batch_size"])))
                    # `planned_jobs` batches over the full plan and then keeps the
                    # pending slice, so a resumed run never re-shuffles the batches.
                    pending_api_calls = len(
                        eu.completion_job_batches(pending_jobs, int(profile["batch_size"]), planned_jobs=jobs)
                    )
                    if args.dry_run:
                        eu.logger.info(
                            "%s",
                            {
                                "dry_run": True,
                                "dataset_id": dataset_id,
                                "variant": variant,
                                "profile": profile["profile_id"],
                                "model": model,
                                "planned_jobs": len(jobs),
                                "planned_batches": total_api_calls,
                                "batch_size": profile["batch_size"],
                                "batch_order": batch_order,
                                "seed": seed,
                                "send_seed": send_seed,
                                "max_retries": max_retries,
                                "estimated_api_calls": total_api_calls,
                                "pending_api_calls": pending_api_calls,
                                "output_path": str(output_path),
                            },
                        )
                        continue

                    current_rows = list(existing_rows)
                    started_monotonic = time.monotonic()
                    emitted_warning_types: set[str] = set()

                    def refresh_live_progress(event_type: str, finished_at_utc: str = "", print_line: bool = True) -> dict[str, Any]:
                        run_rows = eu.select_model_run_rows(current_rows, run_id, model, tasks)
                        pending_jobs_now = eu.pending_completion_jobs(jobs, current_rows, run_id)
                        pending_api_calls_now = len(
                            eu.completion_job_batches(pending_jobs_now, int(profile["batch_size"]), planned_jobs=jobs)
                        )
                        status = "running" if event_type in {"start", "progress"} and pending_jobs_now else None
                        registry_row = eu.run_registry_summary(
                            benchmark_for_run,
                            current_rows,
                            run_id=run_id,
                            run_group_id=run_config["run_group_id"],
                            provider_id=profile["provider_id"],
                            profile_id=profile["profile_id"],
                            model=model,
                            dataset_id=dataset_id,
                            variant=variant,
                            tasks=tasks,
                            expected_stochastic_samples=int(run_config["stochastic"]["samples"]),
                            started_at_utc=started_at,
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
                            server_model_probe=preflight,
                            batch_order=batch_order,
                            notes=rp.run_notes(args),
                        )
                        eu.upsert_run_registry_row(registry_path, registry_row)
                        if logging_config["write_progress_csv"]:
                            eu.write_live_progress_csv(
                                progress_path,
                                benchmark_for_run,
                                run_rows,
                                expected_stochastic_samples=int(run_config["stochastic"]["samples"]),
                            )
                        counters = eu.live_run_counters(
                            run_rows,
                            expected_records=len(jobs),
                            expected_api_calls=total_api_calls,
                            started_monotonic=started_monotonic,
                        )
                        event = {
                            "event_type": event_type,
                            "run_id": run_id,
                            "run_group_id": run_config["run_group_id"],
                            "dataset_id": dataset_id,
                            "benchmark_variant": variant,
                            "provider_id": profile["provider_id"],
                            "profile_id": profile["profile_id"],
                            "model": model,
                            "tasks": tasks,
                            "mode": args.mode,
                            "output_path": str(output_path),
                            "registry_path": str(registry_path),
                            "progress_path": str(progress_path),
                            "planned_jobs": len(jobs),
                            "pending_jobs": len(pending_jobs_now),
                            "planned_api_calls": total_api_calls,
                            "pending_api_calls": pending_api_calls_now,
                            **counters,
                        }
                        if logging_config["write_event_jsonl"]:
                            eu.append_run_event(events_path, event)
                        if print_line and event_type in {"progress", "finish"}:
                            eu.logger.info("%s", eu.format_live_progress_line(run_id, counters))
                        if event_type == "finish":
                            eu.logger.info("%s", eu.format_run_quality_line(run_id, eu.run_quality_counters(run_rows)))
                        return counters

                    start_row = eu.run_registry_summary(
                        benchmark_for_run,
                        current_rows,
                        run_id=run_id,
                        run_group_id=run_config["run_group_id"],
                        provider_id=profile["provider_id"],
                        profile_id=profile["profile_id"],
                        model=model,
                        dataset_id=dataset_id,
                        variant=variant,
                        tasks=tasks,
                        expected_stochastic_samples=int(run_config["stochastic"]["samples"]),
                        started_at_utc=started_at,
                        status="running" if pending_jobs else None,
                        base_url=profile["base_url"],
                        api_key_env=profile["api_key_env"],
                        concurrency=profile["concurrency"],
                        batch_size=profile["batch_size"],
                        timeout_s=profile["timeout_s"],
                        json_mode=bool(profile["json_mode"]),
                        structured_output=str(profile.get("structured_output", "none")),
                        request_extra_body=profile.get("extra_body"),
                        server_model_probe=preflight,
                        batch_order=batch_order,
                        notes=rp.run_notes(args),
                    )
                    eu.upsert_run_registry_row(registry_path, start_row)
                    refresh_live_progress("start", print_line=False)

                    eu.logger.info(
                        "%s",
                        {
                            "run_id": run_id,
                            "dataset_id": dataset_id,
                            "variant": variant,
                            "profile": profile["profile_id"],
                            "model": model,
                            "planned_jobs": len(jobs),
                            "pending_jobs": len(pending_jobs),
                            "batch_size": profile["batch_size"],
                            "batch_order": batch_order,
                            "seed": seed,
                            "send_seed": send_seed,
                            "max_retries": max_retries,
                            "planned_api_calls": total_api_calls,
                            "pending_api_calls": pending_api_calls,
                            "events_path": str(events_path),
                            "progress_path": str(progress_path),
                            "output_path": str(output_path),
                        },
                    )
                    last_progress_record_index = 0
                    last_progress_monotonic = time.monotonic()
                    for index, record in enumerate(
                        eu.run_completion_jobs(
                            pending_jobs,
                            max_workers=int(profile["concurrency"]),
                            completion_fn=completion_fn,
                            batch_size=int(profile["batch_size"]),
                            planned_jobs=jobs,
                        ),
                        start=1,
                    ):
                        # Persist the resolved choice on the run artifact. The
                        # analysis process cannot inherit this process's env.
                        record["semantic_embedding_backend"] = semantic_embedding_backend
                        eu.append_jsonl(output_path, record)
                        current_rows.append(record)
                        now_monotonic = time.monotonic()
                        records_due = index - last_progress_record_index >= int(logging_config["progress_every_records"])
                        seconds_due = (
                            int(logging_config["progress_every_seconds"]) > 0
                            and now_monotonic - last_progress_monotonic >= int(logging_config["progress_every_seconds"])
                        )
                        finished_pending = index == len(pending_jobs)
                        if records_due or seconds_due or finished_pending:
                            counters = refresh_live_progress("progress")
                            last_progress_record_index = index
                            last_progress_monotonic = now_monotonic
                            eu.emit_warning_events(
                                counters,
                                logging_config=logging_config,
                                emitted_warning_types=emitted_warning_types,
                                context={
                                    "run_id": run_id,
                                    "run_group_id": run_config["run_group_id"],
                                    "dataset_id": dataset_id,
                                    "benchmark_variant": variant,
                                    "provider_id": profile["provider_id"],
                                    "profile_id": profile["profile_id"],
                                    "model": model,
                                },
                                events_path=events_path,
                            )

                    finished_rows = current_rows
                    finish_row = eu.run_registry_summary(
                        benchmark_for_run,
                        finished_rows,
                        run_id=run_id,
                        run_group_id=run_config["run_group_id"],
                        provider_id=profile["provider_id"],
                        profile_id=profile["profile_id"],
                        model=model,
                        dataset_id=dataset_id,
                        variant=variant,
                        tasks=tasks,
                        expected_stochastic_samples=int(run_config["stochastic"]["samples"]),
                        started_at_utc=started_at,
                        finished_at_utc=eu.utc_now_iso(),
                        base_url=profile["base_url"],
                        api_key_env=profile["api_key_env"],
                        concurrency=profile["concurrency"],
                        batch_size=profile["batch_size"],
                        timeout_s=profile["timeout_s"],
                        json_mode=bool(profile["json_mode"]),
                        structured_output=str(profile.get("structured_output", "none")),
                        request_extra_body=profile.get("extra_body"),
                        server_model_probe=preflight,
                        batch_order=batch_order,
                        notes=rp.run_notes(args),
                    )
                    eu.upsert_run_registry_row(registry_path, finish_row)
                    refresh_live_progress("finish", finished_at_utc=str(finish_row["finished_at_utc"]))
                    eu.logger.info("Registry status: %s at %s", finish_row["status"], registry_path)


if __name__ == "__main__":
    main()
