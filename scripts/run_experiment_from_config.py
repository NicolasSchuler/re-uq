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
except ModuleNotFoundError:  # pragma: no cover
    from scripts import eval_utils as eu


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_rows_for(rows: list[dict[str, Any]], run_id: str, model: str, tasks: list[str]) -> list[dict[str, Any]]:
    task_set = set(tasks)
    return [
        row
        for row in rows
        if str(row.get("run_id", "")) == str(run_id)
        and str(row.get("model", "")) == str(model)
        and str(row.get("task", "")) in task_set
    ]


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


def selected_values(values: list[str], requested: str | None, name: str) -> list[str]:
    if not requested:
        return values
    normalized = eu.normalize_dataset_id(requested) if name == "dataset" else eu.normalize_benchmark_variant(requested)
    if normalized not in values:
        raise ValueError(f"Requested {name} {normalized!r} is not present in the run config.")
    return [normalized]


def main() -> None:
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
    args = parser.parse_args()

    root = eu.project_root()
    run_config = eu.load_run_config(args.config)
    profiles = eu.filter_run_profiles(run_config, profile_id=args.profile, model=args.model)
    datasets = selected_values(list(run_config["datasets"]), args.dataset, "dataset")
    variants = selected_values(list(run_config["benchmark_variants"]), args.variant, "variant")
    tasks = eu.normalize_task_filter(args.task or run_config["tasks"])
    logging_config = eu.normalize_run_logging_config(
        run_config.get("logging"),
        overrides={
            "progress_every_records": args.progress_every_records,
            "progress_every_seconds": args.progress_every_seconds,
            "warn_after_records": args.warn_after_records,
            "warn_parse_failure_rate": args.warn_parse_failure_rate,
            "warn_request_error_rate": args.warn_request_error_rate,
            "write_progress_csv": False if args.no_progress_artifacts else None,
            "write_event_jsonl": False if args.no_progress_artifacts else None,
        },
    )

    if args.mode == "resume" and not args.run_id:
        raise ValueError("--mode resume requires --run-id.")

    task1_template = eu.load_prompt(root / "prompts/mandatory_entailment.txt")
    task2_template = eu.load_prompt(root / "prompts/modality_extraction.txt")
    completion_fn = fake_completion if args.fake_completion else eu.chat_completion

    for profile in profiles:
        eu.validate_manual_server_profile(profile)
        for model in profile["models"]:
            preflight = eu.provider_preflight(
                host=profile["base_url"],
                model=model,
                api_key_env=profile["api_key_env"],
                timeout_s=int(profile["timeout_s"]),
                json_mode=bool(profile["json_mode"]),
                structured_output=str(profile.get("structured_output", "none")),
                response_format=profile.get("response_format"),
                extra_body=profile.get("extra_body"),
                instructor_mode=str(profile.get("instructor_mode", "json")),
                validation_retries=int(profile.get("validation_retries", 2)),
                completion_fn=completion_fn,
            )
            if not preflight["ok"]:
                raise RuntimeError(f"Provider preflight failed for {profile['profile_id']} / {model}: {preflight}")

            for dataset_id in datasets:
                for variant in variants:
                    prefix = run_prefix(args.mode, variant)
                    run_id = args.run_id or eu.new_run_id(prefix)
                    started_at = utc_now()
                    benchmark_path = eu.artifact_path(root / "data/processed/benchmark_items.csv", dataset_id, variant)
                    output_path = eu.artifact_path(root / "data/processed/model_outputs_raw.jsonl", dataset_id, variant)
                    registry_path = eu.run_registry_path(root, dataset_id, variant)
                    progress_path = eu.run_progress_live_path(root, dataset_id, variant)
                    events_path = eu.run_events_path(root, dataset_id, variant)
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
                        server_model_probe=preflight,
                    )
                    existing_rows = eu.read_jsonl(output_path)
                    pending_jobs = eu.pending_completion_jobs(jobs, existing_rows, run_id)
                    total_api_calls = len(eu.completion_job_batches(jobs, int(profile["batch_size"])))
                    pending_api_calls = len(eu.completion_job_batches(pending_jobs, int(profile["batch_size"])))
                    current_rows = list(existing_rows)
                    started_monotonic = time.monotonic()
                    emitted_warning_types: set[str] = set()

                    def refresh_live_progress(event_type: str, finished_at_utc: str = "", print_line: bool = True) -> dict[str, Any]:
                        run_rows = run_rows_for(current_rows, run_id, model, tasks)
                        pending_jobs_now = eu.pending_completion_jobs(jobs, current_rows, run_id)
                        pending_api_calls_now = len(eu.completion_job_batches(pending_jobs_now, int(profile["batch_size"])))
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
                            notes=f"mode={args.mode}",
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
                            print(eu.format_live_progress_line(run_id, counters), flush=True)
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
                        notes=f"mode={args.mode}",
                    )
                    eu.upsert_run_registry_row(registry_path, start_row)
                    refresh_live_progress("start", print_line=False)

                    print(
                        {
                            "run_id": run_id,
                            "dataset_id": dataset_id,
                            "variant": variant,
                            "profile": profile["profile_id"],
                            "model": model,
                            "planned_jobs": len(jobs),
                            "pending_jobs": len(pending_jobs),
                            "batch_size": profile["batch_size"],
                            "planned_api_calls": total_api_calls,
                            "pending_api_calls": pending_api_calls,
                            "events_path": str(events_path),
                            "progress_path": str(progress_path),
                            "output_path": str(output_path),
                        }
                    )
                    last_progress_record_index = 0
                    last_progress_monotonic = time.monotonic()
                    for index, record in enumerate(
                        eu.run_completion_jobs(
                            pending_jobs,
                            max_workers=int(profile["concurrency"]),
                            completion_fn=completion_fn,
                            batch_size=int(profile["batch_size"]),
                        ),
                        start=1,
                    ):
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
                            for warning_event in eu.warning_events_for_counters(counters, logging_config, emitted_warning_types):
                                emitted_warning_types.add(str(warning_event["warning_type"]))
                                warning_event.update(
                                    {
                                        "run_id": run_id,
                                        "run_group_id": run_config["run_group_id"],
                                        "dataset_id": dataset_id,
                                        "benchmark_variant": variant,
                                        "provider_id": profile["provider_id"],
                                        "profile_id": profile["profile_id"],
                                        "model": model,
                                    }
                                )
                                if logging_config["write_event_jsonl"]:
                                    eu.append_run_event(events_path, warning_event)
                                print(f"WARNING {run_id}: {warning_event['message']}", flush=True)

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
                        finished_at_utc=utc_now(),
                        base_url=profile["base_url"],
                        api_key_env=profile["api_key_env"],
                        concurrency=profile["concurrency"],
                        batch_size=profile["batch_size"],
                        timeout_s=profile["timeout_s"],
                        json_mode=bool(profile["json_mode"]),
                        structured_output=str(profile.get("structured_output", "none")),
                        request_extra_body=profile.get("extra_body"),
                        server_model_probe=preflight,
                        notes=f"mode={args.mode}",
                    )
                    eu.upsert_run_registry_row(registry_path, finish_row)
                    refresh_live_progress("finish", finished_at_utc=str(finish_row["finished_at_utc"]))
                    print(f"Registry status: {finish_row['status']} at {registry_path}")


if __name__ == "__main__":
    main()
