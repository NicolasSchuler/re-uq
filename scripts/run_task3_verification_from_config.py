"""Run the Task 3 modality-preservation diagnostic from a provider config.

Task 3 is deliberately downstream of Task 2: it builds verifier items from a
complete deterministic Task 2 run, then asks the same provider/model whether
each extracted requirement preserved, strengthened, weakened, or changed the
source statement. It never rewrites or corrects the Task 2 raw outputs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

try:
    import eval_utils as eu
except ModuleNotFoundError:  # pragma: no cover
    from scripts import eval_utils as eu


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def selected_values(values: list[str], requested: str | None, name: str) -> list[str]:
    if not requested:
        return values
    normalized = eu.normalize_dataset_id(requested) if name == "dataset" else eu.normalize_benchmark_variant(requested)
    if normalized not in values:
        raise ValueError(f"Requested {name} {normalized!r} is not present in the run config.")
    return [normalized]


def source_run_prefix(variant: str) -> str:
    return "full" if variant == "must" else f"full-{variant}"


def task3_run_prefix(mode: str, variant: str) -> str:
    base = "task3" if variant == "must" else f"task3-{variant}"
    return f"{base}-smoke" if mode == "smoke" else base


def task3_raw_path(root: Path, dataset_id: str, variant: str) -> Path:
    return eu.artifact_path(root / "data/processed/model_outputs_raw_task3_verification.jsonl", dataset_id, variant)


def task3_registry_path(root: Path, dataset_id: str, variant: str) -> Path:
    return eu.artifact_path(root / "data/processed/run_registry_task3_verification.csv", dataset_id, variant)


def task3_progress_path(root: Path, dataset_id: str, variant: str) -> Path:
    return eu.artifact_path(root / "data/processed/run_progress_live_task3_verification.csv", dataset_id, variant)


def task3_events_path(root: Path, dataset_id: str, variant: str) -> Path:
    return eu.artifact_path(root / "data/processed/run_events_task3_verification.jsonl", dataset_id, variant)


def run_rows_for(rows: list[dict[str, Any]], run_id: str, model: str) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row.get("run_id", "")) == str(run_id)
        and str(row.get("model", "")) == str(model)
        and str(row.get("task", "")) == "task3"
    ]


def task3_prompt_for(template: str, item: Mapping[str, Any]) -> str:
    return eu.render_prompt(
        template,
        source_statement=item["source_statement"],
        extracted_requirement=item["task2_requirement"],
        extracted_modality=item["task2_modality"],
    )


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


def source_rows_for_model(source_rows: list[dict[str, Any]], model: str, profile_id: str) -> list[dict[str, Any]]:
    model_rows = [row for row in source_rows if str(row.get("model", "")) == str(model)]
    profile_rows = [row for row in model_rows if str(row.get("profile_id", "")) in {"", profile_id}]
    return profile_rows or model_rows


def require_complete_task2_source(
    benchmark: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    source_run_id: str,
    prefix: str,
    expected_stochastic_samples: int,
) -> None:
    progress = eu.run_progress_summary(
        benchmark,
        source_rows,
        expected_stochastic_samples=expected_stochastic_samples,
    )
    complete = eu.complete_run_ids_from_progress(progress, expected_tasks=["task2"], prefix=prefix)
    if source_run_id not in complete:
        raise ValueError(
            "Task 3 requires a complete full Task 2 source run for the selected model. "
            f"Run {source_run_id!r} is not complete for task2."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 3 modality verification from a provider-aware run config.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--profile")
    parser.add_argument("--model")
    parser.add_argument("--dataset")
    parser.add_argument("--variant")
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--mode", choices=["smoke", "full", "resume"], default="smoke")
    parser.add_argument("--run-id")
    parser.add_argument("--smoke-items", type=int, default=2)
    parser.add_argument("--fake-completion", action="store_true")
    parser.add_argument("--allow-partial-source", action="store_true")
    parser.add_argument("--progress-every-records", type=int)
    parser.add_argument("--progress-every-seconds", type=int)
    parser.add_argument("--warn-after-records", type=int)
    parser.add_argument("--warn-parse-failure-rate", type=float)
    parser.add_argument("--warn-request-error-rate", type=float)
    parser.add_argument("--no-progress-artifacts", action="store_true")
    args = parser.parse_args()

    if args.mode == "resume" and not args.run_id:
        raise ValueError("--mode resume requires --run-id.")

    root = eu.project_root()
    run_config = eu.load_run_config(args.config)
    profiles = eu.filter_run_profiles(run_config, profile_id=args.profile, model=args.model)
    datasets = selected_values(list(run_config["datasets"]), args.dataset, "dataset")
    variants = selected_values(list(run_config["benchmark_variants"]), args.variant, "variant")
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
    task3_template = eu.load_prompt(root / "prompts/modality_verification.txt")
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
                    prefix = task3_run_prefix(args.mode, variant)
                    run_id = args.run_id or eu.new_run_id(prefix)
                    started_at = utc_now()
                    benchmark_path = eu.artifact_path(root / "data/processed/benchmark_items.csv", dataset_id, variant)
                    source_raw_path = eu.artifact_path(root / "data/processed/model_outputs_raw.jsonl", dataset_id, variant)
                    task3_items_path = eu.artifact_path(root / "data/processed/task3_verification_items.csv", dataset_id, variant)
                    output_path = task3_raw_path(root, dataset_id, variant)
                    registry_path = task3_registry_path(root, dataset_id, variant)
                    progress_path = task3_progress_path(root, dataset_id, variant)
                    events_path = task3_events_path(root, dataset_id, variant)

                    benchmark = eu.read_csv_rows(benchmark_path)
                    source_run_id, source_rows = eu.select_run_rows(
                        eu.read_jsonl(source_raw_path),
                        run_id=args.source_run_id,
                        prefix=source_run_prefix(variant),
                    )
                    if not source_run_id or not source_rows:
                        raise ValueError(f"No source rows found for Task 2 run {args.source_run_id!r}.")
                    model_source_rows = source_rows_for_model(source_rows, model, str(profile["profile_id"]))
                    if not model_source_rows:
                        raise ValueError(
                            f"Source run {source_run_id!r} has no rows for model {model!r}; "
                            "Task 3 should verify the same model's Task 2 outputs."
                        )
                    if not args.allow_partial_source:
                        require_complete_task2_source(
                            benchmark,
                            model_source_rows,
                            str(source_run_id),
                            source_run_prefix(variant),
                            expected_stochastic_samples=int(run_config["stochastic"]["samples"]),
                        )

                    all_task3_items = eu.build_task3_verification_items(benchmark, model_source_rows)
                    if not all_task3_items:
                        raise ValueError("No Task 3 verification items were built from valid deterministic Task 2 rows.")
                    eu.write_csv_rows(task3_items_path, all_task3_items, fieldnames=eu.TASK3_VERIFICATION_FIELDS)
                    task3_items_for_run = (
                        all_task3_items[: max(1, args.smoke_items)] if args.mode == "smoke" else all_task3_items
                    )

                    jobs: list[dict[str, Any]] = []
                    stochastic_samples = int(run_config["stochastic"]["samples"])
                    for item in task3_items_for_run:
                        prompt = task3_prompt_for(task3_template, item)
                        jobs.append(
                            eu.completion_request_job(
                                item=item,
                                task="task3",
                                model=model,
                                host=profile["base_url"],
                                run_id=run_id,
                                sample_kind="deterministic",
                                sample_index=0,
                                temperature=float(run_config["deterministic"]["temperature"]),
                                top_p=float(run_config["deterministic"]["top_p"]),
                                prompt=prompt,
                                prompt_version=f"{run_config['prompt_version']}:task3",
                                max_tokens=int(profile["max_tokens"]),
                                timeout_s=int(profile["timeout_s"]),
                                api_key_env=profile["api_key_env"],
                                request_index=len(jobs),
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
                        )
                        for sample_index in range(stochastic_samples):
                            jobs.append(
                                eu.completion_request_job(
                                    item=item,
                                    task="task3",
                                    model=model,
                                    host=profile["base_url"],
                                    run_id=run_id,
                                    sample_kind="stochastic",
                                    sample_index=sample_index,
                                    temperature=float(run_config["stochastic"]["temperature"]),
                                    top_p=float(run_config["stochastic"]["top_p"]),
                                    prompt=prompt,
                                    prompt_version=f"{run_config['prompt_version']}:task3",
                                    max_tokens=int(profile["max_tokens"]),
                                    timeout_s=int(profile["timeout_s"]),
                                    api_key_env=profile["api_key_env"],
                                    request_index=len(jobs),
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
                            )

                    existing_rows = eu.read_jsonl(output_path)
                    pending_jobs = eu.pending_completion_jobs(jobs, existing_rows, run_id)
                    total_api_calls = len(eu.completion_job_batches(jobs, int(profile["batch_size"])))
                    current_rows = list(existing_rows)
                    started_monotonic = time.monotonic()
                    emitted_warning_types: set[str] = set()

                    def refresh_live_progress(event_type: str, finished_at_utc: str = "", print_line: bool = True) -> dict[str, Any]:
                        run_rows = run_rows_for(current_rows, run_id, model)
                        pending_jobs_now = eu.pending_completion_jobs(jobs, current_rows, run_id)
                        pending_api_calls_now = len(eu.completion_job_batches(pending_jobs_now, int(profile["batch_size"])))
                        status = "running" if event_type in {"start", "progress"} and pending_jobs_now else None
                        registry_row = eu.run_registry_summary(
                            task3_items_for_run,
                            current_rows,
                            run_id=run_id,
                            run_group_id=run_config["run_group_id"],
                            provider_id=profile["provider_id"],
                            profile_id=profile["profile_id"],
                            model=model,
                            dataset_id=dataset_id,
                            variant=variant,
                            tasks=["task3"],
                            expected_stochastic_samples=stochastic_samples,
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
                            notes=f"mode={args.mode}; source_run_id={source_run_id}",
                        )
                        eu.upsert_run_registry_row(registry_path, registry_row)
                        if logging_config["write_progress_csv"]:
                            eu.write_live_progress_csv(
                                progress_path,
                                task3_items_for_run,
                                run_rows,
                                expected_stochastic_samples=stochastic_samples,
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
                            "source_run_id": source_run_id,
                            "dataset_id": dataset_id,
                            "benchmark_variant": variant,
                            "provider_id": profile["provider_id"],
                            "profile_id": profile["profile_id"],
                            "model": model,
                            "tasks": ["task3"],
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

                    eu.upsert_run_registry_row(
                        registry_path,
                        eu.run_registry_summary(
                            task3_items_for_run,
                            current_rows,
                            run_id=run_id,
                            run_group_id=run_config["run_group_id"],
                            provider_id=profile["provider_id"],
                            profile_id=profile["profile_id"],
                            model=model,
                            dataset_id=dataset_id,
                            variant=variant,
                            tasks=["task3"],
                            expected_stochastic_samples=stochastic_samples,
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
                            notes=f"mode={args.mode}; source_run_id={source_run_id}",
                        ),
                    )
                    refresh_live_progress("start", print_line=False)
                    print(
                        {
                            "run_id": run_id,
                            "source_run_id": source_run_id,
                            "dataset_id": dataset_id,
                            "variant": variant,
                            "profile": profile["profile_id"],
                            "model": model,
                            "task3_items": len(task3_items_for_run),
                            "planned_jobs": len(jobs),
                            "pending_jobs": len(pending_jobs),
                            "batch_size": profile["batch_size"],
                            "output_path": str(output_path),
                            "registry_path": str(registry_path),
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

                    finish_row = eu.run_registry_summary(
                        task3_items_for_run,
                        current_rows,
                        run_id=run_id,
                        run_group_id=run_config["run_group_id"],
                        provider_id=profile["provider_id"],
                        profile_id=profile["profile_id"],
                        model=model,
                        dataset_id=dataset_id,
                        variant=variant,
                        tasks=["task3"],
                        expected_stochastic_samples=stochastic_samples,
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
                        notes=f"mode={args.mode}; source_run_id={source_run_id}",
                    )
                    eu.upsert_run_registry_row(registry_path, finish_row)
                    refresh_live_progress("finish", finished_at_utc=str(finish_row["finished_at_utc"]))
                    print(f"Task 3 registry status: {finish_row['status']} at {registry_path}")


if __name__ == "__main__":
    main()
