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
from pathlib import Path
from typing import Any, Mapping

try:
    import eval_utils as eu
    import run_provenance as rp
except ModuleNotFoundError:  # pragma: no cover
    from scripts import eval_utils as eu
    from scripts import run_provenance as rp


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


def run_from_config(run_config: dict[str, Any], args: Any) -> None:
    """Execute a normalized run config.

    Shared by the argparse CLI above and by the Hydra entry point in
    `scripts/run.py`, which composes the same dictionary from `conf/`.
    """
    eu.configure_run_logging(args.log_level)
    audit_mode = eu.normalize_task3_audit_mode(args.audit_mode)
    dry_run = bool(getattr(args, "dry_run", False))

    if args.mode == "resume" and not args.run_id:
        raise ValueError("--mode resume requires --run-id.")

    root = eu.project_root()
    profiles = eu.filter_run_profiles(
        run_config, profile_id=args.profile, model=args.model
    )
    datasets = eu.selected_values(list(run_config["datasets"]), args.dataset, "dataset")
    variants = eu.selected_values(
        list(run_config["benchmark_variants"]), args.variant, "variant"
    )
    logging_config = eu.logging_config_from_args(run_config, args)
    task3_template = eu.load_prompt(task3_prompt_path(root, audit_mode))
    completion_fn = fake_completion if args.fake_completion else eu.chat_completion
    # Smoke/fake Task 3 runs are isolated from the paper-facing artifact tree.
    smoke_tree = (
        bool(args.fake_completion)
        or args.mode == "smoke"
        or eu.is_smoke_run_id(args.run_id)
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
                if dry_run
                else eu.preflight_profile(
                    profile,
                    model=model,
                    prompt_version=run_config["prompt_version"],
                    completion_fn=completion_fn,
                )
            )

            for dataset_id in datasets:
                for variant in variants:
                    prefix = task3_run_prefix(args.mode, variant, audit_mode)
                    run_id = args.run_id or eu.new_run_id(prefix)
                    started_at = eu.utc_now_iso()
                    benchmark_path = eu.artifact_path(
                        root / "data/processed/benchmark_items.csv", dataset_id, variant
                    )
                    # A smoke-* source run lives in the parallel smoke tree.
                    source_raw_path = eu.model_outputs_raw_path(
                        root, dataset_id, variant, run_id=args.source_run_id
                    )
                    output_path = eu.task3_raw_path(
                        root, dataset_id, variant, smoke=smoke_tree
                    )
                    registry_path = eu.task3_registry_path(
                        root, dataset_id, variant, smoke=smoke_tree
                    )
                    progress_path = eu.task3_progress_path(
                        root, dataset_id, variant, smoke=smoke_tree
                    )
                    events_path = eu.task3_events_path(
                        root, dataset_id, variant, smoke=smoke_tree
                    )
                    if not dry_run:
                        eu.configure_run_logging(
                            args.log_level, log_path=eu.run_log_path(root, run_id)
                        )
                        # No-op unless the run was composed by scripts/run.py.
                        rp.write_resolved_config(
                            root, run_id, getattr(args, "resolved_config_yaml", "")
                        )

                    benchmark = eu.read_csv_rows(benchmark_path)
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
                            benchmark,
                            model_source_rows,
                            str(source_run_id),
                            accepted_prefixes,
                            expected_stochastic_samples=int(
                                run_config["stochastic"]["samples"]
                            ),
                        )

                    all_task3_items = eu.build_task3_verification_items(
                        benchmark, model_source_rows, audit_mode=audit_mode
                    )
                    if not all_task3_items:
                        raise ValueError(
                            "No Task 3 text-audit items were built from valid deterministic Task 2 rows."
                        )
                    task3_items_path = eu.task3_verification_items_path(
                        root,
                        dataset_id,
                        variant,
                        str(source_run_id),
                        model,
                        audit_mode,
                        smoke=smoke_tree,
                    )
                    if not dry_run:
                        eu.write_csv_rows(
                            task3_items_path,
                            all_task3_items,
                            fieldnames=eu.TASK3_VERIFICATION_FIELDS,
                        )
                    task3_items_for_run = (
                        all_task3_items[: max(1, args.smoke_items)]
                        if args.mode == "smoke"
                        else all_task3_items
                    )

                    stochastic_samples = int(run_config["stochastic"]["samples"])
                    jobs = eu.planned_completion_jobs_for_items(
                        task3_items_for_run,
                        prompt_fn=lambda item: task3_prompt_for(
                            task3_template, item, audit_mode=audit_mode
                        ),
                        prompt_version=f"{run_config['prompt_version']}:task3:{audit_mode}",
                        model=model,
                        host=profile["base_url"],
                        run_id=run_id,
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
                    pending_jobs = eu.pending_completion_jobs(
                        jobs, existing_rows, run_id
                    )
                    total_api_calls = len(
                        eu.completion_job_batches(jobs, int(profile["batch_size"]))
                    )
                    pending_api_calls = len(
                        eu.completion_job_batches(
                            pending_jobs,
                            int(profile["batch_size"]),
                            planned_jobs=jobs,
                        )
                    )
                    if dry_run:
                        eu.logger.info(
                            "%s",
                            {
                                "dry_run": True,
                                "run_id": run_id,
                                "source_run_id": source_run_id,
                                "dataset_id": dataset_id,
                                "variant": variant,
                                "profile": profile["profile_id"],
                                "model": model,
                                "audit_mode": audit_mode,
                                "task3_items": len(task3_items_for_run),
                                "planned_jobs": len(jobs),
                                "pending_jobs": len(pending_jobs),
                                "planned_batches": total_api_calls,
                                "pending_api_calls": pending_api_calls,
                                "batch_size": profile["batch_size"],
                                "batch_order": batch_order,
                                "output_path": str(output_path),
                                "task3_items_path": str(task3_items_path),
                            },
                        )
                        continue
                    current_rows = list(existing_rows)
                    started_monotonic = time.monotonic()
                    emitted_warning_types: set[str] = set()

                    def refresh_live_progress(
                        event_type: str,
                        finished_at_utc: str = "",
                        print_line: bool = True,
                    ) -> dict[str, Any]:
                        run_rows = eu.select_model_run_rows(
                            current_rows, run_id, model, ("task3",)
                        )
                        pending_jobs_now = eu.pending_completion_jobs(
                            jobs, current_rows, run_id
                        )
                        # `planned_jobs` batches over the full plan and then keeps the
                        # pending slice, so a resumed run never re-shuffles the batches.
                        pending_api_calls_now = len(
                            eu.completion_job_batches(
                                pending_jobs_now,
                                int(profile["batch_size"]),
                                planned_jobs=jobs,
                            )
                        )
                        status = (
                            "running"
                            if event_type in {"start", "progress"} and pending_jobs_now
                            else None
                        )
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
                            structured_output=str(
                                profile.get("structured_output", "none")
                            ),
                            request_extra_body=profile.get("extra_body"),
                            server_model_probe=preflight,
                            batch_order=batch_order,
                            notes=rp.run_notes(
                                args,
                                f"audit_mode={audit_mode}; source_run_id={source_run_id}",
                            ),
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
                            "audit_mode": audit_mode,
                            "output_path": str(output_path),
                            "task3_items_path": str(task3_items_path),
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
                            eu.logger.info(
                                "%s", eu.format_live_progress_line(run_id, counters)
                            )
                        if event_type == "finish":
                            eu.logger.info(
                                "%s",
                                eu.format_run_quality_line(
                                    run_id, eu.run_quality_counters(run_rows)
                                ),
                            )
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
                            structured_output=str(
                                profile.get("structured_output", "none")
                            ),
                            request_extra_body=profile.get("extra_body"),
                            server_model_probe=preflight,
                            batch_order=batch_order,
                            notes=rp.run_notes(
                                args,
                                f"audit_mode={audit_mode}; source_run_id={source_run_id}",
                            ),
                        ),
                    )
                    refresh_live_progress("start", print_line=False)
                    eu.logger.info(
                        "%s",
                        {
                            "run_id": run_id,
                            "source_run_id": source_run_id,
                            "dataset_id": dataset_id,
                            "variant": variant,
                            "profile": profile["profile_id"],
                            "model": model,
                            "audit_mode": audit_mode,
                            "task3_items": len(task3_items_for_run),
                            "planned_jobs": len(jobs),
                            "pending_jobs": len(pending_jobs),
                            "batch_size": profile["batch_size"],
                            "batch_order": batch_order,
                            "seed": seed,
                            "send_seed": send_seed,
                            "max_retries": max_retries,
                            "output_path": str(output_path),
                            "task3_items_path": str(task3_items_path),
                            "registry_path": str(registry_path),
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
                        eu.append_jsonl(output_path, record)
                        current_rows.append(record)
                        now_monotonic = time.monotonic()
                        records_due = index - last_progress_record_index >= int(
                            logging_config["progress_every_records"]
                        )
                        seconds_due = int(
                            logging_config["progress_every_seconds"]
                        ) > 0 and now_monotonic - last_progress_monotonic >= int(
                            logging_config["progress_every_seconds"]
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
                        notes=rp.run_notes(
                            args,
                            f"audit_mode={audit_mode}; source_run_id={source_run_id}",
                        ),
                    )
                    eu.upsert_run_registry_row(registry_path, finish_row)
                    refresh_live_progress(
                        "finish", finished_at_utc=str(finish_row["finished_at_utc"])
                    )
                    eu.logger.info(
                        "Task 3 registry status: %s at %s",
                        finish_row["status"],
                        registry_path,
                    )


if __name__ == "__main__":
    main()
