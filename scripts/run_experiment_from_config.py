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
                        "confidence": 90,
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
                        "confidence": 90,
                    }
                )
        raw_text = json.dumps({"results": results})
    elif "Candidate requirement:" in prompt or '"decision"' in prompt:
        raw_text = '{"decision":"yes","confidence":90,"brief_reason":"fake smoke"}'
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
            + f'","modality":"{modality}","confidence":90}}'
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
    args = parser.parse_args()

    root = eu.project_root()
    run_config = eu.load_run_config(args.config)
    profiles = eu.filter_run_profiles(run_config, profile_id=args.profile, model=args.model)
    datasets = selected_values(list(run_config["datasets"]), args.dataset, "dataset")
    variants = selected_values(list(run_config["benchmark_variants"]), args.variant, "variant")
    tasks = eu.normalize_task_filter(args.task or run_config["tasks"])

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
                response_format=profile.get("response_format"),
                extra_body=profile.get("extra_body"),
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
                        response_format=profile.get("response_format"),
                        extra_body=profile.get("extra_body"),
                        server_model_probe=preflight,
                    )
                    existing_rows = eu.read_jsonl(output_path)
                    pending_jobs = eu.pending_completion_jobs(jobs, existing_rows, run_id)
                    start_row = eu.run_registry_summary(
                        benchmark_for_run,
                        existing_rows,
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
                        request_extra_body=profile.get("extra_body"),
                        server_model_probe=preflight,
                        notes=f"mode={args.mode}",
                    )
                    eu.upsert_run_registry_row(registry_path, start_row)

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
                            "planned_api_calls": len(eu.completion_job_batches(pending_jobs, int(profile["batch_size"]))),
                            "output_path": str(output_path),
                        }
                    )
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
                        if index % 100 == 0 or index == len(pending_jobs):
                            print(f"{run_id}: completed {index}/{len(pending_jobs)} pending calls")

                    finished_rows = eu.read_jsonl(output_path)
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
                        request_extra_body=profile.get("extra_body"),
                        server_model_probe=preflight,
                        notes=f"mode={args.mode}",
                    )
                    eu.upsert_run_registry_row(registry_path, finish_row)
                    print(f"Registry status: {finish_row['status']} at {registry_path}")


if __name__ == "__main__":
    main()
