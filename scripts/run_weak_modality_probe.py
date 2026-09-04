"""Run the weak-phrasing probe: four weak templates over the pilot seeds.

The benchmark renders weak stakeholder intent with one fixed template ("It
would be useful if the system could ..."), so a strengthening rate measured on
it could be a fact about that phrase rather than about weak intent. The probe
re-asks Task 2 with four weak phrasings of the same capabilities and reports
what modality each one is read as.

It used to exist only inside ``notebooks/02b_weak_modality_robustness_probe.ipynb``,
which read the legacy ``config.json``, planned its own requests, and wrote raw
rows with no registry row, no resume, and no lease -- so a probe run could not
be continued, audited, or driven from the same place as everything else. This
runs it through the shared cell lifecycle instead: same run config, same
registry, same resume protocol, same request transcripts.

The construct sanity check gates execution: every template must be marked
``weaker_than_should=yes`` in ``outputs/weak_modality_template_sanity_check.csv``
before any request is sent, because a template that is not weaker than SHOULD
does not measure weak intent at all.

    .venv/bin/python scripts/run_weak_modality_probe.py \\
        --config run_configs/current_run.json --profile zai --model glm-5.1 \\
        --dataset nice --mode full
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import eval_utils as eu
    import run_provenance as rp
    import run_transcripts as rt
    import runner_lifecycle as rl
    from runner_args import RunnerArgs
except ModuleNotFoundError:  # pragma: no cover - invocation-path fallback
    from scripts import (
        eval_utils as eu,
        run_provenance as rp,
        run_transcripts as rt,
        runner_lifecycle as rl,
    )
    from scripts.runner_args import RunnerArgs


PROBE_RUN_PREFIX = "weak-modality-probe"
PROBE_TASKS = ("task2",)


class ProbeSanityError(RuntimeError):
    """The template construct review is incomplete, so nothing is sent."""


def probe_run_prefix(mode: str, variant: str) -> str:
    base = PROBE_RUN_PREFIX if variant == "must" else f"{PROBE_RUN_PREFIX}-{variant}"
    return f"{base}-smoke" if mode == "smoke" else base


def probe_items_path(root: Path, dataset_id: str, variant: str, *, run_id: str) -> Path:
    return eu.resolve_run_artifact_path(
        eu.artifact_path(
            root / "data/processed/weak_modality_probe_items.csv", dataset_id, variant
        ),
        run_id=run_id,
    )


def probe_summary_dir(root: Path, run_id: str) -> Path:
    """`outputs/` for a real run, `outputs/smoke/` for a smoke or fake one.

    Same split the per-cell analysis makes: a smoke run must never overwrite a
    paper-facing summary.
    """
    return (
        root / "outputs" / "smoke" if eu.is_smoke_run_id(run_id) else root / "outputs"
    )


def probe_raw_path(root: Path, dataset_id: str, variant: str, *, run_id: str) -> Path:
    return eu.resolve_run_artifact_path(
        eu.artifact_path(
            root / "data/processed/model_outputs_raw_weak_modality_probe.jsonl",
            dataset_id,
            variant,
        ),
        run_id=run_id,
    )


def probe_registry_path(root: Path, dataset_id: str, variant: str, *, run_id: str):
    return eu.resolve_run_artifact_path(
        eu.artifact_path(
            root / "data/processed/run_registry_weak_modality_probe.csv",
            dataset_id,
            variant,
        ),
        run_id=run_id,
    )


def pilot_seeds(root: Path, dataset_id: str, variant: str, count: int) -> list[dict]:
    """The first `count` benchmark seeds, in benchmark order.

    Same selection the notebook made: the probe re-phrases capabilities the
    benchmark already carries, so its items line up with the pilot cells.
    """
    benchmark = eu.read_csv_rows(
        eu.artifact_path(
            root / "data/processed/benchmark_items.csv", dataset_id, variant
        )
    )
    seed_rows = eu.read_csv_rows(
        eu.artifact_path(root / "data/processed/seeds_selected.csv", dataset_id)
    )
    wanted = sorted({row["seed_id"] for row in benchmark})[:count]
    order = {seed_id: index for index, seed_id in enumerate(wanted)}
    seeds = sorted(
        (row for row in seed_rows if row["seed_id"] in order),
        key=lambda row: order[row["seed_id"]],
    )
    if len(seeds) != len(wanted):
        missing = sorted(set(wanted) - {row["seed_id"] for row in seeds})
        raise ValueError(
            f"seeds_selected.csv is missing {len(missing)} pilot seed(s): "
            f"{', '.join(missing[:5])}"
        )
    return seeds


def require_sanity_check(root: Path, dataset_id: str) -> dict[str, Any]:
    """Refuse to send requests until every template is reviewed as weak."""
    paths = eu.write_weak_modality_template_sanity_check(
        root / "outputs", suffix=eu.dataset_suffix(dataset_id)
    )
    status = eu.weak_modality_sanity_status(eu.read_csv_rows(paths["csv"]))
    if not status.get("valid"):
        raise ProbeSanityError(
            f"{paths['csv']} does not mark every template as weaker than SHOULD "
            f"({status}). Review the templates before running the probe."
        )
    return status


@dataclass
class ProbeCell:
    """One planned probe cell; satisfies `runner_lifecycle.LifecycleCell`."""

    profile: dict[str, Any]
    model: str
    dataset_id: str
    variant: str
    run_id: str
    started_at: str
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
        return {
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "benchmark_variant": self.variant,
            "provider_id": self.provider_id,
            "profile_id": self.profile_id,
            "model": self.model,
        }


def plan_cell(
    root: Path,
    run_config: dict[str, Any],
    profile: dict[str, Any],
    model: str,
    dataset_id: str,
    variant: str,
    args: RunnerArgs,
    *,
    run_id: str,
    preflight: dict[str, Any],
    resume: rl.ResumeState | None,
) -> ProbeCell:
    """Build the probe items and the requests that answer them."""
    seed_count = int(eu.DEFAULT_CONFIG["project"]["pilot_seed_count"])
    seeds = pilot_seeds(root, dataset_id, variant, seed_count)
    items = eu.build_weak_modality_probe_items(seeds)
    if args.mode == "smoke":
        items = items[: max(1, int(args.smoke_items))]
    items_path = probe_items_path(root, dataset_id, variant, run_id=run_id)
    if not args.dry_run:
        eu.write_csv_rows(items_path, items, fieldnames=eu.WEAK_MODALITY_PROBE_FIELDS)

    output_path = probe_raw_path(root, dataset_id, variant, run_id=run_id)
    existing_rows = eu.read_jsonl(output_path) if output_path.exists() else []
    jobs = eu.planned_completion_jobs(
        items,
        tasks=PROBE_TASKS,
        model=model,
        host=str(profile["base_url"]),
        run_id=run_id,
        prompt_version=str(run_config["prompt_version"]),
        task1_template=eu.load_prompt("prompts/mandatory_entailment.txt"),
        task2_template=eu.load_prompt("prompts/modality_extraction.txt"),
        deterministic=run_config["deterministic"],
        stochastic=run_config["stochastic"],
        max_tokens=int(profile["max_tokens"]),
        timeout_s=int(profile["timeout_s"]),
        api_key_env=str(profile["api_key_env"]),
        provider_id=str(profile["provider_id"]),
        profile_id=str(profile["profile_id"]),
        run_group_id=str(run_config["run_group_id"]),
        json_mode=bool(profile["json_mode"]),
        structured_output=str(profile.get("structured_output", "none")),
        extra_body=profile.get("extra_body"),
        seed=int(profile["seed"]),
        send_seed=bool(profile["send_seed"]),
        max_retries=int(profile["max_retries"]),
        batch_order=str(profile["batch_order"]),
        batch_size=int(profile["batch_size"]),
        server_model_probe=preflight,
    )
    pending_jobs = eu.pending_completion_jobs(jobs, existing_rows, run_id)
    return ProbeCell(
        profile=profile,
        model=model,
        dataset_id=dataset_id,
        variant=variant,
        run_id=run_id,
        started_at=resume.started_at_utc if resume else eu.utc_now_iso(),
        batch_order=str(profile["batch_order"]),
        preflight=preflight,
        items=items,
        jobs=jobs,
        existing_rows=existing_rows,
        pending_jobs=pending_jobs,
        planned_api_calls=len(
            eu.completion_job_batches(jobs, int(profile["batch_size"]))
        ),
        # Batching over the full plan keeps a resumed run's batch membership.
        pending_api_calls=len(
            eu.completion_job_batches(
                pending_jobs, int(profile["batch_size"]), planned_jobs=jobs
            )
        ),
        output_path=output_path,
        registry_path=probe_registry_path(root, dataset_id, variant, run_id=run_id),
        progress_path=eu.resolve_run_artifact_path(
            eu.artifact_path(
                root / "data/processed/run_progress_live_weak_modality_probe.csv",
                dataset_id,
                variant,
            ),
            run_id=run_id,
        ),
        events_path=eu.resolve_run_artifact_path(
            eu.artifact_path(
                root / "data/processed/run_events_weak_modality_probe.jsonl",
                dataset_id,
                variant,
            ),
            run_id=run_id,
        ),
        items_path=items_path,
        resume=resume,
    )


def registry_row(
    run_config: dict[str, Any],
    cell: ProbeCell,
    args: RunnerArgs,
    raw_rows: list[dict[str, Any]],
    *,
    status: str | None = None,
    finished_at_utc: str = "",
) -> dict[str, Any]:
    profile = cell.profile
    return eu.run_registry_summary(
        cell.items,
        raw_rows,
        run_id=cell.run_id,
        run_group_id=str(run_config["run_group_id"]),
        provider_id=cell.provider_id,
        profile_id=cell.profile_id,
        model=cell.model,
        dataset_id=cell.dataset_id,
        variant=cell.variant,
        tasks=list(PROBE_TASKS),
        expected_stochastic_samples=int(run_config["stochastic"]["samples"]),
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
        notes=cell.resume.notes
        if cell.resume
        else rp.run_notes(args, "probe=weak_modality"),
    )


def write_summary(root: Path, dataset_id: str, cell: ProbeCell) -> dict[str, Path]:
    """Per-template summary of what each weak phrasing was read as."""
    rows = eu.read_jsonl(cell.output_path) if cell.output_path.exists() else []
    rows = [row for row in rows if str(row.get("run_id", "")) == cell.run_id]
    summary = eu.weak_modality_probe_summary(cell.items, rows)
    return eu.write_weak_modality_probe_summary(
        summary,
        probe_summary_dir(root, cell.run_id),
        suffix=eu.dataset_suffix(dataset_id),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        parents=[rl.common_runner_parser()],
    )
    # Declared here rather than in the shared parent for the same reason the
    # other two runners do it: the help text names what this runner would send.
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the probe items and print the request plan without sending it.",
    )
    return parser


def run_from_config(run_config: dict[str, Any], args: RunnerArgs) -> None:
    """Run the probe for one profile/model/dataset/variant cell."""
    root = eu.project_root()
    profiles = eu.filter_run_profiles(run_config, args.profile, args.model)
    profile = profiles[0]
    model = args.model or str(profile["models"][0])
    dataset_id = eu.selected_values(
        list(run_config["datasets"]), args.dataset, "dataset"
    )[0]
    variant = eu.selected_values(
        list(run_config["benchmark_variants"]), args.variant, "benchmark variant"
    )[0]
    require_sanity_check(root, dataset_id)

    run_id = args.run_id or eu.new_run_id(probe_run_prefix(args.mode, variant))
    resume = None
    if args.mode == "resume":
        resume = rl.resolve_resume(
            root,
            probe_registry_path(root, dataset_id, variant, run_id=run_id),
            run_id=run_id,
            provider_id=str(profile["provider_id"]),
            profile_id=str(profile["profile_id"]),
            model=model,
            dataset_id=dataset_id,
            variant=variant,
            tasks=PROBE_TASKS,
            record=not args.dry_run,
        )
    rl.prepare_run_directory(root, run_id, args, resume=resume)

    completion_fn = eu.chat_completion
    preflight: dict[str, Any] = {"dry_run": True}
    if args.fake_completion:
        from run_experiment_from_config import fake_completion

        completion_fn = fake_completion
        preflight = {"fake_completion": True}
    elif not args.dry_run:
        preflight = eu.preflight_profile(
            profile,
            model=model,
            prompt_version=str(run_config["prompt_version"]),
        )

    cell = plan_cell(
        root,
        run_config,
        profile,
        model,
        dataset_id,
        variant,
        args,
        run_id=run_id,
        preflight=preflight,
        resume=resume,
    )
    if args.dry_run:
        eu.logger.info(
            "%s",
            {
                "dry_run": True,
                "run_id": cell.run_id,
                "probe_items": len(cell.items),
                "planned_jobs": len(cell.jobs),
                "planned_api_calls": cell.planned_api_calls,
                "output_path": str(cell.output_path),
            },
        )
        return

    logging_config = eu.logging_config_from_args(run_config, args)
    rl.execute_cell(
        rl.CellExecution(
            cell=cell,
            tasks=PROBE_TASKS,
            progress_items=cell.items,
            expected_stochastic_samples=int(run_config["stochastic"]["samples"]),
            logging_config=logging_config,
            completion_fn=completion_fn,
            context=cell.identity(),
            mode=args.mode,
            registry_row=lambda raw_rows, **kwargs: registry_row(
                run_config, cell, args, raw_rows, **kwargs
            ),
            start_log={
                "run_id": cell.run_id,
                "dataset_id": cell.dataset_id,
                "variant": cell.variant,
                "profile": cell.profile_id,
                "model": cell.model,
                "probe_items": len(cell.items),
                "templates": len(eu.WEAK_MODALITY_PROBE_TEMPLATES),
                "planned_jobs": len(cell.jobs),
                "pending_jobs": len(cell.pending_jobs),
                "batch_size": cell.batch_size,
                "output_path": str(cell.output_path),
                "items_path": str(cell.items_path),
            },
            event_fields={"probe": "weak_modality"},
            registry_label="Weak-modality probe registry status",
            transcript=rt.TranscriptWriter.for_run(
                root,
                cell.run_id,
                enabled=bool(logging_config["write_request_transcripts"]),
            ),
            lease=rl.CellLease.for_cell(
                root,
                run_id=cell.run_id,
                profile_id=cell.profile_id,
                model=cell.model,
                dataset_id=cell.dataset_id,
                variant=cell.variant,
            ),
        )
    )
    paths = write_summary(root, dataset_id, cell)
    eu.logger.info("Probe summary: %s", paths["csv"])


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    eu.configure_run_logging(args.log_level)
    run_from_config(eu.load_run_config(args.config), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
