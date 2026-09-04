"""Run the whole rerun: cohort, Task 3, ablations, analysis -- one command.

Driving a full rerun by hand means about eight commands across two config
systems, in an order where a mistake is invisible until the paper numbers come
out wrong: a Task 3 run pointed at a superseded Task 2 run, an export that
silently selected the archived run group, an aggregator whose snapshot
regeneration cannot retarget a group. This walks the whole sequence in the
right order, records what it did, and refuses to start the analysis stage
against an incomplete cohort.

    .venv/bin/python scripts/rerun_all.py                 # everything
    .venv/bin/python scripts/rerun_all.py --dry-run       # print the plan
    .venv/bin/python scripts/rerun_all.py --only analysis # one stage

It adds no execution logic: every step shells out to the CLI that already owns
it (`scripts/run.py` for provider runs, the analysis scripts for everything
after), so a stage can always be re-run by hand exactly as printed.

**Resumable.** `outputs/rerun_state.json` records each cell's status and run id.
Re-invoking the command skips what is done and continues. A cell that fails is
retried once as `mode=resume` on the same run id -- never as a fresh `mode=full`,
which would re-request everything it already paid for -- and then left failed
so the rest of the cohort still runs. The failures are listed at the end.

**Config.** `conf/rerun/default.yaml` says which profiles are the cohort, which
are local, and which models carry the ablations. Model lists, endpoints and key
variables stay in `conf/profile/<id>.yaml`. The resolved run config the
JSON-based tools need is generated into `outputs/rerun/run_config.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

try:
    import eval_utils as eu
    import task3_sources as ts
except ModuleNotFoundError:  # pragma: no cover - invocation-path fallback
    from scripts import eval_utils as eu, task3_sources as ts


DEFAULT_RERUN_CONFIG = Path("conf/rerun/default.yaml")
STATE_NAME = "rerun_state.json"
GENERATED_RUN_CONFIG = Path("outputs/rerun/run_config.json")
STAGES = ("preflight", "cohort", "task3", "ablations", "analysis")


class StageError(RuntimeError):
    """A stage cannot run; the driver stops rather than guessing."""


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class RerunState:
    """What has already run, so a second invocation continues instead of repeating."""

    path: Path
    cells: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: A dry run keeps its bookkeeping in memory -- printing a plan must never
    #: mark cells complete, or the next real invocation would skip them.
    dry_run: bool = False

    @classmethod
    def load(cls, path: Path, *, dry_run: bool = False) -> RerunState:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls(path=path, dry_run=dry_run)
        return cls(path=path, cells=dict(payload.get("cells", {})), dry_run=dry_run)

    def save(self) -> None:
        if self.dry_run:
            return
        eu.write_json(
            self.path,
            {"updated_at_utc": eu.utc_now_iso(), "cells": self.cells},
        )

    def status(self, key: str) -> str:
        return str(self.cells.get(key, {}).get("status", ""))

    def run_id(self, key: str) -> str:
        return str(self.cells.get(key, {}).get("run_id", ""))

    def record(self, key: str, status: str, **fields: Any) -> None:
        entry = dict(self.cells.get(key, {}))
        entry.update(fields)
        entry["status"] = status
        entry["updated_at_utc"] = eu.utc_now_iso()
        self.cells[key] = entry
        self.save()

    def done(self, key: str) -> bool:
        return self.status(key) == "complete"

    def failures(self) -> list[str]:
        return sorted(key for key in self.cells if self.status(key) == "failed")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_rerun_config(root: Path, path: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else root / path
    if not resolved.is_file():
        raise StageError(f"rerun config not found: {resolved}")
    return OmegaConf.to_container(OmegaConf.load(resolved), resolve=True)


def load_profile(root: Path, profile_id: str) -> dict[str, Any]:
    """One `conf/profile/<id>.yaml`, with its `${oc.env:...}` names resolved."""
    path = root / "conf/profile" / f"{profile_id}.yaml"
    if not path.is_file():
        raise StageError(f"profile config not found: {path}")
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def generated_run_config(
    root: Path, rerun: dict[str, Any], profiles: list[dict[str, Any]]
) -> dict[str, Any]:
    """The JSON-shaped run config the non-Hydra tools read.

    Derived from `conf/`, not from `run_configs/current_run.json`, so the
    driver has exactly one source of truth and leaves the hand-maintained
    working config alone.
    """
    sampling = OmegaConf.to_container(
        OmegaConf.load(root / "conf/sampling/default.yaml"), resolve=True
    )
    config = {
        "run_group_id": str(rerun["run_group_id"]),
        "datasets": list(rerun["datasets"]),
        "benchmark_variants": list(rerun["variants"]),
        "tasks": ["task1", "task2"],
        "prompt_version": str(
            OmegaConf.load(root / "conf/config.yaml").get("prompt_version", "v2-conf01")
        ),
        "seed": int(OmegaConf.load(root / "conf/config.yaml").get("seed", 20260518)),
        "batch_order": str(
            OmegaConf.load(root / "conf/config.yaml").get("batch_order", "grouped")
        ),
        "deterministic": sampling["deterministic"],
        "stochastic": sampling["stochastic"],
        "profiles": profiles,
    }
    return eu.normalize_run_config(config)


def cohort_models(
    root: Path, rerun: dict[str, Any]
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(profile, model) pairs of the official cohort and of the local cohort."""

    def pairs(profile_ids: list[str]) -> list[tuple[str, str]]:
        selected: list[tuple[str, str]] = []
        for profile_id in profile_ids:
            profile = load_profile(root, profile_id)
            models = [str(model) for model in profile.get("models", [])]
            if not models:
                raise StageError(
                    f"conf/profile/{profile_id}.yaml lists no models; the rerun "
                    "cohort comes from the profile."
                )
            selected.extend((profile_id, model) for model in models)
        return selected

    return pairs(list(rerun["cohort_profiles"])), pairs(
        list(rerun.get("local_profiles", []))
    )


# ---------------------------------------------------------------------------
# Command execution
# ---------------------------------------------------------------------------


@dataclass
class Runner:
    """Runs the project's CLIs, or prints them under --dry-run."""

    root: Path
    dry_run: bool = False
    #: Synthesize every answer locally and route every artifact into the smoke
    #: tree. This is how the whole chain is verified before any budget is spent.
    fake: bool = False
    smoke_items: int = 8

    @property
    def mode(self) -> str:
        return "smoke" if self.fake else "full"

    def run_overrides(self) -> list[str]:
        """Overrides every provider run of this invocation carries."""
        if not self.fake:
            return ["mode=full"]
        return [
            "mode=smoke",
            f"smoke_items={self.smoke_items}",
            "fake_completion=true",
            # The verification must not depend on the optional MLX package;
            # the TF-IDF reference backend exercises the same code path.
            "embedding=tfidf_proxy",
        ]

    @property
    def python(self) -> str:
        return str(self.root / ".venv/bin/python")

    def run(self, argv: list[str], *, label: str) -> int:
        command = [self.python, *argv]
        printable = " ".join(command)
        if self.dry_run:
            print(f"[dry-run] {printable}")
            return 0
        print(f"[{label}] {printable}", flush=True)
        return subprocess.run(command, cwd=self.root, check=False).returncode


def hydra_run(overrides: list[str]) -> list[str]:
    return ["scripts/run.py", *overrides]


def latest_run_id(
    root: Path,
    *,
    profile_id: str,
    model: str,
    dataset_id: str,
    variant: str,
    tasks: str,
    since_utc: str,
    registry_path: Path,
) -> str:
    """The run this invocation just wrote for this cell, whatever its status.

    The runner always leaves a terminal registry row behind, so this is how the
    driver learns the id to resume or to hand to the analysis. It is pinned to
    the launch time and the task set, because several runs of one cell coexist
    in a registry: the cohort's Task 1+2 run and the batching arms' Task 2 runs
    all share (profile, model, dataset, variant), and "newest" alone would
    hand back whichever ran last.
    """
    if not registry_path.exists():
        return ""
    candidates = [
        row
        for row in eu.read_csv_rows(registry_path)
        if str(row.get("profile_id", "")) == profile_id
        and str(row.get("model", "")) == model
        and str(row.get("dataset_id", "")) == dataset_id
        and str(row.get("benchmark_variant", "")) == variant
        and str(row.get("tasks", "")) == tasks
        and str(row.get("started_at_utc", "")) >= since_utc
    ]
    if not candidates:
        return ""
    newest = max(candidates, key=lambda row: str(row.get("started_at_utc", "")))
    return str(newest.get("run_id", ""))


def run_cell_with_retry(
    runner: Runner,
    state: RerunState,
    *,
    key: str,
    overrides: list[str],
    registry_path: Path,
    profile_id: str,
    model: str,
    dataset_id: str,
    variant: str,
    tasks: str,
) -> bool:
    """Run one cell, retrying once as a resume; record the outcome either way."""
    if state.done(key):
        print(f"[skip] {key} already complete ({state.run_id(key)})")
        return True
    since_utc = eu.utc_now_iso()
    state.record(key, "running")
    code = runner.run(hydra_run(overrides), label=key)
    run_id = latest_run_id(
        runner.root,
        profile_id=profile_id,
        model=model,
        dataset_id=dataset_id,
        variant=variant,
        tasks=tasks,
        since_utc=since_utc,
        registry_path=registry_path,
    )
    if code == 0:
        state.record(key, "complete", run_id=run_id, attempts=1)
        return True

    # One retry, as a resume of the run that failed: a fresh `mode=full` would
    # re-request every item the failed attempt already paid for.
    if run_id:
        print(f"[retry] {key}: resuming {run_id}", flush=True)
        resume_overrides = [
            override
            for override in overrides
            if not override.startswith(("mode=", "run_id="))
        ] + ["mode=resume", f"run_id={run_id}"]
        code = runner.run(hydra_run(resume_overrides), label=f"{key} (resume)")
        if code == 0:
            state.record(key, "complete", run_id=run_id, attempts=2)
            return True
    state.record(key, "failed", run_id=run_id, attempts=2 if run_id else 1)
    print(f"[failed] {key} (exit {code}); continuing", file=sys.stderr, flush=True)
    return False


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def embedding_backend_problems(root: Path, *, fake: bool = False) -> list[str]:
    """The ACSE backend the analysis stage will use, checked before any run.

    `conf/config.yaml` defaults to the MLX backend, whose package is optional
    and not a project dependency. Without it the runs succeed and the analysis
    stage fails hours later, so it is checked up front.
    """
    if fake:
        return []
    backend = str(
        OmegaConf.load(root / "conf/config.yaml").get("acse_embedding_backend", "")
    ).strip()
    if backend != "mlx":
        return []
    try:
        import mlx_embeddings  # noqa: F401
    except ImportError:
        return [
            (
                "conf/config.yaml sets acse_embedding_backend: mlx, but the "
                "optional `mlx-embeddings` package is not installed in .venv. "
                "Install it, or compose with `embedding=tfidf_proxy` and say so "
                "in the paper."
            )
        ]
    return []


def stage_preflight(
    root: Path,
    rerun: dict[str, Any],
    profiles: list[dict[str, Any]],
    cohort: list[tuple[str, str]],
    local: list[tuple[str, str]],
    *,
    fake: bool = False,
) -> None:
    """Every problem that would stop the rerun, reported at once."""
    problems: list[str] = []
    for profile in profiles:
        key_env = str(profile.get("api_key_env", ""))
        if not fake and key_env and not os.getenv(key_env):
            problems.append(
                f"{profile['profile_id']}: ${key_env} is not set in this shell"
            )
        if profile.get("requires_manual_server") and len(profile.get("models", [])) > 1:
            problems.append(
                f"{profile['profile_id']}: serves one model at a time but "
                f"{len(profile['models'])} are configured; run them in separate "
                "invocations, restarting the server between them"
            )
    for dataset_id in rerun["datasets"]:
        for variant in rerun["variants"]:
            path = eu.artifact_path(
                root / "data/processed/benchmark_items.csv", dataset_id, variant
            )
            if not path.exists():
                problems.append(f"benchmark missing: {path}")
    problems.extend(embedding_backend_problems(root, fake=fake))
    if not cohort:
        problems.append("no cohort models: conf/rerun/default.yaml lists no profiles")
    if problems:
        raise StageError("preflight failed:\n  - " + "\n  - ".join(problems))
    print(
        f"[preflight] {len(cohort)} cohort model(s), {len(local)} local model(s), "
        f"{len(rerun['datasets']) * len(rerun['variants'])} cells, "
        f"run group {rerun['run_group_id']}"
    )


def stage_cohort(
    runner: Runner,
    state: RerunState,
    rerun: dict[str, Any],
    models: list[tuple[str, str]],
) -> None:
    """Task 1 + Task 2 for every (profile, model, dataset, variant)."""
    for profile_id, model in models:
        for dataset_id in rerun["datasets"]:
            for variant in rerun["variants"]:
                run_cell_with_retry(
                    runner,
                    state,
                    key=f"cohort:{profile_id}:{model}:{dataset_id}:{variant}",
                    overrides=[
                        f"profile={profile_id}",
                        f"model={model}",
                        f"dataset={dataset_id}",
                        f"variant={variant}",
                        "task=both",
                        *runner.run_overrides(),
                        f"run_group_id={rerun['run_group_id']}",
                    ],
                    registry_path=eu.run_registry_path(
                        runner.root, dataset_id, variant, smoke=runner.fake
                    ),
                    profile_id=profile_id,
                    model=model,
                    dataset_id=dataset_id,
                    variant=variant,
                    tasks="task1,task2",
                )


def stage_task3(
    runner: Runner,
    state: RerunState,
    rerun: dict[str, Any],
    run_config: dict[str, Any],
) -> None:
    """Blind audits, each pinned to the Task 2 run it reads."""
    sources, gaps = ts.resolve_task3_sources(run_config, runner.root, mode=runner.mode)
    for gap in gaps:
        print(f"[task3] {gap.as_line()}", file=sys.stderr)
    audit_mode = str(rerun.get("task3", {}).get("audit_mode", "blind"))
    for source in sources:
        run_cell_with_retry(
            runner,
            state,
            key=(
                f"task3:{source.profile_id}:{source.model}:"
                f"{source.dataset_id}:{source.variant}"
            ),
            overrides=[
                f"profile={source.profile_id}",
                # Task 3 does not honour "all models": it would audit every
                # model of the profile against this one source run.
                f"model={source.model}",
                f"dataset={source.dataset_id}",
                f"variant={source.variant}",
                "task=task3",
                *runner.run_overrides(),
                f"source_run_id={source.source_run_id}",
                f"audit_mode={audit_mode}",
                # A fake cohort run answers a truncated benchmark on purpose,
                # so its audit has to accept a partial source. A real run never
                # does: an incomplete Task 2 run is a run to finish, not audit.
                *(["allow_partial_source=true"] if runner.fake else []),
                f"run_group_id={rerun['run_group_id']}",
            ],
            registry_path=eu.task3_registry_path(
                runner.root, source.dataset_id, source.variant, smoke=runner.fake
            ),
            profile_id=source.profile_id,
            model=source.model,
            dataset_id=source.dataset_id,
            variant=source.variant,
            tasks="task3",
        )


def stage_ablations(
    runner: Runner,
    state: RerunState,
    rerun: dict[str, Any],
    run_config_path: Path,
) -> None:
    """Batching, document context, and the weak-phrasing probe."""
    ablations = rerun.get("ablations", {})

    batching = ablations.get("batching", {})
    dataset_id = str(batching.get("dataset", "mlm_tapt"))
    variant = str(batching.get("variant", "must"))
    for entry in batching.get("models", []):
        profile_id, model = str(entry["profile"]), str(entry["model"])
        # Composed explicitly rather than through +experiment=batching_ablation:
        # the preset carries a sweeper, and one job per arm is unambiguous.
        for arm, batch_size, batch_order in (
            ("grouped", 16, "grouped"),
            ("shuffled", 16, "shuffled"),
            ("single", 1, "grouped"),
        ):
            run_cell_with_retry(
                runner,
                state,
                key=f"batching:{arm}:{profile_id}:{model}:{dataset_id}:{variant}",
                overrides=[
                    f"profile={profile_id}",
                    f"model={model}",
                    f"dataset={dataset_id}",
                    f"variant={variant}",
                    "task=task2",
                    *runner.run_overrides(),
                    "sampling=deterministic_only",
                    f"profile.batch_size={batch_size}",
                    f"profile.batch_order={batch_order}",
                    f"run_group_id={rerun['run_group_id']}",
                ],
                registry_path=eu.run_registry_path(
                    runner.root, dataset_id, variant, smoke=runner.fake
                ),
                profile_id=profile_id,
                model=model,
                dataset_id=dataset_id,
                variant=variant,
                tasks="task2",
            )

    context = ablations.get("context", {})
    context_group = str(context.get("run_group_id", "context-ablation-2026-09"))
    for entry in context.get("models", []):
        profile_id, model = str(entry["profile"]), str(entry["model"])
        for arm in ("bare", "document"):
            run_cell_with_retry(
                runner,
                state,
                key=f"context:{arm}:{profile_id}:{model}",
                overrides=[
                    f"profile={profile_id}",
                    f"model={model}",
                    "dataset=pure",
                    "variant=must",
                    "task=task2",
                    *runner.run_overrides(),
                    "sampling=deterministic_only",
                    "profile.batch_size=16",
                    "profile.batch_order=grouped",
                    f"item_context={arm}",
                    f"run_group_id={context_group}",
                ],
                registry_path=eu.run_registry_path(
                    runner.root, "pure", "must", smoke=runner.fake
                ),
                profile_id=profile_id,
                model=model,
                dataset_id="pure",
                variant="must",
                tasks="task2",
            )

    probe = ablations.get("weak_phrasing", {})
    probe_dataset = str(probe.get("dataset", "nice"))
    probe_variant = str(probe.get("variant", "must"))
    for entry in probe.get("models", []):
        profile_id, model = str(entry["profile"]), str(entry["model"])
        key = f"weak_phrasing:{profile_id}:{model}"
        if state.done(key):
            print(f"[skip] {key} already complete")
            continue
        code = runner.run(
            [
                "scripts/run_weak_modality_probe.py",
                "--config",
                str(run_config_path),
                "--profile",
                profile_id,
                "--model",
                model,
                "--dataset",
                probe_dataset,
                "--variant",
                probe_variant,
                "--mode",
                runner.mode,
                *(["--fake-completion"] if runner.fake else []),
                *(["--smoke-items", str(runner.smoke_items)] if runner.fake else []),
            ],
            label=key,
        )
        state.record(key, "complete" if code == 0 else "failed")


def stage_analysis(
    runner: Runner,
    state: RerunState,
    rerun: dict[str, Any],
    cohort: list[tuple[str, str]],
    local: list[tuple[str, str]],
) -> None:
    """Every table, macro and figure, in dependency order."""
    incomplete = [
        key for key in state.cells if key.startswith("cohort:") and not state.done(key)
    ]
    missing = [
        f"cohort:{profile_id}:{model}:{dataset_id}:{variant}"
        for profile_id, model in cohort
        for dataset_id in rerun["datasets"]
        for variant in rerun["variants"]
        if not state.done(f"cohort:{profile_id}:{model}:{dataset_id}:{variant}")
    ]
    if incomplete or missing:
        raise StageError(
            "the cohort is incomplete, so the paper tables would describe a "
            "partial rerun:\n  - " + "\n  - ".join(sorted(set(incomplete + missing)))
        )

    analysis = rerun.get("analysis", {})
    bootstrap = str(analysis.get("bootstrap_samples", 1000))
    models = [model for _, model in cohort]
    local_models = [model for _, model in local]
    # A fake run scores smoke runs, so every paper-facing artifact goes to the
    # smoke tree and the selectors are told to look there. Nothing under
    # outputs/ is touched.
    outputs_dir = "outputs/smoke" if runner.fake else "outputs"
    smoke_flags = ["--include-smoke"] if runner.fake else []

    steps: list[tuple[str, list[str]]] = []
    analysis_dirs: list[str] = []
    for dataset_id in rerun["datasets"]:
        for variant in rerun["variants"]:
            for profile_id, model in cohort + local:
                key = f"cohort:{profile_id}:{model}:{dataset_id}:{variant}"
                task3_key = f"task3:{profile_id}:{model}:{dataset_id}:{variant}"
                run_id = state.run_id(key)
                if not run_id:
                    continue
                analysis_dir = (
                    f"{outputs_dir}/evaluation_{dataset_id}_{variant}_"
                    f"{eu.safe_identifier(run_id)}"
                )
                analysis_dirs.append(analysis_dir)
                argv = [
                    "scripts/generate_evaluation_analysis.py",
                    "--run-id",
                    run_id,
                    "--dataset",
                    dataset_id,
                    "--variant",
                    variant,
                    "--model",
                    model,
                    "--profile",
                    profile_id,
                    # Named rather than derived, so a fake run's per-cell
                    # analysis lands in the smoke tree like everything else.
                    "--output-dir",
                    analysis_dir,
                ]
                if state.run_id(task3_key):
                    argv += ["--task3-run-id", state.run_id(task3_key)]
                if runner.fake:
                    # A fake cell answers a truncated benchmark, which is
                    # exactly what --allow-partial is for. A real cell must
                    # cover all of it, so the gate stays on.
                    argv.append("--allow-partial")
                steps.append((f"analysis:{key}", argv))

    # Named directories, not "everything under outputs/": without them the
    # cache pass walks the archived evaluation dirs too and fails on runs this
    # rerun never touched.
    acse_argv = ["scripts/compute_acse_semantic_artifacts.py"]
    if runner.fake:
        acse_argv += ["--output-root", outputs_dir]
    for analysis_dir in analysis_dirs:
        acse_argv += ["--analysis-dir", analysis_dir]
    steps.append(("analysis:acse", acse_argv))
    # The embedding probe, Figure 2 and the macro file all read MLX embeddings,
    # which a fake run does not produce (it stamps the TF-IDF backend so the
    # verification needs no optional package). They are skipped with a note
    # rather than failed: preflight already refuses a real run without MLX.
    mlx_steps = [
        (
            "analysis:embedding-probe",
            [
                "scripts/diagnose_embedding_separability.py",
                "--output-dir",
                f"{outputs_dir}/embedding_diagnostic",
            ],
        ),
        (
            "analysis:figure2",
            [
                "scripts/plot_embedding_diagnostic_figure_v2.py",
                "--diagnostic-dir",
                f"{outputs_dir}/embedding_diagnostic",
            ],
        ),
    ]
    if not runner.fake:
        steps.extend(mlx_steps)
    export_argv = [
        "scripts/export_paper_tables.py",
        # The cells this rerun actually covers: the exporter otherwise expects
        # all four and fails on the ones the config left out.
        *[
            argument
            for dataset_id in rerun["datasets"]
            for variant in rerun["variants"]
            for argument in ("--cell", f"{dataset_id}/{variant}")
        ],
        "--run-group-id",
        str(rerun["run_group_id"]),
        "--bootstrap-samples",
        bootstrap,
        "--overwrite-snapshots",
        "--output-dir",
        outputs_dir,
        *smoke_flags,
        "--models",
        *models,
        *local_models,
    ]
    for model in local_models:
        export_argv += ["--local-model", model]
    steps.append(("analysis:paper-tables", export_argv))
    # Plain, without --regenerate-snapshots: that path re-runs the tables
    # exporter with its default run group and would undo the line above.
    headline_step = (
        "analysis:headline-metrics",
        [
            "scripts/aggregate_paper_headline_metrics.py",
            "--task2",
            f"{outputs_dir}/paper_task2_text_drift_metrics.csv",
            "--confidence",
            f"{outputs_dir}/paper_text_drift_confidence_and_stability.csv",
            "--output",
            f"{outputs_dir}/paper_headline_metrics.csv",
        ],
    )
    # Skipped in a fake run for the same reason as the macro file: a truncated
    # cell leaves metrics with no value at all, which the aggregator reads as a
    # malformed ratio. A real cell always has all of them.
    if not runner.fake:
        steps.append(headline_step)
    if not runner.fake:
        steps.append(
            (
                "analysis:numbers",
                [
                    "scripts/export_paper_numbers.py",
                    "--output",
                    str(analysis.get("numbers_output", "outputs/paper_numbers.tex")),
                ],
            )
        )
    steps.append(
        (
            "analysis:context-ablation",
            [
                "scripts/compare_context_ablation.py",
                *smoke_flags,
                "--output-prefix",
                f"{outputs_dir}/context_ablation_summary",
            ],
        )
    )
    batching = rerun.get("ablations", {}).get("batching", {})
    steps.append(
        (
            "analysis:batching-ablation",
            [
                "scripts/compare_batching_ablation.py",
                # The comparison defaults to the paper's cell; the arms were run
                # on whichever cell the rerun config names.
                "--dataset",
                str(batching.get("dataset", "mlm_tapt")),
                "--variant",
                str(batching.get("variant", "must")),
                "--run-group-id",
                str(rerun["run_group_id"]),
                *smoke_flags,
                "--output-prefix",
                f"{outputs_dir}/batching_ablation_summary",
            ],
        )
    )

    if runner.fake:
        print(
            "[analysis] skipped (need a complete cell and the MLX embeddings a "
            "fake run does not produce): "
            + ", ".join(
                key
                for key, _ in [
                    *mlx_steps,
                    headline_step,
                    ("analysis:numbers", []),
                ]
            )
        )
    for key, argv in steps:
        if state.done(key):
            print(f"[skip] {key} already complete")
            continue
        code = runner.run(argv, label=key)
        state.record(key, "complete" if code == 0 else "failed")
        if code != 0:
            raise StageError(
                f"{key} failed (exit {code}); the later analysis steps read its "
                "output, so the stage stops here."
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_RERUN_CONFIG,
        help=f"Rerun config (default: {DEFAULT_RERUN_CONFIG}).",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=STAGES,
        default=[],
        help="Run only these stages. Repeatable; default is all of them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print every command that would run, without running any of them.",
    )
    parser.add_argument(
        "--fake-completion",
        action="store_true",
        help=(
            "Verify the whole chain without a provider: synthesize every answer "
            "locally, run in smoke mode, and route every artifact into the smoke "
            "tree. Costs nothing and touches no paper-facing file."
        ),
    )
    parser.add_argument(
        "--smoke-items",
        type=int,
        default=8,
        help="Benchmark items per cell under --fake-completion (default: 8).",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=None,
        help=f"State file (default: outputs/{STATE_NAME}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    except StageError as error:
        # A stage that cannot run is an operator problem, not a crash: print
        # what is wrong, not a traceback through the driver.
        print(f"error: {error}", file=sys.stderr)
        return 2


def _main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = eu.project_root()
    stages = args.only or list(STAGES)
    rerun = load_rerun_config(root, args.config)
    cohort, local = cohort_models(root, rerun)
    profiles = [
        load_profile(root, profile_id)
        for profile_id in list(rerun["cohort_profiles"])
        + list(rerun.get("local_profiles", []))
    ]
    run_config = generated_run_config(root, rerun, profiles)
    run_config_path = root / GENERATED_RUN_CONFIG
    if not args.dry_run:
        eu.write_json(run_config_path, run_config)

    if args.fake_completion:
        # The runners route by run id, but this process and the analysis
        # scripts also need to look in the same tree.
        os.environ[eu.SMOKE_TREE_ENV_VAR] = "1"
    state_name = "rerun_state_smoke.json" if args.fake_completion else STATE_NAME
    state = RerunState.load(
        args.state or root / "outputs" / state_name, dry_run=args.dry_run
    )
    runner = Runner(
        root=root,
        dry_run=args.dry_run,
        fake=args.fake_completion,
        smoke_items=args.smoke_items,
    )

    if "preflight" in stages:
        stage_preflight(root, rerun, profiles, cohort, local, fake=args.fake_completion)
    if "cohort" in stages:
        stage_cohort(runner, state, rerun, cohort + local)
    if "task3" in stages:
        stage_task3(runner, state, rerun, run_config)
    if "ablations" in stages:
        stage_ablations(runner, state, rerun, run_config_path)
    if "analysis" in stages:
        stage_analysis(runner, state, rerun, cohort, local)

    failures = state.failures()
    if failures:
        print(
            "\nFailed, re-run this command to retry:\n  - " + "\n  - ".join(failures),
            file=sys.stderr,
        )
        return 1
    print("\nAll requested stages complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
