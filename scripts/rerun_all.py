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

**Resumable.** `outputs/rerun/<run_group_id>/state.json` records status, configuration and run IDs.
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
import copy
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

try:
    import build_pure_benchmark as pure_benchmark
    import eval_utils as eu
except ModuleNotFoundError:  # pragma: no cover - invocation-path fallback
    from scripts import build_pure_benchmark as pure_benchmark, eval_utils as eu


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
    configuration: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path, *, dry_run: bool = False) -> RerunState:
        payload = cls._read_payload(path)
        return cls(
            path=path,
            cells=dict(payload.get("cells", {})),
            dry_run=dry_run,
            configuration=dict(payload.get("configuration", {})),
        )

    @staticmethod
    def _read_payload(path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as error:
            raise StageError(f"Cannot read rerun state {path}: {error}") from error

    @staticmethod
    def experiment_identity(configuration: dict[str, Any]) -> dict[str, Any]:
        """The recorded configuration without deployment details.

        An endpoint address (``base_url``) says where a profile was served
        from, not what was asked of it; the run registries keep the address
        each run actually used. Leaving it out lets a moved server, or the
        published placeholder, resume the same experiment.
        """
        identity = copy.deepcopy(configuration)
        for profile in identity.get("run_config", {}).get("profiles", []):
            profile.pop("base_url", None)
        return identity

    def bind_configuration(self, configuration: dict[str, Any]) -> None:
        if self.cells and self.experiment_identity(
            self.configuration
        ) != self.experiment_identity(configuration):
            raise StageError(
                f"{self.path} belongs to a different or unrecorded configuration. "
                "Use a new run_group_id for a new experiment (or --state with a new path). "
                "Existing runs have been preserved."
            )
        self.configuration = configuration

    def save(self) -> None:
        if self.dry_run:
            return
        payload = {
            "updated_at_utc": eu.utc_now_iso(),
            "configuration": self.configuration,
            "cells": self.cells,
        }
        eu.atomic_write_text(
            self.path,
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
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
        if self.dry_run:
            self.cells[key] = entry
            return
        # Two drivers may share this file, one per endpoint (`--profile`), so a
        # record merges its one key into what is on disk under a lock. Writing
        # this process's whole snapshot would overwrite the other driver's
        # entries with the stale copies this process loaded at startup.
        with eu.file_lock(self.path):
            cells = dict(self._read_payload(self.path).get("cells", {}))
            cells[key] = entry
            self.cells = cells
            self.save()

    def done(self, key: str) -> bool:
        return self.status(key) == "complete"

    def failures(self, profile_ids: Iterable[str] = ()) -> list[str]:
        """Failed cells, on disk now; narrowed to `profile_ids` when given."""
        cells = (
            self.cells
            if self.dry_run
            else {**self.cells, **self._read_payload(self.path).get("cells", {})}
        )
        selected = set(profile_ids)
        return sorted(
            key
            for key, entry in cells.items()
            if str(entry.get("status", "")) == "failed"
            and (not selected or selected & set(key.split(":")))
        )


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
    base = OmegaConf.load(root / "conf/config.yaml")
    embedding = OmegaConf.to_container(
        OmegaConf.load(
            root / "conf/embedding" / f"{rerun.get('embedding', 'qwen3_06b')}.yaml"
        ),
        resolve=True,
    )
    logging = OmegaConf.to_container(
        OmegaConf.load(root / "conf/logging/default.yaml"), resolve=True
    )
    logging.update(
        write_progress_csv=True, write_event_jsonl=True, write_request_transcripts=True
    )
    # The ablations read this config too, and their cells may lie outside the
    # cohort's datasets (the weak probe runs on `nice`, the context arms on
    # `pure`), so the selector lists must cover them as well.
    ablations = rerun.get("ablations", {})
    datasets = list(rerun["datasets"])
    variants = list(rerun["variants"])
    for name, ablation in ablations.items():
        dataset_id = "pure" if name == "context" else ablation.get("dataset")
        variant = "must" if name == "context" else ablation.get("variant")
        if dataset_id and dataset_id not in datasets:
            datasets.append(str(dataset_id))
        if variant and variant not in variants:
            variants.append(str(variant))
    config = {
        "run_group_id": str(rerun["run_group_id"]),
        "datasets": datasets,
        "benchmark_variants": variants,
        "tasks": ["task1", "task2"],
        "prompt_version": str(base.get("prompt_version", "v2-conf01")),
        "seed": int(base.get("seed", 20260518)),
        "batch_order": str(base.get("batch_order", "grouped")),
        **embedding,
        "logging": logging,
        "deterministic": sampling["deterministic"],
        "stochastic": sampling["stochastic"],
        "profiles": profiles,
    }
    return eu.normalize_run_config(config)


def cohort_models(
    root: Path, rerun: dict[str, Any], *, profiles: list[dict[str, Any]] | None = None
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(profile, model) pairs of the official cohort and of the local cohort."""

    def pairs(profile_ids: list[str]) -> list[tuple[str, str]]:
        selected: list[tuple[str, str]] = []
        for profile_id in profile_ids:
            profile = (
                next((p for p in profiles if p["profile_id"] == profile_id), None)
                if profiles is not None
                else load_profile(root, profile_id)
            )
            if profile is None:
                raise StageError(
                    f"Recorded configuration has no profile {profile_id!r}"
                )
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


def resolve_ablation_models(
    rerun: dict[str, Any], profiles: list[dict[str, Any]]
) -> None:
    """Resolve representative models from the same lists used for the cohort."""
    configured = {str(p["profile_id"]): list(p["models"]) for p in profiles}
    for name, ablation in rerun.get("ablations", {}).items():
        if "models" not in ablation:
            ablation["models"] = []
            for profile in ablation.get("profiles", configured):
                if profile not in configured:
                    raise StageError(
                        f"{name}: profile {profile!r} is not a cohort or local "
                        f"profile of this rerun config (known: "
                        f"{', '.join(configured)})"
                    )
                if not configured[profile]:
                    raise StageError(
                        f"{name}: conf/profile/{profile}.yaml lists no models, "
                        "so it has no representative for this ablation"
                    )
                ablation["models"].append(
                    {"profile": profile, "model": configured[profile][0]}
                )
        for entry in ablation["models"]:
            if entry["model"] not in configured.get(entry["profile"], []):
                raise StageError(
                    f"{name}: ablation model is not in the configured cohort: {entry}"
                )


def select_profiles(requested: list[str], rerun: dict[str, Any]) -> list[str]:
    """The profiles this invocation drives: `--profile` values, or all of them.

    The hosted and the local endpoint are independent, so one driver per
    profile can run at the same time against the same state file; each only
    touches its own cells (the state merges per key).
    """
    known = [
        str(profile_id)
        for profile_id in [*rerun["cohort_profiles"], *rerun.get("local_profiles", [])]
    ]
    unknown = [profile_id for profile_id in requested if profile_id not in known]
    if unknown:
        raise StageError(
            f"--profile {', '.join(unknown)}: not a cohort or local profile of this "
            f"rerun config (known: {', '.join(known)})"
        )
    return [profile_id for profile_id in known if profile_id in requested] or known


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
    embedding: str = "qwen3_06b"
    environment: dict[str, str] = field(default_factory=dict)

    @property
    def mode(self) -> str:
        return "smoke" if self.fake else "full"

    def run_overrides(self) -> list[str]:
        """Overrides every provider run of this invocation carries."""
        common = [
            "logging.write_progress_csv=true",
            "logging.write_event_jsonl=true",
            "logging.write_request_transcripts=true",
        ]
        if not self.fake:
            return ["mode=full", f"embedding={self.embedding}", *common]
        return [
            "mode=smoke",
            f"smoke_items={self.smoke_items}",
            "fake_completion=true",
            # The verification must not depend on the optional MLX package;
            # the TF-IDF reference backend exercises the same code path.
            "embedding=tfidf_proxy",
            *common,
        ]

    @property
    def python(self) -> str:
        return str(self.root / ".venv/bin/python")

    def run(self, argv: list[str], *, label: str) -> int:
        command = [self.python, "-u", *argv]
        printable = shlex.join(command)
        if self.dry_run:
            print(f"[dry-run] {printable}")
            return 0
        print(f"[{label}] {printable}", flush=True)
        log_dir = self.root / (
            "outputs/smoke/rerun/logs" if self.fake else "outputs/rerun/logs"
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / f"{eu.safe_identifier(label)}.log").open(
            "a", encoding="utf-8"
        ) as log:
            log.write(f"\n[{eu.utc_now_iso()}] {printable}\n")
            log.flush()
            with subprocess.Popen(
                command,
                cwd=self.root,
                env={**os.environ, **self.environment},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            ) as process:
                assert process.stdout is not None
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    code = process.wait()
                except KeyboardInterrupt:
                    process.terminate()
                    process.wait()
                    log.write("Interrupted; rerun this command to resume.\n")
                    raise
                log.write(f"[{eu.utc_now_iso()}] exit={code}\n")
                return code


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
    arm: Mapping[str, str] | None = None,
) -> str:
    """The run this invocation just wrote for this cell, whatever its status.

    The runner always leaves a terminal registry row behind, so this is how the
    driver learns the id to resume or to hand to the analysis. It is pinned to
    the launch time and the task set, because several runs of one cell coexist
    in a registry: the cohort's Task 1+2 run and the batching arms' Task 2 runs
    all share (profile, model, dataset, variant), and "newest" alone would
    hand back whichever ran last.

    `arm` pins the registry columns that tell the ablation arms of one cell
    apart (`batch_size`, `batch_order`, `item_context`): the three batching
    arms and the two context arms share everything above, and a driver that
    resumed after an early failure would otherwise hand one arm's run to the
    next, which would resume it, find nothing pending, and report a null effect.
    """
    if not registry_path.exists():
        return ""
    expected = {key: str(value) for key, value in (arm or {}).items()}
    candidates = [
        row
        for row in eu.read_csv_rows(registry_path)
        if str(row.get("profile_id", "")) == profile_id
        and str(row.get("model", "")) == model
        and str(row.get("dataset_id", "")) == dataset_id
        and str(row.get("benchmark_variant", "")) == variant
        and str(row.get("tasks", "")) == tasks
        and str(row.get("started_at_utc", "")) >= since_utc
        and all(str(row.get(key, "")) == value for key, value in expected.items())
    ]
    if not candidates:
        return ""
    newest = max(candidates, key=lambda row: str(row.get("started_at_utc", "")))
    return str(newest.get("run_id", ""))


#: Hydra overrides that distinguish the ablation arms of one cell, and the
#: registry column each one lands in.
ARM_OVERRIDE_COLUMNS = {
    "profile.batch_size": "batch_size",
    "profile.batch_order": "batch_order",
    "item_context": "item_context",
}


def arm_columns(overrides: list[str]) -> dict[str, str]:
    """The registry values an arm's overrides pin, for `latest_run_id`."""
    columns: dict[str, str] = {}
    for override in overrides:
        name, _, value = override.partition("=")
        if name in ARM_OVERRIDE_COLUMNS:
            columns[ARM_OVERRIDE_COLUMNS[name]] = value
    return columns


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
    previous = state.cells.get(key, {})
    since_utc = str(previous.get("started_at_utc") or eu.utc_now_iso())

    def discover_run() -> str:
        return latest_run_id(
            runner.root,
            profile_id=profile_id,
            model=model,
            dataset_id=dataset_id,
            variant=variant,
            tasks=tasks,
            since_utc=since_utc,
            registry_path=registry_path,
            arm=arm_columns(overrides),
        )

    run_id = state.run_id(key) or (discover_run() if previous else "")
    if run_id:
        overrides = [
            value for value in overrides if not value.startswith(("mode=", "run_id="))
        ]
        overrides += ["mode=resume", f"run_id={run_id}"]
    state.record(key, "running", started_at_utc=since_utc, run_id=run_id)
    code = runner.run(hydra_run(overrides), label=key)
    run_id = run_id or discover_run()
    if runner.dry_run:
        run_id = run_id or f"planned-{eu.safe_identifier(key)}"
    if code == 0 and run_id:
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
    # Native Metal initialization can abort or leave a partially imported module
    # after failure. Probe in a child so preflight remains a readable error.
    result = subprocess.run(
        [sys.executable, "-c", "import mlx_embeddings"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip().splitlines()
        return [
            "The configured MLX embedding backend is unavailable: "
            + (detail[-1] if detail else f"import exited {result.returncode}")
            + ". Run uv sync --group dev --locked, and run from a session with Metal GPU access."
        ]

    return []


def primary_protocol_problems(
    rerun: dict[str, Any], profiles: list[dict[str, Any]]
) -> list[str]:
    return [
        f"{profile['profile_id']}: primary protocol requires {field}={expected!r}, got {profile.get(field)!r}"
        for profile in profiles
        for field, expected in rerun.get("primary_protocol", {}).items()
        if profile.get(field) != expected
    ]


def primary_sampling_problems(
    rerun: dict[str, Any], run_config: dict[str, Any]
) -> list[str]:
    return [
        f"primary sampling requires {kind}.samples={expected}, got {run_config.get(kind, {}).get('samples')!r}"
        for kind, expected in rerun.get("primary_sampling", {}).items()
        if run_config.get(kind, {}).get("samples") != expected
    ]


def stage_preflight(
    root: Path,
    rerun: dict[str, Any],
    profiles: list[dict[str, Any]],
    cohort: list[tuple[str, str]],
    local: list[tuple[str, str]],
    *,
    fake: bool = False,
    dry_run: bool = False,
    analysis: bool = True,
) -> None:
    """Every problem that would stop the rerun, reported at once.

    The MLX check only applies when this invocation will reach the analysis
    stage: generation runs on any host that can reach the endpoints, and the
    analysis is run separately on a Mac with Metal (docs/experiment_runbook.md).
    """
    problems = primary_protocol_problems(rerun, profiles)
    for profile in profiles:
        key_env = str(profile.get("api_key_env", ""))
        if not fake and key_env and not os.getenv(key_env):
            message = f"{profile['profile_id']}: ${key_env} is not set in this shell"
            # A plan may be printed without keys; the real run may not start.
            if dry_run:
                print(f"[preflight] warning: {message}", file=sys.stderr)
            else:
                problems.append(message)
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
    context = rerun.get("ablations", {}).get("context", {})
    if context.get("models"):
        path = eu.artifact_path(
            root / "data/processed/benchmark_items.csv", "pure", "must"
        )
        if not path.exists():
            problems.append(f"context benchmark missing: {path}")
        elif not fake:
            try:
                pure_benchmark.validate_existing_benchmark(root)
            except (ValueError, FileNotFoundError) as error:
                message = str(error)
                if dry_run:
                    print(f"[preflight] warning: {message}", file=sys.stderr)
                else:
                    problems.append(message)
    problems.extend(
        embedding_backend_problems(root, fake=fake or dry_run or not analysis)
    )
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
    audit_mode = str(rerun.get("task3", {}).get("audit_mode", "blind"))
    for profile in run_config["profiles"]:
        profile_id = str(profile["profile_id"])
        for model in profile["models"]:
            for dataset_id in rerun["datasets"]:
                for variant in rerun["variants"]:
                    suffix = f"{profile_id}:{model}:{dataset_id}:{variant}"
                    source_key = f"cohort:{suffix}"
                    if not state.done(source_key) or not state.run_id(source_key):
                        print(f"[task3] waiting for {source_key}", file=sys.stderr)
                        continue
                    run_cell_with_retry(
                        runner,
                        state,
                        key=f"task3:{suffix}",
                        overrides=[
                            f"profile={profile_id}",
                            f"model={model}",
                            f"dataset={dataset_id}",
                            f"variant={variant}",
                            "task=task3",
                            *runner.run_overrides(),
                            f"source_run_id={state.run_id(source_key)}",
                            f"audit_mode={audit_mode}",
                            *(["allow_partial_source=true"] if runner.fake else []),
                            f"run_group_id={rerun['run_group_id']}",
                        ],
                        registry_path=eu.task3_registry_path(
                            runner.root, dataset_id, variant, smoke=runner.fake
                        ),
                        profile_id=profile_id,
                        model=model,
                        dataset_id=dataset_id,
                        variant=variant,
                        tasks="task3",
                    )


def batching_arm_specs(config: dict[str, Any]) -> list[tuple[str, int, str]]:
    """Explicit size/composition sweep; old states retain their three arms."""
    sizes = config.get("batch_sizes", [16, 1])
    orders = config.get("batch_orders", ["grouped", "shuffled"])
    if not sizes or not orders:
        raise StageError("batching ablation requires batch sizes and orders")
    arms = {}
    for size in sizes:
        for order in orders:
            size = eu.positive_int(size, "batch_size")
            order = eu.normalize_batch_order(order)
            arm = eu.batching_arm_name(size, order)
            arms[arm] = (arm, size, "grouped" if size == 1 else order)
    baseline = config.get("baseline_arm", "grouped")
    if baseline not in arms:
        raise StageError(f"batching baseline {baseline!r} is absent from its arms")
    return list(arms.values())


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
        for arm, batch_size, batch_order in batching_arm_specs(batching):
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
                    f"profile.batch_size={context.get('batch_size', 16)}",
                    f"profile.batch_order={context.get('batch_order', 'grouped')}",
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
        run_id = state.run_id(key) or eu.new_run_id(
            ("smoke-" if runner.fake else "") + f"weak-probe-{probe_variant}"
        )
        try:
            from scripts.run_weak_modality_probe import probe_registry_path
        except ModuleNotFoundError:
            from run_weak_modality_probe import probe_registry_path
        registry = probe_registry_path(
            runner.root, probe_dataset, probe_variant, run_id=run_id
        )
        resume = registry.exists() and any(
            row.get("run_id") == run_id for row in eu.read_csv_rows(registry)
        )
        state.record(key, "running", run_id=run_id)
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
                "resume" if resume else runner.mode,
                "--run-id",
                run_id,
                "--bootstrap-samples",
                str(rerun.get("analysis", {}).get("bootstrap_samples", 1000)),
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
    *,
    refresh: bool = False,
) -> None:
    """Every table, macro and figure, in dependency order."""
    incomplete = [
        key for key in state.cells if key.startswith("cohort:") and not state.done(key)
    ]
    missing = [
        f"cohort:{profile_id}:{model}:{dataset_id}:{variant}"
        for profile_id, model in cohort + local
        for dataset_id in rerun["datasets"]
        for variant in rerun["variants"]
        if not state.done(f"cohort:{profile_id}:{model}:{dataset_id}:{variant}")
        or not state.run_id(f"cohort:{profile_id}:{model}:{dataset_id}:{variant}")
    ]
    missing += [
        f"task3:{profile_id}:{model}:{dataset_id}:{variant}"
        for profile_id, model in cohort + local
        for dataset_id in rerun["datasets"]
        for variant in rerun["variants"]
        if not state.done(f"task3:{profile_id}:{model}:{dataset_id}:{variant}")
        or not state.run_id(f"task3:{profile_id}:{model}:{dataset_id}:{variant}")
    ]
    incomplete += [
        key
        for key in state.cells
        if key.startswith(("batching:", "context:", "weak_phrasing:"))
        and not state.done(key)
    ]
    for name, ablation in rerun.get("ablations", {}).items():
        for entry in ablation.get("models", []):
            profile_id, model = entry["profile"], entry["model"]
            if name == "batching":
                keys = [
                    f"batching:{arm}:{profile_id}:{model}:{ablation.get('dataset', 'mlm_tapt')}:{ablation.get('variant', 'must')}"
                    for arm, _, _ in batching_arm_specs(ablation)
                ]
            elif name == "context":
                keys = [
                    f"context:{arm}:{profile_id}:{model}"
                    for arm in ("bare", "document")
                ]
            else:
                keys = [f"weak_phrasing:{profile_id}:{model}"]
            missing += [key for key in keys if not state.done(key)]
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
    selected_manifest = f"{outputs_dir}/rerun/acse_selected_manifest.csv"
    acse_argv = [
        "scripts/compute_acse_semantic_artifacts.py",
        "--selected-manifest",
        selected_manifest,
    ]
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
                "--bootstrap-samples",
                bootstrap,
                "--manifest",
                selected_manifest,
                "--models",
                "hgb",
                "--output-dir",
                f"{outputs_dir}/embedding_diagnostic",
            ],
        ),
        (
            "analysis:figure2",
            [
                "scripts/plot_embedding_diagnostic_figure_v2.py",
                "--output",
                f"{outputs_dir}/rerun/figures/embedding_diagnostic.pdf",
                "--diagnostic-dir",
                f"{outputs_dir}/embedding_diagnostic",
            ],
        ),
    ]
    mlx_steps.append(
        (
            "analysis:figure3",
            [
                "scripts/plot_embedding_diagnostic_tsne_supp.py",
                "--manifest",
                selected_manifest,
                "--diagnostic-dir",
                f"{outputs_dir}/embedding_diagnostic",
                "--output",
                f"{outputs_dir}/rerun/figures/embedding_diagnostic_tsne_supp.pdf",
            ],
        )
    )
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
    protocols = {
        (int(profile.get("batch_size", 16)), str(profile.get("batch_order", "grouped")))
        for profile in state.configuration.get("run_config", {}).get("profiles", [])
    }
    if len(protocols) > 1:
        raise StageError("Main paper export requires one matched request protocol")
    primary = rerun.get("primary_protocol", {})
    batch_size, batch_order = next(
        iter(protocols),
        (primary.get("batch_size", 16), primary.get("batch_order", "grouped")),
    )
    export_argv += [
        "--expected-batch-size",
        str(batch_size),
        "--expected-batch-order",
        batch_order,
    ]
    for key, entry in state.cells.items():
        if key.startswith("cohort:"):
            export_argv += ["--run-id", str(entry["run_id"])]
        elif key.startswith("task3:"):
            export_argv += ["--task3-run-id", str(entry["run_id"])]
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
                    "--outputs-dir",
                    outputs_dir,
                    "--strict",
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
                "--run-group-id",
                str(
                    rerun.get("ablations", {})
                    .get("context", {})
                    .get("run_group_id", "context-ablation-2026-09")
                ),
                "--bootstrap-samples",
                bootstrap,
                *[
                    arg
                    for key in state.cells
                    if key.startswith("context:")
                    for arg in ("--run-id", state.run_id(key))
                ],
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
                "--baseline-arm",
                str(batching.get("baseline_arm", "grouped")),
                "--bootstrap-samples",
                bootstrap,
                *[
                    arg
                    for key in state.cells
                    if key.startswith("batching:")
                    for arg in ("--run-id", state.run_id(key))
                ],
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

    # The manuscript's remaining artifacts, which read the tables above: the
    # commitment-transition counts and TikZ figure, the ablation-delta figure
    # (batching, context and weak phrasing side by side), and the
    # meaning-variation sensitivity grid. The grid reads the MLX caches, so it
    # is skipped with the other MLX steps in a fake run.
    ablations = rerun.get("ablations", {})
    paper_steps = [
        (
            "analysis:commitment-transitions",
            [
                "scripts/export_commitment_transitions.py",
                "--provenance",
                f"{outputs_dir}/paper_snapshot_provenance.json",
                "--output-dir",
                outputs_dir,
                "--tikz-output",
                f"{outputs_dir}/rerun/figures/commitment_transitions.tex",
            ],
        ),
        (
            "analysis:ablation-figure",
            [
                "scripts/plot_ablation_deltas.py",
                "--batching",
                f"{outputs_dir}/batching_ablation_summary_deltas.csv",
                "--context",
                f"{outputs_dir}/context_ablation_summary_deltas.csv",
                "--state",
                str(state.path),
                "--output",
                f"{outputs_dir}/rerun/figures/ablation_deltas.pdf",
            ],
        ),
        (
            "analysis:meaning-variation",
            [
                "scripts/meaning_variation_sensitivity.py",
                "--provenance",
                f"{outputs_dir}/paper_snapshot_provenance.json",
                "--manifest",
                selected_manifest,
                "--rq-table",
                f"{outputs_dir}/paper_per_model_rq_table.csv",
                "--output-dir",
                f"{outputs_dir}/meaning_variation_sensitivity",
            ],
        ),
    ]
    if not runner.fake:
        steps.extend(
            (key, argv)
            for key, argv in paper_steps
            # The figure needs all three ablations; the other two only the tables.
            if key != "analysis:ablation-figure"
            or all(
                ablations.get(name, {}).get("models")
                for name in ("batching", "context", "weak_phrasing")
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
                    *paper_steps,
                ]
            )
        )
    steps = [
        (key, argv)
        for key, argv in steps
        if all(
            key != f"analysis:{name}-ablation"
            or rerun.get("ablations", {}).get(name, {}).get("models")
            for name in ("context", "batching")
        )
    ]
    if refresh:
        # Invalidate dependants before the first write. If refresh fails, an
        # ordinary resume must not skip stale downstream artifacts as complete.
        for key, _ in steps:
            state.record(key, "pending")
    skipped = 0
    for key, argv in steps:
        if state.done(key) and not refresh:
            print(f"[skip] {key} already complete")
            skipped += 1
            continue
        code = runner.run(argv, label=key)
        state.record(key, "complete" if code == 0 else "failed")
        if code != 0:
            raise StageError(
                f"{key} failed (exit {code}); the later analysis steps read its "
                "output, so the stage stops here."
            )
    if steps and skipped == len(steps):
        # A tracked state marks every step complete, so a plain `--only analysis`
        # on a fresh clone recomputes nothing; say how to recompute instead.
        print(
            f"\nEvery analysis step is recorded as complete in {state.path}. To "
            "recompute them from the raw outputs, add "
            f"--refresh-analysis --state {state.path}."
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
        "--refresh-analysis",
        action="store_true",
        help="Recompute derived analysis in dependency order; requires --only analysis. Generation is never repeated.",
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
        help="State file (default: outputs/rerun/<run_group_id>/state.json).",
    )
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        help=(
            "Drive only this profile's models (repeatable). The hosted and the "
            "local endpoint are independent, so one driver per profile can run "
            "concurrently against the same state file. Not combined with the "
            "analysis stage, which covers every profile."
        ),
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
    if args.refresh_analysis and stages != ["analysis"]:
        raise StageError(
            "--refresh-analysis requires --only analysis; generation is not allowed"
        )
    if args.refresh_analysis:
        if args.state is None:
            raise StageError(
                "--refresh-analysis requires an explicit --state with recorded configuration"
            )
        recorded = RerunState.load(args.state, dry_run=args.dry_run).configuration
        if not recorded.get("rerun") or not recorded.get("run_config", {}).get(
            "profiles"
        ):
            raise StageError(
                "Analysis refresh needs a state with recorded run configuration"
            )
        if bool(recorded.get("fake_completion")) != args.fake_completion:
            raise StageError(
                "Analysis refresh must match the state's fake-completion mode"
            )
        rerun = copy.deepcopy(recorded["rerun"])
        run_config = copy.deepcopy(recorded["run_config"])
        profiles = run_config["profiles"]
        cohort, local = cohort_models(root, rerun, profiles=profiles)
        args.smoke_items = recorded.get("smoke_items", args.smoke_items)
        print(
            "[analysis] using the explicitly selected state's recorded cohort and settings; live profiles are not consulted"
        )
    else:
        rerun = load_rerun_config(root, args.config)
        cohort, local = cohort_models(root, rerun)
        profiles = [
            load_profile(root, profile_id)
            for profile_id in list(rerun["cohort_profiles"])
            + list(rerun.get("local_profiles", []))
        ]
        resolve_ablation_models(rerun, profiles)
        run_config = generated_run_config(root, rerun, profiles)
    protocol_problems = primary_protocol_problems(rerun, profiles)
    protocol_problems += primary_sampling_problems(rerun, run_config)
    if protocol_problems:
        raise StageError("\n".join(protocol_problems))
    run_config_path = root / GENERATED_RUN_CONFIG
    selected = select_profiles(list(args.profile), rerun)
    if args.profile and "analysis" in stages:
        raise StageError(
            "--profile drives one endpoint's generation; the analysis stage covers "
            "every profile, so run it without --profile once all drivers are done."
        )
    if args.fake_completion:
        run_config_path = root / "outputs/smoke/rerun/run_config.json"
        run_config["acse_embedding_backend"] = eu.ACSE_PROXY_EMBEDDING_BACKEND
        run_config["acse_embedding_mlx_model"] = ""

    if args.fake_completion:
        # The runners route by run id, but this process and the analysis
        # scripts also need to look in the same tree.
        os.environ[eu.SMOKE_TREE_ENV_VAR] = "1"
    state_root = root / (
        "outputs/smoke/rerun" if args.fake_completion else "outputs/rerun"
    )
    state = RerunState.load(
        args.state or state_root / str(rerun["run_group_id"]) / "state.json",
        dry_run=args.dry_run,
    )
    state.bind_configuration(
        {
            "rerun": rerun,
            "run_config": run_config,
            "fake_completion": args.fake_completion,
            "smoke_items": args.smoke_items,
        }
    )
    if not args.dry_run:
        eu.write_json(run_config_path, run_config)
    runner = Runner(
        root=root,
        dry_run=args.dry_run,
        fake=args.fake_completion,
        smoke_items=args.smoke_items,
        embedding=str(rerun.get("embedding", "qwen3_06b")),
        environment={
            eu.ACSE_EMBEDDING_BACKEND_ENV: str(
                run_config.get("acse_embedding_backend", "")
            ),
            eu.ACSE_MLX_MODEL_ENV: str(run_config.get("acse_embedding_mlx_model", "")),
            eu.SMOKE_TREE_ENV_VAR: "1" if args.fake_completion else "0",
        },
    )

    if "preflight" in stages:
        stage_preflight(
            root,
            rerun,
            [profile for profile in profiles if profile["profile_id"] in selected],
            cohort,
            local,
            fake=args.fake_completion,
            dry_run=args.dry_run,
            analysis="analysis" in stages,
        )
    # Finish all requests for one model before letting the router load the next.
    for profile_id, model in cohort + local:
        if profile_id not in selected:
            continue
        model_config = {
            **run_config,
            "profiles": [
                {**profile, "models": [model]}
                for profile in profiles
                if str(profile["profile_id"]) == profile_id
            ],
        }
        model_rerun = {
            **rerun,
            "ablations": {
                name: {
                    **ablation,
                    "models": [
                        entry
                        for entry in ablation.get("models", [])
                        if (entry["profile"], entry["model"]) == (profile_id, model)
                    ],
                }
                for name, ablation in rerun.get("ablations", {}).items()
            },
        }
        if "cohort" in stages:
            stage_cohort(runner, state, rerun, [(profile_id, model)])
        if "task3" in stages:
            stage_task3(runner, state, rerun, model_config)
        if "ablations" in stages:
            stage_ablations(runner, state, model_rerun, run_config_path)
    if "analysis" in stages:
        stage_analysis(
            runner, state, rerun, cohort, local, refresh=args.refresh_analysis
        )

    failures = state.failures(selected)
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
