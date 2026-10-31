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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import eval_utils as eu
    import run_provenance as rp
    import run_transcripts as rt
    import runner_lifecycle as rl
    from runner_args import Task3RunnerArgs
except ModuleNotFoundError:  # pragma: no cover
    from scripts import (
        eval_utils as eu,
        run_provenance as rp,
        run_transcripts as rt,
        runner_lifecycle as rl,
    )
    from scripts.runner_args import Task3RunnerArgs


def source_run_prefix(variant: str) -> str:
    """Return the canonical (paper-facing) Task 2 source-run prefix for a variant.

    Args:
        variant: Benchmark variant; `must` gives `full`, any other variant
            `full-<variant>`.

    Returns:
        The run id prefix a full Task 2 run of this variant carries.
    """
    return "full" if variant == "must" else f"full-{variant}"


def smoke_source_run_prefix(variant: str) -> str:
    """Return the run id prefix of a smoke Task 2 run for a benchmark variant.

    Args:
        variant: Benchmark variant; `must` gives `smoke`, any other variant
            `smoke-<variant>`.

    Returns:
        The run id prefix a smoke Task 2 run of this variant carries.
    """
    return "smoke" if variant == "must" else f"smoke-{variant}"


def source_run_prefixes(variant: str, mode: str = "full") -> list[str]:
    """Return the accepted Task 2 source-run prefixes.

    Official Task 3 runs always audit a `full-*` Task 2 run. `--mode smoke` also
    accepts a `smoke-*` source so the no-credentials fake-completion chain can
    run end to end without paper-facing data.

    Args:
        variant: Benchmark variant of the cell.
        mode: Task 3 run mode; only `smoke` adds the smoke prefix.

    Returns:
        The full-run prefix, followed by the smoke-run prefix in smoke mode.
    """
    prefixes = [source_run_prefix(variant)]
    if mode == "smoke":
        prefixes.append(smoke_source_run_prefix(variant))
    return prefixes


def task3_run_prefix(
    mode: str, variant: str, audit_mode: str = eu.OFFICIAL_TASK3_AUDIT_MODE
) -> str:
    """Return the prefix a new Task 3 run id gets.

    Args:
        mode: Run mode; `smoke` appends `-smoke`.
        variant: Benchmark variant; any variant other than `must` is appended
            as `-<variant>`.
        audit_mode: Task 3 audit mode; any mode other than the official blind
            audit is inserted after `task3` with hyphens for underscores.

    Returns:
        The run id prefix, for example `task3`, `task3-declared-text-<variant>`
        or `task3-smoke`.
    """
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
    """Return the prompt template file for a Task 3 audit mode.

    Args:
        root: Project root that contains `prompts/`.
        audit_mode: Task 3 audit mode; the official blind audit uses the blind
            prompt, both declared-modality modes share the declared prompt.

    Returns:
        The path of `modality_verification.txt` or
        `modality_verification_declared.txt` under `root/prompts`.
    """
    audit_mode = eu.normalize_task3_audit_mode(audit_mode)
    if audit_mode == eu.OFFICIAL_TASK3_AUDIT_MODE:
        return root / "prompts/modality_verification.txt"
    return root / "prompts/modality_verification_declared.txt"


def task3_declared_modality_for_prompt(item: Mapping[str, Any], audit_mode: str) -> str:
    """Return the declared modality a declared-modality audit shows the model.

    Args:
        item: Task 3 audit item.
        audit_mode: Task 3 audit mode; `declared_text` shows the modality of
            the Task 2 requirement text, `declared_source` the source modality.

    Returns:
        The item's `task2_text_modality` or `source_modality`, or "" for the
        blind audit.
    """
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
    """Render the Task 3 prompt for one audit item.

    Args:
        template: Prompt template from `task3_prompt_path`.
        item: Task 3 audit item supplying the source statement and the Task 2
            requirement.
        audit_mode: Task 3 audit mode; any mode other than the blind audit
            also fills in the declared extracted modality.

    Returns:
        The rendered prompt text.
    """
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
    """Answer a prompt offline, so smoke runs never contact a provider.

    Every audited item is reported as `preserves` at confidence 0.9; a prompt
    asking for a `decision` (the provider preflight) gets a `yes`.

    Args:
        **kwargs: The keyword arguments a real completion function receives;
            `prompt` selects the reply, and `model`, `temperature`, `top_p`,
            `max_tokens` and `seed` go into the synthesized request payload.

    Returns:
        A successful completion dict in the shape `eval_utils.chat_completion`
        returns, including the request payload and its fingerprint.
    """
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
    # Reported like a real request so a fake run exercises the transcript.
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
        "raw_text": raw_text,
        "response_json": {"fake": True, "model": kwargs.get("model", "")},
        "latency_s": 0.0,
        "error": "",
        "request_payload": eu.transcript_request_payload(request_kwargs),
        "request_payload_sha": eu.request_payload_sha(request_kwargs),
        "request_seed": kwargs.get("seed"),
        "attempt_errors": [],
    }


class Task3SourceProfileMismatchError(ValueError):
    """The requested profile has no Task 2 rows in the selected source run."""


def _profiles_present(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    """Return the distinct `profile_id` values on `rows`, with the legacy blank named.

    Args:
        rows: Task 2 raw rows.

    Returns:
        The sorted profile ids, with a blank id reported as
        `eval_utils.LEGACY_IDENTITY`.
    """
    return tuple(
        sorted(
            {str(row.get("profile_id", "") or "") or eu.LEGACY_IDENTITY for row in rows}
        )
    )


def source_rows_for_model(
    source_rows: list[dict[str, Any]],
    model: str,
    profile_id: str,
    *,
    allow_profile_mismatch: bool = False,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    """Return the Task 2 rows of `model` under `profile_id`, and their profiles.

    Rows written before `profile_id` was recorded carry no profile at all. They
    are accepted as this profile's own output -- a raw file is scoped to one
    dataset/variant and predates the provider matrix -- and reported as
    `eval_utils.LEGACY_IDENTITY`.

    Rows belonging to a DIFFERENT profile are never silently audited. Task 3
    stamps the REQUESTED profile on every row it writes, so falling back to
    "any row for this model" (as this function used to) attributes one
    provider's extraction to another and leaves no trace of it in the artifact.
    Without `allow_profile_mismatch` that raises
    `Task3SourceProfileMismatchError`; with it the mismatch is logged as a
    warning and the profiles actually audited are returned so the caller can
    record them in the run's audit provenance.

    Args:
        source_rows: Rows of the selected Task 2 source run.
        model: Model whose Task 2 output is audited.
        profile_id: Provider profile the Task 3 cell runs under.
        allow_profile_mismatch: Whether to audit the model's rows of another
            profile (with a warning) when none exist under `profile_id`.

    Returns:
        The selected rows and the profile ids they were produced under. Returns
        `([], ())` when the source run holds no rows for `model` at all, which
        is a different failure the caller reports separately.

    Raises:
        Task3SourceProfileMismatchError: If the source run holds rows for
            `model` only under other profiles and `allow_profile_mismatch` is
            false.
    """
    model_rows = [row for row in source_rows if str(row.get("model", "")) == str(model)]
    if not model_rows:
        return [], ()
    profile_rows = [
        row for row in model_rows if str(row.get("profile_id", "")) in {"", profile_id}
    ]
    if profile_rows:
        return profile_rows, _profiles_present(profile_rows)

    other_profiles = _profiles_present(model_rows)
    listed = ", ".join(repr(value) for value in other_profiles)
    if not allow_profile_mismatch:
        raise Task3SourceProfileMismatchError(
            f"The Task 2 source run has rows for model {model!r}, but none under "
            f"profile {profile_id!r}: they belong to profile(s) {listed}. Task 3 "
            "audits the Task 2 output of the profile it is run under. Re-run with "
            f"--profile {other_profiles[0]}, or pass "
            "--allow-source-profile-mismatch to audit them anyway (the audited "
            "source profile is then recorded in the run registry notes)."
        )
    eu.logger.warning(
        "%s",
        {
            "warning": "task3_source_profile_mismatch",
            "requested_profile_id": profile_id,
            "audited_profile_ids": list(other_profiles),
            "model": model,
            "detail": (
                f"auditing Task 2 rows of profile(s) {listed} under profile "
                f"{profile_id!r} because --allow-source-profile-mismatch was passed"
            ),
        },
    )
    return model_rows, other_profiles


def require_complete_task2_source(
    benchmark: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    source_run_id: str,
    prefix: str | list[str],
    expected_stochastic_samples: int,
) -> None:
    """Check that the Task 2 source run is complete for the selected model.

    Args:
        benchmark: Benchmark items of the cell's dataset and variant.
        source_rows: The model's rows of the source run.
        source_run_id: Run id of the Task 2 source run.
        prefix: Run id prefix or prefixes a source run may carry.
        expected_stochastic_samples: Stochastic samples the run config plans
            per item, used to judge completeness.

    Raises:
        ValueError: If `source_run_id` is not a complete Task 2 run with an
            accepted prefix.
    """
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


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI for this runner: the shared options plus the Task 3 ones.

    Returns:
        The argument parser, with `runner_lifecycle.common_runner_parser` as
        its parent.
    """
    parser = argparse.ArgumentParser(
        description="Run Task 3 blind text audit from a provider-aware run config.",
        parents=[rl.common_runner_parser()],
    )
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument(
        "--audit-mode",
        choices=eu.TASK3_AUDIT_MODES,
        default=eu.OFFICIAL_TASK3_AUDIT_MODE,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-partial-source", action="store_true")
    parser.add_argument(
        "--allow-source-profile-mismatch",
        action="store_true",
        help=(
            "Audit Task 2 rows produced under a DIFFERENT provider profile. Off "
            "by default: the audited source profile is recorded in the registry "
            "notes and the mismatch is logged as a warning."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, load the JSON run config, and run the audit.

    Args:
        argv: Command-line arguments; None reads `sys.argv`.
    """
    args = build_parser().parse_args(argv)
    eu.configure_run_logging(args.log_level)
    run_from_config(eu.load_run_config(args.config), args)


@dataclass(frozen=True, slots=True)
class Task3Run:
    """State shared by every cell of one Task 3 audit run.

    Attributes:
        run_config: Normalized run config the audit is built from.
        args: Runner options from the JSON CLI or the Hydra bridge.
        root: Project root all artifact paths are resolved under.
        audit_mode: Normalized audit mode (`blind`, `declared_text` or
            `declared_source`).
        logging_config: Run logging settings with the command-line overrides
            applied.
        completion_fn: `eval_utils.chat_completion`, or `fake_completion`
            under `--fake-completion`.
        task3_template: Prompt template for the audit mode.
        smoke_tree: Whether every cell writes into the parallel smoke tree
            (fake completions, smoke mode, or a smoke run id).
        dry_run: Whether cells are only planned and logged, without writing
            audit items or contacting a provider.
    """

    run_config: dict[str, Any]
    args: Task3RunnerArgs
    root: Path
    audit_mode: str
    logging_config: dict[str, Any]
    completion_fn: Callable[..., dict[str, Any]]
    task3_template: str
    smoke_tree: bool
    dry_run: bool

    @property
    def run_group_id(self) -> str:
        """Return the run group id shared by every cell of the audit."""
        return str(self.run_config["run_group_id"])

    @property
    def expected_stochastic_samples(self) -> int:
        """Return the stochastic samples the run config plans per item."""
        return int(self.run_config["stochastic"]["samples"])


@dataclass(frozen=True, slots=True)
class Task3Cell:
    """One (profile, model, dataset, variant) Task 3 cell, fully planned.

    Attributes:
        profile: Provider profile the cell runs under.
        model: Model alias of the cell; the same model's Task 2 output is
            audited.
        dataset_id: Dataset of the cell.
        variant: Benchmark variant of the cell.
        run_id: Run id the cell's Task 3 rows and registry row carry.
        source_run_id: Task 2 run whose output is audited.
        source_profile_ids: Profile(s) the audited Task 2 rows were produced
            under. Equal to `(profile_id,)` for a normal cell; a different
            value means `--allow-source-profile-mismatch` was used and is
            recorded in the registry notes so the artifact says whose output
            was audited.
        started_at: Start time for the registry row; a resumed cell keeps the
            one its first attempt recorded.
        seed: Request seed for the cell's jobs.
        send_seed: Whether the seed is sent to the provider.
        max_retries: Transport retries per provider request.
        batch_order: How jobs are grouped into batches (`grouped` or
            `shuffled`).
        preflight: Server model probe result, or `{"dry_run": True}` in a dry
            run.
        items: Audit items in scope; only the first `--smoke-items` in smoke
            mode.
        jobs: Every planned completion job of the cell.
        existing_rows: Raw rows already in the cell's Task 3 output file.
        pending_jobs: Planned jobs this run has no reusable raw row for yet.
        planned_api_calls: Provider requests the full plan needs.
        pending_api_calls: Provider requests the pending jobs still need.
        output_path: Task 3 raw JSONL file the records are appended to.
        registry_path: Task 3 run registry CSV of the dataset and variant.
        progress_path: Task 3 live progress CSV of the dataset and variant.
        events_path: Task 3 run events JSONL of the dataset and variant.
        items_path: CSV holding every audit item built for the cell.
        resume: Identity of the run being continued; set only under
            `--mode resume`.
    """

    profile: dict[str, Any]
    model: str
    dataset_id: str
    variant: str
    run_id: str
    source_run_id: str
    #: Profile(s) the audited Task 2 rows were produced under. Equal to
    #: `(profile_id,)` for a normal cell; a different value means
    #: `--allow-source-profile-mismatch` was used and is recorded in the
    #: registry notes so the artifact says whose output was audited.
    source_profile_ids: tuple[str, ...]
    started_at: str
    seed: int
    send_seed: bool
    max_retries: int
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
    #: Set only under `--mode resume`; carries the first attempt's identity so
    #: this cell reports the original start time and provenance notes.
    resume: rl.ResumeState | None = None

    @property
    def profile_id(self) -> str:
        """Return the id of the provider profile the cell runs under."""
        return str(self.profile["profile_id"])

    @property
    def provider_id(self) -> str:
        """Return the id of the provider behind the cell's profile."""
        return str(self.profile["provider_id"])

    @property
    def batch_size(self) -> int:
        """Return how many planned jobs the profile sends per provider request."""
        return int(self.profile["batch_size"])

    def identity(self) -> dict[str, Any]:
        """Return the fields that identify this cell in logs and warning events.

        Returns:
            The run id, dataset, benchmark variant, provider, profile and model.
        """
        return {
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "benchmark_variant": self.variant,
            "provider_id": self.provider_id,
            "profile_id": self.profile_id,
            "model": self.model,
        }


def cell_context(matrix: Task3Run, cell: Task3Cell) -> dict[str, Any]:
    """Return the identity fields stamped onto every run event and warning event.

    Args:
        matrix: Audit run the cell belongs to; supplies the run group id.
        cell: Planned cell; supplies the rest of the identity.

    Returns:
        `Task3Cell.identity` plus `run_group_id`.
    """
    return {**cell.identity(), "run_group_id": matrix.run_group_id}


def resolve_source_rows(
    matrix: Task3Run,
    profile: Mapping[str, Any],
    model: str,
    dataset_id: str,
    variant: str,
) -> tuple[str, list[dict[str, Any]], tuple[str, ...]]:
    """Locate the complete Task 2 run this cell audits.

    Raises if no source run matches, if it holds no rows for `model`, if its
    rows for `model` belong to another profile (see `source_rows_for_model`),
    or if it is incomplete and `--allow-partial-source` was not passed.

    Args:
        matrix: Audit run the cell belongs to; supplies `--source-run-id`,
            the mode and the partial-source and profile-mismatch switches.
        profile: Provider profile the cell runs under.
        model: Model whose Task 2 output is audited.
        dataset_id: Dataset of the cell.
        variant: Benchmark variant of the cell.

    Returns:
        The source run id, its rows for this cell, and the profile(s) those
        rows were produced under.

    Raises:
        ValueError: If no source run with an accepted prefix matches, or the
            source run holds no rows for `model`.
    """
    args = matrix.args
    # A smoke-* source run lives in the parallel smoke tree.
    source_raw_path = eu.model_outputs_raw_path(
        matrix.root, dataset_id, variant, run_id=args.source_run_id
    )
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
    model_source_rows, source_profile_ids = source_rows_for_model(
        source_rows,
        model,
        str(profile["profile_id"]),
        allow_profile_mismatch=bool(args.allow_source_profile_mismatch),
    )
    if not model_source_rows:
        raise ValueError(
            f"Source run {source_run_id!r} has no rows for model {model!r}; "
            "Task 3 should audit the same model's Task 2 outputs."
        )
    if not args.allow_partial_source:
        require_complete_task2_source(
            eu.read_csv_rows(
                eu.artifact_path(
                    matrix.root / "data/processed/benchmark_items.csv",
                    dataset_id,
                    variant,
                )
            ),
            model_source_rows,
            str(source_run_id),
            accepted_prefixes,
            expected_stochastic_samples=matrix.expected_stochastic_samples,
        )
    return str(source_run_id), model_source_rows, source_profile_ids


def plan_cell(
    matrix: Task3Run,
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
) -> Task3Cell:
    """Resolve one Task 3 cell: its source run, audit items, and job plan.

    Outside a dry run this writes every built audit item to the cell's items
    CSV. A resumed cell is checked here, before any provider call, against the
    source provenance its first attempt recorded.

    Args:
        matrix: Audit run the cell belongs to.
        profile: Provider profile the cell runs under.
        model: Model alias of the cell.
        dataset_id: Dataset of the cell.
        variant: Benchmark variant of the cell.
        run_id: Run id the cell's records carry.
        preflight: Server model probe result recorded with the jobs.
        seed: Request seed for the cell's jobs.
        send_seed: Whether the seed is sent to the provider.
        max_retries: Transport retries per provider request.
        batch_order: Normalized batch order (`grouped` or `shuffled`).
        resume: Identity of the run being continued, or None for a new run.

    Returns:
        The fully planned cell, with pending jobs and API-call counts computed
        against the Task 3 rows already on disk.

    Raises:
        ValueError: If no audit items could be built from valid deterministic
            Task 2 rows.
    """
    args = matrix.args
    audit_mode = matrix.audit_mode
    source_run_id, model_source_rows, source_profile_ids = resolve_source_rows(
        matrix, profile, model, dataset_id, variant
    )
    benchmark = eu.read_csv_rows(
        eu.artifact_path(
            matrix.root / "data/processed/benchmark_items.csv", dataset_id, variant
        )
    )
    all_items = eu.build_task3_verification_items(
        benchmark, model_source_rows, audit_mode=audit_mode
    )
    if not all_items:
        raise ValueError(
            "No Task 3 text-audit items were built from valid deterministic Task 2 rows."
        )
    items_path = eu.task3_verification_items_path(
        matrix.root,
        dataset_id,
        variant,
        source_run_id,
        model,
        audit_mode,
        smoke=matrix.smoke_tree,
    )
    if not matrix.dry_run:
        eu.write_csv_rows(
            items_path, all_items, fieldnames=eu.TASK3_VERIFICATION_FIELDS
        )
    items = all_items[: max(1, args.smoke_items)] if args.mode == "smoke" else all_items
    jobs = eu.planned_completion_jobs_for_items(
        items,
        prompt_fn=lambda item: task3_prompt_for(
            matrix.task3_template, item, audit_mode=audit_mode
        ),
        prompt_version=f"{matrix.run_config['prompt_version']}:task3:{audit_mode}",
        model=model,
        host=profile["base_url"],
        run_id=run_id,
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
    output_path = eu.task3_raw_path(
        matrix.root, dataset_id, variant, smoke=matrix.smoke_tree
    )
    existing_rows = eu.read_jsonl(output_path)
    pending_jobs = eu.pending_completion_jobs(jobs, existing_rows, run_id)
    batch_size = int(profile["batch_size"])
    cell = Task3Cell(
        profile=profile,
        model=model,
        dataset_id=dataset_id,
        variant=variant,
        run_id=run_id,
        source_run_id=source_run_id,
        source_profile_ids=source_profile_ids,
        # A resumed cell keeps the start time its first attempt recorded.
        started_at=resume.started_at_utc if resume else eu.utc_now_iso(),
        seed=seed,
        send_seed=send_seed,
        max_retries=max_retries,
        batch_order=batch_order,
        preflight=preflight,
        items=items,
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
        registry_path=eu.task3_registry_path(
            matrix.root, dataset_id, variant, smoke=matrix.smoke_tree
        ),
        progress_path=eu.task3_progress_path(
            matrix.root, dataset_id, variant, smoke=matrix.smoke_tree
        ),
        events_path=eu.task3_events_path(
            matrix.root, dataset_id, variant, smoke=matrix.smoke_tree
        ),
        items_path=items_path,
        resume=resume,
    )
    if resume is not None:
        # Before any provider call: a resume that would audit a different source
        # than the run it continues must not reach execute_cell.
        rl.resumed_notes(resume.notes, source_note(matrix, cell))
    return cell


def source_note(matrix: Task3Run, cell: Task3Cell) -> str:
    """Return the provenance for the audited Task 2 output as registry-note fragments.

    The source profile is spelled out only when it is not this cell's own, so
    the note stays quiet on the normal path but a
    `--allow-source-profile-mismatch` run is self-describing.

    Args:
        matrix: Audit run the cell belongs to; supplies the audit mode.
        cell: Planned cell; supplies the source run and source profiles.

    Returns:
        `audit_mode=...; source_run_id=...`, plus `source_profile_id=...` when
        the audited rows came from another profile.
    """
    note = f"audit_mode={matrix.audit_mode}; source_run_id={cell.source_run_id}"
    if set(cell.source_profile_ids) - {cell.profile_id}:
        note += f"; source_profile_id={','.join(cell.source_profile_ids)}"
    return note


def registry_row(
    matrix: Task3Run,
    cell: Task3Cell,
    raw_rows: list[dict[str, Any]],
    *,
    status: str | None = None,
    finished_at_utc: str = "",
) -> dict[str, Any]:
    """Summarize `raw_rows` as this cell's Task 3 registry row.

    Args:
        matrix: Audit run the cell belongs to.
        cell: Planned cell the row describes.
        raw_rows: Raw rows of the cell's Task 3 output file seen so far.
        status: Registry status to write; None lets
            `eval_utils.run_registry_summary` derive it from the rows.
        finished_at_utc: Finish time, empty while the cell is still running.

    Returns:
        The registry row for `eval_utils.upsert_run_registry_row`, with the
        source provenance in its notes (or the first attempt's notes for a
        resumed cell).
    """
    profile = cell.profile
    return eu.run_registry_summary(
        cell.items,
        raw_rows,
        run_id=cell.run_id,
        run_group_id=matrix.run_group_id,
        provider_id=cell.provider_id,
        profile_id=cell.profile_id,
        model=cell.model,
        dataset_id=cell.dataset_id,
        variant=cell.variant,
        tasks=["task3"],
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
        # A resumed cell keeps the notes (and resolved-config digest) of the run
        # it continues instead of restamping them with `mode=resume` -- but only
        # if this attempt audits the same source; see rl.resumed_notes.
        notes=cell.resume.notes
        if cell.resume
        else rp.run_notes(matrix.args, source_note(matrix, cell)),
    )


def log_planned_cell(matrix: Task3Run, cell: Task3Cell) -> None:
    """Report a dry-run cell plan without contacting a provider.

    Args:
        matrix: Audit run the cell belongs to.
        cell: Planned cell whose item, job, batch and API-call counts are
            logged.
    """
    eu.logger.info(
        "%s",
        {
            "dry_run": True,
            "run_id": cell.run_id,
            "source_run_id": cell.source_run_id,
            "dataset_id": cell.dataset_id,
            "variant": cell.variant,
            "profile": cell.profile_id,
            "model": cell.model,
            "audit_mode": matrix.audit_mode,
            "task3_items": len(cell.items),
            "planned_jobs": len(cell.jobs),
            "pending_jobs": len(cell.pending_jobs),
            "planned_batches": cell.planned_api_calls,
            "pending_api_calls": cell.pending_api_calls,
            "batch_size": cell.profile["batch_size"],
            "batch_order": cell.batch_order,
            "output_path": str(cell.output_path),
            "task3_items_path": str(cell.items_path),
        },
    )


def cell_execution(matrix: Task3Run, cell: Task3Cell) -> rl.CellExecution:
    """Bind this cell's Task 3 specifics to the shared lifecycle.

    Args:
        matrix: Audit run the cell belongs to.
        cell: Planned cell to execute.

    Returns:
        The execution handed to `runner_lifecycle.execute_cell`, with the
        cell's lease and, unless turned off, its transcript writer.
    """
    return rl.CellExecution(
        cell=cell,
        tasks=("task3",),
        progress_items=cell.items,
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
            "source_run_id": cell.source_run_id,
            "dataset_id": cell.dataset_id,
            "variant": cell.variant,
            "profile": cell.profile_id,
            "model": cell.model,
            "audit_mode": matrix.audit_mode,
            "task3_items": len(cell.items),
            "planned_jobs": len(cell.jobs),
            "pending_jobs": len(cell.pending_jobs),
            "batch_size": cell.profile["batch_size"],
            "batch_order": cell.batch_order,
            "seed": cell.seed,
            "send_seed": cell.send_seed,
            "max_retries": cell.max_retries,
            "output_path": str(cell.output_path),
            "task3_items_path": str(cell.items_path),
            "registry_path": str(cell.registry_path),
        },
        # Task 3 events also name the audited Task 2 run and the audit mode.
        event_fields={
            "source_run_id": cell.source_run_id,
            "audit_mode": matrix.audit_mode,
            "task3_items_path": str(cell.items_path),
        },
        registry_label="Task 3 registry status",
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


def execute_cell(matrix: Task3Run, cell: Task3Cell) -> None:
    """Run one Task 3 cell to completion, streaming progress artifacts.

    Args:
        matrix: Audit run the cell belongs to.
        cell: Planned cell to execute.
    """
    rl.execute_cell(cell_execution(matrix, cell))


def run_from_config(run_config: dict[str, Any], args: Task3RunnerArgs) -> None:
    """Execute a normalized run config.

    Shared by the argparse CLI above and by the Hydra entry point in
    `scripts/run.py`, which composes the same dictionary from `conf/`.

    Args:
        run_config: Normalized run config: profiles, datasets, variants,
            prompt and sampling settings.
        args: Task 3 runner options; the profile, model, dataset and variant
            options narrow the matrix, `source_run_id` and `audit_mode` pick
            what is audited and how, and `mode`, `run_id`, `dry_run` and
            `fake_completion` control how each cell runs.

    Raises:
        ValueError: If `--mode resume` is given without `--run-id`.
    """
    eu.configure_run_logging(args.log_level)
    eu.apply_acse_embedding_env(run_config)
    if args.mode == "resume" and not args.run_id:
        raise ValueError("--mode resume requires --run-id.")

    root = eu.project_root()
    audit_mode = eu.normalize_task3_audit_mode(args.audit_mode)
    profiles = eu.filter_run_profiles(
        run_config, profile_id=args.profile, model=args.model
    )
    datasets = eu.selected_values(list(run_config["datasets"]), args.dataset, "dataset")
    variants = eu.selected_values(
        list(run_config["benchmark_variants"]), args.variant, "variant"
    )
    matrix = Task3Run(
        run_config=run_config,
        args=args,
        root=root,
        audit_mode=audit_mode,
        logging_config=eu.logging_config_from_args(run_config, args),
        completion_fn=fake_completion if args.fake_completion else eu.chat_completion,
        task3_template=eu.load_prompt(task3_prompt_path(root, audit_mode)),
        # Smoke/fake Task 3 runs are isolated from the paper-facing artifact tree.
        smoke_tree=bool(args.fake_completion)
        or args.mode == "smoke"
        or eu.is_smoke_run_id(args.run_id),
        dry_run=bool(getattr(args, "dry_run", False)),
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
                if matrix.dry_run
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
                        task3_run_prefix(args.mode, variant, audit_mode)
                    )
                    resume = (
                        # Fails here, before this run id opens a log file or
                        # writes audit items, if it is not a real run of this
                        # exact cell.
                        rl.resolve_resume(
                            root,
                            eu.task3_registry_path(
                                root, dataset_id, variant, smoke=matrix.smoke_tree
                            ),
                            run_id=run_id,
                            provider_id=profile["provider_id"],
                            profile_id=profile["profile_id"],
                            model=model,
                            dataset_id=dataset_id,
                            variant=variant,
                            tasks=["task3"],
                            record=not matrix.dry_run,
                        )
                        if args.mode == "resume"
                        else None
                    )
                    if not matrix.dry_run:
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
                    if matrix.dry_run:
                        log_planned_cell(matrix, cell)
                    else:
                        execute_cell(matrix, cell)


if __name__ == "__main__":
    main()
