"""The execute-cell lifecycle both provider-matrix runners share.

`scripts/run_experiment_from_config.py` (Task 1/2) and
`scripts/run_task3_verification_from_config.py` (Task 3) plan different work
from different item populations, but once a cell is planned they run it exactly
the same way: claim the cell, write the starting registry row, stream batches
through the provider, refresh live progress/warning artifacts, and reconcile a
terminal registry state. That lifecycle lives here once; planning stays in the
runners.

The module also owns the parts of the lifecycle that must hold *across*
processes, which the per-file storage locks in `eval_utils` deliberately do not
cover:

- `common_runner_parser()` is an `add_help=False` parent holding the CLI options
  both runners accept, so a new shared option cannot be added to one only.
- `resolve_resume()` refuses a `--mode resume` whose run id does not name an
  existing, identity-compatible registry row, *before* the runner writes
  anything. The original `started_at_utc` and provenance notes are carried
  forward instead of being rewritten, and the resume is recorded as one more
  entry in the run's `resumed_at` list.
- `CellLease` is an advisory per-cell claim: a second runner that finds a live
  lease (a process that still exists and a recent heartbeat) refuses the cell
  instead of duplicating paid provider calls, while a stale lease is taken over
  with a warning.
- `execute_cell()` reconciles the registry to `failed` / `interrupted` when the
  run raises, so an aborted run never leaves a row stuck at `running`.
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import socket
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

try:
    import eval_utils as eu
    import run_provenance as rp
    from runner_args import RunnerArgs
except ModuleNotFoundError:  # pragma: no cover
    from scripts import eval_utils as eu, run_provenance as rp
    from scripts.runner_args import RunnerArgs


# =============================================================================
# Shared CLI options
# =============================================================================


def common_runner_parser() -> argparse.ArgumentParser:
    """Parent parser holding every option both runner CLIs accept.

    Declared once so the two front doors cannot drift in type, default, or
    spelling. Task-specific options (`--task`, `--all-models`,
    `--source-run-id`, `--audit-mode`, `--allow-partial-source`) and `--dry-run`
    (whose help text differs between the runners) stay with their runner.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--profile")
    parser.add_argument("--model")
    parser.add_argument("--dataset")
    parser.add_argument("--variant")
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
        "--log-level",
        default="INFO",
        help="Logging level for the re_uq logger (default: INFO).",
    )
    return parser


# =============================================================================
# Resume protocol
# =============================================================================


class ResumeError(ValueError):
    """`--mode resume` did not name an existing, compatible run."""


@dataclass(frozen=True, slots=True)
class ResumeState:
    """The original identity of the run a `--mode resume` cell continues.

    Carried onto the planned cell so the registry row keeps the first run's
    `started_at_utc` and provenance `notes` (including any
    `resolved_config_sha`) instead of being stamped with the resume's own.
    """

    run_id: str
    started_at_utc: str
    notes: str
    resumed_at: tuple[str, ...] = ()


def _registry_identity(row: Mapping[str, Any]) -> dict[str, str]:
    return {
        "provider_id": str(row.get("provider_id", "")),
        "profile_id": str(row.get("profile_id", "")),
        "model": str(row.get("model", "")),
        "dataset_id": str(row.get("dataset_id", "")),
        "benchmark_variant": str(row.get("benchmark_variant", "")),
        "tasks": str(row.get("tasks", "")),
    }


def find_resume_row(
    registry_path: str | Path,
    *,
    run_id: str,
    provider_id: str,
    profile_id: str,
    model: str,
    dataset_id: str,
    variant: str,
    tasks: Iterable[str],
) -> dict[str, Any]:
    """Return the registry row `run_id` resumes, or raise `ResumeError`.

    Read-only: this runs before the runner opens a run log, writes a resolved
    config, or touches the registry, so a mistyped `--run-id` fails instead of
    silently creating a second run that merely looks resumed.
    """
    registry_path = Path(registry_path)
    if not registry_path.exists():
        raise ResumeError(
            f"--mode resume needs an existing run, but no run registry exists at "
            f"{registry_path}. Start the run with --mode full first."
        )
    rows = eu.read_csv_rows(registry_path)
    candidates = [row for row in rows if str(row.get("run_id", "")) == str(run_id)]
    if not candidates:
        known = sorted(
            {str(row.get("run_id", "")) for row in rows if row.get("run_id")}
        )
        raise ResumeError(
            f"--mode resume was given run id {run_id!r}, which is not in "
            f"{registry_path}. Known run ids: {', '.join(known) or '(none)'}."
        )
    wanted = {
        "provider_id": str(provider_id),
        "profile_id": str(profile_id),
        "model": str(model),
        "dataset_id": eu.normalize_dataset_id(dataset_id),
        "benchmark_variant": eu.normalize_benchmark_variant(variant),
        "tasks": ",".join(eu.normalize_task_filter(tasks)),
    }
    mismatches: list[str] = []
    for row in candidates:
        found = _registry_identity(row)
        row_mismatches = [
            f"{field_name}={found[field_name]!r} (resuming with {value!r})"
            for field_name, value in wanted.items()
            if found[field_name] != value
        ]
        if not row_mismatches:
            return dict(row)
        mismatches = row_mismatches
    raise ResumeError(
        f"--mode resume was given run id {run_id!r}, but its row in "
        f"{registry_path} was resolved differently: {'; '.join(mismatches)}. "
        "Resume the run with the same provider/profile/model/dataset/variant/task "
        "selection it was started with, or start a new run."
    )


def resolve_resume(
    root: str | Path,
    registry_path: str | Path,
    *,
    run_id: str,
    provider_id: str,
    profile_id: str,
    model: str,
    dataset_id: str,
    variant: str,
    tasks: Iterable[str],
    record: bool = True,
) -> ResumeState:
    """Validate a resume target and record this attempt in its `resumed_at` log.

    Validation happens first and raises `ResumeError` without writing anything;
    only a resume that is going to run appends to the log (`record=False` for a
    dry run).
    """
    row = find_resume_row(
        registry_path,
        run_id=run_id,
        provider_id=provider_id,
        profile_id=profile_id,
        model=model,
        dataset_id=dataset_id,
        variant=variant,
        tasks=tasks,
    )
    started_at_utc = str(row.get("started_at_utc", "")) or eu.utc_now_iso()
    resumed_at: list[str] = list(rp.read_resume_log(root, run_id))
    if record:
        resumed_at = rp.record_resume(
            root, run_id, eu.utc_now_iso(), started_at_utc=started_at_utc
        )
        eu.logger.info(
            "resuming run %s started at %s (resume #%d)",
            run_id,
            started_at_utc,
            len(resumed_at),
        )
    return ResumeState(
        run_id=str(run_id),
        started_at_utc=started_at_utc,
        notes=str(row.get("notes", "")),
        resumed_at=tuple(resumed_at),
    )


def prepare_run_directory(
    root: str | Path, run_id: str, args: RunnerArgs, *, resume: ResumeState | None
) -> None:
    """Point the logger at this run's log file and stamp its resolved config.

    Configured before planning so planning-time warnings land in this run's log
    file. A resumed run keeps the resolved composition (and therefore the
    `resolved_config_sha` in its registry notes) that its first attempt wrote.
    """
    eu.configure_run_logging(args.log_level, log_path=eu.run_log_path(root, run_id))
    if resume is not None and rp.resolved_config_path(root, run_id).exists():
        return
    # No-op unless the run was composed by scripts/run.py.
    rp.write_resolved_config(root, run_id, getattr(args, "resolved_config_yaml", ""))


# =============================================================================
# Per-cell lease
# =============================================================================

LEASE_SUFFIX = ".lease.json"
#: A lease whose heartbeat is older than this is treated as abandoned even if
#: some process still holds its pid. Long enough to cover a slow provider batch.
LEASE_STALE_AFTER_S = 1800.0


class CellLeaseError(RuntimeError):
    """Another live runner already claimed this cell."""


def cell_lease_key(
    *, profile_id: str, model: str, dataset_id: str, variant: str
) -> str:
    """Identify the (profile, model, dataset, variant) cell inside one run."""
    return f"{profile_id}-{model}-{dataset_id}-{variant}"


def cell_lease_path(root: str | Path, run_id: str, cell_key: str) -> Path:
    """Sibling of the run log holding the lease for one cell of `run_id`."""
    return (
        Path(root)
        / "data/processed/logs"
        / f"{eu.safe_identifier(run_id)}.{eu.safe_identifier(cell_key)}{LEASE_SUFFIX}"
    )


def _iso_to_epoch(value: Any) -> float | None:
    try:
        return float(calendar.timegm(time.strptime(str(value), "%Y-%m-%dT%H:%M:%SZ")))
    except (TypeError, ValueError):
        return None


def _record_pid(record: Mapping[str, Any]) -> int:
    """The owning pid of a lease record, or 0 when it is missing or corrupt."""
    try:
        return int(record.get("pid", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        # PermissionError and friends mean the pid exists but is not ours.
        return True
    return True


@dataclass(slots=True)
class CellLease:
    """Advisory claim on one cell, held for the duration of its provider calls.

    The storage locks in `eval_utils` serialize individual appends and registry
    rewrites; they say nothing about who owns the `read pending -> call
    provider -> append` transaction. This lease is that missing layer: it is
    claimed before the first provider call of a cell and released when the cell
    reaches a terminal state.

    It stays advisory on purpose. A lease whose owning process is gone, or whose
    heartbeat has aged past `stale_after_s`, is taken over with a warning rather
    than blocking recovery after a crash.
    """

    path: Path
    run_id: str
    cell_key: str
    stale_after_s: float = LEASE_STALE_AFTER_S
    held: bool = field(default=False, init=False)

    @classmethod
    def for_cell(
        cls,
        root: str | Path,
        *,
        run_id: str,
        profile_id: str,
        model: str,
        dataset_id: str,
        variant: str,
        stale_after_s: float = LEASE_STALE_AFTER_S,
    ) -> CellLease:
        cell_key = cell_lease_key(
            profile_id=profile_id, model=model, dataset_id=dataset_id, variant=variant
        )
        return cls(
            path=cell_lease_path(root, run_id, cell_key),
            run_id=str(run_id),
            cell_key=cell_key,
            stale_after_s=stale_after_s,
        )

    def _read(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _record(self, claimed_at: str) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "cell_key": self.cell_key,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "claimed_at_utc": claimed_at,
            "heartbeat_utc": eu.utc_now_iso(),
        }

    def _is_live(self, record: Mapping[str, Any]) -> bool:
        """True when another runner still owns this lease."""
        heartbeat = _iso_to_epoch(
            record.get("heartbeat_utc") or record.get("claimed_at_utc")
        )
        if heartbeat is None or time.time() - heartbeat > self.stale_after_s:
            return False
        if str(record.get("hostname", "")) != socket.gethostname():
            # A different machine's pid is not ours to probe; the fresh
            # heartbeat is the only evidence available, and it says "live".
            return True
        return _pid_is_running(_record_pid(record))

    def claim(self) -> None:
        """Take the lease, or raise `CellLeaseError` if a live runner holds it."""
        claimed_at = eu.utc_now_iso()
        with eu.file_lock(self.path):
            existing = self._read()
            if existing and self._is_live(existing):
                raise CellLeaseError(
                    f"Cell {self.cell_key!r} of run {self.run_id!r} is already claimed "
                    f"by pid {existing.get('pid')} on "
                    f"{existing.get('hostname')} since "
                    f"{existing.get('claimed_at_utc')} (lease {self.path}). "
                    "Wait for that runner to finish, or remove the lease file if you "
                    "are certain it is dead."
                )
            if existing:
                eu.logger.warning(
                    "taking over stale lease for cell %s of run %s "
                    "(pid %s on %s, last heartbeat %s)",
                    self.cell_key,
                    self.run_id,
                    existing.get("pid"),
                    existing.get("hostname"),
                    existing.get("heartbeat_utc") or existing.get("claimed_at_utc"),
                )
            eu.write_json(self.path, self._record(claimed_at))
            self.held = True

    def heartbeat(self) -> None:
        """Refresh the timestamp so a long cell is not mistaken for abandoned."""
        if not self.held:
            return
        with eu.file_lock(self.path):
            record = self._read()
            if _record_pid(record) != os.getpid():
                return
            record["heartbeat_utc"] = eu.utc_now_iso()
            eu.write_json(self.path, record)

    def release(self) -> None:
        """Drop the lease. Only ever removes a lease this process still owns."""
        if not self.held:
            return
        self.held = False
        with eu.file_lock(self.path):
            record = self._read()
            if _record_pid(record) == os.getpid():
                self.path.unlink(missing_ok=True)


# =============================================================================
# Shared execute-cell lifecycle
# =============================================================================


@runtime_checkable
class LifecycleCell(Protocol):
    """The attributes `execute_cell` reads off a planned cell.

    Structural, so `RunCell` and `Task3Cell` satisfy it without a shared base
    class; the fields that differ between them (item population, source-run
    provenance) are supplied through `CellExecution` instead.
    """

    #: Read for `concurrency` only; every other provider setting is already
    #: baked into the planned jobs.
    profile: dict[str, Any]
    model: str
    run_id: str
    jobs: list[dict[str, Any]]
    existing_rows: list[dict[str, Any]]
    pending_jobs: list[dict[str, Any]]
    planned_api_calls: int
    output_path: Path
    registry_path: Path
    progress_path: Path
    events_path: Path

    @property
    def batch_size(self) -> int: ...


@dataclass(frozen=True, slots=True)
class CellExecution:
    """One planned cell plus the task-specific hooks the lifecycle calls.

    Everything here is resolved by the runner before execution starts, so the
    lifecycle itself makes no Task 1/2 versus Task 3 decisions.
    """

    cell: LifecycleCell
    #: Task filter for row selection, the registry row, and the `tasks` event
    #: field: `["task1", "task2"]` for the experiment runner, `["task3"]` here.
    tasks: Sequence[str]
    #: The item population progress is measured against: benchmark items for
    #: Task 1/2, built audit items for Task 3.
    progress_items: list[dict[str, Any]]
    expected_stochastic_samples: int
    logging_config: Mapping[str, Any]
    completion_fn: Callable[..., dict[str, Any]]
    #: Identity block stamped onto every run event and warning event.
    context: Mapping[str, Any]
    mode: str
    #: `(raw_rows, *, status=None, finished_at_utc="") -> registry row`.
    registry_row: Callable[..., dict[str, Any]]
    #: The dict logged once, before the first provider call.
    start_log: Mapping[str, Any]
    #: Event fields only one runner has (Task 3's source run and audit mode).
    event_fields: Mapping[str, Any] = field(default_factory=dict)
    #: Applied to every raw record before it is appended (Task 1/2 stamps the
    #: resolved embedding backend, which the analysis process cannot inherit).
    stamp_record: Callable[[dict[str, Any]], None] | None = None
    registry_label: str = "Registry status"
    lease: CellLease | None = None


def execute_cell(execution: CellExecution) -> None:
    """Run one planned cell to completion, streaming progress artifacts.

    Claims the cell's lease, streams the pending jobs through the provider, and
    always leaves the registry in a terminal state: `complete` / `partial` on a
    normal return, `interrupted` on Ctrl-C, `failed` on any other exception.
    """
    cell = execution.cell
    logging_config = execution.logging_config
    current_rows = list(cell.existing_rows)
    started_monotonic = time.monotonic()
    emitted_warning_types: set[str] = set()

    def refresh_live_progress(
        event_type: str, finished_at_utc: str = "", print_line: bool = True
    ) -> dict[str, Any]:
        run_rows = eu.select_model_run_rows(
            current_rows, cell.run_id, cell.model, execution.tasks
        )
        pending_jobs_now = eu.pending_completion_jobs(
            cell.jobs, current_rows, cell.run_id
        )
        pending_api_calls_now = len(
            eu.completion_job_batches(
                pending_jobs_now, cell.batch_size, planned_jobs=cell.jobs
            )
        )
        status = (
            "running"
            if event_type in {"start", "progress"} and pending_jobs_now
            else None
        )
        eu.upsert_run_registry_row(
            cell.registry_path,
            execution.registry_row(
                current_rows, status=status, finished_at_utc=finished_at_utc
            ),
        )
        if logging_config["write_progress_csv"]:
            eu.write_live_progress_csv(
                cell.progress_path,
                execution.progress_items,
                run_rows,
                expected_stochastic_samples=execution.expected_stochastic_samples,
            )
        counters = eu.live_run_counters(
            run_rows,
            expected_records=len(cell.jobs),
            expected_api_calls=cell.planned_api_calls,
            started_monotonic=started_monotonic,
        )
        event = {
            "event_type": event_type,
            **execution.context,
            **execution.event_fields,
            "tasks": list(execution.tasks),
            "mode": execution.mode,
            "output_path": str(cell.output_path),
            "registry_path": str(cell.registry_path),
            "progress_path": str(cell.progress_path),
            "planned_jobs": len(cell.jobs),
            "pending_jobs": len(pending_jobs_now),
            "planned_api_calls": cell.planned_api_calls,
            "pending_api_calls": pending_api_calls_now,
            **counters,
        }
        if logging_config["write_event_jsonl"]:
            eu.append_run_event(cell.events_path, event)
        if print_line and event_type in {"progress", "finish"}:
            eu.logger.info("%s", eu.format_live_progress_line(cell.run_id, counters))
        if event_type == "finish":
            eu.logger.info(
                "%s",
                eu.format_run_quality_line(
                    cell.run_id, eu.run_quality_counters(run_rows)
                ),
            )
        return counters

    def reconcile_terminal(status: str) -> None:
        """Replace the `running` row with a terminal one after an abort."""
        try:
            eu.upsert_run_registry_row(
                cell.registry_path,
                execution.registry_row(
                    current_rows, status=status, finished_at_utc=eu.utc_now_iso()
                ),
            )
        except Exception:  # pragma: no cover - must never mask the real failure
            eu.logger.exception(
                "could not reconcile %s to %s in %s",
                cell.run_id,
                status,
                cell.registry_path,
            )
            return
        eu.logger.warning(
            "%s: %s after %d/%d records; registry reconciled in %s",
            cell.run_id,
            status,
            len(current_rows) - len(cell.existing_rows),
            len(cell.pending_jobs),
            cell.registry_path,
        )

    if execution.lease is not None:
        execution.lease.claim()
    try:
        eu.upsert_run_registry_row(
            cell.registry_path,
            execution.registry_row(
                current_rows, status="running" if cell.pending_jobs else None
            ),
        )
        refresh_live_progress("start", print_line=False)
        eu.logger.info("%s", dict(execution.start_log))

        progress_every_records = int(logging_config["progress_every_records"])
        progress_every_seconds = int(logging_config["progress_every_seconds"])
        last_progress_record_index = 0
        last_progress_monotonic = time.monotonic()
        for index, record in enumerate(
            eu.run_completion_jobs(
                cell.pending_jobs,
                max_workers=int(cell.profile["concurrency"]),
                completion_fn=execution.completion_fn,
                batch_size=cell.batch_size,
                planned_jobs=cell.jobs,
            ),
            start=1,
        ):
            if execution.stamp_record is not None:
                execution.stamp_record(record)
            eu.append_jsonl(cell.output_path, record)
            current_rows.append(record)
            now_monotonic = time.monotonic()
            records_due = index - last_progress_record_index >= progress_every_records
            seconds_due = (
                progress_every_seconds > 0
                and now_monotonic - last_progress_monotonic >= progress_every_seconds
            )
            if records_due or seconds_due or index == len(cell.pending_jobs):
                counters = refresh_live_progress("progress")
                last_progress_record_index = index
                last_progress_monotonic = now_monotonic
                if execution.lease is not None:
                    execution.lease.heartbeat()
                eu.emit_warning_events(
                    counters,
                    logging_config=logging_config,
                    emitted_warning_types=emitted_warning_types,
                    context=execution.context,
                    events_path=cell.events_path,
                )

        finish_row = execution.registry_row(
            current_rows, finished_at_utc=eu.utc_now_iso()
        )
        eu.upsert_run_registry_row(cell.registry_path, finish_row)
        refresh_live_progress(
            "finish", finished_at_utc=str(finish_row["finished_at_utc"])
        )
        eu.logger.info(
            "%s: %s at %s",
            execution.registry_label,
            finish_row["status"],
            cell.registry_path,
        )
    except KeyboardInterrupt:
        reconcile_terminal("interrupted")
        raise
    except BaseException:
        reconcile_terminal("failed")
        raise
    finally:
        if execution.lease is not None:
            execution.lease.release()
