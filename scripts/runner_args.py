"""The attribute contract the Task 1/2/3 runners expect from their arguments.

Both runners accept an options object rather than a long argument list, and
that object is built by two different front doors: `argparse.Namespace` from
the JSON CLIs and `types.SimpleNamespace` from the Hydra bridge. Neither is a
declared type, so the runners used to take `args: Any` and the contract lived
only in the bodies that read it.

These protocols write that contract down. They are structural, so no caller has
to change or subclass anything, and they stay the single place to look when a
new option has to be threaded through both entry points.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RunnerArgs(Protocol):
    """Options common to every runner entry point."""

    profile: str | None
    model: str | None
    all_models: bool
    dataset: str | None
    variant: str | None
    task: str | None
    mode: str
    run_id: str | None
    smoke_items: int
    fake_completion: bool
    dry_run: bool
    log_level: str

    # Live-progress overrides; `None` means "fall back to the run config".
    progress_every_records: int | None
    progress_every_seconds: int | None
    warn_after_records: int | None
    warn_parse_failure_rate: float | None
    warn_request_error_rate: float | None
    no_progress_artifacts: bool
    #: Opt out of the per-request transcript sidecar.
    no_request_transcripts: bool


@runtime_checkable
class Task3RunnerArgs(RunnerArgs, Protocol):
    """Options the Task 3 verification runner adds on top of `RunnerArgs`."""

    source_run_id: str | None
    audit_mode: str
    allow_partial_source: bool
    #: Audit Task 2 rows produced under another provider profile.
    allow_source_profile_mismatch: bool
