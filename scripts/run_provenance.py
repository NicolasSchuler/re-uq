"""Provenance helpers shared by the JSON and Hydra run entry points.

The runners record *how* a run was configured, not just what it produced. When
a run is launched through the Hydra layer (`scripts/run.py`) the fully resolved
composition is dumped next to the run log as
`data/processed/logs/<run_id>.resolved.yaml`, and its SHA-256 is appended to the
run-registry `notes` field as `resolved_config_sha=<sha>`.

The legacy JSON path passes no resolved config, so `run_notes()` reproduces the
previous `notes` strings byte-for-byte and `write_resolved_config()` is a no-op.
This module deliberately depends on the standard library only, so the runners
stay importable without Hydra/OmegaConf installed.

A resumed run keeps that first composition. Rather than rewriting the run's
start time and resolved provenance, each `--mode resume` attempt is appended to
a `data/processed/logs/<run_id>.resume.json` sidecar, so the registry keeps
saying when the run started while the sidecar says how often it was continued.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

try:
    import eval_utils as eu
except ModuleNotFoundError:  # pragma: no cover
    from scripts import eval_utils as eu


RESOLVED_CONFIG_SUFFIX = ".resolved.yaml"
RESUME_LOG_SUFFIX = ".resume.json"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolved_config_path(root: str | Path, run_id: str) -> Path:
    """Sibling of `eval_utils.run_log_path` holding the resolved composition."""
    return (
        Path(root)
        / "data/processed/logs"
        / f"{eu.safe_identifier(run_id)}{RESOLVED_CONFIG_SUFFIX}"
    )


def write_resolved_config(root: str | Path, run_id: str, yaml_text: str) -> str:
    """Persist the resolved config for `run_id` and return its SHA-256.

    Returns "" (and writes nothing) when there is no resolved config, which is
    the case for every run launched from the legacy JSON CLI.
    """
    if not yaml_text:
        return ""
    path = resolved_config_path(root, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml_text, encoding="utf-8")
    return sha256_text(yaml_text)


def resume_log_path(root: str | Path, run_id: str) -> Path:
    """Sibling of `resolved_config_path` holding this run's `resumed_at` list."""
    return (
        Path(root)
        / "data/processed/logs"
        / f"{eu.safe_identifier(run_id)}{RESUME_LOG_SUFFIX}"
    )


def read_resume_log(root: str | Path, run_id: str) -> list[str]:
    """Timestamps at which `run_id` was resumed, oldest first ([] if never)."""
    path = resume_log_path(root, run_id)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []
    return [str(value) for value in payload.get("resumed_at", []) or []]


def record_resume(
    root: str | Path, run_id: str, at: str, *, started_at_utc: str = ""
) -> list[str]:
    """Append one resume timestamp for `run_id` and return the full list.

    Serialized with the same advisory lock the storage helpers use, so two
    runners resuming the same run cannot lose an entry.
    """
    path = resume_log_path(root, run_id)
    with eu.file_lock(path):
        resumed_at = read_resume_log(root, run_id)
        resumed_at.append(str(at))
        eu.write_json(
            path,
            {
                "run_id": str(run_id),
                "started_at_utc": str(started_at_utc),
                "resumed_at": resumed_at,
            },
        )
    return resumed_at


def run_notes(args: Any, extra: str = "") -> str:
    """Registry `notes` value for a run launched with `args`.

    Identical to the historical `f"mode={...}"` strings unless the caller
    supplied a resolved Hydra config, in which case its digest is appended.
    """
    parts = [f"mode={getattr(args, 'mode', '')}"]
    if extra:
        parts.append(extra)
    resolved_sha = str(getattr(args, "resolved_config_sha", "") or "")
    if resolved_sha:
        parts.append(f"resolved_config_sha={resolved_sha}")
    return "; ".join(parts)
