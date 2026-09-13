"""Compact append-only raw JSONL files into their Parquet siblings.

Every raw file under ``data/processed`` (model outputs, Task 3 audits, run
events, request transcripts) is appended one JSON line at a time while a run
is in flight. Once the runs are finished, this moves the accumulated lines into
``<stem>.parquet`` (zstd, ~40x smaller) and truncates the JSONL to an empty
tail. Readers (`eval_utils.read_jsonl`, `read_raw_store`) return both halves,
so nothing downstream changes; a later run appends to the empty tail and the
next compaction folds it in.

    .venv/bin/python scripts/compact_raw_store.py            # everything
    .venv/bin/python scripts/compact_raw_store.py --dry-run  # sizes only
    .venv/bin/python scripts/compact_raw_store.py data/processed/model_outputs_raw_pure.jsonl

Never run this while a generation run is appending to the same file: the
compaction takes the file's exclusive lock, so a runner would simply wait,
but a runner that crashed mid-line would lose that line's tail. Refuses a
file whose ``.lock`` sidecar is currently held.
"""

from __future__ import annotations

import argparse
import fcntl
import sys
from pathlib import Path

try:
    import eval_utils as eu
except ImportError:  # pragma: no cover - exercised when run as a module
    from scripts import eval_utils as eu

DEFAULT_GLOBS = (
    "data/processed/model_outputs_raw*.jsonl",
    "data/processed/run_events*.jsonl",
    "data/processed/smoke/model_outputs_raw*.jsonl",
    "data/processed/smoke/run_events*.jsonl",
    "data/processed/logs/*.transcript.jsonl",
)


def default_targets(root: Path) -> list[Path]:
    seen: dict[Path, None] = {}
    for pattern in DEFAULT_GLOBS:
        for path in sorted(root.glob(pattern)):
            seen.setdefault(path, None)
    return list(seen)


def lock_is_held(path: Path) -> bool:
    """Whether another process currently holds the raw file's advisory lock."""
    lock_path = Path(str(path) + ".lock")
    if not lock_path.exists():
        return False
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return False


def human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:7.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Raw JSONL files to compact (default: every raw file under data/processed).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report sizes without writing."
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the read-back comparison before truncating (not recommended).",
    )
    args = parser.parse_args(argv)
    root = eu.project_root()
    targets = args.paths or default_targets(root)

    total_before = total_after = 0
    failures = 0
    for path in targets:
        if not path.exists():
            print(f"skip   {path}: no JSONL tail")
            continue
        tail_bytes = path.stat().st_size
        if tail_bytes == 0:
            continue
        if lock_is_held(path):
            print(f"busy   {path}: lock held by another process, skipped")
            failures += 1
            continue
        if args.dry_run:
            print(f"would  {path}: {human(tail_bytes)} of JSONL tail")
            total_before += tail_bytes
            continue
        try:
            summary = eu.compact_jsonl(path, verify=not args.no_verify)
        except Exception as error:
            print(f"FAILED {path}: {error}", file=sys.stderr)
            failures += 1
            continue
        total_before += summary["tail_bytes_before"]
        total_after += summary["store_bytes_after"]
        print(
            f"done   {path}: {summary['tail_rows']} rows, "
            f"{human(summary['tail_bytes_before'])} -> "
            f"{human(summary['store_bytes_after'])} "
            f"({summary['store_rows_after']} rows in store)"
        )
    if args.dry_run:
        print(f"\n{human(total_before)} of JSONL tail would be compacted.")
    else:
        print(
            f"\nCompacted {human(total_before)} of JSONL into "
            f"{human(total_after)} of Parquet."
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
