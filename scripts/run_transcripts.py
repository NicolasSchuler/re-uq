"""Per-request transcripts: what was sent, and what came back.

Raw rows record derived fields plus fingerprints -- `request_payload_sha` for
the request, `batch_prompt_hash` for a batch wrapper -- which is enough to show
that two runs asked the same thing, but not to read what a provider was asked
or what it answered. On a batched row `prompt` holds the single-item prompt
rather than the 16-item text that was actually sent, and `raw_text` holds the
per-item slice the parser cut out of the batch response, so neither column
reconstructs the exchange. The full response body is discarded entirely once
`eval_utils.extract_response_fields` has taken the provenance off it.

This module writes that exchange to a sidecar beside the run log,
``data/processed/logs/<run_id>.transcript.jsonl``, in the same family as
``<run_id>.resolved.yaml`` and ``<run_id>.resume.json``.

One row per provider CALL and per attempt, not per item: a 16-item request
stores its prompt once instead of sixteen times, and the transport attempts
that `eval_utils.call_with_retries` otherwise only counts each get a row of
their own. API keys never appear: the key lives on the client, not in the
payload that is hashed and recorded here.

Writing is opt-out through the run logging config key
``write_request_transcripts`` (see `eval_utils.DEFAULT_RUN_LOGGING`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import eval_utils as eu
except ModuleNotFoundError:  # pragma: no cover
    from scripts import eval_utils as eu


TRANSCRIPT_SUFFIX = ".transcript.jsonl"

#: Column order of a transcript row, documented for readers of the JSONL. The
#: rows are written with sorted keys (append_jsonl), so this is the reading
#: order, not the on-disk order.
TRANSCRIPT_FIELDS = (
    # which call
    "run_id",
    "run_group_id",
    "profile_id",
    "provider_id",
    "model",
    "task",
    "sample_kind",
    "sample_index",
    "batch_id",
    "batch_size",
    "request_indices",
    "attempt",
    "attempt_kind",
    "started_at_utc",
    # what was sent
    "request_payload",
    "request_payload_sha",
    "request_seed",
    # what came back
    "response_json",
    "raw_text",
    "finish_reason",
    "usage_prompt_tokens",
    "usage_completion_tokens",
    "usage_total_tokens",
    "served_model",
    "system_fingerprint",
    "response_id",
    "latency_s",
    "retry_count",
    "error",
)

#: `attempt_kind` values: a failed transport attempt that `call_with_retries`
#: retried, versus the attempt whose outcome the raw row reports.
ATTEMPT_RETRIED = "retried"
ATTEMPT_FINAL = "final"


def transcript_path(root: str | Path, run_id: str) -> Path:
    """Sibling of `eval_utils.run_log_path` holding this run's transcript."""
    return (
        Path(root)
        / "data/processed/logs"
        / f"{eu.safe_identifier(run_id)}{TRANSCRIPT_SUFFIX}"
    )


class TranscriptWriter:
    """Appends one row per provider call to a run's transcript sidecar.

    Threads share one writer: `eval_utils.append_jsonl` appends under an
    advisory file lock, so concurrent batches cannot interleave inside a line.
    """

    __slots__ = ("path",)

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def for_run(
        cls, root: str | Path, run_id: str, *, enabled: bool = True
    ) -> TranscriptWriter | None:
        """Writer for `run_id`, or None when transcripts are turned off."""
        if not enabled:
            return None
        return cls(transcript_path(root, run_id))

    def record(
        self,
        job: Any,
        completion: Any,
        *,
        batch_id: str = "",
        batch_size: int = 1,
        request_indices: Any = (),
    ) -> None:
        """Write the attempts behind one provider call.

        `job` is any planned completion job of the call (the first one, for a
        batch) and supplies the run identity; `completion` is the dict the
        completion function returned, carrying `request_payload`, the response
        body, and the errors of any attempts that were retried.
        """
        started_at = eu.utc_now_iso()
        base = {
            "run_id": str(job.get("run_id", "")),
            "run_group_id": str(job.get("run_group_id", "")),
            "profile_id": str(job.get("profile_id", "")),
            "provider_id": str(job.get("provider_id", "")),
            "model": str(job.get("model", "")),
            "task": str(job.get("task", "")),
            "sample_kind": str(job.get("sample_kind", "")),
            "sample_index": int(job.get("sample_index", 0)),
            "batch_id": str(batch_id),
            "batch_size": int(batch_size),
            "request_indices": [int(index) for index in request_indices],
            "started_at_utc": started_at,
            "request_payload": completion.get("request_payload"),
            "request_payload_sha": str(completion.get("request_payload_sha", "")),
            "request_seed": completion.get("request_seed"),
        }

        # Attempts that failed and were retried: the provider never returned a
        # body, so only the error is recordable. Without these rows a run that
        # eventually succeeded looks like it succeeded first time.
        for attempt, error in enumerate(completion.get("attempt_errors") or []):
            eu.append_jsonl(
                self.path,
                {
                    **base,
                    "attempt": attempt,
                    "attempt_kind": ATTEMPT_RETRIED,
                    "response_json": None,
                    "raw_text": "",
                    **eu.EMPTY_RESPONSE_FIELDS,
                    "latency_s": None,
                    "retry_count": attempt,
                    "error": str(error),
                },
            )

        retry_count = int(completion.get("retry_count", 0) or 0)
        response_json = completion.get("response_json")
        eu.append_jsonl(
            self.path,
            {
                **base,
                "attempt": retry_count,
                "attempt_kind": ATTEMPT_FINAL,
                "response_json": response_json,
                "raw_text": str(completion.get("raw_text", "")),
                **eu.extract_response_fields(response_json),
                "latency_s": completion.get("latency_s"),
                "retry_count": retry_count,
                "error": str(completion.get("error", "")),
            },
        )
