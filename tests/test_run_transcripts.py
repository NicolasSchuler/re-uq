"""Per-request transcripts: one row per provider call, payload and body kept.

The raw rows keep fingerprints (`request_payload_sha`, `batch_prompt_hash`) but
neither the text that was sent nor the body that came back, and on a batched row
`prompt`/`raw_text` describe the item rather than the request. These tests pin
the sidecar that closes that gap: that it writes per CALL and not per item, that
what it stores re-hashes to the fingerprint the raw row reports, that the bodies
lost on a failed batch and on a retried attempt are recorded, and that the
logging knob turns the whole thing off.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import eval_utils as eu, run_transcripts as rt

SEEDS = [
    {
        "seed_id": "S0001",
        "source_dataset": "NICE",
        "original_requirement": "The system shall export reports.",
        "capability_text_final": "export reports",
    },
    {
        "seed_id": "S0002",
        "source_dataset": "NICE",
        "original_requirement": "The system shall archive invoices.",
        "capability_text_final": "archive invoices",
    },
]

TASK2_ANSWER = {
    "requirement": "The system should export reports.",
    "modality": "recommended",
    "confidence": 0.7,
}


def batch_body(request_indices):
    """A well-formed batch response covering every request index."""
    return json.dumps(
        {
            "results": [
                {"request_index": index, **TASK2_ANSWER} for index in request_indices
            ]
        }
    )


class TranscriptTestCase(unittest.TestCase):
    """Plans real Task 2 jobs and runs them through a scripted provider."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = rt.transcript_path(self.root, "full-1")

    def plan(self, **overrides):
        kwargs = {
            "tasks": ["task2"],
            "model": "m1",
            "host": "http://localhost:8000/v1",
            "run_id": "full-1",
            "prompt_version": "v2-conf01",
            "task1_template": eu.load_prompt("prompts/mandatory_entailment.txt"),
            "task2_template": eu.load_prompt("prompts/modality_extraction.txt"),
            "deterministic": {"temperature": 0.0, "top_p": 1.0, "samples": 1},
            "stochastic": {"temperature": 0.7, "top_p": 1.0, "samples": 0},
            "max_tokens": 64,
            "timeout_s": 30,
            "api_key_env": "LOCAL_OPENAI_API_KEY",
        }
        kwargs.update(overrides)
        return eu.planned_completion_jobs(eu.build_benchmark_items(SEEDS), **kwargs)

    def rows(self):
        with self.path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle]

    def run_jobs(self, jobs, completion_fn, *, batch_size, transcript=None):
        writer = rt.TranscriptWriter(self.path) if transcript is None else transcript
        return list(
            eu.run_completion_jobs(
                jobs,
                max_workers=1,
                completion_fn=completion_fn,
                batch_size=batch_size,
                transcript=writer,
            )
        )


class BatchedTranscriptTest(TranscriptTestCase):
    def _completion(self, **kwargs):
        """Stand-in provider returning `self.body` with full call provenance."""
        return {
            "ok": True,
            "raw_text": self.body,
            "response_json": {
                "id": "chatcmpl-1",
                "model": "served-m1",
                "choices": [{"finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 9},
            },
            "latency_s": 0.25,
            "error": "",
            "retry_count": 0,
            "request_payload": {
                "model": kwargs["model"],
                "messages": [{"role": "user", "content": kwargs["prompt"]}],
                "temperature": kwargs["temperature"],
                "top_p": kwargs["top_p"],
                "max_tokens": kwargs["max_tokens"],
            },
            "request_payload_sha": eu.request_payload_sha(
                {
                    "model": kwargs["model"],
                    "messages": [{"role": "user", "content": kwargs["prompt"]}],
                    "temperature": kwargs["temperature"],
                    "top_p": kwargs["top_p"],
                    "max_tokens": kwargs["max_tokens"],
                }
            ),
            "request_seed": kwargs.get("seed"),
            "attempt_errors": [],
        }

    def test_one_row_per_request_not_per_item(self):
        jobs = self.plan()
        self.body = batch_body(range(len(jobs)))

        records = self.run_jobs(jobs, self._completion, batch_size=len(jobs))
        rows = self.rows()

        # Eight benchmark items, one request, one transcript row.
        self.assertEqual(len(records), len(jobs))
        self.assertGreater(len(jobs), 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["batch_size"], len(jobs))
        self.assertEqual(rows[0]["request_indices"], list(range(len(jobs))))
        self.assertEqual(rows[0]["attempt_kind"], rt.ATTEMPT_FINAL)

    def test_stored_payload_rehashes_to_the_raw_row_fingerprint(self):
        jobs = self.plan()
        self.body = batch_body(range(len(jobs)))

        records = self.run_jobs(jobs, self._completion, batch_size=len(jobs))
        row = self.rows()[0]

        self.assertEqual(
            eu.request_payload_sha(row["request_payload"]),
            row["request_payload_sha"],
        )
        # ...and that is the fingerprint every item of the batch reports.
        self.assertEqual(
            {record["request_payload_sha"] for record in records},
            {row["request_payload_sha"]},
        )

    def test_the_batch_prompt_is_stored_once_and_in_full(self):
        jobs = self.plan()
        self.body = batch_body(range(len(jobs)))

        records = self.run_jobs(jobs, self._completion, batch_size=len(jobs))
        row = self.rows()[0]
        sent = row["request_payload"]["messages"][0]["content"]

        # The raw rows carry the single-item prompt, which is not even a
        # substring of the batch wrapper; only the transcript has the text the
        # provider actually saw, and it carries every item of the request.
        self.assertEqual(sent, eu.batch_prompt_for_completion_jobs(jobs))
        self.assertNotEqual(sent, records[0]["prompt"])
        for job in jobs:
            self.assertIn(job["item"]["source_statement"], sent)

    def test_response_provenance_is_extracted_next_to_the_body(self):
        jobs = self.plan()
        self.body = batch_body(range(len(jobs)))

        self.run_jobs(jobs, self._completion, batch_size=len(jobs))
        row = self.rows()[0]

        self.assertEqual(row["raw_text"], self.body)
        self.assertEqual(row["response_json"]["id"], "chatcmpl-1")
        self.assertEqual(row["served_model"], "served-m1")
        self.assertEqual(row["finish_reason"], "stop")
        self.assertEqual(row["usage_completion_tokens"], 9)
        self.assertEqual(row["latency_s"], 0.25)

    def test_unparsable_batch_keeps_its_body_and_logs_every_resend(self):
        jobs = self.plan()
        self.body = "not json at all"
        single_calls = {"count": 0}

        def completion_fn(**kwargs):
            if "request_index" in kwargs["prompt"]:
                return self._completion(**kwargs)
            single_calls["count"] += 1
            return {
                **self._completion(**kwargs),
                "raw_text": json.dumps(TASK2_ANSWER),
            }

        self.run_jobs(jobs, completion_fn, batch_size=len(jobs))
        rows = self.rows()

        # The batch row keeps the body the parser rejected -- today that text is
        # dropped and only a reason string survives on the fallback record.
        self.assertEqual(rows[0]["raw_text"], "not json at all")
        self.assertEqual(rows[0]["batch_size"], len(jobs))
        # One further row per single-item re-send.
        self.assertEqual(single_calls["count"], len(jobs))
        self.assertEqual(len(rows), 1 + len(jobs))
        self.assertTrue(all(row["batch_size"] == 1 for row in rows[1:]))


class RetryTranscriptTest(TranscriptTestCase):
    def test_retried_attempts_get_their_own_rows(self):
        jobs = self.plan()[:1]

        def completion_fn(**kwargs):
            return {
                "ok": True,
                "raw_text": json.dumps(TASK2_ANSWER),
                "response_json": {"model": "served-m1"},
                "latency_s": 1.5,
                "error": "",
                "retry_count": 2,
                "request_payload": {"model": kwargs["model"]},
                "request_payload_sha": "sha-1",
                "request_seed": kwargs.get("seed"),
                "attempt_errors": ["RateLimitError()", "APITimeoutError()"],
            }

        self.run_jobs(jobs, completion_fn, batch_size=1)
        rows = self.rows()

        self.assertEqual(len(rows), 3)
        self.assertEqual(
            [row["attempt_kind"] for row in rows],
            [rt.ATTEMPT_RETRIED, rt.ATTEMPT_RETRIED, rt.ATTEMPT_FINAL],
        )
        self.assertEqual([row["attempt"] for row in rows], [0, 1, 2])
        self.assertEqual(rows[0]["error"], "RateLimitError()")
        self.assertIsNone(rows[1]["response_json"])
        self.assertEqual(rows[2]["error"], "")

    def test_call_with_retries_reports_every_retried_attempt(self):
        seen: list[int] = []
        calls = {"count": 0}

        def flaky():
            calls["count"] += 1
            if calls["count"] < 3:
                raise TimeoutError("slow")
            return "done"

        result, retry_count, error = eu.call_with_retries(
            flaky,
            max_retries=3,
            base_delay_s=0.0,
            on_attempt_error=lambda attempt, _exc: seen.append(attempt),
        )

        self.assertEqual((result, retry_count, error), ("done", 2, None))
        self.assertEqual(seen, [0, 1])


class TranscriptWiringTest(TranscriptTestCase):
    def test_for_run_returns_none_when_transcripts_are_disabled(self):
        self.assertIsNone(
            rt.TranscriptWriter.for_run(self.root, "full-1", enabled=False)
        )
        writer = rt.TranscriptWriter.for_run(self.root, "full-1")
        self.assertEqual(writer.path, self.path)

    def test_no_transcript_is_written_without_a_writer(self):
        jobs = self.plan()[:1]

        def completion_fn(**_kwargs):
            return {
                "ok": True,
                "raw_text": json.dumps(TASK2_ANSWER),
                "response_json": {},
                "latency_s": 0.0,
                "error": "",
            }

        records = list(
            eu.run_completion_jobs(
                jobs, max_workers=1, completion_fn=completion_fn, batch_size=1
            )
        )

        self.assertEqual(len(records), 1)
        self.assertFalse(self.path.exists())

    def test_logging_config_carries_the_transcript_switch(self):
        self.assertTrue(eu.normalize_run_logging_config()["write_request_transcripts"])
        self.assertFalse(
            eu.normalize_run_logging_config({"write_request_transcripts": False})[
                "write_request_transcripts"
            ]
        )

    def test_transcript_path_is_a_run_log_sibling(self):
        self.assertEqual(self.path.parent, eu.run_log_path(self.root, "full-1").parent)
        self.assertTrue(self.path.name.endswith(rt.TRANSCRIPT_SUFFIX))

    def test_payload_render_keeps_json_and_names_the_response_model(self):
        class Task2Batch:
            pass

        payload = eu.transcript_request_payload(
            {"model": "m1", "max_tokens": 64, "response_model": Task2Batch}
        )

        self.assertEqual(payload["model"], "m1")
        self.assertEqual(payload["max_tokens"], 64)
        self.assertEqual(payload["response_model"], "Task2Batch")
        json.dumps(payload)  # must stay serializable


if __name__ == "__main__":
    unittest.main()
