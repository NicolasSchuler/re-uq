"""Generated failures remain samples; reference-free audits remain observable."""

import json
from typing import ClassVar
from unittest.mock import patch

from scripts import eval_utils as eu
from tests.test_run_transcripts import SEEDS, TASK2_ANSWER, TranscriptTestCase


class NativeOutputError(Exception):
    status_code = 500
    body: ClassVar[dict[str, str]] = {
        "error": "The model produced output that does not match the expected peg-native format"
    }


class GenerationFailureTests(TranscriptTestCase):
    def test_instructor_does_not_resample_native_failure_internally(self):
        import httpx
        from openai import OpenAI

        calls = []

        def respond(request):
            calls.append(request)
            return httpx.Response(500, json=NativeOutputError.body)

        client = OpenAI(
            api_key="dummy",
            base_url="http://unused/v1",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        )
        self.addCleanup(client.close)
        with patch.object(eu, "OpenAI", return_value=client):
            result = eu.instructor_completion(
                "http://unused/v1", "m1", "JSON please", 0.7, 1.0, validation_retries=3
            )
        self.assertEqual(len(calls), 1)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error_kind"], "model_output_format")
        self.assertEqual(result["response_json"], NativeOutputError.body)

    def test_native_rejection_is_not_retried_or_hidden_in_transcripts(self):
        with (
            patch.object(eu, "OpenAI") as client,
            patch.object(eu.time, "sleep") as sleep,
        ):
            client.return_value.chat.completions.create.side_effect = NativeOutputError(
                "rejected"
            )
            completion = eu.chat_completion("http://unused/v1", "m1", "request", 0.7, 1)
        self.assertEqual(client.return_value.chat.completions.create.call_count, 1)
        sleep.assert_not_called()
        self.assertEqual(completion["error_kind"], "model_output_format")
        records = self.run_jobs(self.plan()[:1], lambda **_: completion, batch_size=1)
        self.assertEqual(records[0]["parse_status"], "model_output_error")
        self.assertEqual(self.rows()[0]["error_kind"], "model_output_format")

    def test_native_rejection_has_no_batch_fallback_in_either_driver(self):
        failure = {
            "ok": False,
            **eu.provider_error_record(NativeOutputError("rejected")),
        }
        for mode in ("none", "instructor"):
            with (
                self.subTest(mode=mode),
                patch.object(
                    eu, "instructor_completion", return_value=failure
                ) as instructor,
            ):
                jobs = self.plan(structured_output=mode)[:2]
                calls = []

                def completion(calls=calls, **kwargs):
                    calls.append(kwargs)
                    return failure

                records = self.run_jobs(jobs, completion, batch_size=2)
                self.assertEqual(len(records), 2)
                self.assertTrue(
                    all(r["parse_status"] == "model_output_error" for r in records)
                )
                self.assertTrue(all(r["batch_size"] == 2 for r in records))
                self.assertLessEqual(len(calls) + instructor.call_count, 1)

    def test_native_failure_retains_five_sample_denominator(self):
        jobs = self.plan(stochastic={"temperature": 0.7, "top_p": 1.0, "samples": 5})[
            :6
        ]

        def completion(**kwargs):
            if kwargs["seed"] == jobs[-1]["seed"]:
                return {
                    "ok": False,
                    **eu.provider_error_record(NativeOutputError("rejected")),
                }
            return {"ok": True, "raw_text": json.dumps(TASK2_ANSWER)}

        rows = self.run_jobs(jobs, completion, batch_size=1)
        self.assertEqual(len(rows), 6)
        self.assertEqual(
            sum(r["parse_status"] == "model_output_error" for r in rows), 1
        )
        self.assertEqual(eu.pending_completion_jobs(jobs, rows, "full-1"), [])
        changed = [{**jobs[-1], "job_config_sha": "different-config"}]
        self.assertEqual(len(eu.pending_completion_jobs(changed, rows, "full-1")), 1)
        with patch.object(
            eu,
            "acse_semantic_proxy_diagnostics",
            return_value={"semantic_uncertainty_score": 0.0},
        ):
            scores = eu.build_uq_scores(
                eu.build_benchmark_items(SEEDS),
                rows,
                sampling_plan=eu.SamplingPlan(stochastic_samples=5),
            )
        repeated = next(r for r in scores if r["uq_method"] == "modality_consistency")
        self.assertEqual(
            (repeated["valid_n"], repeated["total_n"], repeated["parse_failures"]),
            (4, 5, 1),
        )
        self.assertFalse(repeated["stochastic_complete"])

    def test_transport_failure_remains_resumable(self):
        job = self.plan()[0]
        row = eu.run_completion_job(
            job, completion_fn=lambda **_: {"ok": False, "error_kind": "provider_error"}
        )
        self.assertEqual(len(eu.pending_completion_jobs([job], [row], "full-1")), 1)


class AuditCoverageTests(TranscriptTestCase):
    def sources(self):
        texts = [
            "The system must export reports.",
            "export reports",
            "The system must not export reports.",
            "   ",
        ]
        jobs = self.plan()[:4]
        return [
            eu.run_completion_job(
                job,
                completion_fn=lambda text=text, **_: {
                    "ok": True,
                    "raw_text": json.dumps({**TASK2_ANSWER, "requirement": text}),
                },
            )
            for job, text in zip(jobs, texts, strict=True)
        ]

    def test_unknown_and_negated_sources_are_audited_not_scored(self):
        items = eu.build_task3_verification_items(
            eu.build_benchmark_items(SEEDS), self.sources()
        )
        self.assertEqual(len(items), 3)
        self.assertEqual(
            [i["task3_reference_status"] for i in items],
            ["available", "unknown_text_modality", "negated_text_modality"],
        )
        raw = [
            {
                **i,
                "model": "m1",
                "run_id": "audit",
                "task": "task3",
                "sample_kind": "deterministic",
                "sample_index": 0,
                "parse_status": "ok",
                "parsed_json": {"relation": "preserves", "confidence": 0.9},
                "prompt_version": "v2-conf01",
            }
            for i in items
        ]
        self.assertEqual(len(eu.task3_items_from_raw_rows(raw)), 3)
        scores = eu.build_task3_scores(items, raw)
        self.assertEqual(len(scores), 1)
        review = eu.task3_audit_review_rows(items, raw[:2])
        coverage = eu.task3_audit_coverage_rows(review)[0]
        self.assertEqual(coverage["auditable_source_items"], 3)
        self.assertEqual(coverage["reference_unavailable_items"], 2)
        self.assertEqual(coverage["parsed_unscored_audits"], 1)
        self.assertEqual(coverage["missing_deterministic_audits"], 1)
        self.assertFalse(any("correct" in row or "y_true" in row for row in review))
        self.assertFalse(any(row["stochastic_complete"] for row in review))

    def test_task3_missing_sample_uses_planned_denominator(self):
        item = eu.build_task3_verification_items(
            eu.build_benchmark_items(SEEDS), self.sources()
        )[0]
        raw = [
            {
                **item,
                "model": "m1",
                "run_id": "audit",
                "task": "task3",
                "sample_kind": "stochastic",
                "sample_index": index,
                "parse_status": "ok",
                "parsed_json": {"relation": "preserves", "confidence": 0.9},
                "prompt_version": "v2-conf01",
            }
            for index in range(4)
        ]
        with patch.object(
            eu,
            "acse_semantic_proxy_diagnostics",
            return_value={"semantic_uncertainty_score": 0.0},
        ):
            scores = eu.build_task3_scores(
                [item], raw, sampling_plan=eu.SamplingPlan(stochastic_samples=5)
            )
        repeated = next(r for r in scores if r["uq_method"] == "relation_consistency")
        self.assertEqual(
            (repeated["valid_n"], repeated["total_n"], repeated["parse_failures"]),
            (4, 5, 1),
        )
        self.assertFalse(repeated["stochastic_complete"])

    def test_legacy_blank_reference_is_not_treated_as_explicitly_unscored(self):
        item = eu.build_task3_verification_items(
            eu.build_benchmark_items(SEEDS), self.sources()
        )[1]
        raw = {**item, "task": "task3"}
        raw.pop("task3_reference_status")
        self.assertEqual(eu.task3_items_from_raw_rows([raw]), [])
