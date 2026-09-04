"""Tests for the canonical observation identity and the attempt ledger.

These cover the four boundaries that used to reconstruct "which observation is
this?" from different subsets of fields (AB#1) and the two views of an
append-only raw file (AB#7):

- `eval_utils.ObservationIdentity` / `completion_record_key()` and the resume
  cache built on them, including the documented legacy fallback;
- the Task 2 -> Task 3 handoff in `run_task3_verification_from_config.py`;
- the paper-facing joins in `export_paper_tables.py`;
- `AttemptLedger` and the registry/progress/live counters derived from it.
"""

from __future__ import annotations

import itertools
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

from scripts import (
    eval_utils as eu,
    export_paper_tables as export,
    run_task3_verification_from_config as task3_cli,
)

#: Raw files are scoped to one dataset/variant by path, so a row that predates
#: the provider matrix carries neither provider/profile nor dataset/variant.
LEGACY_ROW = {
    "run_id": "full-legacy",
    "model": "m1",
    "task": "task2",
    "item_id": "S0001_mandatory",
    "sample_kind": "deterministic",
    "sample_index": 0,
    "parse_status": "ok",
}


def raw_attempt(**overrides: Any) -> dict[str, Any]:
    """One raw row with a complete identity."""
    row = {
        "run_id": "full-1",
        "provider_id": "prov_a",
        "profile_id": "prof_a",
        "model": "m1",
        "task": "task2",
        "item_id": "S0001_mandatory",
        "sample_kind": "deterministic",
        "sample_index": 0,
        "parse_status": "ok",
        "job_config_sha": "sha-1",
    }
    row.update(overrides)
    return row


def job_for(row: dict[str, Any]) -> dict[str, Any]:
    """The planned job a runner would build for the same observation."""
    return {
        "provider_id": row.get("provider_id", ""),
        "profile_id": row.get("profile_id", ""),
        "model": row.get("model", ""),
        "task": row.get("task", ""),
        "item": {"item_id": row.get("item_id", "")},
        "sample_kind": row.get("sample_kind", ""),
        "sample_index": row.get("sample_index", 0),
        "job_config_sha": row.get("job_config_sha", ""),
    }


def sampled_real_raw_rows(limit: int = 200) -> tuple[list[dict[str, Any]], str]:
    """A sample of the checkout's own raw rows, or `[]` when none are present.

    The paper's raw JSONL files are large and not part of the repository, so
    this reads only the first `limit` lines of the first file that exists.
    """
    processed = eu.project_root() / "data/processed"
    for path in sorted(processed.glob("model_outputs_raw*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            rows = [
                json.loads(line)
                for line in itertools.islice(handle, limit)
                if line.strip()
            ]
        if rows:
            return rows, str(path)
    return [], ""


class ObservationIdentityTest(unittest.TestCase):
    """The identity carries every field that distinguishes two observations."""

    def test_identity_fields_are_all_part_of_the_completion_key(self) -> None:
        base = raw_attempt()
        base_key = eu.completion_record_key(base)
        for field, value in [
            ("provider_id", "prov_b"),
            ("profile_id", "prof_b"),
            ("model", "m2"),
            ("dataset_id", "mlm_tapt"),
            ("benchmark_variant", "shall"),
            ("task", "task1"),
            ("item_id", "S0002_mandatory"),
            ("sample_kind", "stochastic"),
            ("sample_index", 1),
        ]:
            with self.subTest(field=field):
                self.assertNotEqual(
                    eu.completion_record_key({**base, field: value}), base_key
                )

    def test_dataset_and_variant_aliases_resolve_to_one_identity(self) -> None:
        # Registry rows spell them `dataset_id`/`benchmark_variant`; score rows
        # and cell dicts spell them `dataset`/`variant`.
        canonical = eu.ObservationIdentity.from_raw_row(
            {**LEGACY_ROW, "dataset_id": "nice", "benchmark_variant": "shall"}
        )
        aliased = eu.ObservationIdentity.from_raw_row(
            {**LEGACY_ROW, "dataset": "nice", "variant": "shall"}
        )
        self.assertEqual(canonical, aliased)
        self.assertEqual(canonical.dataset_id, "nice")
        self.assertEqual(canonical.variant, "shall")

    def test_legacy_rows_fall_back_to_the_documented_sentinel(self) -> None:
        identity = eu.ObservationIdentity.from_raw_row(LEGACY_ROW)
        self.assertEqual(identity.provider_id, eu.LEGACY_IDENTITY)
        self.assertEqual(identity.profile_id, eu.LEGACY_IDENTITY)
        self.assertEqual(identity.dataset_id, eu.LEGACY_IDENTITY)
        self.assertEqual(identity.variant, eu.LEGACY_IDENTITY)
        # The fallback is explicit, never an empty string that could compare
        # equal to a row whose provider simply was not written yet.
        self.assertNotIn("", identity.record_key)

    def test_file_context_fills_only_the_fields_a_row_omits(self) -> None:
        inherited = eu.ObservationIdentity.from_raw_row(
            LEGACY_ROW, dataset_id="pure", variant="shall"
        )
        self.assertEqual(inherited.dataset_id, "pure")
        self.assertEqual(inherited.variant, "shall")
        recorded = eu.ObservationIdentity.from_raw_row(
            {**LEGACY_ROW, "dataset_id": "nice", "benchmark_variant": "must"},
            dataset_id="pure",
            variant="shall",
        )
        self.assertEqual(recorded.dataset_id, "nice")
        self.assertEqual(recorded.variant, "must")

    def test_planned_job_and_its_record_share_one_identity(self) -> None:
        row = raw_attempt()
        self.assertEqual(
            eu.completion_record_key(job_for(row)), eu.completion_record_key(row)
        )


class ResumeIdentityTest(unittest.TestCase):
    """Resume reuses a cached record only for the same experiment cell."""

    def test_another_provider_or_profile_is_not_already_complete(self) -> None:
        rows = [raw_attempt()]
        same = job_for(rows[0])
        self.assertEqual(eu.pending_completion_jobs([same], rows, "full-1"), [])
        for field, value in [("provider_id", "prov_b"), ("profile_id", "prof_b")]:
            with self.subTest(field=field):
                other = {**same, field: value}
                self.assertEqual(
                    eu.pending_completion_jobs([other], rows, "full-1"), [other]
                )

    def test_legacy_rows_still_resolve_as_complete(self) -> None:
        # A legacy plan (no provider/profile on the jobs either) matches its own
        # legacy rows, so an old raw file is not re-requested wholesale.
        rows = [{**LEGACY_ROW, "job_config_sha": "sha-1"}]
        job = job_for(rows[0])
        self.assertEqual(eu.pending_completion_jobs([job], rows, "full-legacy"), [])

    def test_this_checkouts_raw_rows_stay_complete_under_the_new_key(self) -> None:
        rows, source = sampled_real_raw_rows()
        if not rows:
            self.skipTest("no raw JSONL in this checkout; synthetic rows cover it")
        run_id = str(rows[0].get("run_id", ""))
        completed = [
            row
            for row in rows
            if str(row.get("run_id", "")) == run_id
            and str(row.get("parse_status", "")) == "ok"
        ]
        self.assertTrue(completed, source)
        jobs = [job_for(row) for row in completed]
        self.assertEqual(
            eu.pending_completion_jobs(jobs, completed, run_id), [], source
        )


class Task3SourceProfileTest(unittest.TestCase):
    """Task 3 never audits another profile's Task 2 output by accident."""

    @staticmethod
    def _rows(profile_id: str) -> list[dict[str, Any]]:
        return [
            {
                "run_id": "full-source",
                "model": "m1",
                "profile_id": profile_id,
                "task": "task2",
                "item_id": "S0001_mandatory",
                "sample_kind": "deterministic",
                "sample_index": 0,
                "parse_status": "ok",
            }
        ]

    def test_exact_profile_rows_are_selected(self) -> None:
        rows, profiles = task3_cli.source_rows_for_model(
            self._rows("prof_a") + self._rows("prof_b"), "m1", "prof_a"
        )
        self.assertEqual([row["profile_id"] for row in rows], ["prof_a"])
        self.assertEqual(profiles, ("prof_a",))

    def test_rows_without_a_profile_are_accepted_as_legacy(self) -> None:
        rows, profiles = task3_cli.source_rows_for_model(self._rows(""), "m1", "prof_a")
        self.assertEqual(len(rows), 1)
        self.assertEqual(profiles, (eu.LEGACY_IDENTITY,))

    def test_another_profiles_rows_are_refused(self) -> None:
        with self.assertRaises(task3_cli.Task3SourceProfileMismatchError) as caught:
            task3_cli.source_rows_for_model(self._rows("prof_b"), "m1", "prof_a")
        message = str(caught.exception)
        self.assertIn("prof_a", message)
        self.assertIn("prof_b", message)
        self.assertIn("--allow-source-profile-mismatch", message)

    def test_the_override_returns_the_profile_actually_audited(self) -> None:
        with self.assertLogs(eu.logger, level="WARNING") as logs:
            rows, profiles = task3_cli.source_rows_for_model(
                self._rows("prof_b"), "m1", "prof_a", allow_profile_mismatch=True
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(profiles, ("prof_b",))
        self.assertIn("prof_b", "\n".join(logs.output))

    def test_a_model_with_no_rows_at_all_is_reported_separately(self) -> None:
        rows, profiles = task3_cli.source_rows_for_model(
            self._rows("prof_a"), "m2", "prof_a"
        )
        self.assertEqual((rows, profiles), ([], ()))

    def test_resolve_source_rows_fails_instead_of_auditing_another_profile(
        self,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_path = eu.model_outputs_raw_path(
                root, "nice", "must", run_id="full-source"
            )
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(
                "\n".join(json.dumps(row) for row in self._rows("prof_b")) + "\n",
                encoding="utf-8",
            )
            matrix = SimpleNamespace(
                root=root,
                args=SimpleNamespace(
                    source_run_id="full-source",
                    mode="full",
                    allow_partial_source=True,
                    allow_source_profile_mismatch=False,
                ),
                expected_stochastic_samples=0,
            )
            with self.assertRaises(task3_cli.Task3SourceProfileMismatchError):
                task3_cli.resolve_source_rows(
                    matrix, {"profile_id": "prof_a"}, "m1", "nice", "must"
                )


def _score_row(
    *,
    model: str,
    dataset: str,
    variant: str,
    item_id: str,
    uq_method: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "model": model,
        "dataset_id": dataset,
        "benchmark_variant": variant,
        "item_id": item_id,
        "task": "task2",
        "uq_method": uq_method,
        **extra,
    }


class PaperJoinTest(unittest.TestCase):
    """Pooled paper joins keep one row per (model, dataset, variant, item).

    Every benchmark item id is reused between the `must` and `shall` variants of
    a family, so a (model, item) key silently collapses the four-cell matrix
    onto one cell.
    """

    CELLS = (("nice", "must"), ("nice", "shall"), ("pure", "must"), ("pure", "shall"))
    ITEM = "S0001_mandatory"

    def _cell_scores(self, dataset: str, variant: str) -> list[dict[str, Any]]:
        return [
            _score_row(
                model="m1",
                dataset=dataset,
                variant=variant,
                item_id=self.ITEM,
                uq_method="verbalized_confidence",
            ),
            _score_row(
                model="m1",
                dataset=dataset,
                variant=variant,
                item_id=self.ITEM,
                uq_method="modality_consistency",
                valid_n=5,
                total_n=5,
                stochastic_complete=True,
                uncertainty_score=0.0,
                label_distribution=eu.label_distribution_json({"mandatory": 1.0}),
            ),
        ]

    def test_overlapping_item_ids_do_not_collapse_the_four_cell_matrix(self) -> None:
        pooled_strict: list[dict[str, Any]] = []
        pooled_consistency: dict[tuple[str, ...], dict[str, Any]] = {}
        for dataset, variant in self.CELLS:
            scores = self._cell_scores(dataset, variant)
            pooled_strict.extend(export.task2_deterministic_rows(scores))
            pooled_consistency.update(
                export.stochastic_rows_by_method(scores, "modality_consistency")
            )

        self.assertEqual(len(pooled_strict), 4)
        self.assertEqual(len(pooled_consistency), 4)
        agreement, complete = export.agreement_for_strict_rows(
            pooled_strict, pooled_consistency
        )
        self.assertEqual(agreement["agreement_n_complete"], 4)
        self.assertEqual(agreement["agreement_n_incomplete_excluded"], 0)
        self.assertEqual(len(complete), 4)
        self.assertEqual(
            {(row["dataset_id"], row["benchmark_variant"]) for row in complete},
            set(self.CELLS),
        )

    def test_a_missing_stochastic_row_is_excluded_per_cell(self) -> None:
        # The `nice/must` strict row must not borrow the `nice/shall` stochastic
        # row just because both carry the same (model, item_id).
        strict = export.task2_deterministic_rows(self._cell_scores("nice", "must"))
        consistency = export.stochastic_rows_by_method(
            self._cell_scores("nice", "shall"), "modality_consistency"
        )
        agreement, complete = export.agreement_for_strict_rows(strict, consistency)
        self.assertEqual(agreement["agreement_n_complete"], 0)
        self.assertEqual(agreement["agreement_n_incomplete_excluded"], 1)
        self.assertEqual(complete, [])

    def test_scored_cells_are_stamped_with_their_dataset_and_variant(self) -> None:
        stamped = export.stamp_cell_identity(
            [{"model": "m1", "item_id": self.ITEM}], "nice", "shall"
        )
        self.assertEqual(
            export.paper_join_key(stamped[0]), ("m1", "nice", "shall", self.ITEM)
        )


class AttemptLedgerTest(unittest.TestCase):
    """Physical attempts and logical observations stay separable."""

    @staticmethod
    def _retried_rows() -> list[dict[str, Any]]:
        """One request that failed and was re-requested successfully on resume."""
        return [
            raw_attempt(parse_status="request_error", retry_count=1, latency_s=2.0),
            raw_attempt(parse_status="ok", retry_count=0, latency_s=1.0),
        ]

    def test_two_attempts_are_one_logical_observation(self) -> None:
        ledger = eu.AttemptLedger.from_raw_rows(self._retried_rows())
        self.assertEqual(len(ledger.all_attempts), 2)
        self.assertEqual(len(ledger.latest_logical_observations), 1)
        self.assertEqual(
            ledger.latest_logical_observations[0]["parse_status"], eu.PARSE_STATUS_OK
        )
        self.assertEqual(ledger.view(eu.ATTEMPT_VIEW_ALL), ledger.all_attempts)
        self.assertEqual(
            ledger.view(eu.ATTEMPT_VIEW_LATEST), ledger.latest_logical_observations
        )
        with self.assertRaises(ValueError):
            ledger.view("whatever")

    def test_task_progress_reports_one_of_one(self) -> None:
        benchmark = [{"item_id": "S0001_mandatory"}]
        progress = eu.run_progress_summary(
            benchmark, self._retried_rows(), expected_stochastic_samples=0
        )
        self.assertEqual(len(progress), 1)
        self.assertEqual(progress[0]["observed_records"], 1)
        self.assertEqual(progress[0]["expected_records"], 1)
        self.assertEqual(progress[0]["record_completion_rate"], 1.0)
        self.assertEqual(progress[0]["parse_success_rate"], 1.0)

    def test_progress_can_be_asked_for_the_attempt_view(self) -> None:
        benchmark = [{"item_id": "S0001_mandatory"}]
        progress = eu.run_progress_summary(
            benchmark,
            self._retried_rows(),
            expected_stochastic_samples=0,
            view=eu.ATTEMPT_VIEW_ALL,
        )
        self.assertEqual(progress[0]["observed_records"], 2)
        self.assertEqual(progress[0]["parse_success_rate"], 0.5)

    def test_registry_counts_observations_and_attempts_separately(self) -> None:
        row = eu.run_registry_summary(
            [{"item_id": "S0001_mandatory"}],
            self._retried_rows(),
            run_id="full-1",
            run_group_id="g1",
            provider_id="prov_a",
            profile_id="prof_a",
            model="m1",
            dataset_id="nice",
            variant="must",
            tasks=["task2"],
            expected_stochastic_samples=0,
            started_at_utc="2026-09-04T00:00:00Z",
        )
        self.assertEqual(row["expected_records"], 1)
        self.assertEqual(row["observed_records"], 1)
        self.assertEqual(row["observed_attempts"], 2)
        # The recovered run parses cleanly and is complete...
        self.assertEqual(row["parse_success_rate"], 1.0)
        self.assertEqual(row["status"], "complete")
        # ...while the physical request record keeps both attempts.
        self.assertEqual(row["retry_total"], 1)
        self.assertEqual(row["observed_api_calls"], 2)
        self.assertIn("request_error", row["parse_status_histogram"])
        self.assertEqual(set(row) - set(eu.RUN_REGISTRY_FIELDS), set())

    def test_live_counters_split_records_from_api_calls(self) -> None:
        counters = eu.live_run_counters(
            self._retried_rows(), expected_records=1, expected_api_calls=1
        )
        self.assertEqual(counters["observed_records"], 1)
        self.assertEqual(counters["observed_attempts"], 2)
        self.assertEqual(counters["ok_records"], 1)
        self.assertEqual(counters["parse_failure_records"], 0)
        self.assertEqual(counters["record_completion_rate"], 1.0)
        self.assertEqual(counters["observed_api_calls"], 2)
        self.assertEqual(counters["retry_total"], 1)

    def test_a_ledger_is_accepted_wherever_rows_are(self) -> None:
        ledger = eu.AttemptLedger.from_raw_rows(self._retried_rows())
        self.assertEqual(
            eu.live_run_counters(ledger, expected_records=1, expected_api_calls=1),
            eu.live_run_counters(
                self._retried_rows(), expected_records=1, expected_api_calls=1
            ),
        )


class ParseRepairReportingTest(unittest.TestCase):
    """Tolerant repairs (T7) are visible in run quality and the registry row."""

    @staticmethod
    def _repaired_rows() -> list[dict[str, Any]]:
        return [
            raw_attempt(
                item_id="S0001_mandatory",
                parsed_json={
                    "modality": "mandatory",
                    eu.PARSE_REPAIRS_FIELD: [eu.PARSE_REPAIR_PROSE_WRAPPER],
                },
            ),
            raw_attempt(
                item_id="S0002_mandatory", parsed_json={"modality": "optional"}
            ),
        ]

    def test_run_quality_counters_carry_the_repair_histogram(self) -> None:
        quality = eu.run_quality_counters(self._repaired_rows())
        self.assertEqual(quality["parse_repairs"]["repaired_records"], 1)
        self.assertEqual(quality["parse_repairs"][eu.PARSE_REPAIR_PROSE_WRAPPER], 1)

    def test_registry_row_and_live_counters_report_repairs(self) -> None:
        row = eu.run_registry_summary(
            [{"item_id": "S0001_mandatory"}, {"item_id": "S0002_mandatory"}],
            self._repaired_rows(),
            run_id="full-1",
            run_group_id="g1",
            provider_id="prov_a",
            profile_id="prof_a",
            model="m1",
            dataset_id="nice",
            variant="must",
            tasks=["task2"],
            expected_stochastic_samples=0,
            started_at_utc="2026-09-04T00:00:00Z",
        )
        self.assertIn(eu.PARSE_REPAIR_PROSE_WRAPPER, row["parse_repairs"])
        self.assertIn('"repaired_records":1', row["parse_repairs"])
        self.assertEqual(set(row) - set(eu.RUN_REGISTRY_FIELDS), set())

        counters = eu.live_run_counters(
            self._repaired_rows(), expected_records=2, expected_api_calls=2
        )
        self.assertEqual(counters["parse_repairs"]["repaired_records"], 1)

    def test_a_clean_run_reports_no_repairs(self) -> None:
        row = eu.run_registry_summary(
            [{"item_id": "S0001_mandatory"}],
            [raw_attempt(parsed_json={"modality": "mandatory"})],
            run_id="full-1",
            run_group_id="g1",
            provider_id="prov_a",
            profile_id="prof_a",
            model="m1",
            dataset_id="nice",
            variant="must",
            tasks=["task2"],
            expected_stochastic_samples=0,
            started_at_utc="2026-09-04T00:00:00Z",
        )
        self.assertEqual(row["parse_repairs"], "")


if __name__ == "__main__":
    unittest.main()
