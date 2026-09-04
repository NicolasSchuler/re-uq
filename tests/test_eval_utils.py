import io
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from pydantic import ValidationError

try:
    from helpers import FakeResponse, raw_record
except ModuleNotFoundError:  # pragma: no cover - invocation-path fallback
    from tests.helpers import FakeResponse, raw_record

from scripts import (
    compare_run_matrix as compare_matrix,
    eval_utils as eu,
    evaluate_external_ai_probe as external_eval,
    generate_evaluation_analysis as analysis_cli,
    plot_acse_global_embedding_projection as acse_global_projection,
    run_experiment_from_config as run_config_cli,
    run_task3_verification_from_config as task3_cli,
    show_run_progress,
    structured_outputs as so,
)


def export_report_seeds():
    """Return the canonical single ``export reports`` seed row as a fresh list."""
    return [
        {
            "seed_id": "S0001",
            "source_dataset": "NICE",
            "original_requirement": "The system shall export reports.",
            "capability_text_final": "export reports",
        }
    ]


def _scaffold_project_root(root, *, with_prompts=True):
    """Create the standard temp-project layout used by CLI integration tests."""
    if with_prompts:
        (root / "prompts").mkdir(parents=True)
    (root / "data/processed").mkdir(parents=True)
    (root / "AGENTS.md").write_text("", encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs/evaluation.md").write_text("", encoding="utf-8")


class EvalUtilsTest(unittest.TestCase):
    _INSTRUCTOR_EXTRA_BODY_DEFAULT = object()

    def _instructor_task1_jobs(
        self,
        seed_count=1,
        *,
        extra_body=_INSTRUCTOR_EXTRA_BODY_DEFAULT,
        validation_retries=3,
    ):
        if extra_body is self._INSTRUCTOR_EXTRA_BODY_DEFAULT:
            extra_body = {
                "thinking": {"type": "disabled"},
                "response_format": {"type": "json_object"},
            }
        seeds = [
            {
                "seed_id": f"S{index + 1:04d}",
                "source_dataset": "NICE",
                "original_requirement": f"The system shall export report {index + 1}.",
                "capability_text_final": f"export report {index + 1}",
            }
            for index in range(seed_count)
        ]
        benchmark = eu.build_benchmark_items(seeds)
        mandatory_items = [
            row for row in benchmark if row["source_modality"] == "mandatory"
        ]
        return eu.planned_completion_jobs(
            mandatory_items,
            tasks=["task1"],
            model="m1",
            host="http://localhost:1234/v1",
            run_id="full-1",
            prompt_version="v2-instructor-conf01",
            task1_template=eu.load_prompt("prompts/mandatory_entailment.txt"),
            task2_template=eu.load_prompt("prompts/modality_extraction.txt"),
            deterministic={"temperature": 0.0, "top_p": 1.0, "samples": 1},
            stochastic={"temperature": 0.7, "top_p": 1.0, "samples": 0},
            max_tokens=64,
            timeout_s=30,
            api_key_env="LOCAL_OPENAI_API_KEY",
            structured_output="instructor",
            extra_body=extra_body,
            validation_retries=validation_retries,
        )

    def _task1_two_seed_jobs(self, **overrides):
        """Two mandatory Task 1 jobs (one per seed) on the plain (non-Instructor) path."""
        seeds = [
            {
                "seed_id": "S0001",
                "source_dataset": "NICE",
                "original_requirement": "The system shall export reports.",
                "capability_text_final": "export reports",
            },
            {
                "seed_id": "S0002",
                "source_dataset": "NICE",
                "original_requirement": "The system shall print invoices.",
                "capability_text_final": "print invoices",
            },
        ]
        benchmark = eu.build_benchmark_items(seeds)
        kwargs = {
            "tasks": ["task1"],
            "model": "m1",
            "host": "http://localhost:1234/v1",
            "run_id": "full-1",
            "prompt_version": "v1",
            "task1_template": eu.load_prompt("prompts/mandatory_entailment.txt"),
            "task2_template": eu.load_prompt("prompts/modality_extraction.txt"),
            "deterministic": {"temperature": 0.0, "top_p": 1.0, "samples": 1},
            "stochastic": {"temperature": 0.7, "top_p": 1.0, "samples": 0},
            "max_tokens": 64,
            "timeout_s": 30,
            "api_key_env": "LOCAL_OPENAI_API_KEY",
        }
        kwargs.update(overrides)
        return eu.planned_completion_jobs(
            [row for row in benchmark if row["source_modality"] == "mandatory"],
            **kwargs,
        )

    def test_benchmark_labels(self):
        seeds = export_report_seeds()
        items = eu.build_benchmark_items(seeds)
        self.assertEqual(len(items), 4)
        by_modality = {row["source_modality"]: row for row in items}
        self.assertEqual(by_modality["mandatory"]["task1_gold_decision"], "yes")
        self.assertEqual(by_modality["recommended"]["task1_gold_decision"], "no")
        self.assertEqual(by_modality["optional"]["task2_gold_modality"], "optional")
        self.assertGreater(
            by_modality["mandatory"]["ordinal_strength"],
            by_modality["nice_to_have"]["ordinal_strength"],
        )

    def test_benchmark_statement_review_export(self):
        seeds = export_report_seeds()
        benchmark = eu.build_benchmark_items(seeds)
        with tempfile.TemporaryDirectory() as tmpdir:
            export_paths = eu.write_benchmark_statement_review(benchmark, tmpdir)
            frame = eu.benchmark_statement_review_frame(benchmark)

            self.assertEqual(len(frame), 1)
            self.assertEqual(
                frame.iloc[0]["MUST source"], "The system MUST export reports."
            )
            self.assertEqual(
                frame.iloc[0]["Nice-to-have source"],
                "It would be useful if the system could export reports.",
            )
            self.assertTrue(export_paths["markdown"].exists())
            self.assertTrue(export_paths["csv"].exists())

    def test_shall_benchmark_uses_shall_but_keeps_labels(self):
        seeds = export_report_seeds()
        items = eu.build_benchmark_items(seeds, mandatory_keyword="SHALL")
        self.assertEqual(len(items), 4)
        self.assertEqual(len({row["item_id"] for row in items}), 4)
        by_modality = {row["source_modality"]: row for row in items}
        self.assertEqual(
            by_modality["mandatory"]["source_statement"],
            "The system SHALL export reports.",
        )
        self.assertEqual(
            by_modality["mandatory"]["candidate_requirement"],
            "The system SHALL export reports.",
        )
        self.assertEqual(by_modality["mandatory"]["task1_gold_decision"], "yes")
        self.assertEqual(by_modality["recommended"]["task1_gold_decision"], "no")

    def test_build_benchmark_items_passthrough_fields_default_to_none(self):
        seeds = [{**export_report_seeds()[0], "context_marker": "O", "extra": "x"}]
        plain = eu.build_benchmark_items(seeds)
        carried = eu.build_benchmark_items(
            seeds, passthrough_fields=["context_marker", "context_section"]
        )

        self.assertNotIn("context_marker", plain[0])
        self.assertNotIn("extra", plain[0])
        self.assertEqual(carried[0]["context_marker"], "O")
        # A named field the seed lacks becomes an empty column, not a KeyError.
        self.assertEqual(carried[0]["context_section"], "")
        self.assertEqual(
            set(carried[0]) - set(plain[0]), {"context_marker", "context_section"}
        )
        self.assertEqual(
            [row["source_statement"] for row in plain],
            [row["source_statement"] for row in carried],
        )

    def test_weak_modality_probe_generation(self):
        seeds = [
            {
                "seed_id": f"S{i:04d}",
                "source_dataset": "NICE",
                "original_requirement": "The system shall export reports.",
                "capability_text_final": f"export reports {i}",
            }
            for i in range(1, 21)
        ]

        items = eu.build_weak_modality_probe_items(seeds)

        self.assertEqual(len(items), 80)
        self.assertEqual(len({row["item_id"] for row in items}), 80)
        self.assertEqual(
            {row["task2_gold_modality"] for row in items}, {"nice_to_have"}
        )
        self.assertEqual({row["source_modality"] for row in items}, {"nice_to_have"})
        self.assertEqual(
            {row["template_id"] for row in items},
            {row["template_id"] for row in eu.WEAK_MODALITY_PROBE_TEMPLATES},
        )
        for row in items:
            text = row["source_statement"].lower()
            self.assertNotIn("nice_to_have", text)
            self.assertNotIn("nice-to-have", text)
            self.assertTrue(text.endswith("."))

    def test_main_modality_template_inventory_covers_every_condition(self):
        rows = eu.main_modality_template_rows()
        self.assertEqual(set(rows[0]), set(eu.MAIN_MODALITY_TEMPLATE_INVENTORY_FIELDS))
        self.assertEqual(len(rows), 5 + len(eu.WEAK_MODALITY_PROBE_TEMPLATES))
        self.assertEqual(len({row["template_id"] for row in rows}), len(rows))

        by_variant = {}
        for row in rows:
            by_variant.setdefault(row["variant"], []).append(row)
        self.assertEqual(set(by_variant), {"must", "shall", "weak_probe"})
        self.assertEqual(
            [row["condition"] for row in by_variant["must"]],
            eu.MODALITIES,
        )
        self.assertEqual(len(by_variant["shall"]), 1)
        self.assertEqual(
            len(by_variant["weak_probe"]), len(eu.WEAK_MODALITY_PROBE_TEMPLATES)
        )

        for row in rows:
            self.assertIn("{capability}", row["source_statement_template"])
            self.assertNotIn("{capability}", row["example_source_statement"])
            self.assertEqual(
                row["example_source_statement"],
                row["source_statement_template"].format(capability="export reports"),
            )
            self.assertIn(row["intended_gold_modality"], eu.MODALITIES)

        must_templates = {
            row["condition"]: row["source_statement_template"]
            for row in by_variant["must"]
        }
        self.assertEqual(must_templates["mandatory"], "The system MUST {capability}.")
        self.assertEqual(
            must_templates["recommended"], "The system SHOULD {capability}."
        )
        self.assertEqual(must_templates["optional"], "The system MAY {capability}.")
        self.assertEqual(
            must_templates["nice_to_have"],
            "It would be useful if the system could {capability}.",
        )
        self.assertEqual(
            by_variant["shall"][0]["source_statement_template"],
            "The system SHALL {capability}.",
        )
        self.assertEqual(
            {row["source_statement_template"] for row in by_variant["weak_probe"]},
            {
                template["source_template"]
                for template in eu.WEAK_MODALITY_PROBE_TEMPLATES
            },
        )

    def test_write_main_modality_template_inventory_writes_csv_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "modality_template_inventory.csv"
            paths = eu.write_main_modality_template_inventory(csv_path)
            self.assertEqual(paths["csv"], csv_path)
            self.assertEqual(paths["markdown"], csv_path.with_suffix(".md"))

            written = eu.read_csv_rows(paths["csv"])
            self.assertEqual(len(written), len(eu.main_modality_template_rows()))
            self.assertEqual(
                list(written[0]), eu.MAIN_MODALITY_TEMPLATE_INVENTORY_FIELDS
            )

            markdown = paths["markdown"].read_text(encoding="utf-8")
            self.assertIn("The system SHALL {capability}.", markdown)
            self.assertIn("probe_future_enhancement", markdown)

    def test_weak_modality_sanity_validation(self):
        rows = eu.weak_modality_template_sanity_rows()
        incomplete = eu.weak_modality_sanity_status(rows)
        self.assertFalse(incomplete["valid"])
        self.assertEqual(
            set(incomplete["incomplete_template_ids"]),
            {row["template_id"] for row in eu.WEAK_MODALITY_PROBE_TEMPLATES},
        )

        agreed = [
            {**row, "weaker_than_should": "yes", "reviewer": "r1"} for row in rows
        ]
        self.assertTrue(eu.weak_modality_sanity_status(agreed)["valid"])

        disagreed = list(agreed)
        disagreed[0] = {**disagreed[0], "weaker_than_should": "no"}
        status = eu.weak_modality_sanity_status(disagreed)
        self.assertFalse(status["valid"])
        self.assertEqual(
            status["disagreeing_template_ids"], [disagreed[0]["template_id"]]
        )

    def test_weak_modality_construct_review_requires_two_agreeing_reviewers(self):
        rows = eu.weak_modality_construct_review_rows()
        self.assertEqual(len(rows), 8)
        self.assertEqual(set(rows[0]), set(eu.WEAK_MODALITY_CONSTRUCT_REVIEW_FIELDS))

        incomplete = eu.weak_modality_construct_review_status(rows)
        self.assertFalse(incomplete["valid"])
        self.assertEqual(
            set(incomplete["incomplete_template_ids"]),
            {row["template_id"] for row in eu.WEAK_MODALITY_PROBE_TEMPLATES},
        )

        agreed = [{**row, "weaker_than_should": "yes"} for row in rows]
        self.assertTrue(eu.weak_modality_construct_review_status(agreed)["valid"])

        one_reviewer_only = [row for row in agreed if row["reviewer_id"] == "R1"]
        one_status = eu.weak_modality_construct_review_status(one_reviewer_only)
        self.assertFalse(one_status["valid"])
        self.assertEqual(
            set(one_status["insufficient_template_ids"]),
            {row["template_id"] for row in eu.WEAK_MODALITY_PROBE_TEMPLATES},
        )

        disagreed = list(agreed)
        disagreed[0] = {**disagreed[0], "weaker_than_should": "no"}
        status = eu.weak_modality_construct_review_status(disagreed)
        self.assertFalse(status["valid"])
        self.assertEqual(
            status["disagreeing_template_ids"], [disagreed[0]["template_id"]]
        )

    def test_construct_review_gate_accepts_shipped_csv(self):
        shipped = eu.project_root() / "docs/weak_modality_construct_review.csv"
        status = eu.weak_modality_construct_review_status(eu.read_csv_rows(shipped))
        self.assertTrue(status["valid"])
        self.assertEqual(status["missing_template_ids"], [])
        self.assertEqual(status["incomplete_template_ids"], [])
        self.assertEqual(status["disagreeing_template_ids"], [])
        # Should not raise now that both reviewer slots are filled.
        analysis_cli.require_construct_review_complete(shipped)

    def test_construct_review_gate_reports_incomplete_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            blank_path = Path(tmp) / "weak_modality_construct_review.csv"
            blank_rows = [
                {**row, "weaker_than_should": ""}
                for row in eu.weak_modality_construct_review_rows()
            ]
            eu.write_csv_rows(
                blank_path,
                blank_rows,
                fieldnames=eu.WEAK_MODALITY_CONSTRUCT_REVIEW_FIELDS,
            )
            status = eu.weak_modality_construct_review_status(
                eu.read_csv_rows(blank_path)
            )
            self.assertFalse(status["valid"])
            # Both reviewer slots present but no judgments filled.
            self.assertEqual(status["missing_template_ids"], [])
            self.assertEqual(
                set(status["incomplete_template_ids"]),
                {row["template_id"] for row in eu.WEAK_MODALITY_PROBE_TEMPLATES},
            )
            with self.assertRaises(ValueError) as ctx:
                analysis_cli.require_construct_review_complete(blank_path)
            message = str(ctx.exception)
            self.assertIn("--skip-construct-review-check", message)
            self.assertIn("R1", message)
            self.assertIn("R2", message)
            self.assertIn(str(blank_path), message)

    def test_construct_review_gate_accepts_filled_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            filled_path = Path(tmp) / "weak_modality_construct_review.csv"
            filled_rows = [
                {**row, "weaker_than_should": "yes"}
                for row in eu.weak_modality_construct_review_rows()
            ]
            eu.write_csv_rows(
                filled_path,
                filled_rows,
                fieldnames=eu.WEAK_MODALITY_CONSTRUCT_REVIEW_FIELDS,
            )
            # Should not raise once both author slots mark every template "yes".
            analysis_cli.require_construct_review_complete(filled_path)

    def test_requirement_text_modality_parser(self):
        cases = [
            ("The system SHALL export reports.", "mandatory"),
            ("The system SHOULD export reports.", "recommended"),
            ("The system MAY export reports.", "optional"),
            ("The system could export reports.", "optional"),
            ("The system can export reports.", "optional"),
            ("It would be nice if the system could export reports.", "nice_to_have"),
            ("The system exports reports.", "mandatory"),
            ("System provides export reports.", "mandatory"),
        ]

        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(
                    eu.requirement_text_modality_diagnostic(text)["text_modality"],
                    expected,
                )

        self.assertEqual(
            eu.requirement_text_modality_diagnostic(
                "The system SHOULD export reports."
            )["text_modality_basis"],
            "explicit_modal",
        )
        self.assertEqual(
            eu.requirement_text_modality_diagnostic(
                "It would be nice if the system could export reports."
            )["text_modality_basis"],
            "weak_phrase",
        )
        self.assertEqual(
            eu.requirement_text_modality_diagnostic("The system exports reports.")[
                "text_modality_basis"
            ],
            "heuristic_system_verb",
        )

    def test_positive_modal_cue_wins_over_a_negated_one(self):
        """A prohibition inside a mandatory obligation is still mandatory."""
        cases = [
            # Positive cue present -> it wins, priority mandatory > recommended > optional.
            (
                "The system must ensure that users cannot delete records.",
                "mandatory",
                "explicit_modal",
            ),
            (
                "Not only must the system export reports, it must archive them.",
                "mandatory",
                "explicit_modal",
            ),
            (
                "Users may not delete records; the system must log attempts.",
                "mandatory",
                "explicit_modal",
            ),
            # No positive cue -> the negated modal stands.
            ("The system must not export reports.", "negated", "negated_modal"),
            ("The system cannot export reports.", "negated", "negated_modal"),
        ]

        for text, expected_modality, expected_basis in cases:
            with self.subTest(text=text):
                diagnostic = eu.requirement_text_modality_diagnostic(text)
                self.assertEqual(diagnostic["text_modality"], expected_modality)
                self.assertEqual(diagnostic["text_modality_basis"], expected_basis)

        # Mixed categories are flagged; a same-category negation is not.
        self.assertTrue(
            eu.requirement_text_modality_diagnostic(
                "Users may not delete records; the system must log attempts."
            )["text_modality_multi_modal"]
        )
        self.assertFalse(
            eu.requirement_text_modality_diagnostic(
                "Not only must the system export reports, it must archive them."
            )["text_modality_multi_modal"]
        )
        # A weak phrase still outranks every modal cue.
        self.assertEqual(
            eu.requirement_text_modality_diagnostic(
                "It would be nice if the system must export reports."
            )["text_modality"],
            "nice_to_have",
        )

    def test_text_modality_overcommitment_fields(self):
        optional_to_shall = eu.text_modality_fields(
            "The system shall export reports.", "optional", "optional", 0.98
        )
        nice_to_could = eu.text_modality_fields(
            "The system could export reports.", "nice_to_have", "nice_to_have", 0.94
        )
        weak_phrase = eu.text_modality_fields(
            "It would be nice if the system could export reports.",
            "nice_to_have",
            "nice_to_have",
            0.94,
        )

        self.assertTrue(optional_to_shall["text_overcommit"])
        self.assertTrue(nice_to_could["text_overcommit"])
        self.assertTrue(nice_to_could["strict_text_overcommit"])
        self.assertFalse(weak_phrase["text_overcommit"])
        self.assertEqual(weak_phrase["text_modality"], "nice_to_have")

    def test_manifest_hash_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "rows.csv"
            prompt_path = root / "prompt.txt"
            manifest_path = root / "manifest.json"
            eu.write_csv_rows(csv_path, [{"id": "1"}, {"id": "2"}])
            prompt_path.write_text("prompt\n", encoding="utf-8")

            manifest = eu.write_benchmark_manifest(
                [csv_path, prompt_path],
                manifest_path,
                root=root,
                metadata={"kind": "test"},
            )
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest["metadata"]["kind"], "test")
            self.assertEqual(loaded["artifacts"][0]["rows"], 2)
            self.assertEqual(loaded["artifacts"][0]["sha256"], eu.sha256_file(csv_path))
            self.assertEqual(loaded["artifacts"][1]["rows"], "")

    def test_verify_benchmark_manifest_passes_on_real_manifest(self):
        root = eu.project_root()
        summary = eu.verify_benchmark_manifest(
            root / "outputs/benchmark_manifest.json", root
        )
        self.assertGreater(summary["checked"], 0)
        self.assertEqual(summary["missing"], [])

    def test_verify_benchmark_manifest_fails_on_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_path = root / "data.csv"
            data_path.write_text("real\n", encoding="utf-8")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps({"artifacts": [{"path": "data.csv", "sha256": "0" * 64}]}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                eu.verify_benchmark_manifest(manifest_path, root)

    def test_verify_benchmark_manifest_missing_required_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "artifacts": [
                            {
                                "path": "prompts/mandatory_entailment.txt",
                                "sha256": "0" * 64,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                eu.verify_benchmark_manifest(manifest_path, root)

    def test_verify_benchmark_manifest_missing_untracked_is_non_fatal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "artifacts": [
                            {
                                "path": "data/processed/seeds_review.csv",
                                "sha256": "0" * 64,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            summary = eu.verify_benchmark_manifest(manifest_path, root)
            self.assertEqual(summary["checked"], 0)
            self.assertEqual(summary["missing"], ["data/processed/seeds_review.csv"])

    def test_rule_based_parser_maps_modalities(self):
        cases = {
            "The system MUST export reports.": "mandatory",
            "The system SHALL export reports.": "mandatory",
            "The system SHOULD export reports.": "recommended",
            "The system MAY export reports.": "optional",
            "It would be useful if the system could export reports.": "nice_to_have",
        }
        for text, expected in cases.items():
            self.assertEqual(eu.rule_based_source_modality(text), expected)

    def test_rule_baseline_scores_are_perfect_on_controlled_benchmark(self):
        benchmark = eu.build_benchmark_items(export_report_seeds())
        scores = eu.build_rule_baseline_scores(benchmark)
        summary = eu.metric_summary_by_model_task_method(scores)

        self.assertEqual(len(scores), 8)
        self.assertEqual({row["model"] for row in scores}, {eu.RULE_BASELINE_MODEL})
        self.assertTrue(all(row["accuracy"] == 1.0 for row in summary))
        self.assertTrue(
            all(row["uq_method"] == eu.RULE_BASELINE_METHOD for row in summary)
        )

    def test_parse_task1_response(self):
        raw = 'Some preface {"decision": "yes", "confidence": 87, "brief_reason": "MUST matches"}'
        parsed, status = eu.parse_task_response("task1", raw)
        self.assertEqual(status, "ok")
        self.assertEqual(parsed["decision"], "yes")
        self.assertEqual(parsed["confidence"], 87.0)

    def test_instructor_response_models_enforce_confidence_probability(self):
        parsed = so.Task2Response.model_validate(
            {
                "requirement": "The system MAY export reports.",
                "modality": "optional",
                "confidence": 0.95,
            }
        )

        self.assertEqual(parsed.confidence, 0.95)
        for bad_confidence in ["0.95", -0.1, 1.1, 95]:
            with (
                self.subTest(confidence=bad_confidence),
                self.assertRaises(ValidationError),
            ):
                so.Task2Response.model_validate(
                    {
                        "requirement": "The system MAY export reports.",
                        "modality": "optional",
                        "confidence": bad_confidence,
                    }
                )

    def test_confidence_probability_handles_legacy_and_instructor_rows(self):
        legacy = {"parsed_json": {"confidence": 90.0}}
        prompt_v2_row = {
            "prompt_version": "v2-conf01",
            "parsed_json": {"confidence": 0.9},
        }
        prompt_contract_row = {
            "parsed_json": {"confidence": 0.85},
            "output_contract_version": so.PROMPT_OUTPUT_CONTRACT_VERSION,
        }
        instructor_row = {
            "parsed_json": {"confidence": 0.9},
            "output_contract_version": so.INSTRUCTOR_OUTPUT_CONTRACT_VERSION,
            "confidence_scale": so.INSTRUCTOR_CONFIDENCE_SCALE,
        }
        instructor_contract_only = {
            "parsed_json": {"confidence": 0.8},
            "output_contract_version": so.INSTRUCTOR_OUTPUT_CONTRACT_VERSION,
        }
        instructor_scale_only = {
            "parsed_json": {"confidence": 0.7},
            "confidence_scale": so.INSTRUCTOR_CONFIDENCE_SCALE,
        }

        self.assertEqual(eu.confidence_probability(legacy), 0.9)
        self.assertEqual(eu.confidence_probability(prompt_v2_row), 0.9)
        self.assertEqual(eu.confidence_probability(prompt_contract_row), 0.85)
        self.assertEqual(eu.confidence_probability(instructor_row), 0.9)
        self.assertEqual(eu.confidence_probability(instructor_contract_only), 0.8)
        self.assertEqual(eu.confidence_probability(instructor_scale_only), 0.7)

    def test_v2_raw_records_mark_probability_confidence_for_non_instructor_runs(self):
        item = eu.build_benchmark_items(export_report_seeds())[0]

        record = eu.build_raw_record(
            run_id="full-1",
            model="m1",
            host="http://localhost:8000/v1",
            task="task1",
            item=item,
            sample_index=0,
            sample_kind="deterministic",
            temperature=0.0,
            top_p=1.0,
            prompt_version="v2-conf01",
            prompt="prompt",
            completion={
                "ok": True,
                "raw_text": '{"decision": "yes", "confidence": 0.9, "brief_reason": "ok"}',
                "latency_s": 0.01,
                "error": "",
            },
            structured_output="json_schema",
        )

        self.assertEqual(record["parse_status"], "ok")
        self.assertEqual(record["confidence_scale"], eu.CONFIDENCE_SCALE_0_1)
        self.assertEqual(
            record["output_contract_version"], so.PROMPT_OUTPUT_CONTRACT_VERSION
        )
        self.assertEqual(eu.confidence_probability(record), 0.9)

    def test_raw_records_infer_probability_scale_from_prompt_contract(self):
        item = eu.build_benchmark_items(export_report_seeds())[0]

        record = eu.build_raw_record(
            run_id="full-1",
            model="m1",
            host="http://localhost:8000/v1",
            task="task1",
            item=item,
            sample_index=0,
            sample_kind="deterministic",
            temperature=0.0,
            top_p=1.0,
            prompt_version="v1",
            prompt='Return JSON only: {"confidence": 0.0-1.0}\nUse confidence as a decimal probability.',
            completion={
                "ok": True,
                "raw_text": '{"decision": "yes", "confidence": 0.9, "brief_reason": "ok"}',
                "latency_s": 0.01,
                "error": "",
            },
        )

        self.assertEqual(record["parse_status"], "ok")
        self.assertEqual(record["confidence_scale"], eu.CONFIDENCE_SCALE_0_1)
        self.assertEqual(
            record["output_contract_version"], so.PROMPT_OUTPUT_CONTRACT_VERSION
        )
        self.assertEqual(eu.confidence_probability(record), 0.9)

    def test_v2_raw_records_reject_percentage_confidence_without_legacy_marker(self):
        item = eu.build_benchmark_items(export_report_seeds())[0]

        record = eu.build_raw_record(
            run_id="full-1",
            model="m1",
            host="http://localhost:8000/v1",
            task="task1",
            item=item,
            sample_index=0,
            sample_kind="deterministic",
            temperature=0.0,
            top_p=1.0,
            prompt_version="v2-conf01",
            prompt="prompt",
            completion={
                "ok": True,
                "raw_text": '{"decision": "yes", "confidence": 95, "brief_reason": "bad scale"}',
                "latency_s": 0.01,
                "error": "",
            },
        )

        self.assertEqual(record["parse_status"], "invalid_confidence")
        self.assertIsNotNone(record["parsed_json"])

    def test_task1_selected_label_confidence_sets_p_yes_for_yes_and_no(self):
        benchmark = eu.build_benchmark_items(export_report_seeds())
        item_by_modality = {row["source_modality"]: row for row in benchmark}
        raw_rows = [
            {
                "run_id": "full-1",
                "model": "m1",
                "task": "task1",
                "item_id": item_by_modality["mandatory"]["item_id"],
                "sample_kind": "deterministic",
                "sample_index": 0,
                "parse_status": "ok",
                "parsed_json": {
                    "decision": "yes",
                    "confidence": 0.8,
                    "brief_reason": "mandatory",
                },
                "output_contract_version": so.INSTRUCTOR_OUTPUT_CONTRACT_VERSION,
                "confidence_scale": so.INSTRUCTOR_CONFIDENCE_SCALE,
            },
            {
                "run_id": "full-1",
                "model": "m1",
                "task": "task1",
                "item_id": item_by_modality["optional"]["item_id"],
                "sample_kind": "deterministic",
                "sample_index": 0,
                "parse_status": "ok",
                "parsed_json": {
                    "decision": "no",
                    "confidence": 0.7,
                    "brief_reason": "optional",
                },
                "output_contract_version": so.INSTRUCTOR_OUTPUT_CONTRACT_VERSION,
                "confidence_scale": so.INSTRUCTOR_CONFIDENCE_SCALE,
            },
        ]

        scores = eu.build_uq_scores(benchmark, raw_rows)
        score_by_item = {row["item_id"]: row for row in scores}

        self.assertEqual(
            score_by_item[item_by_modality["mandatory"]["item_id"]]["confidence"], 0.8
        )
        self.assertEqual(
            score_by_item[item_by_modality["mandatory"]["item_id"]]["p_yes"], 0.8
        )
        self.assertEqual(
            score_by_item[item_by_modality["optional"]["item_id"]]["confidence"], 0.7
        )
        self.assertAlmostEqual(
            score_by_item[item_by_modality["optional"]["item_id"]]["p_yes"], 0.3
        )

    def test_auto_capability_text_strips_quotes_and_boilerplate(self):
        self.assertEqual(
            eu.auto_capability_text(
                "'The system shall refresh the display every 60 seconds.'"
            ),
            "refresh the display every 60 seconds",
        )
        self.assertEqual(
            eu.auto_capability_text('"The application must export reports as CSV."'),
            "export reports as CSV",
        )
        self.assertEqual(
            eu.auto_capability_text(
                "'The system shall interface with CampusConnect's central server.'"
            ),
            "interface with CampusConnect's central server",
        )
        self.assertEqual(
            eu.auto_capability_text(
                "The TMT Observatory shall monitor subsystem performance."
            ),
            "monitor subsystem performance",
        )
        self.assertEqual(
            eu.auto_capability_text(
                "The system shall be able to display a summary which will include cohort progress."
            ),
            "display a summary which will include cohort progress",
        )

    def test_refresh_capability_suggestions_preserves_manual_edits(self):
        rows = [
            {
                "original_requirement": "'The system shall refresh the display every 60 seconds.'",
                "capability_text_auto": "'The system shall refresh the display every 60 seconds.'",
                "include": "yes",
                "capability_text_final": "'The system shall refresh the display every 60 seconds.'",
            },
            {
                "original_requirement": "'The system shall display Events or Activities.'",
                "capability_text_auto": "'The system shall display Events or Activities.'",
                "include": "yes",
                "capability_text_final": "show events and activities",
            },
        ]
        refreshed, updated = eu.refresh_capability_suggestions(rows)
        self.assertEqual(updated, 1)
        self.assertEqual(
            refreshed[0]["capability_text_final"],
            "refresh the display every 60 seconds",
        )
        self.assertEqual(
            refreshed[1]["capability_text_final"], "show events and activities"
        )

    def test_write_included_capability_review(self):
        rows = [
            {
                "seed_id": "S0001",
                "original_requirement": "'The system shall refresh the display every 60 seconds.'",
                "capability_text_final": "refresh the display every 60 seconds.",
                "include": "yes",
            },
            {
                "seed_id": "S0002",
                "original_requirement": "'The product shall reject invalid input.'",
                "capability_text_final": "reject invalid input",
                "include": "no",
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            review_path = Path(tmpdir) / "seeds_review.csv"
            eu.write_csv_rows(review_path, rows)
            export_paths = eu.write_included_capability_review(
                review_path, Path(tmpdir) / "outputs"
            )
            frame = eu.included_capability_review_frame(review_path)

            self.assertEqual(len(frame), 1)
            self.assertEqual(
                frame.iloc[0]["Final capability text"],
                "refresh the display every 60 seconds",
            )
            self.assertTrue(export_paths["markdown"].exists())
            self.assertTrue(export_paths["csv"].exists())

    def test_write_csv_rows_if_changed_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.csv"
            candidate_path = Path(tmpdir) / "rows_candidate.csv"
            eu.write_csv_rows(path, [{"id": "1", "value": "reviewed"}])

            result = eu.write_csv_rows_if_changed(
                path,
                [{"id": "1", "value": "regenerated"}],
                candidate_path=candidate_path,
            )

            self.assertEqual(result["status"], "candidate_written")
            self.assertEqual(eu.read_csv_rows(path)[0]["value"], "reviewed")
            self.assertEqual(
                eu.read_csv_rows(candidate_path)[0]["value"], "regenerated"
            )

    def test_write_csv_rows_if_changed_reports_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.csv"
            rows = [{"id": "1", "value": "reviewed"}]
            eu.write_csv_rows(path, rows)

            result = eu.write_csv_rows_if_changed(path, rows)

            self.assertEqual(result["status"], "unchanged")
            self.assertFalse((Path(tmpdir) / "rows_candidate.csv").exists())

    def test_automatic_filter_rejects_internal_sentence_boundary(self):
        original = "'The product shall ensure that it can only be accessed by authorized users. The product will be able to distinguish between authorized and unauthorized users in all access attempts'"
        capability = eu.auto_capability_text(original)
        include, reason = eu.automatic_filter(original, capability)
        self.assertFalse(include)
        self.assertIn("multi_sentence", reason)

    def test_automatic_filter_rejects_stranded_preposition_capability(self):
        include, reason = eu.automatic_filter(
            "The system shall interface with CampusConnect's central server.",
            "with CampusConnect's central server",
        )
        self.assertFalse(include)
        self.assertIn("stranded_preposition", reason)

    def test_automatic_filter_rejects_residual_modal_capability(self):
        include, reason = eu.automatic_filter(
            "The system shall display a summary which will include cohort progress.",
            "display a summary which will include cohort progress",
        )
        self.assertFalse(include)
        self.assertIn("residual_modal_in_capability", reason)
        include, reason = eu.automatic_filter(
            "There shall be possibility to power the equipment set from the vehicle alternator.",
            "be possibility to power the equipment set from the vehicle alternator",
        )
        self.assertFalse(include)
        self.assertIn("residual_modal_in_capability", reason)

    def test_mlm_tapt_filter_rejects_fragments_and_messy_rows(self):
        cases = {
            "Initialization Data": "too_short",
            "The input data shall include: 1. Name 2. Address": "list_or_heading_marker",
            "The algorithm shall produce the output. NOTE 1 applies": "multi_sentence",
            "The scheduler shall set the interval timer. The display shall refresh": "multi_sentence",
            "The field shall match Table 8-1.": "table_or_figure_reference",
        }
        for requirement, expected_reason in cases.items():
            with self.subTest(requirement=requirement):
                include, reason = eu.mlm_tapt_filter(
                    requirement,
                    eu.auto_capability_text(requirement),
                    source_corpus="source_WEB",
                )
                self.assertFalse(include)
                self.assertIn(expected_reason, reason)

    def test_mlm_tapt_filter_accepts_clean_requirement(self):
        requirement = "The scheduler shall set the interval timer."
        include, reason = eu.mlm_tapt_filter(
            requirement,
            eu.auto_capability_text(requirement),
            source_corpus="source_WEB",
        )

        self.assertTrue(include, reason)

    def test_make_mlm_tapt_seed_candidates_preserves_source_excludes_pure_and_dedupes(
        self,
    ):
        rows = [
            {
                "source": "alpha_WEB",
                "reqs": "The scheduler shall set the interval timer.",
            },
            {
                "source": "alpha_WEB",
                "reqs": "The scheduler shall set the interval timer.",
            },
            {
                "source": "beta_PURE",
                "reqs": "The report shall include diagnostic data.",
            },
            {"source": "gamma_WEB", "reqs": "Initialization Data"},
            {"source": "delta_WEB", "reqs": "The display should show recent alerts."},
            {"source": "epsilon_WEB", "reqs": "Users may export reports."},
        ]

        candidates = eu.make_mlm_tapt_seed_candidates(rows, target_count=2, seed=123)
        selected = [row for row in candidates if row["include"] == "yes"]

        self.assertEqual(len(candidates), 5)
        self.assertEqual(len(selected), 2)
        self.assertEqual({row["source_dataset"] for row in candidates}, {"mlm_tapt"})
        self.assertIn("source_corpus", candidates[0])
        pure = next(row for row in candidates if row["source_corpus"] == "beta_PURE")
        self.assertEqual(pure["auto_include"], "no")
        self.assertIn("excluded_source", pure["auto_exclusion_reason"])

    PURE_FIXTURE_XML = """<?xml version="1.0"?>
<req_document xmlns="req_document.xsd">
  <title>Fixture FRS</title>
  <version>1</version>
  <p id="1"><title>Introduction</title>
    <p id="1.1"><text_body>(M) = Mandatory (O) = Optional</text_body></p>
  </p>
  <p id="2"><title>2 Network requirements</title>
    <p id="2.2"><title>2.2 Voice services</title>
      <req id="2.2.1"><text_body>2.2.1 The system shall support point-to-point voice calls. (M) Group calls</text_body></req>
      <req id="2.2.2"><text_body>2.2.2 The network should support fax transmissions between users. (O)</text_body></req>
      <req id="2.2.3"><text_body>2.2.3 Such calls include 112 calls. (I)</text_body></req>
      <req id="2.2.4"><text_body>2.2.4 The network shall provide: text messages; (O) automatic fax; (M)</text_body></req>
    </p>
  </p>
  <p><title>Functions</title>
    <req id="3.1a"><text_body>ETCS shall supervise train movements.</text_body><modifier>M</modifier></req>
    <req id="3.1b"><text_body>Train data may be entered automatically.</text_body><modifier>O</modifier></req>
  </p>
</req_document>
"""

    def test_parse_pure_document_extracts_markers_sections_and_neighbours(self):
        rows = eu.parse_pure_document(self.PURE_FIXTURE_XML, "fixture")
        by_id = {row["requirement_id"]: row for row in rows}

        self.assertEqual(
            [row["requirement_id"] for row in rows],
            ["2.2.1", "2.2.2", "2.2.3", "2.2.4", "3.1a", "3.1b"],
        )
        self.assertEqual(
            by_id["2.2.1"]["text"],
            "The system shall support point-to-point voice calls.",
        )
        self.assertEqual(by_id["2.2.1"]["marker"], "M")
        self.assertEqual(
            by_id["2.2.1"]["section_path"],
            "2 Network requirements > 2.2 Voice services",
        )
        self.assertEqual(by_id["2.2.2"]["marker"], "O")
        self.assertEqual(by_id["2.2.3"]["marker"], "I")
        self.assertEqual(by_id["2.2.4"]["marker_count"], 2)
        self.assertEqual(by_id["3.1a"]["marker"], "M")
        self.assertEqual(by_id["3.1b"]["marker"], "O")
        self.assertEqual(by_id["3.1b"]["section_path"], "Functions")
        self.assertEqual(by_id["3.1b"]["marker_count"], 1)
        self.assertEqual(by_id["3.1a"]["document_title"], "Fixture FRS")

        self.assertEqual(by_id["2.2.1"]["neighbour_before"], "")
        self.assertEqual(
            by_id["2.2.1"]["neighbour_after"],
            "2.2.2 (O): The network should support fax transmissions between users.",
        )
        self.assertEqual(by_id["3.1b"]["neighbour_after"], "")
        # List-shaped neighbours keep their per-item markers verbatim.
        self.assertEqual(
            by_id["3.1a"]["neighbour_before"],
            "2.2.4: The network shall provide: text messages; (O) automatic fax; (M)",
        )

    def test_pure_filter_reason_table(self):
        cases = [
            ("The network should support fax transmissions.", "O", 1, None),
            (
                "The network should support fax transmissions.",
                "I",
                1,
                "informative_marker",
            ),
            ("The network should support fax transmissions.", "", 0, "no_marker"),
            (
                "The network shall provide text messages and fax.",
                "M",
                2,
                "multiple_markers",
            ),
            (
                "It shall be possible to implement one level on a line.",
                "O",
                1,
                "impersonal_construction",
            ),
            (
                "The radio should comprise the following components:",
                "O",
                1,
                "colon_structure",
            ),
            ("The weight of the radio should not exceed 250g.", "O", 1, "negation"),
        ]
        for requirement, marker, count, expected in cases:
            with self.subTest(requirement=requirement, marker=marker):
                include, reason = eu.pure_filter(
                    requirement, eu.auto_capability_text(requirement), marker, count
                )
                if expected is None:
                    self.assertTrue(include, reason)
                else:
                    self.assertFalse(include)
                    self.assertIn(expected, reason)

    def test_make_pure_seed_candidates_keeps_every_optional_and_samples_mandatory(
        self,
    ):
        def row(rid, text, marker, doc="docA", count=1):
            return {
                "document_id": doc,
                "document_title": f"{doc} title",
                "requirement_id": rid,
                "section_path": "1 Section",
                "text": text,
                "marker": marker,
                "marker_count": count,
                "neighbour_before": "",
                "neighbour_after": "",
            }

        rows = [
            row("o1", "The network should support fax transmissions.", "O"),
            row("o2", "The network should support text messages.", "O"),
            row("o2dup", "The network should support text messages.", "O"),
            row("i1", "Such calls include emergency calls.", "I"),
            *[
                row(f"a{i}", f"The system shall support the {name} voice channel.", "M")
                for i, name in enumerate(
                    ["red", "blue", "green", "amber", "grey", "white"]
                )
            ],
            *[
                row(
                    f"b{i}",
                    f"The unit shall record the {name} diagnostic.",
                    "M",
                    "docB",
                )
                for i, name in enumerate(["thermal", "voltage", "timing"])
            ],
        ]

        candidates = eu.make_pure_seed_candidates(rows, target_count=5, seed=7)
        again = eu.make_pure_seed_candidates(rows, target_count=5, seed=7)
        selected = [c for c in candidates if c["include"] == "yes"]

        self.assertEqual(len(candidates), 12)  # duplicate text dropped
        self.assertEqual(len(selected), 5)
        self.assertEqual(
            sorted(c["context_marker"] for c in selected), ["M", "M", "M", "O", "O"]
        )
        self.assertEqual({c["source_dataset"] for c in candidates}, {"PURE"})
        self.assertEqual(
            {c["source_corpus"] for c in selected if c["context_marker"] == "M"},
            {"docA", "docB"},
        )
        self.assertEqual(
            [c["seed_id"] for c in selected],
            [c["seed_id"] for c in again if c["include"] == "yes"],
        )
        informative = next(c for c in candidates if c["context_requirement_id"] == "i1")
        self.assertEqual(informative["auto_include"], "no")
        self.assertIn("informative_marker", informative["auto_exclusion_reason"])
        not_sampled = [
            c for c in candidates if c["auto_include"] == "yes" and c["include"] == "no"
        ]
        self.assertTrue(not_sampled)
        self.assertEqual(
            {c["exclusion_reason"] for c in not_sampled}, {"not_sampled_mandatory_pool"}
        )
        self.assertEqual(
            set(eu.seed_review_fields("pure")) - set(eu.BASE_SEED_REVIEW_FIELDS),
            set(eu.PURE_CONTEXT_FIELDS),
        )
        self.assertTrue(set(eu.PURE_CONTEXT_FIELDS) <= set(selected[0]))

        with self.assertRaises(ValueError):
            eu.make_pure_seed_candidates(rows, target_count=50, seed=7)

    def test_weighted_sample_candidate_indices_is_deterministic_and_caps_sources(self):
        candidates = []
        for source in ["a", "b", "c", "d", "e", "f"]:
            for index in range(50):
                candidates.append(
                    {
                        "auto_include": "yes",
                        "source_corpus": source,
                        "id": f"{source}{index}",
                    }
                )

        first = eu.weighted_sample_candidate_indices(
            candidates, target_count=180, seed=42, source_cap=30
        )
        second = eu.weighted_sample_candidate_indices(
            candidates, target_count=180, seed=42, source_cap=30
        )
        counts = {}
        for index in first:
            source = candidates[index]["source_corpus"]
            counts[source] = counts.get(source, 0) + 1

        self.assertEqual(first, second)
        self.assertEqual(len(first), 180)
        self.assertLessEqual(max(counts.values()), 30)

    def test_artifact_path_combines_dataset_and_variant_suffixes(self):
        base = Path("data/processed/benchmark_items.csv")

        self.assertEqual(eu.artifact_path(base, "nice"), base)
        self.assertEqual(
            eu.artifact_path(base, "mlm_tapt"),
            Path("data/processed/benchmark_items_mlm_tapt.csv"),
        )
        self.assertEqual(
            eu.artifact_path(base, "mlm_tapt", "shall"),
            Path("data/processed/benchmark_items_mlm_tapt_shall.csv"),
        )
        self.assertEqual(
            eu.artifact_path(base, "pure"),
            Path("data/processed/benchmark_items_pure.csv"),
        )
        self.assertEqual(eu.normalize_dataset_id("PURE"), "pure")

    def test_task3_verification_items_path_is_run_specific(self):
        path_a = eu.task3_verification_items_path(
            Path("/tmp/re-uq"), "nice", "must", "full-a", "m1", "blind"
        )
        path_b = eu.task3_verification_items_path(
            Path("/tmp/re-uq"), "nice", "must", "full-b", "m1", "blind"
        )
        path_c = eu.task3_verification_items_path(
            Path("/tmp/re-uq"), "nice", "must", "full-a", "m2", "blind"
        )

        self.assertNotEqual(path_a, path_b)
        self.assertNotEqual(path_a, path_c)
        self.assertEqual(path_a.parent.name, "task3_verification_items")
        self.assertIn("full_a", path_a.name)
        self.assertIn("m1", path_a.name)

    def test_task3_verification_items_path_routes_smoke_runs_into_the_smoke_tree(self):
        default_path = eu.task3_verification_items_path(
            Path("/tmp/re-uq"), "nice", "must", "full-a", "m1", "blind"
        )
        smoke_path = eu.task3_verification_items_path(
            Path("/tmp/re-uq"), "nice", "must", "full-a", "m1", "blind", smoke=True
        )
        by_run_id = eu.task3_verification_items_path(
            Path("/tmp/re-uq"),
            "nice",
            "must",
            "full-a",
            "m1",
            "blind",
            run_id="smoke-1",
        )

        self.assertEqual(smoke_path.parent.name, "smoke")
        self.assertEqual(smoke_path.parent.parent.name, "task3_verification_items")
        self.assertEqual(smoke_path.name, default_path.name)
        self.assertEqual(by_run_id, smoke_path)
        # A full run keeps the unchanged paper-facing path.
        self.assertEqual(
            default_path,
            Path("/tmp/re-uq")
            / "data/processed/task3_verification_items"
            / default_path.name,
        )
        self.assertEqual(
            eu.task3_verification_items_path(
                Path("/tmp/re-uq"),
                "nice",
                "must",
                "full-a",
                "m1",
                "blind",
                run_id="full-a",
            ),
            default_path,
        )

    def test_parse_task2_response(self):
        raw = '{"requirement": "The system SHOULD export reports.", "modality": "should", "confidence": 80}'
        parsed, status = eu.parse_task_response("task2", raw)
        self.assertEqual(status, "ok")
        self.assertEqual(parsed["modality"], "recommended")

    def test_parse_task3_response_and_relation_aliases(self):
        raw = '{"relation": "stronger", "confidence": 82, "evidence_phrase": "MAY", "brief_reason": "upgraded"}'
        parsed, status = eu.parse_task_response("task3", raw)

        self.assertEqual(status, "ok")
        self.assertEqual(parsed["relation"], "strengthens")
        self.assertEqual(eu.normalize_relation("same modality"), "preserves")
        self.assertEqual(eu.normalize_relation("content mismatch"), "content_changed")

        _, status = eu.parse_task_response(
            "task3", '{"relation":"preserves","confidence":50}'
        )
        self.assertEqual(status, "missing_fields")

        _, status = eu.parse_task_response(
            "task3", '{"relation":"unclear","confidence":50,"evidence_phrase":"MAY"}'
        )
        self.assertEqual(status, "invalid_label")

    def test_task3_gold_relation_from_ordinal_modality(self):
        self.assertEqual(
            eu.task3_gold_relation("nice_to_have", "recommended"), "strengthens"
        )
        self.assertEqual(eu.task3_gold_relation("optional", "mandatory"), "strengthens")
        self.assertEqual(eu.task3_gold_relation("mandatory", "optional"), "weakens")
        self.assertEqual(
            eu.task3_gold_relation("recommended", "recommended"), "preserves"
        )

    def test_task3_prompt_modes_are_distinct(self):
        item = {
            "source_statement": "It would be useful if the system could export reports.",
            "task2_requirement": "The system SHOULD export reports.",
            "source_modality": "nice_to_have",
            "task2_text_modality": "recommended",
        }
        blind_template = Path("prompts/modality_verification.txt").read_text(
            encoding="utf-8"
        )
        declared_template = Path(
            "prompts/modality_verification_declared.txt"
        ).read_text(encoding="utf-8")

        blind_prompt = task3_cli.task3_prompt_for(
            blind_template, item, audit_mode="blind"
        )
        declared_text_prompt = task3_cli.task3_prompt_for(
            declared_template, item, audit_mode="declared_text"
        )
        declared_source_prompt = task3_cli.task3_prompt_for(
            declared_template, item, audit_mode="declared_source"
        )

        self.assertNotIn("Declared extracted modality", blind_prompt)
        self.assertIn("Declared extracted modality", declared_text_prompt)
        self.assertIn('"recommended"', declared_text_prompt)
        self.assertIn('"nice_to_have"', declared_source_prompt)
        self.assertNotEqual(blind_prompt, declared_text_prompt)
        self.assertNotEqual(declared_text_prompt, declared_source_prompt)

    def test_task3_batch_prompt_is_blind_unless_audit_mode_declared(self):
        base_item = {
            "item_id": "i1",
            "source_statement": "It would be useful if the system could export reports.",
            "task2_requirement": "The system SHOULD export reports.",
            "source_modality": "nice_to_have",
            "task2_modality": "nice_to_have",
            "task2_text_modality": "recommended",
            "task3_audit_mode": "blind",
        }
        blind_prompt = eu.batch_prompt_for_completion_jobs(
            [{"task": "task3", "request_index": 0, "item": base_item}]
        )
        declared_prompt = eu.batch_prompt_for_completion_jobs(
            [
                {
                    "task": "task3",
                    "request_index": 0,
                    "item": {**base_item, "task3_audit_mode": "declared_text"},
                }
            ]
        )

        self.assertNotIn("extracted_modality", blind_prompt)
        self.assertNotIn("declared_extracted_modality", blind_prompt)
        self.assertIn("declared_extracted_modality", declared_prompt)
        self.assertIn("recommended", declared_prompt)

    def test_metrics(self):
        y_true = [1, 0, 1, 0]
        p = [0.9, 0.2, 0.8, 0.1]
        self.assertLess(eu.brier_score(y_true, p), 0.05)
        self.assertEqual(eu.auroc_score(y_true, p), 1.0)
        self.assertFalse(math.isnan(eu.ece_score(y_true, p)))
        rank_strength = [1.0, 0.67, 0.33, 0.0]
        recoded_strength = [1.0, 0.75, 0.33, 0.0]
        p_yes = [1.0, 0.05, 0.0, 0.05]
        self.assertEqual(
            eu.spearman_corr(rank_strength, p_yes),
            eu.spearman_corr(recoded_strength, p_yes),
        )
        self.assertNotEqual(
            eu.pearson_corr(rank_strength, p_yes),
            eu.pearson_corr(recoded_strength, p_yes),
        )

    def test_distribution_uncertainty_helpers(self):
        distribution = eu.label_distribution(["yes", "yes", "no"], ["yes", "no"])

        self.assertEqual(distribution, {"yes": 2 / 3, "no": 1 / 3})
        # Ties break toward the weakest label so they do not inflate over-commitment.
        self.assertEqual(
            eu.majority_label({"yes": 0.5, "no": 0.5}, ["yes", "no"]), "no"
        )
        self.assertAlmostEqual(eu.variation_ratio(distribution), 1 / 3)
        self.assertGreater(eu.normalized_predictive_entropy(distribution), 0.0)
        self.assertLess(eu.normalized_predictive_entropy(distribution), 1.0)

    def test_majority_label_tie_breaks_toward_weakest_label(self):
        modalities = eu.MODALITIES
        # (a) task2 2-2 tie between strongest and weakest -> weakest.
        self.assertEqual(
            eu.majority_label({"mandatory": 0.5, "nice_to_have": 0.5}, modalities),
            "nice_to_have",
        )
        # (b) task1 even split -> weaker "no".
        self.assertEqual(
            eu.majority_label({"yes": 0.5, "no": 0.5}, ["yes", "no"]), "no"
        )
        # (c) uniform 4-way tie -> weakest modality.
        uniform = dict.fromkeys(modalities, 0.25)
        self.assertEqual(eu.majority_label(uniform, modalities), "nice_to_have")
        # (d) no-tie cases resolve to the true maximum regardless of order.
        self.assertEqual(
            eu.majority_label({"mandatory": 0.6, "nice_to_have": 0.4}, modalities),
            "mandatory",
        )
        self.assertEqual(
            eu.majority_label(
                {"mandatory": 0.1, "optional": 0.7, "nice_to_have": 0.2}, modalities
            ),
            "optional",
        )

    def test_acse_semantic_proxy_distinguishes_identical_from_divergent_samples(self):
        identical = eu.acse_semantic_proxy_diagnostics(["decision: yes"] * 5)
        divergent = eu.acse_semantic_proxy_diagnostics(
            [
                "decision: yes because export is mandatory",
                "decision: no because export is optional",
                "decision: yes because logging is mandatory",
                "decision: no because deletion is out of scope",
                "decision: yes because payment is required",
            ]
        )

        self.assertEqual(identical["semantic_cluster_count"], 1)
        self.assertEqual(identical["semantic_uncertainty_score"], 0.0)
        self.assertGreater(divergent["semantic_cluster_count"], 1)
        self.assertGreater(
            divergent["semantic_uncertainty_score"],
            identical["semantic_uncertainty_score"],
        )

    def test_acse_semantic_proxy_can_use_lazy_mlx_backend(self):
        fake_embeddings = eu.np.asarray(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
            ],
            dtype=float,
        )
        with mock.patch.object(
            eu, "_mlx_text_embedding_matrix", return_value=fake_embeddings
        ) as embed:
            diagnostics = eu.acse_semantic_proxy_diagnostics(
                ["alpha", "alpha-ish", "beta"],
                embedding_backend="mlx",
                mlx_model_name="mlx-community/fake-embedding",
            )

        embed.assert_called_once()
        self.assertEqual(
            diagnostics["semantic_embedding_backend"],
            "mlx:mlx-community/fake-embedding",
        )
        self.assertGreater(diagnostics["semantic_cluster_count"], 1)

    def test_global_acse_loader_refits_tfidf_caches_in_shared_space(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_rows = []
            for index, width in enumerate([2, 3]):
                artifact_dir = root / f"run{index}"
                artifact_dir.mkdir()
                eu.np.savez_compressed(
                    artifact_dir / "task2_acse_sample_embeddings.npz",
                    embeddings=eu.np.ones((1, width), dtype=float),
                )
                eu.write_csv_rows(
                    artifact_dir / "task2_acse_samples.csv",
                    [
                        {
                            "embedding_index": 0,
                            "item_id": f"item{index}",
                            "seed_id": f"S{index:04d}",
                            "sample_index": 0,
                            "semantic_text": f"requirement: {'alpha' if index == 0 else 'beta gamma'}",
                            "requirement": "",
                            "pred_modality": "optional",
                        }
                    ],
                )
                eu.write_csv_rows(
                    artifact_dir / "task2_acse_items.csv",
                    [
                        {
                            "item_id": f"item{index}",
                            "source_modality": "optional",
                            "strict_text_overcommit": "0",
                            "text_overcommit": "0",
                            "acse_uncertainty_score": "0.1",
                        }
                    ],
                )
                manifest_rows.append(
                    {
                        "artifact_dir": str(artifact_dir),
                        "dataset_id": "nice",
                        "benchmark_variant": "must",
                        "run_id": f"run{index}",
                        "model": "m",
                        "profile_id": "p",
                        "embedding_backend": eu.ACSE_PROXY_EMBEDDING_BACKEND,
                    }
                )

            embeddings, rows = acse_global_projection.load_embeddings_and_rows(
                manifest_rows
            )

        self.assertEqual(embeddings.shape[0], 2)
        self.assertGreater(embeddings.shape[1], 0)
        self.assertEqual([row["global_embedding_index"] for row in rows], [0, 1])
        self.assertEqual(
            {row["global_embedding_source"] for row in rows}, {"shared_tfidf_refit"}
        )

    def test_acse_normalization_and_calibration_diagnostics(self):
        base = {
            "run_id": "r1",
            "model": "m1",
            "task": "task1",
            "uq_method": eu.ACSE_PROXY_METHOD,
            "seed_id": "S0001",
            "source_modality": "mandatory",
            "semantic_embedding_backend": "mlx:fake",
            "valid_n": 5,
            "total_n": 5,
        }
        scores = [
            {
                **base,
                "item_id": "low-risk",
                "y_true": 1,
                "y_pred": 1,
                "uncertainty_score": 0.2,
                "semantic_cluster_count": 1,
            },
            {
                **base,
                "item_id": "high-risk",
                "y_true": 0,
                "y_pred": 1,
                "uncertainty_score": 0.8,
                "semantic_cluster_count": 2,
            },
        ]

        normalized = eu.acse_normalized_score_rows(scores)
        calibration = eu.acse_calibration_diagnostic_rows(
            normalized, target_error_rates=(0.0,)
        )

        by_item = {row["item_id"]: row for row in normalized}
        self.assertEqual(by_item["low-risk"]["acse_normalized_uncertainty_score"], 0.0)
        self.assertEqual(by_item["high-risk"]["acse_normalized_uncertainty_score"], 1.0)
        self.assertEqual(len(calibration), 1)
        self.assertEqual(calibration[0]["selected_normalized_threshold"], 0.0)
        self.assertEqual(calibration[0]["calibration_coverage"], 0.5)
        self.assertEqual(calibration[0]["calibration_accepted_error_rate"], 0.0)

    def test_monotonicity_violation(self):
        rows = [
            {"seed_id": "S1", "source_modality": "mandatory", "p_yes": 0.9},
            {"seed_id": "S1", "source_modality": "recommended", "p_yes": 0.6},
            {"seed_id": "S1", "source_modality": "optional", "p_yes": 0.3},
            {"seed_id": "S1", "source_modality": "nice_to_have", "p_yes": 0.1},
            {"seed_id": "S2", "source_modality": "mandatory", "p_yes": 0.9},
            {"seed_id": "S2", "source_modality": "recommended", "p_yes": 0.94},
            {"seed_id": "S2", "source_modality": "optional", "p_yes": 0.2},
            {"seed_id": "S2", "source_modality": "nice_to_have", "p_yes": 0.1},
            {"seed_id": "S3", "source_modality": "mandatory", "p_yes": 0.9},
            {"seed_id": "S3", "source_modality": "recommended", "p_yes": 0.96},
            {"seed_id": "S3", "source_modality": "optional", "p_yes": 0.2},
            {"seed_id": "S3", "source_modality": "nice_to_have", "p_yes": 0.1},
        ]
        diagnostics = eu.monotonicity_violation_diagnostics(rows)

        self.assertEqual(
            eu.monotonicity_violation_diagnostics(rows, tolerance=0.0)[
                "monotonicity_violations"
            ],
            2 / 3,
        )
        self.assertEqual(diagnostics["monotonicity_violations"], 1 / 3)
        self.assertEqual(diagnostics["monotonicity_strict_violations"], 2 / 3)
        self.assertEqual(
            diagnostics["monotonicity_tolerance"], eu.MONOTONICITY_TOLERANCE
        )
        self.assertAlmostEqual(
            diagnostics["monotonicity_mean_max_increase"], (0.0 + 0.04 + 0.06) / 3
        )
        self.assertAlmostEqual(diagnostics["monotonicity_max_increase"], 0.06)

    def test_high_confidence_overcommitment_metrics(self):
        task1_rows = [
            {"task": "task1", "y_true": 0, "p_yes": 0.95},
            {"task": "task1", "y_true": 0, "p_yes": 0.20},
            {"task": "task1", "y_true": 1, "p_yes": 0.95},
        ]
        task2_rows = [
            {
                "task": "task2",
                "gold_modality": "optional",
                "pred_modality": "mandatory",
                "confidence": 0.90,
            },
            {
                "task": "task2",
                "gold_modality": "recommended",
                "pred_modality": "optional",
                "confidence": 0.95,
            },
        ]

        self.assertEqual(
            eu.high_confidence_overcommitment_rate(task1_rows, "task1", 0.80), 0.5
        )
        self.assertEqual(
            eu.high_confidence_overcommitment_rate(task2_rows, "task2", 0.80), 0.5
        )

    def test_task2_prompt_sensitivity_summary_counts_nice_to_have_upgrade(self):
        benchmark = eu.build_benchmark_items(export_report_seeds())
        item = next(
            row for row in benchmark if row["source_modality"] == "nice_to_have"
        )
        raw_rows = [
            {
                "run_id": "r1-default",
                "model": "m1:default",
                "task": "task2",
                "item_id": item["item_id"],
                "seed_id": item["seed_id"],
                "source_modality": item["source_modality"],
                "parsed_json": {
                    "requirement": "The system should export reports.",
                    "modality": "recommended",
                    "confidence": 95.0,
                },
                "parse_status": "ok",
            },
            {
                "run_id": "r1-labels-only",
                "model": "m1:labels_only",
                "task": "task2",
                "item_id": item["item_id"],
                "seed_id": item["seed_id"],
                "source_modality": item["source_modality"],
                "parsed_json": {
                    "requirement": "It would be useful if the system could export reports.",
                    "modality": "nice_to_have",
                    "confidence": 90.0,
                },
                "parse_status": "ok",
            },
        ]

        summary = eu.task2_prompt_sensitivity_summary(benchmark, raw_rows)
        by_model = {row["model"]: row for row in summary}

        self.assertEqual(by_model["m1:default"]["nice_to_have_accuracy"], 0.0)
        self.assertEqual(
            by_model["m1:default"]["nice_to_have_to_recommended_rate"], 1.0
        )
        self.assertEqual(by_model["m1:default"]["high_conf_overcommit_90"], 1.0)
        self.assertEqual(by_model["m1:labels_only"]["nice_to_have_accuracy"], 1.0)
        self.assertEqual(by_model["m1:labels_only"]["over_commitment"], 0.0)

    def test_weak_modality_probe_summary_counts_overcommitment(self):
        seeds = export_report_seeds()
        items = eu.build_weak_modality_probe_items(seeds)
        item = next(row for row in items if row["template_id"] == "useful_if")
        raw_record = eu.build_raw_record(
            run_id="weak-probe-r1",
            model="m1",
            host="http://localhost:8000/v1",
            task="task2",
            item=item,
            sample_index=0,
            sample_kind="deterministic",
            temperature=0.0,
            top_p=1.0,
            prompt_version="v1",
            prompt="prompt",
            completion={
                "ok": True,
                "raw_text": '{"requirement":"The system should export reports.","modality":"recommended","confidence":95}',
                "latency_s": 0.1,
                "error": "",
            },
        )

        summary = eu.weak_modality_probe_summary(items, [raw_record])

        self.assertEqual(raw_record["template_id"], "useful_if")
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["template_id"], "useful_if")
        self.assertEqual(summary[0]["valid_n"], 1)
        self.assertEqual(summary[0]["accuracy"], 0.0)
        self.assertEqual(summary[0]["to_recommended_rate"], 1.0)
        self.assertEqual(summary[0]["over_commitment"], 1.0)
        self.assertEqual(summary[0]["high_conf_overcommit_90"], 1.0)

    def test_weak_modality_probe_summary_uses_instructor_confidence_scale(self):
        seeds = export_report_seeds()
        items = eu.build_weak_modality_probe_items(seeds)
        item = next(row for row in items if row["template_id"] == "useful_if")
        raw_record = {
            "run_id": "weak-probe-r1",
            "model": "m1",
            "task": "task2",
            "item_id": item["item_id"],
            "seed_id": item["seed_id"],
            "source_modality": item["source_modality"],
            "sample_kind": "deterministic",
            "sample_index": 0,
            "parse_status": "ok",
            "parsed_json": {
                "requirement": "The system should export reports.",
                "modality": "recommended",
                "confidence": 0.95,
            },
            "output_contract_version": so.INSTRUCTOR_OUTPUT_CONTRACT_VERSION,
            "confidence_scale": so.INSTRUCTOR_CONFIDENCE_SCALE,
        }

        summary = eu.weak_modality_probe_summary(items, [raw_record])

        self.assertEqual(summary[0]["high_conf_overcommit_90"], 1.0)
        self.assertEqual(summary[0]["mean_confidence"], 0.95)

    def test_qualitative_examples_sorted_by_risk(self):
        benchmark = eu.build_benchmark_items(export_report_seeds())
        optional_item = next(
            row for row in benchmark if row["source_modality"] == "optional"
        )
        nice_item = next(
            row for row in benchmark if row["source_modality"] == "nice_to_have"
        )
        scores = [
            {
                "model": "m1",
                "task": "task1",
                "uq_method": "verbalized_confidence",
                "item_id": optional_item["item_id"],
                "seed_id": optional_item["seed_id"],
                "source_modality": optional_item["source_modality"],
                "y_true": 0,
                "p_yes": 0.85,
            },
            {
                "model": "m1",
                "task": "task1",
                "uq_method": "verbalized_confidence",
                "item_id": nice_item["item_id"],
                "seed_id": nice_item["seed_id"],
                "source_modality": nice_item["source_modality"],
                "y_true": 0,
                "p_yes": 0.95,
            },
        ]
        examples = eu.qualitative_overcommitment_examples(
            scores, benchmark, limit=2, threshold=0.80
        )

        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0]["risk_score"], 0.95)
        self.assertEqual(examples[0]["source_modality"], "nice_to_have")

    def test_build_uq_scores_and_summary(self):
        benchmark = eu.build_benchmark_items(export_report_seeds())
        raw_rows = []
        for item in benchmark:
            decision = "yes" if item["source_modality"] == "mandatory" else "no"
            raw_rows.append(
                raw_record(
                    item,
                    task="task1",
                    parsed_json={
                        "decision": decision,
                        "confidence": 90.0,
                        "brief_reason": "",
                    },
                )
            )
        scores = eu.build_uq_scores(benchmark, raw_rows)
        summary = eu.metric_summary_by_model_task_method(scores)
        self.assertEqual(len(scores), 4)
        self.assertEqual(summary[0]["accuracy"], 1.0)
        self.assertTrue(all("uncertainty_score" in row for row in scores))

    def _task2_raw_row(self, item, requirement, **overrides):
        row = raw_record(
            item,
            task="task2",
            parsed_json={
                "requirement": requirement,
                "modality": "mandatory",
                "confidence": 0.9,
            },
        )
        row["raw_text"] = json.dumps(row["parsed_json"])
        row.update(overrides)
        return row

    def test_duplicate_completed_rows_score_once_from_the_latest_ok_row(self):
        benchmark = eu.build_benchmark_items(export_report_seeds())
        item = next(row for row in benchmark if row["source_modality"] == "mandatory")
        rows = [
            self._task2_raw_row(item, "The system may export reports."),
            self._task2_raw_row(item, "The system must export reports."),
        ]

        scores = [
            row
            for row in eu.build_uq_scores(benchmark, rows)
            if row["uq_method"] == "verbalized_confidence"
        ]

        self.assertEqual(len(scores), 1)
        # The later row wins, so the score reflects the latest answer's text.
        self.assertEqual(scores[0]["text_modality"], "mandatory")
        self.assertEqual(len(eu.dedupe_raw_rows(rows)), 1)

    def test_a_failed_row_followed_by_an_ok_retry_scores_once(self):
        benchmark = eu.build_benchmark_items(export_report_seeds())
        item = next(row for row in benchmark if row["source_modality"] == "mandatory")
        failed = self._task2_raw_row(
            item, "", parse_status="invalid_json", parsed_json=None
        )
        rows = [failed, self._task2_raw_row(item, "The system must export reports.")]

        scores = [
            row
            for row in eu.build_uq_scores(benchmark, rows)
            if row["uq_method"] == "verbalized_confidence"
        ]

        self.assertEqual(len(scores), 1)
        self.assertEqual(scores[0]["text_modality"], "mandatory")
        # The ok row wins whichever order it was appended in.
        self.assertEqual(
            [row["parse_status"] for row in eu.dedupe_raw_rows(rows)], ["ok"]
        )
        self.assertEqual(
            [row["parse_status"] for row in eu.dedupe_raw_rows(list(reversed(rows)))],
            ["ok"],
        )
        # With no ok row at all the last attempt survives, so the failure is visible.
        self.assertEqual(
            [row["parse_status"] for row in eu.dedupe_raw_rows([failed, failed])],
            ["invalid_json"],
        )

    def test_dedupe_keeps_distinct_requests_and_runs_apart(self):
        benchmark = eu.build_benchmark_items(export_report_seeds())
        item = next(row for row in benchmark if row["source_modality"] == "mandatory")
        other_item = next(
            row for row in benchmark if row["source_modality"] == "optional"
        )
        rows = [
            self._task2_raw_row(item, "a"),
            self._task2_raw_row(item, "b", run_id="r2"),
            self._task2_raw_row(other_item, "c"),
            self._task2_raw_row(item, "d", sample_kind="stochastic", sample_index=0),
            self._task2_raw_row(item, "e", sample_kind="stochastic", sample_index=1),
            self._task2_raw_row(item, "f", model="m2"),
        ]

        self.assertEqual(len(eu.dedupe_raw_rows(rows)), len(rows))
        # Order of the survivors follows first appearance.
        self.assertEqual(
            [
                row["parsed_json"]["requirement"]
                for row in eu.dedupe_raw_rows([*rows, self._task2_raw_row(item, "z")])
            ],
            ["z", "b", "c", "d", "e", "f"],
        )

    def test_duplicate_rows_do_not_inflate_run_progress(self):
        benchmark = eu.build_benchmark_items(export_report_seeds())
        item = next(row for row in benchmark if row["source_modality"] == "mandatory")
        rows = [self._task2_raw_row(item, "a"), self._task2_raw_row(item, "b")]

        progress = eu.run_progress_summary(
            benchmark, rows, expected_stochastic_samples=0
        )

        self.assertEqual(len(progress), 1)
        self.assertEqual(progress[0]["observed_records"], 1)
        self.assertEqual(progress[0]["deterministic_records"], 1)

    def test_duplicate_task2_rows_build_one_task3_item(self):
        benchmark = eu.build_benchmark_items(export_report_seeds())
        item = next(
            row for row in benchmark if row["source_modality"] == "nice_to_have"
        )
        rows = [
            self._task2_raw_row(item, "The system may export reports."),
            self._task2_raw_row(item, "The system must export reports."),
        ]

        items = eu.build_task3_verification_items(benchmark, rows)

        self.assertEqual(len(items), 1)
        self.assertEqual(
            items[0]["task2_requirement"], "The system must export reports."
        )
        self.assertEqual(items[0]["task2_text_modality"], "mandatory")

    def test_stochastic_completeness_counts_samples_that_were_never_written(self):
        benchmark = eu.build_benchmark_items(export_report_seeds())
        item = next(row for row in benchmark if row["source_modality"] == "mandatory")
        rows = [
            self._task2_raw_row(
                item,
                "The system must export reports.",
                sample_kind="stochastic",
                sample_index=sample_index,
            )
            for sample_index in range(4)
        ]

        without_expectation = [
            row
            for row in eu.build_uq_scores(benchmark, rows)
            if row["uq_method"] == "modality_consistency"
        ]
        with_expectation = [
            row
            for row in eu.build_uq_scores(
                benchmark, rows, expected_stochastic_samples=5
            )
            if row["uq_method"] == "modality_consistency"
        ]

        self.assertEqual([row["total_n"] for row in without_expectation], [4])
        self.assertTrue(without_expectation[0]["stochastic_complete"])
        self.assertEqual([row["total_n"] for row in with_expectation], [5])
        self.assertEqual(with_expectation[0]["valid_n"], 4)
        self.assertEqual(with_expectation[0]["parse_failures"], 1)
        self.assertFalse(with_expectation[0]["stochastic_complete"])

    def test_stale_raw_prompt_rows_are_excluded_from_scores(self):
        benchmark = eu.build_benchmark_items(
            [
                {
                    "seed_id": "S0088",
                    "source_dataset": "NICE",
                    "original_requirement": "The system shall interface with CampusConnect's central server.",
                    "capability_text_final": "interface with CampusConnect's central server",
                }
            ]
        )
        item = next(row for row in benchmark if row["source_modality"] == "mandatory")
        raw = {
            "run_id": "r1",
            "model": "m1",
            "host": "http://localhost:8000/v1",
            "task": "task1",
            "item_id": item["item_id"],
            "seed_id": item["seed_id"],
            "source_modality": item["source_modality"],
            "sample_index": 0,
            "sample_kind": "deterministic",
            "temperature": 0.0,
            "top_p": 1.0,
            "prompt_version": "v1",
            "raw_text": "",
            "parsed_json": {"decision": "yes", "confidence": 90.0, "brief_reason": ""},
            "parse_status": "ok",
            "latency_s": 0.1,
            "error": "",
            "prompt": 'Source statement:\n"The system MUST with CampusConnect\'s central server."',
        }

        self.assertEqual(
            eu.benchmark_rows_with_current_raw_outputs(benchmark, [raw]), []
        )
        self.assertEqual(eu.build_uq_scores(benchmark, [raw]), [])

        fresh = {**raw, "prompt": f'Source statement:\n"{item["source_statement"]}"'}
        self.assertEqual(
            len(eu.benchmark_rows_with_current_raw_outputs(benchmark, [fresh])), 1
        )
        self.assertEqual(len(eu.build_uq_scores(benchmark, [fresh])), 1)

    def test_build_task3_items_scores_and_summary(self):
        benchmark = eu.build_benchmark_items(export_report_seeds())
        source_item = next(
            row for row in benchmark if row["source_modality"] == "nice_to_have"
        )
        task2_raw = [
            {
                "run_id": "full-r1",
                "model": "m1",
                "host": "http://localhost:8000/v1",
                "task": "task2",
                "item_id": source_item["item_id"],
                "seed_id": source_item["seed_id"],
                "source_modality": source_item["source_modality"],
                "sample_index": 0,
                "sample_kind": "deterministic",
                "temperature": 0.0,
                "top_p": 1.0,
                "prompt_version": "v1",
                "raw_text": "",
                "parsed_json": {
                    "requirement": "The system SHOULD export reports.",
                    "modality": "nice_to_have",
                    "confidence": 95.0,
                },
                "parse_status": "ok",
                "latency_s": 0.1,
                "error": "",
            }
        ]
        task3_items = eu.build_task3_verification_items(benchmark, task2_raw)
        item = task3_items[0]

        self.assertEqual(len(task3_items), 1)
        self.assertEqual(item["source_item_id"], source_item["item_id"])
        self.assertEqual(item["task2_modality"], "nice_to_have")
        self.assertEqual(item["task2_text_modality"], "recommended")
        self.assertEqual(item["task2_text_modality_basis"], "explicit_modal")
        self.assertEqual(item["task3_declared_relation"], "preserves")
        self.assertEqual(item["task3_gold_relation"], "strengthens")
        self.assertEqual(item["task3_audit_mode"], "blind")

        raw_record = eu.build_raw_record(
            run_id="task3-r1",
            model="m1",
            host="http://localhost:8000/v1",
            task="task3",
            item=item,
            sample_index=0,
            sample_kind="deterministic",
            temperature=0.0,
            top_p=1.0,
            prompt_version="v1:task3:blind",
            prompt="prompt",
            completion={
                "ok": True,
                "raw_text": (
                    '{"relation":"preserves","confidence":0.9,'
                    '"evidence_phrase":"It would be useful if","brief_reason":"missed"}'
                ),
                "latency_s": 0.1,
                "error": "",
            },
        )
        self.assertEqual(
            raw_record["task2_requirement"], "The system SHOULD export reports."
        )
        self.assertEqual(raw_record["task2_text_modality"], "recommended")
        self.assertEqual(raw_record["task3_declared_relation"], "preserves")
        reconstructed_items = eu.task3_items_from_raw_rows([raw_record])
        self.assertEqual(reconstructed_items[0]["task3_gold_relation"], "strengthens")

        raw_rows = [
            {
                "run_id": "task3-r1",
                "model": "m1",
                "host": "http://localhost:8000/v1",
                "task": "task3",
                "item_id": item["item_id"],
                "seed_id": item["seed_id"],
                "source_modality": item["source_modality"],
                "sample_index": 0,
                "sample_kind": "deterministic",
                "temperature": 0.0,
                "top_p": 1.0,
                "prompt_version": "v1:task3",
                "raw_text": "",
                "parsed_json": {
                    "relation": "preserves",
                    "confidence": 0.9,
                    "evidence_phrase": "It would be useful if",
                    "brief_reason": "missed upgrade",
                },
                "parse_status": "ok",
                "latency_s": 0.1,
                "error": "",
                "output_contract_version": so.INSTRUCTOR_OUTPUT_CONTRACT_VERSION,
                "confidence_scale": so.INSTRUCTOR_CONFIDENCE_SCALE,
            },
            {
                "run_id": "task3-r1",
                "model": "m1",
                "host": "http://localhost:8000/v1",
                "task": "task3",
                "item_id": item["item_id"],
                "seed_id": item["seed_id"],
                "source_modality": item["source_modality"],
                "sample_index": 0,
                "sample_kind": "stochastic",
                "temperature": 0.7,
                "top_p": 1.0,
                "prompt_version": "v1:task3",
                "raw_text": "",
                "parsed_json": {
                    "relation": "strengthens",
                    "confidence": 80.0,
                    "evidence_phrase": "It would be useful if",
                },
                "parse_status": "ok",
                "latency_s": 0.1,
                "error": "",
            },
            {
                "run_id": "task3-r1",
                "model": "m1",
                "host": "http://localhost:8000/v1",
                "task": "task3",
                "item_id": item["item_id"],
                "seed_id": item["seed_id"],
                "source_modality": item["source_modality"],
                "sample_index": 1,
                "sample_kind": "stochastic",
                "temperature": 0.7,
                "top_p": 1.0,
                "prompt_version": "v1:task3",
                "raw_text": "",
                "parsed_json": {
                    "relation": "strengthens",
                    "confidence": 75.0,
                    "evidence_phrase": "useful",
                },
                "parse_status": "ok",
                "latency_s": 0.1,
                "error": "",
            },
            {
                "run_id": "task3-r1",
                "model": "m1",
                "host": "http://localhost:8000/v1",
                "task": "task3",
                "item_id": item["item_id"],
                "seed_id": item["seed_id"],
                "source_modality": item["source_modality"],
                "sample_index": 2,
                "sample_kind": "stochastic",
                "temperature": 0.7,
                "top_p": 1.0,
                "prompt_version": "v1:task3",
                "raw_text": "",
                "parsed_json": None,
                "parse_status": "invalid_json",
                "latency_s": 0.1,
                "error": "",
            },
        ]

        scores = eu.build_task3_scores(task3_items, raw_rows)
        summary = eu.metric_summary_by_model_task_method(scores)
        by_method = {row["uq_method"]: row for row in summary}

        self.assertEqual({row["task"] for row in scores}, {"task3"})
        verbalized_score = next(
            row for row in scores if row["uq_method"] == "verbalized_confidence"
        )
        self.assertEqual(verbalized_score["confidence"], 0.9)
        self.assertEqual(by_method["verbalized_confidence"]["accuracy"], 0.0)
        self.assertEqual(by_method["verbalized_confidence"]["f1_or_macro_f1"], 0.0)
        self.assertEqual(
            by_method["verbalized_confidence"]["strengthening_recall"], 0.0
        )
        self.assertEqual(by_method["verbalized_confidence"]["false_preserve_rate"], 1.0)
        self.assertEqual(
            by_method["verbalized_confidence"]["evidence_phrase_source_rate"], 1.0
        )
        self.assertEqual(by_method["relation_consistency"]["accuracy"], 1.0)
        self.assertEqual(by_method["relation_consistency"]["f1_or_macro_f1"], 1.0)
        self.assertAlmostEqual(
            by_method["relation_consistency"]["parse_failure_rate"], 1 / 3
        )

    def test_task2_summary_accuracy_uses_modality_labels(self):
        scores = [
            {
                "model": "m1",
                "task": "task2",
                "uq_method": "verbalized_confidence",
                "item_id": "S0001_nice_to_have",
                "seed_id": "S0001",
                "source_modality": "nice_to_have",
                "ordinal_strength": 0,
                "numeric_strength": 0.0,
                "valid_n": 1,
                "total_n": 1,
                "parse_failures": 0,
                "y_true": 0,
                "y_pred": 0,
                "p_yes": "",
                "confidence": 0.95,
                "uncertainty_score": 0.05,
                "uncertainty_measure": "one_minus_confidence",
                "label_distribution": "",
                "gold_modality": "nice_to_have",
                "pred_modality": "recommended",
            },
            {
                "model": "m1",
                "task": "task2",
                "uq_method": "verbalized_confidence",
                "item_id": "S0001_optional",
                "seed_id": "S0001",
                "source_modality": "optional",
                "ordinal_strength": 1,
                "numeric_strength": 0.33,
                "valid_n": 1,
                "total_n": 1,
                "parse_failures": 0,
                "y_true": 1,
                "y_pred": 1,
                "p_yes": "",
                "confidence": 0.90,
                "uncertainty_score": 0.10,
                "uncertainty_measure": "one_minus_confidence",
                "label_distribution": "",
                "gold_modality": "optional",
                "pred_modality": "optional",
            },
        ]

        summary = eu.metric_summary_by_model_task_method(scores)
        acc_point, _, _ = eu.bootstrap_seed_metric(
            scores,
            lambda sample_rows: eu.task_accuracy(sample_rows, "task2"),
            iterations=10,
        )

        self.assertEqual(summary[0]["accuracy"], 0.5)
        self.assertEqual(acc_point, 0.5)
        self.assertEqual(
            summary[0]["f1_or_macro_f1"],
            eu.macro_f1_score(
                ["nice_to_have", "optional"], ["recommended", "optional"], eu.MODALITIES
            ),
        )
        self.assertEqual(summary[0]["over_commitment"], 0.5)
        self.assertEqual(summary[0]["over_commitment_severity"], 1.0)
        self.assertEqual(summary[0]["over_commitment_severity_all"], 1.0)
        self.assertEqual(
            summary[0]["over_commitment_severity_given_overcommitment"], 2.0
        )
        self.assertEqual(summary[0]["weak_recall"], 0.0)
        self.assertEqual(summary[0]["weak_strengthening_90"], 1.0)
        self.assertEqual(summary[0]["high_conf_overcommit_all_90"], 0.5)
        self.assertEqual(summary[0]["high_conf_overcommit_overcommittable_90"], 0.5)

    def test_high_confidence_denominators_are_explicit(self):
        task1_rows = [
            {"task": "task1", "y_true": 0, "p_yes": 0.95},
            {"task": "task1", "y_true": 0, "p_yes": 0.40},
            {"task": "task1", "y_true": 1, "p_yes": 0.99},
        ]
        task2_rows = [
            {
                "task": "task2",
                "gold_modality": "nice_to_have",
                "pred_modality": "recommended",
                "confidence": 0.95,
            },
            {
                "task": "task2",
                "gold_modality": "optional",
                "pred_modality": "optional",
                "confidence": 0.95,
            },
            {
                "task": "task2",
                "gold_modality": "mandatory",
                "pred_modality": "mandatory",
                "confidence": 0.95,
            },
        ]

        self.assertEqual(
            eu.unsupported_mandatory_acceptance_rate(task1_rows, 0.90), 0.5
        )
        self.assertEqual(
            eu.high_confidence_overcommitment_rate(task1_rows, "task1", 0.90), 0.5
        )
        self.assertEqual(
            eu.task2_high_confidence_overcommitment_rate(
                task2_rows, 0.90, denominator="all"
            ),
            1 / 3,
        )
        self.assertEqual(
            eu.task2_high_confidence_overcommitment_rate(
                task2_rows, 0.90, denominator="overcommittable"
            ),
            0.5,
        )
        self.assertEqual(eu.weak_strengthening_rate(task2_rows, 0.90), 1.0)

    def test_task2_deterministic_scores_include_text_modality_fields(self):
        benchmark = eu.build_benchmark_items(export_report_seeds())
        optional_item = next(
            row for row in benchmark if row["source_modality"] == "optional"
        )
        raw_rows = [
            raw_record(
                optional_item,
                task="task2",
                parsed_json={
                    "requirement": "The system SHALL export reports.",
                    "modality": "optional",
                    "confidence": 98.0,
                },
            )
        ]

        scores = eu.build_uq_scores(benchmark, raw_rows)
        summary = eu.metric_summary_by_model_task_method(scores)

        self.assertEqual(scores[0]["text_modality"], "mandatory")
        self.assertEqual(scores[0]["text_modality_basis"], "explicit_modal")
        self.assertFalse(scores[0]["label_text_consistent"])
        self.assertTrue(scores[0]["text_overcommit"])
        self.assertTrue(scores[0]["strict_text_overcommit"])
        self.assertEqual(summary[0]["accuracy"], 1.0)
        self.assertEqual(summary[0]["text_modality_accuracy"], 0.0)
        self.assertEqual(summary[0]["text_modality_accuracy_all"], 0.0)
        self.assertEqual(summary[0]["text_modality_parse_coverage"], 1.0)
        self.assertEqual(summary[0]["strict_text_over_commitment"], 1.0)
        self.assertEqual(summary[0]["label_correct_text_overcommit_90"], 1.0)
        self.assertEqual(summary[0]["text_high_conf_overcommit_90"], 1.0)

    def test_text_modality_summary_reports_coverage_and_all_row_accuracy(self):
        rows = [
            {
                "text_modality_parse_status": "ok",
                "text_modality_correct": True,
                "label_text_consistent": True,
                "text_overcommit": False,
                "text_undercommit": False,
                "text_high_conf_overcommit_80": False,
                "text_high_conf_overcommit_90": False,
            },
            {
                "text_modality_parse_status": "unknown",
                "text_modality_correct": False,
                "label_text_consistent": False,
                "text_overcommit": False,
                "text_undercommit": False,
                "text_high_conf_overcommit_80": False,
                "text_high_conf_overcommit_90": False,
            },
        ]

        metrics = eu.text_modality_summary_metrics(rows)

        self.assertEqual(metrics["text_modality_accuracy"], 1.0)
        self.assertEqual(metrics["text_modality_accuracy_all"], 0.5)
        self.assertEqual(metrics["text_modality_parse_coverage"], 0.5)

        empty_metrics = eu.text_modality_summary_metrics(
            [{"text_modality_parse_status": ""}]
        )
        self.assertEqual(empty_metrics["text_modality_accuracy"], "")
        self.assertEqual(empty_metrics["text_modality_accuracy_all"], "")
        self.assertEqual(empty_metrics["text_modality_parse_coverage"], "")

    def test_external_evaluator_counts_label_text_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gold_path = root / "gold.csv"
            output_path = root / "outputs.jsonl"
            eu.write_csv_rows(
                gold_path,
                [
                    {
                        "external_item_id": "EXT0001",
                        "source_kind": "main_benchmark",
                        "original_item_id": "S0001_optional",
                        "seed_id": "S0001",
                        "source_condition": "optional",
                        "source_modality": "optional",
                        "task2_gold_modality": "optional",
                        "capability_text": "export reports",
                        "source_statement": "The system MAY export reports.",
                    }
                ],
            )
            output_path.write_text(
                json.dumps(
                    {
                        "external_item_id": "EXT0001",
                        "requirement": "The system SHALL export reports.",
                        "modality": "optional",
                        "confidence": 0.98,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            scored, validation = external_eval.evaluate_outputs(output_path, gold_path)
            summary = external_eval.source_condition_summary(scored)

            self.assertEqual(validation["parse_errors"], 0)
            self.assertEqual(validation["confidence_scale"], eu.CONFIDENCE_SCALE_0_1)
            self.assertTrue(validation["paper_ready"])
            self.assertIn("raw_output_sha256", validation)
            self.assertTrue(bool(scored.iloc[0]["correct"]))
            self.assertFalse(bool(scored.iloc[0]["label_text_consistent"]))
            self.assertFalse(bool(scored.iloc[0]["text_modality_correct"]))
            self.assertTrue(bool(scored.iloc[0]["text_overcommit"]))
            self.assertEqual(summary.iloc[0]["accuracy"], 1.0)
            self.assertEqual(summary.iloc[0]["text_modality_accuracy"], 0.0)

    def test_external_evaluator_rejects_duplicate_external_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gold_path = root / "gold.csv"
            output_path = root / "outputs.jsonl"
            eu.write_csv_rows(
                gold_path,
                [
                    {
                        "external_item_id": "EXT0001",
                        "source_kind": "main_benchmark",
                        "original_item_id": "S0001_optional",
                        "seed_id": "S0001",
                        "source_condition": "optional",
                        "source_modality": "optional",
                        "task2_gold_modality": "optional",
                        "capability_text": "export reports",
                        "source_statement": "The system MAY export reports.",
                    }
                ],
            )
            duplicate_row = {
                "external_item_id": "EXT0001",
                "requirement": "The system MAY export reports.",
                "modality": "optional",
                "confidence": 0.9,
            }
            output_path.write_text(
                json.dumps(duplicate_row)
                + "\n"
                + json.dumps({**duplicate_row, "confidence": 0.8})
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError, "duplicate external_item_id.*EXT0001"
            ):
                external_eval.evaluate_outputs(output_path, gold_path)

    def test_external_comparison_skips_scored_items_without_current_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows = [
                {
                    "correct": True,
                    "overcommit": False,
                    "high_conf_overcommit_90": False,
                    "text_modality_correct": True,
                    "label_text_consistency": True,
                    "text_overcommit": False,
                    "text_high_conf_overcommit_90": False,
                }
            ]
            eu.write_csv_rows(root / "current_model_scored_items.csv", rows)
            eu.write_csv_rows(root / "stale_model_scored_items.csv", rows)
            (root / "current_model_evaluation.md").write_text(
                "- Paper-ready under current contract: no\n",
                encoding="utf-8",
            )

            report_path = external_eval.write_external_comparison_report(root)

            self.assertIsNotNone(report_path)
            text = report_path.read_text(encoding="utf-8")
            self.assertIn("current_model", text)
            self.assertIn("paper_ready", text)
            self.assertNotIn("stale_model", text)

    def test_external_evaluator_uses_confidence_probability_scale(self):
        self.assertTrue(external_eval.valid_probability_confidence(0.95))
        self.assertFalse(external_eval.valid_probability_confidence(95))
        self.assertFalse(external_eval.valid_probability_confidence("95%"))

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gold_path = root / "gold.csv"
            output_path = root / "outputs.jsonl"
            eu.write_csv_rows(
                gold_path,
                [
                    {
                        "external_item_id": "EXT0001",
                        "source_kind": "main_benchmark",
                        "original_item_id": "S0001_optional",
                        "seed_id": "S0001",
                        "source_condition": "optional",
                        "source_modality": "optional",
                        "task2_gold_modality": "optional",
                        "capability_text": "export reports",
                        "source_statement": "The system MAY export reports.",
                    },
                    {
                        "external_item_id": "EXT0002",
                        "source_kind": "main_benchmark",
                        "original_item_id": "S0001_recommended",
                        "seed_id": "S0001",
                        "source_condition": "recommended",
                        "source_modality": "recommended",
                        "task2_gold_modality": "recommended",
                        "capability_text": "export reports",
                        "source_statement": "The system SHOULD export reports.",
                    },
                ],
            )
            output_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "external_item_id": "EXT0001",
                                "requirement": "The system MAY export reports.",
                                "modality": "optional",
                                "confidence": 0.95,
                            }
                        ),
                        json.dumps(
                            {
                                "external_item_id": "EXT0002",
                                "requirement": "The system SHOULD export reports.",
                                "modality": "recommended",
                                "confidence": 95,
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            scored, validation = external_eval.evaluate_outputs(output_path, gold_path)

            self.assertEqual(validation["invalid_confidence_count"], 1)
            self.assertFalse(validation["paper_ready"])
            self.assertIn("invalid_confidence", validation["paper_ready_blockers"])
            self.assertEqual(
                float(
                    scored.loc[
                        scored["external_item_id"].eq("EXT0001"), "confidence_num"
                    ].iloc[0]
                ),
                0.95,
            )

    def test_stochastic_uq_scores(self):
        benchmark = eu.build_benchmark_items(export_report_seeds())
        item = benchmark[0]
        raw_rows = [
            {
                "run_id": "r1",
                "model": "m1",
                "host": "http://localhost:8000/v1",
                "task": "task1",
                "item_id": item["item_id"],
                "seed_id": item["seed_id"],
                "source_modality": item["source_modality"],
                "sample_index": index,
                "sample_kind": "stochastic",
                "temperature": 0.7,
                "top_p": 1.0,
                "prompt_version": "v1",
                "raw_text": "",
                "parsed_json": {
                    "decision": "yes" if index < 4 else "no",
                    "confidence": 70.0,
                    "brief_reason": "",
                },
                "parse_status": "ok",
                "latency_s": 0.1,
                "error": "",
            }
            for index in range(5)
        ]
        scores = eu.build_uq_scores(benchmark, raw_rows)
        by_method = {row["uq_method"]: row for row in scores}

        self.assertEqual(
            set(by_method),
            {
                "label_self_consistency",
                "predictive_entropy",
                "variation_ratio",
                eu.ACSE_PROXY_METHOD,
            },
        )
        self.assertAlmostEqual(by_method["label_self_consistency"]["p_yes"], 0.8)
        self.assertAlmostEqual(by_method["variation_ratio"]["uncertainty_score"], 0.2)
        self.assertEqual(
            by_method["predictive_entropy"]["uncertainty_measure"], "normalized_entropy"
        )
        self.assertGreater(
            by_method["predictive_entropy"]["uncertainty_score"],
            by_method["variation_ratio"]["uncertainty_score"],
        )
        self.assertEqual(
            by_method[eu.ACSE_PROXY_METHOD]["uncertainty_measure"],
            eu.ACSE_PROXY_MEASURE,
        )
        self.assertEqual(
            by_method[eu.ACSE_PROXY_METHOD]["semantic_embedding_backend"],
            eu.ACSE_PROXY_EMBEDDING_BACKEND,
        )
        self.assertEqual(by_method[eu.ACSE_PROXY_METHOD]["valid_n"], 5)

    def test_stochastic_parse_failures_are_retained_in_score_counts(self):
        benchmark = eu.build_benchmark_items(export_report_seeds())
        item = benchmark[0]
        raw_rows = [
            {
                "run_id": "r1",
                "model": "m1",
                "host": "http://localhost:8000/v1",
                "task": "task1",
                "item_id": item["item_id"],
                "seed_id": item["seed_id"],
                "source_modality": item["source_modality"],
                "sample_index": index,
                "sample_kind": "stochastic",
                "temperature": 0.7,
                "top_p": 1.0,
                "prompt_version": "v1",
                "raw_text": "",
                "parsed_json": {
                    "decision": "yes",
                    "confidence": 70.0,
                    "brief_reason": "",
                },
                "parse_status": "ok",
                "latency_s": 0.1,
                "error": "",
            }
            for index in range(4)
        ]
        raw_rows.append(
            {
                **raw_rows[0],
                "sample_index": 4,
                "parsed_json": None,
                "parse_status": "invalid_json",
            }
        )

        scores = eu.build_uq_scores(benchmark, raw_rows)

        self.assertTrue(all(row["valid_n"] == 4 for row in scores))
        self.assertTrue(all(row["total_n"] == 5 for row in scores))
        self.assertTrue(all(row["parse_failures"] == 1 for row in scores))

    def test_ensemble_disagreement_scores_task1(self):
        benchmark = eu.build_benchmark_items(export_report_seeds())
        item = benchmark[0]
        raw_rows = []
        for model, decision in [("m1", "yes"), ("m2", "no")]:
            raw_rows.append(
                raw_record(
                    item,
                    task="task1",
                    parsed_json={
                        "decision": decision,
                        "confidence": 80.0,
                        "brief_reason": "",
                    },
                    model=model,
                )
            )

        scores = eu.build_uq_scores(benchmark, raw_rows)
        ensemble = [
            row for row in scores if row["uq_method"] == "model_ensemble_disagreement"
        ]

        self.assertEqual(len(ensemble), 1)
        self.assertEqual(ensemble[0]["model"], "ensemble:2_models")
        # 50/50 yes/no tie breaks toward the weaker "no" so it does not inflate over-commitment.
        self.assertEqual(ensemble[0]["y_pred"], 0)
        self.assertAlmostEqual(ensemble[0]["p_yes"], 0.5)
        self.assertAlmostEqual(ensemble[0]["uncertainty_score"], 0.5)

    def test_ensemble_disagreement_scores_task2(self):
        benchmark = eu.build_benchmark_items(export_report_seeds())
        item = benchmark[2]
        raw_rows = []
        for model, modality in [
            ("m1", "optional"),
            ("m2", "mandatory"),
            ("m3", "mandatory"),
        ]:
            raw_rows.append(
                raw_record(
                    item,
                    task="task2",
                    parsed_json={
                        "requirement": "The system MUST export reports.",
                        "modality": modality,
                        "confidence": 80.0,
                    },
                    model=model,
                )
            )

        scores = eu.build_uq_scores(benchmark, raw_rows)
        ensemble = [
            row for row in scores if row["uq_method"] == "model_ensemble_disagreement"
        ]

        self.assertEqual(len(ensemble), 1)
        self.assertEqual(ensemble[0]["pred_modality"], "mandatory")
        self.assertEqual(ensemble[0]["gold_modality"], "optional")
        self.assertAlmostEqual(ensemble[0]["confidence"], 2 / 3)

    def test_error_detection_auroc_uses_uncertainty_score(self):
        rows = [
            {"task": "task1", "y_true": 1, "y_pred": 1, "uncertainty_score": 0.1},
            {"task": "task1", "y_true": 0, "y_pred": 1, "uncertainty_score": 0.9},
            {"task": "task1", "y_true": 0, "y_pred": 0, "uncertainty_score": 0.2},
            {"task": "task1", "y_true": 1, "y_pred": 0, "uncertainty_score": 0.8},
        ]

        self.assertEqual(eu.error_detection_auroc(rows, "task1"), 1.0)

    def test_selective_deferral_reports_retained_error_rate(self):
        rows = [
            {"task": "task1", "y_true": 1, "y_pred": 1, "uncertainty_score": 0.1},
            {"task": "task1", "y_true": 0, "y_pred": 1, "uncertainty_score": 0.9},
            {"task": "task1", "y_true": 0, "y_pred": 0, "uncertainty_score": 0.2},
            {"task": "task1", "y_true": 1, "y_pred": 1, "uncertainty_score": 0.3},
        ]

        metrics = eu.selective_deferral_metrics(rows, "task1", defer_fractions=(0.25,))

        self.assertEqual(metrics["selective_coverage_defer_25"], 0.75)
        self.assertEqual(metrics["selective_error_defer_25"], 0.0)

    def test_headline_risk_ci_fields_cover_task1_and_task2_metrics(self):
        task1_rows = [
            {"seed_id": "S0001", "task": "task1", "y_true": 0, "p_yes": 0.95},
            {"seed_id": "S0002", "task": "task1", "y_true": 0, "p_yes": 0.40},
            {"seed_id": "S0003", "task": "task1", "y_true": 1, "p_yes": 0.99},
        ]
        task2_rows = [
            {
                "seed_id": "S0001",
                "task": "task2",
                "gold_modality": "nice_to_have",
                "pred_modality": "recommended",
                "confidence": 0.95,
            },
            {
                "seed_id": "S0002",
                "task": "task2",
                "gold_modality": "optional",
                "pred_modality": "optional",
                "confidence": 0.95,
            },
        ]

        task1_fields = eu.headline_risk_ci_fields(task1_rows, "task1", iterations=10)
        task2_fields = eu.headline_risk_ci_fields(task2_rows, "task2", iterations=10)

        self.assertIn("unsupported_mandatory_acceptance_90_ci_low", task1_fields)
        self.assertIn("unsupported_mandatory_acceptance_90_ci_high", task1_fields)
        self.assertIn("high_conf_overcommit_overcommittable_90_ci_low", task2_fields)
        self.assertIn("weak_strengthening_90_ci_high", task2_fields)

    def test_response_logprob_tokens_are_detected(self):
        response_json = {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {"token": " yes", "logprob": -0.1, "top_logprobs": []},
                        ]
                    }
                }
            ]
        }

        tokens = eu.response_logprob_tokens(response_json)

        self.assertTrue(tokens)
        self.assertEqual(tokens[0]["token"], " yes")
        self.assertEqual(tokens[0]["logprob"], -0.1)

    def test_responses_logprob_tokens_are_detected(self):
        response_json = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "yes",
                            "logprobs": [
                                {
                                    "token": "yes",
                                    "logprob": -0.2,
                                    "top_logprobs": [{"token": "no", "logprob": -2.0}],
                                },
                            ],
                        }
                    ],
                }
            ]
        }

        tokens = eu.response_logprob_tokens(response_json)

        self.assertTrue(tokens)
        self.assertEqual(eu.responses_output_text(response_json), "yes")
        self.assertEqual(tokens[0]["token"], "yes")
        self.assertEqual(tokens[0]["top_logprobs"][0]["token"], "no")

    def test_responses_endpoint_url(self):
        self.assertEqual(
            eu.responses_endpoint_url("http://localhost:1234/v1"),
            "http://localhost:1234/v1/responses",
        )
        self.assertEqual(
            eu.responses_endpoint_url("http://localhost:1234"),
            "http://localhost:1234/v1/responses",
        )
        self.assertEqual(
            eu.responses_endpoint_url("http://localhost:1234/v1/responses"),
            "http://localhost:1234/v1/responses",
        )

    def test_resolve_llm_concurrency_uses_config_and_env_override(self):
        self.assertEqual(
            eu.resolve_llm_concurrency({"llm": {"concurrency": 3}}, env={}), 3
        )
        self.assertEqual(
            eu.resolve_llm_concurrency(
                {"llm": {"concurrency": 3}}, env={"LLM_CONCURRENCY": "7"}
            ),
            7,
        )
        with self.assertRaises(ValueError):
            eu.resolve_llm_concurrency(
                {"llm": {"concurrency": 3}}, env={"LLM_CONCURRENCY": "0"}
            )
        with self.assertRaises(ValueError):
            eu.resolve_llm_concurrency({"llm": {"concurrency": "nope"}}, env={})

    def test_run_config_parsing_and_manual_server_guard(self):
        config = eu.normalize_run_config(
            {
                "run_group_id": "group1",
                "datasets": ["nice", "mlm_tapt"],
                "benchmark_variants": ["must"],
                "logging": {
                    "progress_every_records": 5,
                    "warn_parse_failure_rate": 0.1,
                },
                "profiles": [
                    {
                        "profile_id": "local",
                        "provider_id": "llama_cpp",
                        "base_url": "http://127.0.0.1:1234/v1/",
                        "api_key_env": "LOCAL_OPENAI_API_KEY",
                        "models": ["m1", "m2"],
                        "requires_manual_server": True,
                        "batch_size": 4,
                    },
                    {
                        "profile_id": "zai",
                        "provider_id": "zai",
                        "base_url": "https://api.z.ai/api/coding/paas/v4/",
                        "api_key_env": "ZAI_API_KEY",
                        "models": ["glm-5.1"],
                        "json_mode": True,
                        "extra_body": {
                            "thinking": {"type": "disabled"},
                            "response_format": {"type": "json_object"},
                        },
                    },
                ],
            }
        )

        self.assertEqual(config["profiles"][0]["base_url"], "http://127.0.0.1:1234/v1")
        self.assertEqual(config["profiles"][0]["batch_size"], 4)
        self.assertEqual(config["logging"]["progress_every_records"], 5)
        self.assertEqual(config["logging"]["warn_parse_failure_rate"], 0.1)
        self.assertEqual(
            config["profiles"][1]["base_url"], "https://api.z.ai/api/coding/paas/v4"
        )
        self.assertEqual(config["profiles"][1]["response_format"], None)
        self.assertEqual(config["profiles"][1]["structured_output"], "json_object")
        self.assertEqual(
            config["profiles"][1]["extra_body"]["thinking"]["type"], "disabled"
        )
        with self.assertRaises(ValueError):
            eu.validate_manual_server_profile(config["profiles"][0])

        selected = eu.filter_run_profiles(config, profile_id="local", model="m2")
        eu.validate_manual_server_profile(selected[0])
        self.assertEqual(selected[0]["models"], ["m2"])

    def test_load_run_config_accepts_trailing_commas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "current_run.json"
            config_path.write_text(
                """
{
  "run_group_id": "group1",
  "datasets": ["nice",],
  "benchmark_variants": ["must",],
  "profiles": [
    {
      "profile_id": "zai",
      "provider_id": "zai",
      "base_url": "https://api.z.ai/api/coding/paas/v4",
      "api_key_env": "ZAI_API_KEY",
      "models": ["glm-5.1",],
      "json_mode": true,
      "structured_output": "json_object",
    },
  ],
}
""",
                encoding="utf-8",
            )

            config = eu.load_run_config(config_path)

        self.assertEqual(config["datasets"], ["nice"])
        self.assertEqual(config["profiles"][0]["models"], ["glm-5.1"])
        self.assertEqual(config["profiles"][0]["structured_output"], "json_object")

    def test_run_logging_config_override_precedence(self):
        logging_config = eu.normalize_run_logging_config(
            {"progress_every_records": 50, "write_progress_csv": False},
            overrides={
                "progress_every_records": 10,
                "warn_after_records": 0,
                "write_event_jsonl": False,
            },
        )

        self.assertEqual(logging_config["progress_every_records"], 10)
        self.assertEqual(logging_config["warn_after_records"], 0)
        self.assertFalse(logging_config["write_progress_csv"])
        self.assertFalse(logging_config["write_event_jsonl"])

    def test_provider_request_metadata_and_extra_body_are_preserved(self):
        seeds = export_report_seeds()
        benchmark = eu.build_benchmark_items(seeds)
        captured = {}
        jobs = eu.planned_completion_jobs(
            benchmark[:1],
            tasks=["task1"],
            model="glm-5.1",
            host="https://api.z.ai/api/coding/paas/v4",
            run_id="full-1",
            prompt_version="v1",
            task1_template=eu.load_prompt("prompts/mandatory_entailment.txt"),
            task2_template=eu.load_prompt("prompts/modality_extraction.txt"),
            deterministic={"temperature": 0.0, "top_p": 1.0, "samples": 1},
            stochastic={"temperature": 0.7, "top_p": 1.0, "samples": 0},
            max_tokens=64,
            timeout_s=30,
            api_key_env="ZAI_API_KEY",
            provider_id="zai",
            profile_id="zai",
            run_group_id="group1",
            json_mode=True,
            extra_body={
                "thinking": {"type": "disabled"},
                "response_format": {"type": "json_object"},
            },
        )

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return {
                "ok": True,
                "raw_text": '{"decision": "yes", "confidence": 0.8, "brief_reason": "ok"}',
                "response_json": {},
                "latency_s": 0.01,
                "error": "",
            }

        record = eu.run_completion_job(jobs[0], completion_fn=fake_completion)

        self.assertEqual(captured["extra_body"]["thinking"]["type"], "disabled")
        self.assertEqual(record["provider_id"], "zai")
        self.assertEqual(record["profile_id"], "zai")
        self.assertEqual(record["run_group_id"], "group1")
        self.assertTrue(record["json_mode"])
        self.assertEqual(
            record["request_extra_body"]["response_format"]["type"], "json_object"
        )

    def test_provider_preflight_does_not_duplicate_extra_body_response_format(self):
        captured = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return {
                "ok": True,
                "raw_text": '{"decision": "yes", "confidence": 100, "brief_reason": "probe"}',
                "response_json": {},
                "latency_s": 0.01,
                "error": "",
            }

        preflight = eu.provider_preflight(
            host="https://api.z.ai/api/coding/paas/v4",
            model="glm-5.1",
            api_key_env="ZAI_API_KEY",
            timeout_s=30,
            json_mode=True,
            extra_body={
                "thinking": {"type": "disabled"},
                "response_format": {"type": "json_object"},
            },
            completion_fn=fake_completion,
        )

        self.assertTrue(preflight["ok"])
        self.assertIsNone(captured["response_format"])
        self.assertEqual(
            captured["extra_body"]["response_format"]["type"], "json_object"
        )

    def test_provider_preflight_rejects_v2_percentage_confidence(self):
        def run_preflight(raw_text):
            def fake_completion(**kwargs):
                return {
                    "ok": True,
                    "raw_text": raw_text,
                    "response_json": {},
                    "latency_s": 0.01,
                    "error": "",
                }

            return eu.provider_preflight(
                host="https://api.z.ai/api/coding/paas/v4",
                model="glm-5.1",
                api_key_env="ZAI_API_KEY",
                timeout_s=30,
                prompt_version="v2-conf01",
                json_mode=True,
                extra_body={"response_format": {"type": "json_object"}},
                completion_fn=fake_completion,
            )

        valid = run_preflight(
            '{"decision": "yes", "confidence": 0.8, "brief_reason": "ok"}'
        )
        self.assertTrue(valid["ok"])

        for bad_confidence in [95, 100, "95%"]:
            with self.subTest(confidence=bad_confidence):
                preflight = run_preflight(
                    json.dumps(
                        {
                            "decision": "yes",
                            "confidence": bad_confidence,
                            "brief_reason": "bad scale",
                        }
                    )
                )
                self.assertFalse(preflight["ok"])
                self.assertEqual(preflight["parse_status"], "invalid_confidence")

    def test_json_schema_structured_output_uses_task_schema(self):
        profile = eu.normalize_provider_profile(
            {
                "profile_id": "institutional",
                "provider_id": "institutional_llm",
                "base_url": "https://institutional-llm.example.invalid/api/v1",
                "api_key_env": "INSTITUTIONAL_LLM_API_KEY",
                "models": ["azure.gpt-4.1-mini"],
                "json_schema": True,
            }
        )
        seeds = export_report_seeds()
        benchmark = eu.build_benchmark_items(seeds)
        jobs = eu.planned_completion_jobs(
            benchmark[:1],
            tasks=["task2"],
            model="azure.gpt-4.1-mini",
            host=profile["base_url"],
            run_id="full-1",
            prompt_version="v1",
            task1_template=eu.load_prompt("prompts/mandatory_entailment.txt"),
            task2_template=eu.load_prompt("prompts/modality_extraction.txt"),
            deterministic={"temperature": 0.0, "top_p": 1.0, "samples": 1},
            stochastic={"temperature": 0.7, "top_p": 1.0, "samples": 0},
            max_tokens=64,
            timeout_s=30,
            api_key_env=profile["api_key_env"],
            json_mode=profile["json_mode"],
            structured_output=profile["structured_output"],
            response_format=profile["response_format"],
            extra_body=profile["extra_body"],
        )

        response_format = jobs[0]["response_format"]
        self.assertEqual(profile["structured_output"], "json_schema")
        self.assertEqual(response_format["type"], "json_schema")
        schema = response_format["json_schema"]["schema"]
        self.assertEqual(
            schema["properties"]["modality"]["enum"],
            ["mandatory", "recommended", "optional", "nice_to_have"],
        )
        self.assertTrue(response_format["json_schema"]["strict"])

    def test_json_schema_batch_uses_results_schema(self):
        seeds = [
            {
                "seed_id": "S0001",
                "source_dataset": "NICE",
                "original_requirement": "The system shall export reports.",
                "capability_text_final": "export reports",
            },
            {
                "seed_id": "S0002",
                "source_dataset": "NICE",
                "original_requirement": "The system shall print invoices.",
                "capability_text_final": "print invoices",
            },
        ]
        benchmark = eu.build_benchmark_items(seeds)
        jobs = eu.planned_completion_jobs(
            [row for row in benchmark if row["source_modality"] == "mandatory"],
            tasks=["task1"],
            model="m1",
            host="http://localhost:1234/v1",
            run_id="full-1",
            prompt_version="v1",
            task1_template=eu.load_prompt("prompts/mandatory_entailment.txt"),
            task2_template=eu.load_prompt("prompts/modality_extraction.txt"),
            deterministic={"temperature": 0.0, "top_p": 1.0, "samples": 1},
            stochastic={"temperature": 0.7, "top_p": 1.0, "samples": 0},
            max_tokens=64,
            timeout_s=30,
            api_key_env="LOCAL_OPENAI_API_KEY",
            json_mode=True,
            structured_output="json_schema",
        )
        captured = {}

        def fake_batch_completion(**kwargs):
            captured.update(kwargs)
            items = json.loads(kwargs["prompt"].split("Items:\n", 1)[1])
            return {
                "ok": True,
                "raw_text": json.dumps(
                    {
                        "results": [
                            {
                                "request_index": item["request_index"],
                                "decision": "yes",
                                "confidence": 0.8,
                                "brief_reason": "batched",
                            }
                            for item in items
                        ]
                    }
                ),
                "response_json": {},
                "latency_s": 0.01,
                "error": "",
            }

        records = list(
            eu.run_completion_jobs(
                jobs, max_workers=1, completion_fn=fake_batch_completion, batch_size=2
            )
        )

        schema = captured["response_format"]["json_schema"]["schema"]
        self.assertIn("results", schema["properties"])
        self.assertIn(
            "request_index", schema["properties"]["results"]["items"]["properties"]
        )
        self.assertEqual({row["structured_output"] for row in records}, {"json_schema"})
        self.assertEqual(
            {row["response_format"]["json_schema"]["name"] for row in records},
            {"re_uq_task1_batch"},
        )

    def test_json_schema_replaces_extra_body_response_format(self):
        response_format, extra_body = eu.resolve_response_format_args(
            "task1",
            structured_output="json_schema",
            extra_body={
                "thinking": {"type": "disabled"},
                "response_format": {"type": "json_object"},
            },
        )

        self.assertIsNone(response_format)
        self.assertEqual(extra_body["thinking"]["type"], "disabled")
        self.assertEqual(extra_body["response_format"]["type"], "json_schema")

    def test_instructor_profile_strips_response_format_and_sets_defaults(self):
        profile = eu.normalize_provider_profile(
            {
                "profile_id": "zai",
                "provider_id": "zai",
                "base_url": "https://api.z.ai/api/coding/paas/v4",
                "models": ["glm-5.1"],
                "structured_output": "instructor",
                "extra_body": {
                    "thinking": {"type": "disabled"},
                    "response_format": {"type": "json_object"},
                },
            }
        )

        self.assertEqual(profile["structured_output"], "instructor")
        self.assertEqual(profile["instructor_mode"], "json")
        self.assertEqual(profile["validation_retries"], 2)
        self.assertEqual(profile["fallback_batch_size"], 1)
        self.assertEqual(profile["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertIsNone(profile["response_format"])

    def test_instructor_completion_passes_response_model_retries_mode_and_clean_extra_body(
        self,
    ):
        import instructor

        captured = {}

        class ParsedResponse:
            def model_dump(self, mode):
                captured["dump_mode"] = mode
                return {"decision": "yes", "confidence": 0.82, "brief_reason": "valid"}

        class DummyCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return ParsedResponse()

        class DummyChat:
            completions = DummyCompletions()

        class DummyInstructorClient:
            chat = DummyChat()

        with (
            mock.patch("scripts.eval_utils.OpenAI") as openai_cls,
            mock.patch(
                "instructor.from_openai", return_value=DummyInstructorClient()
            ) as from_openai,
        ):
            result = eu.instructor_completion(
                host="http://localhost:1234/v1",
                model="m1",
                prompt="prompt",
                temperature=0.0,
                top_p=1.0,
                max_tokens=64,
                timeout_s=30,
                api_key_env="LOCAL_OPENAI_API_KEY",
                response_format={"type": "json_object"},
                extra_body={
                    "thinking": {"type": "disabled"},
                    "response_format": {"type": "json_object"},
                },
                task="task1",
                instructor_mode="json",
                validation_retries=3,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(json.loads(result["raw_text"])["confidence"], 0.82)
        self.assertEqual(captured["response_model"], so.Task1Response)
        self.assertEqual(captured["max_retries"], 3)
        self.assertEqual(captured["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertNotIn("response_format", captured)
        self.assertEqual(captured["dump_mode"], "json")
        openai_cls.assert_called_once()
        self.assertEqual(from_openai.call_args.kwargs["mode"], instructor.Mode.JSON)

    def test_instructor_completion_validation_retry_failure_is_marked_for_parser(self):
        class InstructorValidationError(Exception):
            pass

        class DummyCompletions:
            def create(self, **kwargs):
                exc = InstructorValidationError("bad model output")
                exc.last_completion = (
                    '{"decision":"yes","confidence":95,"brief_reason":"bad scale"}'
                )
                raise exc

        class DummyChat:
            completions = DummyCompletions()

        class DummyInstructorClient:
            chat = DummyChat()

        with (
            mock.patch("scripts.eval_utils.OpenAI"),
            mock.patch("instructor.from_openai", return_value=DummyInstructorClient()),
        ):
            result = eu.instructor_completion(
                host="http://localhost:1234/v1",
                model="m1",
                prompt="prompt",
                temperature=0.0,
                top_p=1.0,
                task="task1",
                validation_retries=2,
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["parse_status_override"], "instructor_validation_error")
        self.assertIn("confidence", result["raw_text"])

    def test_instructor_single_item_writes_validated_contract_markers(self):
        jobs = self._instructor_task1_jobs(seed_count=1, validation_retries=2)
        captured = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return {
                "ok": True,
                "raw_text": '{"decision": "yes", "confidence": 0.8, "brief_reason": "ok"}',
                "response_json": {},
                "latency_s": 0.01,
                "error": "",
            }

        record = eu.run_completion_job(jobs[0], completion_fn=fake_completion)

        self.assertEqual(captured["task"], "task1")
        self.assertFalse(captured["batched"])
        self.assertEqual(captured["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertEqual(record["parse_status"], "ok")
        self.assertEqual(record["parsed_json"]["confidence"], 0.8)
        self.assertEqual(
            record["output_contract_version"], so.INSTRUCTOR_OUTPUT_CONTRACT_VERSION
        )
        self.assertEqual(record["confidence_scale"], so.INSTRUCTOR_CONFIDENCE_SCALE)

    def test_instructor_batch_partial_results_fall_back_unbatched(self):
        jobs = self._instructor_task1_jobs(
            seed_count=2, extra_body=None, validation_retries=2
        )
        calls = []

        def fake_completion(**kwargs):
            calls.append(kwargs)
            if kwargs["batched"]:
                items = json.loads(kwargs["prompt"].split("Items:\n", 1)[1])
                return {
                    "ok": True,
                    "raw_text": json.dumps(
                        {
                            "results": [
                                {
                                    "request_index": items[0]["request_index"],
                                    "decision": "yes",
                                    "confidence": 0.8,
                                    "brief_reason": "batched",
                                }
                            ]
                        }
                    ),
                    "response_json": {},
                    "latency_s": 0.01,
                    "error": "",
                }
            return {
                "ok": True,
                "raw_text": '{"decision": "yes", "confidence": 0.9, "brief_reason": "fallback"}',
                "response_json": {},
                "latency_s": 0.01,
                "error": "",
            }

        records = list(
            eu.run_completion_jobs(
                jobs, max_workers=1, completion_fn=fake_completion, batch_size=2
            )
        )

        self.assertEqual([call["batched"] for call in calls], [True, False])
        self.assertEqual(len(records), 2)
        self.assertEqual({record["parse_status"] for record in records}, {"ok"})
        self.assertEqual(records[0]["parsed_json"]["confidence"], 0.8)
        self.assertEqual(records[1]["parsed_json"]["confidence"], 0.9)

    def test_instructor_batch_invalid_item_falls_back_unbatched(self):
        jobs = self._instructor_task1_jobs(seed_count=2)
        calls = []

        def fake_completion(**kwargs):
            calls.append(kwargs)
            if kwargs["batched"]:
                items = json.loads(kwargs["prompt"].split("Items:\n", 1)[1])
                return {
                    "ok": True,
                    "raw_text": json.dumps(
                        {
                            "results": [
                                {
                                    "request_index": items[0]["request_index"],
                                    "decision": "yes",
                                    "confidence": 0.8,
                                    "brief_reason": "valid batch item",
                                },
                                {
                                    "request_index": items[1]["request_index"],
                                    "decision": "yes",
                                    "confidence": 95,
                                    "brief_reason": "bad scale",
                                },
                            ]
                        }
                    ),
                    "response_json": {},
                    "latency_s": 0.01,
                    "error": "",
                }
            return {
                "ok": True,
                "raw_text": '{"decision": "yes", "confidence": 0.91, "brief_reason": "fallback"}',
                "response_json": {},
                "latency_s": 0.01,
                "error": "",
            }

        records = list(
            eu.run_completion_jobs(
                jobs, max_workers=1, completion_fn=fake_completion, batch_size=2
            )
        )

        self.assertEqual([call["batched"] for call in calls], [True, False])
        self.assertEqual({record["parse_status"] for record in records}, {"ok"})
        self.assertEqual(records[0]["parsed_json"]["confidence"], 0.8)
        self.assertEqual(records[1]["parsed_json"]["confidence"], 0.91)

    def test_instructor_batch_unknown_request_index_falls_back_unbatched(self):
        jobs = self._instructor_task1_jobs(seed_count=2)
        calls = []

        def fake_completion(**kwargs):
            calls.append(kwargs)
            if kwargs["batched"]:
                return {
                    "ok": True,
                    "raw_text": json.dumps(
                        {
                            "results": [
                                {
                                    "request_index": 999,
                                    "decision": "yes",
                                    "confidence": 0.8,
                                    "brief_reason": "unknown index",
                                }
                            ]
                        }
                    ),
                    "response_json": {},
                    "latency_s": 0.01,
                    "error": "",
                }
            return {
                "ok": True,
                "raw_text": '{"decision": "yes", "confidence": 0.77, "brief_reason": "fallback"}',
                "response_json": {},
                "latency_s": 0.01,
                "error": "",
            }

        records = list(
            eu.run_completion_jobs(
                jobs, max_workers=1, completion_fn=fake_completion, batch_size=2
            )
        )

        self.assertEqual([call["batched"] for call in calls], [True, False, False])
        self.assertEqual({record["parse_status"] for record in records}, {"ok"})
        self.assertEqual(
            [record["parsed_json"]["confidence"] for record in records], [0.77, 0.77]
        )

    def test_instructor_batch_duplicate_request_index_falls_back_unbatched(self):
        jobs = self._instructor_task1_jobs(seed_count=2)
        calls = []

        def fake_completion(**kwargs):
            calls.append(kwargs)
            if kwargs["batched"]:
                items = json.loads(kwargs["prompt"].split("Items:\n", 1)[1])
                duplicate_index = items[0]["request_index"]
                return {
                    "ok": True,
                    "raw_text": json.dumps(
                        {
                            "results": [
                                {
                                    "request_index": duplicate_index,
                                    "decision": "yes",
                                    "confidence": 0.8,
                                    "brief_reason": "first copy",
                                },
                                {
                                    "request_index": duplicate_index,
                                    "decision": "yes",
                                    "confidence": 0.9,
                                    "brief_reason": "duplicate copy",
                                },
                            ]
                        }
                    ),
                    "response_json": {},
                    "latency_s": 0.01,
                    "error": "",
                }
            return {
                "ok": True,
                "raw_text": '{"decision": "yes", "confidence": 0.76, "brief_reason": "fallback"}',
                "response_json": {},
                "latency_s": 0.01,
                "error": "",
            }

        records = list(
            eu.run_completion_jobs(
                jobs, max_workers=1, completion_fn=fake_completion, batch_size=2
            )
        )

        parsed_results, parse_status = eu.parse_batch_completion_results(
            json.dumps(
                {
                    "results": [
                        {
                            "request_index": 1,
                            "decision": "yes",
                            "confidence": 0.8,
                            "brief_reason": "a",
                        },
                        {
                            "request_index": 1,
                            "decision": "yes",
                            "confidence": 0.9,
                            "brief_reason": "b",
                        },
                    ]
                }
            )
        )
        self.assertEqual(parse_status, "duplicate_request_index")
        self.assertEqual(set(parsed_results), {1})
        self.assertEqual([call["batched"] for call in calls], [True, False, False])
        self.assertEqual({record["parse_status"] for record in records}, {"ok"})
        self.assertEqual(
            [record["parsed_json"]["confidence"] for record in records], [0.76, 0.76]
        )

    def test_instructor_malformed_batch_and_failed_fallback_remain_pending(self):
        jobs = self._instructor_task1_jobs(seed_count=2)
        calls = []

        def fake_completion(**kwargs):
            calls.append(kwargs)
            if kwargs["batched"]:
                return {
                    "ok": True,
                    "raw_text": "not json",
                    "response_json": {},
                    "latency_s": 0.01,
                    "error": "",
                }
            return {
                "ok": True,
                "raw_text": '{"decision": "yes", "confidence": 95, "brief_reason": "bad scale"}',
                "response_json": {},
                "latency_s": 0.01,
                "error": "",
            }

        records = list(
            eu.run_completion_jobs(
                jobs, max_workers=1, completion_fn=fake_completion, batch_size=2
            )
        )

        self.assertEqual([call["batched"] for call in calls], [True, False, False])
        self.assertEqual(
            {record["parse_status"] for record in records},
            {"instructor_validation_error"},
        )
        self.assertEqual(
            len(eu.pending_completion_jobs(jobs, records, "full-1")), len(jobs)
        )

    def test_instructor_failed_fallback_stays_pending(self):
        jobs = self._instructor_task1_jobs(
            seed_count=1, extra_body=None, validation_retries=2
        )

        def fake_completion(**kwargs):
            return {
                "ok": True,
                "raw_text": '{"decision": "yes", "confidence": 95, "brief_reason": "bad scale"}',
                "response_json": {},
                "latency_s": 0.01,
                "error": "",
            }

        records = list(
            eu.run_completion_jobs(
                jobs, max_workers=1, completion_fn=fake_completion, batch_size=1
            )
        )

        self.assertEqual(records[0]["parse_status"], "instructor_validation_error")
        self.assertEqual(len(eu.pending_completion_jobs(jobs, records, "full-1")), 1)

    def test_zai_example_uses_coding_plan_endpoint(self):
        config = eu.load_run_config("run_configs/full_matrix.example.json")
        zai_profile = next(
            profile for profile in config["profiles"] if profile["profile_id"] == "zai"
        )

        self.assertEqual(zai_profile["base_url"], "https://api.z.ai/api/coding/paas/v4")
        self.assertIn("Coding Plan", zai_profile["notes"])

    def test_batched_completion_splits_results_to_raw_records(self):
        seeds = [
            {
                "seed_id": "S0001",
                "source_dataset": "NICE",
                "original_requirement": "The system shall export reports.",
                "capability_text_final": "export reports",
            },
            {
                "seed_id": "S0002",
                "source_dataset": "NICE",
                "original_requirement": "The system shall print invoices.",
                "capability_text_final": "print invoices",
            },
        ]
        benchmark = eu.build_benchmark_items(seeds)
        jobs = eu.planned_completion_jobs(
            [row for row in benchmark if row["source_modality"] == "mandatory"],
            tasks=["task1"],
            model="m1",
            host="http://localhost:1234/v1",
            run_id="full-1",
            prompt_version="v1",
            task1_template=eu.load_prompt("prompts/mandatory_entailment.txt"),
            task2_template=eu.load_prompt("prompts/modality_extraction.txt"),
            deterministic={"temperature": 0.0, "top_p": 1.0, "samples": 1},
            stochastic={"temperature": 0.7, "top_p": 1.0, "samples": 0},
            max_tokens=64,
            timeout_s=30,
            api_key_env="LOCAL_OPENAI_API_KEY",
        )
        calls = []

        def fake_batch_completion(**kwargs):
            calls.append(kwargs)
            items = json.loads(kwargs["prompt"].split("Items:\n", 1)[1])
            return {
                "ok": True,
                "raw_text": json.dumps(
                    {
                        "results": [
                            {
                                "request_index": item["request_index"],
                                "decision": "yes",
                                "confidence": 0.8,
                                "brief_reason": "batched",
                            }
                            for item in items
                        ]
                    }
                ),
                "response_json": {"batched": True},
                "latency_s": 0.01,
                "error": "",
            }

        records = list(
            eu.run_completion_jobs(
                jobs, max_workers=1, completion_fn=fake_batch_completion, batch_size=2
            )
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["max_tokens"], 128)
        self.assertEqual(len(records), 2)
        self.assertEqual({row["parse_status"] for row in records}, {"ok"})
        self.assertEqual(len({row["batch_id"] for row in records}), 1)
        self.assertTrue(all(row["batch_size"] == 2 for row in records))
        self.assertTrue(all(row.get("job_config_sha") for row in records))
        self.assertEqual(len(eu.pending_completion_jobs(jobs, records, "full-1")), 0)

    def test_batched_completion_missing_results_fall_back_to_single_item_requests(self):
        """An unusable batch response is re-sent per item, mirroring the Instructor path."""
        seeds = [
            {
                "seed_id": "S0001",
                "source_dataset": "NICE",
                "original_requirement": "The system shall export reports.",
                "capability_text_final": "export reports",
            },
            {
                "seed_id": "S0002",
                "source_dataset": "NICE",
                "original_requirement": "The system shall print invoices.",
                "capability_text_final": "print invoices",
            },
        ]
        benchmark = eu.build_benchmark_items(seeds)
        jobs = eu.planned_completion_jobs(
            [row for row in benchmark if row["source_modality"] == "mandatory"],
            tasks=["task1"],
            model="m1",
            host="http://localhost:1234/v1",
            run_id="full-1",
            prompt_version="v1",
            task1_template=eu.load_prompt("prompts/mandatory_entailment.txt"),
            task2_template=eu.load_prompt("prompts/modality_extraction.txt"),
            deterministic={"temperature": 0.0, "top_p": 1.0, "samples": 1},
            stochastic={"temperature": 0.7, "top_p": 1.0, "samples": 0},
            max_tokens=64,
            timeout_s=30,
            api_key_env="LOCAL_OPENAI_API_KEY",
        )

        prompts = []

        def fake_ignored_batch_completion(**kwargs):
            prompts.append(kwargs["prompt"])
            if "Items:\n" in kwargs["prompt"]:
                # A batch answer that carries no per-item results at all.
                return {
                    "ok": True,
                    "raw_text": '{"decision": "yes", "confidence": 0.8, "brief_reason": "ignored the batch"}',
                    "response_json": {},
                    "latency_s": 0.01,
                    "error": "",
                }
            return {
                "ok": True,
                "raw_text": '{"decision": "yes", "confidence": 0.9, "brief_reason": "single"}',
                "response_json": {},
                "latency_s": 0.01,
                "error": "",
            }

        records = list(
            eu.run_completion_jobs(
                jobs,
                max_workers=1,
                completion_fn=fake_ignored_batch_completion,
                batch_size=2,
            )
        )

        # One batch call plus one single-item call per item of that batch.
        self.assertEqual(len(prompts), 1 + len(jobs))
        self.assertEqual({row["parse_status"] for row in records}, {"ok"})
        self.assertEqual(
            {row["parsed_json"]["brief_reason"] for row in records}, {"single"}
        )
        # The rows were really sent alone: batch_size 1 and no batch_id.
        self.assertEqual({row["batch_size"] for row in records}, {1})
        self.assertTrue(all("batch_id" not in row for row in records))
        self.assertEqual(eu.pending_completion_jobs(jobs, records, "full-1"), [])

    def test_batched_completion_keeps_failing_when_the_fallback_also_fails(self):
        jobs = self._task1_two_seed_jobs()

        def fake_broken_completion(**kwargs):
            if "Items:\n" in kwargs["prompt"]:
                return {
                    "ok": True,
                    "raw_text": "not json at all",
                    "response_json": {},
                    "latency_s": 0.01,
                    "error": "",
                }
            return {
                "ok": True,
                "raw_text": "still not json",
                "response_json": {},
                "latency_s": 0.01,
                "error": "",
            }

        records = list(
            eu.run_completion_jobs(
                jobs, max_workers=1, completion_fn=fake_broken_completion, batch_size=2
            )
        )

        self.assertEqual({row["parse_status"] for row in records}, {"invalid_json"})
        self.assertTrue(all(row["parsed_json"] is None for row in records))
        # The fallback's own status stands, but the batch-level cause is kept.
        self.assertTrue(
            all(row["error"].startswith("batch_fallback:") for row in records)
        )
        self.assertTrue(all("missing request_index" in row["error"] for row in records))
        self.assertEqual(
            len(eu.pending_completion_jobs(jobs, records, "full-1")), len(jobs)
        )

    def test_batched_completion_falls_back_when_the_batch_request_errors(self):
        jobs = self._task1_two_seed_jobs()

        def fake_failing_batch_completion(**kwargs):
            if "Items:\n" in kwargs["prompt"]:
                return {
                    "ok": False,
                    "raw_text": "",
                    "response_json": None,
                    "latency_s": 0.01,
                    "error": "RateLimitError(429)",
                }
            return {
                "ok": True,
                "raw_text": '{"decision": "yes", "confidence": 0.9, "brief_reason": "single"}',
                "response_json": {},
                "latency_s": 0.01,
                "error": "",
            }

        records = list(
            eu.run_completion_jobs(
                jobs,
                max_workers=1,
                completion_fn=fake_failing_batch_completion,
                batch_size=2,
            )
        )

        self.assertEqual({row["parse_status"] for row in records}, {"ok"})
        self.assertEqual({row["batch_size"] for row in records}, {1})
        self.assertEqual(eu.pending_completion_jobs(jobs, records, "full-1"), [])

    def test_batched_completion_count_matches_but_no_index_overlap_is_not_positional(
        self,
    ):
        jobs = self._task1_two_seed_jobs()

        def fake_no_index_batch_completion(**kwargs):
            if "Items:\n" not in kwargs["prompt"]:
                return {
                    "ok": True,
                    "raw_text": '{"decision": "no", "confidence": 0.6, "brief_reason": "fallback"}',
                    "response_json": {},
                    "latency_s": 0.01,
                    "error": "",
                }
            items = json.loads(kwargs["prompt"].split("Items:\n", 1)[1])
            # Right count, but each result omits request_index (parse assigns
            # synthetic negative indices -> no overlap with expected indices).
            return {
                "ok": True,
                "raw_text": json.dumps(
                    {
                        "results": [
                            {
                                "decision": "yes",
                                "confidence": 0.8,
                                "brief_reason": "batched",
                            }
                            for _ in items
                        ]
                    }
                ),
                "response_json": {"batched": True},
                "latency_s": 0.01,
                "error": "",
            }

        records = list(
            eu.run_completion_jobs(
                jobs,
                max_workers=1,
                completion_fn=fake_no_index_batch_completion,
                batch_size=2,
            )
        )

        self.assertEqual(len(records), 2)
        # Index-less batch results are never matched positionally: every row is
        # the single-item fallback answer, not the batch's first result.
        self.assertEqual({row["parse_status"] for row in records}, {"ok"})
        self.assertEqual(
            {row["parsed_json"]["brief_reason"] for row in records}, {"fallback"}
        )
        self.assertEqual({row["parsed_json"]["decision"] for row in records}, {"no"})
        self.assertEqual({row["batch_size"] for row in records}, {1})

    def test_resume_reruns_on_config_sha_mismatch_and_reuses_on_match(self):
        job = {
            "run_id": "full-1",
            "model": "m1",
            "task": "task1",
            "item_id": "S0001_mandatory",
            "sample_kind": "deterministic",
            "sample_index": 0,
            "job_config_sha": "sha-new",
        }
        matching = [{**job, "parse_status": "ok"}]
        mismatching = [{**job, "parse_status": "ok", "job_config_sha": "sha-old"}]

        self.assertEqual(eu.pending_completion_jobs([job], matching, "full-1"), [])
        self.assertEqual(
            len(eu.pending_completion_jobs([job], mismatching, "full-1")), 1
        )

    def test_resume_reuses_legacy_row_without_config_sha_with_warning(self):
        job = {
            "run_id": "full-1",
            "model": "m1",
            "task": "task1",
            "item_id": "S0001_mandatory",
            "sample_kind": "deterministic",
            "sample_index": 0,
            "job_config_sha": "sha-new",
        }
        legacy = [
            {
                "run_id": "full-1",
                "model": "m1",
                "task": "task1",
                "item_id": "S0001_mandatory",
                "sample_kind": "deterministic",
                "sample_index": 0,
                "parse_status": "ok",
            }
        ]

        with self.assertLogs("re_uq", level="WARNING") as captured:
            pending = eu.pending_completion_jobs([job], legacy, "full-1")

        self.assertEqual(pending, [])
        self.assertIn("job_config_sha", "\n".join(captured.output))

    def test_resume_planning_skips_completed_records(self):
        job = {
            "run_id": "full-1",
            "model": "m1",
            "task": "task1",
            "item_id": "S0001_mandatory",
            "sample_kind": "deterministic",
            "sample_index": 0,
        }
        completed = [{**job, "parse_status": "ok"}]
        failed = [{**job, "parse_status": "request_error"}]

        self.assertEqual(eu.pending_completion_jobs([job], completed, "full-1"), [])
        self.assertEqual(len(eu.pending_completion_jobs([job], failed, "full-1")), 1)

    def test_run_registry_summary_and_upsert(self):
        benchmark = eu.build_benchmark_items(export_report_seeds())[:1]
        raw_rows = []
        for task in ["task1", "task2"]:
            item = benchmark[0]
            raw_rows.append(
                {
                    "run_id": "full-1",
                    "model": "m1",
                    "task": task,
                    "item_id": item["item_id"],
                    "seed_id": item["seed_id"],
                    "source_modality": item["source_modality"],
                    "sample_kind": "deterministic",
                    "sample_index": 0,
                    "parse_status": "ok",
                    "parsed_json": {"decision": "yes", "confidence": 90}
                    if task == "task1"
                    else {
                        "requirement": "The system MUST export reports.",
                        "modality": "mandatory",
                        "confidence": 90,
                    },
                    "prompt": item["source_statement"],
                    "prompt_version": "v1",
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "job_config_sha": f"sha-{task}",
                }
            )
        row = eu.run_registry_summary(
            benchmark,
            raw_rows,
            run_id="full-1",
            run_group_id="group1",
            provider_id="local",
            profile_id="local",
            model="m1",
            dataset_id="nice",
            variant="must",
            tasks=["task1", "task2"],
            expected_stochastic_samples=0,
            started_at_utc="2026-05-21T00:00:00Z",
            finished_at_utc="2026-05-21T00:01:00Z",
        )

        self.assertEqual(row["status"], "complete")
        self.assertEqual(row["expected_records"], 2)
        self.assertEqual(row["observed_records"], 2)
        self.assertEqual(row["prompt_version"], "v1")
        self.assertEqual(row["temperature"], 0.0)
        self.assertEqual(row["top_p"], 1.0)
        self.assertTrue(row["config_sha"])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "run_registry.csv"
            eu.upsert_run_registry_row(path, row)
            eu.upsert_run_registry_row(path, {**row, "status": "partial"})
            rows = eu.read_csv_rows(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "partial")
            self.assertEqual(rows[0]["prompt_version"], "v1")
            self.assertEqual(rows[0]["config_sha"], row["config_sha"])

    def test_live_run_counters_and_warning_events(self):
        raw_rows = [
            {"run_id": "r1", "request_index": 0, "parse_status": "ok"},
            {"run_id": "r1", "request_index": 1, "parse_status": "invalid_json"},
            {"run_id": "r1", "request_index": 2, "parse_status": "request_error"},
        ]

        counters = eu.live_run_counters(
            raw_rows,
            expected_records=10,
            expected_api_calls=10,
            started_monotonic=0.0,
            now_monotonic=3.0,
        )
        early_warnings = eu.warning_events_for_counters(
            counters,
            {
                "warn_after_records": 4,
                "warn_parse_failure_rate": 0.1,
                "warn_request_error_rate": 0.1,
            },
            set(),
        )
        warnings = eu.warning_events_for_counters(
            counters,
            {
                "warn_after_records": 3,
                "warn_parse_failure_rate": 0.1,
                "warn_request_error_rate": 0.1,
            },
            set(),
        )

        self.assertEqual(counters["observed_records"], 3)
        self.assertEqual(counters["ok_records"], 1)
        self.assertEqual(counters["request_error_records"], 1)
        self.assertEqual(counters["records_per_s"], 1.0)
        self.assertEqual(early_warnings, [])
        self.assertEqual(
            {warning["warning_type"] for warning in warnings},
            {"parse_failure_rate", "request_error_rate"},
        )

    def test_run_event_jsonl_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "events.jsonl"
            eu.append_run_event(path, {"event_type": "start", "run_id": "r1"})
            eu.append_run_event(
                path, {"event_type": "finish", "run_id": "r1", "observed_records": 1}
            )

            rows = eu.read_jsonl(path)
            self.assertEqual([row["event_type"] for row in rows], ["start", "finish"])
            self.assertTrue(all("created_at_utc" in row for row in rows))

    def test_run_group_ensemble_disagreement_across_run_ids(self):
        benchmark = eu.build_benchmark_items(export_report_seeds())
        item = benchmark[0]
        raw_rows = [
            {
                "run_id": "full-a",
                "run_group_id": "group1",
                "provider_id": "p1",
                "model": "m1",
                "task": "task2",
                "item_id": item["item_id"],
                "seed_id": item["seed_id"],
                "source_modality": item["source_modality"],
                "sample_kind": "deterministic",
                "sample_index": 0,
                "parse_status": "ok",
                "parsed_json": {
                    "requirement": item["source_statement"],
                    "modality": "mandatory",
                    "confidence": 90,
                },
                "prompt": item["source_statement"],
            },
            {
                "run_id": "full-b",
                "run_group_id": "group1",
                "provider_id": "p2",
                "model": "m2",
                "task": "task2",
                "item_id": item["item_id"],
                "seed_id": item["seed_id"],
                "source_modality": item["source_modality"],
                "sample_kind": "deterministic",
                "sample_index": 0,
                "parse_status": "ok",
                "parsed_json": {
                    "requirement": item["source_statement"],
                    "modality": "recommended",
                    "confidence": 90,
                },
                "prompt": item["source_statement"],
            },
        ]

        scores = eu.build_run_group_ensemble_disagreement_scores(
            benchmark, raw_rows, run_group_id="group1"
        )

        self.assertEqual(len(scores), 1)
        self.assertEqual(
            scores[0]["uq_method"], "model_ensemble_disagreement_run_group"
        )
        self.assertEqual(scores[0]["valid_n"], 2)

    def test_run_matrix_completed_rows_filter_excludes_smoke_by_default(self):
        rows = [
            {"run_group_id": "group1", "run_id": "full-1", "status": "complete"},
            {"run_group_id": "group1", "run_id": "smoke-1", "status": "complete"},
            {"run_group_id": "group1", "run_id": "full-2", "status": "partial"},
            {"run_group_id": "other", "run_id": "full-3", "status": "complete"},
        ]

        selected = compare_matrix.completed_registry_rows(
            rows, "group1", include_smoke=False
        )
        selected_with_smoke = compare_matrix.completed_registry_rows(
            rows, "group1", include_smoke=True
        )

        self.assertEqual([row["run_id"] for row in selected], ["full-1"])
        self.assertEqual(
            [row["run_id"] for row in selected_with_smoke], ["full-1", "smoke-1"]
        )

    def test_run_matrix_completed_rows_can_exclude_model_prefixes(self):
        rows = [
            {
                "run_group_id": "group1",
                "run_id": "full-1",
                "status": "complete",
                "model": "local.gemma4-31b-it",
            },
            {
                "run_group_id": "group1",
                "run_id": "full-2",
                "status": "complete",
                "model": "azure.gpt-5.4",
            },
            {
                "run_group_id": "group1",
                "run_id": "full-3",
                "status": "complete",
                "model": "glm-5",
            },
        ]

        selected = compare_matrix.completed_registry_rows(
            rows,
            "group1",
            include_smoke=False,
            exclude_model_prefixes=["azure."],
        )

        self.assertEqual([row["run_id"] for row in selected], ["full-1", "full-3"])

    def test_fake_cli_smoke_writes_canonical_jsonl_and_registry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _scaffold_project_root(root)
            (root / "prompts/mandatory_entailment.txt").write_text(
                Path("prompts/mandatory_entailment.txt").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (root / "prompts/modality_extraction.txt").write_text(
                Path("prompts/modality_extraction.txt").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            benchmark = eu.build_benchmark_items(
                [
                    {
                        "seed_id": "S0001",
                        "source_dataset": "NICE",
                        "original_requirement": "The system shall export reports.",
                        "capability_text_final": "export reports",
                    },
                    {
                        "seed_id": "S0002",
                        "source_dataset": "NICE",
                        "original_requirement": "The system shall print invoices.",
                        "capability_text_final": "print invoices",
                    },
                ]
            )
            eu.write_csv_rows(root / "data/processed/benchmark_items.csv", benchmark)
            config_path = root / "run_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "run_group_id": "smoke-group",
                        "datasets": ["nice"],
                        "benchmark_variants": ["must"],
                        "stochastic": {"temperature": 0.7, "top_p": 1.0, "samples": 1},
                        "logging": {
                            "progress_every_records": 2,
                            "progress_every_seconds": 999,
                            "warn_after_records": 2,
                        },
                        "profiles": [
                            {
                                "profile_id": "fake",
                                "provider_id": "fake",
                                "base_url": "http://127.0.0.1:1234/v1",
                                "api_key_env": "LOCAL_OPENAI_API_KEY",
                                "models": ["fake-model"],
                                "concurrency": 1,
                                "batch_size": 2,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            argv = [
                "run_experiment_from_config.py",
                "--config",
                str(config_path),
                "--profile",
                "fake",
                "--mode",
                "smoke",
                "--smoke-items",
                "2",
                "--fake-completion",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(run_config_cli.eu, "project_root", return_value=root),
                redirect_stdout(io.StringIO()),
            ):
                run_config_cli.main()

            # Fake/smoke runs are isolated from the paper-facing artifact tree.
            self.assertFalse((root / "data/processed/model_outputs_raw.jsonl").exists())
            rows = eu.read_jsonl(root / "data/processed/smoke/model_outputs_raw.jsonl")
            registry = eu.read_csv_rows(root / "data/processed/smoke/run_registry.csv")
            progress = eu.read_csv_rows(
                root / "data/processed/smoke/run_progress_live.csv"
            )
            events = eu.read_jsonl(root / "data/processed/smoke/run_events.jsonl")
            self.assertEqual(len(rows), 8)
            self.assertEqual(len({row["batch_id"] for row in rows}), 4)
            self.assertEqual(registry[0]["status"], "complete")
            self.assertEqual(registry[0]["run_group_id"], "smoke-group")
            self.assertEqual(registry[0]["batch_size"], "2")
            self.assertEqual(registry[0]["expected_api_calls"], "4")
            self.assertEqual(registry[0]["observed_api_calls"], "4")
            self.assertEqual({row["task"] for row in progress}, {"task1", "task2"})
            self.assertEqual(
                {row["event_type"] for row in events}, {"start", "progress", "finish"}
            )
            finish_event = next(
                row for row in reversed(events) if row["event_type"] == "finish"
            )
            self.assertEqual(finish_event["pending_jobs"], 0)
            self.assertEqual(finish_event["pending_api_calls"], 0)

    def test_task3_cli_fake_run_writes_diagnostic_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _scaffold_project_root(root)
            (root / "prompts/modality_verification.txt").write_text(
                Path("prompts/modality_verification.txt").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            benchmark = eu.build_benchmark_items(
                [
                    {
                        "seed_id": "S0001",
                        "source_dataset": "NICE",
                        "original_requirement": "The system shall export reports.",
                        "capability_text_final": "export reports",
                    },
                    {
                        "seed_id": "S0002",
                        "source_dataset": "NICE",
                        "original_requirement": "The system shall print invoices.",
                        "capability_text_final": "print invoices",
                    },
                ]
            )
            eu.write_csv_rows(root / "data/processed/benchmark_items.csv", benchmark)
            for item in benchmark:
                eu.append_jsonl(
                    root / "data/processed/model_outputs_raw.jsonl",
                    eu.build_raw_record(
                        run_id="full-source",
                        model="fake-model",
                        host="http://127.0.0.1:1234/v1",
                        task="task2",
                        item=item,
                        sample_index=0,
                        sample_kind="deterministic",
                        temperature=0.0,
                        top_p=1.0,
                        prompt_version="v2-conf01",
                        prompt=item["source_statement"],
                        completion={
                            "ok": True,
                            "raw_text": json.dumps(
                                {
                                    "requirement": item["source_statement"],
                                    "modality": item["task2_gold_modality"],
                                    "confidence": 0.9,
                                }
                            ),
                            "latency_s": 0.0,
                            "error": "",
                        },
                        provider_id="fake",
                        profile_id="fake",
                        run_group_id="group1",
                    ),
                )
            config_path = root / "run_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "run_group_id": "task3-group",
                        "datasets": ["nice"],
                        "benchmark_variants": ["must"],
                        "stochastic": {"temperature": 0.7, "top_p": 1.0, "samples": 0},
                        "profiles": [
                            {
                                "profile_id": "fake",
                                "provider_id": "fake",
                                "base_url": "http://127.0.0.1:1234/v1",
                                "api_key_env": "LOCAL_OPENAI_API_KEY",
                                "models": ["fake-model"],
                                "concurrency": 1,
                                "batch_size": 2,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            argv = [
                "run_task3_verification_from_config.py",
                "--config",
                str(config_path),
                "--profile",
                "fake",
                "--model",
                "fake-model",
                "--dataset",
                "nice",
                "--source-run-id",
                "full-source",
                "--mode",
                "smoke",
                "--smoke-items",
                "2",
                "--fake-completion",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(task3_cli.eu, "project_root", return_value=root),
                redirect_stdout(io.StringIO()),
            ):
                task3_cli.main()

            task3_items_path = eu.task3_verification_items_path(
                root, "nice", "must", "full-source", "fake-model", "blind", smoke=True
            )
            task3_items = eu.read_csv_rows(task3_items_path)
            # The fake run must not touch the paper-facing item file.
            self.assertEqual(task3_items_path.parent.name, "smoke")
            self.assertFalse(
                eu.task3_verification_items_path(
                    root, "nice", "must", "full-source", "fake-model", "blind"
                ).exists()
            )
            task3_rows = eu.read_jsonl(
                root / "data/processed/smoke/model_outputs_raw_task3_verification.jsonl"
            )
            registry = eu.read_csv_rows(
                root / "data/processed/smoke/run_registry_task3_verification.csv"
            )
            progress = eu.read_csv_rows(
                root / "data/processed/smoke/run_progress_live_task3_verification.csv"
            )
            self.assertFalse(
                (
                    root / "data/processed/model_outputs_raw_task3_verification.jsonl"
                ).exists()
            )
            self.assertEqual(len(task3_items), len(benchmark))
            self.assertEqual(len(task3_rows), 2)
            self.assertFalse(
                (root / "data/processed/task3_verification_items.csv").exists()
            )
            self.assertEqual({row["task"] for row in task3_rows}, {"task3"})
            self.assertEqual({row["parse_status"] for row in task3_rows}, {"ok"})
            self.assertEqual({row["task3_audit_mode"] for row in task3_rows}, {"blind"})
            self.assertTrue(
                all(
                    "Declared extracted modality" not in row["prompt"]
                    for row in task3_rows
                )
            )
            self.assertEqual(registry[0]["status"], "complete")
            self.assertEqual(registry[0]["tasks"], "task3")
            self.assertIn("audit_mode=blind", registry[0]["notes"])
            self.assertEqual({row["task"] for row in progress}, {"task3"})

    def test_analysis_cli_generates_publication_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "prompts").mkdir(parents=True)
            (root / "data/processed").mkdir(parents=True)
            (root / "docs").mkdir(parents=True)
            for prompt_name in [
                "mandatory_entailment.txt",
                "modality_extraction.txt",
                "modality_verification.txt",
                "modality_verification_declared.txt",
            ]:
                (root / f"prompts/{prompt_name}").write_text(
                    Path(f"prompts/{prompt_name}").read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
            review_rows = [
                {**row, "weaker_than_should": "yes"}
                for row in eu.weak_modality_construct_review_rows()
            ]
            eu.write_csv_rows(
                root / "docs/weak_modality_construct_review.csv",
                review_rows,
                fieldnames=eu.WEAK_MODALITY_CONSTRUCT_REVIEW_FIELDS,
            )
            benchmark = eu.build_benchmark_items(export_report_seeds())
            eu.write_csv_rows(root / "data/processed/benchmark_items.csv", benchmark)
            task1_template = eu.load_prompt(root / "prompts/mandatory_entailment.txt")
            task2_template = eu.load_prompt(root / "prompts/modality_extraction.txt")
            raw_rows = []
            for item in benchmark:
                for task in ["task1", "task2"]:
                    parsed = (
                        {
                            "decision": item["task1_gold_decision"],
                            "confidence": 0.9,
                            "brief_reason": "test",
                        }
                        if task == "task1"
                        else {
                            "requirement": item["source_statement"],
                            "modality": item["task2_gold_modality"],
                            "confidence": 0.9,
                        }
                    )
                    raw = eu.build_raw_record(
                        run_id="full-analysis",
                        model="fake-model",
                        host="http://127.0.0.1:1234/v1",
                        task=task,
                        item=item,
                        sample_index=0,
                        sample_kind="deterministic",
                        temperature=0.0,
                        top_p=1.0,
                        prompt_version="v2-conf01",
                        prompt=eu.prompt_for_benchmark_task(
                            task, item, task1_template, task2_template
                        ),
                        completion={
                            "ok": True,
                            "raw_text": json.dumps(parsed),
                            "latency_s": 0.0,
                            "error": "",
                        },
                        provider_id="fake",
                        profile_id="fake",
                        run_group_id="group1",
                    )
                    raw_rows.append(raw)
                    eu.append_jsonl(
                        root / "data/processed/model_outputs_raw.jsonl", raw
                    )
            registry_row = eu.run_registry_summary(
                benchmark,
                raw_rows,
                run_id="full-analysis",
                run_group_id="group1",
                provider_id="fake",
                profile_id="fake",
                model="fake-model",
                dataset_id="nice",
                variant="must",
                tasks=["task1", "task2"],
                expected_stochastic_samples=0,
                started_at_utc="2026-05-22T00:00:00Z",
                finished_at_utc="2026-05-22T00:01:00Z",
            )
            eu.upsert_run_registry_row(
                root / "data/processed/run_registry.csv", registry_row
            )
            eu.write_benchmark_manifest(
                [
                    root / "data/processed/benchmark_items.csv",
                    root / "prompts/mandatory_entailment.txt",
                    root / "prompts/modality_extraction.txt",
                    root / "prompts/modality_verification.txt",
                    root / "prompts/modality_verification_declared.txt",
                ],
                root / "outputs/benchmark_manifest.json",
                root=root,
            )
            output_dir = root / "outputs/final"
            argv = [
                "generate_evaluation_analysis.py",
                "--dataset",
                "nice",
                "--variant",
                "must",
                "--run-id",
                "full-analysis",
                "--model",
                "fake-model",
                "--profile",
                "fake",
                "--output-dir",
                str(output_dir),
                "--bootstrap-iterations",
                "5",
                "--max-parse-failure-rate",
                "0",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(analysis_cli.eu, "project_root", return_value=root),
                redirect_stdout(io.StringIO()),
            ):
                analysis_cli.main()

            self.assertTrue((output_dir / "metrics_summary.csv").exists())
            self.assertTrue((output_dir / "paper_results_table.md").exists())
            self.assertTrue(
                (output_dir / "acse_semantic_normalized_scores.csv").exists()
            )
            self.assertTrue((output_dir / "acse_semantic_calibration.csv").exists())
            self.assertTrue((output_dir / "acse_semantic_calibration.md").exists())
            self.assertTrue((output_dir / "task1_p_yes_by_modality.svg").exists())
            self.assertTrue((output_dir / "provenance_manifest.json").exists())
            provenance = json.loads(
                (output_dir / "provenance_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(provenance["run_id"], "full-analysis")
            self.assertEqual(provenance["stale_item_count"], 0)
            self.assertEqual(provenance["acse_normalized_rows"], 0)

    def test_analysis_rejects_legacy_task3_as_official_blind(self):
        with self.assertRaisesRegex(ValueError, "legacy"):
            analysis_cli.require_task3_audit_mode(
                [{"task": "task3", "item_id": "legacy-item"}], "blind"
            )

    def test_show_run_progress_reads_outputs_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _scaffold_project_root(root, with_prompts=False)
            benchmark = eu.build_benchmark_items(export_report_seeds())[:1]
            eu.write_csv_rows(root / "data/processed/benchmark_items.csv", benchmark)
            raw_rows = [
                {
                    "run_id": "full-1",
                    "model": "m1",
                    "task": "task1",
                    "item_id": benchmark[0]["item_id"],
                    "seed_id": benchmark[0]["seed_id"],
                    "source_modality": benchmark[0]["source_modality"],
                    "sample_kind": "deterministic",
                    "sample_index": 0,
                    "parse_status": "ok",
                    "parsed_json": {"decision": "yes", "confidence": 90},
                    "prompt": benchmark[0]["source_statement"],
                }
            ]
            for row in raw_rows:
                eu.append_jsonl(root / "data/processed/model_outputs_raw.jsonl", row)
            registry_row = eu.run_registry_summary(
                benchmark,
                raw_rows,
                run_id="full-1",
                run_group_id="group1",
                provider_id="fake",
                profile_id="fake",
                model="m1",
                dataset_id="nice",
                variant="must",
                tasks=["task1"],
                expected_stochastic_samples=0,
                started_at_utc="2026-05-21T00:00:00Z",
            )
            eu.upsert_run_registry_row(
                root / "data/processed/run_registry.csv", registry_row
            )

            before = sorted(
                path.relative_to(root) for path in root.rglob("*") if path.is_file()
            )
            output = io.StringIO()
            with redirect_stdout(output):
                show_run_progress.print_progress(root, "nice", "must", "full-1")
            after = sorted(
                path.relative_to(root) for path in root.rglob("*") if path.is_file()
            )

            self.assertEqual(before, after)
            self.assertIn("full-1: records 1/1", output.getvalue())
            self.assertIn("task_progress", output.getvalue())

    def test_show_run_progress_requires_disambiguation_for_reused_run_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _scaffold_project_root(root, with_prompts=False)
            benchmark = eu.build_benchmark_items(export_report_seeds())[:1]
            eu.write_csv_rows(root / "data/processed/benchmark_items.csv", benchmark)
            raw_rows = [
                {
                    "run_id": "full-1",
                    "profile_id": "p1",
                    "model": "m1",
                    "task": "task1",
                    "item_id": benchmark[0]["item_id"],
                    "seed_id": benchmark[0]["seed_id"],
                    "source_modality": benchmark[0]["source_modality"],
                    "sample_kind": "deterministic",
                    "sample_index": 0,
                    "parse_status": "ok",
                    "parsed_json": {"decision": "yes", "confidence": 90},
                    "prompt": benchmark[0]["source_statement"],
                },
                {
                    "run_id": "full-1",
                    "profile_id": "p2",
                    "model": "m2",
                    "task": "task1",
                    "item_id": benchmark[0]["item_id"],
                    "seed_id": benchmark[0]["seed_id"],
                    "source_modality": benchmark[0]["source_modality"],
                    "sample_kind": "deterministic",
                    "sample_index": 0,
                    "parse_status": "request_error",
                    "parsed_json": {},
                    "prompt": benchmark[0]["source_statement"],
                },
            ]
            for row in raw_rows:
                eu.append_jsonl(root / "data/processed/model_outputs_raw.jsonl", row)
            for profile_id, model in [("p1", "m1"), ("p2", "m2")]:
                registry_row = eu.run_registry_summary(
                    benchmark,
                    raw_rows,
                    run_id="full-1",
                    run_group_id="group1",
                    provider_id="fake",
                    profile_id=profile_id,
                    model=model,
                    dataset_id="nice",
                    variant="must",
                    tasks=["task1"],
                    expected_stochastic_samples=0,
                    started_at_utc="2026-05-21T00:00:00Z",
                )
                eu.upsert_run_registry_row(
                    root / "data/processed/run_registry.csv", registry_row
                )

            with self.assertRaisesRegex(ValueError, "matches multiple registry rows"):
                show_run_progress.print_progress(root, "nice", "must", "full-1")

            output = io.StringIO()
            with redirect_stdout(output):
                show_run_progress.print_progress(
                    root, "nice", "must", "full-1", model="m2"
                )

            text = output.getvalue()
            self.assertIn("full-1: records 1/1", text)
            self.assertIn("parse_status: {'request_error': 1}", text)

    def test_run_completion_jobs_returns_records_from_fake_completion(self):
        item = {
            "item_id": "S0001_mandatory",
            "seed_id": "S0001",
            "source_modality": "mandatory",
        }
        jobs = [
            {
                "request_index": index,
                "run_id": "r1",
                "model": "m1",
                "host": "http://localhost:8000/v1",
                "task": "task1",
                "item": item,
                "sample_index": index,
                "sample_kind": "stochastic",
                "temperature": 0.7,
                "top_p": 1.0,
                "prompt_version": "v1",
                "prompt": f"prompt {index}",
                "max_tokens": 32,
                "timeout_s": 5,
                "api_key_env": "LOCAL_OPENAI_API_KEY",
            }
            for index in range(4)
        ]

        def fake_completion(**kwargs):
            return {
                "ok": True,
                "raw_text": '{"decision": "yes", "confidence": 80, "brief_reason": "ok"}',
                "response_json": {"prompt": kwargs["prompt"]},
                "latency_s": 0.01,
                "error": "",
            }

        records = list(
            eu.run_completion_jobs(jobs, max_workers=2, completion_fn=fake_completion)
        )

        self.assertEqual(len(records), 4)
        self.assertEqual({row["request_index"] for row in records}, {0, 1, 2, 3})
        self.assertTrue(all(row["parse_status"] == "ok" for row in records))
        self.assertTrue(all(row["parsed_json"]["decision"] == "yes" for row in records))

    def test_select_run_rows_uses_latest_full_run_by_default(self):
        raw_rows = [
            {"run_id": "full-20260518-100000-aaaa", "value": 1},
            {"run_id": "pilot-20260518-110000-bbbb", "value": 2},
            {"run_id": "full-20260518-120000-cccc", "value": 3},
            {"run_id": "full-20260518-120000-cccc", "value": 4},
            {"run_id": "full-shall-20260518-130000-dddd", "value": 5},
        ]
        selected_run_id, rows = eu.select_run_rows(raw_rows, prefix="full")
        self.assertEqual(selected_run_id, "full-20260518-120000-cccc")
        self.assertEqual([row["value"] for row in rows], [3, 4])
        self.assertFalse(
            eu.run_id_matches_prefix("full-shall-20260518-130000-dddd", "full")
        )
        self.assertTrue(
            eu.run_id_matches_prefix("full-shall-20260518-130000-dddd", "full-shall")
        )

    def test_select_run_rows_honors_explicit_run_id(self):
        raw_rows = [
            {"run_id": "full-20260518-100000-aaaa", "value": 1},
            {"run_id": "full-20260518-120000-cccc", "value": 2},
            {"run_id": "full-shall-20260518-130000-dddd", "value": 3},
        ]
        selected_run_id, rows = eu.select_run_rows(
            raw_rows, run_id="full-20260518-100000-aaaa", prefix="full"
        )
        self.assertEqual(selected_run_id, "full-20260518-100000-aaaa")
        self.assertEqual([row["value"] for row in rows], [1])

        selected_run_id, rows = eu.select_run_rows(
            raw_rows, run_id="full-shall-20260518-130000-dddd", prefix="full"
        )
        self.assertEqual(selected_run_id, "full-shall-20260518-130000-dddd")
        self.assertEqual(rows, [])

    def test_calibration_probabilities_task1_use_p_yes(self):
        rows = [
            {"p_yes": 0.1, "confidence": 0.9},
            {"p_yes": 0.8, "confidence": 0.8},
        ]
        self.assertEqual(eu.calibration_probabilities(rows, "task1"), [0.1, 0.8])
        self.assertEqual(eu.calibration_probabilities(rows, "task2"), [0.9, 0.8])

    def test_run_progress_summary_for_partial_outputs(self):
        benchmark = eu.build_benchmark_items(export_report_seeds())
        item = benchmark[0]
        raw_rows = [
            {
                "run_id": "r1",
                "model": "m1",
                "task": "task1",
                "item_id": item["item_id"],
                "sample_kind": "deterministic",
                "parse_status": "ok",
                "parsed_json": {
                    "decision": "yes",
                    "confidence": 90.0,
                    "brief_reason": "",
                },
            },
            {
                "run_id": "r1",
                "model": "m1",
                "task": "task1",
                "item_id": item["item_id"],
                "sample_kind": "stochastic",
                "parse_status": "invalid_json",
                "parsed_json": None,
            },
        ]

        progress = eu.run_progress_summary(
            benchmark, raw_rows, expected_stochastic_samples=5
        )

        self.assertEqual(len(progress), 1)
        self.assertEqual(progress[0]["observed_records"], 2)
        self.assertAlmostEqual(progress[0]["parse_success_rate"], 0.5)
        self.assertAlmostEqual(progress[0]["record_completion_rate"], 2 / 24)
        self.assertAlmostEqual(progress[0]["deterministic_item_coverage"], 0.25)

    def test_complete_run_ids_from_progress_can_be_variant_scoped(self):
        progress = []
        for run_id in ["full-20260518-120000-aaaa", "full-shall-20260518-130000-bbbb"]:
            for task in ["task1", "task2"]:
                progress.append(
                    {
                        "run_id": run_id,
                        "task": task,
                        "record_completion_rate": 1.0,
                        "deterministic_item_coverage": 1.0,
                        "stochastic_complete_item_rate": 1.0,
                    }
                )

        self.assertEqual(
            eu.complete_run_ids_from_progress(progress, prefix="full"),
            ["full-20260518-120000-aaaa"],
        )
        self.assertEqual(
            eu.complete_run_ids_from_progress(progress, prefix="full-shall"),
            ["full-shall-20260518-130000-bbbb"],
        )

    def test_run_progress_summary_ignores_rows_for_removed_benchmark_items(self):
        benchmark = eu.build_benchmark_items(export_report_seeds())
        raw_rows = [
            {
                "run_id": "r1",
                "model": "m1",
                "task": "task1",
                "item_id": "removed_item",
                "sample_kind": "deterministic",
                "parse_status": "ok",
                "parsed_json": {
                    "decision": "yes",
                    "confidence": 90.0,
                    "brief_reason": "",
                },
            }
        ]

        progress = eu.run_progress_summary(
            benchmark, raw_rows, expected_stochastic_samples=5
        )

        self.assertEqual(progress, [])

    def test_write_preliminary_result_snapshot(self):
        benchmark = eu.build_benchmark_items(export_report_seeds())
        item = benchmark[0]
        raw_rows = [
            raw_record(
                item,
                task="task1",
                parsed_json={"decision": "yes", "confidence": 90.0, "brief_reason": ""},
            )
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = eu.write_preliminary_result_snapshot(
                benchmark,
                raw_rows,
                tmpdir,
                expected_stochastic_samples=5,
            )

            self.assertGreater(snapshot["score_rows"], 0)
            self.assertGreater(snapshot["summary_rows"], 0)
            self.assertEqual(snapshot["progress_rows"], 1)
            for path in snapshot["paths"].values():
                self.assertTrue(path.exists())
            self.assertIn(
                "Preliminary Results",
                snapshot["paths"]["table"].read_text(encoding="utf-8"),
            )

    def test_matplotlib_figure_export(self):
        rows = [
            {
                "model": "m1",
                "task": "task1",
                "uq_method": "verbalized_confidence",
                "source_modality": modality,
                "p_yes": value,
            }
            for modality, value in [
                ("mandatory", 0.9),
                ("recommended", 0.6),
                ("optional", 0.3),
                ("nice_to_have", 0.1),
            ]
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "figure.svg"
            eu.write_task1_modality_svg(rows, path)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 100)

    def test_uq_method_inventory_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = eu.write_uq_method_inventory(tmpdir)

            self.assertTrue(paths["markdown"].exists())
            self.assertTrue(paths["csv"].exists())
            self.assertIn(
                "predictive_entropy", paths["markdown"].read_text(encoding="utf-8")
            )
            self.assertIn(
                eu.ACSE_PROXY_METHOD, paths["markdown"].read_text(encoding="utf-8")
            )


_MINIMAL_PROFILE = {
    "profile_id": "p",
    "provider_id": "p",
    "base_url": "http://x/v1",
    "models": ["m"],
}


class ResponseProvenanceAndRetryTest(unittest.TestCase):
    """Provider-response provenance, seeding, retries, and batch composition."""

    def _benchmark_items(self):
        return eu.build_benchmark_items(export_report_seeds())

    def _plan(self, **overrides):
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
        rows = kwargs.pop("benchmark_rows", self._benchmark_items())
        return eu.planned_completion_jobs(rows, **kwargs)

    # --- item 1: response provenance -------------------------------------

    def test_extract_response_fields_from_provider_payload(self):
        fields = eu.extract_response_fields(
            {
                "id": "chatcmpl-abc",
                "model": "served-m1",
                "system_fingerprint": "fp_123",
                "choices": [{"finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                },
            }
        )

        self.assertEqual(fields["finish_reason"], "stop")
        self.assertEqual(fields["usage_prompt_tokens"], 11)
        self.assertEqual(fields["usage_completion_tokens"], 7)
        self.assertEqual(fields["usage_total_tokens"], 18)
        self.assertEqual(fields["served_model"], "served-m1")
        self.assertEqual(fields["system_fingerprint"], "fp_123")
        self.assertEqual(fields["response_id"], "chatcmpl-abc")

    def test_extract_response_fields_defaults_when_absent(self):
        self.assertEqual(eu.extract_response_fields(None), eu.EMPTY_RESPONSE_FIELDS)
        self.assertEqual(eu.extract_response_fields({}), eu.EMPTY_RESPONSE_FIELDS)

    def test_raw_record_persists_response_fields_and_derived_counts(self):
        jobs = self._plan()

        def fake_completion(**kwargs):
            return {
                "ok": True,
                "raw_text": '{"requirement": "The system must export reports.", "modality": "mandatory", "confidence": 0.9}',
                "response_json": {
                    "id": "chatcmpl-xyz",
                    "model": "served-m1",
                    "system_fingerprint": "fp_9",
                    "choices": [{"finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 30,
                        "completion_tokens": 12,
                        "total_tokens": 42,
                    },
                },
                "latency_s": 0.01,
                "error": "",
            }

        record = eu.run_completion_job(jobs[0], completion_fn=fake_completion)

        self.assertEqual(record["finish_reason"], "stop")
        self.assertEqual(record["usage_prompt_tokens"], 30)
        self.assertEqual(record["usage_completion_tokens"], 12)
        self.assertEqual(record["usage_total_tokens"], 42)
        self.assertEqual(record["served_model"], "served-m1")
        self.assertEqual(record["system_fingerprint"], "fp_9")
        self.assertEqual(record["response_id"], "chatcmpl-xyz")
        self.assertEqual(record["response_chars"], len(record["raw_text"]))
        self.assertEqual(record["requirement_word_count"], 5)
        self.assertEqual(record["system_prompt"], "")
        self.assertEqual(record["request_messages_role_layout"], "user_only")
        self.assertEqual(record["job_config_sha_version"], eu.JOB_CONFIG_SHA_VERSION)

    def test_batched_rows_copy_batch_usage_under_explicit_name(self):
        jobs = self._plan()

        def fake_completion(**kwargs):
            items = json.loads(kwargs["prompt"].split("Items:\n", 1)[1])
            return {
                "ok": True,
                "raw_text": json.dumps(
                    {
                        "results": [
                            {
                                "request_index": item["request_index"],
                                "requirement": "The system must export reports.",
                                "modality": "mandatory",
                                "confidence": 0.8,
                            }
                            for item in items
                        ]
                    }
                ),
                "response_json": {
                    "model": "served-m1",
                    "choices": [{"finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 80,
                        "total_tokens": 180,
                    },
                },
                "latency_s": 0.02,
                "error": "",
            }

        records = list(
            eu.run_completion_jobs(
                jobs, max_workers=1, completion_fn=fake_completion, batch_size=4
            )
        )

        self.assertGreater(len(records), 1)
        for record in records:
            self.assertEqual(record["usage_completion_tokens"], 80)
            self.assertEqual(record["batch_usage_completion_tokens"], 80)
            self.assertEqual(record["finish_reason"], "stop")

    # --- item 2: truncation ----------------------------------------------

    def test_length_finish_reason_marks_parse_failure_as_truncated(self):
        jobs = self._plan()

        def fake_completion(**kwargs):
            return {
                "ok": True,
                "raw_text": '{"requirement": "The system must expo',
                "response_json": {
                    "choices": [{"finish_reason": "length"}],
                    "model": "m1",
                },
                "latency_s": 0.01,
                "error": "",
            }

        record = eu.run_completion_job(jobs[0], completion_fn=fake_completion)

        self.assertEqual(record["parse_status"], "truncated")
        self.assertNotEqual(record["parse_status"], "ok")

    def test_length_finish_reason_keeps_ok_status_when_parse_succeeds(self):
        jobs = self._plan()

        def fake_completion(**kwargs):
            return {
                "ok": True,
                "raw_text": '{"requirement": "The system must export reports.", "modality": "mandatory", "confidence": 0.9}',
                "response_json": {
                    "choices": [{"finish_reason": "length"}],
                    "model": "m1",
                },
                "latency_s": 0.01,
                "error": "",
            }

        self.assertEqual(
            eu.run_completion_job(jobs[0], completion_fn=fake_completion)[
                "parse_status"
            ],
            "ok",
        )

    def test_request_error_is_not_relabelled_as_truncated(self):
        jobs = self._plan()

        def fake_completion(**kwargs):
            return {
                "ok": False,
                "raw_text": "",
                "response_json": {"choices": [{"finish_reason": "length"}]},
                "latency_s": 0.01,
                "error": "boom",
            }

        self.assertEqual(
            eu.run_completion_job(jobs[0], completion_fn=fake_completion)[
                "parse_status"
            ],
            "request_error",
        )

    # --- item 3: seed -----------------------------------------------------

    def test_profile_defaults_expose_seed_retry_and_batch_order_knobs(self):
        profile = eu.normalize_provider_profile(
            {
                "profile_id": "p1",
                "provider_id": "prov",
                "base_url": "http://localhost:8000/v1",
                "models": ["m1"],
            }
        )

        self.assertEqual(profile["seed"], eu.DEFAULT_REQUEST_SEED)
        self.assertTrue(profile["send_seed"])
        self.assertEqual(profile["max_retries"], eu.DEFAULT_MAX_RETRIES)
        self.assertEqual(profile["batch_order"], eu.DEFAULT_BATCH_ORDER)

    def test_profile_seed_and_batch_order_overrides(self):
        profile = eu.normalize_provider_profile(
            {
                "profile_id": "p1",
                "provider_id": "prov",
                "base_url": "http://localhost:8000/v1",
                "models": ["m1"],
                "seed": 7,
                "send_seed": False,
                "max_retries": 5,
                "batch_order": "shuffled",
            }
        )

        self.assertEqual(profile["seed"], 7)
        self.assertFalse(profile["send_seed"])
        self.assertEqual(profile["max_retries"], 5)
        self.assertEqual(profile["batch_order"], "shuffled")
        with self.assertRaises(ValueError):
            eu.normalize_batch_order("interleaved")

    def test_chat_completion_sends_seed_and_records_it(self):
        captured = {}

        class FakeClient:
            def __init__(self, **_kwargs):
                self.chat = type("Chat", (), {"completions": self})()

            def create(self, **kwargs):
                captured.update(kwargs)
                return FakeResponse(
                    dump={"model": "m1", "choices": [{"finish_reason": "stop"}]}
                )

        with mock.patch.object(eu, "OpenAI", FakeClient):
            result = eu.chat_completion(
                "http://x/v1", "m1", "prompt", 0.0, 1.0, seed=1234
            )

        self.assertEqual(captured["seed"], 1234)
        self.assertEqual(captured["messages"], [{"role": "user", "content": "prompt"}])
        self.assertEqual(result["request_seed"], 1234)
        self.assertEqual(result["retry_count"], 0)

        captured.clear()
        with mock.patch.object(eu, "OpenAI", FakeClient):
            result = eu.chat_completion(
                "http://x/v1", "m1", "prompt", 0.0, 1.0, seed=None
            )

        self.assertNotIn("seed", captured)
        self.assertIsNone(result["request_seed"])

    def test_send_seed_false_drops_seed_from_job_and_request(self):
        jobs = self._plan(send_seed=False)
        captured = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return {
                "ok": True,
                "raw_text": "{}",
                "response_json": {},
                "latency_s": 0.0,
                "error": "",
            }

        record = eu.run_completion_job(jobs[0], completion_fn=fake_completion)

        self.assertIsNone(jobs[0]["seed"])
        self.assertIsNone(captured["seed"])
        self.assertIsNone(record["request_seed"])

        seeded_jobs = self._plan(seed=99)
        self.assertEqual(seeded_jobs[0]["seed"], 99)

    # --- item 4: fingerprints ---------------------------------------------

    def test_request_payload_sha_is_stable_and_field_sensitive(self):
        payload = {
            "model": "m1",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 64,
            "seed": 20260518,
            "response_format": {"type": "json_object"},
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        first = eu.request_payload_sha(payload)

        self.assertEqual(
            first, eu.request_payload_sha(dict(reversed(list(payload.items()))))
        )
        # Transport-only kwargs do not move the fingerprint...
        self.assertEqual(first, eu.request_payload_sha({**payload, "logprobs": True}))
        # ...but any hashed request field does.
        self.assertNotEqual(first, eu.request_payload_sha({**payload, "seed": 1}))
        self.assertNotEqual(
            first, eu.request_payload_sha({**payload, "max_tokens": 65})
        )

    def test_chat_completion_reports_request_payload_sha(self):
        class FakeClient:
            def __init__(self, **_kwargs):
                self.chat = type("Chat", (), {"completions": self})()

            def create(self, **kwargs):
                return FakeResponse()

        with mock.patch.object(eu, "OpenAI", FakeClient):
            result = eu.chat_completion(
                "http://x/v1", "m1", "prompt", 0.0, 1.0, max_tokens=64, seed=5
            )

        self.assertEqual(
            result["request_payload_sha"],
            eu.request_payload_sha(
                {
                    "model": "m1",
                    "messages": [{"role": "user", "content": "prompt"}],
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 64,
                    "seed": 5,
                }
            ),
        )

    def test_record_falls_back_to_locally_computed_payload_sha(self):
        jobs = self._plan()

        def fake_completion(**kwargs):
            return {
                "ok": True,
                "raw_text": "{}",
                "response_json": {},
                "latency_s": 0.0,
                "error": "",
            }

        record = eu.run_completion_job(jobs[0], completion_fn=fake_completion)
        again = eu.run_completion_job(jobs[0], completion_fn=fake_completion)

        self.assertTrue(record["request_payload_sha"])
        self.assertEqual(record["request_payload_sha"], again["request_payload_sha"])

    def test_job_config_sha_covers_extra_body_and_new_inputs(self):
        base = {
            "prompt": "p",
            "prompt_version": "v2-conf01",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 64,
            "structured_output": "json_object",
            "json_mode": True,
        }
        baseline = eu.compute_job_config_sha(**base)

        self.assertNotEqual(
            baseline,
            eu.compute_job_config_sha(
                **base, extra_body={"thinking": {"type": "disabled"}}
            ),
        )
        self.assertNotEqual(
            baseline,
            eu.compute_job_config_sha(**base, response_format={"type": "json_object"}),
        )
        self.assertNotEqual(
            baseline, eu.compute_job_config_sha(**base, instructor_mode="tools")
        )
        self.assertNotEqual(
            baseline, eu.compute_job_config_sha(**base, validation_retries=5)
        )
        self.assertNotEqual(baseline, eu.compute_job_config_sha(**base, seed=1))
        self.assertEqual(baseline, eu.compute_job_config_sha(**base))

    def test_job_config_sha_covers_the_batching_setup(self):
        base = {
            "prompt": "p",
            "prompt_version": "v2-conf01",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 64,
            "structured_output": "json_object",
            "json_mode": True,
        }
        baseline = eu.compute_job_config_sha(**base)
        batched = eu.compute_job_config_sha(**base, task="task2", batch_size=16)

        self.assertNotEqual(baseline, batched)
        self.assertNotEqual(
            batched, eu.compute_job_config_sha(**base, task="task2", batch_size=8)
        )
        self.assertNotEqual(
            batched,
            eu.compute_job_config_sha(
                **base, task="task2", batch_size=16, batch_order="shuffled"
            ),
        )
        self.assertNotEqual(
            baseline, eu.compute_job_config_sha(**base, fallback_batch_size=4)
        )
        # Different tasks render different batch wrappers.
        self.assertNotEqual(
            batched, eu.compute_job_config_sha(**base, task="task1", batch_size=16)
        )

        # The wrapper text is hashed for batched plans ...
        with mock.patch.object(
            eu, "batch_prompt_wrapper_sha", return_value="EDITED-WRAPPER"
        ):
            self.assertNotEqual(
                batched, eu.compute_job_config_sha(**base, task="task2", batch_size=16)
            )
            # ... and never for single-item plans, whatever the wrapper says.
            self.assertEqual(
                baseline, eu.compute_job_config_sha(**base, task="task2", batch_size=1)
            )
            self.assertEqual(baseline, eu.compute_job_config_sha(**base, task="task1"))

    def test_paper_condition_fingerprints_are_pinned(self):
        """Regression pin: the bare Task 2 request is the paper condition.

        Every reported run was a 16-item grouped batch built by
        `batch_prompt_for_completion_jobs` with no item context. These digests
        must not move when ablation knobs (batch_order, item_context) are
        added, otherwise `--mode resume` would re-request every archived
        batched row and the bare arm of an ablation would no longer be the
        paper condition.
        """
        self.assertEqual(
            eu.batch_prompt_wrapper_sha("task2"),
            "d8a5b71a8e621ca55dcfbf4f52a3248581cda1cda333ae372c0b75b0e230a35a",
        )
        base = {
            "prompt": "p",
            "prompt_version": "v2-conf01",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 64,
            "structured_output": "json_object",
            "json_mode": True,
        }
        self.assertEqual(
            eu.compute_job_config_sha(**base),
            "9c1bac1710cbae8ce430c3891173021c8d34e937c39ed55acbe1afb977fc198b",
        )
        self.assertEqual(
            eu.compute_job_config_sha(**base, task="task2", batch_size=16),
            "9ac6e6ad20307371e4774b4ecc36a06f8e1e04c7e1ed2c7b37d637574343db7a",
        )

    def test_item_context_knob_normalizes_and_rejects_unknown_values(self):
        self.assertEqual(eu.normalize_item_context(None), "bare")
        self.assertEqual(eu.normalize_item_context(""), "bare")
        self.assertEqual(eu.normalize_item_context(" Document "), "document")
        with self.assertRaises(ValueError):
            eu.normalize_item_context("envelope")
        run_config = eu.normalize_run_config(
            {
                "run_group_id": "g",
                "profiles": [_MINIMAL_PROFILE],
                "item_context": "document",
            }
        )
        self.assertEqual(run_config["item_context"], "document")
        self.assertEqual(
            eu.normalize_run_config(
                {"run_group_id": "g", "profiles": [_MINIMAL_PROFILE]}
            )["item_context"],
            "bare",
        )

    def _pure_item(self, **overrides):
        item = {
            "item_id": "S0001_optional",
            "seed_id": "S0001",
            "source_modality": "optional",
            "source_statement": "The system MAY export reports.",
            "candidate_requirement": "The system MUST export reports.",
            "context_document": "Fixture FRS, version 1",
            "context_legend": "(M) mandatory, (O) optional",
            "context_section": "2 Reporting > 2.1 Exports",
            "context_requirement_id": "2.1.2",
            "context_marker": "O",
            "context_before": "2.1.1 (M): The system shall store reports.",
            "context_after": "2.1.3 (O): The system should archive reports.",
        }
        return {**item, **overrides}

    def test_document_context_text_renders_every_field_once(self):
        text = eu.document_context_text(self._pure_item())
        self.assertEqual(
            text.splitlines(),
            [
                "Document: Fixture FRS, version 1 (markers: (M) mandatory, (O) optional)",
                "Section: 2 Reporting > 2.1 Exports",
                "Preceding requirement 2.1.1 (M): The system shall store reports.",
                "This requirement: 2.1.2, marker (O)",
                "Following requirement 2.1.3 (O): The system should archive reports.",
            ],
        )
        # The source statement itself is never repeated in the context.
        self.assertNotIn("MAY export", text)
        edge = eu.document_context_text(
            self._pure_item(context_before="", context_after="")
        )
        self.assertNotIn("Preceding", edge)
        self.assertNotIn("Following", edge)

    def test_prompt_for_benchmark_task_renders_context_only_for_the_document_arm(
        self,
    ):
        item = self._pure_item()
        task2 = eu.load_prompt("prompts/modality_extraction.txt")
        task2_context = eu.load_prompt("prompts/modality_extraction_context.txt")
        bare = eu.prompt_for_benchmark_task(
            "task2", item, "t1 {source_statement}", task2
        )
        document = eu.prompt_for_benchmark_task(
            "task2",
            item,
            "t1 {source_statement}",
            task2,
            item_context="document",
            task2_context_template=task2_context,
        )

        self.assertNotIn("Document context", bare)
        self.assertIn(
            "Document context (where the source statement appears):", document
        )
        self.assertIn("This requirement: 2.1.2, marker (O)", document)
        self.assertIn('Source:\n"The system MAY export reports."', document)
        self.assertIn("not from its context", document)
        # The fake smoke completion locates the statement by this exact shape.
        self.assertEqual(
            run_config_cli.source_from_prompt(document),
            "The system MAY export reports.",
        )
        with self.assertRaises(ValueError):
            eu.prompt_for_benchmark_task(
                "task2", item, "t1", task2, item_context="document"
            )
        # Task 1 ignores the knob entirely.
        self.assertEqual(
            eu.prompt_for_benchmark_task(
                "task1", item, "t1 {source_statement}", task2, item_context="document"
            ),
            "t1 The system MAY export reports.",
        )

    def test_batch_prompt_document_arm_adds_context_per_item_and_keeps_bare_intact(
        self,
    ):
        jobs = [
            {
                "task": "task2",
                "request_index": index,
                "item": self._pure_item(item_id=f"S000{index}_optional"),
            }
            for index in range(2)
        ]
        bare = eu.batch_prompt_for_completion_jobs(jobs)
        document = eu.batch_prompt_for_completion_jobs(
            [{**job, "item_context": "document"} for job in jobs]
        )

        self.assertEqual(
            bare,
            eu.batch_prompt_for_completion_jobs(
                [{**job, "item_context": "bare"} for job in jobs]
            ),
        )
        self.assertNotIn("context", bare)
        bare_items = json.loads(bare.split("Items:\n", 1)[1])
        document_items = json.loads(document.split("Items:\n", 1)[1])
        self.assertEqual(set(bare_items[0]), {"request_index", "source_statement"})
        self.assertEqual(
            set(document_items[0]), {"request_index", "source_statement", "context"}
        )
        self.assertIn(
            "This requirement: 2.1.2, marker (O)", document_items[1]["context"]
        )
        self.assertIn("Extract from the source statement only.", document)
        self.assertNotIn("Extract from the source statement only.", bare)
        self.assertEqual(
            [item["source_statement"] for item in bare_items],
            [item["source_statement"] for item in document_items],
        )
        with self.assertRaises(ValueError):
            eu.batch_prompt_for_completion_jobs(
                [jobs[0], {**jobs[1], "item_context": "document"}]
            )
        # Task 1 and Task 3 wrappers do not change with the knob.
        for task, extra in (
            ("task1", {}),
            ("task3", {"task2_requirement": "The system may export reports."}),
        ):
            task_jobs = [
                {"task": task, "request_index": 0, "item": self._pure_item(**extra)}
            ]
            self.assertEqual(
                eu.batch_prompt_for_completion_jobs(task_jobs),
                eu.batch_prompt_for_completion_jobs(
                    [{**task_jobs[0], "item_context": "document"}]
                ),
            )

    def test_document_arm_has_its_own_wrapper_and_job_fingerprints(self):
        self.assertNotEqual(
            eu.batch_prompt_wrapper_sha("task2", "document"),
            eu.batch_prompt_wrapper_sha("task2"),
        )
        self.assertEqual(
            eu.batch_prompt_wrapper_sha("task2", "bare"),
            eu.batch_prompt_wrapper_sha("task2"),
        )
        base = {
            "prompt": "p",
            "prompt_version": "v2-conf01",
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 64,
            "structured_output": "json_object",
            "json_mode": True,
        }
        for batch_size in (1, 16):
            with self.subTest(batch_size=batch_size):
                bare = eu.compute_job_config_sha(
                    **base, task="task2", batch_size=batch_size
                )
                self.assertEqual(
                    bare,
                    eu.compute_job_config_sha(
                        **base, task="task2", batch_size=batch_size, item_context="bare"
                    ),
                )
                self.assertNotEqual(
                    bare,
                    eu.compute_job_config_sha(
                        **base,
                        task="task2",
                        batch_size=batch_size,
                        item_context="document",
                    ),
                )

    def test_planned_jobs_and_raw_records_carry_item_context(self):
        rows = [self._pure_item()]
        common = {
            "tasks": ["task2"],
            "model": "m",
            "host": "http://x/v1",
            "run_id": "r",
            "prompt_version": "v2-conf01",
            "task1_template": "t1 {source_statement} {candidate_requirement}",
            "task2_template": eu.load_prompt("prompts/modality_extraction.txt"),
            "deterministic": {"temperature": 0.0, "top_p": 1.0, "samples": 1},
            "stochastic": {"temperature": 0.7, "top_p": 1.0, "samples": 0},
            "max_tokens": 64,
            "timeout_s": 10,
            "api_key_env": "K",
            "batch_size": 16,
        }
        bare_job = eu.planned_completion_jobs(rows, **common)[0]
        document_job = eu.planned_completion_jobs(
            rows,
            **common,
            item_context="document",
            task2_context_template=eu.load_prompt(
                "prompts/modality_extraction_context.txt"
            ),
        )[0]

        self.assertEqual(bare_job["item_context"], "bare")
        self.assertEqual(document_job["item_context"], "document")
        self.assertNotEqual(bare_job["job_config_sha"], document_job["job_config_sha"])
        self.assertIn("Document context", document_job["prompt"])
        self.assertNotIn("Document context", bare_job["prompt"])

        completion = {
            "ok": True,
            "raw_text": '{"requirement": "x", "modality": "optional", "confidence": 0.5}',
            "response_json": {},
            "latency_s": 0.0,
            "error": "",
        }
        record = eu._job_record(
            document_job,
            completion=completion,
            request_index=0,
            response_format=None,
            request_extra_body=None,
        )
        self.assertEqual(record["item_context"], "document")
        self.assertEqual(record["context_marker"], "O")
        self.assertEqual(record["context_requirement_id"], "2.1.2")
        plain = eu._job_record(
            {**bare_job, "item": export_report_seeds()[0] | bare_job["item"]},
            completion=completion,
            request_index=0,
            response_format=None,
            request_extra_body=None,
        )
        self.assertEqual(plain["item_context"], "bare")
        legacy = eu.build_raw_record(
            run_id="r",
            model="m",
            host="h",
            task="task2",
            item={"item_id": "i", "seed_id": "s", "source_modality": "optional"},
            sample_index=0,
            sample_kind="deterministic",
            temperature=0.0,
            top_p=1.0,
            prompt_version="v2-conf01",
            prompt="p",
            completion=completion,
        )
        self.assertEqual(legacy["item_context"], "bare")
        self.assertNotIn("context_marker", legacy)

    def test_batch_prompt_wrapper_sha_digests_the_rendered_wrapper(self):
        probe_jobs = [
            {
                "task": "task2",
                "request_index": index,
                "item": dict(eu.BATCH_WRAPPER_PROBE_ITEM),
            }
            for index in range(2)
        ]

        self.assertEqual(
            eu.batch_prompt_wrapper_sha("task2"),
            eu.sha256_text(eu.batch_prompt_for_completion_jobs(probe_jobs)),
        )
        self.assertNotEqual(
            eu.batch_prompt_wrapper_sha("task1"), eu.batch_prompt_wrapper_sha("task2")
        )
        self.assertNotEqual(
            eu.batch_prompt_wrapper_sha("task2"), eu.batch_prompt_wrapper_sha("task3")
        )

    def test_planned_jobs_record_the_batch_size(self):
        unbatched = self._plan()
        batched = self._plan(batch_size=16)

        self.assertEqual({job["batch_size"] for job in unbatched}, {1})
        self.assertEqual({job["batch_size"] for job in batched}, {16})
        # The planned batch size is part of the resume fingerprint.
        self.assertNotEqual(
            batched[0]["job_config_sha"], unbatched[0]["job_config_sha"]
        )

    def test_planned_task3_jobs_record_the_batch_size(self):
        items = [
            {
                "item_id": "S0001_mandatory__task3__m1__blind",
                "seed_id": "S0001",
                "source_modality": "mandatory",
                "source_statement": "The system MUST export reports.",
                "task2_requirement": "The system must export reports.",
            }
        ]
        jobs = eu.planned_completion_jobs_for_items(
            items,
            prompt_fn=lambda item: "audit this",
            prompt_version="v2-conf01",
            model="m1",
            host="http://localhost:8000/v1",
            run_id="task3-1",
            deterministic={"temperature": 0.0, "top_p": 1.0, "samples": 1},
            stochastic={"temperature": 0.7, "top_p": 1.0, "samples": 0},
            max_tokens=64,
            timeout_s=30,
            api_key_env="LOCAL_OPENAI_API_KEY",
            batch_size=16,
        )

        self.assertEqual([job["batch_size"] for job in jobs], [16])
        self.assertEqual(jobs[0]["task"], "task3")

    def test_planned_jobs_record_sha_version(self):
        jobs = self._plan()

        self.assertEqual(jobs[0]["job_config_sha_version"], eu.JOB_CONFIG_SHA_VERSION)
        self.assertEqual(eu.JOB_CONFIG_SHA_VERSION, 3)

    # --- item 5: retries ---------------------------------------------------

    def _fake_client_raising(self, errors):
        calls = {"count": 0}

        class FakeClient:
            def __init__(self, **_kwargs):
                self.chat = type("Chat", (), {"completions": self})()

            def create(self, **kwargs):
                index = calls["count"]
                calls["count"] += 1
                if index < len(errors):
                    raise errors[index]
                return FakeResponse(dump={"model": "m1"})

        return FakeClient, calls

    def test_chat_completion_retries_rate_limit_then_succeeds(self):
        import httpx
        import openai

        request = httpx.Request("POST", "http://x/v1/chat/completions")
        rate_limited = openai.RateLimitError(
            "slow down",
            response=httpx.Response(429, request=request),
            body=None,
        )
        FakeClient, calls = self._fake_client_raising([rate_limited])

        with (
            mock.patch.object(eu, "OpenAI", FakeClient),
            mock.patch.object(eu.time, "sleep") as sleeper,
        ):
            result = eu.chat_completion(
                "http://x/v1", "m1", "prompt", 0.0, 1.0, max_retries=3
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["retry_count"], 1)
        self.assertEqual(calls["count"], 2)
        sleeper.assert_called_once()

    def test_chat_completion_stops_after_max_attempts(self):
        import httpx
        import openai

        request = httpx.Request("POST", "http://x/v1/chat/completions")
        server_error = openai.InternalServerError(
            "boom",
            response=httpx.Response(503, request=request),
            body=None,
        )
        FakeClient, calls = self._fake_client_raising([server_error] * 10)

        with (
            mock.patch.object(eu, "OpenAI", FakeClient),
            mock.patch.object(eu.time, "sleep"),
        ):
            result = eu.chat_completion(
                "http://x/v1", "m1", "prompt", 0.0, 1.0, max_retries=3
            )

        self.assertFalse(result["ok"])
        self.assertEqual(calls["count"], 3)
        self.assertEqual(result["retry_count"], 2)

    def test_chat_completion_does_not_retry_bad_request(self):
        import httpx
        import openai

        request = httpx.Request("POST", "http://x/v1/chat/completions")
        budget_error = openai.BadRequestError(
            "ExceededBudget",
            response=httpx.Response(400, request=request),
            body=None,
        )
        FakeClient, calls = self._fake_client_raising([budget_error] * 5)

        with (
            mock.patch.object(eu, "OpenAI", FakeClient),
            mock.patch.object(eu.time, "sleep") as sleeper,
        ):
            result = eu.chat_completion(
                "http://x/v1", "m1", "prompt", 0.0, 1.0, max_retries=3
            )

        self.assertFalse(result["ok"])
        self.assertEqual(calls["count"], 1)
        self.assertEqual(result["retry_count"], 0)
        sleeper.assert_not_called()
        self.assertIn("ExceededBudget", result["error"])

    def test_is_transient_provider_error_status_classification(self):
        class StatusError(Exception):
            def __init__(self, status_code):
                self.status_code = status_code

        for status in (408, 429, 500, 502, 503):
            self.assertTrue(eu.is_transient_provider_error(StatusError(status)), status)
        for status in (400, 401, 403, 404, 422):
            self.assertFalse(
                eu.is_transient_provider_error(StatusError(status)), status
            )
        self.assertTrue(eu.is_transient_provider_error(TimeoutError()))
        self.assertFalse(eu.is_transient_provider_error(ValueError("nope")))

    def test_retry_count_reaches_the_raw_record(self):
        jobs = self._plan()

        def fake_completion(**kwargs):
            return {
                "ok": True,
                "raw_text": '{"requirement": "r", "modality": "mandatory", "confidence": 0.9}',
                "response_json": {},
                "latency_s": 0.0,
                "error": "",
                "retry_count": 2,
            }

        self.assertEqual(
            eu.run_completion_job(jobs[0], completion_fn=fake_completion)[
                "retry_count"
            ],
            2,
        )

    # --- item 6: batch composition ----------------------------------------

    def _multi_seed_jobs(self, **overrides):
        seeds = [
            {
                "seed_id": f"S000{index}",
                "source_dataset": "NICE",
                "original_requirement": "The system shall export reports.",
                "capability_text_final": f"export reports {index}",
            }
            for index in range(1, 5)
        ]
        return self._plan(benchmark_rows=eu.build_benchmark_items(seeds), **overrides)

    def test_grouped_batches_keep_consecutive_request_indices(self):
        jobs = self._multi_seed_jobs()
        batches = eu.completion_job_batches(jobs, batch_size=4)

        self.assertEqual(
            [[int(job["request_index"]) for job in batch] for batch in batches],
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
        )
        # Grouped batching packs all four modality variants of one seed together.
        for batch in batches:
            self.assertEqual(len({job["item"]["seed_id"] for job in batch}), 1)
            self.assertEqual(len({job["item"]["source_modality"] for job in batch}), 4)

    def test_shuffled_batches_are_deterministic_and_mix_seeds(self):
        jobs = self._multi_seed_jobs(batch_order="shuffled")
        first = eu.completion_job_batches(jobs, batch_size=4)
        second = eu.completion_job_batches(jobs, batch_size=4)

        indices = [[int(job["request_index"]) for job in batch] for batch in first]
        self.assertEqual(
            indices, [[int(job["request_index"]) for job in batch] for batch in second]
        )
        self.assertNotEqual(
            indices, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
        )
        self.assertEqual(
            sorted(index for batch in indices for index in batch), list(range(16))
        )
        self.assertTrue(
            any(len({job["item"]["seed_id"] for job in batch}) > 1 for batch in first)
        )

        other_seed = eu.completion_job_batches(
            jobs, batch_size=4, batch_order="shuffled", seed=1
        )
        self.assertNotEqual(
            indices,
            [[int(job["request_index"]) for job in batch] for batch in other_seed],
        )

    def test_batch_composition_fields_are_recorded(self):
        jobs = self._multi_seed_jobs(batch_order="shuffled")

        def fake_completion(**kwargs):
            items = json.loads(kwargs["prompt"].split("Items:\n", 1)[1])
            return {
                "ok": True,
                "raw_text": json.dumps(
                    {
                        "results": [
                            {
                                "request_index": item["request_index"],
                                "requirement": "The system must export reports.",
                                "modality": "mandatory",
                                "confidence": 0.8,
                            }
                            for item in items
                        ]
                    }
                ),
                "response_json": {
                    "model": "m1",
                    "choices": [{"finish_reason": "stop"}],
                },
                "latency_s": 0.0,
                "error": "",
            }

        records = list(
            eu.run_completion_jobs(
                jobs, max_workers=1, completion_fn=fake_completion, batch_size=4
            )
        )

        self.assertEqual({row["batch_order"] for row in records}, {"shuffled"})
        by_batch = {}
        for row in records:
            by_batch.setdefault(row["batch_id"], []).append(row)
        for batch_rows in by_batch.values():
            seed_ids = {tuple(row["batch_seed_ids"]) for row in batch_rows}
            self.assertEqual(len(seed_ids), 1)
            self.assertEqual(sorted(next(iter(seed_ids))), list(next(iter(seed_ids))))
            self.assertEqual(
                {row["batch_variant_mix"] for row in batch_rows},
                {len({r["source_modality"] for r in batch_rows})},
            )
        self.assertTrue(any(len(row["batch_seed_ids"]) > 1 for row in records))

    def test_single_job_records_carry_singleton_batch_composition(self):
        jobs = self._plan()

        def fake_completion(**kwargs):
            return {
                "ok": True,
                "raw_text": "{}",
                "response_json": {},
                "latency_s": 0.0,
                "error": "",
            }

        record = eu.run_completion_job(jobs[0], completion_fn=fake_completion)

        self.assertEqual(record["batch_seed_ids"], ["S0001"])
        self.assertEqual(record["batch_variant_mix"], 1)
        self.assertEqual(record["batch_order"], "grouped")

    def test_run_config_seed_and_batch_order_flow_into_profiles(self):
        run_config = eu.normalize_run_config(
            {
                "run_group_id": "g1",
                "seed": 4242,
                "batch_order": "shuffled",
                "profiles": [
                    {
                        "profile_id": "p1",
                        "provider_id": "prov",
                        "base_url": "http://localhost:8000/v1",
                        "models": ["m1"],
                    },
                    {
                        "profile_id": "p2",
                        "provider_id": "prov",
                        "base_url": "http://localhost:8000/v1",
                        "models": ["m2"],
                        "seed": 1,
                        "batch_order": "grouped",
                    },
                ],
            }
        )

        self.assertEqual(run_config["seed"], 4242)
        self.assertEqual(run_config["batch_order"], "shuffled")
        self.assertEqual(run_config["profiles"][0]["seed"], 4242)
        self.assertEqual(run_config["profiles"][0]["batch_order"], "shuffled")
        self.assertEqual(run_config["profiles"][1]["seed"], 1)
        self.assertEqual(run_config["profiles"][1]["batch_order"], "grouped")

    # --- item 7: shuffled batches never repeat a seed ----------------------

    def _full_matrix_task2_jobs(self, **overrides):
        """The real 180-seed x 4-variant benchmark cell, planned as 720 Task 2 jobs."""
        benchmark = eu.read_csv_rows("data/processed/benchmark_items_mlm_tapt.csv")
        self.assertEqual(len(benchmark), 720)
        self.assertEqual(len({row["seed_id"] for row in benchmark}), 180)
        return self._plan(benchmark_rows=benchmark, **overrides)

    def test_shuffled_batches_never_put_two_variants_of_a_seed_together(self):
        jobs = self._full_matrix_task2_jobs(batch_order="shuffled", batch_size=16)
        first = eu.completion_job_batches(jobs, 16)
        second = eu.completion_job_batches(jobs, 16)

        self.assertEqual(len(first), 45)
        self.assertEqual({len(batch) for batch in first}, {16})
        for batch in first:
            self.assertEqual(len({job["item"]["seed_id"] for job in batch}), 16)
        self.assertEqual(
            sorted(int(job["request_index"]) for batch in first for job in batch),
            list(range(720)),
        )

        indices = [[int(job["request_index"]) for job in batch] for batch in first]
        self.assertEqual(
            indices, [[int(job["request_index"]) for job in batch] for batch in second]
        )

        grouped = eu.completion_job_batches(jobs, 16, batch_order="grouped")
        self.assertNotEqual(
            indices, [[int(job["request_index"]) for job in batch] for batch in grouped]
        )
        # The grouped default is exactly what the ablation breaks up: 16 jobs
        # from only 4 seeds, i.e. all four variants of each seed side by side.
        self.assertEqual(
            {len({job["item"]["seed_id"] for job in batch}) for batch in grouped}, {4}
        )

    def test_shuffled_batching_falls_back_loudly_when_seeds_are_too_few(self):
        # 4 seeds x 4 variants into batches of 8: a seed collision is unavoidable.
        jobs = self._multi_seed_jobs(batch_order="shuffled")

        with self.assertLogs(eu.logger, level="WARNING") as captured:
            batches = eu.completion_job_batches(jobs, 8)

        self.assertEqual([len(batch) for batch in batches], [8, 8])
        self.assertIn("shuffled batching", "\n".join(captured.output))
        self.assertEqual(
            sorted(int(job["request_index"]) for batch in batches for job in batch),
            list(range(16)),
        )

    # --- item 8: resume keeps the original batch composition ---------------

    def test_resume_batches_over_the_full_plan_instead_of_reshuffling(self):
        jobs = self._full_matrix_task2_jobs(batch_order="shuffled", batch_size=16)
        planned_batches = eu.completion_job_batches(jobs, 16)
        dropped = planned_batches[7]
        done_keys = {eu.completion_record_key(job) for job in dropped}
        pending = [
            job for job in jobs if eu.completion_record_key(job) not in done_keys
        ]

        resumed = eu.completion_job_batches(pending, 16, planned_jobs=jobs)

        self.assertEqual(len(resumed), len(planned_batches) - 1)
        self.assertEqual(
            [
                sorted(eu.completion_record_key(job) for job in batch)
                for batch in resumed
            ],
            [
                sorted(eu.completion_record_key(job) for job in batch)
                for batch in planned_batches
                if batch is not dropped
            ],
        )
        # Re-batching the pending subset on its own would NOT reproduce them.
        self.assertNotEqual(
            [
                sorted(eu.completion_record_key(job) for job in batch)
                for batch in resumed
            ],
            [
                sorted(eu.completion_record_key(job) for job in batch)
                for batch in eu.completion_job_batches(pending, 16)
            ],
        )

    def test_resume_drops_partially_completed_batches_to_their_pending_items(self):
        jobs = self._multi_seed_jobs(batch_order="shuffled")
        planned_batches = eu.completion_job_batches(jobs, 4)
        done = {
            int(planned_batches[0][0]["request_index"]),
            int(planned_batches[1][1]["request_index"]),
        }
        pending = [job for job in jobs if int(job["request_index"]) not in done]

        resumed = eu.completion_job_batches(pending, 4, planned_jobs=jobs)

        # Each planned batch keeps its identity, minus the items already done.
        self.assertEqual([len(batch) for batch in resumed], [3, 3, 4, 4])
        self.assertEqual(
            sorted(int(job["request_index"]) for batch in resumed for job in batch),
            sorted(int(job["request_index"]) for job in pending),
        )

    def test_run_completion_jobs_accepts_the_full_plan_for_resume(self):
        jobs = self._multi_seed_jobs(batch_order="shuffled")
        planned_batches = eu.completion_job_batches(jobs, 4)
        pending = [job for job in jobs if job not in planned_batches[0]]
        batch_sizes = []

        def fake_completion(**kwargs):
            items = json.loads(kwargs["prompt"].split("Items:\n", 1)[1])
            batch_sizes.append(len(items))
            return {
                "ok": True,
                "raw_text": json.dumps(
                    {
                        "results": [
                            {
                                "request_index": item["request_index"],
                                "requirement": "The system must export reports.",
                                "modality": "mandatory",
                                "confidence": 0.8,
                            }
                            for item in items
                        ]
                    }
                ),
                "response_json": {
                    "model": "m1",
                    "choices": [{"finish_reason": "stop"}],
                },
                "latency_s": 0.0,
                "error": "",
            }

        records = list(
            eu.run_completion_jobs(
                pending,
                max_workers=1,
                completion_fn=fake_completion,
                batch_size=4,
                planned_jobs=jobs,
            )
        )

        self.assertEqual(len(records), len(pending))
        self.assertEqual(sorted(batch_sizes), [4, 4, 4])
        self.assertEqual(
            len(eu.completion_job_batches(pending, 4, planned_jobs=jobs)), 3
        )

    # --- item 9: batched rows keep the driver's provenance ------------------

    def test_batched_records_carry_the_driver_seed_sha_and_retry_count(self):
        jobs = self._multi_seed_jobs()[:2]

        def fake_completion(**kwargs):
            items = json.loads(kwargs["prompt"].split("Items:\n", 1)[1])
            return {
                "ok": True,
                "raw_text": json.dumps(
                    {
                        "results": [
                            {
                                "request_index": item["request_index"],
                                "requirement": "The system must export reports.",
                                "modality": "mandatory",
                                "confidence": 0.8,
                            }
                            for item in items
                        ]
                    }
                ),
                "response_json": {
                    "model": "m1",
                    "choices": [{"finish_reason": "stop"}],
                },
                "latency_s": 0.0,
                "error": "",
                "retry_count": 2,
                "request_seed": 20260518,
                "request_payload_sha": "SHA_FROM_DRIVER",
            }

        records = eu.run_completion_batch(jobs, completion_fn=fake_completion)

        self.assertEqual(len(records), 2)
        self.assertEqual({row["retry_count"] for row in records}, {2})
        self.assertEqual({row["request_seed"] for row in records}, {20260518})
        # The batch request's payload sha, not a per-item recomputation.
        self.assertEqual(
            {row["request_payload_sha"] for row in records}, {"SHA_FROM_DRIVER"}
        )

    def test_instructor_batched_records_carry_the_driver_provenance(self):
        jobs = self._multi_seed_jobs(
            structured_output="instructor", prompt_version="v2-instructor-conf01"
        )[:2]

        def fake_completion(**kwargs):
            items = json.loads(kwargs["prompt"].split("Items:\n", 1)[1])
            return {
                "ok": True,
                "raw_text": json.dumps(
                    {
                        "results": [
                            {
                                "request_index": item["request_index"],
                                "requirement": "The system must export reports.",
                                "modality": "mandatory",
                                "confidence": 0.8,
                            }
                            for item in items
                        ]
                    }
                ),
                "response_json": {},
                "latency_s": 0.0,
                "error": "",
                "retry_count": 2,
                "request_seed": 20260518,
                "request_payload_sha": "SHA_FROM_DRIVER",
            }

        records = eu.run_instructor_completion_batch(
            jobs, completion_fn=fake_completion
        )

        self.assertEqual({row["parse_status"] for row in records}, {"ok"})
        self.assertEqual({row["retry_count"] for row in records}, {2})
        self.assertEqual({row["request_seed"] for row in records}, {20260518})
        self.assertEqual(
            {row["request_payload_sha"] for row in records}, {"SHA_FROM_DRIVER"}
        )

    # --- item 10: the SDK must not retry behind our back --------------------

    def _recording_openai_class(self):
        seen = {}

        class FakeClient:
            def __init__(self, **kwargs):
                seen.update(kwargs)
                self.chat = type("Chat", (), {"completions": self})()

            def create(self, **kwargs):
                return FakeResponse(dump={"model": "m1"})

        return FakeClient, seen

    def test_chat_completion_disables_the_sdk_retry_loop(self):
        FakeClient, seen = self._recording_openai_class()

        with mock.patch.object(eu, "OpenAI", FakeClient):
            result = eu.chat_completion(
                "http://x/v1", "m1", "prompt", 0.0, 1.0, max_retries=3
            )

        self.assertTrue(result["ok"])
        self.assertEqual(seen["max_retries"], 0)
        self.assertEqual(result["retry_count"], 0)

    def test_instructor_completion_disables_the_sdk_retry_loop(self):
        FakeClient, seen = self._recording_openai_class()

        class FakeInstructorClient:
            def __init__(self):
                self.chat = type("Chat", (), {"completions": self})()

            def create(self, **kwargs):
                raise RuntimeError("stop after client construction")

        with (
            mock.patch.object(eu, "OpenAI", FakeClient),
            mock.patch.dict(
                sys.modules,
                {
                    "instructor": mock.Mock(
                        from_openai=lambda *_args, **_kwargs: FakeInstructorClient()
                    )
                },
            ),
        ):
            result = eu.instructor_completion(
                "http://x/v1", "m1", "prompt", 0.0, 1.0, max_retries=0
            )

        self.assertFalse(result["ok"])
        self.assertEqual(seen["max_retries"], 0)


if __name__ == "__main__":
    unittest.main()
