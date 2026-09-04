"""Expected-failure regressions for confirmed high-risk contract gaps.

Each expected failure documents a current defect rather than accepting the
behavior. The passing malformed-JSON test pins the surrounding external-probe
validation path.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from scripts import (
    compare_run_matrix as compare_matrix,
    eval_utils as eu,
    evaluate_external_ai_probe as external_eval,
    generate_evaluation_analysis as analysis_cli,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _gold_row() -> dict[str, str]:
    return {
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


def _external_output(**overrides: object) -> dict[str, object]:
    return {
        "external_item_id": "EXT0001",
        "requirement": "The system MAY export reports.",
        "modality": "optional",
        "confidence": 0.9,
        **overrides,
    }


def _write_external_fixture(root: Path, records: list[str]) -> tuple[Path, Path]:
    gold_path = root / "gold.csv"
    output_path = root / "outputs.jsonl"
    eu.write_csv_rows(gold_path, [_gold_row()])
    output_path.write_text("\n".join(records) + "\n", encoding="utf-8")
    return output_path, gold_path


class BenchmarkManifestFailClosedTest(unittest.TestCase):
    @unittest.expectedFailure
    def test_existing_required_artifact_with_blank_digest_is_rejected(self) -> None:
        """Known bug: a blank digest silently disables integrity verification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benchmark = root / "data/processed/benchmark_items.csv"
            benchmark.parent.mkdir(parents=True)
            benchmark.write_text("item_id\nS0001_optional\n", encoding="utf-8")
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "artifacts": [
                            {
                                "path": "data/processed/benchmark_items.csv",
                                "sha256": "",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "sha256|digest"):
                eu.verify_benchmark_manifest(manifest_path, root)

    @unittest.expectedFailure
    def test_manifest_with_no_artifacts_is_rejected(self) -> None:
        """Known bug: an empty manifest reports success with zero checked files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path = root / "manifest.json"
            manifest_path.write_text('{"artifacts": []}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "artifact|empty"):
                eu.verify_benchmark_manifest(manifest_path, root)


class ExternalProbeContractTest(unittest.TestCase):
    def test_malformed_json_line_is_a_paper_readiness_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path, gold_path = _write_external_fixture(
                Path(tmpdir), ["{not json", json.dumps(_external_output())]
            )
            scored, validation = external_eval.evaluate_outputs(output_path, gold_path)

        self.assertEqual(len(scored), 1)
        self.assertEqual(validation["parse_errors"], 1)
        self.assertFalse(validation["paper_ready"])
        self.assertIn("parse_errors", validation["paper_ready_blockers"])

    @unittest.expectedFailure
    def test_external_output_with_undeclared_fields_is_not_paper_ready(self) -> None:
        """Known bug: the advertised exact JSON schema is treated as a subset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path, gold_path = _write_external_fixture(
                Path(tmpdir),
                [json.dumps(_external_output(unexpected_field="leaked metadata"))],
            )
            _, validation = external_eval.evaluate_outputs(output_path, gold_path)

        self.assertFalse(validation["paper_ready"])
        self.assertIn("unexpected_fields", validation["paper_ready_blockers"])

    @unittest.expectedFailure
    def test_external_output_with_empty_requirement_is_not_paper_ready(self) -> None:
        """Known bug: an empty extraction can currently pass every validation gate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path, gold_path = _write_external_fixture(
                Path(tmpdir), [json.dumps(_external_output(requirement=""))]
            )
            _, validation = external_eval.evaluate_outputs(output_path, gold_path)

        self.assertFalse(validation["paper_ready"])
        self.assertIn("empty_requirements", validation["paper_ready_blockers"])

    @unittest.expectedFailure
    def test_external_output_with_whitespace_requirement_is_not_paper_ready(
        self,
    ) -> None:
        """Known bug: whitespace-only extractions pass every validation gate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path, gold_path = _write_external_fixture(
                Path(tmpdir), [json.dumps(_external_output(requirement="   \t"))]
            )
            _, validation = external_eval.evaluate_outputs(output_path, gold_path)

        self.assertFalse(validation["paper_ready"])
        self.assertIn("empty_requirements", validation["paper_ready_blockers"])


class AnalysisProvenanceContractTest(unittest.TestCase):
    @staticmethod
    def _probability_row() -> dict[str, object]:
        return {
            "run_id": "full-1",
            "task": "task2",
            "item_id": "S0001_optional",
            "prompt_version": "v2-conf01",
            "parse_status": "ok",
            "parsed_json": {
                "requirement": "The system MAY export reports.",
                "modality": "optional",
                "confidence": 0.9,
            },
        }

    def _assert_probability_row_rejected(self, row: dict[str, object]) -> None:
        with self.assertRaisesRegex(ValueError, "confidence.*scale|scale.*confidence"):
            analysis_cli.require_probability_confidence("fixture", [row])

    @unittest.expectedFailure
    def test_v2_probability_requires_its_confidence_scale_marker(self) -> None:
        """Known bug: probability rows pass when confidence_scale is absent."""
        self._assert_probability_row_rejected(self._probability_row())

    @unittest.expectedFailure
    def test_v2_probability_rejects_a_blank_confidence_scale(self) -> None:
        """Known bug: a blank confidence_scale is accepted as probability provenance."""
        row = self._probability_row()
        row["confidence_scale"] = ""
        self._assert_probability_row_rejected(row)

    @unittest.expectedFailure
    def test_v2_probability_rejects_a_percentage_confidence_scale(self) -> None:
        """Known bug: 0_100 provenance is accepted for a v2 probability row."""
        row = self._probability_row()
        row["confidence_scale"] = eu.CONFIDENCE_SCALE_0_100
        self._assert_probability_row_rejected(row)

    @unittest.expectedFailure
    def test_v2_probability_rejects_an_unknown_confidence_scale(self) -> None:
        """Known bug: unknown confidence provenance is ignored for a v2 row."""
        row = self._probability_row()
        row["confidence_scale"] = "unknown"
        self._assert_probability_row_rejected(row)


class ExpectedStochasticSampleContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.root = Path(self._tmpdir.name)
        (self.root / "data/processed").mkdir(parents=True)
        (self.root / "prompts").mkdir()
        for name in ("mandatory_entailment.txt", "modality_extraction.txt"):
            shutil.copyfile(REPO_ROOT / "prompts" / name, self.root / "prompts" / name)
        seed = {
            "seed_id": "S0001",
            "source_dataset": "NICE",
            "original_requirement": "The system shall export reports.",
            "capability_text_final": "export reports",
        }
        self.item = next(
            row
            for row in eu.build_benchmark_items([seed])
            if row["source_modality"] == "optional"
        )
        eu.write_csv_rows(self.root / "data/processed/benchmark_items.csv", [self.item])
        self._write_four_of_five_stochastic_samples()

    def _write_four_of_five_stochastic_samples(self) -> None:
        task2_template = eu.load_prompt(self.root / "prompts/modality_extraction.txt")
        samples = [
            ("deterministic", 0, "optional"),
            ("stochastic", 0, "optional"),
            ("stochastic", 1, "optional"),
            ("stochastic", 2, "recommended"),
            ("stochastic", 3, "optional"),
        ]
        for sample_kind, sample_index, modality in samples:
            modal = "SHOULD" if modality == "recommended" else "MAY"
            parsed = {
                "requirement": f"The system {modal} export reports.",
                "modality": modality,
                "confidence": 0.9,
            }
            raw = eu.build_raw_record(
                run_id="full-1",
                model="model-1",
                host="http://offline.invalid/v1",
                task="task2",
                item=self.item,
                sample_index=sample_index,
                sample_kind=sample_kind,
                temperature=0.0 if sample_kind == "deterministic" else 0.7,
                top_p=1.0,
                prompt_version="v2-conf01",
                prompt=eu.prompt_for_benchmark_task(
                    "task2", self.item, "", task2_template
                ),
                completion={
                    "ok": True,
                    "raw_text": json.dumps(parsed),
                    "latency_s": 0.0,
                    "error": "",
                },
                provider_id="provider-1",
                profile_id="profile-1",
                run_group_id="group-1",
            )
            eu.append_jsonl(self.root / "data/processed/model_outputs_raw.jsonl", raw)

    def test_analysis_cli_counts_a_never_written_sample_as_incomplete(self) -> None:
        """Known bug: the CLI records the expected count but does not score with it."""
        output_dir = self.root / "outputs/analysis"
        argv = [
            "generate_evaluation_analysis.py",
            "--run-id",
            "full-1",
            "--model",
            "model-1",
            "--profile",
            "profile-1",
            "--output-dir",
            str(output_dir),
            "--bootstrap-iterations",
            "1",
            "--expected-stochastic-samples",
            "5",
            "--allow-partial",
            "--skip-registry-check",
            "--skip-construct-review-check",
            "--skip-manifest-check",
            "--max-parse-failure-rate",
            "0",
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(analysis_cli.eu, "project_root", return_value=self.root),
            redirect_stdout(io.StringIO()),
        ):
            analysis_cli.main()

        scores = eu.read_csv_rows(output_dir / "uq_scores.csv")
        consistency = next(
            row for row in scores if row["uq_method"] == "modality_consistency"
        )
        self.assertEqual(consistency["total_n"], "5")
        self.assertEqual(consistency["stochastic_complete"], "False")

    def test_run_matrix_counts_a_never_written_sample_as_incomplete(self) -> None:
        """Known bug: matrix comparison also omits the configured sample count."""
        eu.write_csv_rows(
            self.root / "data/processed/run_registry.csv",
            [
                {
                    "run_id": "full-1",
                    "run_group_id": "group-1",
                    "provider_id": "provider-1",
                    "profile_id": "profile-1",
                    "model": "model-1",
                    "dataset_id": "nice",
                    "benchmark_variant": "must",
                    "status": "complete",
                    "expected_stochastic_samples": 5,
                }
            ],
            fieldnames=eu.RUN_REGISTRY_FIELDS,
        )
        config_path = self.root / "run_config.json"
        config_path.write_text(
            json.dumps(
                {
                    "run_group_id": "group-1",
                    "datasets": ["nice"],
                    "benchmark_variants": ["must"],
                    "stochastic": {
                        "temperature": 0.7,
                        "top_p": 1.0,
                        "samples": 5,
                    },
                    "profiles": [
                        {
                            "profile_id": "profile-1",
                            "provider_id": "provider-1",
                            "base_url": "http://offline.invalid/v1",
                            "api_key_env": "OFFLINE_KEY",
                            "models": ["model-1"],
                            "concurrency": 1,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        argv = [
            "compare_run_matrix.py",
            "--config",
            str(config_path),
            "--bootstrap-samples",
            "1",
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(
                compare_matrix.eu, "project_root", return_value=self.root
            ),
            redirect_stdout(io.StringIO()),
        ):
            compare_matrix.main()

        rows = eu.read_csv_rows(self.root / "outputs/run_matrix_summary.csv")
        consistency = next(
            row
            for row in rows
            if row["model"] == "model-1" and row["uq_method"] == "modality_consistency"
        )
        self.assertEqual(consistency["agreement_n_complete"], "0")
        self.assertEqual(consistency["agreement_n_incomplete_excluded"], "1")


class ReproductionWrapperContractTest(unittest.TestCase):
    @unittest.expectedFailure
    def test_full_command_forwards_the_selected_benchmark_variant(self) -> None:
        """Known bug: RE_UQ_VARIANT is printed but omitted from the paid runner."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "scripts").mkdir()
            (root / "run_configs").mkdir()
            (root / ".venv/bin").mkdir(parents=True)
            shutil.copyfile(
                REPO_ROOT / "scripts/reproduce.sh", root / "scripts/reproduce.sh"
            )
            (root / "run_configs/current.json").write_text("{}\n", encoding="utf-8")
            capture_path = root / "captured_args.txt"
            fake_python = root / ".venv/bin/python"
            fake_python.write_text(
                '#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE_PATH"\n',
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            env = {
                **os.environ,
                "CAPTURE_PATH": str(capture_path),
                "RE_UQ_CONFIG": "run_configs/current.json",
                "RE_UQ_VARIANT": "shall",
            }

            completed = subprocess.run(
                ["bash", "scripts/reproduce.sh", "full"],
                cwd=root,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            captured = capture_path.read_text(encoding="utf-8").splitlines()
            self.assertIn(
                "--variant", captured, f"runner argv omitted variant: {captured}"
            )
            variant_index = captured.index("--variant")
            self.assertEqual(captured[variant_index + 1], "shall")


if __name__ == "__main__":
    unittest.main()
