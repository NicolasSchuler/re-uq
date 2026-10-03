"""Keep public scientific prose and ablation recipes tied to shipped evidence."""

from __future__ import annotations

import csv
import json
import re
import shlex
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(name: str) -> list[dict[str, str]]:
    with (ROOT / "outputs" / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def paragraph(name: str, phrase: str) -> str:
    text = (ROOT / "docs" / name).read_text(encoding="utf-8")
    matches = [part for part in text.split("\n\n") if phrase in part]
    if len(matches) != 1:
        raise AssertionError(f"Expected one paragraph containing {phrase!r} in {name}")
    return " ".join(matches[0].split())


class DocumentedResultsTest(unittest.TestCase):
    def test_agreement_uses_the_complete_strengthened_cohort(self):
        pooled = next(
            row
            for row in rows("paper_per_model_rq_table.csv")
            if row["model"] == row["dataset"] == row["variant"] == "all"
        )
        description = paragraph("experimental_setup.md", "unanimous sampled")
        for column in (
            "task2_strict_agreement_n",
            "task2_strict_agreement_denominator",
        ):
            self.assertIn(f"{int(pooled[column]):,}", description)
        self.assertIn(
            f"{100 * float(pooled['task2_strict_agreement_rate']):.1f}%", description
        )
        self.assertEqual(0, int(pooled["task2_strict_agreement_incomplete_excluded"]))
        self.assertIn("complete sample groups", description)
        self.assertIn("declared labels", description)

    def test_heuristic_and_unclassified_counts_remain_distinct(self):
        cells = rows("paper_task2_text_drift_metrics.csv")
        total = sum(int(row["n"]) for row in cells)
        unknown = sum(
            int(row["strict_text_over_commitment_n_unknown_excluded"]) for row in cells
        )
        heuristic = sum(
            round(
                float(row["heuristic_text_modality_rate"])
                * int(row["strict_text_over_commitment_n_denominator"])
            )
            for row in cells
        )
        for name, phrase in (
            ("experimental_setup.md", "heuristic-only classification"),
            ("faq.md", "this heuristic-only classification"),
        ):
            with self.subTest(document=name):
                description = paragraph(name, phrase)
                for count in (total, heuristic, unknown):
                    self.assertIn(f"{count:,}", description)
                for count in (heuristic, unknown):
                    self.assertIn(f"{100 * count / total:.1f}%", description)
                self.assertIn("unclassified wording", description)
                self.assertIn("denominator", description)

    def test_length_comparison_pools_valid_word_counts(self):
        pooled = [
            row
            for row in rows("paper_per_model_modality_pooled.csv")
            if row["model"] == row["dataset"] == row["variant"] == "all"
        ]
        description = paragraph("experimental_setup.md", "Answer length is recorded")
        for weak in (True, False):
            selected = [
                row
                for row in pooled
                if (row["source_modality"] == "nice_to_have") == weak
            ]
            count = sum(int(row["requirement_word_count_n"]) for row in selected)
            mean = (
                sum(
                    int(row["requirement_word_count_n"])
                    * float(row["mean_requirement_word_count"])
                    for row in selected
                )
                / count
            )
            self.assertIn(f"{mean:.2f} words", description)

    def test_comparison_recipes_select_the_reported_groups(self):
        pattern = re.compile(
            r"\.venv/bin/python scripts/compare_(context|batching)_ablation\.py([^`\n]*)"
        )
        checked = 0
        for name in ("reproduction.md", "context_ablation.md"):
            text = (ROOT / "docs" / name).read_text(encoding="utf-8")
            for kind, arguments in pattern.findall(text.replace("\\\n", " ")):
                with self.subTest(document=name, comparison=kind):
                    provenance = (
                        ROOT / "outputs" / f"{kind}_ablation_summary_provenance.json"
                    )
                    runs = json.loads(provenance.read_text(encoding="utf-8"))["runs"]
                    groups = {run["run_group_id"] for run in runs}
                    tokens = shlex.split(arguments)
                    self.assertIn("--run-group-id", tokens)
                    self.assertEqual(
                        groups, {tokens[tokens.index("--run-group-id") + 1]}
                    )
                    checked += 1
        self.assertEqual(3, checked)

    def test_context_inference_names_the_exported_cluster(self):
        description = paragraph("context_ablation.md", "Their delta rows")
        deltas = rows("context_ablation_summary_deltas.csv")
        for cluster in {row["delta_cluster_field"] for row in deltas}:
            self.assertIn(f"delta_cluster_field={cluster}", description)
        for note in {row["cluster_note"] for row in deltas}:
            self.assertIn(note, description)


if __name__ == "__main__":
    unittest.main()
