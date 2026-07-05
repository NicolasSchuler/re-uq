"""Tests for scripts/aggregate_paper_headline_metrics.py.

The per-cell paper snapshots the production script reads
(outputs/paper_task2_text_drift_metrics.csv, ...) are *not* tracked in git, so
these tests build a small synthetic fixture that mirrors the real snapshot
columns and the exact shipped values. The asserted headline reconstructions are
the known review targets: 8.6% strict, 13.9% macro / 13.8% pooled broad,
29.8% weak (mlm_tapt/must cell), 98.4% macro high-confidence, 100.0% agreement.
"""

import io
import unittest
from contextlib import redirect_stdout

from scripts import aggregate_paper_headline_metrics as agg


# Synthetic fixtures mirroring the shipped per-cell snapshot columns and values.
TASK2_ROWS = [
    {"dataset": "mlm_tapt", "variant": "must", "text_over_commitment": "0.2088235294117647", "strict_text_over_commitment": "0.10686274509803921"},
    {"dataset": "mlm_tapt", "variant": "shall", "text_over_commitment": "0.09302325581395349", "strict_text_over_commitment": "0.07267441860465117"},
    {"dataset": "nice", "variant": "must", "text_over_commitment": "0.17162698412698413", "strict_text_over_commitment": "0.09424603174603174"},
    {"dataset": "nice", "variant": "shall", "text_over_commitment": "0.08079847908745247", "strict_text_over_commitment": "0.07034220532319392"},
]

CONFIDENCE_ROWS = [
    {"dataset": "mlm_tapt", "variant": "must", "n": "4080", "broad_text_oc_n": "852", "strict_text_oc_n": "436", "strict_text_oc_conf_ge_90": "1.0", "strict_text_oc_unanimous_modality_samples": "1.0"},
    {"dataset": "mlm_tapt", "variant": "shall", "n": "4128", "broad_text_oc_n": "384", "strict_text_oc_n": "300", "strict_text_oc_conf_ge_90": "0.9733333333333334", "strict_text_oc_unanimous_modality_samples": "1.0"},
    {"dataset": "nice", "variant": "must", "n": "4032", "broad_text_oc_n": "692", "strict_text_oc_n": "380", "strict_text_oc_conf_ge_90": "0.9894736842105263", "strict_text_oc_unanimous_modality_samples": "1.0"},
    {"dataset": "nice", "variant": "shall", "n": "4208", "broad_text_oc_n": "340", "strict_text_oc_n": "296", "strict_text_oc_conf_ge_90": "0.972972972972973", "strict_text_oc_unanimous_modality_samples": "1.0"},
]

BLIND_ROWS = [
    {"dataset": "mlm_tapt", "variant": "must", "weak_strict_text_strengthening_90": "29.8%"},
    {"dataset": "mlm_tapt", "variant": "shall", "weak_strict_text_strengthening_90": "28.3%"},
    {"dataset": "nice", "variant": "must", "weak_strict_text_strengthening_90": "31.7%"},
    {"dataset": "nice", "variant": "shall", "weak_strict_text_strengthening_90": "27.4%"},
]


class AggregatePaperHeadlineMetricsTest(unittest.TestCase):
    def setUp(self):
        rows = agg.build_headline_rows(TASK2_ROWS, CONFIDENCE_ROWS, BLIND_ROWS)
        self.rows = {row["headline_key"]: row for row in rows}

    def test_strict_text_strengthening_matches_pooled_and_macro(self):
        row = self.rows["strict_text_strengthening"]
        # Pooled = 1412 / 16448; macro of per-cell strict rates.
        self.assertAlmostEqual(float(row["value_pooled"]), 0.08585, places=4)
        self.assertAlmostEqual(float(row["value_macro_over_cells"]), 0.08603, places=4)
        self.assertEqual(round(float(row["value_pooled"]), 3), 0.086)
        self.assertEqual(round(float(row["value_macro_over_cells"]), 3), 0.086)

    def test_broad_text_strengthening_pooled_and_macro_diverge(self):
        row = self.rows["broad_text_strengthening"]
        # README 13.9% is the macro; the pooled figure is 13.8%.
        self.assertAlmostEqual(float(row["value_pooled"]), 0.13789, places=4)
        self.assertAlmostEqual(float(row["value_macro_over_cells"]), 0.13857, places=4)
        self.assertEqual(round(float(row["value_macro_over_cells"]), 3), 0.139)
        self.assertEqual(round(float(row["value_pooled"]), 3), 0.138)

    def test_weak_strict_strengthening_is_single_cell(self):
        row = self.rows["weak_strict_text_strengthening_90"]
        self.assertEqual(row["value_pooled"], "")
        self.assertEqual(row["confidence_threshold"], "0.90")
        per_cell = [float(v) for v in row["per_cell_values"].split("|")]
        # mlm_tapt/must cell (first) is the README 29.8% figure.
        self.assertEqual(round(per_cell[0], 3), 0.298)
        # Cross-cell range 27.4%-31.7%; macro ~29.3%.
        self.assertAlmostEqual(min(per_cell), 0.274, places=4)
        self.assertAlmostEqual(max(per_cell), 0.317, places=4)
        self.assertAlmostEqual(float(row["value_macro_over_cells"]), 0.2930, places=4)

    def test_high_confidence_share_macro(self):
        row = self.rows["strict_text_oc_high_conf_90"]
        # README 98.4% is the unweighted macro; pooled is 98.6%.
        self.assertAlmostEqual(float(row["value_macro_over_cells"]), 0.98394, places=4)
        self.assertEqual(round(float(row["value_macro_over_cells"]), 3), 0.984)
        self.assertAlmostEqual(float(row["value_pooled"]), 0.98584, places=4)
        self.assertEqual(row["confidence_threshold"], "0.90")
        # Per-cell n span 296-436.
        cell_n = [int(v) for v in row["cell_n"].split("|")]
        self.assertEqual(min(cell_n), 296)
        self.assertEqual(max(cell_n), 436)

    def test_repeated_sample_agreement_unanimous(self):
        row = self.rows["strict_text_oc_repeated_sample_agreement"]
        self.assertEqual(float(row["value_pooled"]), 1.0)
        self.assertEqual(float(row["value_macro_over_cells"]), 1.0)
        per_cell = [float(v) for v in row["per_cell_values"].split("|")]
        self.assertTrue(all(v == 1.0 for v in per_cell))

    def test_stdout_comparison_reports_all_matches(self):
        rows = agg.build_headline_rows(TASK2_ROWS, CONFIDENCE_ROWS, BLIND_ROWS)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            agg.print_readme_comparison(rows)
        output = buffer.getvalue()
        self.assertNotIn("MISMATCH", output)
        for key in agg.README_VALUES:
            self.assertIn(key, output)


if __name__ == "__main__":
    unittest.main()
