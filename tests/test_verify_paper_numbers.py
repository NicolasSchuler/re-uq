"""The no-MLX check of the paper tables: hand-counted fixture, match and mismatch."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import eval_utils as eu, verify_paper_numbers as verify

# One model, one cell. Hand counts: unsupported acceptance 1/2, no cue 1/4,
# strict 2/3, broad 2/3, weak strict 2/2 (escalation 1/2, frame-only 1/2),
# high confidence 1/2, agreement 1/2, called strengthened 1/2, called
# preserved 1/2; meaning-variation AUROC 0.5 and confidence AUROC 1.0.
HAND_COUNTS = {
    "task1_unsupported_acceptance_90": (1, 2),
    "task2_no_cue": (1, 4),
    "task2_strict_strengthening": (2, 3),
    "task2_broad_strengthening": (2, 3),
    "task2_weak_strict_strengthening": (2, 2),
    "task2_weak_strict_escalation": (1, 2),
    "task2_weak_strict_frame_only": (1, 2),
    "task2_strict_high_conf_90": (1, 2),
    "task2_strict_agreement": (1, 2),
    "task3_strict_flagged": (1, 2),
    "task3_strict_called_preserved": (1, 2),
}
HAND_AUROCS = {
    "task2_meaning_variation_auroc": 0.5,
    "task2_verbalized_confidence_auroc": 1.0,
}


def _row(task, method, item, **fields):
    return {
        "run_id": "run-1",
        "task": task,
        "uq_method": method,
        "item_id": item,
        "source_item_id": "",
        "source_modality": "",
        "text_modality": "",
        "text_modality_parse_status": "",
        "strict_text_overcommit": "",
        "text_overcommit": "",
        "y_true": "",
        "y_pred": "",
        "confidence": "",
        "uncertainty_score": "",
        "stochastic_complete": "",
        "pred_relation": "",
        **fields,
    }


def _score_rows():
    det = "verbalized_confidence"
    weak, rec, man = "nice_to_have", "recommended", "mandatory"
    return [
        _row("task1", det, "t1a", y_true="0", y_pred="1", confidence="0.95"),
        _row("task1", det, "t1b", y_true="0", y_pred="0", confidence="0.99"),
        _row(
            "task2",
            det,
            "a",
            source_modality=weak,
            text_modality="recommended",
            text_modality_parse_status="ok",
            strict_text_overcommit="1",
            text_overcommit="1",
            confidence="0.95",
            uncertainty_score="0.05",
        ),
        _row(
            "task2",
            det,
            "b",
            source_modality=weak,
            text_modality="optional",
            text_modality_parse_status="ok",
            strict_text_overcommit="1",
            text_overcommit="1",
            confidence="0.8",
            uncertainty_score="0.2",
        ),
        _row(
            "task2",
            det,
            "c",
            source_modality=rec,
            text_modality="recommended",
            text_modality_parse_status="ok",
            strict_text_overcommit="0",
            text_overcommit="0",
            confidence="1.0",
            uncertainty_score="0.0",
        ),
        _row(
            "task2",
            det,
            "d",
            source_modality=man,
            text_modality="unknown",
            text_modality_parse_status="unknown",
            strict_text_overcommit="0",
            text_overcommit="0",
            confidence="1.0",
            uncertainty_score="0.0",
        ),
        _row(
            "task2",
            "modality_consistency",
            "a",
            stochastic_complete="1",
            uncertainty_score="0.0",
        ),
        _row(
            "task2",
            "modality_consistency",
            "b",
            stochastic_complete="1",
            uncertainty_score="0.4",
        ),
        _row("task2", eu.ACSE_PROXY_METHOD, "a", uncertainty_score="0.3"),
        _row("task2", eu.ACSE_PROXY_METHOD, "b", uncertainty_score="0.1"),
        _row("task2", eu.ACSE_PROXY_METHOD, "c", uncertainty_score="0.2"),
        # Task 3 rows carry their own run id; the join must not depend on it.
        {
            **_row("task3", det, "v1", source_item_id="a", pred_relation="strengthens"),
            "run_id": "task3-1",
        },
        {
            **_row("task3", det, "v2", source_item_id="b", pred_relation="preserves"),
            "run_id": "task3-1",
        },
    ]


def _tracked_rq_rows(counts=HAND_COUNTS):
    row = {"dataset": "all", "variant": "all"}
    for name, (n, d) in counts.items():
        row[f"{name}_n"], row[f"{name}_denominator"] = n, d
    row.update(HAND_AUROCS)
    return [{**row, "model": "m"}, {**row, "model": "all"}]


def _tracked_modality_rows():
    cells = {"nice_to_have": (2, 2, 2), "recommended": (0, 0, 1)}
    return [
        {
            "model": model,
            "dataset": "all",
            "variant": "all",
            "source_modality": modality,
            "strict_strengthening_n": strict,
            "strict_strengthening_denominator": d,
            "broad_strengthening_n": broad,
            "broad_strengthening_denominator": d,
        }
        for model in ("m", "all")
        for modality, (strict, broad, d) in cells.items()
    ]


class VerifyPaperNumbersTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.manifest = self.root / "manifest.csv"
        eu.write_csv_rows(
            self.manifest,
            [
                {
                    "model": "m",
                    "dataset_id": "nice",
                    "benchmark_variant": "must",
                    "run_id": "run-1",
                    "analysis_dir": "outputs/evaluation_x",
                }
            ],
        )
        eu.write_csv_rows(
            self.root / "outputs/evaluation_x/uq_scores.csv", _score_rows()
        )
        self.rq_table = self.root / "rq.csv"
        self.modality_table = self.root / "modality.csv"
        eu.write_csv_rows(self.modality_table, _tracked_modality_rows())

    def _lines(self, rq_rows):
        eu.write_csv_rows(self.rq_table, rq_rows)
        rows = verify.load_score_rows(self.manifest, self.root)
        return verify.compare(rows, self.rq_table, self.modality_table)

    def test_hand_counted_fixture_matches_every_number(self) -> None:
        lines = self._lines(_tracked_rq_rows())
        failed = [line for line in lines if not line[4]]
        self.assertEqual(failed, [])
        # 11 rates + 2 AUROCs + 2 modalities x (strict, broad), for model and pool.
        self.assertEqual(len(lines), 2 * (11 + 2 + 4))

    def test_a_changed_tracked_number_is_reported(self) -> None:
        counts = {**HAND_COUNTS, "task2_strict_strengthening": (3, 3)}
        failed = [line for line in self._lines(_tracked_rq_rows(counts)) if not line[4]]
        self.assertEqual(
            {(group, metric) for group, metric, *_ in failed},
            {
                ("m", "task2_strict_strengthening"),
                ("all", "task2_strict_strengthening"),
            },
        )

    def test_missing_scores_point_to_the_archive(self) -> None:
        (self.root / "outputs/evaluation_x/uq_scores.csv").unlink()
        with self.assertRaises(SystemExit) as caught:
            verify.load_score_rows(self.manifest, self.root)
        self.assertIn("Zenodo raw archive", str(caught.exception))

    def test_auroc_agrees_with_the_exporter_on_ties(self) -> None:
        labels = [1, 1, 0, 0, 1, 0]
        scores = [0.4, 0.2, 0.2, 0.1, 0.4, 0.4]
        self.assertAlmostEqual(
            verify.auroc(labels, scores), eu.auroc_score(labels, scores)
        )
        self.assertIsNone(verify.auroc([1, 1], [0.1, 0.2]))


if __name__ == "__main__":
    unittest.main()
