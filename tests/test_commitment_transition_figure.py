"""Display checks for zero and rare commitment transitions."""

import unittest

from scripts.commitment_transition_figure import LEVELS, render_tikz


class CommitmentTransitionFigureTests(unittest.TestCase):
    def test_only_nonzero_strict_transitions_are_drawn(self):
        accounting = [
            {
                "model": "all",
                "dataset": "all",
                "variant": "all",
                "source_modality": level,
                "n_readable": 10000,
                "n_equal": 0,
                "n_down": 0,
                "n_broad_only_up": 0,
                "n_unknown": 0,
                "n_negated": 0,
                "n_invalid": 0,
            }
            for level in LEVELS
        ]
        matrix = [
            {
                "model": "all",
                "dataset": "all",
                "variant": "all",
                "source_modality": source,
                "text_modality": target,
                "n_strict": count,
            }
            for source, target, count in [
                ("nice_to_have", "optional", 1),
                ("nice_to_have", "recommended", 2000),
                ("optional", "recommended", 0),
            ]
        ]
        drawing = render_tikz(matrix, accounting, 9, 4)
        self.assertIn(r"$<0.1\%$", drawing)
        self.assertIn(r"20.0\%", drawing)
        self.assertEqual(drawing.count("node[rate, midway]"), 2)
        self.assertIn(r"\draw[transition, dashed] (level0.north)", drawing)
        self.assertNotIn("(level1.south)", drawing)
        accounting[0]["n_readable"] = 0
        with self.assertRaisesRegex(ValueError, "denominator"):
            render_tikz(matrix, accounting, 9, 4)
        empty = render_tikz([], accounting, 9, 4)
        self.assertNotIn("node[rate, midway]", empty)


if __name__ == "__main__":
    unittest.main()
