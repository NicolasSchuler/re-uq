"""The README's results table quotes tracked numbers, never typed ones.

Each row of the "Results at a glance" table ends in a marker naming the
``outputs/paper_numbers.tex`` macros it quotes (``<!-- num:... -->``) or a
column of a tracked CSV (``<!-- csv:file:row_key:column -->``). Every numeric
token of the macro value must appear in the row. A second test checks that the
tracked macro file regenerates from the tracked tables, so the chain from CSV
to README stays closed.
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
NUMBERS = ROOT / "outputs" / "paper_numbers.tex"
MACRO = re.compile(r"\\newcommand\{\\(num\w+)\}\{([^}]*)\}")
MARKER = re.compile(r"<!--\s*(num|csv):(.*?)-->")
TOKEN = re.compile(r"[0-9][0-9,]*(?:\.[0-9]+)?")


def macro_values(text: str) -> dict[str, str]:
    return dict(MACRO.findall(text))


class ReadmeNumbersTest(unittest.TestCase):
    def test_marked_rows_quote_the_tracked_values(self):
        values = macro_values(NUMBERS.read_text(encoding="utf-8"))
        checked = 0
        for line in README.read_text(encoding="utf-8").splitlines():
            for kind, spec in MARKER.findall(line):
                row_text = line.split("<!--", 1)[0]
                if kind == "num":
                    for name in spec.split():
                        self.assertIn(
                            name, values, f"README names unknown macro {name}"
                        )
                        for token in TOKEN.findall(values[name]):
                            self.assertIn(
                                token,
                                row_text,
                                f"{name}={values[name]!r} is not quoted in: {row_text}",
                            )
                        checked += 1
                else:
                    file_name, row_key, column = spec.strip().split(":")
                    with (ROOT / "outputs" / file_name).open(encoding="utf-8") as fh:
                        rows = {row["headline_key"]: row for row in csv.DictReader(fh)}
                    expected = f"{100 * float(rows[row_key][column]):.1f}"
                    self.assertIn(
                        expected, row_text, f"{spec} = {expected} is not quoted"
                    )
                    checked += 1
        self.assertGreaterEqual(
            checked, 15, "the README results table lost its markers"
        )

    def test_tracked_macros_regenerate_from_the_tracked_tables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "numbers.tex"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/export_paper_numbers.py",
                    "--strict",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
            )
            regenerated = MACRO.findall(output.read_text(encoding="utf-8"))
        self.assertEqual(
            regenerated, MACRO.findall(NUMBERS.read_text(encoding="utf-8"))
        )


if __name__ == "__main__":
    unittest.main()
