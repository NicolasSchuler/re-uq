"""Contract tests that every module under scripts/ stays loadable and callable.

Several scripts -- the figure generators in particular -- have no other test
coverage, so a broken import or a renamed entry point would only surface when
someone regenerates a paper figure. These tests are deliberately shallow: they
assert the module boundary, not the plotting itself.
"""

from __future__ import annotations

import ast
import importlib
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
SCRIPT_PATHS = sorted(SCRIPTS_DIR.glob("*.py"))


def _has_main_guard(path: Path) -> bool:
    """True if the module ends in the `if __name__ == "__main__":` CLI guard."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
        ):
            return True
    return False


class ScriptModuleContractTest(unittest.TestCase):
    """Every script must import, and every CLI must expose a callable main()."""

    def test_scripts_directory_is_discovered(self) -> None:
        # Guards against the glob silently matching nothing after a move.
        self.assertGreater(len(SCRIPT_PATHS), 10, SCRIPTS_DIR)

    def test_every_script_module_imports(self) -> None:
        for path in SCRIPT_PATHS:
            with self.subTest(module=path.name):
                importlib.import_module(f"scripts.{path.stem}")

    def test_every_cli_script_exposes_a_callable_main(self) -> None:
        for path in SCRIPT_PATHS:
            if not _has_main_guard(path):
                continue
            with self.subTest(module=path.name):
                module = importlib.import_module(f"scripts.{path.stem}")
                main = getattr(module, "main", None)
                self.assertIsNotNone(main, f"{path.name} has a CLI guard but no main")
                self.assertTrue(callable(main), f"{path.name}: main is not callable")


if __name__ == "__main__":
    unittest.main()
