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


def _contains_json_dumps(node: ast.AST) -> bool:
    """True if `node` contains a `json.dumps(...)` call anywhere below it."""
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "dumps"
        for child in ast.walk(node)
    )


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


class ManifestWriterContractTest(unittest.TestCase):
    """Manifest JSON is serialized in exactly one place.

    Manifests participate in provenance hashing, so a second hand-rolled
    `json.dumps(...) + "\\n"` writer is a place where the bytes can drift.
    `eval_utils.write_json` is the single writer.
    """

    def test_no_script_hand_rolls_a_json_manifest_writer(self) -> None:
        for path in SCRIPT_PATHS:
            offenders = sorted(
                {
                    node.lineno
                    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "write_text"
                    and _contains_json_dumps(node)
                }
            )
            with self.subTest(module=path.name):
                self.assertEqual(
                    offenders,
                    [],
                    f"{path.name}: use eval_utils.write_json instead of "
                    f"write_text(json.dumps(...)) at line(s) {offenders}",
                )


if __name__ == "__main__":
    unittest.main()
