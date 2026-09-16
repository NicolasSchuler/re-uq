"""Relative links in tracked Markdown must resolve to tracked files.

The documentation is read on GitHub, where an ignored or moved file is a
dead link. Every ``[text](path)`` and ``![alt](path)`` in a tracked ``.md``
file is resolved against that file's directory and must name a tracked file
or a directory that holds tracked files. Fenced code blocks are skipped.
"""

from __future__ import annotations

import os
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
FENCE = re.compile(r"```.*?```", re.S)
EXTERNAL = ("http://", "https://", "mailto:", "#")


def tracked_files() -> set[Path]:
    try:
        completed = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return set()
    return {Path(name) for name in completed.stdout.decode().split("\0") if name}


def link_targets(markdown: Path) -> list[str]:
    text = FENCE.sub("", markdown.read_text(encoding="utf-8"))
    return [target for target in LINK.findall(text) if not target.startswith(EXTERNAL)]


class DocsLinkTest(unittest.TestCase):
    def test_relative_links_resolve_to_tracked_files(self):
        tracked = tracked_files()
        if not tracked:
            self.skipTest("not inside a git checkout")
        directories = {parent for path in tracked for parent in path.parents}
        broken: list[str] = []
        for relative in sorted(path for path in tracked if path.suffix == ".md"):
            for target in link_targets(ROOT / relative):
                target_path = target.split("#", 1)[0]
                if not target_path:
                    continue
                resolved = Path(os.path.normpath(relative.parent / target_path))
                if resolved not in tracked and resolved not in directories:
                    broken.append(f"{relative}: {target}")
        self.assertEqual([], broken, "\n" + "\n".join(broken))


if __name__ == "__main__":
    unittest.main()
