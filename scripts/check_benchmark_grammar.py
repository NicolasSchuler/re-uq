"""Grammar check of every benchmark sentence a model saw, with LanguageTool.

Covers the rendered source statements and Task 1 candidates of the NICE, MLM TAPT
and PURE item files (both keyword variants) and the three alternative weak-intent
phrasings of the phrasing probe. LanguageTool runs locally at its default rule
level (no picky rules) through the optional ``language-tool-python`` package,
which needs Java and downloads LanguageTool on first use:

    uv run --with language-tool-python python scripts/check_benchmark_grammar.py

Every match is kept in the CSV, but only the GRAMMAR and MISC categories count as
grammar flags. Spelling (unknown domain terms and acronyms), punctuation,
redundancy, style, typography and casing matches do not. Each row also says
whether the matched span lies inside the reviewed capability clause, which the
templates copy from the source requirement, or in the fixed template wording.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

try:
    import eval_utils as eu
except ModuleNotFoundError:  # pragma: no cover
    from scripts import eval_utils as eu

ITEM_FILES = {
    "nice/must": "benchmark_items.csv",
    "nice/shall": "benchmark_items_shall.csv",
    "mlm_tapt/must": "benchmark_items_mlm_tapt.csv",
    "mlm_tapt/shall": "benchmark_items_mlm_tapt_shall.csv",
    "pure": "benchmark_items_pure.csv",
}
PROBE_ITEM_FILE = "benchmark_items.csv"  # the phrasing probe uses the NICE capabilities
GRAMMAR_CATEGORIES = {"GRAMMAR", "MISC"}
FIELDNAMES = [
    "sentence",
    "sources",
    "capability_text",
    "rule_id",
    "category",
    "counts_as_grammar",
    "in_capability_clause",
    "matched_text",
    "message",
    "suggestions",
]


def collect_sentences(items_dir: Path) -> dict[str, dict[str, object]]:
    """Map each distinct sentence to its capability clause and where it occurs."""
    sentences: dict[str, dict[str, object]] = defaultdict(
        lambda: {"capability_text": "", "sources": set()}
    )
    for tag, filename in ITEM_FILES.items():
        for row in eu.read_csv_rows(items_dir / filename):
            for text, role in (
                (row["source_statement"], row["source_modality"]),
                (row["candidate_requirement"], "candidate"),
            ):
                sentences[text]["capability_text"] = row["capability_text"]
                sentences[text]["sources"].add(f"{tag}/{role}")
    for row in eu.read_csv_rows(items_dir / PROBE_ITEM_FILE):
        if row["source_modality"] != "nice_to_have":
            continue
        for template in eu.WEAK_MODALITY_PROBE_TEMPLATES:
            text = template["source_template"].format(capability=row["capability_text"])
            sentences[text]["capability_text"] = row["capability_text"]
            sentences[text]["sources"].add(f"probe/{template['template_id']}")
    return dict(sentences)


def in_clause(sentence: str, capability: str, offset: int, length: int) -> bool:
    start = sentence.find(capability)
    return start >= 0 and start <= offset and offset + length <= start + len(capability)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/benchmark_grammar_check.csv")
    )
    parser.add_argument("--language", default="en-US")
    args = parser.parse_args()

    import language_tool_python
    from language_tool_python.download_lt import LTP_DOWNLOAD_VERSION

    root = eu.project_root()
    items_dir = (
        args.items_dir if args.items_dir.is_absolute() else root / args.items_dir
    )
    output = args.output if args.output.is_absolute() else root / args.output

    sentences = collect_sentences(items_dir)
    rows = []
    with language_tool_python.LanguageTool(args.language) as tool:
        for text in sorted(sentences):
            info = sentences[text]
            for match in tool.check(text):
                rows.append(
                    {
                        "sentence": text,
                        "sources": ";".join(sorted(info["sources"])),
                        "capability_text": info["capability_text"],
                        "rule_id": match.rule_id,
                        "category": match.category,
                        "counts_as_grammar": match.category in GRAMMAR_CATEGORIES,
                        "in_capability_clause": in_clause(
                            text,
                            str(info["capability_text"]),
                            match.offset,
                            match.error_length,
                        ),
                        "matched_text": match.matched_text,
                        "message": match.message,
                        "suggestions": " | ".join(match.replacements[:3]),
                    }
                )
    eu.write_csv_rows(output, rows, fieldnames=FIELDNAMES)

    grammar = [row for row in rows if row["counts_as_grammar"]]
    flagged = {row["capability_text"] for row in grammar}
    in_template = [row for row in grammar if not row["in_capability_clause"]]
    by_category = Counter(row["category"] for row in rows)
    capabilities = {str(info["capability_text"]) for info in sentences.values()}
    categories = ", ".join(f"{k} {v}" for k, v in by_category.most_common())
    flagged_sentences = len({row["sentence"] for row in grammar})
    summary = [
        "# Benchmark grammar check",
        "",
        (
            f"LanguageTool {LTP_DOWNLOAD_VERSION} ({args.language}), default rule "
            "level, run locally via language-tool-python."
        ),
        (
            "Command: `uv run --with language-tool-python python "
            "scripts/check_benchmark_grammar.py`"
        ),
        "",
        f"- Sentences checked: {len(sentences)} ({len(capabilities)} capability clauses)",
        f"- Matches, all categories: {len(rows)} ({categories})",
        (
            f"- Grammar flags (GRAMMAR, MISC): {len(grammar)} in {flagged_sentences} "
            f"sentences, {len(flagged)} capability clauses"
        ),
        f"- Grammar flags in the fixed template wording: {len(in_template)}",
        "",
        f"Row-level matches, including the ones that do not count, are in `{output.name}`.",
        "",
    ]
    eu.atomic_write_text(output.with_suffix(".md"), "\n".join(summary))
    print("\n".join(summary))


if __name__ == "__main__":
    main()
