# Benchmark grammar check

LanguageTool 6.8 (en-US), default rule level, run locally via language-tool-python.
Command: `uv run --with language-tool-python python scripts/check_benchmark_grammar.py`

- Sentences checked: 3060 (540 capability clauses)
- Matches, all categories: 609 (TYPOS 542, MISC 34, REDUNDANCY 13, PUNCTUATION 10, GRAMMAR 9, STYLE 1)
- Grammar flags (GRAMMAR, MISC): 43 in 43 sentences, 7 capability clauses
- Grammar flags in the fixed template wording: 0

Row-level matches, including the ones that do not count, are in `benchmark_grammar_check.csv`.
