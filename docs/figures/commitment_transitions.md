# Commitment transitions

Regenerate the RQ1 figure and its count tables from cached main-study outputs:

```bash
.venv/bin/python scripts/export_commitment_transitions.py
```

The exporter reads `outputs/paper_snapshot_provenance.json` and pins its exact
model/run pairs, benchmark cells, and sampling plan. It scores only deterministic
Task 2 responses, with existing retry selection and wording rules. It performs no
model calls or embedding calculations and does not regenerate headline metrics.

Outputs:

- `outputs/commitment_transition_counts.csv`: the full source-by-generated-level
  matrix, including strict and heuristic counts, and source-conditional strict
  percentages.
- `outputs/commitment_transition_accounting.csv`: source-level denominators and
  mutually exclusive outcomes, including invalid, unknown, and negated outputs.
- `manuscript/figures/commitment_transitions.tex`: the generated TikZ figure.

Both CSVs include model/cell detail and pooled rows. `all` in model, dataset, or
variant identifies an aggregate. Do not sum aggregate rows with their components.

## Reading the figure

An arrow goes from the source commitment to the commitment expressed in the
generated requirement text, independently of the model's declared label. Its
percentage divides strict transitions by all classifiable outputs from that
source level. This denominator includes heuristic-only classifications, matching
the paper's strict strengthening rate. The arrows themselves require strict
evidence. Their widths are uniform and do not encode frequency.

The dashed weak-intent-to-optional arrow denotes removal of wish framing while
optional wording remains. Solid arrows denote explicit modal escalation.
Node order is ordinal, not an equal-distance semantic scale. Equal-level wording
does not establish preservation of functional content.

The accounting CSV retains the outcomes omitted from the diagram: equal levels,
weakening, broad-only strengthening inferred from wording without a modal word,
invalid responses, and unclassified wording. Unknown and negated outputs are
counted separately. Heuristic-only classification
is also recorded as an overlapping diagnostic count: a heuristic classification
can be equal-level rather than upward.

The pooled destinations are not typical of every model. Inspect model-specific
matrix rows before interpreting the pooled pattern. Detailed counts stay in the
replication outputs so the figure and caption remain compact.

## Verification

```bash
.venv/bin/python -m unittest tests.test_commitment_transitions tests.test_commitment_transition_figure tests.test_paper_exports -v
```

Compile the manuscript with an isolated temporary output directory, then inspect
the figure and adjacent pages at actual reading size. Keep auxiliary files out
of the shared manuscript directory to avoid interference with background builds.
