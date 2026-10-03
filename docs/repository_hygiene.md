# Artifact distribution

The public repository contains the material needed to understand the study,
inspect its evidence and reproduce its analyses. The artifact is divided
between Git and the separately archived raw outputs.

| Location | Contents |
| --- | --- |
| Git | Analysis and preparation scripts, tests, frozen prompts, reviewed seeds, benchmark CSVs, review decisions, configuration examples, manifests, paper tables and figures, and methods and reproduction documentation |
| Zenodo dataset record | Raw model outputs, per-request transcripts, run registries, per-item scores and embedding caches; see [reproduction](reproduction.md) |
| Local development storage | Planning notes, implementation reviews, launch logs, exploratory notebooks, manuscript drafts, credentials and machine-specific settings |

The explicit allow-list in `.gitignore` identifies the compact outputs that
belong to the public artifact. Raw run outputs and caches are excluded from
Git and distributed through the dataset archive when needed for reproduction.
The historical scientific snapshots under `outputs/archive/` and
`data/processed/archive/` remain public and are labelled by campaign.

The two review CSVs in `docs/` are scientific inputs:
`weak_modality_construct_review.csv` records the weak-template validation used
by the analysis gate; `pure_capability_revisions.csv` records the accepted
capability revisions used to build the document-context benchmark. They must
remain available with the code that consumes them. Human validation scope
and limitations are described in [validation_review.md](validation_review.md).

Preparation and analysis use the scripts documented in
[reproduction.md](reproduction.md). Notebooks are not required. The test suite
checks benchmark reconstruction, result-table consistency and documentation
links. Use [results_mapping.md](results_mapping.md) to trace a reported result
to its evidence.
