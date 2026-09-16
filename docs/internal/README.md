# Internal engineering records

Working notes kept for traceability: how the code was reviewed, and how the
September 2026 campaign was qualified before launch. None of them is required
to reproduce the paper; the reader-facing pages are indexed in
[`../repository_layout.md`](../repository_layout.md). Each file is a record of
its date and is not updated afterwards.

| File | What it records | Date |
| --- | --- | --- |
| `abstraction_boundary_review.md` | Architecture review of the pipeline's boundaries: observation identity, resume, Task 2 to Task 3 handoff, export joins | 2026-09-04 |
| `python_simplification_review.md` | Simplification review of the Python modules and their tests | 2026-09-04 |
| `review_implementation_tasks.md` | Task list derived from the two reviews above | 2026-09-04 |
| `ablation_proposals.md` | The ablation studies agreed for the resubmission and how each is reported | 2026-09-11 |
| `final_run_readiness.md` | Pre-launch gates for the final campaign and their status on 2026-09-11 | 2026-09-11 |
| `live_smoke_2026-09-11.md` | Live smoke of the local llama.cpp path across the seven local models | 2026-09-11 |
| `generation_qualification_2026-09-11.md` | Operational qualification of the schema-constrained local profile | 2026-09-11 |
| `pure_capability_ai_review.md` | AI-assisted source review of the 180 PURE capabilities (129 accepted, 51 revised); the decisions are `../pure_capability_revisions.csv` | 2026-09-11 |
| `external_validity_datasets.md` | Survey of public datasets with natural stakeholder language, for future work | 2026-09-04 |
