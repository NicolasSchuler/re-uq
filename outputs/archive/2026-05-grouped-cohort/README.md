# Archived: May 2026 grouped-batching campaign

These summaries come from the original submission's campaign (run group
`provider-matrix-2026-05`, tag `v1.0.0`): six models, five GLM models on the
z.ai endpoint and `kit.gemma4-31b-it` on the KIT endpoint, queried with 16
benchmark items per request. The revised manuscript reports the September
2026 `manuscript-final` campaign instead (nine models, one item per request);
its tables live one level up under `outputs/`.

Kept for the record and not cited by the revised manuscript. The headline
values this campaign produced were strict text strengthening 8.6%, broad
13.8%, and weak-intent strict strengthening at confidence >= 0.90 of 29.8% in
the mlm_tapt/MUST cell.

| File | Content |
| --- | --- |
| `pilot_results_summary.md` | Pilot-run notes (2026-05-22) |
| `logprob_probe.json` | Provider log-probability capability probe |
| `prompt_sensitivity_summary.csv`, `task2_prompt_sensitivity_summary.csv` | Task 1 and Task 2 prompt-wording sensitivity on the pilot subset |
| `weak_modality_probe_summary.csv`, `.md` | Weak-intent phrasing probe of that campaign; the final campaign's per-run probes are under `../../weak_modality_probe/` |
| `external_ai_service_probe/` | Blind Task 2 probe of an external chat service: inputs, gold key, prompt, evaluation and comparison. Row-level outputs and scored items stay local. |
