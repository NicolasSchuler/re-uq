# External AI Service Probe

Purpose: test whether a larger web-based model preserves weak stakeholder
intent better than the local 9B pilot model, and whether it remains highly
confident when it upgrades weak modality.

Files:

- `external_task2_prompt.md`: copy/paste this into the web AI service.
- `external_task2_inputs.csv`: upload this file to the service.
- `external_task2_inputs.jsonl`: same inputs in JSONL form, useful for services that prefer JSONL.
- `external_task2_gold_key.csv`: keep this local; it contains the gold labels and should not be shown to the model.
- `*_evaluation.md`: curated evaluation reports only when they include provenance and pass the current confidence-scale contract.

Design:

- 20 reviewed pilot seeds.
- 4 main modality statements per seed: `MUST`, `SHOULD`, `MAY`, and `It would be useful if...`.
- 3 additional weak-modality statements per seed from the robustness probe.
- 140 total items.

Main comparison:

- Does the larger model preserve `nice_to_have` more often?
- When it fails, does it upgrade weak intent to `optional`, `recommended`, or `mandatory`?
- Are those upgrades high-confidence?

Save the returned JSONL from the service as something like
`external_model_outputs_<model-name>.jsonl` in this folder.

Current confidence contract:

- The prompt asks for numeric confidence from `0.0` to `1.0`.
- Percentages such as `95` and strings such as `"95%"` are invalid.
- Raw `*_outputs.jsonl` and row-level `*_scored_items.csv` files stay ignored locally by default.
- Curated reports require zero invalid confidence values plus prompt version, confidence scale, raw-output SHA-256, gold-key SHA-256, and prompt SHA-256.
- The comparison report only includes scored outputs that have a matching current evaluation report with paper-ready status metadata.

Legacy note:

Reports produced from old `0-100` confidence outputs are diagnostics only. Regenerate them with the current prompt before using them as paper-facing evidence.
