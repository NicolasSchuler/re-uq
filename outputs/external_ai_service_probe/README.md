# External AI Service Probe

Purpose: test whether a larger web-based model preserves weak stakeholder
intent better than the local 9B pilot model, and whether it remains highly
confident when it upgrades weak modality.

Files:

- `external_task2_prompt.md`: copy/paste this into the web AI service.
- `external_task2_inputs.csv`: upload this file to the service.
- `external_task2_inputs.jsonl`: same inputs in JSONL form, useful for services that prefer JSONL.
- `external_task2_gold_key.csv`: keep this local; it contains the gold labels and should not be shown to the model.

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
