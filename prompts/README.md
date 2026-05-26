# Prompts

Frozen task prompt contracts used by the CLI runner and recorded in benchmark manifests for provenance.

| File | Task | Notes |
| --- | --- | --- |
| `mandatory_entailment.txt` | Task 1 | Capability/control: does the source statement entail a mandatory requirement? |
| `mandatory_entailment_strict.txt` | Task 1 | Prompt-sensitivity variant of Task 1 used on the pilot subset. |
| `modality_extraction.txt` | Task 2 | Main extraction task: preserve source modality (`mandatory` / `recommended` / `optional` / `nice_to_have`). |
| `modality_extraction_labels_only.txt` | Task 2 | Prompt-validity variant: states allowed labels without deterministic mapping rules or examples. |
| `modality_verification.txt` | Task 3 | Official blind text audit over deterministic Task 2 outputs; does not reveal the declared Task 2 modality. |
| `modality_verification_declared.txt` | Task 3 ablation | Declared-modality anchoring prompt for Task 3 ablations only. |

Prompts are content-addressed by SHA-256 in `outputs/benchmark_manifest*.json`. Changing a prompt without updating the manifest will be caught by the analysis gate.
