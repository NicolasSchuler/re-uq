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

## These files are the contract, not the request body

All reported runs sent **batched** prompts, not these single-item files. `batch_prompt_for_completion_jobs` in `scripts/eval_utils.py` builds one request carrying 16 benchmark items and asks for an array of results keyed by `request_index`. The batched wrapper restates the same task, the same label set, and the same `0.0-1.0` confidence contract in a different surface form; it does not read these `.txt` files.

The files here remain authoritative for the task definition, they are what the manifest hashes, and they are what a `batch_size=1` run would send. The batched prompt bodies for Task 1, Task 2, and Task 3 are reproduced verbatim in [`docs/experimental_setup.md`](../docs/experimental_setup.md), together with the batching policy and its known confound.

No system message is sent with any prompt. The prompt is the entire user message.
