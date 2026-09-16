# Prompts

Frozen task prompt contracts used by the CLI runner and recorded in benchmark manifests for provenance.

| File | Task | Notes |
| --- | --- | --- |
| `mandatory_entailment.txt` | Task 1 | Capability/control: does the source statement entail a mandatory requirement? |
| `mandatory_entailment_strict.txt` | Task 1 | Prompt-sensitivity variant of Task 1 used on the pilot subset. |
| `modality_extraction.txt` | Task 2 | Main extraction task: preserve source modality (`mandatory` / `recommended` / `optional` / `nice_to_have`). |
| `modality_extraction_labels_only.txt` | Task 2 | Prompt-validity variant: states allowed labels without deterministic mapping rules or examples. |
| `modality_extraction_context.txt` | Task 2 ablation | Same contract as `modality_extraction.txt` plus a `Document context:` block (document, section, author marker, neighbouring requirements). Sent only when a run sets `item_context: document` (`docs/context_ablation.md`). |
| `modality_verification.txt` | Task 3 | Official blind text audit over deterministic Task 2 outputs; does not reveal the declared Task 2 modality. |
| `modality_verification_declared.txt` | Task 3 ablation | Declared-modality anchoring prompt for Task 3 ablations only. |

Prompts are content-addressed by SHA-256 in `outputs/benchmark_manifest*.json`. Changing a prompt without updating the manifest will be caught by the analysis gate.

## Delivery: one item per request

The reported campaign sent each benchmark item on its own, rendered from these
files by `prompt_for_benchmark_task`; the prompt is the entire user message and
no system message is sent. The files are therefore both the task contract and
the request body.

`batch_prompt_for_completion_jobs` in `scripts/eval_utils.py` builds a request
that carries several items and asks for an array of results keyed by
`request_index`. That wrapper was the delivery mode of the archived May 2026
campaign and is now the request-composition ablation (4 and 16 items per
request, grouped or with the four conditions of a capability spread across
requests). Its prompt bodies are reproduced in
[`docs/experimental_setup.md`](../docs/experimental_setup.md).
