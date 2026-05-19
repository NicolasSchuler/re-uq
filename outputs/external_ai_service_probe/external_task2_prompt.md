# External High-Capacity Model Probe: Modality-Preserving Extraction

Please process the uploaded file `external_task2_inputs.csv`.

## Task

For each row, extract exactly one requirement from `source_statement`.
Preserve the modality of the source.

Use only one of these modality labels:

- `mandatory`
- `recommended`
- `optional`
- `nice_to_have`

## Per-Item Prompt

Apply this prompt independently to every row:

```text
Extract exactly one requirement from the source statement.
Preserve the modality of the source.

Source:
"{source_statement}"

Return JSON only:
{
  "requirement": "...",
  "modality": "mandatory" | "recommended" | "optional" | "nice_to_have",
  "confidence": 0-100
}
```

## Output Format

Return JSONL only, with one object per input row.
Do not include Markdown fences, comments, analysis, or extra prose.

Each JSONL object must have exactly these fields:

```json
{
  "external_item_id": "EXT0001",
  "requirement": "...",
  "modality": "mandatory",
  "confidence": 95
}
```

Requirements:

- Preserve every `external_item_id` exactly as given.
- Produce exactly one JSONL object for every row in the uploaded file.
- Use numeric confidence from `0` to `100`.
- Do not add any labels beyond the four allowed modality labels.
- Treat the external IDs as opaque identifiers; do not infer labels from them.
