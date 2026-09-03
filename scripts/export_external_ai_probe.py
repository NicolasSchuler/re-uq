"""Export the blind Task 2 input bundle for external AI-service probes."""

from __future__ import annotations

import argparse
import csv
import json
import random
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:
    import eval_utils as eu
except (
    ModuleNotFoundError
):  # pragma: no cover - exercised when imported as package in tests
    from scripts import eval_utils as eu


OUTPUT_DIR_NAME = "external_ai_service_probe"
RANDOM_SEED = 20260519
EXCLUDED_DUPLICATE_WEAK_TEMPLATE = "useful_if"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_external_probe_rows(
    root: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Return blinded external-probe inputs plus the local gold key."""
    config = eu.load_config(root / "config.example.json")
    pilot_seed_count = int(config["project"]["pilot_seed_count"])

    benchmark_rows = eu.read_csv_rows(root / "data/processed/benchmark_items.csv")
    weak_probe_rows = eu.read_csv_rows(
        root / "data/processed/weak_modality_probe_items.csv"
    )

    pilot_seed_ids = sorted({row["seed_id"] for row in benchmark_rows})[
        :pilot_seed_count
    ]
    selected: list[dict[str, str]] = []

    for row in benchmark_rows:
        if row["seed_id"] not in pilot_seed_ids:
            continue
        selected.append(
            {
                "source_kind": "main_benchmark",
                "original_item_id": row["item_id"],
                "seed_id": row["seed_id"],
                "source_condition": row["source_modality"],
                "source_modality": row["source_modality"],
                "task2_gold_modality": row["task2_gold_modality"],
                "capability_text": row["capability_text"],
                "source_statement": row["source_statement"],
            }
        )

    for row in weak_probe_rows:
        if row["seed_id"] not in pilot_seed_ids:
            continue
        if row["template_id"] == EXCLUDED_DUPLICATE_WEAK_TEMPLATE:
            continue
        selected.append(
            {
                "source_kind": "weak_modality_probe",
                "original_item_id": row["item_id"],
                "seed_id": row["seed_id"],
                "source_condition": f"weak_{row['template_id']}",
                "source_modality": row["source_modality"],
                "task2_gold_modality": row["task2_gold_modality"],
                "capability_text": row["capability_text"],
                "source_statement": row["source_statement"],
            }
        )

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(selected)

    inputs: list[dict[str, str]] = []
    key: list[dict[str, str]] = []
    for index, row in enumerate(selected, start=1):
        external_id = f"EXT{index:04d}"
        inputs.append(
            {
                "external_item_id": external_id,
                "source_statement": row["source_statement"],
            }
        )
        key.append({"external_item_id": external_id, **row})

    return inputs, key


def prompt_markdown(input_filename: str) -> str:
    return textwrap.dedent(
        f"""\
        # External High-Capacity Model Probe: Modality-Preserving Extraction

        Prompt version: `external-task2-v2-conf01`

        Please process the uploaded file `{input_filename}`.

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
        "{{source_statement}}"

        Return JSON only:
        {{
          "requirement": "...",
          "modality": "mandatory" | "recommended" | "optional" | "nice_to_have",
          "confidence": 0.0-1.0
        }}
        ```

        ## Output Format

        Return JSONL only, with one object per input row.
        Do not include Markdown fences, comments, analysis, or extra prose.

        Each JSONL object must have exactly these fields:

        ```json
        {{
          "external_item_id": "EXT0001",
          "requirement": "...",
          "modality": "mandatory",
          "confidence": 0.95
        }}
        ```

        Requirements:

        - Preserve every `external_item_id` exactly as given.
        - Produce exactly one JSONL object for every row in the uploaded file.
        - Use numeric confidence from `0.0` to `1.0`.
        - Confidence is for the selected modality label; do not return percentages such as `95` or strings such as `"95%"`.
        - Do not add any labels beyond the four allowed modality labels.
        - Treat the external IDs as opaque identifiers; do not infer labels from them.
        """
    )


def readme_markdown() -> str:
    return textwrap.dedent(
        """\
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

        Legacy note:

        Reports produced from old `0-100` confidence outputs are diagnostics only. Regenerate them with the current prompt before using them as paper-facing evidence.
        """
    )


def export_external_probe_bundle(
    root: Path, output_dir: Path, *, dry_run: bool = False
) -> list[Path]:
    """Export the blind input bundle, or report target paths without writing."""
    inputs, key = build_external_probe_rows(root)

    input_csv = output_dir / "external_task2_inputs.csv"
    input_jsonl = output_dir / "external_task2_inputs.jsonl"
    key_csv = output_dir / "external_task2_gold_key.csv"
    prompt_path = output_dir / "external_task2_prompt.md"
    readme_path = output_dir / "README.md"
    paths = [input_csv, input_jsonl, key_csv, prompt_path, readme_path]

    if dry_run:
        print(f"Would write {len(inputs)} blind input rows")
        for path in paths:
            print(f"Would write: {path}")
        return paths

    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(input_csv, inputs, ["external_item_id", "source_statement"])
    write_jsonl(input_jsonl, inputs)
    write_csv(
        key_csv,
        key,
        [
            "external_item_id",
            "source_kind",
            "original_item_id",
            "seed_id",
            "source_condition",
            "source_modality",
            "task2_gold_modality",
            "capability_text",
            "source_statement",
        ],
    )
    prompt_path.write_text(prompt_markdown(input_csv.name), encoding="utf-8")
    readme_path.write_text(readme_markdown(), encoding="utf-8")

    print(f"Wrote {len(inputs)} blind input rows")
    print(f"Prompt: {prompt_path}")
    print(f"Inputs CSV: {input_csv}")
    print(f"Inputs JSONL: {input_jsonl}")
    print(f"Gold key: {key_csv}")
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for exporting or dry-running the external probe bundle."""
    parser = argparse.ArgumentParser(
        description="Export the blind Task 2 input bundle for external AI-service probes."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=eu.project_root(),
        help="Repository root. Defaults to this checkout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=f"Output directory. Defaults to outputs/{OUTPUT_DIR_NAME} under --root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and show target files without writing.",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    output_dir = args.output_dir or root / "outputs" / OUTPUT_DIR_NAME
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    export_external_probe_bundle(root, output_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
