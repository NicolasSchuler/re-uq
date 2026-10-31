"""Evaluate blind external web-model outputs for the Task 2 probe."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import eval_utils as eu
except (
    ModuleNotFoundError
):  # pragma: no cover - exercised when imported as package in tests
    from scripts import eval_utils as eu


ALLOWED_MODALITIES = set(eu.MODALITIES)
EXTERNAL_CONFIDENCE_SCALE = eu.CONFIDENCE_SCALE_0_1


def validation_blockers(validation: dict[str, Any]) -> list[str]:
    """List the reasons an external evaluation is not paper-ready.

    Args:
        validation: Validation summary from `evaluate_outputs`.

    Returns:
        Blocker names among `parse_errors`, `missing_ids`, `extra_ids`,
        `invalid_labels`, `invalid_confidence` and `row_count_mismatch`, empty
        when none applies.
    """
    blockers: list[str] = []
    if validation.get("parse_errors"):
        blockers.append("parse_errors")
    if validation.get("missing_ids"):
        blockers.append("missing_ids")
    if validation.get("extra_ids"):
        blockers.append("extra_ids")
    if validation.get("invalid_label_count"):
        blockers.append("invalid_labels")
    if validation.get("invalid_confidence_count"):
        blockers.append("invalid_confidence")
    if validation.get("output_rows") != validation.get("gold_rows"):
        blockers.append("row_count_mismatch")
    return blockers


def valid_probability_confidence(value: Any) -> bool:
    """Check that a confidence value is a number on the 0-1 probability scale.

    Args:
        value: Confidence as parsed from the output JSON. Booleans and strings
            are rejected even when they would convert to a number.

    Returns:
        True for a number between 0 and 1 inclusive.
    """
    if isinstance(value, (bool, str)):
        return False
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return False
    return 0.0 <= confidence <= 1.0


def slugify(value: str) -> str:
    """Turn a model name into the slug that prefixes its output files.

    Args:
        value: Model name from `--model-name`.

    Returns:
        The lower-case name with every run of other characters than letters
        and digits replaced by `_`, or `external_model` when nothing is left.
    """
    return eu.safe_identifier(value, fallback="external_model")


def read_jsonl_records(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read a JSONL file, collecting unparseable lines instead of raising.

    Args:
        path: JSONL file of external model outputs; blank lines are skipped.

    Returns:
        The parsed records, and one error record per unparseable line with its
        line number, the parser message and the line's first 160 characters.
    """
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                errors.append(
                    {
                        "line_number": line_number,
                        "error": str(exc),
                        "raw_prefix": stripped[:160],
                    }
                )
    return records, errors


def evaluate_outputs(
    output_path: Path, gold_key_path: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score external outputs against the blind gold key and validate the file.

    Each gold item is joined to its output by `external_item_id`, and both its
    label and its requirement text are scored for correctness and for over-
    and under-commitment, with high-confidence variants at 0.80 and 0.90.

    Args:
        output_path: JSONL file of external outputs with `external_item_id`,
            `requirement`, `modality` and `confidence`.
        gold_key_path: CSV gold key with `external_item_id` and
            `task2_gold_modality`.

    Returns:
        The scored items, one per gold row, and a validation summary with row
        and error counts, file hashes and the paper-ready verdict.

    Raises:
        ValueError: If a required output field is missing or an item id is
            scored more than once.
    """
    records, parse_errors = read_jsonl_records(output_path)
    outputs = pd.DataFrame.from_records(records)
    gold = eu.read_csv_frame(gold_key_path)

    required_fields = {"external_item_id", "requirement", "modality", "confidence"}
    missing_fields = sorted(required_fields - set(outputs.columns))
    if missing_fields:
        raise ValueError(f"Output file is missing required fields: {missing_fields}")

    outputs["valid_label"] = outputs["modality"].isin(ALLOWED_MODALITIES)
    outputs["valid_confidence"] = outputs["confidence"].map(
        valid_probability_confidence
    )

    duplicate_ids = int(outputs["external_item_id"].duplicated().sum())
    if duplicate_ids:
        duplicated_values = sorted(
            str(value)
            for value in outputs.loc[
                outputs["external_item_id"].duplicated(keep=False), "external_item_id"
            ].unique()
        )
        preview = ", ".join(duplicated_values[:10])
        if len(duplicated_values) > 10:
            preview += ", ..."
        raise ValueError(
            "Output file contains duplicate external_item_id values; "
            f"each item must be scored exactly once. Duplicates: {preview}"
        )
    missing_ids = sorted(
        set(gold["external_item_id"]) - set(outputs["external_item_id"])
    )
    extra_ids = sorted(set(outputs["external_item_id"]) - set(gold["external_item_id"]))

    merged = gold.merge(
        outputs, on="external_item_id", how="left", suffixes=("_gold", "_pred")
    )
    merged["confidence_num"] = pd.to_numeric(merged["confidence"], errors="coerce")
    merged["correct"] = merged["modality"].eq(merged["task2_gold_modality"])
    merged["gold_strength"] = merged["task2_gold_modality"].map(eu.ORDINAL_STRENGTH)
    merged["pred_strength"] = merged["modality"].map(eu.ORDINAL_STRENGTH)
    merged["overcommit"] = merged["pred_strength"] > merged["gold_strength"]
    merged["undercommit"] = merged["pred_strength"] < merged["gold_strength"]
    merged["high_conf_overcommit_80"] = merged["overcommit"] & (
        merged["confidence_num"] >= 0.80
    )
    merged["high_conf_overcommit_90"] = merged["overcommit"] & (
        merged["confidence_num"] >= 0.90
    )
    text_rows = [
        eu.text_modality_fields(
            row.get("requirement", ""),
            row.get("task2_gold_modality", ""),
            row.get("modality", ""),
            float(row.get("confidence_num") or 0.0),
        )
        for row in merged.to_dict(orient="records")
    ]
    text_frame = pd.DataFrame.from_records(text_rows)
    for column in text_frame.columns:
        merged[column] = text_frame[column]

    validation = {
        "output_rows": len(outputs),
        "gold_rows": len(gold),
        "parse_errors": len(parse_errors),
        "duplicate_ids": duplicate_ids,
        "missing_ids": missing_ids,
        "extra_ids": extra_ids,
        "invalid_label_count": int((~outputs["valid_label"]).sum()),
        "invalid_confidence_count": int((~outputs["valid_confidence"]).sum()),
        "confidence_scale": EXTERNAL_CONFIDENCE_SCALE,
        "raw_output_sha256": eu.sha256_file(output_path),
        "gold_key_sha256": eu.sha256_file(gold_key_path),
    }
    blockers = validation_blockers(validation)
    validation["paper_ready"] = not blockers
    validation["paper_ready_blockers"] = blockers
    return merged, validation


def source_condition_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """Summarize scored items per source condition.

    Args:
        scored: Scored items from `evaluate_outputs`.

    Returns:
        One row per `source_condition` with the item count, label and
        text-level accuracy, over- and under-commitment rates, high-confidence
        over-commitment rates and mean confidence.
    """
    return (
        scored.groupby("source_condition", dropna=False)
        .agg(
            n=("external_item_id", "size"),
            accuracy=("correct", "mean"),
            overcommit_rate=("overcommit", "mean"),
            undercommit_rate=("undercommit", "mean"),
            high_conf_overcommit_80=("high_conf_overcommit_80", "mean"),
            high_conf_overcommit_90=("high_conf_overcommit_90", "mean"),
            text_modality_accuracy=("text_modality_correct", "mean"),
            label_text_consistency=("label_text_consistent", "mean"),
            text_overcommit_rate=("text_overcommit", "mean"),
            text_undercommit_rate=("text_undercommit", "mean"),
            text_high_conf_overcommit_80=("text_high_conf_overcommit_80", "mean"),
            text_high_conf_overcommit_90=("text_high_conf_overcommit_90", "mean"),
            mean_confidence=("confidence_num", "mean"),
        )
        .reset_index()
    )


def overall_summary(scored: pd.DataFrame) -> dict[str, Any]:
    """Summarize scored items overall and for the weak `nice_to_have` gold items.

    Args:
        scored: Scored items from `evaluate_outputs`.

    Returns:
        Label and text-level accuracy, over- and under-commitment rates,
        high-confidence over-commitment rates and mean confidence, with the
        `weak_*` entries restricted to `nice_to_have` gold items.
    """
    weak = scored[scored["task2_gold_modality"].eq("nice_to_have")]
    return {
        "n": len(scored),
        "accuracy": float(scored["correct"].mean()),
        "overcommit_rate": float(scored["overcommit"].mean()),
        "undercommit_rate": float(scored["undercommit"].mean()),
        "high_conf_overcommit_80": float(scored["high_conf_overcommit_80"].mean()),
        "high_conf_overcommit_90": float(scored["high_conf_overcommit_90"].mean()),
        "mean_confidence": float(scored["confidence_num"].mean()),
        "weak_n": len(weak),
        "weak_accuracy": float(weak["correct"].mean()),
        "weak_overcommit_rate": float(weak["overcommit"].mean()),
        "weak_high_conf_overcommit_80": float(weak["high_conf_overcommit_80"].mean()),
        "weak_high_conf_overcommit_90": float(weak["high_conf_overcommit_90"].mean()),
        "text_modality_accuracy": float(scored["text_modality_correct"].mean()),
        "label_text_consistency": float(scored["label_text_consistent"].mean()),
        "text_overcommit_rate": float(scored["text_overcommit"].mean()),
        "text_undercommit_rate": float(scored["text_undercommit"].mean()),
        "text_high_conf_overcommit_80": float(
            scored["text_high_conf_overcommit_80"].mean()
        ),
        "text_high_conf_overcommit_90": float(
            scored["text_high_conf_overcommit_90"].mean()
        ),
        "weak_text_modality_accuracy": float(weak["text_modality_correct"].mean()),
        "weak_text_overcommit_rate": float(weak["text_overcommit"].mean()),
    }


def markdown_report(
    model_name: str,
    validation: dict[str, Any],
    overall: dict[str, Any],
    by_condition: pd.DataFrame,
    confusion: pd.DataFrame,
    text_confusion: pd.DataFrame,
) -> str:
    """Render the Markdown evaluation report of one external model.

    A report that is not paper-ready opens with a notice listing its blockers.

    Args:
        model_name: Model name for the title.
        validation: Validation summary, including the prompt version, prompt
            hash and evaluation time that `main` adds.
        overall: Overall summary from `overall_summary`.
        by_condition: Per-condition summary from `source_condition_summary`.
        confusion: Gold-by-predicted label confusion matrix.
        text_confusion: Gold-by-text-modality confusion matrix.

    Returns:
        The report text.
    """
    lines = [
        f"# External Probe Evaluation: {model_name}",
        "",
    ]
    if not validation.get("paper_ready"):
        blockers = ", ".join(validation.get("paper_ready_blockers", [])) or "unknown"
        lines.extend(
            [
                "Legacy/non-paper-ready status: this report does not satisfy the current external-probe contract.",
                f"Blockers: {blockers}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Validation",
            "",
            f"- Output rows: {validation['output_rows']}",
            f"- Gold rows: {validation['gold_rows']}",
            f"- Parse errors: {validation['parse_errors']}",
            f"- Duplicate IDs: {validation['duplicate_ids']}",
            f"- Missing IDs: {len(validation['missing_ids'])}",
            f"- Extra IDs: {len(validation['extra_ids'])}",
            f"- Invalid labels: {validation['invalid_label_count']}",
            f"- Invalid confidence values: {validation['invalid_confidence_count']}",
            f"- Confidence scale: {validation.get('confidence_scale', '')}",
            f"- Prompt version: {validation.get('prompt_version', '')}",
            f"- Evaluated at UTC: {validation.get('evaluated_at_utc', '')}",
            f"- Raw output SHA-256: {validation.get('raw_output_sha256', '')}",
            f"- Gold key SHA-256: {validation.get('gold_key_sha256', '')}",
            f"- Prompt SHA-256: {validation.get('prompt_sha256', '')}",
            f"- Paper-ready under current contract: {'yes' if validation.get('paper_ready') else 'no'}",
            f"- Paper-ready blockers: {', '.join(validation.get('paper_ready_blockers', [])) or 'none'}",
            "",
            "## Overall",
            "",
            f"- Accuracy: {overall['accuracy']:.3f}",
            f"- Over-commitment rate: {overall['overcommit_rate']:.3f}",
            f"- Under-commitment rate: {overall['undercommit_rate']:.3f}",
            f"- High-confidence over-commitment >= 0.80: {overall['high_conf_overcommit_80']:.3f}",
            f"- High-confidence over-commitment >= 0.90: {overall['high_conf_overcommit_90']:.3f}",
            f"- Mean confidence: {overall['mean_confidence']:.3f}",
            f"- Weak-modality accuracy: {overall['weak_accuracy']:.3f}",
            f"- Weak-modality over-commitment rate: {overall['weak_overcommit_rate']:.3f}",
            f"- Text-modality accuracy: {overall['text_modality_accuracy']:.3f}",
            f"- Label-text consistency: {overall['label_text_consistency']:.3f}",
            f"- Text-level over-commitment rate: {overall['text_overcommit_rate']:.3f}",
            f"- Text-level high-confidence over-commitment >= 0.80: {overall['text_high_conf_overcommit_80']:.3f}",
            f"- Text-level high-confidence over-commitment >= 0.90: {overall['text_high_conf_overcommit_90']:.3f}",
            f"- Weak text-modality accuracy: {overall['weak_text_modality_accuracy']:.3f}",
            "",
            "## By Source Condition",
            "",
            by_condition.to_markdown(index=False, floatfmt=".3f"),
            "",
            "## Confusion Matrix",
            "",
            confusion.to_markdown(),
            "",
            "## Text-Modality Confusion Matrix",
            "",
            text_confusion.to_markdown(),
            "",
        ]
    )
    return "\n".join(lines)


def write_external_comparison_report(output_dir: Path) -> Path | None:
    """Write a comparison table of every evaluated external model in a directory.

    A model is included only when its evaluation report records a paper-ready
    verdict, which becomes a column, and its scored items are not empty.

    Args:
        output_dir: Directory with `*_scored_items.csv` and `*_evaluation.md`
            files.

    Returns:
        The path of `external_probe_comparison.md`, or `None` when no model
        qualifies and nothing is written.
    """
    rows: list[dict[str, Any]] = []
    for scored_path in sorted(output_dir.glob("*_scored_items.csv")):
        model_slug = scored_path.name.removesuffix("_scored_items.csv")
        paper_ready = evaluation_report_paper_ready(
            output_dir / f"{model_slug}_evaluation.md"
        )
        if paper_ready is None:
            continue
        scored = eu.read_csv_frame(scored_path)
        if scored.empty:
            continue
        for column in [
            "correct",
            "overcommit",
            "high_conf_overcommit_90",
            "text_modality_correct",
            "label_text_consistent",
            "text_overcommit",
            "text_high_conf_overcommit_90",
        ]:
            if column not in scored.columns:
                scored[column] = False
            scored[column] = (
                scored[column].astype(str).str.lower().isin({"true", "1", "yes"})
            )
        rows.append(
            {
                "model_slug": model_slug,
                "paper_ready": "yes" if paper_ready else "no",
                "n": len(scored),
                "label_accuracy": scored["correct"].mean(),
                "label_overcommit": scored["overcommit"].mean(),
                "label_high_conf_overcommit_90": scored[
                    "high_conf_overcommit_90"
                ].mean(),
                "text_accuracy": scored["text_modality_correct"].mean(),
                "label_text_consistency": scored["label_text_consistent"].mean(),
                "text_overcommit": scored["text_overcommit"].mean(),
                "text_high_conf_overcommit_90": scored[
                    "text_high_conf_overcommit_90"
                ].mean(),
            }
        )
    if not rows:
        return None
    frame = pd.DataFrame.from_records(rows)
    report = "\n".join(
        [
            "# External Probe Comparison",
            "",
            "Use this table only as an orientation aid; check each individual evaluation report for paper-ready provenance and confidence-scale validity.",
            "",
            frame.to_markdown(index=False, floatfmt=".3f"),
            "",
        ]
    )
    path = output_dir / "external_probe_comparison.md"
    path.write_text(report, encoding="utf-8")
    return path


def evaluation_report_paper_ready(path: Path) -> bool | None:
    """Read the paper-ready verdict from an evaluation report.

    Args:
        path: An `*_evaluation.md` report.

    Returns:
        The verdict of its "Paper-ready under current contract" line, or `None`
        when the report or the line is missing.
    """
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"Paper-ready under current contract:\s*(yes|no)", text, flags=re.IGNORECASE
    )
    if match is None:
        return None
    return match.group(1).lower() == "yes"


def main() -> None:
    """Evaluate one external model's outputs and refresh the comparison report.

    Writes the scored items, the per-condition summary, the confusion matrix
    and the Markdown report under `--output-dir`, named after the model slug.
    A missing prompt file adds the `missing_prompt_hash` blocker.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate external web-model outputs for the Task 2 probe."
    )
    parser.add_argument("outputs_jsonl", type=Path)
    parser.add_argument("--model-name", default="external_model")
    parser.add_argument(
        "--gold-key",
        type=Path,
        default=eu.project_root()
        / "outputs/archive/2026-05-grouped-cohort/external_ai_service_probe/external_task2_gold_key.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=eu.project_root()
        / "outputs/archive/2026-05-grouped-cohort/external_ai_service_probe",
    )
    parser.add_argument("--prompt-version", default="external-task2-v2-conf01")
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=eu.project_root()
        / "outputs/archive/2026-05-grouped-cohort/external_ai_service_probe/external_task2_prompt.md",
    )
    args = parser.parse_args()

    scored, validation = evaluate_outputs(args.outputs_jsonl, args.gold_key)
    validation["prompt_version"] = args.prompt_version
    validation["prompt_sha256"] = (
        eu.sha256_file(args.prompt_path) if args.prompt_path.exists() else ""
    )
    validation["evaluated_at_utc"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not validation["prompt_sha256"]:
        validation["paper_ready_blockers"] = [
            *validation.get("paper_ready_blockers", []),
            "missing_prompt_hash",
        ]
        validation["paper_ready"] = False
    by_condition = source_condition_summary(scored)
    confusion = pd.crosstab(
        scored["task2_gold_modality"], scored["modality"], dropna=False
    )
    text_confusion = pd.crosstab(
        scored["task2_gold_modality"], scored["text_modality"], dropna=False
    )
    overall = overall_summary(scored)

    slug = slugify(args.model_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = args.output_dir / f"{slug}_scored_items.csv"
    condition_path = args.output_dir / f"{slug}_source_condition_summary.csv"
    confusion_path = args.output_dir / f"{slug}_confusion_matrix.csv"
    report_path = args.output_dir / f"{slug}_evaluation.md"
    comparison_path = args.output_dir / "external_probe_comparison.md"

    scored.to_csv(scored_path, index=False)
    by_condition.to_csv(condition_path, index=False)
    confusion.to_csv(confusion_path)
    report_path.write_text(
        markdown_report(
            args.model_name,
            validation,
            overall,
            by_condition,
            confusion,
            text_confusion,
        ),
        encoding="utf-8",
    )
    written_comparison_path = write_external_comparison_report(args.output_dir)

    print(f"Wrote scored items: {scored_path}")
    print(f"Wrote condition summary: {condition_path}")
    print(f"Wrote confusion matrix: {confusion_path}")
    print(f"Wrote report: {report_path}")
    print(f"Wrote comparison report: {written_comparison_path or comparison_path}")


if __name__ == "__main__":
    main()
