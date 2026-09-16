#!/usr/bin/env python3
"""Export frozen deterministic Task 2 wording transitions without semantic scoring."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

try:
    from scripts import eval_utils as eu, export_paper_tables as paper
except ImportError:
    import eval_utils as eu
    import export_paper_tables as paper

LEVELS = tuple(sorted(eu.ORDINAL_STRENGTH, key=eu.ORDINAL_STRENGTH.get))
OUTCOMES = (
    "strict_up",
    "broad_only_up",
    "equal",
    "down",
    "unknown",
    "negated",
    "invalid",
)


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        # Frozen snapshots contain original checkout paths; keep their artifact suffix.
        if "data" in path.parts:
            return root.joinpath(*path.parts[path.parts.index("data") :])
        return path
    return root / path


def load_observations(root: Path, provenance_path: Path) -> list[dict]:
    snapshot = json.loads(provenance_path.read_text())
    models = snapshot["models_cohort"]
    if not models or len(set(models)) != len(models) or "all" in models:
        raise ValueError("Frozen model cohort must be nonempty and unique")
    observations, seen, cells = [], set(), set()
    for cell in snapshot["cells"]:
        dataset, variant = cell["dataset"], cell["variant"]
        logging.getLogger(__name__).info(
            "Loading frozen Task 2 outputs: %s / %s", dataset, variant
        )
        if (dataset, variant) in cells:
            raise ValueError("Duplicate frozen cell")
        cells.add((dataset, variant))
        runs = cell["run_ids"]
        if set(runs) != set(models) or len(set(runs.values())) != len(runs):
            raise ValueError("Unexpected model/run selection in frozen cell")
        if cell["sampling_plan_source"] != "planned":
            raise ValueError("Frozen sampling plan must be planned")
        plan = eu.SamplingPlan(
            stochastic_samples=int(cell["expected_stochastic_samples"])
        )
        benchmark = eu.read_csv_rows(_path(root, cell["benchmark_path"]))
        items = {str(item["item_id"]): item for item in benchmark}
        if len(items) != len(benchmark) or len(items) != cell["n_benchmark_items"]:
            raise ValueError("Benchmark size or duplicate item mismatch")
        for item in benchmark:
            if (
                item["source_modality"] not in LEVELS
                or item["source_modality"] != item["task2_gold_modality"]
            ):
                raise ValueError("Benchmark source/gold modality mismatch")
        raw_rows = paper.task2_deterministic_raw_rows(
            [
                row
                for path in cell.get("raw_paths", [cell["raw_path"]])
                for row in paper.stream_raw_rows(_path(root, path), set(runs.values()))
            ]
        )
        for row in raw_rows:
            if row.get("model") not in runs or row.get("run_id") != runs[row["model"]]:
                raise ValueError("Unexpected model/run in frozen raw records")
            item = items.get(str(row.get("item_id")))
            if item is None or not eu.raw_record_matches_benchmark_item(row, item):
                raise ValueError("Raw/benchmark item mismatch")
            for field, expected in [
                ("dataset_id", dataset),
                ("benchmark_variant", variant),
                ("source_modality", item["source_modality"]),
            ]:
                if row.get(field) not in (None, "", expected):
                    raise ValueError(f"Raw/benchmark {field} mismatch")
        raw_rows = eu.dedupe_raw_rows(raw_rows)
        paper.stamp_cell_identity(raw_rows, dataset, variant)
        raw_by_key = {}
        for row in raw_rows:
            key = paper.paper_join_key(row)
            if key in raw_by_key or key in seen:
                raise ValueError(f"Duplicate paper key: {key}")
            raw_by_key[key] = row
        expected = {
            (
                model,
                eu.normalize_dataset_id(dataset),
                eu.normalize_benchmark_variant(variant),
                item_id,
            )
            for model in models
            for item_id in items
        }
        if set(raw_by_key) != expected:
            raise ValueError(
                f"Missing planned deterministic rows: {len(expected - set(raw_by_key))}"
            )
        scores = paper.task2_deterministic_rows(
            eu.build_uq_scores(benchmark, raw_rows, sampling_plan=plan)
        )
        paper.stamp_cell_identity(scores, dataset, variant)
        score_by_key = {paper.paper_join_key(row): row for row in scores}
        if len(score_by_key) != len(scores):
            raise ValueError("Duplicate deterministic score paper key")
        for key, raw in raw_by_key.items():
            item = items[key[-1]]
            score = score_by_key.get(key)
            observation = {
                field: raw.get(field, "")
                for field in (
                    "model",
                    "run_id",
                    "run_group_id",
                    "provider_id",
                    "profile_id",
                    "dataset_id",
                    "benchmark_variant",
                    "item_id",
                )
            }
            observation.update(
                source_modality=item["source_modality"],
                text_modality="",
                text_modality_basis="",
                strict_text_overcommit=False,
                strict_text_overcommit_kind="",
                outcome="invalid",
            )
            if score is not None:
                observation.update(
                    {
                        field: score[field]
                        for field in (
                            "text_modality",
                            "text_modality_basis",
                            "strict_text_overcommit",
                            "strict_text_overcommit_kind",
                        )
                    }
                )
                target = score["text_modality"]
                if target in ("unknown", "negated"):
                    outcome = target
                elif score["strict_text_overcommit"]:
                    outcome = "strict_up"
                else:
                    delta = (
                        eu.ORDINAL_STRENGTH[target]
                        - eu.ORDINAL_STRENGTH[item["source_modality"]]
                    )
                    outcome = (
                        "broad_only_up"
                        if delta > 0
                        else "down"
                        if delta < 0
                        else "equal"
                    )
                observation["outcome"] = outcome
            observations.append(observation)
        seen.update(raw_by_key)
        logging.getLogger(__name__).info(
            "Validated %d deterministic outputs: %s / %s",
            len(raw_by_key),
            dataset,
            variant,
        )
    if not cells:
        raise ValueError("Frozen snapshot has no cells")
    return observations


def aggregate_observations(observations: list[dict]) -> tuple[list[dict], list[dict]]:
    groups = defaultdict(list)
    for row in observations:
        model, dataset, variant = (
            row["model"],
            row["dataset_id"],
            row["benchmark_variant"],
        )
        for key in (
            (model, dataset, variant),
            (model, "all", "all"),
            ("all", dataset, variant),
            ("all", "all", "all"),
        ):
            groups[key].append(row)
    matrix, accounting = [], []
    for (model, dataset, variant), group in sorted(groups.items()):
        for source in LEVELS:
            rows = [row for row in group if row["source_modality"] == source]
            readable = [row for row in rows if row["outcome"] in OUTCOMES[:4]]
            base = {
                "model": model,
                "dataset": dataset,
                "variant": variant,
                "source_modality": source,
            }
            counts = {
                f"n_{outcome}": sum(row["outcome"] == outcome for row in rows)
                for outcome in OUTCOMES
            }
            accounting.append(
                dict(
                    base,
                    n_planned=len(rows),
                    n_readable=len(readable),
                    **counts,
                    n_heuristic=sum(
                        row["text_modality_basis"] == "heuristic_system_verb"
                        for row in rows
                    ),
                )
            )
            if sum(counts.values()) != len(rows):
                raise ValueError("Unrecognized outcome in observation accounting")
            for target in LEVELS:
                transitions = [
                    row for row in readable if row["text_modality"] == target
                ]
                strict = sum(row["outcome"] == "strict_up" for row in transitions)
                frame = sum(
                    row["strict_text_overcommit_kind"]
                    == eu.STRICT_OVERCOMMIT_FRAME_ONLY
                    for row in transitions
                )
                escalation = sum(
                    row["strict_text_overcommit_kind"]
                    == eu.STRICT_OVERCOMMIT_ESCALATION
                    for row in transitions
                )
                if strict != frame + escalation:
                    raise ValueError("Strict strengthening kind accounting mismatch")
                matrix.append(
                    dict(
                        base,
                        text_modality=target,
                        n=len(transitions),
                        n_strict=strict,
                        n_heuristic=sum(
                            row["text_modality_basis"] == "heuristic_system_verb"
                            for row in transitions
                        ),
                        n_frame_only=frame,
                        n_escalation=escalation,
                        source_readable_n=len(readable),
                        strict_percentage=100 * strict / len(readable)
                        if readable
                        else "",
                    )
                )
    return matrix, accounting


def reconcile(accounting: list[dict], pooled_path: Path) -> None:
    actual = {
        (row["model"], row["source_modality"]): row
        for row in accounting
        if row["model"] != "all" and row["dataset"] == row["variant"] == "all"
    }
    published = {
        (row["model"], row["source_modality"]): row
        for row in eu.read_csv_rows(pooled_path)
        if row["model"] != "all"
    }
    if actual.keys() != published.keys():
        raise ValueError("Published modality table cohort mismatch")
    for key, row in actual.items():
        expected = published[key]
        comparisons = {
            "n_items": row["n_planned"],
            "n_valid": row["n_planned"] - row["n_invalid"],
            "n_parse_failures": row["n_invalid"],
            "n_text_unclassified": row["n_unknown"] + row["n_negated"],
            "strict_strengthening_n": row["n_strict_up"],
            "strict_strengthening_denominator": row["n_readable"],
            "broad_strengthening_n": row["n_strict_up"] + row["n_broad_only_up"],
            "broad_strengthening_denominator": row["n_readable"],
        }
        for field, value in comparisons.items():
            if int(expected[field]) != value:
                raise ValueError(
                    f"Published reconciliation mismatch for {key}, {field}: {value} != {expected[field]}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("outputs/paper_snapshot_provenance.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--tikz-output",
        type=Path,
        default=Path("manuscript/figures/commitment_transitions.tex"),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    provenance = args.root / args.provenance
    observations = load_observations(args.root, provenance)
    matrix, accounting = aggregate_observations(observations)
    snapshot = json.loads(provenance.read_text())
    reconcile(accounting, args.root / snapshot["outputs"]["per_model_modality_pooled"])
    try:
        from scripts.commitment_transition_figure import render_tikz
    except ImportError:
        from commitment_transition_figure import render_tikz
    tikz = render_tikz(
        matrix,
        accounting,
        model_count=len(snapshot["models_cohort"]),
        cell_count=len(snapshot["cells"]),
    )
    output_dir = args.root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [
        ("commitment_transition_counts.csv", matrix),
        ("commitment_transition_accounting.csv", accounting),
    ]:
        with (output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    tikz_path = args.root / args.tikz_output
    tikz_path.parent.mkdir(parents=True, exist_ok=True)
    tikz_path.write_text(tikz)
    totals = [
        row
        for row in accounting
        if row["model"] == row["dataset"] == row["variant"] == "all"
    ]
    print(
        f"Exported {len(observations)} planned outputs, {sum(row['n_readable'] for row in totals)} readable, {sum(row['n_strict_up'] for row in totals)} strict strengthening; published table reconciliation passed."
    )


if __name__ == "__main__":
    main()
