from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip() + "\n")


COMMON_SETUP = r"""
from pathlib import Path
import os
import sys

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
while not (PROJECT_ROOT / "AGENTS.md").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import eval_utils as eu

CONFIG_PATH = PROJECT_ROOT / "config.json"
if not CONFIG_PATH.exists():
    CONFIG_PATH = PROJECT_ROOT / "config.example.json"
CONFIG = eu.load_config(CONFIG_PATH)
eu.ensure_project_dirs(PROJECT_ROOT)

PROJECT_ROOT, CONFIG_PATH
"""


def write_notebook(name: str, cells: list[nbf.NotebookNode]) -> None:
    path = NOTEBOOK_DIR / name
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {
        "name": "python",
        "pygments_lexer": "ipython3",
    }
    nbf.write(notebook, path)


def notebook_00() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # 00 Prepare Data

            Objective: load the NICE/PROMISE-derived requirements dataset, create a transparent seed review table, and keep exactly 120 accepted seed capabilities for the main experiment.

            The automatic filter is intentionally conservative. Manual review remains part of the protocol: inspect `data/processed/seeds_review.csv`, edit `include`, `exclusion_reason`, and `capability_text_final` if needed, then rerun the validation cells.
            """
        ),
        code(COMMON_SETUP),
        md("## Load or Download NICE"),
        code(
            r"""
            nice_path = PROJECT_ROOT / CONFIG["datasets"]["nice_local_path"]
            nice_url = CONFIG["datasets"]["nice_url"]

            if not nice_path.exists():
                print(f"NICE CSV not found at {nice_path}. Downloading from Zenodo...")
                eu.download_file(nice_url, nice_path, timeout_s=CONFIG["llm"]["timeout_s"])
            else:
                print(f"Using existing NICE CSV: {nice_path}")

            rows = eu.read_csv_rows(nice_path)
            text_column = eu.find_requirement_text_column(rows)
            print(f"Loaded {len(rows)} rows. Requirement text column: {text_column}")
            print(rows[0][text_column][:240])
            """
        ),
        md("## Build Review Table"),
        code(
            r"""
            target_count = int(CONFIG["project"]["target_seed_count"])
            candidates = eu.make_seed_candidates(rows, target_count=target_count)

            review_fields = [
                "seed_id",
                "source_dataset",
                "original_requirement",
                "capability_text_auto",
                "auto_include",
                "auto_exclusion_reason",
                "include",
                "exclusion_reason",
                "capability_text_final",
            ]
            review_path = PROJECT_ROOT / "data/processed/seeds_review.csv"
            eu.write_csv_rows(review_path, candidates, fieldnames=review_fields)

            auto_ok = sum(1 for row in candidates if row["auto_include"] == "yes")
            selected = eu.load_reviewed_seeds(review_path, target_count=target_count, strict=False)
            print(f"Wrote review table: {review_path}")
            print(f"Automatic candidates passing filters: {auto_ok}")
            print(f"Currently included seeds: {len(selected)} / {target_count}")
            """
        ),
        md("## Validate Reviewed Seeds"),
        code(
            r"""
            selected = eu.load_reviewed_seeds(review_path, target_count=target_count, strict=False)
            selected_path = PROJECT_ROOT / "data/processed/seeds_selected.csv"

            if len(selected) == target_count:
                eu.write_csv_rows(selected_path, selected)
                print(f"OK: exactly {target_count} included seeds.")
                print(f"Wrote selected seeds: {selected_path}")
            else:
                print(f"Review needed: found {len(selected)} included seeds, expected {target_count}.")
                print("Edit data/processed/seeds_review.csv, then rerun this cell.")

            selected[:3]
            """
        ),
    ]


def notebook_01() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # 01 Build Modality Benchmark

            Objective: convert the reviewed seed capabilities into controlled modality minimal pairs and gold labels for the two tasks.
            """
        ),
        code(COMMON_SETUP),
        md("## Load Reviewed Seeds"),
        code(
            r"""
            target_count = int(CONFIG["project"]["target_seed_count"])
            review_path = PROJECT_ROOT / "data/processed/seeds_review.csv"
            seeds = eu.load_reviewed_seeds(review_path, target_count=target_count, strict=True)
            print(f"Loaded {len(seeds)} reviewed seeds.")
            seeds[:2]
            """
        ),
        md("## Generate Four Modality Variants Per Seed"),
        code(
            r"""
            benchmark = eu.build_benchmark_items(seeds)
            benchmark_path = PROJECT_ROOT / "data/processed/benchmark_items.csv"
            eu.write_csv_rows(benchmark_path, benchmark)
            print(f"Wrote benchmark: {benchmark_path}")
            print(f"Items: {len(benchmark)}")
            benchmark[:4]
            """
        ),
        md("## Label and Shape Checks"),
        code(
            r"""
            expected_items = target_count * len(eu.MODALITIES)
            assert len(benchmark) == expected_items, (len(benchmark), expected_items)
            assert len({row["item_id"] for row in benchmark}) == expected_items
            assert all(row["task1_gold_decision"] == ("yes" if row["source_modality"] == "mandatory" else "no") for row in benchmark)
            assert all(row["task2_gold_modality"] == row["source_modality"] for row in benchmark)
            assert eu.ORDINAL_STRENGTH["mandatory"] > eu.ORDINAL_STRENGTH["recommended"] > eu.ORDINAL_STRENGTH["optional"] > eu.ORDINAL_STRENGTH["nice_to_have"]
            print("OK: benchmark shape, labels, and modality ordering are valid.")
            """
        ),
    ]


PROMPT_RUNNER = r"""
task1_template = eu.load_prompt(PROJECT_ROOT / "prompts/mandatory_entailment.txt")
task2_template = eu.load_prompt(PROJECT_ROOT / "prompts/modality_extraction.txt")

def prompt_for(task, item):
    if task == "task1":
        return eu.render_prompt(
            task1_template,
            source_statement=item["source_statement"],
            candidate_requirement=item["candidate_requirement"],
        )
    if task == "task2":
        return eu.render_prompt(task2_template, source_statement=item["source_statement"])
    raise ValueError(task)

def run_one(item, task, model, sample_kind, sample_index, temperature, top_p, output_path, run_id):
    prompt = prompt_for(task, item)
    completion = eu.chat_completion(
        host=HOST,
        model=model,
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=int(CONFIG["llm"]["max_tokens"]),
        timeout_s=int(CONFIG["llm"]["timeout_s"]),
        api_key_env=CONFIG["llm"]["api_key_env"],
    )
    record = eu.build_raw_record(
        run_id=run_id,
        model=model,
        host=HOST,
        task=task,
        item=item,
        sample_index=sample_index,
        sample_kind=sample_kind,
        temperature=temperature,
        top_p=top_p,
        prompt_version=CONFIG["project"]["prompt_version"],
        prompt=prompt,
        completion=completion,
    )
    eu.append_jsonl(output_path, record)
    return record
"""


def notebook_02() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # 02 Pilot Local LLMs

            Objective: run a bounded pilot against one local OpenAI-compatible model and check JSON parse rate before the full experiment.

            By default this notebook does not call the model. Set `RUN_PILOT = True` after the local endpoint is running.
            """
        ),
        code(COMMON_SETUP),
        md("## Configure Local Endpoint"),
        code(
            r"""
            HOST = os.getenv("HOST", CONFIG["llm"]["host"])
            configured_models = [m.strip() for m in os.getenv("MODELS", ",".join(CONFIG["llm"]["models"])).split(",") if m.strip()]
            MODEL = os.getenv("MODEL", configured_models[0])
            RUN_PILOT = os.getenv("RUN_PILOT", "false").lower() in {"1", "true", "yes"}

            deterministic = CONFIG["llm"]["deterministic"]
            stochastic = CONFIG["llm"]["stochastic"]
            print({"HOST": HOST, "MODEL": MODEL, "RUN_PILOT": RUN_PILOT})
            """
        ),
        md("## Select Pilot Items"),
        code(
            r"""
            benchmark = eu.read_csv_rows(PROJECT_ROOT / "data/processed/benchmark_items.csv")
            pilot_seed_count = int(CONFIG["project"]["pilot_seed_count"])
            pilot_seed_ids = sorted({row["seed_id"] for row in benchmark})[:pilot_seed_count]
            pilot_items = [row for row in benchmark if row["seed_id"] in pilot_seed_ids]
            planned_calls = len(pilot_items) * 2 * (1 + int(stochastic["samples"]))
            print(f"Pilot items: {len(pilot_items)} ({pilot_seed_count} seeds)")
            print(f"Planned calls for one model across both tasks: {planned_calls}")
            """
        ),
        md("## Run Pilot"),
        code(PROMPT_RUNNER),
        code(
            r"""
            output_path = PROJECT_ROOT / "data/processed/model_outputs_raw_pilot.jsonl"
            run_id = eu.new_run_id("pilot")
            records = []

            if RUN_PILOT:
                for item in pilot_items:
                    for task in ["task1", "task2"]:
                        records.append(run_one(
                            item=item,
                            task=task,
                            model=MODEL,
                            sample_kind="deterministic",
                            sample_index=0,
                            temperature=float(deterministic["temperature"]),
                            top_p=float(deterministic["top_p"]),
                            output_path=output_path,
                            run_id=run_id,
                        ))
                        for sample_index in range(int(stochastic["samples"])):
                            records.append(run_one(
                                item=item,
                                task=task,
                                model=MODEL,
                                sample_kind="stochastic",
                                sample_index=sample_index,
                                temperature=float(stochastic["temperature"]),
                                top_p=float(stochastic["top_p"]),
                                output_path=output_path,
                                run_id=run_id,
                            ))
                print(f"Wrote {len(records)} pilot records to {output_path}")
            else:
                print("Pilot not run. Set RUN_PILOT=true in the environment or edit RUN_PILOT to True.")
            """
        ),
        md("## Pilot Gate"),
        code(
            r"""
            pilot_rows = [row for row in eu.read_jsonl(output_path) if row.get("run_id") == run_id] if RUN_PILOT else []
            if pilot_rows:
                ok = sum(1 for row in pilot_rows if row["parse_status"] == "ok")
                parse_rate = ok / len(pilot_rows)
                avg_latency = sum(float(row["latency_s"] or 0) for row in pilot_rows) / len(pilot_rows)
                print(f"Parse success: {ok}/{len(pilot_rows)} = {parse_rate:.3f}")
                print(f"Average latency: {avg_latency:.2f}s")
                if parse_rate < 0.95:
                    print("Gate failed: inspect invalid outputs before the full run.")
                else:
                    print("Gate passed: parse success is >= 95%.")
            """
        ),
    ]


def notebook_03() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # 03 Run Experiments

            Objective: run both tasks for each locally provided model, cache every raw response, and preserve invalid outputs for auditability.

            This notebook is intentionally guarded. Set `RUN_FULL_EXPERIMENT = True` only after the pilot passes.
            """
        ),
        code(COMMON_SETUP),
        md("## Configure Run"),
        code(
            r"""
            HOST = os.getenv("HOST", CONFIG["llm"]["host"])
            MODELS = [m.strip() for m in os.getenv("MODELS", ",".join(CONFIG["llm"]["models"])).split(",") if m.strip()]
            RUN_FULL_EXPERIMENT = os.getenv("RUN_FULL_EXPERIMENT", "false").lower() in {"1", "true", "yes"}
            deterministic = CONFIG["llm"]["deterministic"]
            stochastic = CONFIG["llm"]["stochastic"]
            output_path = PROJECT_ROOT / "data/processed/model_outputs_raw.jsonl"
            run_id = eu.new_run_id("full")

            benchmark = eu.read_csv_rows(PROJECT_ROOT / "data/processed/benchmark_items.csv")
            planned_calls = len(benchmark) * len(MODELS) * 2 * (1 + int(stochastic["samples"]))
            print({"HOST": HOST, "MODELS": MODELS, "RUN_FULL_EXPERIMENT": RUN_FULL_EXPERIMENT, "run_id": run_id})
            print(f"Benchmark items: {len(benchmark)}")
            print(f"Planned calls: {planned_calls}")
            """
        ),
        md("## Run Full Experiment"),
        code(PROMPT_RUNNER),
        code(
            r"""
            if RUN_FULL_EXPERIMENT:
                total = 0
                for model in MODELS:
                    print(f"Running model: {model}")
                    for item in benchmark:
                        for task in ["task1", "task2"]:
                            run_one(
                                item=item,
                                task=task,
                                model=model,
                                sample_kind="deterministic",
                                sample_index=0,
                                temperature=float(deterministic["temperature"]),
                                top_p=float(deterministic["top_p"]),
                                output_path=output_path,
                                run_id=run_id,
                            )
                            total += 1
                            for sample_index in range(int(stochastic["samples"])):
                                run_one(
                                    item=item,
                                    task=task,
                                    model=model,
                                    sample_kind="stochastic",
                                    sample_index=sample_index,
                                    temperature=float(stochastic["temperature"]),
                                    top_p=float(stochastic["top_p"]),
                                    output_path=output_path,
                                    run_id=run_id,
                                )
                                total += 1
                            if total % 100 == 0:
                                print(f"Completed {total}/{planned_calls} calls")
                print(f"Done. Wrote records to {output_path}")
            else:
                print("Full experiment not run. Set RUN_FULL_EXPERIMENT=true after the pilot gate passes.")
            """
        ),
        md("## Parse-Failure Audit"),
        code(
            r"""
            all_rows = eu.read_jsonl(output_path)
            run_rows = [row for row in all_rows if row.get("run_id") == run_id]
            if run_rows:
                status_counts = {}
                for row in run_rows:
                    status_counts[row["parse_status"]] = status_counts.get(row["parse_status"], 0) + 1
                print(status_counts)
                print(f"Parse success rate: {status_counts.get('ok', 0) / len(run_rows):.3f}")
            else:
                print("No rows for this run_id yet.")
            """
        ),
    ]


def notebook_04() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # 04 Compute UQ and Metrics

            Objective: convert cached model outputs into UQ scores and paper-facing metrics, including calibration and modality-specific diagnostics.
            """
        ),
        code(COMMON_SETUP),
        md("## Load Raw Outputs and Benchmark"),
        code(
            r"""
            benchmark_path = PROJECT_ROOT / "data/processed/benchmark_items.csv"
            raw_path = PROJECT_ROOT / "data/processed/model_outputs_raw.jsonl"

            benchmark = eu.read_csv_rows(benchmark_path)
            all_raw_rows = eu.read_jsonl(raw_path)
            requested_run_id = os.getenv("RUN_ID") or os.getenv("ANALYSIS_RUN_ID")
            selected_run_id, raw_rows = eu.select_run_rows(all_raw_rows, run_id=requested_run_id, prefix="full")
            print(f"Benchmark items: {len(benchmark)}")
            print(f"Raw output rows: {len(all_raw_rows)}")
            print(f"Selected run_id: {selected_run_id}")
            print(f"Selected raw rows: {len(raw_rows)}")
            if all_raw_rows and not raw_rows:
                available = sorted({row.get("run_id", "") for row in all_raw_rows if row.get("run_id")})
                raise ValueError(f"No rows found for selected run_id. Available run_ids: {available[-10:]}")
            """
        ),
        md("## Build UQ Scores"),
        code(
            r"""
            scores = eu.build_uq_scores(benchmark, raw_rows)
            scores_path = PROJECT_ROOT / "data/processed/uq_scores.csv"
            eu.write_csv_rows(scores_path, scores)
            print(f"Wrote UQ scores: {scores_path}")
            print(f"Score rows: {len(scores)}")
            scores[:3]
            """
        ),
        md("## Metric Summary"),
        code(
            r"""
            summary = eu.metric_summary_by_model_task_method(scores)
            summary_path = PROJECT_ROOT / "data/processed/metrics_summary.csv"
            eu.write_csv_rows(summary_path, summary)
            print(f"Wrote summary: {summary_path}")
            fields = [
                "model",
                "task",
                "uq_method",
                "n",
                "accuracy",
                "f1_or_macro_f1",
                "over_commitment",
                "brier",
                "ece",
                "auroc",
                "monotonicity_violations",
                "parse_failure_rate",
            ]
            print(eu.markdown_table(summary, fields))
            """
        ),
        md("## Bootstrap Confidence Intervals Over Seeds"),
        code(
            r"""
            ci_rows = []
            for key, rows in eu.grouped(scores, ["model", "task", "uq_method"]).items():
                model, task, uq_method = key

                def acc_metric(sample_rows):
                    return eu.accuracy_score(
                        [int(row["y_true"]) for row in sample_rows],
                        [int(row["y_pred"]) for row in sample_rows],
                    )

                def brier_metric(sample_rows):
                    return eu.brier_score(
                        [int(row["y_true"]) for row in sample_rows],
                        eu.calibration_probabilities(sample_rows, task),
                    )

                acc_point, acc_low, acc_high = eu.bootstrap_seed_metric(rows, acc_metric, iterations=1000)
                brier_point, brier_low, brier_high = eu.bootstrap_seed_metric(rows, brier_metric, iterations=1000)
                ci_rows.append({
                    "model": model,
                    "task": task,
                    "uq_method": uq_method,
                    "accuracy": acc_point,
                    "accuracy_ci_low": acc_low,
                    "accuracy_ci_high": acc_high,
                    "brier": brier_point,
                    "brier_ci_low": brier_low,
                    "brier_ci_high": brier_high,
                })

            ci_path = PROJECT_ROOT / "data/processed/bootstrap_seed_ci.csv"
            eu.write_csv_rows(ci_path, ci_rows)
            print(f"Wrote bootstrap CIs: {ci_path}")
            print(eu.markdown_table(ci_rows, ["model", "task", "uq_method", "accuracy", "accuracy_ci_low", "accuracy_ci_high", "brier", "brier_ci_low", "brier_ci_high"]))
            """
        ),
        md("## Sensitivity Check: Recommended Strength = 0.75"),
        code(
            r"""
            benchmark_075 = []
            for row in benchmark:
                row = dict(row)
                row["numeric_strength"] = eu.NUMERIC_STRENGTH_RECOMMENDED_075[row["source_modality"]]
                benchmark_075.append(row)

            scores_075 = eu.build_uq_scores(benchmark_075, raw_rows)
            summary_075 = eu.metric_summary_by_model_task_method(scores_075)
            sensitivity_path = PROJECT_ROOT / "data/processed/metrics_summary_recommended075.csv"
            eu.write_csv_rows(sensitivity_path, summary_075)
            print(f"Wrote sensitivity summary: {sensitivity_path}")
            print(eu.markdown_table(summary_075, ["model", "task", "uq_method", "spearman_modality_p_yes"]))
            """
        ),
    ]


def notebook_05() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # 05 Analyze and Export Results

            Objective: export the compact table, one figure, and a short observation template for the IST manuscript.
            """
        ),
        code(COMMON_SETUP),
        md("## Load Scores and Summaries"),
        code(
            r"""
            scores = eu.read_csv_rows(PROJECT_ROOT / "data/processed/uq_scores.csv")
            summary = eu.read_csv_rows(PROJECT_ROOT / "data/processed/metrics_summary.csv")
            ci_rows = eu.read_csv_rows(PROJECT_ROOT / "data/processed/bootstrap_seed_ci.csv")
            print(f"Scores: {len(scores)}")
            print(f"Summary rows: {len(summary)}")
            """
        ),
        md("## Export Paper-Facing Table"),
        code(
            r"""
            paper_fields = [
                "model",
                "task",
                "uq_method",
                "accuracy",
                "f1_or_macro_f1",
                "over_commitment",
                "brier",
                "ece",
                "auroc",
                "monotonicity_violations",
            ]
            table_md = eu.markdown_table(summary, paper_fields)
            table_path = PROJECT_ROOT / "outputs/paper_results_table.md"
            table_path.write_text(table_md + "\n", encoding="utf-8")
            print(f"Wrote table: {table_path}")
            print(table_md)
            """
        ),
        md("## Export Compact Figure"),
        code(
            r"""
            figure_path = PROJECT_ROOT / "outputs/task1_p_yes_by_modality.svg"
            eu.write_task1_modality_svg(scores, figure_path)
            print(f"Wrote figure: {figure_path}")
            """
        ),
        md("## Export Manuscript Observation Notes"),
        code(
            r"""
            notes = [
                "# Result Notes for IST Manuscript",
                "",
                "Fill this file after inspecting the metric table and figure.",
                "",
                "## Observations",
                "- Observation: <grounded result from metrics_summary.csv>.",
                "- Observation: <grounded result from task1_p_yes_by_modality.svg>.",
                "",
                "## Interpretation",
                "- Hypothesis: <what the observed pattern may imply>.",
                "",
                "## Caveats",
                "- Controlled variants are synthetic minimal pairs.",
                "- Confidence values are verbalized or consistency-derived, not direct internal model uncertainty.",
                "- Local model IDs and endpoint configuration must be reported exactly.",
                "",
                "## Recommended Next Step",
                "- Recommendation: <best follow-up experiment or paper edit>.",
            ]
            notes_path = PROJECT_ROOT / "outputs/result_notes_template.md"
            notes_path.write_text("\n".join(notes) + "\n", encoding="utf-8")
            print(f"Wrote notes template: {notes_path}")
            """
        ),
    ]


def main() -> None:
    write_notebook("00_prepare_data.ipynb", notebook_00())
    write_notebook("01_build_modality_benchmark.ipynb", notebook_01())
    write_notebook("02_pilot_local_llms.ipynb", notebook_02())
    write_notebook("03_run_experiments.ipynb", notebook_03())
    write_notebook("04_compute_uq_and_metrics.ipynb", notebook_04())
    write_notebook("05_analyze_and_export_results.ipynb", notebook_05())


if __name__ == "__main__":
    main()
