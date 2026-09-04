"""Generate the stripped companion notebooks from version-controlled sources."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(source).strip() + "\n")


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(source).strip() + "\n")


COMMON_SETUP = r"""
from pathlib import Path
import os
import sys
import importlib

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
while not (PROJECT_ROOT / "AGENTS.md").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import eval_utils as eu
eu = importlib.reload(eu)

CONFIG_PATH = PROJECT_ROOT / "config.json"
if not CONFIG_PATH.exists():
    CONFIG_PATH = PROJECT_ROOT / "config.example.json"
CONFIG = eu.load_config(CONFIG_PATH)
eu.ensure_project_dirs(PROJECT_ROOT)
DATASET_ID = eu.normalize_dataset_id(os.getenv("DATASET_ID", "nice"))
DATASET_SUFFIX = eu.dataset_suffix(DATASET_ID)
BENCHMARK_VARIANT = os.getenv("BENCHMARK_VARIANT", "must").strip().lower()
VARIANT_SUFFIX = eu.variant_suffix(BENCHMARK_VARIANT)
ARTIFACT_SUFFIX = eu.dataset_variant_suffix(DATASET_ID, BENCHMARK_VARIANT)

PROJECT_ROOT, CONFIG_PATH, DATASET_ID, BENCHMARK_VARIANT
"""


def notebook_document(cells: list[nbf.NotebookNode]) -> nbf.NotebookNode:
    """Build a stripped notebook document with stable kernel metadata."""
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
    return notebook


def write_notebook(
    name: str, cells: list[nbf.NotebookNode], notebook_dir: Path = NOTEBOOK_DIR
) -> None:
    """Write one generated companion notebook into the target directory."""
    path = notebook_dir / name
    notebook = notebook_document(cells)
    nbf.write(notebook, path)


def populate_notebooks(
    notebook_dir: Path = NOTEBOOK_DIR, *, dry_run: bool = False
) -> None:
    """Generate all companion notebooks, optionally validating without writes."""
    for name, builder in NOTEBOOK_BUILDERS:
        cells = builder()
        path = notebook_dir / name
        # Exercise full notebook serialization even in dry-run mode so a reviewer
        # can probe the generator without rewriting tracked notebooks.
        nbf.writes(notebook_document(cells))
        if dry_run:
            print(f"Would write: {path}")
        else:
            write_notebook(name, cells, notebook_dir)
            print(f"Wrote: {path}")


def notebook_00() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # 00 Prepare Data

            Objective: load the NICE/PROMISE-derived requirements dataset, create a transparent seed review table, and keep the configured number of accepted seed capabilities for the main experiment.

            The automatic filter and capability cleanup are intentionally conservative. Manual review remains part of the protocol, but `capability_text_final` is pre-filled with a cleaned suggestion so most rows should only need inspection rather than hand rewriting.
            """
        ),
        code(COMMON_SETUP),
        md("## Load Source Dataset"),
        code(
            r"""
            if DATASET_ID == "nice":
                nice_path = PROJECT_ROOT / CONFIG["datasets"]["nice_local_path"]
                nice_url = CONFIG["datasets"]["nice_url"]

                if not nice_path.exists():
                    print(f"NICE CSV not found at {nice_path}. Downloading from Zenodo...")
                    eu.download_file(nice_url, nice_path, timeout_s=CONFIG["llm"]["timeout_s"])
                else:
                    print(f"Using existing NICE CSV: {nice_path}")

                rows = eu.read_csv_rows(nice_path)
                text_column = eu.find_requirement_text_column(rows)
                print(f"Loaded {len(rows)} NICE rows. Requirement text column: {text_column}")
                print(rows[0][text_column][:240])
            elif DATASET_ID == "mlm_tapt":
                rows = eu.load_mlm_tapt_rows(CONFIG)
                text_column = "reqs"
                print(f"Loaded {len(rows)} mlm_tapt rows from Hugging Face.")
                print(rows[0][text_column][:240])
            else:
                raise ValueError(f"Unsupported DATASET_ID: {DATASET_ID}")
            """
        ),
        md("## Build Review Table"),
        code(
            r"""
            target_count = eu.dataset_target_seed_count(CONFIG, DATASET_ID)
            if DATASET_ID == "nice":
                candidates = eu.make_seed_candidates(rows, target_count=target_count)
            else:
                candidates = eu.make_mlm_tapt_seed_candidates(
                    rows,
                    target_count=target_count,
                    seed=int(CONFIG["project"]["seed"]),
                    exclude_source_regex=CONFIG["datasets"]["mlm_tapt_exclude_source_regex"],
                    source_cap=30,
                )

            review_fields = eu.seed_review_fields(DATASET_ID)
            review_path = eu.artifact_path(PROJECT_ROOT / "data/processed/seeds_review.csv", DATASET_ID)
            auto_candidates_path = eu.auto_candidates_path(review_path)

            if review_path.exists():
                eu.write_csv_rows(auto_candidates_path, candidates, fieldnames=review_fields)
                print(f"Existing review table found: {review_path}")
                print("Did not overwrite reviewed/manual capability edits.")
                print(f"Wrote fresh automatic candidates for comparison: {auto_candidates_path}")
            else:
                eu.write_csv_rows(review_path, candidates, fieldnames=review_fields)
                print(f"Wrote new review table: {review_path}")

            auto_ok = sum(1 for row in candidates if row["auto_include"] == "yes")
            selected = eu.load_reviewed_seeds(review_path, target_count=target_count, strict=False)
            if DATASET_ID == "mlm_tapt":
                included_sources = {}
                for row in selected:
                    source = row.get("source_corpus", "")
                    included_sources[source] = included_sources.get(source, 0) + 1
                print(f"Included source_corpus counts: {included_sources}")
            print(f"Automatic candidates passing filters: {auto_ok}")
            print(f"Currently included seeds: {len(selected)} / {target_count}")
            print("Inspect capability_text_final for included rows; edit only unclear or awkward suggestions.")
            """
        ),
        md("## Optionally Refresh Existing Capability Suggestions"),
        code(
            r"""
            # Manual review edits are preserved by default.
            # Set RUN_REFRESH = True only if you want to refresh unedited capability suggestions.
            RUN_REFRESH = False
            review_path = eu.artifact_path(PROJECT_ROOT / "data/processed/seeds_review.csv", DATASET_ID)

            if RUN_REFRESH:
                refreshed_count = eu.refresh_capability_suggestions_file(review_path)
                print(f"Refreshed capability_text_final suggestions: {refreshed_count}")
            else:
                print("Skipped refresh; reviewed/manual capability edits were left untouched.")
            """
        ),
        md("## Inspect Included Capability Suggestions"),
        code(
            r"""
            import pandas as pd

            review_df = pd.read_csv(review_path, dtype=str, keep_default_na=False)
            included = review_df[review_df["include"] == "yes"].copy()
            suspicious = included[
                included["capability_text_final"].str.contains(r"\b(shall|must|should|may|system|product|application)\b", case=False, regex=True)
                | included["capability_text_final"].str.contains(r"[.;:]", regex=True)
                | included["capability_text_final"].str.contains(r"^(with|to|from|for|of|in|on|at|by|about|into|onto|through|across|under|over|between|among)\b", case=False, regex=True)
            ]
            print(f"Included rows: {len(included)}")
            print(f"Rows worth closer manual review: {len(suspicious)}")
            suspicious[["seed_id", "original_requirement", "capability_text_final"]].head(20)
            """
        ),
        md("## Show Full Included Capability Review Table"),
        code(
            r"""
            capability_review = eu.included_capability_review_frame(review_path)
            export_paths = eu.write_included_capability_review(review_path, PROJECT_ROOT / "outputs", suffix=DATASET_SUFFIX)

            print(f"Included rows: {len(capability_review)}")
            print(f"Wrote Markdown review table: {export_paths['markdown']}")
            print(f"Wrote CSV review table: {export_paths['csv']}")

            with pd.option_context("display.max_rows", None, "display.max_colwidth", 180, "display.width", 240):
                display(capability_review)
            """
        ),
        md("## Validate Reviewed Seeds"),
        code(
            r"""
            selected = eu.load_reviewed_seeds(review_path, target_count=target_count, strict=False)
            selected_path = eu.artifact_path(PROJECT_ROOT / "data/processed/seeds_selected.csv", DATASET_ID)

            if len(selected) == target_count:
                eu.write_csv_rows(selected_path, selected)
                print(f"OK: exactly {target_count} included seeds.")
                print(f"Wrote selected seeds: {selected_path}")
            else:
                print(f"Review needed: found {len(selected)} included seeds, expected {target_count}.")
                print(f"Edit {review_path}, then rerun this cell.")

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
            target_count = eu.dataset_target_seed_count(CONFIG, DATASET_ID)
            review_path = eu.artifact_path(PROJECT_ROOT / "data/processed/seeds_review.csv", DATASET_ID)
            seeds = eu.load_reviewed_seeds(review_path, target_count=target_count, strict=True)
            print(f"Loaded {len(seeds)} reviewed seeds.")
            seeds[:2]
            """
        ),
        md("## Generate Four Modality Variants Per Seed"),
        code(
            r"""
            benchmark = eu.build_benchmark_items(seeds)
            benchmark_path = eu.artifact_path(PROJECT_ROOT / "data/processed/benchmark_items.csv", DATASET_ID)
            candidate_benchmark_path = eu.candidate_path(benchmark_path)

            # Existing benchmark artifacts are preserved by default after review.
            # Set FORCE_REBUILD_BENCHMARK = True only when you intentionally accept regenerated items.
            FORCE_REBUILD_BENCHMARK = False
            write_result = eu.write_csv_rows_if_changed(
                benchmark_path,
                benchmark,
                candidate_path=candidate_benchmark_path,
                overwrite=FORCE_REBUILD_BENCHMARK,
            )

            if write_result["status"] == "written":
                print(f"Wrote benchmark: {benchmark_path}")
            elif write_result["status"] == "overwritten":
                print(f"Overwrote benchmark by explicit request: {benchmark_path}")
            elif write_result["status"] == "unchanged":
                print(f"Existing benchmark matches regenerated items: {benchmark_path}")
            else:
                print(f"Existing benchmark preserved: {benchmark_path}")
                print(f"Regenerated candidate differs and was written for review: {write_result['candidate_path']}")

            benchmark = eu.read_csv_rows(benchmark_path)
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
        md("## Export Benchmark Statements For Review"),
        code(
            r"""
            import pandas as pd

            benchmark_review = eu.benchmark_statement_review_frame(benchmark)
            review_paths = eu.write_benchmark_statement_review(benchmark, PROJECT_ROOT / "outputs", suffix=DATASET_SUFFIX)

            print(f"Review rows: {len(benchmark_review)}")
            print(f"Wrote Markdown review table: {review_paths['markdown']}")
            print(f"Wrote CSV review table: {review_paths['csv']}")

            with pd.option_context("display.max_rows", 20, "display.max_colwidth", 160, "display.width", 240):
                display(benchmark_review.head(20))
            """
        ),
        md("## Build SHALL Robustness Benchmark"),
        code(
            r"""
            shall_benchmark = eu.build_benchmark_items(seeds, mandatory_keyword="SHALL")
            shall_path = eu.artifact_path(PROJECT_ROOT / "data/processed/benchmark_items.csv", DATASET_ID, "shall")
            candidate_shall_path = eu.candidate_path(shall_path)

            FORCE_REBUILD_SHALL_BENCHMARK = False
            shall_write_result = eu.write_csv_rows_if_changed(
                shall_path,
                shall_benchmark,
                candidate_path=candidate_shall_path,
                overwrite=FORCE_REBUILD_SHALL_BENCHMARK,
            )

            if shall_write_result["status"] == "written":
                print(f"Wrote SHALL benchmark: {shall_path}")
            elif shall_write_result["status"] == "overwritten":
                print(f"Overwrote SHALL benchmark by explicit request: {shall_path}")
            elif shall_write_result["status"] == "unchanged":
                print(f"Existing SHALL benchmark matches regenerated items: {shall_path}")
            else:
                print(f"Existing SHALL benchmark preserved: {shall_path}")
                print(f"Regenerated candidate differs and was written for review: {shall_write_result['candidate_path']}")

            shall_benchmark = eu.read_csv_rows(shall_path)
            assert len(shall_benchmark) == expected_items
            assert len({row["item_id"] for row in shall_benchmark}) == expected_items
            assert all(row["task1_gold_decision"] == ("yes" if row["source_modality"] == "mandatory" else "no") for row in shall_benchmark)
            assert all(row["task2_gold_modality"] == row["source_modality"] for row in shall_benchmark)
            assert all("SHALL" in row["source_statement"] for row in shall_benchmark if row["source_modality"] == "mandatory")
            assert all("SHALL" in row["candidate_requirement"] for row in shall_benchmark)

            shall_review_paths = eu.write_benchmark_statement_review(shall_benchmark, PROJECT_ROOT / "outputs", suffix=eu.dataset_variant_suffix(DATASET_ID, "shall"))
            print(f"SHALL items: {len(shall_benchmark)}")
            print(f"Wrote SHALL Markdown review table: {shall_review_paths['markdown']}")
            print(f"Wrote SHALL CSV review table: {shall_review_paths['csv']}")
            """
        ),
        md("## Write Benchmark Manifest"),
        code(
            r"""
            manifest_path = eu.artifact_path(PROJECT_ROOT / "outputs/benchmark_manifest.json", DATASET_ID)
            manifest_paths = [
                eu.artifact_path(PROJECT_ROOT / "data/processed/seeds_review.csv", DATASET_ID),
                eu.artifact_path(PROJECT_ROOT / "data/processed/seeds_selected.csv", DATASET_ID),
                eu.artifact_path(PROJECT_ROOT / "data/processed/benchmark_items.csv", DATASET_ID),
                eu.artifact_path(PROJECT_ROOT / "data/processed/benchmark_items.csv", DATASET_ID, "shall"),
                PROJECT_ROOT / "prompts/mandatory_entailment.txt",
                PROJECT_ROOT / "prompts/mandatory_entailment_strict.txt",
                PROJECT_ROOT / "prompts/modality_extraction.txt",
                PROJECT_ROOT / "prompts/modality_extraction_labels_only.txt",
                PROJECT_ROOT / "prompts/modality_verification.txt",
                PROJECT_ROOT / "prompts/modality_verification_declared.txt",
            ]
            manifest = eu.write_benchmark_manifest(
                manifest_paths,
                manifest_path,
                root=PROJECT_ROOT,
                metadata={
                    "main_benchmark": "MUST",
                    "robustness_benchmark": "SHALL",
                    "dataset_id": DATASET_ID,
                    "seed_count": target_count,
                    "source_modalities": eu.MODALITIES,
                },
            )
            print(f"Wrote manifest: {manifest_path}")
            print(f"Artifacts recorded: {len(manifest['artifacts'])}")
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

def request_job(
    item,
    task,
    model,
    sample_kind,
    sample_index,
    temperature,
    top_p,
    run_id,
    request_index,
    prompt=None,
    prompt_version=None,
):
    prompt = prompt if prompt is not None else prompt_for(task, item)
    return {
        "request_index": request_index,
        "run_id": run_id,
        "model": model,
        "host": HOST,
        "task": task,
        "item": item,
        "sample_index": sample_index,
        "sample_kind": sample_kind,
        "temperature": temperature,
        "top_p": top_p,
        "prompt_version": prompt_version or CONFIG["project"]["prompt_version"],
        "prompt": prompt,
        "max_tokens": int(CONFIG["llm"]["max_tokens"]),
        "timeout_s": int(CONFIG["llm"]["timeout_s"]),
        "api_key_env": CONFIG["llm"]["api_key_env"],
    }
"""


def notebook_02() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # 02 Pilot Local LLMs

            Objective: run a bounded pilot against one local OpenAI-compatible model and check JSON parse rate before the full experiment.

            By default this notebook calls the configured model. Set `RUN_PILOT=false` to skip requests.
            """
        ),
        code(COMMON_SETUP),
        md("## Configure Local Endpoint"),
        code(
            r"""
            HOST = os.getenv("HOST", CONFIG["llm"]["host"])
            configured_models = [m.strip() for m in os.getenv("MODELS", ",".join(CONFIG["llm"]["models"])).split(",") if m.strip()]
            MODEL = os.getenv("MODEL", configured_models[0])
            RUN_PILOT = os.getenv("RUN_PILOT", "true").lower() in {"1", "true", "yes"}
            REQUEST_CONCURRENCY = eu.resolve_llm_concurrency(CONFIG)

            deterministic = CONFIG["llm"]["deterministic"]
            stochastic = CONFIG["llm"]["stochastic"]
            print({
                "HOST": HOST,
                "MODEL": MODEL,
                "RUN_PILOT": RUN_PILOT,
                "REQUEST_CONCURRENCY": REQUEST_CONCURRENCY,
                "BENCHMARK_VARIANT": BENCHMARK_VARIANT,
            })
            """
        ),
        md("## Select Pilot Items"),
        code(
            r"""
            benchmark_path = eu.artifact_path(PROJECT_ROOT / "data/processed/benchmark_items.csv", DATASET_ID, BENCHMARK_VARIANT)
            benchmark = eu.read_csv_rows(benchmark_path)
            pilot_seed_count = int(CONFIG["project"]["pilot_seed_count"])
            pilot_seed_ids = sorted({row["seed_id"] for row in benchmark})[:pilot_seed_count]
            pilot_items = [row for row in benchmark if row["seed_id"] in pilot_seed_ids]
            planned_calls = len(pilot_items) * 2 * (1 + int(stochastic["samples"]))
            print(f"Benchmark path: {benchmark_path}")
            print(f"Pilot items: {len(pilot_items)} ({pilot_seed_count} seeds)")
            print(f"Planned calls for one model across both tasks: {planned_calls}")
            """
        ),
        md("## Run Pilot"),
        code(PROMPT_RUNNER),
        code(
            r"""
            output_path = eu.artifact_path(PROJECT_ROOT / "data/processed/model_outputs_raw_pilot.jsonl", DATASET_ID, BENCHMARK_VARIANT)
            run_id = eu.new_run_id("pilot" if BENCHMARK_VARIANT == "must" else f"pilot-{BENCHMARK_VARIANT}")
            records = []
            jobs = []

            for item in pilot_items:
                for task in ["task1", "task2"]:
                    jobs.append(request_job(
                        item=item,
                        task=task,
                        model=MODEL,
                        sample_kind="deterministic",
                        sample_index=0,
                        temperature=float(deterministic["temperature"]),
                        top_p=float(deterministic["top_p"]),
                        run_id=run_id,
                        request_index=len(jobs),
                    ))
                    for sample_index in range(int(stochastic["samples"])):
                        jobs.append(request_job(
                            item=item,
                            task=task,
                            model=MODEL,
                            sample_kind="stochastic",
                            sample_index=sample_index,
                            temperature=float(stochastic["temperature"]),
                            top_p=float(stochastic["top_p"]),
                            run_id=run_id,
                            request_index=len(jobs),
                        ))
            assert len(jobs) == planned_calls

            if RUN_PILOT:
                print(f"Dispatching {len(jobs)} pilot calls with concurrency={REQUEST_CONCURRENCY}")
                for record in eu.run_completion_jobs(jobs, max_workers=REQUEST_CONCURRENCY):
                    eu.append_jsonl(output_path, record)
                    records.append(record)
                    if len(records) % 25 == 0 or len(records) == len(jobs):
                        print(f"Completed {len(records)}/{len(jobs)} pilot calls")
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
        md("## Survey-Aligned UQ Pilot Diagnostics"),
        code(
            r"""
            if pilot_rows:
                pilot_scores = eu.build_uq_scores(benchmark, pilot_rows, sampling_plan=eu.SamplingPlan.from_run_config(CONFIG))
                fields = [
                    "model",
                    "task",
                    "uq_method",
                    "source_modality",
                    "p_yes",
                    "confidence",
                    "uncertainty_score",
                    "uncertainty_measure",
                    "valid_n",
                    "total_n",
                ]
                diagnostic_rows = [
                    row for row in pilot_scores
                    if row["uq_method"] in {"label_self_consistency", "modality_consistency", "predictive_entropy", "variation_ratio", eu.ACSE_PROXY_METHOD}
                ]
                print(eu.markdown_table(diagnostic_rows[:24], fields))
            else:
                print("No pilot rows available. Run the pilot to inspect stochastic UQ diagnostics.")
            """
        ),
        md("## Optional Logprob Capability Probe"),
        code(
            r"""
            RUN_LOGPROB_PROBE = os.getenv("RUN_LOGPROB_PROBE", "true").lower() in {"1", "true", "yes"}
            logprob_probe_path = eu.artifact_path(PROJECT_ROOT / "outputs/logprob_probe.json", DATASET_ID, BENCHMARK_VARIANT)

            if RUN_LOGPROB_PROBE:
                probe = eu.logprob_support_probe(
                    host=HOST,
                    model=MODEL,
                    api_key_env=CONFIG["llm"]["api_key_env"],
                    timeout_s=int(CONFIG["llm"]["timeout_s"]),
                )
                eu.write_json(logprob_probe_path, probe)
                print(probe)
            else:
                print("Logprob probe not run. Set RUN_LOGPROB_PROBE=true to test token-level UQ support via /v1/responses.")
                print(f"Probe output path when enabled: {logprob_probe_path}")
            """
        ),
        md("## Prompt Sensitivity Check"),
        code(
            r"""
            strict_template = eu.load_prompt(PROJECT_ROOT / "prompts/mandatory_entailment_strict.txt")
            sensitivity_raw_path = eu.artifact_path(PROJECT_ROOT / "data/processed/model_outputs_raw_prompt_sensitivity.jsonl", DATASET_ID, BENCHMARK_VARIANT)
            sensitivity_summary_path = eu.artifact_path(PROJECT_ROOT / "outputs/prompt_sensitivity_summary.csv", DATASET_ID, BENCHMARK_VARIANT)
            RUN_PROMPT_SENSITIVITY = os.getenv("RUN_PROMPT_SENSITIVITY", "true").lower() in {"1", "true", "yes"}

            sensitivity_records = []
            sensitivity_run_id = eu.new_run_id("prompt-sensitivity" if BENCHMARK_VARIANT == "must" else f"prompt-sensitivity-{BENCHMARK_VARIANT}")
            task1_pilot_items = list(pilot_items)
            sensitivity_jobs = []

            if RUN_PROMPT_SENSITIVITY:
                for prompt_name, template in [("default", task1_template), ("strict", strict_template)]:
                    for item in task1_pilot_items:
                        prompt = eu.render_prompt(
                            template,
                            source_statement=item["source_statement"],
                            candidate_requirement=item["candidate_requirement"],
                        )
                        sensitivity_jobs.append(request_job(
                            run_id=f"{sensitivity_run_id}-{prompt_name}",
                            model=f"{MODEL}:{prompt_name}",
                            task="task1",
                            item=item,
                            sample_index=0,
                            sample_kind="deterministic",
                            temperature=float(deterministic["temperature"]),
                            top_p=float(deterministic["top_p"]),
                            prompt_version=f"{CONFIG['project']['prompt_version']}:{prompt_name}",
                            prompt=prompt,
                            request_index=len(sensitivity_jobs),
                        ))
                print(f"Dispatching {len(sensitivity_jobs)} prompt-sensitivity calls with concurrency={REQUEST_CONCURRENCY}")
                for record in eu.run_completion_jobs(sensitivity_jobs, max_workers=REQUEST_CONCURRENCY):
                    eu.append_jsonl(sensitivity_raw_path, record)
                    sensitivity_records.append(record)
                    if len(sensitivity_records) % 25 == 0 or len(sensitivity_records) == len(sensitivity_jobs):
                        print(f"Completed {len(sensitivity_records)}/{len(sensitivity_jobs)} prompt-sensitivity calls")
                print(f"Wrote {len(sensitivity_records)} prompt-sensitivity records to {sensitivity_raw_path}")
            else:
                print("Prompt sensitivity not run. Set RUN_PROMPT_SENSITIVITY=true or RUN_PILOT=true.")

            sensitivity_summary = eu.prompt_sensitivity_summary(benchmark, sensitivity_records)
            eu.write_csv_rows(
                sensitivity_summary_path,
                sensitivity_summary,
                fieldnames=["model", "prompt_run_id", "n", "accuracy", "weak_source_high_p_yes_80", "weak_source_high_p_yes_90", "mean_weak_p_yes"],
            )
            print(f"Wrote prompt sensitivity summary: {sensitivity_summary_path}")
            print(eu.markdown_table(sensitivity_summary, ["model", "prompt_run_id", "n", "accuracy", "weak_source_high_p_yes_80", "weak_source_high_p_yes_90", "mean_weak_p_yes"]))
            """
        ),
        md("## Task 2 Modality Prompt Validity Check"),
        code(
            r"""
            task2_labels_only_template = eu.load_prompt(PROJECT_ROOT / "prompts/modality_extraction_labels_only.txt")
            task2_sensitivity_raw_path = eu.artifact_path(PROJECT_ROOT / "data/processed/model_outputs_raw_task2_prompt_sensitivity.jsonl", DATASET_ID, BENCHMARK_VARIANT)
            task2_sensitivity_summary_path = eu.artifact_path(PROJECT_ROOT / "outputs/task2_prompt_sensitivity_summary.csv", DATASET_ID, BENCHMARK_VARIANT)
            RUN_TASK2_PROMPT_SENSITIVITY = os.getenv("RUN_TASK2_PROMPT_SENSITIVITY", "true").lower() in {"1", "true", "yes"}

            task2_sensitivity_items = [row for row in pilot_items if row["source_modality"] == "nice_to_have"]
            task2_sensitivity_records = []
            task2_sensitivity_run_id = eu.new_run_id("task2-prompt-sensitivity" if BENCHMARK_VARIANT == "must" else f"task2-prompt-sensitivity-{BENCHMARK_VARIANT}")
            task2_sensitivity_jobs = []

            if RUN_TASK2_PROMPT_SENSITIVITY:
                for prompt_name, template in [("default", task2_template), ("labels_only", task2_labels_only_template)]:
                    for item in task2_sensitivity_items:
                        prompt = eu.render_prompt(template, source_statement=item["source_statement"])
                        task2_sensitivity_jobs.append(request_job(
                            run_id=f"{task2_sensitivity_run_id}-{prompt_name}",
                            model=f"{MODEL}:task2_{prompt_name}",
                            task="task2",
                            item=item,
                            sample_index=0,
                            sample_kind="deterministic",
                            temperature=float(deterministic["temperature"]),
                            top_p=float(deterministic["top_p"]),
                            prompt_version=f"{CONFIG['project']['prompt_version']}:task2_{prompt_name}",
                            prompt=prompt,
                            request_index=len(task2_sensitivity_jobs),
                        ))
                print(f"Dispatching {len(task2_sensitivity_jobs)} Task 2 prompt-validity calls with concurrency={REQUEST_CONCURRENCY}")
                for record in eu.run_completion_jobs(task2_sensitivity_jobs, max_workers=REQUEST_CONCURRENCY):
                    eu.append_jsonl(task2_sensitivity_raw_path, record)
                    task2_sensitivity_records.append(record)
                    if len(task2_sensitivity_records) % 10 == 0 or len(task2_sensitivity_records) == len(task2_sensitivity_jobs):
                        print(f"Completed {len(task2_sensitivity_records)}/{len(task2_sensitivity_jobs)} Task 2 prompt-validity calls")
                print(f"Wrote {len(task2_sensitivity_records)} Task 2 prompt-validity records to {task2_sensitivity_raw_path}")
            else:
                print("Task 2 prompt-validity check not run. Set RUN_TASK2_PROMPT_SENSITIVITY=true or edit the flag to True.")

            task2_sensitivity_summary = eu.task2_prompt_sensitivity_summary(benchmark, task2_sensitivity_records)
            task2_sensitivity_fields = [
                "model",
                "prompt_run_id",
                "n",
                "valid_n",
                "parse_success_rate",
                "accuracy",
                "nice_to_have_n",
                "nice_to_have_accuracy",
                "nice_to_have_to_recommended_rate",
                "over_commitment",
                "high_conf_overcommit_80",
                "high_conf_overcommit_90",
            ]
            eu.write_csv_rows(task2_sensitivity_summary_path, task2_sensitivity_summary, fieldnames=task2_sensitivity_fields)
            print(f"Wrote Task 2 prompt-validity summary: {task2_sensitivity_summary_path}")
            print(eu.markdown_table(task2_sensitivity_summary, task2_sensitivity_fields))
            """
        ),
    ]


def notebook_02b() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # 02b Weak-Modality Robustness Probe

            Objective: test whether the pilot `nice_to_have` to `recommended` collapse is tied to one wording or generalizes across weak stakeholder-intent phrasings.

            This formative probe uses the same 20 pilot seeds, Task 2 only, and the existing modality extraction prompt. It does not change the main benchmark.
            """
        ),
        code(COMMON_SETUP),
        md("## Configure Probe"),
        code(
            r"""
            HOST = os.getenv("HOST", CONFIG["llm"]["host"])
            configured_models = [m.strip() for m in os.getenv("MODELS", ",".join(CONFIG["llm"]["models"])).split(",") if m.strip()]
            MODEL = os.getenv("MODEL", configured_models[0])
            RUN_WEAK_MODALITY_PROBE = os.getenv("RUN_WEAK_MODALITY_PROBE", "true").lower() in {"1", "true", "yes"}
            RUN_WEAK_MODALITY_STOCHASTIC = os.getenv("RUN_WEAK_MODALITY_STOCHASTIC", "false").lower() in {"1", "true", "yes"}
            REQUEST_CONCURRENCY = eu.resolve_llm_concurrency(CONFIG)

            deterministic = CONFIG["llm"]["deterministic"]
            stochastic = CONFIG["llm"]["stochastic"]
            print({
                "HOST": HOST,
                "MODEL": MODEL,
                "RUN_WEAK_MODALITY_PROBE": RUN_WEAK_MODALITY_PROBE,
                "RUN_WEAK_MODALITY_STOCHASTIC": RUN_WEAK_MODALITY_STOCHASTIC,
                "REQUEST_CONCURRENCY": REQUEST_CONCURRENCY,
                "BENCHMARK_VARIANT": BENCHMARK_VARIANT,
            })
            """
        ),
        md("## Build Probe Items"),
        code(
            r"""
            benchmark_path = eu.artifact_path(PROJECT_ROOT / "data/processed/benchmark_items.csv", DATASET_ID, BENCHMARK_VARIANT)
            seeds_path = eu.artifact_path(PROJECT_ROOT / "data/processed/seeds_selected.csv", DATASET_ID)
            probe_items_path = eu.artifact_path(PROJECT_ROOT / "data/processed/weak_modality_probe_items.csv", DATASET_ID, BENCHMARK_VARIANT)

            benchmark = eu.read_csv_rows(benchmark_path)
            seed_rows = eu.read_csv_rows(seeds_path)
            pilot_seed_count = int(CONFIG["project"]["pilot_seed_count"])
            pilot_seed_ids = sorted({row["seed_id"] for row in benchmark})[:pilot_seed_count]
            pilot_seed_order = {seed_id: index for index, seed_id in enumerate(pilot_seed_ids)}
            pilot_seeds = sorted(
                [row for row in seed_rows if row["seed_id"] in pilot_seed_order],
                key=lambda row: pilot_seed_order[row["seed_id"]],
            )
            assert len(pilot_seeds) == pilot_seed_count

            probe_items = eu.build_weak_modality_probe_items(pilot_seeds)
            expected_items = pilot_seed_count * len(eu.WEAK_MODALITY_PROBE_TEMPLATES)
            assert len(probe_items) == expected_items
            assert len({row["item_id"] for row in probe_items}) == expected_items
            assert {row["task2_gold_modality"] for row in probe_items} == {"nice_to_have"}

            eu.write_csv_rows(probe_items_path, probe_items, fieldnames=eu.WEAK_MODALITY_PROBE_FIELDS)
            print(f"Probe items: {len(probe_items)}")
            print(f"Wrote probe items: {probe_items_path}")
            print(eu.markdown_table(probe_items[:8], ["item_id", "template_id", "source_statement", "task2_gold_modality"]))
            """
        ),
        md("## Pre-Model Sanity Check"),
        code(
            r"""
            sanity_paths = eu.write_weak_modality_template_sanity_check(PROJECT_ROOT / "outputs", suffix=DATASET_SUFFIX)
            sanity_rows = eu.read_csv_rows(sanity_paths["csv"])
            sanity_status = eu.weak_modality_sanity_status(sanity_rows)
            PROBE_READY = bool(sanity_status["valid"])

            print(f"Sanity CSV: {sanity_paths['csv']}")
            print(f"Sanity Markdown: {sanity_paths['markdown']}")
            print(sanity_status)
            if not PROBE_READY:
                print("Probe is gated: mark each template as weaker_than_should=yes in the sanity CSV before model execution.")
            print(eu.markdown_table(sanity_rows, eu.WEAK_MODALITY_SANITY_FIELDS))
            """
        ),
        md("## Run Task 2 Probe"),
        code(PROMPT_RUNNER),
        code(
            r"""
            output_path = eu.artifact_path(PROJECT_ROOT / "data/processed/model_outputs_raw_weak_modality_probe.jsonl", DATASET_ID, BENCHMARK_VARIANT)
            run_id = eu.new_run_id("weak-modality-probe" if BENCHMARK_VARIANT == "must" else f"weak-modality-probe-{BENCHMARK_VARIANT}")
            records = []
            jobs = []

            for item in probe_items:
                jobs.append(request_job(
                    item=item,
                    task="task2",
                    model=MODEL,
                    sample_kind="deterministic",
                    sample_index=0,
                    temperature=float(deterministic["temperature"]),
                    top_p=float(deterministic["top_p"]),
                    run_id=run_id,
                    request_index=len(jobs),
                ))
                if RUN_WEAK_MODALITY_STOCHASTIC:
                    for sample_index in range(int(stochastic["samples"])):
                        jobs.append(request_job(
                            item=item,
                            task="task2",
                            model=MODEL,
                            sample_kind="stochastic",
                            sample_index=sample_index,
                            temperature=float(stochastic["temperature"]),
                            top_p=float(stochastic["top_p"]),
                            run_id=run_id,
                            request_index=len(jobs),
                        ))

            if RUN_WEAK_MODALITY_PROBE and PROBE_READY:
                print(f"Dispatching {len(jobs)} weak-modality probe calls with concurrency={REQUEST_CONCURRENCY}")
                for record in eu.run_completion_jobs(jobs, max_workers=REQUEST_CONCURRENCY):
                    eu.append_jsonl(output_path, record)
                    records.append(record)
                    if len(records) % 20 == 0 or len(records) == len(jobs):
                        print(f"Completed {len(records)}/{len(jobs)} weak-modality probe calls")
                print(f"Wrote {len(records)} weak-modality probe records to {output_path}")
            elif not PROBE_READY:
                print("Weak-modality probe not run because the sanity check is incomplete.")
            else:
                print("Weak-modality probe not run. Set RUN_WEAK_MODALITY_PROBE=true or edit the flag to True.")
            """
        ),
        md("## Summarize Probe"),
        code(
            r"""
            if records:
                probe_run_id, probe_rows = run_id, records
            elif output_path.exists():
                probe_run_id, probe_rows = eu.select_run_rows(
                    eu.read_jsonl(output_path),
                    prefix="weak-modality-probe" if BENCHMARK_VARIANT == "must" else f"weak-modality-probe-{BENCHMARK_VARIANT}",
                )
            else:
                probe_run_id, probe_rows = None, []

            summary = eu.weak_modality_probe_summary(probe_items, probe_rows)
            summary_paths = eu.write_weak_modality_probe_summary(summary, PROJECT_ROOT / "outputs", suffix=DATASET_SUFFIX)
            print(f"Selected probe run: {probe_run_id}")
            print(f"Wrote summary CSV: {summary_paths['csv']}")
            print(f"Wrote summary Markdown: {summary_paths['markdown']}")
            print(eu.markdown_table(summary, eu.WEAK_MODALITY_PROBE_SUMMARY_FIELDS))
            """
        ),
        md("## Decision Rule"),
        code(
            r"""
            print("Interpretation guide:")
            print("- Most templates collapse to recommended: proceed to full runs with robustness note.")
            print("- Only useful_if collapses: treat the pilot as phrase-specific and revise or narrow the claim.")
            print("- Mixed labels: proceed cautiously and frame weak modality as lexically sensitive.")
            print("- Sanity check not valid: hold and revise the weak-modality construct.")
            """
        ),
    ]


def notebook_03() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # 03 Run Experiments

            Objective: run both tasks for selected models, cache every raw response, and preserve invalid outputs for auditability.

            The provider-aware CLI runner is the recommended path for GPU/server runs. The notebook cells below remain available for direct OpenAI-compatible local runs.
            """
        ),
        code(COMMON_SETUP),
        md("## Provider-Aware CLI Runner"),
        code(
            r"""
            import shlex

            RUN_CONFIG = Path(os.getenv("RUN_CONFIG", PROJECT_ROOT / "run_configs/current_run.json"))
            profile_hint = os.getenv("RUN_PROFILE", "local_llama_cpp")
            model_hint = os.getenv("RUN_MODEL", "")
            mode_hint = os.getenv("RUN_MODE", "smoke")

            command = [
                ".venv/bin/python",
                "scripts/run_experiment_from_config.py",
                "--config",
                str(RUN_CONFIG),
                "--profile",
                profile_hint,
                "--dataset",
                DATASET_ID,
                "--variant",
                BENCHMARK_VARIANT,
                "--mode",
                mode_hint,
            ]
            if model_hint:
                command.extend(["--model", model_hint])

            print("Recommended server command:")
            print(" ".join(shlex.quote(part) for part in command))
            print()
            print("Copy run_configs/full_matrix.example.json to run_configs/current_run.json and edit it for the provider/model matrix.")
            print("For llama.cpp, start the server with one model, then pass --model for that loaded model.")
            print("Set batch_size in the selected provider profile to reduce API calls while preserving one JSONL row per item/sample.")
            """
        ),
        md("## Configure Run"),
        code(
            r"""
            HOST = os.getenv("HOST", CONFIG["llm"]["host"])
            MODELS = [m.strip() for m in os.getenv("MODELS", ",".join(CONFIG["llm"]["models"])).split(",") if m.strip()]
            RUN_FULL_EXPERIMENT = os.getenv("RUN_FULL_EXPERIMENT", "true").lower() in {"1", "true", "yes"}
            deterministic = CONFIG["llm"]["deterministic"]
            stochastic = CONFIG["llm"]["stochastic"]
            REQUEST_CONCURRENCY = eu.resolve_llm_concurrency(CONFIG)
            SAVE_PRELIMINARY_RESULTS = os.getenv("SAVE_PRELIMINARY_RESULTS", "true").lower() in {"1", "true", "yes"}
            PRELIMINARY_EVERY_N_CALLS = max(1, int(os.getenv("PRELIMINARY_EVERY_N_CALLS", "50")))
            output_path = eu.artifact_path(PROJECT_ROOT / "data/processed/model_outputs_raw.jsonl", DATASET_ID, BENCHMARK_VARIANT)
            run_id = eu.new_run_id("full" if BENCHMARK_VARIANT == "must" else f"full-{BENCHMARK_VARIANT}")

            benchmark_path = eu.artifact_path(PROJECT_ROOT / "data/processed/benchmark_items.csv", DATASET_ID, BENCHMARK_VARIANT)
            benchmark = eu.read_csv_rows(benchmark_path)
            planned_calls = len(benchmark) * len(MODELS) * 2 * (1 + int(stochastic["samples"]))
            print({
                "HOST": HOST,
                "MODELS": MODELS,
                "RUN_FULL_EXPERIMENT": RUN_FULL_EXPERIMENT,
                "REQUEST_CONCURRENCY": REQUEST_CONCURRENCY,
                "SAVE_PRELIMINARY_RESULTS": SAVE_PRELIMINARY_RESULTS,
                "PRELIMINARY_EVERY_N_CALLS": PRELIMINARY_EVERY_N_CALLS,
                "run_id": run_id,
                "BENCHMARK_VARIANT": BENCHMARK_VARIANT,
            })
            print(f"Benchmark path: {benchmark_path}")
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
                last_snapshot_at = 0
                jobs = []

                def write_preliminary_snapshot(total_calls):
                    run_rows = [row for row in eu.read_jsonl(output_path) if row.get("run_id") == run_id]
                    snapshot = eu.write_preliminary_result_snapshot(
                        benchmark,
                        run_rows,
                        PROJECT_ROOT,
                        variant=BENCHMARK_VARIANT,
                        dataset_id=DATASET_ID,
                        sampling_plan=eu.SamplingPlan.from_run_config(CONFIG),
                    )
                    print(
                        f"Saved preliminary snapshot after {total_calls}/{planned_calls} calls: "
                        f"{snapshot['summary_rows']} summary rows, {snapshot['progress_rows']} progress rows"
                    )
                    print(f"Preliminary table: {snapshot['paths']['table']}")

                for model in MODELS:
                    for item in benchmark:
                        for task in ["task1", "task2"]:
                            jobs.append(request_job(
                                item=item,
                                task=task,
                                model=model,
                                sample_kind="deterministic",
                                sample_index=0,
                                temperature=float(deterministic["temperature"]),
                                top_p=float(deterministic["top_p"]),
                                run_id=run_id,
                                request_index=len(jobs),
                            ))
                            for sample_index in range(int(stochastic["samples"])):
                                jobs.append(request_job(
                                    item=item,
                                    task=task,
                                    model=model,
                                    sample_kind="stochastic",
                                    sample_index=sample_index,
                                    temperature=float(stochastic["temperature"]),
                                    top_p=float(stochastic["top_p"]),
                                    run_id=run_id,
                                    request_index=len(jobs),
                                ))
                assert len(jobs) == planned_calls

                print(f"Dispatching {len(jobs)} full-experiment calls with concurrency={REQUEST_CONCURRENCY}")
                for record in eu.run_completion_jobs(jobs, max_workers=REQUEST_CONCURRENCY):
                    eu.append_jsonl(output_path, record)
                    total += 1
                    if total % 100 == 0 or total == planned_calls:
                        print(f"Completed {total}/{planned_calls} calls")
                    if SAVE_PRELIMINARY_RESULTS and total - last_snapshot_at >= PRELIMINARY_EVERY_N_CALLS:
                        write_preliminary_snapshot(total)
                        last_snapshot_at = total
                if SAVE_PRELIMINARY_RESULTS:
                    write_preliminary_snapshot(total)
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
        md("## Preliminary Snapshot Files"),
        code(
            r"""
            paths = eu.preliminary_result_paths(PROJECT_ROOT, BENCHMARK_VARIANT, dataset_id=DATASET_ID)
            for name, path in paths.items():
                print(f"{name}: {path} ({'exists' if path.exists() else 'missing'})")
            """
        ),
    ]


def notebook_03b() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # 03b Run Modality Self-Audit

            Objective: audit deterministic Task 2 extracted text with a source-grounded blind Task 3 prompt.

            This diagnostic does not revise Task 2 outputs. It asks the same model whether its extracted requirement preserves, strengthens, weakens, or changes the source.
            """
        ),
        code(COMMON_SETUP),
        md("## Configure Self-Audit Run"),
        code(
            r"""
            HOST = os.getenv("HOST", CONFIG["llm"]["host"])
            RUN_TASK3_VERIFICATION = os.getenv("RUN_TASK3_VERIFICATION", "true").lower() in {"1", "true", "yes"}
            deterministic = CONFIG["llm"]["deterministic"]
            stochastic = CONFIG["llm"]["stochastic"]
            REQUEST_CONCURRENCY = eu.resolve_llm_concurrency(CONFIG)
            TASK3_AUDIT_MODE = eu.normalize_task3_audit_mode(os.getenv("TASK3_AUDIT_MODE", eu.OFFICIAL_TASK3_AUDIT_MODE))

            benchmark_path = eu.artifact_path(PROJECT_ROOT / "data/processed/benchmark_items.csv", DATASET_ID, BENCHMARK_VARIANT)
            source_raw_path = eu.artifact_path(PROJECT_ROOT / "data/processed/model_outputs_raw.jsonl", DATASET_ID, BENCHMARK_VARIANT)
            output_path = eu.artifact_path(PROJECT_ROOT / "data/processed/model_outputs_raw_task3_verification.jsonl", DATASET_ID, BENCHMARK_VARIANT)

            benchmark = eu.read_csv_rows(benchmark_path)
            all_source_rows = eu.read_jsonl(source_raw_path)
            requested_source_run_id = os.getenv("TASK3_SOURCE_RUN_ID") or os.getenv("RUN_ID")
            run_prefix = "full" if BENCHMARK_VARIANT == "must" else f"full-{BENCHMARK_VARIANT}"

            if requested_source_run_id:
                source_run_id, source_rows = eu.select_run_rows(all_source_rows, run_id=requested_source_run_id, prefix=run_prefix)
            else:
                progress = eu.run_progress_summary(
                    benchmark,
                    all_source_rows,
                    expected_stochastic_samples=int(stochastic["samples"]),
                )
                complete_run_ids = eu.complete_run_ids_from_progress(progress, prefix=run_prefix)
                if not complete_run_ids:
                    available = sorted({row.get("run_id", "") for row in all_source_rows if str(row.get("run_id", "")).startswith(run_prefix)})
                    raise ValueError(
                        "No complete full run found for Task 3 text audit. "
                        f"Set TASK3_SOURCE_RUN_ID explicitly or finish a full run. Available run_ids: {available[-10:]}"
                    )
                source_run_id, source_rows = eu.select_run_rows(all_source_rows, run_id=complete_run_ids[-1], prefix=run_prefix)

            run_id = eu.new_run_id("task3" if BENCHMARK_VARIANT == "must" else f"task3-{BENCHMARK_VARIANT}")
            task3_items_path = eu.task3_verification_items_path(
                PROJECT_ROOT,
                DATASET_ID,
                BENCHMARK_VARIANT,
                source_run_id,
                "notebook_multi_model",
                TASK3_AUDIT_MODE,
            )
            print({
                "HOST": HOST,
                "RUN_TASK3_VERIFICATION": RUN_TASK3_VERIFICATION,
                "REQUEST_CONCURRENCY": REQUEST_CONCURRENCY,
                "TASK3_AUDIT_MODE": TASK3_AUDIT_MODE,
                "source_run_id": source_run_id,
                "task3_run_id": run_id,
                "BENCHMARK_VARIANT": BENCHMARK_VARIANT,
            })
            print(f"Benchmark path: {benchmark_path}")
            print(f"Task 1/2 raw path: {source_raw_path}")
            print(f"Task 3 output path: {output_path}")
            """
        ),
        md("## Build Task 3 Self-Audit Items"),
        code(
            r"""
            task3_items = eu.build_task3_verification_items(benchmark, source_rows, audit_mode=TASK3_AUDIT_MODE)
            eu.write_csv_rows(task3_items_path, task3_items, fieldnames=eu.TASK3_VERIFICATION_FIELDS)
            print(f"Wrote Task 3 text-audit items: {task3_items_path}")
            print(f"Task 3 items: {len(task3_items)}")
            if not task3_items:
                raise ValueError("No Task 3 items were built. Check that the selected run has valid deterministic Task 2 rows.")
            print(eu.markdown_table(task3_items[:8], ["item_id", "source_modality", "task2_modality", "task2_text_modality", "task3_declared_relation", "task3_gold_relation"]))
            """
        ),
        md("## Run Task 3 Self-Audit"),
        code(
            r"""
            task3_prompt_path = (
                PROJECT_ROOT / "prompts/modality_verification.txt"
                if TASK3_AUDIT_MODE == eu.OFFICIAL_TASK3_AUDIT_MODE
                else PROJECT_ROOT / "prompts/modality_verification_declared.txt"
            )
            task3_template = eu.load_prompt(task3_prompt_path)

            def task3_prompt_for(item):
                values = {
                    "source_statement": item["source_statement"],
                    "extracted_requirement": item["task2_requirement"],
                }
                if TASK3_AUDIT_MODE == "declared_text":
                    values["declared_extracted_modality"] = item["task2_text_modality"]
                elif TASK3_AUDIT_MODE == "declared_source":
                    values["declared_extracted_modality"] = item["source_modality"]
                return eu.render_prompt(task3_template, **values)

            def task3_request_job(item, sample_kind, sample_index, temperature, top_p, request_index):
                return {
                    "request_index": request_index,
                    "run_id": run_id,
                    "model": item["task2_model"],
                    "host": HOST,
                    "task": "task3",
                    "item": item,
                    "sample_index": sample_index,
                    "sample_kind": sample_kind,
                    "temperature": temperature,
                    "top_p": top_p,
                    "prompt_version": f"{CONFIG['project']['prompt_version']}:task3:{TASK3_AUDIT_MODE}",
                    "prompt": task3_prompt_for(item),
                    "max_tokens": int(CONFIG["llm"]["max_tokens"]),
                    "timeout_s": int(CONFIG["llm"]["timeout_s"]),
                    "api_key_env": CONFIG["llm"]["api_key_env"],
                }

            jobs = []
            for item in task3_items:
                jobs.append(task3_request_job(
                    item=item,
                    sample_kind="deterministic",
                    sample_index=0,
                    temperature=float(deterministic["temperature"]),
                    top_p=float(deterministic["top_p"]),
                    request_index=len(jobs),
                ))
                for sample_index in range(int(stochastic["samples"])):
                    jobs.append(task3_request_job(
                        item=item,
                        sample_kind="stochastic",
                        sample_index=sample_index,
                        temperature=float(stochastic["temperature"]),
                        top_p=float(stochastic["top_p"]),
                        request_index=len(jobs),
                    ))

            planned_calls = len(task3_items) * (1 + int(stochastic["samples"]))
            assert len(jobs) == planned_calls
            records = []

            if RUN_TASK3_VERIFICATION:
                print(f"Dispatching {len(jobs)} Task 3 calls with concurrency={REQUEST_CONCURRENCY}")
                for record in eu.run_completion_jobs(jobs, max_workers=REQUEST_CONCURRENCY):
                    eu.append_jsonl(output_path, record)
                    records.append(record)
                    if len(records) % 50 == 0 or len(records) == len(jobs):
                        print(f"Completed {len(records)}/{len(jobs)} Task 3 calls")
                print(f"Wrote {len(records)} Task 3 records to {output_path}")
            else:
                print("Task 3 text audit not run. Set RUN_TASK3_VERIFICATION=true to execute it.")
            """
        ),
        md("## Verification Summary"),
        code(
            r"""
            task3_rows = [row for row in eu.read_jsonl(output_path) if row.get("run_id") == run_id] if RUN_TASK3_VERIFICATION else []
            if task3_rows:
                status_counts = {}
                for row in task3_rows:
                    status_counts[row["parse_status"]] = status_counts.get(row["parse_status"], 0) + 1
                print(status_counts)
                print(f"Parse success rate: {status_counts.get('ok', 0) / len(task3_rows):.3f}")
                task3_scores = eu.build_task3_scores(task3_items, task3_rows)
                summary = eu.metric_summary_by_model_task_method(task3_scores)
                fields = [
                    "model",
                    "task",
                    "uq_method",
                    "n",
                    "accuracy",
                    "f1_or_macro_f1",
                    "strengthening_recall",
                    "false_preserve_rate",
                    "evidence_phrase_source_rate",
                    "brier",
                    "ece",
                    "error_detection_auroc",
                    "selective_error_defer_10",
                    "selective_error_defer_20",
                    "parse_failure_rate",
                ]
                print(eu.markdown_table(summary, fields))
            else:
                print("No Task 3 rows for this run_id yet.")
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
            benchmark_path = eu.artifact_path(PROJECT_ROOT / "data/processed/benchmark_items.csv", DATASET_ID, BENCHMARK_VARIANT)
            raw_path = eu.artifact_path(PROJECT_ROOT / "data/processed/model_outputs_raw.jsonl", DATASET_ID, BENCHMARK_VARIANT)
            task3_raw_path = eu.artifact_path(PROJECT_ROOT / "data/processed/model_outputs_raw_task3_verification.jsonl", DATASET_ID, BENCHMARK_VARIANT)
            TASK3_AUDIT_MODE = eu.normalize_task3_audit_mode(os.getenv("TASK3_AUDIT_MODE", eu.OFFICIAL_TASK3_AUDIT_MODE))

            benchmark = eu.read_csv_rows(benchmark_path)
            all_raw_rows = eu.read_jsonl(raw_path)
            requested_run_id = os.getenv("RUN_ID") or os.getenv("ANALYSIS_RUN_ID")
            run_prefix = "full" if BENCHMARK_VARIANT == "must" else f"full-{BENCHMARK_VARIANT}"
            if requested_run_id:
                selected_run_id, raw_rows = eu.select_run_rows(all_raw_rows, run_id=requested_run_id, prefix=run_prefix)
            else:
                progress = eu.run_progress_summary(
                    benchmark,
                    all_raw_rows,
                    expected_stochastic_samples=int(CONFIG["llm"]["stochastic"]["samples"]),
                )
                complete_run_ids = eu.complete_run_ids_from_progress(progress, prefix=run_prefix)
                if not complete_run_ids:
                    available = sorted({row.get("run_id", "") for row in all_raw_rows if str(row.get("run_id", "")).startswith(run_prefix)})
                    raise ValueError(
                        "No complete full run found for metric computation. "
                        f"Set RUN_ID or ANALYSIS_RUN_ID explicitly. Available run_ids: {available[-10:]}"
                    )
                selected_run_id, raw_rows = eu.select_run_rows(all_raw_rows, run_id=complete_run_ids[-1], prefix=run_prefix)
            all_task3_rows = eu.read_jsonl(task3_raw_path)
            requested_task3_run_id = os.getenv("TASK3_RUN_ID")

            def task3_row_matches_audit_mode(row):
                raw_mode = str(row.get("task3_audit_mode", "")).strip()
                row_mode = eu.normalize_task3_audit_mode(raw_mode) if raw_mode else eu.LEGACY_TASK3_AUDIT_MODE
                return row_mode == TASK3_AUDIT_MODE

            if requested_task3_run_id:
                selected_task3_run_id, task3_raw_rows = eu.select_run_rows(all_task3_rows, run_id=requested_task3_run_id, prefix="task3")
                task3_raw_rows = [row for row in task3_raw_rows if task3_row_matches_audit_mode(row)]
            else:
                source_task3_rows = [
                    row
                    for row in all_task3_rows
                    if row.get("task2_run_id") == selected_run_id and task3_row_matches_audit_mode(row)
                ]
                selected_task3_run_id = eu.latest_run_id(source_task3_rows, prefix="task3")
                task3_raw_rows = [row for row in source_task3_rows if row.get("run_id") == selected_task3_run_id] if selected_task3_run_id else []
            task3_items = eu.task3_items_from_raw_rows(task3_raw_rows)
            task3_item_models = sorted({row.get("task2_model") or row.get("model", "") for row in task3_raw_rows})
            task3_items_path = None
            if not task3_items and len(task3_item_models) == 1:
                task3_items_path = eu.task3_verification_items_path(
                    PROJECT_ROOT,
                    DATASET_ID,
                    BENCHMARK_VARIANT,
                    selected_run_id,
                    task3_item_models[0],
                    TASK3_AUDIT_MODE,
                )
                all_task3_items = eu.read_csv_rows(task3_items_path) if task3_items_path.exists() else []
                task3_items = [row for row in all_task3_items if row.get("task2_run_id") == selected_run_id]
            result_benchmark = eu.benchmark_rows_with_current_raw_outputs(benchmark, raw_rows)
            stale_item_count = len(benchmark) - len(result_benchmark)
            print(f"Benchmark variant: {BENCHMARK_VARIANT}")
            print(f"Benchmark path: {benchmark_path}")
            print(f"Raw output path: {raw_path}")
            print(f"Benchmark items: {len(benchmark)}")
            print(f"Result-scored benchmark items: {len(result_benchmark)}")
            print(f"Benchmark items without current raw prompts: {stale_item_count}")
            print(f"Raw output rows: {len(all_raw_rows)}")
            print(f"Selected run_id: {selected_run_id}")
            print(f"Selected raw rows: {len(raw_rows)}")
            print(f"Task 3 audit mode: {TASK3_AUDIT_MODE}")
            print(f"Task 3 items path: {task3_items_path} ({'exists' if task3_items_path and task3_items_path.exists() else 'from raw rows or missing'})")
            print(f"Selected Task 3 run_id: {selected_task3_run_id}")
            print(f"Selected Task 3 items: {len(task3_items)}")
            print(f"Selected Task 3 raw rows: {len(task3_raw_rows)}")
            if all_raw_rows and not raw_rows:
                available = sorted({row.get("run_id", "") for row in all_raw_rows if row.get("run_id")})
                raise ValueError(f"No rows found for selected run_id. Available run_ids: {available[-10:]}")
            """
        ),
        md("## Build UQ Scores"),
        code(
            r"""
            scores = eu.build_uq_scores(result_benchmark, raw_rows, sampling_plan=eu.SamplingPlan.from_run_config(CONFIG))
            task3_scores = eu.build_task3_scores(task3_items, task3_raw_rows) if task3_items and task3_raw_rows else []
            baseline_scores = eu.build_rule_baseline_scores(result_benchmark)
            scores.extend(task3_scores)
            scores.extend(baseline_scores)
            scores_path = eu.artifact_path(PROJECT_ROOT / "data/processed/uq_scores.csv", DATASET_ID, BENCHMARK_VARIANT)
            eu.write_csv_rows(scores_path, scores)
            acse_normalized = eu.acse_normalized_score_rows(scores)
            acse_calibration = eu.acse_calibration_diagnostic_rows(acse_normalized)
            acse_normalized_path = eu.artifact_path(PROJECT_ROOT / "data/processed/acse_semantic_normalized_scores.csv", DATASET_ID, BENCHMARK_VARIANT)
            acse_calibration_path = eu.artifact_path(PROJECT_ROOT / "data/processed/acse_semantic_calibration.csv", DATASET_ID, BENCHMARK_VARIANT)
            eu.write_csv_rows(acse_normalized_path, acse_normalized, fieldnames=eu.ACSE_NORMALIZED_SCORE_FIELDS)
            eu.write_csv_rows(acse_calibration_path, acse_calibration, fieldnames=eu.ACSE_CALIBRATION_FIELDS)
            print(f"Wrote UQ scores: {scores_path}")
            print(f"Wrote ACSE normalized scores: {acse_normalized_path}")
            print(f"Wrote ACSE calibration diagnostics: {acse_calibration_path}")
            print(f"Score rows: {len(scores)} including {len(baseline_scores)} rule-baseline rows and {len(task3_scores)} Task 3 rows")
            print(f"ACSE normalized rows: {len(acse_normalized)}; calibration rows: {len(acse_calibration)}")
            scores[:3]
            """
        ),
        md("## Metric Summary"),
        code(
            r"""
            summary = eu.metric_summary_by_model_task_method(scores)
            summary_path = eu.artifact_path(PROJECT_ROOT / "data/processed/metrics_summary.csv", DATASET_ID, BENCHMARK_VARIANT)
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
                "monotonicity_strict_violations",
                "monotonicity_tolerance",
                "monotonicity_mean_max_increase",
                "monotonicity_max_increase",
                "pearson_modality_p_yes",
                "high_conf_overcommit_80",
                "high_conf_overcommit_90",
                "unsupported_mandatory_acceptance_80",
                "unsupported_mandatory_acceptance_90",
                "high_conf_overcommit_all_80",
                "high_conf_overcommit_all_90",
                "high_conf_overcommit_overcommittable_80",
                "high_conf_overcommit_overcommittable_90",
                "weak_recall",
                "weak_strengthening_80",
                "weak_strengthening_90",
                "over_commitment_severity_all",
                "over_commitment_severity_given_overcommitment",
                "text_modality_accuracy",
                "text_modality_accuracy_all",
                "text_modality_parse_coverage",
                "label_text_consistency",
                "text_over_commitment",
                "text_high_conf_overcommit_80",
                "text_high_conf_overcommit_90",
                "strengthening_recall",
                "false_preserve_rate",
                "evidence_phrase_source_rate",
                "error_detection_auroc",
                "selective_error_defer_10",
                "selective_error_defer_20",
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
                    return eu.task_accuracy(sample_rows, task)

                def brier_metric(sample_rows):
                    return eu.brier_score(
                        [int(row["y_true"]) for row in sample_rows],
                        eu.calibration_probabilities(sample_rows, task),
                    )

                acc_point, acc_low, acc_high = eu.bootstrap_seed_metric(rows, acc_metric, iterations=1000)
                brier_point, brier_low, brier_high = eu.bootstrap_seed_metric(rows, brier_metric, iterations=1000)
                ci_row = {
                    "model": model,
                    "task": task,
                    "uq_method": uq_method,
                    "accuracy": acc_point,
                    "accuracy_ci_low": acc_low,
                    "accuracy_ci_high": acc_high,
                    "brier": brier_point,
                    "brier_ci_low": brier_low,
                    "brier_ci_high": brier_high,
                }
                ci_row.update(eu.headline_risk_ci_fields(rows, task, iterations=1000))
                ci_rows.append(ci_row)

            ci_path = eu.artifact_path(PROJECT_ROOT / "data/processed/bootstrap_seed_ci.csv", DATASET_ID, BENCHMARK_VARIANT)
            eu.write_csv_rows(ci_path, ci_rows)
            print(f"Wrote bootstrap CIs: {ci_path}")
            print(eu.markdown_table(ci_rows, [
                "model",
                "task",
                "uq_method",
                "accuracy",
                "accuracy_ci_low",
                "accuracy_ci_high",
                "brier",
                "brier_ci_low",
                "brier_ci_high",
                "unsupported_mandatory_acceptance_80_ci_low",
                "unsupported_mandatory_acceptance_80_ci_high",
                "high_conf_overcommit_overcommittable_80_ci_low",
                "high_conf_overcommit_overcommittable_80_ci_high",
                "weak_strengthening_80_ci_low",
                "weak_strengthening_80_ci_high",
            ]))
            """
        ),
        md("## Sensitivity Check: Recommended Strength = 0.75"),
        code(
            r"""
            benchmark_075 = []
            for row in result_benchmark:
                row = dict(row)
                row["numeric_strength"] = eu.NUMERIC_STRENGTH_RECOMMENDED_075[row["source_modality"]]
                benchmark_075.append(row)

            scores_075 = eu.build_uq_scores(benchmark_075, raw_rows, sampling_plan=eu.SamplingPlan.from_run_config(CONFIG))
            scores_075.extend(eu.build_rule_baseline_scores(benchmark_075))
            summary_075 = eu.metric_summary_by_model_task_method(scores_075)
            sensitivity_path = eu.artifact_path(PROJECT_ROOT / "data/processed/metrics_summary_recommended075.csv", DATASET_ID, BENCHMARK_VARIANT)
            eu.write_csv_rows(sensitivity_path, summary_075)
            print(f"Wrote sensitivity summary: {sensitivity_path}")
            print(eu.markdown_table(summary_075, ["model", "task", "uq_method", "spearman_modality_p_yes", "pearson_modality_p_yes"]))
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
            benchmark_path = eu.artifact_path(PROJECT_ROOT / "data/processed/benchmark_items.csv", DATASET_ID, BENCHMARK_VARIANT)
            scores_path = eu.artifact_path(PROJECT_ROOT / "data/processed/uq_scores.csv", DATASET_ID, BENCHMARK_VARIANT)
            summary_path = eu.artifact_path(PROJECT_ROOT / "data/processed/metrics_summary.csv", DATASET_ID, BENCHMARK_VARIANT)
            ci_path = eu.artifact_path(PROJECT_ROOT / "data/processed/bootstrap_seed_ci.csv", DATASET_ID, BENCHMARK_VARIANT)

            benchmark = eu.read_csv_rows(benchmark_path)
            scores = eu.read_csv_rows(scores_path)
            summary = eu.read_csv_rows(summary_path)
            ci_rows = eu.read_csv_rows(ci_path)
            print(f"Benchmark variant: {BENCHMARK_VARIANT}")
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
                "error_detection_auroc",
                "selective_error_defer_10",
                "selective_error_defer_20",
                "monotonicity_violations",
                "monotonicity_strict_violations",
                "monotonicity_tolerance",
                "monotonicity_mean_max_increase",
                "monotonicity_max_increase",
                "unsupported_mandatory_acceptance_90",
                "high_conf_overcommit_all_90",
                "high_conf_overcommit_overcommittable_90",
                "weak_recall",
                "weak_strengthening_90",
                "over_commitment_severity_all",
                "over_commitment_severity_given_overcommitment",
                "text_modality_parse_coverage",
                "label_text_consistency",
                "strengthening_recall",
                "false_preserve_rate",
                "evidence_phrase_source_rate",
            ]
            table_md = eu.markdown_table(summary, paper_fields)
            table_path = eu.artifact_path(PROJECT_ROOT / "outputs/paper_results_table.md", DATASET_ID, BENCHMARK_VARIANT)
            table_path.write_text(table_md + "\n", encoding="utf-8")
            print(f"Wrote table: {table_path}")
            print(table_md)
            """
        ),
        md("## Export Compact Figure"),
        code(
            r"""
            figure_path = eu.artifact_path(PROJECT_ROOT / "outputs/task1_p_yes_by_modality.svg", DATASET_ID, BENCHMARK_VARIANT)
            eu.write_task1_modality_svg(scores, figure_path)
            print(f"Wrote figure: {figure_path}")
            """
        ),
        md("## Export Qualitative Over-Commitment Examples"),
        code(
            r"""
            example_paths = eu.write_qualitative_overcommitment_examples(
                scores,
                benchmark,
                PROJECT_ROOT / "outputs",
                suffix=VARIANT_SUFFIX,
                limit=5,
                threshold=0.80,
            )
            print(f"Wrote qualitative examples CSV: {example_paths['csv']}")
            print(f"Wrote qualitative examples Markdown: {example_paths['markdown']}")
            """
        ),
        md("## Export UQ Method Inventory"),
        code(
            r"""
            inventory_paths = eu.write_uq_method_inventory(PROJECT_ROOT / "outputs", suffix=VARIANT_SUFFIX)
            print(f"Wrote UQ inventory Markdown: {inventory_paths['markdown']}")
            print(f"Wrote UQ inventory CSV: {inventory_paths['csv']}")
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
                "- Observation: <grounded high-confidence over-commitment result>.",
                "- Observation: <grounded blind Task 3 text-audit result>.",
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
            notes_path = eu.artifact_path(PROJECT_ROOT / "outputs/result_notes_template.md", DATASET_ID, BENCHMARK_VARIANT)
            notes_path.write_text("\n".join(notes) + "\n", encoding="utf-8")
            print(f"Wrote notes template: {notes_path}")
            """
        ),
    ]


NOTEBOOK_BUILDERS = [
    ("00_prepare_data.ipynb", notebook_00),
    ("01_build_modality_benchmark.ipynb", notebook_01),
    ("02_pilot_local_llms.ipynb", notebook_02),
    ("02b_weak_modality_robustness_probe.ipynb", notebook_02b),
    ("03_run_experiments.ipynb", notebook_03),
    ("03b_run_modality_verification.ipynb", notebook_03b),
    ("04_compute_uq_and_metrics.ipynb", notebook_04),
    ("05_analyze_and_export_results.ipynb", notebook_05),
]


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for regenerating or dry-running companion notebooks."""
    parser = argparse.ArgumentParser(
        description="Generate the stripped companion notebooks from source builders."
    )
    parser.add_argument(
        "--notebook-dir",
        type=Path,
        default=NOTEBOOK_DIR,
        help="Directory to write notebooks into. Defaults to this checkout's notebooks/ directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and serialize notebooks without writing files.",
    )
    args = parser.parse_args(argv)

    populate_notebooks(args.notebook_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
