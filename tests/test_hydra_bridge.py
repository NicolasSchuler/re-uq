"""Tests for the Hydra configuration layer (conf/, scripts/run.py, hydra_bridge).

The Hydra path is an alternative front end, not a second configuration
contract: everything here checks that composing `conf/` lands on exactly the
dictionary the legacy JSON path produces, and that the provenance artifacts the
runners write are complete and free of credentials.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest import mock

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts import (
    compute_acse_semantic_artifacts as semantic_cache,
    eval_utils as eu,
    generate_evaluation_analysis as analysis_cli,
    hydra_bridge as hb,
    run_provenance as rp,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = REPO_ROOT / "conf"
EXAMPLE_CONFIG = REPO_ROOT / "run_configs/full_matrix.example.json"
EXAMPLE_PROFILE_IDS = [
    "local_llama_cpp",
    "institutional_llm",
    "kit_toolbox",
    "zai",
    "openai",
    "mistral",
    "google_gemini",
    "ollama_local",
]


def compose_config(config_dir=CONF_DIR, overrides=()):
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="config", overrides=list(overrides))


class HydraCompositionTest(unittest.TestCase):
    def test_default_composition_is_accepted_by_normalize_run_config(self):
        run_config = eu.normalize_run_config(
            hb.hydra_config_to_run_config(compose_config())
        )
        self.assertEqual(run_config["datasets"], ["mlm_tapt"])
        self.assertEqual(run_config["benchmark_variants"], ["must"])
        self.assertEqual(run_config["tasks"], ["task1", "task2"])
        self.assertEqual(run_config["profiles"][0]["profile_id"], "zai")

    def test_zai_profile_matches_the_json_run_config(self):
        """The two official-run profiles must be identical on both config paths."""
        json_config = eu.load_run_config(EXAMPLE_CONFIG)
        for profile_id in ("zai", "kit_toolbox"):
            with self.subTest(profile=profile_id):
                composed = eu.normalize_run_config(
                    hb.hydra_config_to_run_config(
                        compose_config(overrides=[f"profile={profile_id}"])
                    )
                )
                expected = next(
                    p for p in json_config["profiles"] if p["profile_id"] == profile_id
                )
                self.assertEqual(composed["profiles"][0], expected)
                # Every official run delivered 16 items per request.
                self.assertEqual(composed["profiles"][0]["batch_size"], 16)
                for key in (
                    "run_group_id",
                    "prompt_version",
                    "seed",
                    "batch_order",
                    "deterministic",
                    "stochastic",
                    "logging",
                ):
                    self.assertEqual(composed[key], json_config[key], key)

    def test_every_example_profile_has_an_equivalent_config_group(self):
        json_config = eu.load_run_config(EXAMPLE_CONFIG)
        for profile_id in EXAMPLE_PROFILE_IDS:
            with self.subTest(profile=profile_id):
                composed = eu.normalize_run_config(
                    hb.hydra_config_to_run_config(
                        compose_config(overrides=[f"profile={profile_id}"])
                    )
                )
                expected = next(
                    p for p in json_config["profiles"] if p["profile_id"] == profile_id
                )
                self.assertEqual(composed["profiles"][0], expected)

    def test_model_override_selects_a_single_model(self):
        cfg = compose_config(overrides=["profile=zai", "model=glm-4.5-air"])
        run_config = eu.normalize_run_config(hb.hydra_config_to_run_config(cfg))
        self.assertEqual(run_config["profiles"][0]["models"], ["glm-4.5-air"])
        args = hb.run_args_namespace(cfg)
        self.assertEqual(args.model, "glm-4.5-air")
        self.assertFalse(args.all_models)

    def test_model_override_rejects_a_model_the_profile_does_not_list(self):
        """`model=` is a selector, never a way to invent an unlisted model."""
        cfg = compose_config(overrides=["profile=zai", "model=kit.gemma4-31b-it"])
        with self.assertRaises(ValueError) as context:
            hb.hydra_config_to_run_config(cfg)
        message = str(context.exception)
        self.assertIn("No provider profiles match the requested filters.", message)
        self.assertIn("kit.gemma4-31b-it", message)
        self.assertIn("zai", message)
        # The same model composes cleanly against the profile that does list it.
        self.assertEqual(
            eu.normalize_run_config(
                hb.hydra_config_to_run_config(
                    compose_config(
                        overrides=["profile=kit_toolbox", "model=kit.gemma4-31b-it"]
                    )
                )
            )["profiles"][0]["models"],
            ["kit.gemma4-31b-it"],
        )

    def test_no_model_override_runs_every_model_of_the_profile(self):
        cfg = compose_config(overrides=["profile=openai"])
        self.assertEqual(
            eu.normalize_run_config(hb.hydra_config_to_run_config(cfg))["profiles"][0][
                "models"
            ],
            ["gpt-5-mini", "gpt-5", "gpt-4.1-mini"],
        )
        self.assertTrue(hb.run_args_namespace(cfg).all_models)

    def test_group_overrides_select_dataset_variant_sampling_and_task(self):
        cfg = compose_config(
            overrides=[
                "dataset=nice",
                "variant=shall",
                "sampling=deterministic_only",
                "task=task2",
            ]
        )
        run_config = eu.normalize_run_config(hb.hydra_config_to_run_config(cfg))
        self.assertEqual(run_config["datasets"], ["nice"])
        self.assertEqual(run_config["benchmark_variants"], ["shall"])
        self.assertEqual(run_config["stochastic"]["samples"], 0)
        self.assertEqual(run_config["tasks"], ["task2"])
        self.assertEqual(hb.run_args_namespace(cfg).task, "task2")

    def test_dotted_override_reaches_per_profile_knobs(self):
        cfg = compose_config(
            overrides=[
                "profile=zai",
                "profile.batch_order=shuffled",
                "profile.batch_size=1",
            ]
        )
        profile = eu.normalize_run_config(hb.hydra_config_to_run_config(cfg))[
            "profiles"
        ][0]
        self.assertEqual(profile["batch_order"], "shuffled")
        self.assertEqual(profile["batch_size"], 1)

    def test_task3_composition_maps_to_the_task3_runner_arguments(self):
        cfg = compose_config(
            overrides=[
                "task=task3",
                "source_run_id=full-1",
                "audit_mode=declared_text",
                "dry_run=true",
            ]
        )
        self.assertEqual(eu.normalize_task_filter(str(cfg.task)), ["task3"])
        args = hb.run_args_namespace(cfg)
        self.assertEqual(args.source_run_id, "full-1")
        self.assertEqual(args.audit_mode, "declared_text")
        self.assertTrue(args.dry_run)
        self.assertIsNone(args.task)

    def test_experiment_presets_compose(self):
        for preset, expected in (
            ("paper_cohort", {"task": "both", "mode": "full"}),
            ("batching_ablation", {"task": "task2", "mode": "full"}),
            ("diverse_families", {"task": "both", "mode": "full"}),
        ):
            with self.subTest(experiment=preset):
                cfg = compose_config(overrides=[f"+experiment={preset}"])
                self.assertEqual(str(cfg.task), expected["task"])
                self.assertEqual(str(cfg.mode), expected["mode"])
                eu.normalize_run_config(hb.hydra_config_to_run_config(cfg))

    def test_batching_ablation_preset_targets_the_todo_section_a_cell(self):
        cfg = compose_config(overrides=["+experiment=batching_ablation"])
        run_config = eu.normalize_run_config(hb.hydra_config_to_run_config(cfg))
        self.assertEqual(run_config["datasets"], ["mlm_tapt"])
        self.assertEqual(run_config["benchmark_variants"], ["must"])
        self.assertEqual(run_config["tasks"], ["task2"])
        self.assertEqual(run_config["stochastic"]["samples"], 0)
        self.assertEqual(run_config["profiles"][0]["models"], ["glm-5.1"])
        # The grouped baseline arm is the paper condition: 16 items per request.
        self.assertEqual(run_config["profiles"][0]["batch_size"], 16)

    def test_paper_cohort_preset_sweeps_the_five_glm_models_and_both_variants(self):
        cfg = compose_config(overrides=["+experiment=paper_cohort"])
        run_config = eu.normalize_run_config(hb.hydra_config_to_run_config(cfg))
        self.assertEqual(
            run_config["profiles"][0]["models"],
            ["glm-4.5-air", "glm-4.7", "glm-5", "glm-5-turbo", "glm-5.1"],
        )
        self.assertEqual(run_config["profiles"][0]["batch_size"], 16)
        # `compose()` strips the `hydra` node, so read the sweep off the preset.
        sweep = OmegaConf.load(
            CONF_DIR / "experiment/paper_cohort.yaml"
        ).hydra.sweeper.params
        self.assertEqual(
            str(sweep["model"]), "glm-4.5-air,glm-4.7,glm-5,glm-5-turbo,glm-5.1"
        )
        self.assertEqual(str(sweep["dataset"]), "nice,mlm_tapt")
        self.assertEqual(str(sweep["variant"]), "must,shall")

    def test_default_embedding_selection_flows_into_the_run_config(self):
        cfg = compose_config()
        run_config = eu.normalize_run_config(hb.hydra_config_to_run_config(cfg))
        self.assertEqual(run_config["acse_embedding_backend"], "mlx")
        self.assertEqual(
            run_config["acse_embedding_mlx_model"], eu.ACSE_MLX_DEFAULT_MODEL
        )

    def test_embedding_group_override_selects_the_ablation_model(self):
        cfg = compose_config(overrides=["embedding=qwen3_4b"])
        run_config = eu.normalize_run_config(hb.hydra_config_to_run_config(cfg))
        self.assertEqual(run_config["acse_embedding_backend"], "mlx")
        self.assertEqual(
            run_config["acse_embedding_mlx_model"],
            "mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
        )

    def test_every_embedding_option_has_a_valid_backend_label(self):
        options = sorted(p.stem for p in (CONF_DIR / "embedding").glob("*.yaml"))
        self.assertIn("qwen3_06b", options)
        for option in options:
            cfg = compose_config(overrides=[f"embedding={option}"])
            run_config = eu.normalize_run_config(hb.hydra_config_to_run_config(cfg))
            label, model = eu.semantic_embedding_backend_label(
                run_config["acse_embedding_backend"],
                run_config["acse_embedding_mlx_model"],
            )
            self.assertTrue(label)
            if run_config["acse_embedding_backend"] == "mlx":
                self.assertEqual(label, f"mlx:{model}")

    def test_apply_acse_embedding_env_sets_variables_without_clobbering(self):
        env = {eu.ACSE_EMBEDDING_BACKEND_ENV, eu.ACSE_MLX_MODEL_ENV}
        saved = {name: os.environ.get(name) for name in env}
        try:
            os.environ.pop(eu.ACSE_EMBEDDING_BACKEND_ENV, None)
            os.environ.pop(eu.ACSE_MLX_MODEL_ENV, None)
            eu.apply_acse_embedding_env(
                {
                    "acse_embedding_backend": "mlx",
                    "acse_embedding_mlx_model": "mlx-community/bge-m3-mlx-8bit",
                }
            )
            self.assertEqual(
                os.environ[eu.ACSE_MLX_MODEL_ENV], "mlx-community/bge-m3-mlx-8bit"
            )
            # An explicitly exported value wins over the composed one.
            os.environ[eu.ACSE_MLX_MODEL_ENV] = "manual/override"
            eu.apply_acse_embedding_env(
                {"acse_embedding_mlx_model": "mlx-community/bge-m3-mlx-8bit"}
            )
            self.assertEqual(os.environ[eu.ACSE_MLX_MODEL_ENV], "manual/override")
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


class ResolvedConfigProvenanceTest(unittest.TestCase):
    def test_resolved_yaml_records_env_variable_names_not_secrets(self):
        yaml_text = hb.resolved_config_yaml(compose_config(overrides=["profile=zai"]))
        self.assertIn("api_key_env: ZAI_API_KEY", yaml_text)
        self.assertNotIn("oc.env", yaml_text)

    def test_credential_shaped_values_are_masked(self):
        masked = hb.mask_secrets(
            {
                "api_key_env": "ZAI_API_KEY",
                "api_key": "sk-live",
                "extra_body": {"auth_token": "t", "token": "t"},
                "max_tokens": 256,
                "n": 1,
            }
        )
        self.assertEqual(masked["api_key_env"], "ZAI_API_KEY")
        self.assertEqual(masked["api_key"], hb.MASK_VALUE)
        self.assertEqual(masked["extra_body"]["auth_token"], hb.MASK_VALUE)
        self.assertEqual(masked["extra_body"]["token"], hb.MASK_VALUE)
        # Request knobs that merely mention "token" are not credentials.
        self.assertEqual(masked["max_tokens"], 256)
        self.assertEqual(masked["n"], 1)

    def test_write_resolved_config_returns_digest_and_is_a_no_op_without_one(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            digest = rp.write_resolved_config(root, "smoke-1", "model: glm-5.1\n")
            path = rp.resolved_config_path(root, "smoke-1")
            self.assertTrue(path.exists())
            self.assertEqual(digest, rp.sha256_text("model: glm-5.1\n"))
            self.assertEqual(rp.write_resolved_config(root, "smoke-2", ""), "")
            self.assertFalse(rp.resolved_config_path(root, "smoke-2").exists())

    def test_run_notes_appends_the_digest_only_for_hydra_runs(self):
        class Args:
            mode = "smoke"

        legacy = Args()
        self.assertEqual(rp.run_notes(legacy), "mode=smoke")
        self.assertEqual(
            rp.run_notes(legacy, "audit_mode=blind"), "mode=smoke; audit_mode=blind"
        )
        hydra_args = hb.run_args_namespace(
            compose_config(), hb.resolved_config_yaml(compose_config())
        )
        self.assertEqual(
            rp.run_notes(hydra_args),
            f"mode={hydra_args.mode}; resolved_config_sha={hydra_args.resolved_config_sha}",
        )


class RunConfigExportTest(unittest.TestCase):
    def test_json_run_config_round_trips_through_the_exported_groups(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "conf"
            shutil.copytree(CONF_DIR, out)
            written = hb.run_config_to_hydra_yaml(
                EXAMPLE_CONFIG, out, name="exported", overwrite=True
            )
            self.assertTrue(written["experiment"].exists())
            for profile_id in EXAMPLE_PROFILE_IDS:
                self.assertIn(f"profile/{profile_id}", written)

            json_config = eu.load_run_config(EXAMPLE_CONFIG)
            experiment = OmegaConf.to_container(
                OmegaConf.load(written["experiment"]), resolve=True
            )
            self.assertIsNone(experiment["model"])
            self.assertEqual(
                experiment["hydra"]["sweeper"]["params"]["profile"].split(","),
                EXAMPLE_PROFILE_IDS,
            )
            self.assertNotIn("model", experiment["hydra"]["sweeper"]["params"])
            for profile_id in EXAMPLE_PROFILE_IDS:
                with self.subTest(profile=profile_id):
                    cfg = compose_config(
                        out,
                        [
                            "+experiment=exported",
                            f"profile={profile_id}",
                            "dataset=nice",
                        ],
                    )
                    composed = eu.normalize_run_config(
                        hb.hydra_config_to_run_config(cfg)
                    )
                    expected = next(
                        p
                        for p in json_config["profiles"]
                        if p["profile_id"] == profile_id
                    )
                    self.assertEqual(composed["profiles"][0], expected)
                    self.assertTrue(hb.run_args_namespace(cfg).all_models)
                    for key in (
                        "run_group_id",
                        "prompt_version",
                        "seed",
                        "batch_order",
                        "acse_embedding_backend",
                        "acse_embedding_mlx_model",
                    ):
                        self.assertEqual(composed[key], json_config[key], key)
                    self.assertEqual(
                        composed["deterministic"], json_config["deterministic"]
                    )
                    self.assertEqual(composed["stochastic"], json_config["stochastic"])
                    self.assertEqual(composed["logging"], json_config["logging"])

    def test_json_embedding_selection_round_trips_through_exported_experiment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "run_config.json"
            payload = json.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
            payload["acse_embedding_backend"] = "mlx"
            payload["acse_embedding_mlx_model"] = (
                "mlx-community/Qwen3-Embedding-4B-4bit-DWQ"
            )
            source.write_text(json.dumps(payload), encoding="utf-8")
            out = tmp / "conf"
            shutil.copytree(CONF_DIR, out)
            hb.run_config_to_hydra_yaml(
                source, out, name="exported_embedding", overwrite=True
            )

            cfg = compose_config(
                out,
                [
                    "+experiment=exported_embedding",
                    "profile=zai",
                    "dataset=nice",
                ],
            )
            composed = eu.normalize_run_config(hb.hydra_config_to_run_config(cfg))
            self.assertEqual(composed["acse_embedding_backend"], "mlx")
            self.assertEqual(
                composed["acse_embedding_mlx_model"],
                "mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
            )

    def test_export_refuses_to_clobber_a_different_file_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "conf"
            (out / "profile").mkdir(parents=True)
            (out / "profile/zai.yaml").write_text(
                "profile_id: something-else\n", encoding="utf-8"
            )
            with self.assertRaises(FileExistsError):
                hb.run_config_to_hydra_yaml(EXAMPLE_CONFIG, out, name="exported")

    def test_exported_profiles_never_contain_a_secret_value(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "conf"
            hb.run_config_to_hydra_yaml(EXAMPLE_CONFIG, out, name="exported")
            for path in sorted((out / "profile").glob("*.yaml")):
                text = path.read_text(encoding="utf-8")
                self.assertIn("api_key_env:", text)
                self.assertNotIn("api_key:", text)


class HydraMultirunSmokeTest(unittest.TestCase):
    """`--multirun` over a 2x2 grid must run every cell into the smoke tree."""

    MODELS: ClassVar[list[str]] = ["fake-a", "fake-b"]
    DATASETS: ClassVar[list[str]] = ["nice", "mlm_tapt"]

    def _scaffold(self, root):
        (root / "docs").mkdir(parents=True)
        (root / "AGENTS.md").write_text("", encoding="utf-8")
        (root / "docs/evaluation.md").write_text("", encoding="utf-8")
        (root / "prompts").mkdir()
        for name in ("mandatory_entailment.txt", "modality_extraction.txt"):
            shutil.copyfile(REPO_ROOT / "prompts" / name, root / "prompts" / name)
        (root / "data/processed").mkdir(parents=True)
        benchmark = eu.build_benchmark_items(
            [
                {
                    "seed_id": "S0001",
                    "source_dataset": "NICE",
                    "original_requirement": "The system shall export reports.",
                    "capability_text_final": "export reports",
                }
            ]
        )
        for dataset_id in self.DATASETS:
            eu.write_csv_rows(
                eu.artifact_path(
                    root / "data/processed/benchmark_items.csv", dataset_id, "must"
                ),
                benchmark,
            )

    def test_multirun_writes_every_cell_into_the_smoke_tree(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._scaffold(root)
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts/run.py"),
                "--multirun",
                "profile=local_llama_cpp",
                f"profile.models=[{','.join(self.MODELS)}]",
                "profile.requires_manual_server=false",
                f"model={','.join(self.MODELS)}",
                f"dataset={','.join(self.DATASETS)}",
                "variant=must",
                "embedding=qwen3_4b",
                "mode=smoke",
                "fake_completion=true",
                "smoke_items=1",
            ]
            env = dict(os.environ, RE_UQ_CACHE_DIR=str(root / ".cache"))
            completed = subprocess.run(
                command, cwd=root, capture_output=True, text=True, env=env
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(
                "Launching 4 jobs locally", completed.stdout + completed.stderr
            )

            # Fake smoke runs never touch the paper-facing tree.
            self.assertFalse((root / "data/processed/model_outputs_raw.jsonl").exists())
            observed = {}
            for dataset_id in self.DATASETS:
                path = eu.model_outputs_raw_path(root, dataset_id, "must", smoke=True)
                self.assertTrue(path.exists(), path)
                rows = eu.read_jsonl(path)
                observed[dataset_id] = {str(row["model"]) for row in rows}
                self.assertTrue(
                    all(str(row["run_id"]).startswith("smoke-") for row in rows)
                )
                self.assertEqual(
                    {row["semantic_embedding_backend"] for row in rows},
                    {"mlx:mlx-community/Qwen3-Embedding-4B-4bit-DWQ"},
                )
            self.assertEqual(
                observed, {dataset_id: set(self.MODELS) for dataset_id in self.DATASETS}
            )

            # Every cell dumped its resolved config and referenced it in the registry.
            registry = eu.read_csv_rows(
                eu.run_registry_path(root, "nice", "must", smoke=True)
            )
            self.assertEqual(len(registry), 2)
            for row in registry:
                self.assertEqual(row["status"], "complete")
                digest = dict(part.split("=", 1) for part in row["notes"].split("; "))[
                    "resolved_config_sha"
                ]
                resolved = rp.resolved_config_path(root, row["run_id"])
                self.assertTrue(resolved.exists(), resolved)
                text = resolved.read_text(encoding="utf-8")
                self.assertEqual(rp.sha256_text(text), digest)
                self.assertIn("api_key_env: LOCAL_OPENAI_API_KEY", text)
                self.assertNotIn(hb.MASK_VALUE, text)

            # The independently invoked analysis consumes the raw run's
            # persisted selection and writes the same label to every ACSE row.
            selected_registry_row = registry[0]
            selected_run_id = selected_registry_row["run_id"]
            selected_model = selected_registry_row["model"]
            expected_backend = "mlx:mlx-community/Qwen3-Embedding-4B-4bit-DWQ"
            analysis_dir = root / "outputs/evaluation_embedding_round_trip"
            analysis_argv = [
                "generate_evaluation_analysis.py",
                "--dataset",
                "nice",
                "--variant",
                "must",
                "--run-id",
                selected_run_id,
                "--model",
                selected_model,
                "--profile",
                "local_llama_cpp",
                "--output-dir",
                str(analysis_dir),
                "--bootstrap-iterations",
                "1",
                "--allow-partial",
                "--skip-construct-review-check",
                "--skip-manifest-check",
            ]

            observed_embedding_args = []

            def fake_embedding_matrix(
                texts, embedding_backend=None, mlx_model_name=None
            ):
                observed_embedding_args.append(
                    (embedding_backend, mlx_model_name)
                )
                return np.ones((len(texts), 2), dtype=float), expected_backend

            with (
                mock.patch.object(sys, "argv", analysis_argv),
                mock.patch.object(analysis_cli.eu, "project_root", return_value=root),
                mock.patch.object(
                    analysis_cli.eu,
                    "semantic_embedding_matrix",
                    side_effect=fake_embedding_matrix,
                ),
            ):
                analysis_cli.main()

            acse_rows = [
                row
                for row in eu.read_csv_rows(analysis_dir / "uq_scores.csv")
                if row["uq_method"] == eu.ACSE_PROXY_METHOD
            ]
            self.assertTrue(acse_rows)
            self.assertEqual(
                {row["semantic_embedding_backend"] for row in acse_rows},
                {expected_backend},
            )
            self.assertEqual(
                set(observed_embedding_args),
                {("mlx", "mlx-community/Qwen3-Embedding-4B-4bit-DWQ")},
            )
            analysis_provenance = json.loads(
                (analysis_dir / "provenance_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                analysis_provenance["semantic_embedding_backend"], expected_backend
            )

            completed_runs = semantic_cache.completed_runs_from_analysis_dirs(
                root, root / "outputs"
            )
            cached_run = next(
                run for run in completed_runs if run.analysis_dir == analysis_dir
            )
            self.assertEqual(
                semantic_cache.backend_specs_for_run(cached_run, None, None),
                [("mlx", "mlx-community/Qwen3-Embedding-4B-4bit-DWQ")],
            )

            # Hydra keeps its own sweep artifacts out of the data tree and does
            # not change the working directory of the job.
            self.assertTrue(sorted((root / "outputs/hydra/multirun").glob("*/*/0")))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
