"""Bridge between the Hydra `conf/` composition and the legacy run-config dict.

The pipeline has exactly one configuration contract: the dictionary produced by
`eval_utils.normalize_run_config`. The JSON files under `run_configs/` are one
way to build it (the one the paper runs used); the Hydra config groups under
`conf/` are another. This module converts a composed `DictConfig` into that same
dictionary shape, builds the argparse-compatible namespace the runners expect,
and exports an existing JSON run config back into `conf/` groups for migration.

No secrets ever enter a config file: provider profiles carry `api_key_env`, the
*name* of the environment variable holding the key. That contract is enforced,
not merely documented. `eval_utils.normalize_provider_profile` rejects an
`api_key_env` that is not an environment-variable name and any credential-shaped
`extra_body` key; `run_config_to_hydra_yaml()` re-runs the same fail-closed
check (`eval_utils.assert_no_credential_shaped_values`) before writing a single
byte, since exported `conf/` groups are committed to the repository; and
`resolved_config_yaml()` additionally masks anything key-shaped before the run's
resolved composition is written to `data/processed/logs/<run_id>.resolved.yaml`.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

try:
    import eval_utils as eu
    import run_provenance as rp
except ModuleNotFoundError:  # pragma: no cover
    from scripts import eval_utils as eu, run_provenance as rp


MASK_VALUE = "***MASKED***"
CONFIG_GROUPS = ("profile", "dataset", "variant", "sampling", "logging", "experiment")

# Masking and the fail-closed export check share one notion of "credential
# shaped" (`eval_utils.is_credential_shaped_key`): `max_tokens` is a request
# knob, `api_key_env` names an environment variable, and neither is a secret.
_is_secret_key = eu.is_credential_shaped_key


def _container(node: Any) -> Any:
    if isinstance(node, (DictConfig, ListConfig)):
        return OmegaConf.to_container(node, resolve=True)
    return node


def mask_secrets(value: Any) -> Any:
    """Recursively replace credential-shaped values with a masking token."""
    if isinstance(value, Mapping):
        return {
            key: MASK_VALUE
            if _is_secret_key(key) and val not in (None, "")
            else mask_secrets(val)
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [mask_secrets(item) for item in value]
    return value


# =============================================================================
# Hydra composition -> run config dict
# =============================================================================


def hydra_config_to_run_config(cfg: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    """Convert a composed Hydra config into a `normalize_run_config` input.

    The returned dictionary has exactly the shape of `run_configs/*.json`, so
    the Hydra path and the JSON path converge before any pipeline code runs.
    """
    raw = _container(cfg) if isinstance(cfg, (DictConfig, ListConfig)) else dict(cfg)
    profile = dict(raw["profile"])
    model = raw.get("model")
    if model:
        # `model=` selects the single model to run for the chosen profile. A
        # model the profile does not list is a typo, not a silent new model:
        # fail exactly the way `eval_utils.filter_run_profiles` does.
        listed = [str(entry) for entry in profile.get("models", [])]
        if str(model) not in listed:
            raise ValueError(
                "No provider profiles match the requested filters. "
                f"Model {str(model)!r} is not listed by profile {str(profile.get('profile_id'))!r} "
                f"(models: {', '.join(listed) or 'none'})."
            )
        profile["models"] = [str(model)]
    sampling = raw.get("sampling") or {}
    dataset = raw.get("dataset") or {}
    variant = raw.get("variant") or {}
    run_config: dict[str, Any] = {
        "run_group_id": raw["run_group_id"],
        "datasets": [dataset["id"] if isinstance(dataset, Mapping) else dataset],
        "benchmark_variants": [
            variant["id"] if isinstance(variant, Mapping) else variant
        ],
        "tasks": eu.normalize_task_filter(raw.get("task") or "both"),
        "prompt_version": raw["prompt_version"],
        "seed": raw["seed"],
        "batch_order": raw["batch_order"],
        # Older exported presets predate the knob; missing means bare.
        "item_context": raw.get("item_context") or eu.DEFAULT_ITEM_CONTEXT,
        "deterministic": dict(sampling.get("deterministic", {})),
        "stochastic": dict(sampling.get("stochastic", {})),
        "logging": dict(raw.get("logging") or {}),
        "profiles": [profile],
        "acse_embedding_backend": raw.get("acse_embedding_backend"),
        "acse_embedding_mlx_model": raw.get("acse_embedding_mlx_model"),
    }
    return run_config


def resolved_config_yaml(cfg: DictConfig) -> str:
    """Fully resolved composition as YAML text, with credentials masked."""
    resolved = OmegaConf.to_container(cfg, resolve=True)
    return OmegaConf.to_yaml(OmegaConf.create(mask_secrets(resolved)))


def run_args_namespace(
    cfg: DictConfig | Mapping[str, Any], resolved_yaml: str = ""
) -> SimpleNamespace:
    """Build the argparse-compatible namespace the runners consume.

    Field names and semantics match the two runner CLIs one-for-one, so the
    Hydra path exercises the same code with the same defaults.
    """
    raw = _container(cfg) if isinstance(cfg, (DictConfig, ListConfig)) else dict(cfg)
    model = raw.get("model")
    task = str(raw.get("task") or "both")
    logging_cfg = dict(raw.get("logging") or {})
    dataset = raw.get("dataset") or {}
    variant = raw.get("variant") or {}
    return SimpleNamespace(
        config=None,
        profile=str(raw["profile"]["profile_id"]),
        model=str(model) if model else None,
        all_models=not bool(model),
        dataset=dataset["id"] if isinstance(dataset, Mapping) else dataset,
        variant=variant["id"] if isinstance(variant, Mapping) else variant,
        task=None if task in {"both", "task3"} else task,
        mode=str(raw.get("mode") or "smoke"),
        run_id=raw.get("run_id") or None,
        smoke_items=int(raw.get("smoke_items", 2)),
        fake_completion=bool(raw.get("fake_completion", False)),
        dry_run=bool(raw.get("dry_run", False)),
        log_level=str(raw.get("log_level") or "INFO"),
        # Task 3 only.
        source_run_id=raw.get("source_run_id") or None,
        audit_mode=str(raw.get("audit_mode") or eu.OFFICIAL_TASK3_AUDIT_MODE),
        allow_partial_source=bool(raw.get("allow_partial_source", False)),
        allow_source_profile_mismatch=bool(
            raw.get("allow_source_profile_mismatch", False)
        ),
        # Logging overrides are already merged into run_config["logging"]; the
        # runners still read these argparse-style attributes.
        progress_every_records=logging_cfg.get("progress_every_records"),
        progress_every_seconds=logging_cfg.get("progress_every_seconds"),
        warn_after_records=logging_cfg.get("warn_after_records"),
        warn_parse_failure_rate=logging_cfg.get("warn_parse_failure_rate"),
        warn_request_error_rate=logging_cfg.get("warn_request_error_rate"),
        no_progress_artifacts=not (
            bool(logging_cfg.get("write_progress_csv", True))
            or bool(logging_cfg.get("write_event_jsonl", True))
        ),
        no_request_transcripts=not bool(
            logging_cfg.get("write_request_transcripts", True)
        ),
        # Provenance: written per run id by the runners.
        resolved_config_yaml=resolved_yaml,
        resolved_config_sha=rp.sha256_text(resolved_yaml) if resolved_yaml else "",
    )


# =============================================================================
# JSON run config -> Hydra config groups
# =============================================================================


def _yaml_text(payload: Mapping[str, Any], header: str) -> str:
    body = OmegaConf.to_yaml(OmegaConf.create(json.loads(json.dumps(payload))))
    return f"{header}\n{body}" if header else body


def _write(path: Path, text: str, overwrite: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite and path.read_text(encoding="utf-8") != text:
        raise FileExistsError(
            f"{path} already exists with different content; pass overwrite=True to replace it."
        )
    path.write_text(text, encoding="utf-8")
    return path


def run_config_to_hydra_yaml(
    path: str | Path,
    output_dir: str | Path | None = None,
    *,
    name: str | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Export a JSON run config into `conf/` groups and return the written paths.

    One `conf/profile/<profile_id>.yaml` per provider profile, plus
    `conf/sampling/<name>.yaml`, `conf/logging/<name>.yaml` and a
    `conf/experiment/<name>.yaml` preset that ties them together and sweeps the
    config's profile / dataset / variant matrix. Each profile then runs only
    its own model list. Composing
    `+experiment=<name>` then reproduces the JSON config cell by cell.
    """
    source = Path(path)
    config = eu.load_run_config(source)
    # Exported groups are committed alongside the paper artifacts, so this runs
    # before the first `_write`: a rejected config leaves nothing on disk.
    eu.assert_no_credential_shaped_values(
        config, where=f"exported run config {source.name}"
    )
    out = Path(output_dir) if output_dir is not None else eu.project_root() / "conf"
    preset = eu.safe_identifier(name or source.stem)
    written: dict[str, Path] = {}

    for profile in config["profiles"]:
        profile_id = str(profile["profile_id"])
        payload = {key: value for key, value in profile.items() if value is not None}
        header = (
            f"# Provider profile: {profile_id}\n"
            f"# Exported from {source.name} by scripts/hydra_bridge.py.\n"
            "# `api_key_env` is the NAME of the environment variable holding the key.\n"
        )
        written[f"profile/{profile_id}"] = _write(
            out / "profile" / f"{profile_id}.yaml",
            _yaml_text(payload, header),
            overwrite,
        )

    written["sampling"] = _write(
        out / "sampling" / f"{preset}.yaml",
        _yaml_text(
            {
                "deterministic": config["deterministic"],
                "stochastic": config["stochastic"],
            },
            f"# Sampling blocks exported from {source.name}.\n",
        ),
        overwrite,
    )
    written["logging"] = _write(
        out / "logging" / f"{preset}.yaml",
        _yaml_text(
            dict(config["logging"]),
            f"# Logging thresholds exported from {source.name}.\n",
        ),
        overwrite,
    )

    profile_ids = [str(profile["profile_id"]) for profile in config["profiles"]]
    experiment = {
        "defaults": [
            {"override /profile": str(config["profiles"][0]["profile_id"])},
            {"override /dataset": config["datasets"][0]},
            {"override /variant": config["benchmark_variants"][0]},
            {"override /sampling": preset},
            {"override /logging": preset},
        ],
        "run_group_id": config["run_group_id"],
        "prompt_version": config["prompt_version"],
        "seed": config["seed"],
        "batch_order": config["batch_order"],
        "item_context": config.get("item_context", eu.DEFAULT_ITEM_CONTEXT),
        # Override the global embedding group with the JSON config's exact
        # values, including null for legacy configs that did not select one.
        "acse_embedding_backend": config["acse_embedding_backend"],
        "acse_embedding_mlx_model": config["acse_embedding_mlx_model"],
        "task": "both" if config["tasks"] == ["task1", "task2"] else config["tasks"][0],
        "mode": "full",
        # Each swept profile owns its model list. Leaving model unset runs those
        # models sequentially and cannot create invalid cross-profile pairs.
        "model": None,
        "hydra": {
            "mode": "MULTIRUN",
            "sweeper": {
                "params": {
                    "profile": ",".join(profile_ids),
                    "dataset": ",".join(config["datasets"]),
                    "variant": ",".join(config["benchmark_variants"]),
                }
            },
        },
    }
    header = (
        "# @package _global_\n"
        f"# Exported from {source.name} by scripts/hydra_bridge.py.\n"
        f"# Reproduce the JSON matrix with:\n"
        f"#   .venv/bin/python scripts/run.py --multirun +experiment={preset}\n"
        "# Profiles are swept while `model: null` runs only the models owned by\n"
        "# each selected profile, preserving every profile/model pairing.\n"
    )
    written["experiment"] = _write(
        out / "experiment" / f"{preset}.yaml", _yaml_text(experiment, header), overwrite
    )
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export a JSON run config into Hydra config groups under conf/."
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="Path to a run_configs/*.json file."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Config root to write into (default: conf/).",
    )
    parser.add_argument(
        "--name", default=None, help="Preset name (default: the JSON file stem)."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files with different content.",
    )
    args = parser.parse_args(argv)
    written = run_config_to_hydra_yaml(
        args.config, args.output_dir, name=args.name, overwrite=args.overwrite
    )
    for key in sorted(written):
        print(f"{key}: {written[key]}")


if __name__ == "__main__":
    main()
