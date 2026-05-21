from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib
import math
import os
import random
import re
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

_CACHE_DIR = Path(os.environ.get("RE_UQ_CACHE_DIR", Path(__file__).resolve().parents[1] / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib
import numpy as np
import pandas as pd
import requests
from openai import OpenAI
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score as sklearn_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODALITIES = ["mandatory", "recommended", "optional", "nice_to_have"]
TASK3_RELATIONS = ["preserves", "strengthens", "weakens", "content_changed"]
MONOTONICITY_TOLERANCE = 0.05
WEAK_MODALITY_PROBE_TEMPLATES = [
    {
        "template_id": "useful_if",
        "source_template": "It would be useful if the system could {capability}.",
    },
    {
        "template_id": "nice_if",
        "source_template": "It would be nice if the system could {capability}.",
    },
    {
        "template_id": "low_priority_enhancement",
        "source_template": "As a low-priority enhancement, the system could {capability}.",
    },
    {
        "template_id": "future_enhancement",
        "source_template": "Stakeholders mentioned that the system could {capability} as a possible future enhancement.",
    },
]
WEAK_MODALITY_PROBE_FIELDS = [
    "item_id",
    "seed_id",
    "template_id",
    "source_modality",
    "source_statement",
    "task2_gold_modality",
    "capability_text",
]
WEAK_MODALITY_SANITY_FIELDS = [
    "template_id",
    "source_statement_template",
    "example_source_statement",
    "intended_gold_modality",
    "weaker_than_should",
    "reviewer",
    "review_note",
]
WEAK_MODALITY_CONSTRUCT_REVIEW_FIELDS = [
    "reviewer_id",
    "reviewer_role",
    "template_id",
    "source_statement_template",
    "example_source_statement",
    "weaker_than_should",
    "ordinal_rank",
    "review_note",
]
WEAK_MODALITY_PROBE_SUMMARY_FIELDS = [
    "model",
    "run_id",
    "template_id",
    "sample_kind",
    "n",
    "valid_n",
    "parse_success_rate",
    "accuracy",
    "to_recommended_rate",
    "over_commitment",
    "high_conf_overcommit_80",
    "high_conf_overcommit_90",
    "pred_mandatory_rate",
    "pred_recommended_rate",
    "pred_optional_rate",
    "pred_nice_to_have_rate",
    "mean_confidence",
]
TASK3_VERIFICATION_FIELDS = [
    "item_id",
    "source_item_id",
    "seed_id",
    "source_dataset",
    "original_requirement",
    "capability_text",
    "source_modality",
    "source_statement",
    "task2_run_id",
    "task2_model",
    "task2_requirement",
    "task2_modality",
    "task2_confidence",
    "task3_gold_relation",
    "ordinal_strength",
    "numeric_strength",
]
ORDINAL_STRENGTH = {
    "mandatory": 3,
    "recommended": 2,
    "optional": 1,
    "nice_to_have": 0,
}
NUMERIC_STRENGTH_DEFAULT = {
    "mandatory": 1.00,
    "recommended": 0.67,
    "optional": 0.33,
    "nice_to_have": 0.00,
}
NUMERIC_STRENGTH_RECOMMENDED_075 = {
    "mandatory": 1.00,
    "recommended": 0.75,
    "optional": 0.33,
    "nice_to_have": 0.00,
}
HIGH_CONFIDENCE_THRESHOLDS = (0.80, 0.90)
RULE_BASELINE_MODEL = "rule_based_baseline"
RULE_BASELINE_METHOD = "deterministic_rules"
ENSEMBLE_MODEL_PREFIX = "ensemble"
LOGPROB_PROBE_PROMPT = 'Return exactly this JSON object: {"decision":"yes","confidence":100,"brief_reason":"probe"}'
DATASET_NICE = "nice"
DATASET_MLM_TAPT = "mlm_tapt"
DATASET_IDS = {DATASET_NICE, DATASET_MLM_TAPT}
DATASET_SUFFIXES = {
    DATASET_NICE: "",
    DATASET_MLM_TAPT: "_mlm_tapt",
}
SOURCE_DATASET_LABELS = {
    DATASET_NICE: "NICE",
    DATASET_MLM_TAPT: "mlm_tapt",
}
BASE_SEED_REVIEW_FIELDS = [
    "seed_id",
    "source_dataset",
    "source_corpus",
    "original_requirement",
    "capability_text_auto",
    "auto_include",
    "auto_exclusion_reason",
    "include",
    "exclusion_reason",
    "capability_text_final",
]
REQUIREMENT_CUE_RE = re.compile(
    r"\b(shall|must|should|may|required|recommended|optional|permitted|will|able\s+to)\b",
    re.I,
)
CAPABILITY_MODAL_RE = re.compile(
    r"\b(shall|must|should|may|will|can|could|able\s+to|capable(?:\s+(?:of|to))?|possible\s+to|possibility\s+to)\b",
    re.I,
)
TABLE_FIGURE_RE = re.compile(r"\b(table|figure|fig\.|annex|section|clause)\s+[A-Za-z0-9.-]+", re.I)
LIST_MARKER_RE = re.compile(r"(^|\s)(\d+\.|\([a-z]\)|[a-z]\)|[ivx]+\.|[-*]\s|[•·]|\t)", re.I)
SYMBOL_HEAVY_RE = re.compile(r"([=<>±×µ%]|\b0x[0-9a-f]+\b)", re.I)

DEFAULT_CONFIG: dict[str, Any] = {
    "project": {
        "seed": 20260518,
        "target_seed_count": 180,
        "pilot_seed_count": 20,
        "prompt_version": "v1",
    },
    "llm": {
        "host": "http://localhost:8000/v1",
        "models": ["local-model"],
        "api_key_env": "LOCAL_OPENAI_API_KEY",
        "timeout_s": 120,
        "max_tokens": 256,
        "concurrency": 4,
        "deterministic": {"temperature": 0.0, "top_p": 1.0, "samples": 1},
        "stochastic": {"temperature": 0.7, "top_p": 1.0, "samples": 5},
    },
    "datasets": {
        "nice_url": "https://zenodo.org/records/14590935/files/PROMISE-relabeled-NICE.csv?download=1",
        "nice_local_path": "data/raw/PROMISE-relabeled-NICE.csv",
        "mlm_tapt_repo": "limsc/mlm-tapt-requirements",
        "mlm_tapt_config": "default",
        "mlm_tapt_splits": ["train", "val"],
        "mlm_tapt_target_seed_count": 180,
        "mlm_tapt_exclude_source_regex": "_PURE$",
    },
}

RUN_REGISTRY_FIELDS = [
    "run_id",
    "run_group_id",
    "provider_id",
    "profile_id",
    "model",
    "dataset_id",
    "benchmark_variant",
    "tasks",
    "status",
    "expected_records",
    "observed_records",
    "parse_success_rate",
    "deterministic_item_coverage",
    "stochastic_complete_item_rate",
    "started_at_utc",
    "finished_at_utc",
    "base_url",
    "api_key_env",
    "concurrency",
    "batch_size",
    "expected_api_calls",
    "observed_api_calls",
    "timeout_s",
    "json_mode",
    "structured_output",
    "request_extra_body",
    "server_model_probe",
    "notes",
]

DEFAULT_RUN_LOGGING: dict[str, Any] = {
    "progress_every_records": 100,
    "progress_every_seconds": 60,
    "warn_after_records": 100,
    "warn_parse_failure_rate": 0.02,
    "warn_request_error_rate": 0.02,
    "write_progress_csv": True,
    "write_event_jsonl": True,
}
STRUCTURED_OUTPUT_MODES = {"none", "json_object", "json_schema"}

RUN_PROGRESS_FIELDS = [
    "run_id",
    "model",
    "task",
    "benchmark_items",
    "expected_records",
    "observed_records",
    "record_completion_rate",
    "parse_success_rate",
    "deterministic_records",
    "deterministic_ok",
    "deterministic_item_coverage",
    "stochastic_records",
    "stochastic_ok",
    "stochastic_item_coverage",
    "stochastic_complete_item_rate",
]


def project_root() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "AGENTS.md").exists() and (candidate / "docs" / "evaluation.md").exists():
            return candidate
    return Path(__file__).resolve().parents[1]


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path = "config.example.json") -> dict[str, Any]:
    root = project_root()
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = root / config_path
    if not config_path.exists():
        return DEFAULT_CONFIG
    with config_path.open("r", encoding="utf-8") as handle:
        return deep_update(DEFAULT_CONFIG, json.load(handle))


def positive_int(value: Any, name: str) -> int:
    try:
        resolved = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if resolved < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return resolved


def nonnegative_int(value: Any, name: str) -> int:
    try:
        resolved = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}") from exc
    if resolved < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return resolved


def nonnegative_float(value: Any, name: str) -> float:
    try:
        resolved = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative number, got {value!r}") from exc
    if resolved < 0:
        raise ValueError(f"{name} must be a non-negative number, got {value!r}")
    return resolved


def bool_config(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def resolve_llm_concurrency(
    config: Mapping[str, Any],
    env: Mapping[str, str] | None = None,
    env_var: str = "LLM_CONCURRENCY",
) -> int:
    env = os.environ if env is None else env
    env_value = str(env.get(env_var, "")).strip()
    if env_value:
        return positive_int(env_value, env_var)

    llm_config = config.get("llm", {}) if isinstance(config, Mapping) else {}
    configured_value = (
        llm_config.get("concurrency", DEFAULT_CONFIG["llm"]["concurrency"])
        if isinstance(llm_config, Mapping)
        else DEFAULT_CONFIG["llm"]["concurrency"]
    )
    return positive_int(configured_value, "llm.concurrency")


def ensure_project_dirs(root: Path | None = None) -> None:
    root = root or project_root()
    for rel in [
        "data/raw",
        "data/processed",
        "notebooks",
        "outputs",
        "prompts",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def read_csv_frame(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    return read_csv_frame(path).to_dict(orient="records")


def _csv_frame(rows: list[dict[str, Any]] | pd.DataFrame, fieldnames: list[str] | None = None) -> pd.DataFrame:
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame.from_records(rows)
    if fieldnames is not None:
        for field in fieldnames:
            if field not in frame.columns:
                frame[field] = ""
        frame = frame.loc[:, fieldnames]
    return frame


def write_csv_rows(
    path: str | Path,
    rows: list[dict[str, Any]] | pd.DataFrame,
    fieldnames: list[str] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _csv_frame(rows, fieldnames=fieldnames).to_csv(path, index=False)


def write_csv_rows_if_changed(
    path: str | Path,
    rows: list[dict[str, Any]] | pd.DataFrame,
    fieldnames: list[str] | None = None,
    candidate_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = _csv_frame(rows, fieldnames=fieldnames)
    candidate_text = frame.to_csv(index=False)
    existed_before = path.exists()

    if overwrite or not existed_before:
        path.write_text(candidate_text, encoding="utf-8")
        return {"status": "overwritten" if existed_before else "written", "path": path, "candidate_path": ""}

    existing_text = path.read_text(encoding="utf-8")
    if existing_text == candidate_text:
        return {"status": "unchanged", "path": path, "candidate_path": ""}

    candidate = Path(candidate_path) if candidate_path is not None else path.with_name(f"{path.stem}_candidate{path.suffix}")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(candidate_text, encoding="utf-8")
    return {"status": "candidate_written", "path": path, "candidate_path": candidate}


def normalize_dataset_id(dataset_id: str | None) -> str:
    normalized = str(dataset_id or DATASET_NICE).strip().lower().replace("-", "_")
    if normalized in {"", "main", "default", DATASET_NICE}:
        return DATASET_NICE
    if normalized in {DATASET_MLM_TAPT, "hf", "hf_requirements", "limsc_mlm_tapt"}:
        return DATASET_MLM_TAPT
    raise ValueError(f"Unknown dataset_id: {dataset_id}")


def dataset_suffix(dataset_id: str | None) -> str:
    return DATASET_SUFFIXES[normalize_dataset_id(dataset_id)]


def variant_suffix(variant: str | None) -> str:
    normalized = str(variant or "must").strip().lower()
    if normalized in {"", "must", "main"}:
        return ""
    if normalized == "shall":
        return "_shall"
    raise ValueError(f"Unknown benchmark variant: {variant}")


def variant_path(path: str | Path, variant: str | None) -> Path:
    path = Path(path)
    suffix = variant_suffix(variant)
    if not suffix:
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def dataset_variant_suffix(dataset_id: str | None, variant: str | None = None) -> str:
    return f"{dataset_suffix(dataset_id)}{variant_suffix(variant)}"


def artifact_path(path: str | Path, dataset_id: str | None = None, variant: str | None = None) -> Path:
    path = Path(path)
    suffix = dataset_variant_suffix(dataset_id, variant)
    if not suffix:
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def candidate_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(f"{path.stem}_candidate{path.suffix}")


def auto_candidates_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(f"{path.stem}_candidates_auto{path.suffix}")


def dataset_target_seed_count(config: Mapping[str, Any], dataset_id: str | None) -> int:
    dataset_id = normalize_dataset_id(dataset_id)
    project_config = config.get("project", {}) if isinstance(config, Mapping) else {}
    datasets_config = config.get("datasets", {}) if isinstance(config, Mapping) else {}
    default_count = project_config.get("target_seed_count", DEFAULT_CONFIG["project"]["target_seed_count"])
    if dataset_id == DATASET_MLM_TAPT and isinstance(datasets_config, Mapping):
        return positive_int(datasets_config.get("mlm_tapt_target_seed_count", default_count), "datasets.mlm_tapt_target_seed_count")
    return positive_int(default_count, "project.target_seed_count")


def seed_review_fields(dataset_id: str | None = None) -> list[str]:
    dataset_id = normalize_dataset_id(dataset_id)
    if dataset_id == DATASET_NICE:
        return [field for field in BASE_SEED_REVIEW_FIELDS if field != "source_corpus"]
    return list(BASE_SEED_REVIEW_FIELDS)


def normalize_task_filter(task: str | Iterable[str] | None = None) -> list[str]:
    if task is None:
        return ["task1", "task2"]
    if isinstance(task, str):
        values = [task]
    else:
        values = [str(value) for value in task]
    resolved: list[str] = []
    for value in values:
        normalized = str(value).strip().lower()
        if normalized in {"", "both", "all"}:
            candidates = ["task1", "task2"]
        elif normalized in {"task1", "mandatory_entailment"}:
            candidates = ["task1"]
        elif normalized in {"task2", "modality_extraction"}:
            candidates = ["task2"]
        else:
            raise ValueError(f"Unknown task filter: {value}")
        for candidate in candidates:
            if candidate not in resolved:
                resolved.append(candidate)
    return resolved or ["task1", "task2"]


def normalize_benchmark_variant(variant: str | None) -> str:
    normalized = str(variant or "must").strip().lower()
    if normalized in {"", "main", "must"}:
        return "must"
    if normalized == "shall":
        return "shall"
    raise ValueError(f"Unknown benchmark variant: {variant}")


def normalize_base_url(value: Any) -> str:
    return str(value or "").strip().rstrip("/")


def _string_list(value: Any, name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",")]
    elif isinstance(value, Iterable):
        values = [str(part).strip() for part in value]
    else:
        raise ValueError(f"{name} must be a string or list of strings.")
    return [value for value in values if value]


def normalize_run_logging_config(
    logging_config: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = dict(DEFAULT_RUN_LOGGING)
    if logging_config:
        config.update(dict(logging_config))
    if overrides:
        config.update({key: value for key, value in overrides.items() if value is not None})

    return {
        "progress_every_records": positive_int(config["progress_every_records"], "logging.progress_every_records"),
        "progress_every_seconds": nonnegative_int(config["progress_every_seconds"], "logging.progress_every_seconds"),
        "warn_after_records": nonnegative_int(config["warn_after_records"], "logging.warn_after_records"),
        "warn_parse_failure_rate": nonnegative_float(config["warn_parse_failure_rate"], "logging.warn_parse_failure_rate"),
        "warn_request_error_rate": nonnegative_float(config["warn_request_error_rate"], "logging.warn_request_error_rate"),
        "write_progress_csv": bool_config(config["write_progress_csv"], "logging.write_progress_csv"),
        "write_event_jsonl": bool_config(config["write_event_jsonl"], "logging.write_event_jsonl"),
    }


def normalize_structured_output_mode(value: Any, *, json_mode: bool = False, json_schema: bool = False) -> str:
    if value is None:
        if json_schema:
            return "json_schema"
        return "json_object" if json_mode else "none"
    if isinstance(value, bool):
        return "json_object" if value else "none"
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "": "none",
        "false": "none",
        "off": "none",
        "none": "none",
        "no": "none",
        "true": "json_object",
        "on": "json_object",
        "json": "json_object",
        "json_mode": "json_object",
        "json_object": "json_object",
        "schema": "json_schema",
        "strict": "json_schema",
        "strict_json": "json_schema",
        "strict_json_schema": "json_schema",
        "json_schema": "json_schema",
    }
    if normalized not in aliases:
        raise ValueError(f"Unknown structured output mode: {value}")
    return aliases[normalized]


def normalize_provider_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    provider_id = str(profile.get("provider_id") or profile.get("provider") or "").strip()
    profile_id = str(profile.get("profile_id") or profile.get("id") or provider_id).strip()
    base_url = normalize_base_url(profile.get("base_url") or profile.get("host"))
    models = _string_list(profile.get("models"), f"profiles.{profile_id}.models")
    if not provider_id:
        raise ValueError("Provider profile is missing provider_id.")
    if not profile_id:
        raise ValueError(f"Provider profile for {provider_id} is missing profile_id.")
    if not base_url:
        raise ValueError(f"Provider profile {profile_id} is missing base_url.")
    if not models:
        raise ValueError(f"Provider profile {profile_id} must list at least one model.")
    extra_body = profile.get("extra_body") or {}
    if not isinstance(extra_body, Mapping):
        raise ValueError(f"Provider profile {profile_id} extra_body must be an object.")
    json_mode = bool(profile.get("json_mode", False))
    structured_output = normalize_structured_output_mode(
        profile.get("structured_output"),
        json_mode=json_mode,
        json_schema=bool(profile.get("json_schema", False)),
    )
    json_mode = json_mode or structured_output in {"json_object", "json_schema"}
    response_format = profile.get("response_format")
    if response_format is None and structured_output == "json_object" and "response_format" not in extra_body:
        response_format = {"type": "json_object"}
    if response_format is not None and not isinstance(response_format, Mapping):
        raise ValueError(f"Provider profile {profile_id} response_format must be an object.")
    return {
        "profile_id": profile_id,
        "provider_id": provider_id,
        "base_url": base_url,
        "api_key_env": str(profile.get("api_key_env") or "LOCAL_OPENAI_API_KEY").strip(),
        "models": models,
        "concurrency": positive_int(profile.get("concurrency", DEFAULT_CONFIG["llm"]["concurrency"]), f"profiles.{profile_id}.concurrency"),
        "batch_size": positive_int(profile.get("batch_size", 1), f"profiles.{profile_id}.batch_size"),
        "timeout_s": positive_int(profile.get("timeout_s", DEFAULT_CONFIG["llm"]["timeout_s"]), f"profiles.{profile_id}.timeout_s"),
        "max_tokens": positive_int(profile.get("max_tokens", DEFAULT_CONFIG["llm"]["max_tokens"]), f"profiles.{profile_id}.max_tokens"),
        "json_mode": json_mode,
        "structured_output": structured_output,
        "response_format": dict(response_format) if response_format is not None else None,
        "extra_body": dict(extra_body),
        "requires_manual_server": bool(profile.get("requires_manual_server", False)),
        "notes": str(profile.get("notes", "")).strip(),
    }


def normalize_run_config(config: Mapping[str, Any]) -> dict[str, Any]:
    run_group_id = str(config.get("run_group_id") or "").strip()
    if not run_group_id:
        raise ValueError("Run config must define run_group_id.")
    profiles_value = config.get("profiles")
    if not isinstance(profiles_value, list) or not profiles_value:
        raise ValueError("Run config must define a non-empty profiles list.")
    project_config = config.get("project", {}) if isinstance(config.get("project"), Mapping) else {}
    llm_config = config.get("llm", {}) if isinstance(config.get("llm"), Mapping) else {}
    deterministic = config.get("deterministic") or llm_config.get("deterministic") or DEFAULT_CONFIG["llm"]["deterministic"]
    stochastic = config.get("stochastic") or llm_config.get("stochastic") or DEFAULT_CONFIG["llm"]["stochastic"]
    datasets = [normalize_dataset_id(value) for value in _string_list(config.get("datasets", [DATASET_NICE]), "datasets")]
    variants = [normalize_benchmark_variant(value) for value in _string_list(config.get("benchmark_variants", ["must"]), "benchmark_variants")]
    tasks = normalize_task_filter(config.get("tasks", ["task1", "task2"]))
    return {
        "run_group_id": run_group_id,
        "datasets": datasets,
        "benchmark_variants": variants,
        "tasks": tasks,
        "prompt_version": str(config.get("prompt_version") or project_config.get("prompt_version") or DEFAULT_CONFIG["project"]["prompt_version"]),
        "deterministic": {
            "temperature": float(deterministic.get("temperature", DEFAULT_CONFIG["llm"]["deterministic"]["temperature"])),
            "top_p": float(deterministic.get("top_p", DEFAULT_CONFIG["llm"]["deterministic"]["top_p"])),
            "samples": positive_int(deterministic.get("samples", 1), "deterministic.samples"),
        },
        "stochastic": {
            "temperature": float(stochastic.get("temperature", DEFAULT_CONFIG["llm"]["stochastic"]["temperature"])),
            "top_p": float(stochastic.get("top_p", DEFAULT_CONFIG["llm"]["stochastic"]["top_p"])),
            "samples": max(0, int(stochastic.get("samples", DEFAULT_CONFIG["llm"]["stochastic"]["samples"]))),
        },
        "logging": normalize_run_logging_config(config.get("logging") if isinstance(config.get("logging"), Mapping) else None),
        "profiles": [normalize_provider_profile(profile) for profile in profiles_value],
    }


def strip_json_trailing_commas(text: str) -> str:
    output: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue
        if char == ",":
            lookahead = index + 1
            while lookahead < len(text) and text[lookahead] in " \t\r\n":
                lookahead += 1
            if lookahead < len(text) and text[lookahead] in "]}":
                index += 1
                continue
        output.append(char)
        index += 1
    return "".join(output)


def load_run_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    try:
        config = json.loads(text)
    except json.JSONDecodeError as first_error:
        normalized_text = strip_json_trailing_commas(text)
        if normalized_text == text:
            raise ValueError(
                f"Invalid JSON in run config {config_path}: line {first_error.lineno} "
                f"column {first_error.colno}: {first_error.msg}"
            ) from first_error
        try:
            config = json.loads(normalized_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in run config {config_path}: line {exc.lineno} "
                f"column {exc.colno}: {exc.msg}"
            ) from exc
    return normalize_run_config(config)


def filter_run_profiles(
    run_config: Mapping[str, Any],
    profile_id: str | None = None,
    model: str | None = None,
) -> list[dict[str, Any]]:
    profiles = [dict(profile) for profile in run_config.get("profiles", [])]
    if profile_id:
        requested = str(profile_id).strip()
        profiles = [profile for profile in profiles if profile["profile_id"] == requested or profile["provider_id"] == requested]
    if model:
        requested_model = str(model).strip()
        selected: list[dict[str, Any]] = []
        for profile in profiles:
            if requested_model in profile["models"]:
                profile = dict(profile)
                profile["models"] = [requested_model]
                selected.append(profile)
        profiles = selected
    if not profiles:
        raise ValueError("No provider profiles match the requested filters.")
    return profiles


def validate_manual_server_profile(profile: Mapping[str, Any]) -> None:
    if profile.get("requires_manual_server") and len(profile.get("models", [])) != 1:
        raise ValueError(
            f"Profile {profile.get('profile_id')} requires a manually restarted server; "
            "select exactly one model with --model."
        )


def run_registry_path(root: str | Path, dataset_id: str | None = None, variant: str | None = None) -> Path:
    return artifact_path(Path(root) / "data/processed/run_registry.csv", dataset_id, variant)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_metadata(path: str | Path, root: str | Path | None = None) -> dict[str, Any]:
    path = Path(path)
    root_path = Path(root).resolve() if root is not None else None
    resolved = path.resolve()
    relative = str(resolved.relative_to(root_path)) if root_path and resolved.is_relative_to(root_path) else str(path)
    metadata: dict[str, Any] = {
        "path": relative,
        "exists": path.exists(),
    }
    if not path.exists():
        metadata.update({"sha256": "", "bytes": 0, "rows": ""})
        return metadata
    metadata["sha256"] = sha256_file(path)
    metadata["bytes"] = path.stat().st_size
    metadata["rows"] = len(read_csv_frame(path)) if path.suffix.lower() == ".csv" else ""
    return metadata


def write_benchmark_manifest(
    paths: list[str | Path],
    output_path: str | Path,
    root: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": metadata or {},
        "artifacts": [artifact_metadata(path, root=root) for path in paths],
    }
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def latest_run_id(raw_rows: list[dict[str, Any]], prefix: str | None = None) -> str | None:
    selected: list[str] = []
    for row in raw_rows:
        run_id = str(row.get("run_id", "")).strip()
        if not run_id:
            continue
        if not run_id_matches_prefix(run_id, prefix):
            continue
        if not selected or selected[-1] != run_id:
            selected.append(run_id)
    return selected[-1] if selected else None


def select_run_rows(
    raw_rows: list[dict[str, Any]],
    run_id: str | None = None,
    prefix: str | None = None,
) -> tuple[str | None, list[dict[str, Any]]]:
    selected_run_id = str(run_id).strip() if run_id else latest_run_id(raw_rows, prefix=prefix)
    if not selected_run_id:
        return None, []
    if not run_id_matches_prefix(selected_run_id, prefix):
        return selected_run_id, []
    return selected_run_id, [row for row in raw_rows if row.get("run_id") == selected_run_id]


def run_id_matches_prefix(run_id: Any, prefix: str | None = None) -> bool:
    if not prefix:
        return True
    candidate = str(run_id or "").strip()
    normalized = str(prefix).strip().strip("-")
    if not candidate:
        return False
    if not normalized:
        return True
    marker = f"{normalized}-"
    if not candidate.startswith(marker):
        return False
    remainder = candidate[len(marker) :]
    # Benchmark variants are encoded inside the run prefix, e.g. full-shall-*.
    # A request for full-* should not also match full-shall-*.
    return remainder.split("-", 1)[0] not in {"shall"}


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def compact_json(value: Any) -> str:
    if value in (None, "", {}, []):
        return ""
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def download_file(url: str, dest: str | Path, timeout_s: int = 120) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=timeout_s, headers={"User-Agent": "re-uq-evaluation/0.1"})
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


SPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
NEGATION_RE = re.compile(r"\b(no|not|never|none|without|cannot|can't|won't|mustn't|shouldn't)\b", re.I)
FORMULA_RE = re.compile(r"(<=|>=|==|!=|[<>]|%|\b\d+\s*[*/+-]\s*\d+\b)")
SENTENCE_END_RE = re.compile(r"[.!?]+")
OUTER_QUOTES_RE = re.compile(r"^[\"'“”‘’]+|[\"'“”‘’]+$")
STRANDED_PREPOSITION_RE = re.compile(
    r"^(?:with|to|from|for|of|in|on|at|by|about|into|onto|through|across|under|over|between|among)\b",
    re.I,
)


def normalize_space(text: str) -> str:
    return SPACE_RE.sub(" ", (text or "").strip())


def unwrap_outer_quotes(text: str) -> str:
    previous = normalize_space(text)
    while previous:
        current = OUTER_QUOTES_RE.sub("", previous).strip()
        if current == previous:
            return current
        previous = current
    return previous


def strip_final_punctuation(text: str) -> str:
    return unwrap_outer_quotes(normalize_space(text)).rstrip(" .;:'\"“”‘’")


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def lower_initial(text: str) -> str:
    text = strip_final_punctuation(text)
    if not text:
        return text
    if len(text) > 1 and text[:2].isupper():
        return text
    return text[0].lower() + text[1:]


LEADING_REQUIREMENT_PATTERNS = [
    re.compile(
        r"^(?:the\s+)?(?:system|software|application|app|product|platform|service|tool|interface|data|table)\s+"
        r"(?:shall|must|should|may|will|can|could)\s+(?:be\s+able\s+to\s+)?",
        re.I,
    ),
    re.compile(
        r"^(?:the\s+)?(?:system|software|application|app|product|platform|service|tool|interface|data|table)\s+"
        r"(?:is|are)\s+(?:required|expected|recommended)\s+to\s+",
        re.I,
    ),
    re.compile(
        r"^(?:users?|administrators?|admins?|customers?|stakeholders?)\s+"
        r"(?:shall|must|should|may|will|can|could)\s+(?:be\s+able\s+to\s+)?",
        re.I,
    ),
]
GENERIC_MODAL_PREFIX_RE = re.compile(
    r"^(?:the\s+)?[A-Za-z0-9][^.;:!?]{0,120}?\s+"
    r"(?:shall|must|should|may|will|can|could)\s+(?:be\s+able\s+to\s+)?",
    re.I,
)


def auto_capability_text(requirement: str) -> str:
    text = strip_final_punctuation(requirement)
    text = re.sub(r"^[\-\*\d.)\s]+", "", text)
    stripped_leading_requirement = False
    for pattern in LEADING_REQUIREMENT_PATTERNS:
        stripped = pattern.sub("", text).strip()
        if stripped != text:
            text = stripped
            stripped_leading_requirement = True
            break
    if not stripped_leading_requirement:
        text = GENERIC_MODAL_PREFIX_RE.sub("", text).strip()
    text = re.sub(r"^(?:the\s+)?(?:system|software|application|app|product|platform|service|tool|data|table)\s+", "", text, flags=re.I)
    text = re.sub(r"^(?:shall|must|should|may|will|can|could)\s+(?:be\s+able\s+to\s+)?", "", text, flags=re.I)
    text = re.sub(r"^(?:be\s+able\s+to|able\s+to)\s+", "", text, flags=re.I)
    return lower_initial(text)


def automatic_filter(requirement: str, capability: str) -> tuple[bool, str]:
    reasons: list[str] = []
    wc = word_count(requirement)
    cleaned_requirement = strip_final_punctuation(requirement)
    cleaned_capability = strip_final_punctuation(capability)
    if wc < 5:
        reasons.append("too_short")
    if wc > 35:
        reasons.append("too_long")
    if SENTENCE_END_RE.search(cleaned_requirement) or SENTENCE_END_RE.search(cleaned_capability):
        reasons.append("multi_sentence")
    if NEGATION_RE.search(requirement):
        reasons.append("negation")
    if FORMULA_RE.search(requirement):
        reasons.append("formula_or_symbol")
    heavy_conjunctions = sum(requirement.lower().count(token) for token in [" and ", " or ", ";"])
    if heavy_conjunctions > 1:
        reasons.append("possibly_multiple_capabilities")
    if not capability or word_count(capability) < 2:
        reasons.append("empty_or_too_short_capability")
    if CAPABILITY_MODAL_RE.search(capability):
        reasons.append("residual_modal_in_capability")
    if STRANDED_PREPOSITION_RE.search(cleaned_capability):
        reasons.append("stranded_preposition")
    return not reasons, ";".join(reasons)


def mlm_tapt_filter(requirement: str, capability: str, source_corpus: str = "", exclude_source_regex: str = "_PURE$") -> tuple[bool, str]:
    auto_include, reason = automatic_filter(requirement, capability)
    reasons = [item for item in reason.split(";") if item]
    if exclude_source_regex and re.search(exclude_source_regex, source_corpus or ""):
        reasons.append("excluded_source")
    if not REQUIREMENT_CUE_RE.search(requirement):
        reasons.append("no_requirement_cue")
    if TABLE_FIGURE_RE.search(requirement):
        reasons.append("table_or_figure_reference")
    if ":" in requirement:
        reasons.append("colon_structure")
    if LIST_MARKER_RE.search(requirement):
        reasons.append("list_or_heading_marker")
    if "NOTE" in requirement.upper():
        reasons.append("note_text")
    if SYMBOL_HEAVY_RE.search(requirement):
        reasons.append("symbol_heavy")
    if CAPABILITY_MODAL_RE.search(capability):
        reasons.append("residual_modal_in_capability")
    unique_reasons = list(dict.fromkeys(reasons))
    return auto_include and not unique_reasons, ";".join(unique_reasons)


def find_requirement_text_column(rows: list[dict[str, str]]) -> str:
    if not rows:
        raise ValueError("No rows found in the source dataset.")
    columns = list(rows[0].keys())
    for preferred in ["RequirementText", "requirementtext", "requirement", "text", "Requirement"]:
        for column in columns:
            if column.lower() == preferred.lower():
                return column
    for column in columns:
        if "requirement" in column.lower() or column.lower() in {"text", "sentence"}:
            return column
    raise ValueError(f"Could not find a requirement text column. Columns: {columns}")


def make_seed_candidates(
    dataset_rows: list[dict[str, str]],
    target_count: int = DEFAULT_CONFIG["project"]["target_seed_count"],
    source_dataset: str = "NICE",
    text_column: str | None = None,
    source_corpus_field: str | None = None,
) -> list[dict[str, Any]]:
    text_column = text_column or find_requirement_text_column(dataset_rows)
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    accepted = 0
    for row in dataset_rows:
        original = normalize_space(row.get(text_column, ""))
        if not original:
            continue
        dedupe_key = original.lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        capability = auto_capability_text(original)
        auto_include, reason = automatic_filter(original, capability)
        include = auto_include and accepted < target_count
        if include:
            accepted += 1
        candidate = {
            "seed_id": f"S{len(candidates) + 1:04d}",
            "source_dataset": source_dataset,
            "original_requirement": original,
            "capability_text_auto": capability,
            "auto_include": "yes" if auto_include else "no",
            "auto_exclusion_reason": reason,
            "include": "yes" if include else "no",
            "exclusion_reason": "" if include else reason,
            "capability_text_final": capability if include else "",
        }
        if source_corpus_field is not None:
            candidate["source_corpus"] = normalize_space(row.get(source_corpus_field, ""))
        candidates.append(candidate)
    return candidates


def load_mlm_tapt_rows(config: Mapping[str, Any]) -> list[dict[str, str]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install the 'datasets' dependency to load limsc/mlm-tapt-requirements.") from exc

    datasets_config = config.get("datasets", {}) if isinstance(config, Mapping) else {}
    repo = str(datasets_config.get("mlm_tapt_repo", DEFAULT_CONFIG["datasets"]["mlm_tapt_repo"]))
    config_name = str(datasets_config.get("mlm_tapt_config", DEFAULT_CONFIG["datasets"]["mlm_tapt_config"]))
    splits = datasets_config.get("mlm_tapt_splits", DEFAULT_CONFIG["datasets"]["mlm_tapt_splits"])
    if isinstance(splits, str):
        splits = [splits]
    rows: list[dict[str, str]] = []
    for split in splits:
        dataset = load_dataset(repo, config_name, split=str(split))
        for row in dataset:
            rows.append(
                {
                    "source": str(row.get("source", "")),
                    "reqs": str(row.get("reqs", "")),
                    "split": str(split),
                }
            )
    return rows


def weighted_sample_candidate_indices(
    candidates: list[dict[str, Any]],
    target_count: int,
    seed: int,
    source_cap: int = 30,
    source_field: str = "source_corpus",
) -> list[int]:
    eligible_indices = [index for index, row in enumerate(candidates) if is_truthy(row.get("auto_include", ""))]
    if len(eligible_indices) < target_count:
        raise ValueError(f"Expected at least {target_count} eligible candidates, found {len(eligible_indices)}.")

    source_counts = Counter(str(candidates[index].get(source_field, "") or "unknown") for index in eligible_indices)
    rng = random.Random(seed)
    remaining = set(eligible_indices)
    selected: list[int] = []
    selected_by_source: Counter[str] = Counter()

    while len(selected) < target_count:
        allowed = [
            index
            for index in sorted(remaining)
            if selected_by_source[str(candidates[index].get(source_field, "") or "unknown")] < source_cap
        ]
        if not allowed:
            raise ValueError(
                f"Could not sample {target_count} candidates with source_cap={source_cap}; selected {len(selected)}."
            )
        weights = [1.0 / source_counts[str(candidates[index].get(source_field, "") or "unknown")] for index in allowed]
        chosen = rng.choices(allowed, weights=weights, k=1)[0]
        selected.append(chosen)
        remaining.remove(chosen)
        selected_by_source[str(candidates[chosen].get(source_field, "") or "unknown")] += 1
    return selected


def make_mlm_tapt_seed_candidates(
    dataset_rows: list[dict[str, str]],
    target_count: int = DEFAULT_CONFIG["datasets"]["mlm_tapt_target_seed_count"],
    seed: int = DEFAULT_CONFIG["project"]["seed"],
    exclude_source_regex: str = "_PURE$",
    source_cap: int = 30,
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for row in dataset_rows:
        source_corpus = normalize_space(row.get("source", ""))
        original = normalize_space(row.get("reqs", ""))
        if not original:
            continue
        dedupe_key = original.lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        capability = auto_capability_text(original)
        auto_include, reason = mlm_tapt_filter(
            original,
            capability,
            source_corpus=source_corpus,
            exclude_source_regex=exclude_source_regex,
        )
        candidates.append(
            {
                "seed_id": f"S{len(candidates) + 1:04d}",
                "source_dataset": SOURCE_DATASET_LABELS[DATASET_MLM_TAPT],
                "source_corpus": source_corpus,
                "original_requirement": original,
                "capability_text_auto": capability,
                "auto_include": "yes" if auto_include else "no",
                "auto_exclusion_reason": reason,
                "include": "no",
                "exclusion_reason": reason,
                "capability_text_final": "",
            }
        )

    selected_indices = set(weighted_sample_candidate_indices(candidates, target_count, seed=seed, source_cap=source_cap))
    for index, candidate in enumerate(candidates):
        if index in selected_indices:
            candidate["include"] = "yes"
            candidate["exclusion_reason"] = ""
            candidate["capability_text_final"] = candidate["capability_text_auto"]
        elif is_truthy(candidate["auto_include"]):
            candidate["exclusion_reason"] = "not_sampled_weighted_pool"
    return candidates


def _review_compare_text(value: Any) -> str:
    return strip_final_punctuation(str(value)).lower()


def refresh_capability_suggestions(rows: list[dict[str, Any]], force: bool = False) -> tuple[list[dict[str, Any]], int]:
    refreshed: list[dict[str, Any]] = []
    updated = 0
    for row in rows:
        row = dict(row)
        original = row.get("original_requirement", "")
        old_auto = row.get("capability_text_auto", "")
        old_final = row.get("capability_text_final", "")
        new_auto = auto_capability_text(original)
        row["capability_text_auto"] = new_auto

        final_is_unedited = _review_compare_text(old_final) in {
            "",
            _review_compare_text(old_auto),
            _review_compare_text(original),
        }
        if is_truthy(row.get("include", "")) and new_auto and (force or final_is_unedited):
            if row.get("capability_text_final", "") != new_auto:
                updated += 1
            row["capability_text_final"] = new_auto
        refreshed.append(row)
    return refreshed, updated


def refresh_capability_suggestions_file(path: str | Path, force: bool = False) -> int:
    rows = read_csv_rows(path)
    refreshed, updated = refresh_capability_suggestions(rows, force=force)
    write_csv_rows(path, refreshed, fieldnames=list(read_csv_frame(path).columns))
    return updated


def is_truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "include"}


def load_reviewed_seeds(
    path: str | Path,
    target_count: int = DEFAULT_CONFIG["project"]["target_seed_count"],
    strict: bool = True,
) -> list[dict[str, Any]]:
    rows = read_csv_rows(path)
    selected: list[dict[str, Any]] = []
    for row in rows:
        if not is_truthy(row.get("include", "")):
            continue
        capability = strip_final_punctuation(row.get("capability_text_final") or row.get("capability_text_auto") or "")
        if not capability:
            continue
        row = dict(row)
        row["capability_text_final"] = capability
        selected.append(row)
    if strict and len(selected) != target_count:
        raise ValueError(f"Expected exactly {target_count} included seeds, found {len(selected)}.")
    return selected


def included_capability_review_frame(path: str | Path) -> pd.DataFrame:
    frame = read_csv_frame(path)
    for column in ["seed_id", "original_requirement", "capability_text_final", "include"]:
        if column not in frame.columns:
            raise ValueError(f"Missing required review column: {column}")
    included = frame[frame["include"].map(is_truthy)].copy()
    included["capability_text_final"] = included["capability_text_final"].map(strip_final_punctuation)
    columns = ["seed_id"]
    if "source_corpus" in included.columns:
        columns.append("source_corpus")
    columns.extend(["original_requirement", "capability_text_final"])
    return included.loc[:, columns].rename(
        columns={
            "seed_id": "Seed",
            "source_corpus": "Source corpus",
            "original_requirement": "Original requirement",
            "capability_text_final": "Final capability text",
        }
    )


def write_included_capability_review(path: str | Path, output_dir: str | Path, suffix: str = "") -> dict[str, Path]:
    frame = included_capability_review_frame(path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / f"included_capabilities_review{suffix}.md"
    csv_path = output_dir / f"included_capabilities_review{suffix}.csv"
    markdown_path.write_text(frame.to_markdown(index=False) + "\n", encoding="utf-8")
    frame.to_csv(csv_path, index=False)
    return {"markdown": markdown_path, "csv": csv_path}


def capability_clause(capability: str) -> str:
    return strip_final_punctuation(capability)


def source_statement(capability: str, modality: str, mandatory_keyword: str = "MUST") -> str:
    cap = capability_clause(capability)
    if modality == "mandatory":
        return f"The system {mandatory_keyword.upper()} {cap}."
    if modality == "recommended":
        return f"The system SHOULD {cap}."
    if modality == "optional":
        return f"The system MAY {cap}."
    if modality == "nice_to_have":
        return f"It would be useful if the system could {cap}."
    raise ValueError(f"Unknown modality: {modality}")


def weak_modality_template_by_id(template_id: str) -> dict[str, str]:
    for template in WEAK_MODALITY_PROBE_TEMPLATES:
        if template["template_id"] == template_id:
            return template
    raise ValueError(f"Unknown weak-modality probe template: {template_id}")


def weak_modality_probe_source_statement(capability: str, template_id: str) -> str:
    template = weak_modality_template_by_id(template_id)
    return template["source_template"].format(capability=capability_clause(capability))


def build_weak_modality_probe_items(
    seed_rows: list[dict[str, Any]],
    templates: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    templates = templates or WEAK_MODALITY_PROBE_TEMPLATES
    items: list[dict[str, Any]] = []
    for seed in seed_rows:
        capability = seed["capability_text_final"]
        for template in templates:
            template_id = template["template_id"]
            source = template["source_template"].format(capability=capability_clause(capability))
            items.append(
                {
                    "item_id": f"{seed['seed_id']}_weak_{template_id}",
                    "seed_id": seed["seed_id"],
                    "template_id": template_id,
                    "source_modality": "nice_to_have",
                    "source_statement": source,
                    "task2_gold_modality": "nice_to_have",
                    "capability_text": capability,
                    "source_dataset": seed.get("source_dataset", "NICE"),
                    "source_corpus": seed.get("source_corpus", ""),
                    "original_requirement": seed.get("original_requirement", ""),
                }
            )
    return items


def build_task3_verification_items(
    benchmark_rows: list[dict[str, Any]],
    task2_raw_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    benchmark_by_item = {row["item_id"]: row for row in benchmark_rows}
    task2_raw_rows = filter_raw_rows_to_current_benchmark(benchmark_rows, task2_raw_rows)
    items: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in task2_raw_rows:
        if raw.get("task") != "task2" or raw.get("sample_kind") != "deterministic":
            continue
        if raw.get("parse_status") != "ok" or not isinstance(raw.get("parsed_json"), dict):
            continue
        source_item = benchmark_by_item.get(str(raw.get("item_id", "")))
        if not source_item:
            continue
        parsed = raw["parsed_json"]
        extracted_modality = normalize_modality(parsed.get("modality"))
        if extracted_modality is None:
            continue
        model = str(raw.get("model", ""))
        source_item_id = str(source_item["item_id"])
        dedupe_key = (model, source_item_id)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        relation = task3_gold_relation(source_item["source_modality"], extracted_modality)
        confidence = parse_confidence(parsed.get("confidence"))
        items.append(
            {
                "item_id": f"{source_item_id}__task3__{safe_identifier(model)}",
                "source_item_id": source_item_id,
                "seed_id": source_item["seed_id"],
                "source_dataset": source_item.get("source_dataset", "NICE"),
                "original_requirement": source_item.get("original_requirement", ""),
                "capability_text": source_item.get("capability_text", ""),
                "source_modality": source_item["source_modality"],
                "source_statement": source_item["source_statement"],
                "task2_run_id": raw.get("run_id", ""),
                "task2_model": model,
                "task2_requirement": str(parsed.get("requirement", "")),
                "task2_modality": extracted_modality,
                "task2_confidence": "" if confidence is None else confidence,
                "task3_gold_relation": relation,
                "ordinal_strength": int(source_item["ordinal_strength"]),
                "numeric_strength": float(source_item["numeric_strength"]),
            }
        )
    return items


def weak_modality_template_sanity_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for template in WEAK_MODALITY_PROBE_TEMPLATES:
        rows.append(
            {
                "template_id": template["template_id"],
                "source_statement_template": template["source_template"],
                "example_source_statement": template["source_template"].format(capability="export reports"),
                "intended_gold_modality": "nice_to_have",
                "weaker_than_should": "",
                "reviewer": "",
                "review_note": "",
            }
        )
    return rows


def write_weak_modality_template_sanity_check(output_dir: str | Path, suffix: str = "") -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"weak_modality_template_sanity_check{suffix}.csv"
    markdown_path = output_dir / f"weak_modality_template_sanity_check{suffix}.md"
    if csv_path.exists():
        rows = read_csv_rows(csv_path)
    else:
        rows = weak_modality_template_sanity_rows()
        write_csv_rows(csv_path, rows, fieldnames=WEAK_MODALITY_SANITY_FIELDS)
    markdown_path.write_text(markdown_table(rows, WEAK_MODALITY_SANITY_FIELDS) + "\n", encoding="utf-8")
    return {"csv": csv_path, "markdown": markdown_path}


def _sanity_answer(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1"}:
        return "yes"
    if text in {"no", "n", "false", "0"}:
        return "no"
    return ""


def weak_modality_sanity_status(rows: list[dict[str, Any]]) -> dict[str, Any]:
    required = [template["template_id"] for template in WEAK_MODALITY_PROBE_TEMPLATES]
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=WEAK_MODALITY_SANITY_FIELDS)
    for field in WEAK_MODALITY_SANITY_FIELDS:
        if field not in frame.columns:
            frame[field] = ""

    missing: list[str] = []
    incomplete: list[str] = []
    disagreeing: list[str] = []
    for template_id in required:
        template_rows = frame[frame["template_id"].astype(str) == template_id]
        if template_rows.empty:
            missing.append(template_id)
            continue
        answers = [_sanity_answer(value) for value in template_rows["weaker_than_should"].tolist()]
        if "no" in answers:
            disagreeing.append(template_id)
        if not answers or any(answer != "yes" for answer in answers):
            incomplete.append(template_id)

    valid = not missing and not incomplete and not disagreeing
    return {
        "valid": valid,
        "missing_template_ids": missing,
        "incomplete_template_ids": incomplete,
        "disagreeing_template_ids": disagreeing,
    }


def require_weak_modality_sanity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status = weak_modality_sanity_status(rows)
    if not status["valid"]:
        raise ValueError(
            "Weak-modality sanity check is incomplete or disagrees with the construct: "
            f"missing={status['missing_template_ids']}, "
            f"incomplete={status['incomplete_template_ids']}, "
            f"disagreeing={status['disagreeing_template_ids']}"
        )
    return status


def weak_modality_construct_review_rows(reviewer_ids: Iterable[str] = ("R1", "R2")) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for reviewer_id in reviewer_ids:
        for template in WEAK_MODALITY_PROBE_TEMPLATES:
            rows.append(
                {
                    "reviewer_id": reviewer_id,
                    "reviewer_role": "",
                    "template_id": template["template_id"],
                    "source_statement_template": template["source_template"],
                    "example_source_statement": template["source_template"].format(capability="export reports"),
                    "weaker_than_should": "",
                    "ordinal_rank": "",
                    "review_note": "",
                }
            )
    return rows


def weak_modality_construct_review_status(
    rows: list[dict[str, Any]],
    expected_reviewers_per_template: int = 2,
) -> dict[str, Any]:
    required = [template["template_id"] for template in WEAK_MODALITY_PROBE_TEMPLATES]
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=WEAK_MODALITY_CONSTRUCT_REVIEW_FIELDS)
    for field in WEAK_MODALITY_CONSTRUCT_REVIEW_FIELDS:
        if field not in frame.columns:
            frame[field] = ""

    missing: list[str] = []
    insufficient: list[str] = []
    incomplete: list[str] = []
    disagreeing: list[str] = []
    reviewer_counts: dict[str, int] = {}

    for template_id in required:
        template_rows = frame[frame["template_id"].astype(str) == template_id]
        if template_rows.empty:
            missing.append(template_id)
            reviewer_counts[template_id] = 0
            continue

        reviewer_ids = {
            str(value).strip()
            for value in template_rows["reviewer_id"].tolist()
            if str(value).strip()
        }
        reviewer_counts[template_id] = len(reviewer_ids)
        if len(reviewer_ids) < expected_reviewers_per_template:
            insufficient.append(template_id)

        answers = [_sanity_answer(value) for value in template_rows["weaker_than_should"].tolist()]
        if "no" in answers:
            disagreeing.append(template_id)
        if len(answers) < expected_reviewers_per_template or any(answer != "yes" for answer in answers):
            incomplete.append(template_id)

    valid = not missing and not insufficient and not incomplete and not disagreeing
    return {
        "valid": valid,
        "expected_reviewers_per_template": expected_reviewers_per_template,
        "reviewer_counts": reviewer_counts,
        "missing_template_ids": missing,
        "insufficient_template_ids": insufficient,
        "incomplete_template_ids": incomplete,
        "disagreeing_template_ids": disagreeing,
    }


def candidate_requirement(capability: str, mandatory_keyword: str = "MUST") -> str:
    return f"The system {mandatory_keyword.upper()} {capability_clause(capability)}."


def build_benchmark_items(
    seed_rows: list[dict[str, Any]],
    numeric_strength: dict[str, float] | None = None,
    mandatory_keyword: str = "MUST",
) -> list[dict[str, Any]]:
    numeric_strength = numeric_strength or NUMERIC_STRENGTH_DEFAULT
    mandatory_keyword = mandatory_keyword.upper()
    if mandatory_keyword not in {"MUST", "SHALL"}:
        raise ValueError(f"Unsupported mandatory keyword: {mandatory_keyword}")
    items: list[dict[str, Any]] = []
    for seed in seed_rows:
        capability = seed["capability_text_final"]
        for modality in MODALITIES:
            items.append(
                {
                    "item_id": f"{seed['seed_id']}_{modality}",
                    "seed_id": seed["seed_id"],
                    "source_dataset": seed.get("source_dataset", "NICE"),
                    "source_corpus": seed.get("source_corpus", ""),
                    "original_requirement": seed.get("original_requirement", ""),
                    "capability_text": capability,
                    "source_modality": modality,
                    "source_statement": source_statement(capability, modality, mandatory_keyword=mandatory_keyword),
                    "candidate_requirement": candidate_requirement(capability, mandatory_keyword=mandatory_keyword),
                    "mandatory_keyword": mandatory_keyword,
                    "task1_gold_decision": "yes" if modality == "mandatory" else "no",
                    "task1_gold_yes": 1 if modality == "mandatory" else 0,
                    "task2_gold_modality": modality,
                    "ordinal_strength": ORDINAL_STRENGTH[modality],
                    "numeric_strength": numeric_strength[modality],
                }
            )
    return items


def benchmark_statement_review_frame(items: list[dict[str, Any]] | pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(items, (str, Path)):
        frame = read_csv_frame(items)
    elif isinstance(items, pd.DataFrame):
        frame = items.copy()
    else:
        frame = pd.DataFrame.from_records(items)
    required = ["seed_id", "capability_text", "candidate_requirement", "source_modality", "source_statement"]
    for column in required:
        if column not in frame.columns:
            raise ValueError(f"Missing required benchmark column: {column}")
    review = frame.pivot(
        index=["seed_id", "capability_text", "candidate_requirement"],
        columns="source_modality",
        values="source_statement",
    ).reset_index()
    for modality in MODALITIES:
        if modality not in review.columns:
            review[modality] = ""
    return review[
        [
            "seed_id",
            "capability_text",
            "mandatory",
            "recommended",
            "optional",
            "nice_to_have",
            "candidate_requirement",
        ]
    ].rename(
        columns={
            "seed_id": "Seed",
            "capability_text": "Capability",
            "mandatory": "MUST source",
            "recommended": "SHOULD source",
            "optional": "MAY source",
            "nice_to_have": "Nice-to-have source",
            "candidate_requirement": "Mandatory candidate",
        }
    )


def write_benchmark_statement_review(
    items: list[dict[str, Any]] | pd.DataFrame | str | Path,
    output_dir: str | Path,
    suffix: str = "",
) -> dict[str, Path]:
    frame = benchmark_statement_review_frame(items)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / f"benchmark_statements_review{suffix}.md"
    csv_path = output_dir / f"benchmark_statements_review{suffix}.csv"
    markdown_path.write_text(frame.to_markdown(index=False) + "\n", encoding="utf-8")
    frame.to_csv(csv_path, index=False)
    return {"markdown": markdown_path, "csv": csv_path}


def load_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def render_prompt(template: str, **values: Any) -> str:
    return template.format(**values)


def prompt_for_benchmark_task(
    task: str,
    item: Mapping[str, Any],
    task1_template: str,
    task2_template: str,
) -> str:
    if task == "task1":
        return render_prompt(
            task1_template,
            source_statement=item["source_statement"],
            candidate_requirement=item["candidate_requirement"],
        )
    if task == "task2":
        return render_prompt(task2_template, source_statement=item["source_statement"])
    raise ValueError(f"Unsupported benchmark task: {task}")


def _json_schema_object(properties: Mapping[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
    }


def task_response_schema(task: str, *, batched: bool = False) -> dict[str, Any]:
    confidence = {"type": "number", "minimum": 0, "maximum": 100}
    if task == "task1":
        properties: dict[str, Any] = {
            "decision": {"type": "string", "enum": ["yes", "no"]},
            "confidence": confidence,
            "brief_reason": {"type": "string", "maxLength": 200},
        }
        required = ["decision", "confidence", "brief_reason"]
    elif task == "task2":
        properties = {
            "requirement": {"type": "string"},
            "modality": {"type": "string", "enum": ["mandatory", "recommended", "optional", "nice_to_have"]},
            "confidence": confidence,
        }
        required = ["requirement", "modality", "confidence"]
    elif task == "task3":
        properties = {
            "relation": {"type": "string", "enum": ["preserves", "strengthens", "weakens", "content_changed"]},
            "confidence": confidence,
            "evidence_phrase": {"type": "string", "maxLength": 240},
            "brief_reason": {"type": "string", "maxLength": 240},
        }
        required = ["relation", "confidence", "evidence_phrase", "brief_reason"]
    else:
        raise ValueError(f"Unsupported task for JSON Schema response format: {task}")

    if batched:
        properties = {"request_index": {"type": "integer"}, **properties}
        required = ["request_index", *required]
        return _json_schema_object(
            {"results": {"type": "array", "items": _json_schema_object(properties, required)}},
            ["results"],
        )
    return _json_schema_object(properties, required)


def response_format_for_task(task: str, structured_output: str, *, batched: bool = False) -> dict[str, Any] | None:
    mode = normalize_structured_output_mode(structured_output)
    if mode == "none":
        return None
    if mode == "json_object":
        return {"type": "json_object"}
    schema_name = f"re_uq_{task}_{'batch' if batched else 'single'}"
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": task_response_schema(task, batched=batched),
        },
    }


def resolve_response_format_args(
    task: str,
    *,
    structured_output: Any = None,
    json_mode: bool = False,
    response_format: Mapping[str, Any] | None = None,
    extra_body: Mapping[str, Any] | None = None,
    batched: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    extra = dict(extra_body) if extra_body else None
    mode = normalize_structured_output_mode(structured_output, json_mode=json_mode)
    if mode == "none":
        return (dict(response_format) if response_format else None), extra

    resolved = response_format_for_task(task, mode, batched=batched)
    if extra is not None and "response_format" in extra:
        extra["response_format"] = resolved
        return None, extra
    return resolved, extra


def completion_request_job(
    *,
    item: Mapping[str, Any],
    task: str,
    model: str,
    host: str,
    run_id: str,
    sample_kind: str,
    sample_index: int,
    temperature: float,
    top_p: float,
    prompt: str,
    prompt_version: str,
    max_tokens: int,
    timeout_s: int,
    api_key_env: str,
    request_index: int,
    provider_id: str = "",
    profile_id: str = "",
    run_group_id: str = "",
    json_mode: bool = False,
    structured_output: str | None = None,
    response_format: Mapping[str, Any] | None = None,
    extra_body: Mapping[str, Any] | None = None,
    server_model_probe: Mapping[str, Any] | str | None = None,
) -> dict[str, Any]:
    resolved_response_format, resolved_extra_body = resolve_response_format_args(
        task,
        structured_output=structured_output,
        json_mode=json_mode,
        response_format=response_format,
        extra_body=extra_body,
        batched=False,
    )
    return {
        "request_index": request_index,
        "run_id": run_id,
        "model": model,
        "host": host,
        "base_url": host,
        "task": task,
        "item": dict(item),
        "sample_index": int(sample_index),
        "sample_kind": sample_kind,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "prompt_version": prompt_version,
        "prompt": prompt,
        "max_tokens": int(max_tokens),
        "timeout_s": int(timeout_s),
        "api_key_env": api_key_env,
        "provider_id": provider_id,
        "profile_id": profile_id,
        "run_group_id": run_group_id,
        "json_mode": bool(json_mode),
        "structured_output": normalize_structured_output_mode(structured_output, json_mode=json_mode),
        "response_format": dict(resolved_response_format) if resolved_response_format else None,
        "extra_body": dict(resolved_extra_body) if resolved_extra_body else None,
        "server_model_probe": server_model_probe,
    }


def planned_completion_jobs(
    benchmark_rows: list[dict[str, Any]],
    *,
    tasks: Iterable[str],
    model: str,
    host: str,
    run_id: str,
    prompt_version: str,
    task1_template: str,
    task2_template: str,
    deterministic: Mapping[str, Any],
    stochastic: Mapping[str, Any],
    max_tokens: int,
    timeout_s: int,
    api_key_env: str,
    provider_id: str = "",
    profile_id: str = "",
    run_group_id: str = "",
    json_mode: bool = False,
    structured_output: str | None = None,
    response_format: Mapping[str, Any] | None = None,
    extra_body: Mapping[str, Any] | None = None,
    server_model_probe: Mapping[str, Any] | str | None = None,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    task_list = normalize_task_filter(tasks)
    stochastic_samples = max(0, int(stochastic.get("samples", 0)))
    for item in benchmark_rows:
        for task in task_list:
            prompt = prompt_for_benchmark_task(task, item, task1_template, task2_template)
            jobs.append(
                completion_request_job(
                    item=item,
                    task=task,
                    model=model,
                    host=host,
                    run_id=run_id,
                    sample_kind="deterministic",
                    sample_index=0,
                    temperature=float(deterministic["temperature"]),
                    top_p=float(deterministic["top_p"]),
                    prompt=prompt,
                    prompt_version=prompt_version,
                    max_tokens=max_tokens,
                    timeout_s=timeout_s,
                    api_key_env=api_key_env,
                    request_index=len(jobs),
                    provider_id=provider_id,
                    profile_id=profile_id,
                    run_group_id=run_group_id,
                    json_mode=json_mode,
                    structured_output=structured_output,
                    response_format=response_format,
                    extra_body=extra_body,
                    server_model_probe=server_model_probe,
                )
            )
            for sample_index in range(stochastic_samples):
                jobs.append(
                    completion_request_job(
                        item=item,
                        task=task,
                        model=model,
                        host=host,
                        run_id=run_id,
                        sample_kind="stochastic",
                        sample_index=sample_index,
                        temperature=float(stochastic["temperature"]),
                        top_p=float(stochastic["top_p"]),
                        prompt=prompt,
                        prompt_version=prompt_version,
                        max_tokens=max_tokens,
                        timeout_s=timeout_s,
                        api_key_env=api_key_env,
                        request_index=len(jobs),
                        provider_id=provider_id,
                        profile_id=profile_id,
                        run_group_id=run_group_id,
                        json_mode=json_mode,
                        structured_output=structured_output,
                        response_format=response_format,
                        extra_body=extra_body,
                        server_model_probe=server_model_probe,
                    )
                )
    return jobs


def extract_json_object(text: str) -> str | None:
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_json_value(text: str) -> Any | None:
    if not text:
        return None
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            return value
        except json.JSONDecodeError:
            continue
    return None


def parse_confidence(value: Any) -> float | None:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= confidence <= 100.0:
        return confidence
    return None


def normalize_decision(value: Any) -> str | None:
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1"}:
        return "yes"
    if text in {"no", "n", "false", "0"}:
        return "no"
    return None


def normalize_modality(value: Any) -> str | None:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "must": "mandatory",
        "shall": "mandatory",
        "required": "mandatory",
        "mandatory": "mandatory",
        "should": "recommended",
        "recommended": "recommended",
        "recommendation": "recommended",
        "may": "optional",
        "can": "optional",
        "optional": "optional",
        "nice_to_have": "nice_to_have",
        "nice": "nice_to_have",
        "wish": "nice_to_have",
    }
    return aliases.get(text)


def normalize_relation(value: Any) -> str | None:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "preserve": "preserves",
        "preserved": "preserves",
        "preserves": "preserves",
        "same": "preserves",
        "same_modality": "preserves",
        "faithful": "preserves",
        "strengthen": "strengthens",
        "strengthened": "strengthens",
        "strengthens": "strengthens",
        "stronger": "strengthens",
        "upgrades": "strengthens",
        "upgrade": "strengthens",
        "overcommit": "strengthens",
        "over_commitment": "strengthens",
        "weaken": "weakens",
        "weakened": "weakens",
        "weakens": "weakens",
        "weaker": "weakens",
        "downgrade": "weakens",
        "downgrades": "weakens",
        "undercommit": "weakens",
        "under_commitment": "weakens",
        "content_changed": "content_changed",
        "content_change": "content_changed",
        "changed_content": "content_changed",
        "content_mismatch": "content_changed",
        "different_content": "content_changed",
        "functionality_changed": "content_changed",
    }
    return aliases.get(text)


def task3_gold_relation(source_modality: str, extracted_modality: str) -> str:
    source = normalize_modality(source_modality)
    extracted = normalize_modality(extracted_modality)
    if source is None:
        raise ValueError(f"Invalid source modality: {source_modality!r}")
    if extracted is None:
        raise ValueError(f"Invalid extracted modality: {extracted_modality!r}")
    source_strength = ORDINAL_STRENGTH[source]
    extracted_strength = ORDINAL_STRENGTH[extracted]
    if extracted_strength > source_strength:
        return "strengthens"
    if extracted_strength < source_strength:
        return "weakens"
    return "preserves"


def safe_identifier(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_").lower()
    return text or "value"


def evidence_phrase_in_source(evidence_phrase: Any, source_statement_text: Any) -> bool:
    evidence = normalize_space(str(evidence_phrase or "")).lower()
    source = normalize_space(str(source_statement_text or "")).lower()
    return bool(evidence and evidence in source)


def raw_record_matches_benchmark_item(raw: Mapping[str, Any], item: Mapping[str, Any]) -> bool:
    prompt = raw.get("prompt")
    if not prompt:
        return True
    source = normalize_space(str(item.get("source_statement", "")))
    if not source:
        return True
    return source in normalize_space(str(prompt))


def filter_raw_rows_to_current_benchmark(
    benchmark_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    item_by_id = {str(row["item_id"]): row for row in benchmark_rows}
    filtered: list[dict[str, Any]] = []
    for raw in raw_rows:
        item = item_by_id.get(str(raw.get("item_id", "")))
        if item and raw_record_matches_benchmark_item(raw, item):
            filtered.append(raw)
    return filtered


def benchmark_rows_with_current_raw_outputs(
    benchmark_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not raw_rows:
        return list(benchmark_rows)
    item_by_id = {str(row["item_id"]): row for row in benchmark_rows}
    fresh_item_ids = {
        str(raw.get("item_id", ""))
        for raw in raw_rows
        if (item := item_by_id.get(str(raw.get("item_id", ""))))
        and raw_record_matches_benchmark_item(raw, item)
    }
    return [row for row in benchmark_rows if str(row.get("item_id", "")) in fresh_item_ids]


def rule_based_source_modality(source_statement_text: str) -> str | None:
    text = normalize_space(source_statement_text).lower()
    if re.match(r"^the system\s+(?:must|shall)\b", text):
        return "mandatory"
    if re.match(r"^the system\s+should\b", text):
        return "recommended"
    if re.match(r"^the system\s+may\b", text):
        return "optional"
    if re.match(r"^it would be useful if the system could\b", text):
        return "nice_to_have"
    return None


def requirement_text_modality(requirement_text: Any) -> str:
    text = normalize_space(str(requirement_text or "")).lower()
    if not text:
        return "unknown"

    weak_patterns = [
        r"\bwould\s+be\s+nice\s+if\b",
        r"\bwould\s+be\s+useful\s+if\b",
        r"\blow[-\s]+priority\s+enhancement\b",
        r"\bfuture\s+enhancement\b",
        r"\bnice[-\s]+to[-\s]+have\b",
        r"\bwishlist\b",
    ]
    if any(re.search(pattern, text) for pattern in weak_patterns):
        return "nice_to_have"
    if re.search(r"\b(?:must|shall)\b", text) or re.search(r"\brequired\s+to\b", text):
        return "mandatory"
    if re.search(r"\b(?:should|recommended)\b", text):
        return "recommended"
    if re.search(r"\b(?:may|optional|could|can)\b", text):
        return "optional"
    if re.match(r"^(?:the\s+)?system\s+\w+", text):
        return "mandatory"
    return "unknown"


def empty_text_modality_fields() -> dict[str, Any]:
    return {
        "text_modality": "",
        "text_modality_parse_status": "",
        "text_modality_correct": "",
        "label_text_consistent": "",
        "text_overcommit": "",
        "text_undercommit": "",
        "text_high_conf_overcommit_80": "",
        "text_high_conf_overcommit_90": "",
    }


def text_modality_fields(
    requirement_text: Any,
    gold_modality: str,
    pred_modality: str,
    confidence: float,
) -> dict[str, Any]:
    text_modality = requirement_text_modality(requirement_text)
    parse_ok = text_modality in MODALITIES
    gold_strength = ORDINAL_STRENGTH.get(str(gold_modality))
    text_strength = ORDINAL_STRENGTH.get(text_modality)
    overcommit = bool(parse_ok and gold_strength is not None and text_strength is not None and text_strength > gold_strength)
    undercommit = bool(parse_ok and gold_strength is not None and text_strength is not None and text_strength < gold_strength)
    return {
        "text_modality": text_modality,
        "text_modality_parse_status": "ok" if parse_ok else "unknown",
        "text_modality_correct": bool(parse_ok and text_modality == gold_modality),
        "label_text_consistent": bool(parse_ok and text_modality == pred_modality),
        "text_overcommit": overcommit,
        "text_undercommit": undercommit,
        "text_high_conf_overcommit_80": bool(overcommit and confidence >= 0.80),
        "text_high_conf_overcommit_90": bool(overcommit and confidence >= 0.90),
    }


def parse_task_response(task: str, raw_text: str) -> tuple[dict[str, Any] | None, str]:
    json_text = extract_json_object(raw_text)
    if json_text is None:
        return None, "invalid_json"
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        return None, "invalid_json"

    confidence = parse_confidence(parsed.get("confidence"))
    if confidence is None:
        return parsed, "invalid_confidence"
    parsed["confidence"] = confidence

    if task == "task1":
        decision = normalize_decision(parsed.get("decision"))
        if decision is None:
            return parsed, "invalid_label"
        parsed["decision"] = decision
        parsed["brief_reason"] = str(parsed.get("brief_reason", ""))[:200]
        return parsed, "ok"

    if task == "task2":
        modality = normalize_modality(parsed.get("modality"))
        if modality is None:
            return parsed, "invalid_label"
        if "requirement" not in parsed:
            return parsed, "missing_fields"
        parsed["modality"] = modality
        parsed["requirement"] = str(parsed.get("requirement", ""))
        return parsed, "ok"

    if task == "task3":
        relation = normalize_relation(parsed.get("relation"))
        if relation is None:
            return parsed, "invalid_label"
        if "evidence_phrase" not in parsed:
            return parsed, "missing_fields"
        parsed["relation"] = relation
        parsed["evidence_phrase"] = str(parsed.get("evidence_phrase", ""))[:240]
        parsed["brief_reason"] = str(parsed.get("brief_reason", ""))[:240]
        return parsed, "ok"

    raise ValueError(f"Unknown task: {task}")


def batch_prompt_for_completion_jobs(jobs: list[Mapping[str, Any]]) -> str:
    if not jobs:
        raise ValueError("Cannot build a batch prompt for an empty job list.")
    task = str(jobs[0]["task"])
    if any(str(job["task"]) != task for job in jobs):
        raise ValueError("Batch prompts must contain jobs for exactly one task.")

    if task == "task1":
        items = [
            {
                "request_index": int(job["request_index"]),
                "source_statement": str(job["item"]["source_statement"]),
                "candidate_requirement": str(job["item"]["candidate_requirement"]),
            }
            for job in jobs
        ]
        return (
            "You are reviewing software requirements.\n"
            "Evaluate each item independently. Do not infer an answer for one item from another item.\n\n"
            'Use "yes" or "no" for decision.\n'
            "Return JSON only as this object:\n"
            '{"results":[{"request_index":0,"decision":"yes","confidence":0,"brief_reason":"<max 12 words>"}]}\n\n'
            "Items:\n"
            f"{json.dumps(items, ensure_ascii=False, indent=2)}"
        )

    if task == "task2":
        items = [
            {
                "request_index": int(job["request_index"]),
                "source_statement": str(job["item"]["source_statement"]),
            }
            for job in jobs
        ]
        return (
            "Extract exactly one requirement from each source statement.\n"
            "Preserve the modality of each source. Evaluate each item independently.\n\n"
            'Use one of: "mandatory", "recommended", "optional", "nice_to_have".\n'
            "Return JSON only as this object:\n"
            '{"results":[{"request_index":0,"requirement":"...","modality":"mandatory","confidence":0}]}\n\n'
            "Items:\n"
            f"{json.dumps(items, ensure_ascii=False, indent=2)}"
        )

    raise ValueError(f"Unsupported benchmark task for batching: {task}")


def parse_batch_completion_results(raw_text: str) -> tuple[dict[int, dict[str, Any]], str]:
    payload = extract_json_value(raw_text)
    if payload is None:
        return {}, "invalid_json"
    if isinstance(payload, dict):
        results = payload.get("results")
    elif isinstance(payload, list):
        results = payload
    else:
        return {}, "invalid_json"
    if not isinstance(results, list):
        return {}, "missing_results"

    parsed: dict[int, dict[str, Any]] = {}
    for position, result in enumerate(results):
        if not isinstance(result, Mapping):
            continue
        if "request_index" in result or "id" in result:
            request_index = result.get("request_index", result.get("id"))
            try:
                request_index_int = int(request_index)
            except (TypeError, ValueError):
                request_index_int = -(position + 1)
        else:
            request_index_int = -(position + 1)
        parsed[request_index_int] = dict(result)
    return parsed, "ok"


def chat_completion(
    host: str,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int = 256,
    timeout_s: int = 120,
    api_key_env: str = "LOCAL_OPENAI_API_KEY",
    logprobs: bool = False,
    top_logprobs: int | None = None,
    response_format: Mapping[str, Any] | None = None,
    extra_body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    api_key = os.getenv(api_key_env, "EMPTY")
    client = OpenAI(base_url=host.rstrip("/") + "/", api_key=api_key, timeout=timeout_s)
    start = time.perf_counter()
    try:
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if logprobs:
            request_kwargs["logprobs"] = True
            if top_logprobs is not None:
                request_kwargs["top_logprobs"] = int(top_logprobs)
        if response_format:
            request_kwargs["response_format"] = dict(response_format)
        if extra_body:
            request_kwargs["extra_body"] = dict(extra_body)
        response = client.chat.completions.create(
            **request_kwargs,
        )
        latency_s = time.perf_counter() - start
        raw_text = response.choices[0].message.content or ""
        return {
            "ok": True,
            "raw_text": raw_text,
            "response_json": response.model_dump(mode="json"),
            "latency_s": latency_s,
            "error": "",
        }
    except Exception as exc:
        return {
            "ok": False,
            "raw_text": "",
            "response_json": None,
            "latency_s": time.perf_counter() - start,
            "error": repr(exc),
        }


def normalize_logprob_tokens(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return []
    tokens: list[dict[str, Any]] = []
    for token_info in content:
        if not isinstance(token_info, dict):
            continue
        token = token_info.get("token", "")
        logprob = token_info.get("logprob")
        try:
            logprob_float = float(logprob)
        except (TypeError, ValueError):
            continue
        tokens.append(
            {
                "token": str(token),
                "logprob": logprob_float,
                "top_logprobs": token_info.get("top_logprobs", []),
            }
        )
    return tokens


def response_logprob_tokens(response_json: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(response_json, dict):
        return []

    choices = response_json.get("choices")
    if isinstance(choices, list) and choices:
        logprobs = choices[0].get("logprobs") if isinstance(choices[0], dict) else None
        if isinstance(logprobs, dict):
            tokens = normalize_logprob_tokens(logprobs.get("content"))
            if tokens:
                return tokens

    response_output = response_json.get("output")
    if isinstance(response_output, list):
        tokens: list[dict[str, Any]] = []
        for output_item in response_output:
            if not isinstance(output_item, dict):
                continue
            tokens.extend(normalize_logprob_tokens(output_item.get("logprobs")))
            content_items = output_item.get("content")
            if isinstance(content_items, list):
                for content_item in content_items:
                    if isinstance(content_item, dict):
                        tokens.extend(normalize_logprob_tokens(content_item.get("logprobs")))
        if tokens:
            return tokens

    return normalize_logprob_tokens(response_json.get("logprobs"))


def completion_has_logprobs(response_json: dict[str, Any] | None) -> bool:
    return bool(response_logprob_tokens(response_json))


def responses_endpoint_url(host: str) -> str:
    base = host.rstrip("/")
    if base.endswith("/responses"):
        return base
    if base.endswith("/v1"):
        return f"{base}/responses"
    return f"{base}/v1/responses"


def responses_output_text(response_json: dict[str, Any] | None) -> str:
    if not isinstance(response_json, dict):
        return ""
    output_text = response_json.get("output_text")
    if isinstance(output_text, str):
        return output_text
    pieces: list[str] = []
    response_output = response_json.get("output")
    if isinstance(response_output, list):
        for output_item in response_output:
            if not isinstance(output_item, dict):
                continue
            text = output_item.get("text")
            if isinstance(text, str):
                pieces.append(text)
            content_items = output_item.get("content")
            if isinstance(content_items, list):
                for content_item in content_items:
                    if isinstance(content_item, dict) and isinstance(content_item.get("text"), str):
                        pieces.append(content_item["text"])
    return "".join(pieces)


def logprob_support_probe(
    host: str,
    model: str,
    api_key_env: str = "LOCAL_OPENAI_API_KEY",
    timeout_s: int = 120,
    max_tokens: int = 48,
) -> dict[str, Any]:
    endpoint = responses_endpoint_url(host)
    api_key = os.getenv(api_key_env, "")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "input": LOGPROB_PROBE_PROMPT,
        "include": ["message.output_text.logprobs"],
        "top_logprobs": 5,
        "max_output_tokens": max_tokens,
    }
    start = time.perf_counter()
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout_s)
        latency_s = time.perf_counter() - start
        try:
            response_json = response.json()
        except ValueError:
            response_json = None
        raw_text = responses_output_text(response_json) or response.text[:500]
        error = "" if response.ok else f"HTTP {response.status_code}: {response.text[:500]}"
    except requests.RequestException as exc:
        latency_s = time.perf_counter() - start
        response_json = None
        raw_text = ""
        error = repr(exc)

    tokens = response_logprob_tokens(response_json)
    return {
        "model": model,
        "host": host,
        "endpoint": endpoint,
        "supported": bool(not error and tokens),
        "token_count": len(tokens),
        "include": ["message.output_text.logprobs"],
        "top_logprobs": 5,
        "raw_text": raw_text,
        "error": error,
        "latency_s": latency_s,
    }


def new_run_id(prefix: str = "run") -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{timestamp}-{uuid.uuid4().hex[:8]}"


def build_raw_record(
    *,
    run_id: str,
    model: str,
    host: str,
    task: str,
    item: dict[str, Any],
    sample_index: int,
    sample_kind: str,
    temperature: float,
    top_p: float,
    prompt_version: str,
    prompt: str,
    completion: dict[str, Any],
    request_index: int | None = None,
    provider_id: str = "",
    profile_id: str = "",
    run_group_id: str = "",
    base_url: str = "",
    api_key_env: str = "",
    json_mode: bool = False,
    structured_output: str = "none",
    response_format: Mapping[str, Any] | None = None,
    request_extra_body: Mapping[str, Any] | None = None,
    server_model_probe: Mapping[str, Any] | str | None = None,
    batch_id: str = "",
    batch_size: int | str = "",
    batch_item_count: int | str = "",
    batch_prompt_hash: str = "",
) -> dict[str, Any]:
    parsed_json, parse_status = parse_task_response(task, completion.get("raw_text", ""))
    parse_status_override = str(completion.get("parse_status_override", "")).strip()
    if parse_status_override:
        parsed_json = None
        parse_status = parse_status_override
    elif not completion.get("ok"):
        parse_status = "request_error"
    record = {
        "run_id": run_id,
        "model": model,
        "host": host,
        "task": task,
        "item_id": item["item_id"],
        "seed_id": item["seed_id"],
        "source_modality": item["source_modality"],
        "sample_index": sample_index,
        "sample_kind": sample_kind,
        "temperature": temperature,
        "top_p": top_p,
        "prompt_version": prompt_version,
        "prompt": prompt,
        "raw_text": completion.get("raw_text", ""),
        "parsed_json": parsed_json,
        "parse_status": parse_status,
        "latency_s": completion.get("latency_s", ""),
        "error": completion.get("error", ""),
    }
    if provider_id:
        record["provider_id"] = provider_id
    if profile_id:
        record["profile_id"] = profile_id
    if run_group_id:
        record["run_group_id"] = run_group_id
    if base_url:
        record["base_url"] = base_url
    if api_key_env:
        record["api_key_env"] = api_key_env
    if json_mode:
        record["json_mode"] = True
    resolved_structured_output = normalize_structured_output_mode(structured_output, json_mode=json_mode)
    if resolved_structured_output != "none":
        record["structured_output"] = resolved_structured_output
    if response_format:
        record["response_format"] = dict(response_format)
    if request_extra_body:
        record["request_extra_body"] = dict(request_extra_body)
    if server_model_probe:
        record["server_model_probe"] = server_model_probe
    if batch_id:
        record["batch_id"] = batch_id
    if batch_size != "":
        record["batch_size"] = int(batch_size)
    if batch_item_count != "":
        record["batch_item_count"] = int(batch_item_count)
    if batch_prompt_hash:
        record["batch_prompt_hash"] = batch_prompt_hash
    if "template_id" in item:
        record["template_id"] = item["template_id"]
    for key in [
        "source_item_id",
        "task2_run_id",
        "task2_model",
        "task2_modality",
        "task3_gold_relation",
    ]:
        if key in item:
            record[key] = item[key]
    if request_index is not None:
        record["request_index"] = request_index
    return record


def run_completion_job(
    job: Mapping[str, Any],
    completion_fn: Callable[..., dict[str, Any]] = chat_completion,
) -> dict[str, Any]:
    completion = completion_fn(
        host=str(job["host"]),
        model=str(job["model"]),
        prompt=str(job["prompt"]),
        temperature=float(job["temperature"]),
        top_p=float(job["top_p"]),
        max_tokens=int(job.get("max_tokens", 256)),
        timeout_s=int(job.get("timeout_s", 120)),
        api_key_env=str(job.get("api_key_env", "LOCAL_OPENAI_API_KEY")),
        response_format=job.get("response_format"),
        extra_body=job.get("extra_body"),
    )
    request_index = job.get("request_index")
    return build_raw_record(
        run_id=str(job["run_id"]),
        model=str(job["model"]),
        host=str(job["host"]),
        task=str(job["task"]),
        item=dict(job["item"]),
        sample_index=int(job["sample_index"]),
        sample_kind=str(job["sample_kind"]),
        temperature=float(job["temperature"]),
        top_p=float(job["top_p"]),
        prompt_version=str(job["prompt_version"]),
        prompt=str(job["prompt"]),
        completion=completion,
        request_index=int(request_index) if request_index is not None else None,
        provider_id=str(job.get("provider_id", "")),
        profile_id=str(job.get("profile_id", "")),
        run_group_id=str(job.get("run_group_id", "")),
        base_url=str(job.get("base_url", job.get("host", ""))),
        api_key_env=str(job.get("api_key_env", "")),
        json_mode=bool(job.get("json_mode", False)),
        structured_output=str(job.get("structured_output", "none")),
        response_format=job.get("response_format"),
        request_extra_body=job.get("extra_body"),
        server_model_probe=job.get("server_model_probe"),
    )


def completion_batch_key(job: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(job.get("host", "")),
        str(job.get("model", "")),
        str(job.get("run_id", "")),
        str(job.get("task", "")),
        str(job.get("sample_kind", "")),
        int(job.get("sample_index", 0)),
        float(job.get("temperature", 0.0)),
        float(job.get("top_p", 1.0)),
        str(job.get("prompt_version", "")),
        int(job.get("max_tokens", 256)),
        int(job.get("timeout_s", 120)),
        str(job.get("api_key_env", "")),
        str(job.get("provider_id", "")),
        str(job.get("profile_id", "")),
        str(job.get("run_group_id", "")),
        bool(job.get("json_mode", False)),
        str(job.get("structured_output", "")),
        compact_json(job.get("response_format")),
        compact_json(job.get("extra_body")),
    )


def completion_job_batches(jobs: Iterable[Mapping[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    job_list = [dict(job) for job in jobs]
    if not job_list:
        return []
    resolved_batch_size = positive_int(batch_size, "batch_size")
    if resolved_batch_size <= 1:
        return [[job] for job in job_list]

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for job in job_list:
        grouped.setdefault(completion_batch_key(job), []).append(job)

    batches: list[list[dict[str, Any]]] = []
    for group_jobs in sorted(grouped.values(), key=lambda rows: min(int(row.get("request_index", 0)) for row in rows)):
        ordered = sorted(group_jobs, key=lambda row: int(row.get("request_index", 0)))
        for start in range(0, len(ordered), resolved_batch_size):
            batches.append(ordered[start : start + resolved_batch_size])
    return batches


def run_completion_batch(
    jobs: list[Mapping[str, Any]],
    completion_fn: Callable[..., dict[str, Any]] = chat_completion,
) -> list[dict[str, Any]]:
    if not jobs:
        return []
    if len(jobs) == 1:
        return [run_completion_job(jobs[0], completion_fn=completion_fn)]

    first = jobs[0]
    batch_prompt = batch_prompt_for_completion_jobs(jobs)
    batch_prompt_hash = hashlib.sha256(batch_prompt.encode("utf-8")).hexdigest()
    batch_response_format, batch_extra_body = resolve_response_format_args(
        str(first["task"]),
        structured_output=first.get("structured_output"),
        json_mode=bool(first.get("json_mode", False)),
        response_format=first.get("response_format"),
        extra_body=first.get("extra_body"),
        batched=True,
    )
    completion = completion_fn(
        host=str(first["host"]),
        model=str(first["model"]),
        prompt=batch_prompt,
        temperature=float(first["temperature"]),
        top_p=float(first["top_p"]),
        max_tokens=int(first.get("max_tokens", 256)) * len(jobs),
        timeout_s=int(first.get("timeout_s", 120)),
        api_key_env=str(first.get("api_key_env", "LOCAL_OPENAI_API_KEY")),
        response_format=batch_response_format,
        extra_body=batch_extra_body,
    )
    batch_id = (
        f"{first['run_id']}:{first['model']}:{first['task']}:"
        f"{first['sample_kind']}:{first['sample_index']}:"
        f"{min(int(job.get('request_index', 0)) for job in jobs)}-"
        f"{max(int(job.get('request_index', 0)) for job in jobs)}"
    )
    parsed_results, batch_parse_status = parse_batch_completion_results(completion.get("raw_text", ""))
    ordered_fallback_results = list(parsed_results.values())
    use_order_fallback = (
        len(ordered_fallback_results) == len(jobs)
        and not any(int(job["request_index"]) in parsed_results for job in jobs)
    )
    records: list[dict[str, Any]] = []
    for position, job in enumerate(jobs):
        request_index = int(job["request_index"])
        result = parsed_results.get(request_index)
        if result is None and use_order_fallback:
            result = ordered_fallback_results[position]
        if completion.get("ok") and result is not None:
            item_completion = {
                "ok": True,
                "raw_text": json.dumps(result, ensure_ascii=False),
                "response_json": completion.get("response_json"),
                "latency_s": completion.get("latency_s", ""),
                "error": "",
            }
        elif completion.get("ok"):
            item_completion = {
                "ok": True,
                "raw_text": completion.get("raw_text", ""),
                "response_json": completion.get("response_json"),
                "latency_s": completion.get("latency_s", ""),
                "error": f"batch_parse_status={batch_parse_status}; missing request_index={request_index}",
                "parse_status_override": "missing_batch_result",
            }
        else:
            item_completion = completion

        records.append(
            build_raw_record(
                run_id=str(job["run_id"]),
                model=str(job["model"]),
                host=str(job["host"]),
                task=str(job["task"]),
                item=dict(job["item"]),
                sample_index=int(job["sample_index"]),
                sample_kind=str(job["sample_kind"]),
                temperature=float(job["temperature"]),
                top_p=float(job["top_p"]),
                prompt_version=str(job["prompt_version"]),
                prompt=str(job["prompt"]),
                completion=item_completion,
                request_index=request_index,
                provider_id=str(job.get("provider_id", "")),
                profile_id=str(job.get("profile_id", "")),
                run_group_id=str(job.get("run_group_id", "")),
                base_url=str(job.get("base_url", job.get("host", ""))),
                api_key_env=str(job.get("api_key_env", "")),
                json_mode=bool(job.get("json_mode", False)),
                structured_output=str(job.get("structured_output", "none")),
                response_format=batch_response_format,
                request_extra_body=batch_extra_body,
                server_model_probe=job.get("server_model_probe"),
                batch_id=batch_id,
                batch_size=len(jobs),
                batch_item_count=len(jobs),
                batch_prompt_hash=batch_prompt_hash,
            )
        )
    return records


def run_completion_jobs(
    jobs: Iterable[Mapping[str, Any]],
    max_workers: int,
    completion_fn: Callable[..., dict[str, Any]] = chat_completion,
    batch_size: int = 1,
) -> Iterable[dict[str, Any]]:
    batches = completion_job_batches(jobs, batch_size=batch_size)
    if not batches:
        return

    worker_count = min(positive_int(max_workers, "max_workers"), len(batches))
    if worker_count == 1:
        for batch in batches:
            yield from run_completion_batch(batch, completion_fn=completion_fn)
        return

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(run_completion_batch, batch, completion_fn=completion_fn)
            for batch in batches
        ]
        for future in as_completed(futures):
            yield from future.result()


def accuracy_score(y_true: list[Any], y_pred: list[Any]) -> float:
    if not y_true:
        return math.nan
    return float(sklearn_accuracy_score(y_true, y_pred))


def binary_f1_score(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return math.nan
    return float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))


def macro_f1_score(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    if not y_true:
        return math.nan
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def brier_score(y_true: list[int], probabilities: list[float]) -> float:
    if not y_true:
        return math.nan
    return float(brier_score_loss(y_true, probabilities))


def ece_score(y_true: list[int], probabilities: list[float], bins: int = 10) -> float:
    if not y_true:
        return math.nan
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    total = len(y)
    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    for bin_index, (low, high) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        if bin_index == bins - 1:
            mask = (p >= low) & (p <= high)
        else:
            mask = (p >= low) & (p < high)
        if not np.any(mask):
            continue
        avg_conf = float(np.mean(p[mask]))
        avg_acc = float(np.mean(y[mask]))
        ece += (int(np.sum(mask)) / total) * abs(avg_acc - avg_conf)
    return float(ece)


def spearman_corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return math.nan
    statistic = spearmanr(xs, ys, nan_policy="omit").statistic
    return float(statistic) if not math.isnan(statistic) else math.nan


def pearson_corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return math.nan
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    if x.size < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def auroc_score(y_true: list[int], probabilities: list[float]) -> float:
    if not y_true or len(set(y_true)) < 2:
        return math.nan
    try:
        return float(roc_auc_score(y_true, probabilities))
    except ValueError:
        return math.nan


def class_order_for_task(task: str) -> list[str]:
    if task == "task1":
        return ["yes", "no"]
    if task == "task2":
        return list(MODALITIES)
    if task == "task3":
        return list(TASK3_RELATIONS)
    raise ValueError(f"Unknown task: {task}")


def label_from_parsed(task: str, parsed: dict[str, Any]) -> str:
    if task == "task1":
        decision = normalize_decision(parsed.get("decision"))
        if decision is None:
            raise ValueError(f"Invalid Task 1 decision: {parsed.get('decision')}")
        return decision
    if task == "task2":
        modality = normalize_modality(parsed.get("modality"))
        if modality is None:
            raise ValueError(f"Invalid Task 2 modality: {parsed.get('modality')}")
        return modality
    if task == "task3":
        relation = normalize_relation(parsed.get("relation"))
        if relation is None:
            raise ValueError(f"Invalid Task 3 relation: {parsed.get('relation')}")
        return relation
    raise ValueError(f"Unknown task: {task}")


def label_distribution(labels: list[str], label_order: list[str]) -> dict[str, float]:
    if not labels:
        return {label: 0.0 for label in label_order}
    counts = Counter(labels)
    total = sum(counts.values())
    return {label: counts.get(label, 0) / total for label in label_order}


def label_distribution_from_rows(task: str, rows: list[dict[str, Any]]) -> dict[str, float]:
    labels = [label_from_parsed(task, row["parsed_json"]) for row in rows]
    return label_distribution(labels, class_order_for_task(task))


def label_distribution_json(distribution: dict[str, float]) -> str:
    normalized = {label: round(float(value), 12) for label, value in distribution.items()}
    return json.dumps(normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def normalized_predictive_entropy(distribution: dict[str, float]) -> float:
    if len(distribution) < 2:
        return 0.0
    probabilities = np.asarray([value for value in distribution.values() if value > 0.0], dtype=float)
    if probabilities.size == 0:
        return math.nan
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / math.log(len(distribution))


def variation_ratio(distribution: dict[str, float]) -> float:
    if not distribution:
        return math.nan
    return 1.0 - max(float(value) for value in distribution.values())


def majority_label(distribution: dict[str, float], label_order: list[str]) -> str:
    if not distribution:
        raise ValueError("Cannot choose a majority label from an empty distribution")
    order_index = {label: index for index, label in enumerate(label_order)}
    return sorted(
        distribution.items(),
        key=lambda pair: (-float(pair[1]), order_index.get(pair[0], len(order_index))),
    )[0][0]


def one_hot_distribution(label: str, label_order: list[str]) -> dict[str, float]:
    return {candidate: 1.0 if candidate == label else 0.0 for candidate in label_order}


def task3_score_fields(item: dict[str, Any], pred_relation: str, evidence_phrase: Any = "") -> dict[str, Any]:
    return {
        "source_item_id": item.get("source_item_id", item.get("item_id", "")),
        "gold_relation": item.get("task3_gold_relation", ""),
        "pred_relation": pred_relation,
        "task2_modality": item.get("task2_modality", ""),
        "task2_requirement": item.get("task2_requirement", ""),
        "evidence_phrase": str(evidence_phrase or ""),
        "evidence_phrase_in_source": evidence_phrase_in_source(evidence_phrase, item.get("source_statement", "")),
    }


def monotonicity_violation_rate(
    rows: list[dict[str, Any]],
    score_field: str = "p_yes",
    tolerance: float = MONOTONICITY_TOLERANCE,
) -> float:
    return monotonicity_violation_diagnostics(rows, score_field=score_field, tolerance=tolerance)["monotonicity_violations"]


def monotonicity_violation_diagnostics(
    rows: list[dict[str, Any]],
    score_field: str = "p_yes",
    tolerance: float = MONOTONICITY_TOLERANCE,
) -> dict[str, float]:
    frame = pd.DataFrame.from_records(rows)
    if frame.empty or score_field not in frame.columns:
        return {
            "monotonicity_violations": math.nan,
            "monotonicity_strict_violations": math.nan,
            "monotonicity_tolerance": float(tolerance),
            "monotonicity_mean_max_increase": math.nan,
            "monotonicity_max_increase": math.nan,
        }
    frame = frame[frame[score_field] != ""].copy()
    if frame.empty:
        return {
            "monotonicity_violations": math.nan,
            "monotonicity_strict_violations": math.nan,
            "monotonicity_tolerance": float(tolerance),
            "monotonicity_mean_max_increase": math.nan,
            "monotonicity_max_increase": math.nan,
        }
    frame[score_field] = pd.to_numeric(frame[score_field])
    frame["_modality_sort"] = frame["source_modality"].map(lambda value: -ORDINAL_STRENGTH[str(value)])
    max_increases: list[float] = []
    for _, seed_frame in frame.groupby("seed_id", sort=False):
        if len(seed_frame) < len(MODALITIES):
            continue
        ordered = seed_frame.sort_values("_modality_sort")
        scores = ordered[score_field].to_numpy(dtype=float)
        diffs = np.diff(scores)
        max_increases.append(max(0.0, float(np.max(diffs))))
    if not max_increases:
        return {
            "monotonicity_violations": math.nan,
            "monotonicity_strict_violations": math.nan,
            "monotonicity_tolerance": float(tolerance),
            "monotonicity_mean_max_increase": math.nan,
            "monotonicity_max_increase": math.nan,
        }
    checked = len(max_increases)
    strict_violations = sum(1 for value in max_increases if value > 1e-12)
    tolerant_violations = sum(1 for value in max_increases if value > float(tolerance) + 1e-12)
    return {
        "monotonicity_violations": tolerant_violations / checked,
        "monotonicity_strict_violations": strict_violations / checked,
        "monotonicity_tolerance": float(tolerance),
        "monotonicity_mean_max_increase": float(np.mean(max_increases)),
        "monotonicity_max_increase": float(np.max(max_increases)),
    }


def bootstrap_ci(
    values: list[Any],
    statistic: Callable[[list[Any]], float],
    iterations: int = 1000,
    seed: int = 20260518,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    if not values:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    point = statistic(values)
    samples = []
    for _ in range(iterations):
        indices = rng.integers(0, len(values), size=len(values))
        resampled = [values[index] for index in indices]
        samples.append(statistic(resampled))
    sample_array = np.asarray([x for x in samples if not math.isnan(x)], dtype=float)
    if sample_array.size == 0:
        return point, math.nan, math.nan
    low, high = np.quantile(sample_array, [alpha / 2, 1 - alpha / 2])
    return float(point), float(low), float(high)


def bootstrap_seed_metric(
    rows: list[dict[str, Any]],
    metric: Callable[[list[dict[str, Any]]], float],
    seed_field: str = "seed_id",
    iterations: int = 1000,
    seed: int = 20260518,
) -> tuple[float, float, float]:
    if not rows:
        return math.nan, math.nan, math.nan
    frame = pd.DataFrame.from_records(rows)
    if seed_field not in frame.columns:
        return math.nan, math.nan, math.nan
    by_seed = {
        str(seed_value): group.to_dict(orient="records")
        for seed_value, group in frame.groupby(seed_field, sort=False)
    }
    seed_keys = np.asarray(list(by_seed.keys()), dtype=object)
    if seed_keys.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    point = metric(rows)
    samples = []
    for _ in range(iterations):
        sample_rows: list[dict[str, Any]] = []
        for seed_key in rng.choice(seed_keys, size=seed_keys.size, replace=True):
            sample_rows.extend(by_seed[str(seed_key)])
        samples.append(metric(sample_rows))
    sample_array = np.asarray([x for x in samples if not math.isnan(x)], dtype=float)
    if sample_array.size == 0:
        return point, math.nan, math.nan
    low, high = np.quantile(sample_array, [0.025, 0.975])
    return float(point), float(low), float(high)


def score_from_distribution(
    raw: dict[str, Any],
    item: dict[str, Any],
    uq_method: str,
    distribution: dict[str, float],
    valid_n: int,
    total_n: int,
    uncertainty_measure: str,
    uncertainty_score: float,
    model_name: str | None = None,
) -> dict[str, Any]:
    task = str(raw.get("task", ""))
    label_order = class_order_for_task(task)
    pred_label = majority_label(distribution, label_order)
    confidence = max(float(value) for value in distribution.values())
    base = score_base(raw, item, uq_method, valid_n, total_n)
    if model_name is not None:
        base["model"] = model_name
    common = {
        **base,
        "confidence": confidence,
        "uncertainty_score": uncertainty_score,
        "uncertainty_measure": uncertainty_measure,
        "label_distribution": label_distribution_json(distribution),
    }
    if task == "task1":
        pred_yes = 1 if pred_label == "yes" else 0
        return {
            **common,
            "y_true": int(item["task1_gold_yes"]),
            "y_pred": pred_yes,
            "p_yes": float(distribution["yes"]),
            "gold_modality": "",
            "pred_modality": "",
            **empty_text_modality_fields(),
        }
    if task == "task2":
        correct = 1 if pred_label == item["task2_gold_modality"] else 0
        return {
            **common,
            "y_true": correct,
            "y_pred": correct,
            "p_yes": "",
            "gold_modality": item["task2_gold_modality"],
            "pred_modality": pred_label,
            **empty_text_modality_fields(),
        }
    if task == "task3":
        gold_relation = item["task3_gold_relation"]
        correct = 1 if pred_label == gold_relation else 0
        return {
            **common,
            "y_true": correct,
            "y_pred": correct,
            "p_yes": "",
            "gold_modality": item["source_modality"],
            "pred_modality": item.get("task2_modality", ""),
            **empty_text_modality_fields(),
            **task3_score_fields(item, pred_label),
        }
    raise ValueError(f"Unknown task: {task}")


def distribution_score_rows(
    raw: dict[str, Any],
    item: dict[str, Any],
    distribution: dict[str, float],
    valid_n: int,
    total_n: int,
    consistency_method: str,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    entropy = normalized_predictive_entropy(distribution)
    vr = variation_ratio(distribution)
    return [
        score_from_distribution(
            raw,
            item,
            consistency_method,
            distribution,
            valid_n,
            total_n,
            "variation_ratio",
            vr,
            model_name=model_name,
        ),
        score_from_distribution(
            raw,
            item,
            "predictive_entropy",
            distribution,
            valid_n,
            total_n,
            "normalized_entropy",
            entropy,
            model_name=model_name,
        ),
        score_from_distribution(
            raw,
            item,
            "variation_ratio",
            distribution,
            valid_n,
            total_n,
            "variation_ratio",
            vr,
            model_name=model_name,
        ),
    ]


def build_ensemble_disagreement_scores(
    benchmark_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    benchmark_by_item = {row["item_id"]: row for row in benchmark_rows}
    raw_rows = filter_raw_rows_to_current_benchmark(benchmark_rows, raw_rows)
    raw_frame = pd.DataFrame.from_records(raw_rows)
    required_columns = {"sample_kind", "run_id", "task", "item_id", "model"}
    if raw_frame.empty or not required_columns.issubset(raw_frame.columns):
        return []
    deterministic_frame = raw_frame[raw_frame["sample_kind"] == "deterministic"]
    if deterministic_frame.empty:
        return []

    scores: list[dict[str, Any]] = []
    for (_, task, item_id), group_frame in deterministic_frame.groupby(["run_id", "task", "item_id"], sort=False):
        item = benchmark_by_item.get(item_id)
        if not item:
            continue
        total_models = group_frame["model"].astype(str).nunique()
        valid_by_model: dict[str, dict[str, Any]] = {}
        for row in group_frame.to_dict(orient="records"):
            if row.get("parse_status") != "ok" or not isinstance(row.get("parsed_json"), dict):
                continue
            valid_by_model.setdefault(str(row.get("model", "")), row)
        if len(valid_by_model) < 2:
            continue
        valid_rows = list(valid_by_model.values())
        distribution = label_distribution_from_rows(str(task), valid_rows)
        model_name = f"{ENSEMBLE_MODEL_PREFIX}:{len(valid_by_model)}_models"
        scores.append(
            score_from_distribution(
                valid_rows[0],
                item,
                "model_ensemble_disagreement",
                distribution,
                valid_n=len(valid_by_model),
                total_n=total_models,
                uncertainty_measure="variation_ratio",
                uncertainty_score=variation_ratio(distribution),
                model_name=model_name,
            )
        )
    return scores


def build_run_group_ensemble_disagreement_scores(
    benchmark_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    run_group_id: str | None = None,
) -> list[dict[str, Any]]:
    benchmark_by_item = {row["item_id"]: row for row in benchmark_rows}
    raw_rows = filter_raw_rows_to_current_benchmark(benchmark_rows, raw_rows)
    raw_frame = pd.DataFrame.from_records(raw_rows)
    required_columns = {"sample_kind", "run_group_id", "task", "item_id", "model"}
    if raw_frame.empty or not required_columns.issubset(raw_frame.columns):
        return []
    deterministic_frame = raw_frame[raw_frame["sample_kind"] == "deterministic"]
    if run_group_id:
        deterministic_frame = deterministic_frame[deterministic_frame["run_group_id"].astype(str) == str(run_group_id)]
    if deterministic_frame.empty:
        return []

    scores: list[dict[str, Any]] = []
    for (group_id, task, item_id), group_frame in deterministic_frame.groupby(["run_group_id", "task", "item_id"], sort=False):
        item = benchmark_by_item.get(item_id)
        if not item:
            continue
        total_models = group_frame.apply(
            lambda row: f"{row.get('provider_id', '')}:{row.get('model', '')}:{row.get('run_id', '')}",
            axis=1,
        ).nunique()
        valid_by_member: dict[str, dict[str, Any]] = {}
        for row in group_frame.to_dict(orient="records"):
            if row.get("parse_status") != "ok" or not isinstance(row.get("parsed_json"), dict):
                continue
            member = f"{row.get('provider_id', '')}:{row.get('model', '')}:{row.get('run_id', '')}"
            valid_by_member.setdefault(member, row)
        if len(valid_by_member) < 2:
            continue
        valid_rows = list(valid_by_member.values())
        distribution = label_distribution_from_rows(str(task), valid_rows)
        model_name = f"{ENSEMBLE_MODEL_PREFIX}:run_group:{group_id}:{len(valid_by_member)}_models"
        scores.append(
            score_from_distribution(
                valid_rows[0],
                item,
                "model_ensemble_disagreement_run_group",
                distribution,
                valid_n=len(valid_by_member),
                total_n=total_models,
                uncertainty_measure="variation_ratio",
                uncertainty_score=variation_ratio(distribution),
                model_name=model_name,
            )
        )
    return scores


def build_uq_scores(benchmark_rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    benchmark_by_item = {row["item_id"]: row for row in benchmark_rows}
    raw_rows = filter_raw_rows_to_current_benchmark(benchmark_rows, raw_rows)
    scores: list[dict[str, Any]] = []

    for raw in raw_rows:
        item = benchmark_by_item.get(raw.get("item_id"))
        parsed = raw.get("parsed_json")
        if not item or raw.get("parse_status") != "ok" or not isinstance(parsed, dict):
            continue
        if raw.get("sample_kind") != "deterministic":
            continue
        base = score_base(raw, item, "verbalized_confidence", valid_n=1, total_n=1)
        confidence = float(parsed["confidence"]) / 100.0
        if raw["task"] == "task1":
            pred_yes = 1 if parsed["decision"] == "yes" else 0
            p_yes = confidence if pred_yes else 1.0 - confidence
            distribution = {"yes": p_yes, "no": 1.0 - p_yes}
            scores.append(
                {
                    **base,
                    "y_true": int(item["task1_gold_yes"]),
                    "y_pred": pred_yes,
                    "p_yes": p_yes,
                    "confidence": confidence,
                    "uncertainty_score": 1.0 - confidence,
                    "uncertainty_measure": "one_minus_confidence",
                    "label_distribution": label_distribution_json(distribution),
                    "gold_modality": "",
                    "pred_modality": "",
                    **empty_text_modality_fields(),
                }
            )
        elif raw["task"] == "task2":
            pred_modality = parsed["modality"]
            correct = 1 if pred_modality == item["task2_gold_modality"] else 0
            text_fields = text_modality_fields(
                parsed.get("requirement", ""),
                item["task2_gold_modality"],
                pred_modality,
                confidence,
            )
            scores.append(
                {
                    **base,
                    "y_true": correct,
                    "y_pred": correct,
                    "p_yes": "",
                    "confidence": confidence,
                    "uncertainty_score": 1.0 - confidence,
                    "uncertainty_measure": "one_minus_confidence",
                    "label_distribution": "",
                    "gold_modality": item["task2_gold_modality"],
                    "pred_modality": pred_modality,
                    **text_fields,
                }
            )

    raw_frame = pd.DataFrame.from_records(raw_rows)
    if raw_frame.empty or "sample_kind" not in raw_frame.columns:
        return scores
    stochastic_frame = raw_frame[raw_frame["sample_kind"] == "stochastic"]

    for (_, task, item_id, _), group_frame in stochastic_frame.groupby(["model", "task", "item_id", "run_id"], sort=False):
        group = group_frame.to_dict(orient="records")
        item = benchmark_by_item.get(item_id)
        if not item:
            continue
        valid = [row for row in group if row.get("parse_status") == "ok" and isinstance(row.get("parsed_json"), dict)]
        if not valid:
            continue
        distribution = label_distribution_from_rows(str(task), valid)
        consistency_method = "label_self_consistency" if task == "task1" else "modality_consistency"
        scores.extend(distribution_score_rows(valid[0], item, distribution, len(valid), len(group), consistency_method))

    scores.extend(build_ensemble_disagreement_scores(benchmark_rows, raw_rows))
    return scores


def build_task3_scores(task3_items: list[dict[str, Any]], raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    item_by_id = {row["item_id"]: row for row in task3_items}
    scores: list[dict[str, Any]] = []

    for raw in raw_rows:
        if raw.get("task") != "task3" or raw.get("sample_kind") != "deterministic":
            continue
        item = item_by_id.get(str(raw.get("item_id", "")))
        parsed = raw.get("parsed_json")
        if not item or raw.get("parse_status") != "ok" or not isinstance(parsed, dict):
            continue
        pred_relation = normalize_relation(parsed.get("relation"))
        if pred_relation is None:
            continue
        confidence = float(parsed["confidence"]) / 100.0
        gold_relation = item["task3_gold_relation"]
        correct = 1 if pred_relation == gold_relation else 0
        base = score_base(raw, item, "verbalized_confidence", valid_n=1, total_n=1)
        scores.append(
            {
                **base,
                "y_true": correct,
                "y_pred": correct,
                "p_yes": "",
                "confidence": confidence,
                "uncertainty_score": 1.0 - confidence,
                "uncertainty_measure": "one_minus_confidence",
                "label_distribution": "",
                "gold_modality": item["source_modality"],
                "pred_modality": item.get("task2_modality", ""),
                **empty_text_modality_fields(),
                **task3_score_fields(item, pred_relation, parsed.get("evidence_phrase", "")),
            }
        )

    raw_frame = pd.DataFrame.from_records(raw_rows)
    if raw_frame.empty or "sample_kind" not in raw_frame.columns:
        return scores
    stochastic_frame = raw_frame[(raw_frame["sample_kind"] == "stochastic") & (raw_frame["task"] == "task3")]
    if stochastic_frame.empty:
        return scores

    for (_, item_id, _), group_frame in stochastic_frame.groupby(["model", "item_id", "run_id"], sort=False):
        group = group_frame.to_dict(orient="records")
        item = item_by_id.get(str(item_id))
        if not item:
            continue
        valid = [row for row in group if row.get("parse_status") == "ok" and isinstance(row.get("parsed_json"), dict)]
        if not valid:
            continue
        distribution = label_distribution_from_rows("task3", valid)
        scores.extend(distribution_score_rows(valid[0], item, distribution, len(valid), len(group), "relation_consistency"))

    return scores


def run_progress_summary(
    benchmark_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    expected_stochastic_samples: int = 5,
) -> list[dict[str, Any]]:
    if not raw_rows:
        return []
    frame = pd.DataFrame.from_records(raw_rows)
    required = {"run_id", "model", "task", "item_id", "sample_kind", "parse_status"}
    if frame.empty or not required.issubset(frame.columns):
        return []
    benchmark_item_ids = {str(row["item_id"]) for row in benchmark_rows}
    benchmark_item_count = len(benchmark_item_ids)
    expected_stochastic_samples = max(0, int(expected_stochastic_samples))
    rows: list[dict[str, Any]] = []
    for (run_id, model, task), group_frame in frame.groupby(["run_id", "model", "task"], sort=False):
        group_frame = group_frame[group_frame["item_id"].astype(str).isin(benchmark_item_ids)]
        if group_frame.empty:
            continue
        deterministic = group_frame[group_frame["sample_kind"] == "deterministic"]
        stochastic = group_frame[group_frame["sample_kind"] == "stochastic"]
        ok_count = int((group_frame["parse_status"] == "ok").sum())
        deterministic_items = deterministic["item_id"].astype(str).nunique()
        stochastic_items = stochastic["item_id"].astype(str).nunique()
        if expected_stochastic_samples:
            stochastic_counts = stochastic.groupby("item_id", sort=False).size()
            stochastic_complete_items = int((stochastic_counts >= expected_stochastic_samples).sum())
        else:
            stochastic_complete_items = benchmark_item_count
        expected_records = benchmark_item_count * (1 + expected_stochastic_samples)
        observed_records = len(group_frame)
        rows.append(
            {
                "run_id": run_id,
                "model": model,
                "task": task,
                "benchmark_items": benchmark_item_count,
                "expected_records": expected_records,
                "observed_records": observed_records,
                "record_completion_rate": observed_records / expected_records if expected_records else math.nan,
                "parse_success_rate": ok_count / observed_records if observed_records else math.nan,
                "deterministic_records": len(deterministic),
                "deterministic_ok": int((deterministic["parse_status"] == "ok").sum()),
                "deterministic_item_coverage": deterministic_items / benchmark_item_count if benchmark_item_count else math.nan,
                "stochastic_records": len(stochastic),
                "stochastic_ok": int((stochastic["parse_status"] == "ok").sum()),
                "stochastic_item_coverage": stochastic_items / benchmark_item_count if benchmark_item_count else math.nan,
                "stochastic_complete_item_rate": stochastic_complete_items / benchmark_item_count if benchmark_item_count else math.nan,
            }
        )
    return rows


def complete_run_ids_from_progress(
    progress_rows: list[dict[str, Any]],
    expected_tasks: Iterable[str] = ("task1", "task2"),
    prefix: str | None = None,
) -> list[str]:
    if not progress_rows:
        return []
    expected_task_set = {str(task) for task in expected_tasks}
    complete: list[str] = []
    frame = pd.DataFrame.from_records(progress_rows)
    if frame.empty or "run_id" not in frame.columns:
        return complete
    for run_id, group_frame in frame.groupby("run_id", sort=False):
        if not run_id_matches_prefix(run_id, prefix):
            continue
        tasks = set(group_frame["task"].astype(str).tolist()) if "task" in group_frame.columns else set()
        if not expected_task_set.issubset(tasks):
            continue
        completion_columns = ["record_completion_rate", "deterministic_item_coverage", "stochastic_complete_item_rate"]
        if all(
            column in group_frame.columns
            and bool((pd.to_numeric(group_frame[column], errors="coerce") >= 1.0).all())
            for column in completion_columns
        ):
            complete.append(str(run_id))
    return complete


def completion_record_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, int]:
    item_id = row.get("item_id", "")
    nested_item = row.get("item")
    if not item_id and isinstance(nested_item, Mapping):
        item_id = nested_item.get("item_id", "")
    return (
        str(row.get("model", "")),
        str(row.get("task", "")),
        str(item_id),
        str(row.get("sample_kind", "")),
        int(row.get("sample_index", 0)),
    )


def pending_completion_jobs(jobs: Iterable[Mapping[str, Any]], raw_rows: Iterable[Mapping[str, Any]], run_id: str) -> list[dict[str, Any]]:
    completed = {
        completion_record_key(row)
        for row in raw_rows
        if str(row.get("run_id", "")) == str(run_id) and str(row.get("parse_status", "")) == "ok"
    }
    return [dict(job) for job in jobs if completion_record_key(job) not in completed]


def run_registry_summary(
    benchmark_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    *,
    run_id: str,
    run_group_id: str,
    provider_id: str,
    profile_id: str,
    model: str,
    dataset_id: str,
    variant: str,
    tasks: Iterable[str],
    expected_stochastic_samples: int,
    started_at_utc: str,
    finished_at_utc: str = "",
    status: str | None = None,
    base_url: str = "",
    api_key_env: str = "",
    concurrency: int | str = "",
    batch_size: int | str = 1,
    timeout_s: int | str = "",
    json_mode: bool = False,
    structured_output: str = "none",
    request_extra_body: Mapping[str, Any] | None = None,
    server_model_probe: Mapping[str, Any] | str | None = None,
    notes: str = "",
) -> dict[str, Any]:
    task_list = normalize_task_filter(tasks)
    run_rows = [
        row
        for row in raw_rows
        if str(row.get("run_id", "")) == str(run_id)
        and str(row.get("model", "")) == str(model)
        and str(row.get("task", "")) in set(task_list)
    ]
    progress = run_progress_summary(benchmark_rows, run_rows, expected_stochastic_samples=expected_stochastic_samples)
    task_progress = [row for row in progress if str(row.get("task", "")) in set(task_list)]
    benchmark_item_count = len({str(row["item_id"]) for row in benchmark_rows})
    expected_records = benchmark_item_count * len(task_list) * (1 + max(0, int(expected_stochastic_samples)))
    resolved_batch_size = positive_int(batch_size, "batch_size")
    expected_api_calls = (
        len(task_list)
        * (1 + max(0, int(expected_stochastic_samples)))
        * math.ceil(benchmark_item_count / resolved_batch_size)
        if benchmark_item_count
        else 0
    )
    observed_records = len(run_rows)
    observed_api_calls = len(
        {
            str(row.get("batch_id") or f"single:{row.get('request_index', index)}")
            for index, row in enumerate(run_rows)
        }
    )
    ok_records = sum(1 for row in run_rows if str(row.get("parse_status", "")) == "ok")
    deterministic_coverages = [float(row.get("deterministic_item_coverage", 0) or 0) for row in task_progress]
    stochastic_coverages = [float(row.get("stochastic_complete_item_rate", 0) or 0) for row in task_progress]
    complete = (
        observed_records >= expected_records
        and len(task_progress) >= len(task_list)
        and all(value >= 1.0 for value in deterministic_coverages)
        and all(value >= 1.0 for value in stochastic_coverages)
    )
    resolved_status = status or ("complete" if complete else "partial")
    return {
        "run_id": run_id,
        "run_group_id": run_group_id,
        "provider_id": provider_id,
        "profile_id": profile_id,
        "model": model,
        "dataset_id": normalize_dataset_id(dataset_id),
        "benchmark_variant": normalize_benchmark_variant(variant),
        "tasks": ",".join(task_list),
        "status": resolved_status,
        "expected_records": expected_records,
        "observed_records": observed_records,
        "parse_success_rate": ok_records / observed_records if observed_records else "",
        "deterministic_item_coverage": min(deterministic_coverages) if deterministic_coverages else "",
        "stochastic_complete_item_rate": min(stochastic_coverages) if stochastic_coverages else "",
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "concurrency": concurrency,
        "batch_size": resolved_batch_size,
        "expected_api_calls": expected_api_calls,
        "observed_api_calls": observed_api_calls,
        "timeout_s": timeout_s,
        "json_mode": "yes" if json_mode else "no",
        "structured_output": normalize_structured_output_mode(structured_output, json_mode=json_mode),
        "request_extra_body": compact_json(request_extra_body),
        "server_model_probe": compact_json(server_model_probe),
        "notes": notes,
    }


def upsert_run_registry_row(path: str | Path, row: Mapping[str, Any]) -> None:
    path = Path(path)
    rows = read_csv_rows(path) if path.exists() else []
    key_fields = ["run_id", "profile_id", "model", "dataset_id", "benchmark_variant"]
    row_key = tuple(str(row.get(field, "")) for field in key_fields)
    updated = False
    output_rows: list[dict[str, Any]] = []
    for existing in rows:
        if tuple(str(existing.get(field, "")) for field in key_fields) == row_key:
            output_rows.append(dict(row))
            updated = True
        else:
            output_rows.append(existing)
    if not updated:
        output_rows.append(dict(row))
    write_csv_rows(path, output_rows, fieldnames=RUN_REGISTRY_FIELDS)


def run_events_path(root: str | Path, dataset_id: str | None = None, variant: str | None = None) -> Path:
    return artifact_path(Path(root) / "data/processed/run_events.jsonl", dataset_id, variant)


def run_progress_live_path(root: str | Path, dataset_id: str | None = None, variant: str | None = None) -> Path:
    return artifact_path(Path(root) / "data/processed/run_progress_live.csv", dataset_id, variant)


def live_run_counters(
    raw_rows: list[dict[str, Any]],
    *,
    expected_records: int,
    expected_api_calls: int,
    started_monotonic: float | None = None,
    now_monotonic: float | None = None,
) -> dict[str, Any]:
    observed_records = len(raw_rows)
    ok_records = sum(1 for row in raw_rows if str(row.get("parse_status", "")) == "ok")
    request_error_records = sum(1 for row in raw_rows if str(row.get("parse_status", "")) == "request_error")
    observed_api_calls = len(
        {
            str(row.get("batch_id") or f"single:{row.get('request_index', index)}")
            for index, row in enumerate(raw_rows)
        }
    )
    elapsed_s = 0.0
    if started_monotonic is not None:
        elapsed_s = max(0.0, float(now_monotonic if now_monotonic is not None else time.monotonic()) - float(started_monotonic))
    records_per_s = observed_records / elapsed_s if elapsed_s > 0 else 0.0
    remaining_records = max(0, int(expected_records) - observed_records)
    eta_s = remaining_records / records_per_s if records_per_s > 0 else ""
    parse_failure_records = observed_records - ok_records
    return {
        "expected_records": int(expected_records),
        "observed_records": observed_records,
        "record_completion_rate": observed_records / expected_records if expected_records else math.nan,
        "ok_records": ok_records,
        "parse_failure_records": parse_failure_records,
        "parse_success_rate": ok_records / observed_records if observed_records else math.nan,
        "parse_failure_rate": parse_failure_records / observed_records if observed_records else 0.0,
        "request_error_records": request_error_records,
        "request_error_rate": request_error_records / observed_records if observed_records else 0.0,
        "expected_api_calls": int(expected_api_calls),
        "observed_api_calls": observed_api_calls,
        "api_call_completion_rate": observed_api_calls / expected_api_calls if expected_api_calls else math.nan,
        "elapsed_s": elapsed_s,
        "records_per_s": records_per_s,
        "eta_s": eta_s,
    }


def _duration_label(seconds: Any) -> str:
    if seconds == "" or seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        return "unknown"
    total = int(max(0, float(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def format_live_progress_line(run_id: str, counters: Mapping[str, Any]) -> str:
    expected_records = int(counters.get("expected_records", 0) or 0)
    observed_records = int(counters.get("observed_records", 0) or 0)
    expected_api_calls = int(counters.get("expected_api_calls", 0) or 0)
    observed_api_calls = int(counters.get("observed_api_calls", 0) or 0)
    completion_pct = 100.0 * float(counters.get("record_completion_rate", 0.0) or 0.0)
    parse_pct = 100.0 * float(counters.get("parse_success_rate", 0.0) or 0.0)
    return (
        f"{run_id}: records {observed_records}/{expected_records} ({completion_pct:.1f}%), "
        f"api {observed_api_calls}/{expected_api_calls}, parse_ok {parse_pct:.1f}%, "
        f"errors {int(counters.get('parse_failure_records', 0) or 0)}, "
        f"elapsed {_duration_label(counters.get('elapsed_s'))}, eta {_duration_label(counters.get('eta_s'))}"
    )


def append_run_event(path: str | Path, event: Mapping[str, Any]) -> None:
    append_jsonl(path, {"created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **dict(event)})


def write_live_progress_csv(
    path: str | Path,
    benchmark_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    expected_stochastic_samples: int,
) -> None:
    progress = run_progress_summary(
        benchmark_rows,
        raw_rows,
        expected_stochastic_samples=expected_stochastic_samples,
    )
    write_csv_rows(path, progress, fieldnames=RUN_PROGRESS_FIELDS)


def warning_events_for_counters(
    counters: Mapping[str, Any],
    logging_config: Mapping[str, Any],
    emitted_warning_types: set[str] | None = None,
) -> list[dict[str, Any]]:
    emitted_warning_types = emitted_warning_types or set()
    observed_records = int(counters.get("observed_records", 0) or 0)
    warn_after = int(logging_config.get("warn_after_records", DEFAULT_RUN_LOGGING["warn_after_records"]))
    if observed_records < warn_after:
        return []

    events: list[dict[str, Any]] = []
    parse_failure_rate = float(counters.get("parse_failure_rate", 0.0) or 0.0)
    parse_threshold = float(logging_config.get("warn_parse_failure_rate", DEFAULT_RUN_LOGGING["warn_parse_failure_rate"]))
    if "parse_failure_rate" not in emitted_warning_types and parse_failure_rate > parse_threshold:
        events.append(
            {
                "event_type": "warning",
                "warning_type": "parse_failure_rate",
                "message": f"Parse failure rate {parse_failure_rate:.3f} exceeded threshold {parse_threshold:.3f}.",
                "threshold": parse_threshold,
                "observed_rate": parse_failure_rate,
                **dict(counters),
            }
        )

    request_error_rate = float(counters.get("request_error_rate", 0.0) or 0.0)
    request_threshold = float(logging_config.get("warn_request_error_rate", DEFAULT_RUN_LOGGING["warn_request_error_rate"]))
    if "request_error_rate" not in emitted_warning_types and request_error_rate > request_threshold:
        events.append(
            {
                "event_type": "warning",
                "warning_type": "request_error_rate",
                "message": f"Request error rate {request_error_rate:.3f} exceeded threshold {request_threshold:.3f}.",
                "threshold": request_threshold,
                "observed_rate": request_error_rate,
                **dict(counters),
            }
        )
    return events


def provider_preflight(
    *,
    host: str,
    model: str,
    api_key_env: str,
    timeout_s: int,
    json_mode: bool = False,
    structured_output: str = "none",
    response_format: Mapping[str, Any] | None = None,
    extra_body: Mapping[str, Any] | None = None,
    completion_fn: Callable[..., dict[str, Any]] = chat_completion,
) -> dict[str, Any]:
    resolved_response_format, resolved_extra_body = resolve_response_format_args(
        "task1",
        structured_output=structured_output,
        json_mode=json_mode,
        response_format=response_format,
        extra_body=extra_body,
        batched=False,
    )
    if resolved_response_format is None and json_mode and structured_output == "none" and not (
        isinstance(resolved_extra_body, Mapping) and "response_format" in resolved_extra_body
    ):
        resolved_response_format = {"type": "json_object"}
    completion = completion_fn(
        host=host,
        model=model,
        prompt=LOGPROB_PROBE_PROMPT,
        temperature=0.0,
        top_p=1.0,
        max_tokens=48,
        timeout_s=timeout_s,
        api_key_env=api_key_env,
        response_format=resolved_response_format,
        extra_body=resolved_extra_body,
    )
    _, parse_status = parse_task_response("task1", completion.get("raw_text", ""))
    return {
        "ok": bool(completion.get("ok")) and parse_status == "ok",
        "parse_status": parse_status,
        "error": completion.get("error", ""),
        "latency_s": completion.get("latency_s", ""),
        "raw_text": str(completion.get("raw_text", ""))[:240],
    }


def preliminary_result_paths(root: str | Path, variant: str | None = None, dataset_id: str | None = None) -> dict[str, Path]:
    root = Path(root)
    return {
        "scores": artifact_path(root / "data/processed/uq_scores_preliminary.csv", dataset_id, variant),
        "summary": artifact_path(root / "data/processed/metrics_summary_preliminary.csv", dataset_id, variant),
        "progress": artifact_path(root / "data/processed/run_progress_preliminary.csv", dataset_id, variant),
        "table": artifact_path(root / "outputs/preliminary_results_table.md", dataset_id, variant),
    }


def write_preliminary_result_snapshot(
    benchmark_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    root: str | Path,
    variant: str | None = None,
    dataset_id: str | None = None,
    expected_stochastic_samples: int = 5,
    include_baseline: bool = True,
) -> dict[str, Any]:
    paths = preliminary_result_paths(root, variant, dataset_id=dataset_id)
    scored_benchmark_rows = benchmark_rows_with_current_raw_outputs(benchmark_rows, raw_rows)
    scores = build_uq_scores(scored_benchmark_rows, raw_rows)
    if include_baseline:
        scores.extend(build_rule_baseline_scores(scored_benchmark_rows))
    summary = metric_summary_by_model_task_method(scores)
    progress = run_progress_summary(
        benchmark_rows,
        raw_rows,
        expected_stochastic_samples=expected_stochastic_samples,
    )
    score_fields = [
        "run_id",
        "model",
        "task",
        "uq_method",
        "item_id",
        "seed_id",
        "source_modality",
        "ordinal_strength",
        "numeric_strength",
        "valid_n",
        "total_n",
        "parse_failures",
        "y_true",
        "y_pred",
        "p_yes",
        "confidence",
        "uncertainty_score",
        "uncertainty_measure",
        "label_distribution",
        "gold_modality",
        "pred_modality",
        "text_modality",
        "text_modality_parse_status",
        "text_modality_correct",
        "label_text_consistent",
        "text_overcommit",
        "text_undercommit",
        "text_high_conf_overcommit_80",
        "text_high_conf_overcommit_90",
    ]
    summary_fields = [
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
        "error_detection_auroc",
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
        "text_under_commitment",
        "text_high_conf_overcommit_80",
        "text_high_conf_overcommit_90",
        "parse_failure_rate",
    ]
    write_csv_rows(paths["scores"], scores, fieldnames=score_fields)
    write_csv_rows(paths["summary"], summary, fieldnames=summary_fields)
    write_csv_rows(paths["progress"], progress, fieldnames=RUN_PROGRESS_FIELDS)

    table_lines = [
        "# Preliminary Results",
        "",
        "These results are computed from the currently cached raw outputs. Treat them as provisional until the run progress is complete.",
        "",
        "## Run Progress",
        markdown_table(progress, RUN_PROGRESS_FIELDS),
        "",
        "## Metric Summary",
        markdown_table(summary, summary_fields),
        "",
    ]
    paths["table"].parent.mkdir(parents=True, exist_ok=True)
    paths["table"].write_text("\n".join(table_lines), encoding="utf-8")
    return {
        "paths": paths,
        "score_rows": len(scores),
        "summary_rows": len(summary),
        "progress_rows": len(progress),
    }


def score_base(raw: dict[str, Any], item: dict[str, Any], uq_method: str, valid_n: int, total_n: int) -> dict[str, Any]:
    return {
        "run_id": raw.get("run_id", ""),
        "run_group_id": raw.get("run_group_id", ""),
        "provider_id": raw.get("provider_id", ""),
        "profile_id": raw.get("profile_id", ""),
        "model": raw.get("model", ""),
        "task": raw.get("task", ""),
        "uq_method": uq_method,
        "item_id": item["item_id"],
        "seed_id": item["seed_id"],
        "source_modality": item["source_modality"],
        "ordinal_strength": int(item["ordinal_strength"]),
        "numeric_strength": float(item["numeric_strength"]),
        "valid_n": valid_n,
        "total_n": total_n,
        "parse_failures": total_n - valid_n,
    }


def baseline_score_base(item: dict[str, Any], task: str) -> dict[str, Any]:
    return {
        "run_id": "rule-based-baseline",
        "model": RULE_BASELINE_MODEL,
        "task": task,
        "uq_method": RULE_BASELINE_METHOD,
        "item_id": item["item_id"],
        "seed_id": item["seed_id"],
        "source_modality": item["source_modality"],
        "ordinal_strength": int(item["ordinal_strength"]),
        "numeric_strength": float(item["numeric_strength"]),
        "valid_n": 1,
        "total_n": 1,
        "parse_failures": 0,
    }


def build_rule_baseline_scores(benchmark_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    for item in benchmark_rows:
        predicted_modality = rule_based_source_modality(item["source_statement"])
        if predicted_modality is None:
            raise ValueError(f"Could not parse source modality for {item.get('item_id')}: {item.get('source_statement')}")

        pred_yes = 1 if predicted_modality == "mandatory" else 0
        task1_distribution = one_hot_distribution("yes" if pred_yes else "no", class_order_for_task("task1"))
        scores.append(
            {
                **baseline_score_base(item, "task1"),
                "y_true": int(item["task1_gold_yes"]),
                "y_pred": pred_yes,
                "p_yes": float(pred_yes),
                "confidence": 1.0,
                "uncertainty_score": 0.0,
                "uncertainty_measure": "deterministic_rule",
                "label_distribution": label_distribution_json(task1_distribution),
                "gold_modality": "",
                "pred_modality": "",
                **empty_text_modality_fields(),
            }
        )

        correct = 1 if predicted_modality == item["task2_gold_modality"] else 0
        task2_distribution = one_hot_distribution(predicted_modality, class_order_for_task("task2"))
        scores.append(
            {
                **baseline_score_base(item, "task2"),
                "y_true": correct,
                "y_pred": correct,
                "p_yes": "",
                "confidence": 1.0,
                "uncertainty_score": 0.0,
                "uncertainty_measure": "deterministic_rule",
                "label_distribution": label_distribution_json(task2_distribution),
                "gold_modality": item["task2_gold_modality"],
                "pred_modality": predicted_modality,
                **empty_text_modality_fields(),
            }
        )
    return scores


def prompt_sensitivity_summary(benchmark_rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scores = build_uq_scores(benchmark_rows, raw_rows)
    frame = pd.DataFrame.from_records(scores)
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for (model, prompt_version), group_frame in frame.groupby(["model", "run_id"], sort=False):
        group_rows = group_frame.to_dict(orient="records")
        task1 = group_frame[group_frame["task"] == "task1"]
        weak = task1[task1["y_true"].astype(int) == 0].copy()
        rows.append(
            {
                "model": model,
                "prompt_run_id": prompt_version,
                "n": len(task1),
                "accuracy": accuracy_score(task1["y_true"].astype(int).tolist(), task1["y_pred"].astype(int).tolist()) if not task1.empty else math.nan,
                "weak_source_high_p_yes_80": high_confidence_overcommitment_rate(group_rows, "task1", 0.80),
                "weak_source_high_p_yes_90": high_confidence_overcommitment_rate(group_rows, "task1", 0.90),
                "mean_weak_p_yes": float(pd.to_numeric(weak["p_yes"]).mean()) if not weak.empty else math.nan,
            }
        )
    return rows


def task2_prompt_sensitivity_summary(benchmark_rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not raw_rows:
        return []
    benchmark_by_item = {row["item_id"]: row for row in benchmark_rows}
    frame = pd.DataFrame.from_records(raw_rows)
    if frame.empty:
        return []
    if "task" not in frame.columns:
        return []
    frame = frame[frame["task"].astype(str) == "task2"].copy()
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for (model, run_id), group_frame in frame.groupby(["model", "run_id"], sort=False):
        score_rows: list[dict[str, Any]] = []
        for raw in group_frame.to_dict(orient="records"):
            item = benchmark_by_item.get(raw.get("item_id"))
            parsed = raw.get("parsed_json")
            if raw.get("task") != "task2" or not item or raw.get("parse_status") != "ok" or not isinstance(parsed, dict):
                continue
            pred_modality = normalize_modality(parsed.get("modality"))
            if pred_modality is None:
                continue
            confidence = float(parsed.get("confidence", 0.0)) / 100.0
            gold_modality = item["task2_gold_modality"]
            correct = 1 if pred_modality == gold_modality else 0
            score_rows.append(
                {
                    "model": model,
                    "run_id": run_id,
                    "task": "task2",
                    "item_id": item["item_id"],
                    "seed_id": item["seed_id"],
                    "source_modality": item["source_modality"],
                    "y_true": correct,
                    "y_pred": correct,
                    "confidence": confidence,
                    "gold_modality": gold_modality,
                    "pred_modality": pred_modality,
                }
            )
        nice_rows = [row for row in score_rows if row["gold_modality"] == "nice_to_have"]
        nice_to_recommended = [row for row in nice_rows if row["pred_modality"] == "recommended"]
        rows.append(
            {
                "model": model,
                "prompt_run_id": run_id,
                "n": len(group_frame),
                "valid_n": len(score_rows),
                "parse_success_rate": len(score_rows) / len(group_frame) if len(group_frame) else math.nan,
                "accuracy": accuracy_score(
                    [row["gold_modality"] for row in score_rows],
                    [row["pred_modality"] for row in score_rows],
                )
                if score_rows
                else math.nan,
                "nice_to_have_n": len(nice_rows),
                "nice_to_have_accuracy": accuracy_score(
                    [row["gold_modality"] for row in nice_rows],
                    [row["pred_modality"] for row in nice_rows],
                )
                if nice_rows
                else math.nan,
                "nice_to_have_to_recommended_rate": len(nice_to_recommended) / len(nice_rows) if nice_rows else math.nan,
                "over_commitment": overcommitment_metrics(score_rows)[0] if score_rows else math.nan,
                "high_conf_overcommit_80": high_confidence_overcommitment_rate(score_rows, "task2", 0.80),
                "high_conf_overcommit_90": high_confidence_overcommitment_rate(score_rows, "task2", 0.90),
            }
        )
    return rows


def weak_modality_probe_summary(probe_items: list[dict[str, Any]], raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not raw_rows:
        return []
    item_by_id = {row["item_id"]: row for row in probe_items}
    frame = pd.DataFrame.from_records(raw_rows)
    if frame.empty:
        return []
    if "task" not in frame.columns:
        return []
    frame = frame[frame["task"].astype(str) == "task2"].copy()
    if frame.empty:
        return []
    for column in ["model", "run_id", "template_id", "sample_kind"]:
        if column not in frame.columns:
            frame[column] = ""

    rows: list[dict[str, Any]] = []
    for (model, run_id, template_id, sample_kind), group_frame in frame.groupby(
        ["model", "run_id", "template_id", "sample_kind"],
        sort=False,
    ):
        valid_predictions: list[dict[str, Any]] = []
        for raw in group_frame.to_dict(orient="records"):
            item = item_by_id.get(str(raw.get("item_id", "")))
            parsed = raw.get("parsed_json")
            if raw.get("task") != "task2" or not item or raw.get("parse_status") != "ok" or not isinstance(parsed, dict):
                continue
            pred_modality = normalize_modality(parsed.get("modality"))
            if pred_modality is None:
                continue
            confidence = float(parsed.get("confidence", 0.0)) / 100.0
            gold_modality = item["task2_gold_modality"]
            valid_predictions.append(
                {
                    "gold_modality": gold_modality,
                    "pred_modality": pred_modality,
                    "confidence": confidence,
                }
            )

        valid_n = len(valid_predictions)
        pred_counts = Counter(row["pred_modality"] for row in valid_predictions)
        over = [
            row
            for row in valid_predictions
            if ORDINAL_STRENGTH[row["pred_modality"]] > ORDINAL_STRENGTH[row["gold_modality"]]
        ]
        row = {
            "model": model,
            "run_id": run_id,
            "template_id": template_id,
            "sample_kind": sample_kind,
            "n": len(group_frame),
            "valid_n": valid_n,
            "parse_success_rate": valid_n / len(group_frame) if len(group_frame) else math.nan,
            "accuracy": (pred_counts.get("nice_to_have", 0) / valid_n) if valid_n else math.nan,
            "to_recommended_rate": (pred_counts.get("recommended", 0) / valid_n) if valid_n else math.nan,
            "over_commitment": len(over) / valid_n if valid_n else math.nan,
            "high_conf_overcommit_80": (
                len([row for row in over if row["confidence"] >= 0.80]) / valid_n if valid_n else math.nan
            ),
            "high_conf_overcommit_90": (
                len([row for row in over if row["confidence"] >= 0.90]) / valid_n if valid_n else math.nan
            ),
            "pred_mandatory_rate": pred_counts.get("mandatory", 0) / valid_n if valid_n else math.nan,
            "pred_recommended_rate": pred_counts.get("recommended", 0) / valid_n if valid_n else math.nan,
            "pred_optional_rate": pred_counts.get("optional", 0) / valid_n if valid_n else math.nan,
            "pred_nice_to_have_rate": pred_counts.get("nice_to_have", 0) / valid_n if valid_n else math.nan,
            "mean_confidence": float(np.mean([row["confidence"] for row in valid_predictions])) if valid_predictions else math.nan,
        }
        rows.append(row)
    return sorted(rows, key=lambda row: (str(row["model"]), str(row["run_id"]), str(row["sample_kind"]), str(row["template_id"])))


def write_weak_modality_probe_summary(summary_rows: list[dict[str, Any]], output_dir: str | Path, suffix: str = "") -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"weak_modality_probe_summary{suffix}.csv"
    markdown_path = output_dir / f"weak_modality_probe_summary{suffix}.md"
    write_csv_rows(csv_path, summary_rows, fieldnames=WEAK_MODALITY_PROBE_SUMMARY_FIELDS)
    markdown_path.write_text(markdown_table(summary_rows, WEAK_MODALITY_PROBE_SUMMARY_FIELDS) + "\n", encoding="utf-8")
    return {"csv": csv_path, "markdown": markdown_path}


def grouped(rows: Iterable[dict[str, Any]], keys: list[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    frame = pd.DataFrame.from_records(list(rows))
    if frame.empty:
        return {}
    for key in keys:
        if key not in frame.columns:
            frame[key] = ""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for group_key, group in frame.groupby(keys, sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        groups[group_key] = group.to_dict(orient="records")
    return groups


def overcommitment_summary_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "over_commitment": math.nan,
            "under_commitment": math.nan,
            "over_commitment_severity_all": math.nan,
            "over_commitment_severity_given_overcommitment": math.nan,
        }
    over = 0
    under = 0
    severity_sum = 0.0
    for row in rows:
        gold = ORDINAL_STRENGTH[row["gold_modality"]]
        pred = ORDINAL_STRENGTH[row["pred_modality"]]
        if pred > gold:
            over += 1
            severity_sum += pred - gold
        if pred < gold:
            under += 1
    total = len(rows)
    return {
        "over_commitment": over / total,
        "under_commitment": under / total,
        "over_commitment_severity_all": severity_sum / total,
        "over_commitment_severity_given_overcommitment": (severity_sum / over) if over else math.nan,
    }


def overcommitment_metrics(rows: list[dict[str, Any]]) -> tuple[float, float, float]:
    metrics = overcommitment_summary_metrics(rows)
    return (
        metrics["over_commitment"],
        metrics["under_commitment"],
        metrics["over_commitment_severity_all"],
    )


def unsupported_mandatory_acceptance_rate(rows: list[dict[str, Any]], threshold: float) -> float:
    weak_rows = [row for row in rows if str(row.get("task", "")) == "task1" and int(row["y_true"]) == 0]
    if not weak_rows:
        return math.nan
    accepted = [row for row in weak_rows if row.get("p_yes") != "" and float(row["p_yes"]) >= threshold]
    return len(accepted) / len(weak_rows)


def task2_high_confidence_overcommitment_rate(
    rows: list[dict[str, Any]],
    threshold: float,
    denominator: str = "all",
) -> float:
    task_rows = [row for row in rows if str(row.get("task", "")) == "task2"]
    if denominator == "overcommittable":
        task_rows = [
            row
            for row in task_rows
            if ORDINAL_STRENGTH[str(row["gold_modality"])] < max(ORDINAL_STRENGTH.values())
        ]
    elif denominator != "all":
        raise ValueError(f"Unknown Task 2 high-confidence denominator: {denominator}")
    if not task_rows:
        return math.nan
    over = []
    for row in task_rows:
        gold = ORDINAL_STRENGTH[row["gold_modality"]]
        pred = ORDINAL_STRENGTH[row["pred_modality"]]
        if pred > gold and float(row["confidence"]) >= threshold:
            over.append(row)
    return len(over) / len(task_rows)


def weak_strengthening_rate(rows: list[dict[str, Any]], threshold: float | None = None) -> float:
    weak_rows = [
        row
        for row in rows
        if str(row.get("task", "")) == "task2" and str(row.get("gold_modality", "")) == "nice_to_have"
    ]
    if not weak_rows:
        return math.nan
    strengthened = []
    for row in weak_rows:
        pred_strength = ORDINAL_STRENGTH[str(row["pred_modality"])]
        if pred_strength <= ORDINAL_STRENGTH["nice_to_have"]:
            continue
        if threshold is not None and float(row["confidence"]) < threshold:
            continue
        strengthened.append(row)
    return len(strengthened) / len(weak_rows)


def weak_modality_recall(rows: list[dict[str, Any]]) -> float:
    weak_rows = [
        row
        for row in rows
        if str(row.get("task", "")) == "task2" and str(row.get("gold_modality", "")) == "nice_to_have"
    ]
    if not weak_rows:
        return math.nan
    preserved = [row for row in weak_rows if str(row.get("pred_modality", "")) == "nice_to_have"]
    return len(preserved) / len(weak_rows)


def high_confidence_overcommitment_rate(rows: list[dict[str, Any]], task: str, threshold: float) -> float:
    task_rows = [row for row in rows if str(row.get("task", "")) == task]
    if not task_rows:
        return math.nan
    if task == "task1":
        return unsupported_mandatory_acceptance_rate(task_rows, threshold)
    if task == "task2":
        return task2_high_confidence_overcommitment_rate(task_rows, threshold, denominator="all")
    if task == "task3":
        return math.nan
    return math.nan


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def text_modality_summary_metrics(rows: list[dict[str, Any]]) -> dict[str, float | str]:
    diagnostic_rows = [
        row
        for row in rows
        if str(row.get("text_modality_parse_status", "")) in {"ok", "unknown"}
    ]
    if not diagnostic_rows:
        return {
            "text_modality_accuracy": "",
            "text_modality_accuracy_all": "",
            "text_modality_parse_coverage": "",
            "label_text_consistency": "",
            "text_over_commitment": "",
            "text_under_commitment": "",
            "text_high_conf_overcommit_80": "",
            "text_high_conf_overcommit_90": "",
        }
    total_rows = len(diagnostic_rows)
    text_rows = [row for row in diagnostic_rows if str(row.get("text_modality_parse_status", "")) == "ok"]
    coverage = len(text_rows) / total_rows
    correct_over_all = (
        sum(1 for row in diagnostic_rows if _truthy(row.get("text_modality_correct"))) / total_rows
    )
    if not text_rows:
        return {
            "text_modality_accuracy": "",
            "text_modality_accuracy_all": correct_over_all,
            "text_modality_parse_coverage": coverage,
            "label_text_consistency": "",
            "text_over_commitment": "",
            "text_under_commitment": "",
            "text_high_conf_overcommit_80": "",
            "text_high_conf_overcommit_90": "",
        }
    total = len(text_rows)
    return {
        "text_modality_accuracy": sum(1 for row in text_rows if _truthy(row.get("text_modality_correct"))) / total,
        "text_modality_accuracy_all": correct_over_all,
        "text_modality_parse_coverage": coverage,
        "label_text_consistency": sum(1 for row in text_rows if _truthy(row.get("label_text_consistent"))) / total,
        "text_over_commitment": sum(1 for row in text_rows if _truthy(row.get("text_overcommit"))) / total,
        "text_under_commitment": sum(1 for row in text_rows if _truthy(row.get("text_undercommit"))) / total,
        "text_high_conf_overcommit_80": sum(1 for row in text_rows if _truthy(row.get("text_high_conf_overcommit_80"))) / total,
        "text_high_conf_overcommit_90": sum(1 for row in text_rows if _truthy(row.get("text_high_conf_overcommit_90"))) / total,
    }


def qualitative_overcommitment_examples(
    scores: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
    limit: int = 5,
    threshold: float = 0.80,
) -> list[dict[str, Any]]:
    benchmark_by_item = {row["item_id"]: row for row in benchmark_rows}
    examples: list[dict[str, Any]] = []
    for row in scores:
        item = benchmark_by_item.get(row.get("item_id"))
        if not item or row.get("model") == RULE_BASELINE_MODEL:
            continue
        task = str(row.get("task", ""))
        if task == "task1":
            if int(row.get("y_true", 0)) != 0 or row.get("p_yes") == "":
                continue
            risk_score = float(row["p_yes"])
            if risk_score < threshold:
                continue
            examples.append(
                {
                    "risk_score": risk_score,
                    "model": row.get("model", ""),
                    "task": task,
                    "uq_method": row.get("uq_method", ""),
                    "seed_id": row.get("seed_id", ""),
                    "source_modality": row.get("source_modality", ""),
                    "source_statement": item.get("source_statement", ""),
                    "candidate_requirement": item.get("candidate_requirement", ""),
                    "gold": "no",
                    "prediction": "yes",
                    "confidence_or_p_yes": risk_score,
                    "why_it_matters": "weak source accepted as mandatory",
                }
            )
        elif task == "task2":
            gold = str(row.get("gold_modality", ""))
            pred = str(row.get("pred_modality", ""))
            if not gold or not pred or ORDINAL_STRENGTH[pred] <= ORDINAL_STRENGTH[gold]:
                continue
            confidence = float(row["confidence"])
            if confidence < threshold:
                continue
            severity = ORDINAL_STRENGTH[pred] - ORDINAL_STRENGTH[gold]
            examples.append(
                {
                    "risk_score": severity + confidence,
                    "model": row.get("model", ""),
                    "task": task,
                    "uq_method": row.get("uq_method", ""),
                    "seed_id": row.get("seed_id", ""),
                    "source_modality": row.get("source_modality", ""),
                    "source_statement": item.get("source_statement", ""),
                    "candidate_requirement": item.get("candidate_requirement", ""),
                    "gold": gold,
                    "prediction": pred,
                    "confidence_or_p_yes": confidence,
                    "why_it_matters": "extracted modality is stronger than source",
                }
            )
    return sorted(examples, key=lambda row: (-float(row["risk_score"]), row["model"], row["seed_id"]))[:limit]


def write_qualitative_overcommitment_examples(
    scores: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
    output_dir: str | Path,
    suffix: str = "",
    limit: int = 5,
    threshold: float = 0.80,
) -> dict[str, Path]:
    examples = qualitative_overcommitment_examples(scores, benchmark_rows, limit=limit, threshold=threshold)
    columns = [
        "risk_score",
        "model",
        "task",
        "uq_method",
        "seed_id",
        "source_modality",
        "source_statement",
        "candidate_requirement",
        "gold",
        "prediction",
        "confidence_or_p_yes",
        "why_it_matters",
    ]
    frame = pd.DataFrame.from_records(examples, columns=columns)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"qualitative_overcommitment_examples{suffix}.csv"
    markdown_path = output_dir / f"qualitative_overcommitment_examples{suffix}.md"
    frame.to_csv(csv_path, index=False)
    markdown_path.write_text(frame.to_markdown(index=False) + "\n" if not frame.empty else "_No high-confidence over-commitment examples._\n", encoding="utf-8")
    return {"csv": csv_path, "markdown": markdown_path}


def calibration_probabilities(rows: list[dict[str, Any]], task: str) -> list[float]:
    if task == "task1":
        return [float(row["p_yes"]) for row in rows]
    return [float(row["confidence"]) for row in rows]


def prediction_error_labels(rows: list[dict[str, Any]], task: str) -> list[int]:
    labels: list[int] = []
    for row in rows:
        if task == "task1":
            labels.append(1 if int(row["y_true"]) != int(row["y_pred"]) else 0)
        elif task in {"task2", "task3"}:
            labels.append(1 - int(row["y_true"]))
        else:
            raise ValueError(f"Unknown task: {task}")
    return labels


def error_detection_auroc(rows: list[dict[str, Any]], task: str) -> float:
    errors: list[int] = []
    uncertainty_scores: list[float] = []
    for row, error in zip(rows, prediction_error_labels(rows, task)):
        value = row.get("uncertainty_score", "")
        if value in {"", None}:
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(score):
            continue
        errors.append(error)
        uncertainty_scores.append(score)
    return auroc_score(errors, uncertainty_scores)


def task3_strengthening_recall(rows: list[dict[str, Any]]) -> float:
    strengthened = [row for row in rows if str(row.get("gold_relation", "")) == "strengthens"]
    if not strengthened:
        return math.nan
    detected = [row for row in strengthened if str(row.get("pred_relation", "")) == "strengthens"]
    return len(detected) / len(strengthened)


def task3_false_preserve_rate(rows: list[dict[str, Any]]) -> float:
    strengthened = [row for row in rows if str(row.get("gold_relation", "")) == "strengthens"]
    if not strengthened:
        return math.nan
    false_preserve = [row for row in strengthened if str(row.get("pred_relation", "")) == "preserves"]
    return len(false_preserve) / len(strengthened)


def task3_evidence_phrase_source_rate(rows: list[dict[str, Any]]) -> float | str:
    evidence_rows = [row for row in rows if str(row.get("evidence_phrase", "")).strip()]
    if not evidence_rows:
        return ""
    return sum(1 for row in evidence_rows if _truthy(row.get("evidence_phrase_in_source"))) / len(evidence_rows)


def task_accuracy(rows: list[dict[str, Any]], task: str) -> float:
    if not rows:
        return math.nan
    if task == "task2":
        return accuracy_score(
            [str(row["gold_modality"]) for row in rows],
            [str(row["pred_modality"]) for row in rows],
        )
    if task == "task3":
        return accuracy_score(
            [str(row["gold_relation"]) for row in rows],
            [str(row["pred_relation"]) for row in rows],
        )
    return accuracy_score(
        [int(row["y_true"]) for row in rows],
        [int(row["y_pred"]) for row in rows],
    )


def labels_present_in_rows(rows: list[dict[str, Any]], field: str, label_order: list[str]) -> list[str]:
    present = {str(row.get(field, "")) for row in rows}
    labels = [label for label in label_order if label in present]
    return labels or list(label_order)


def metric_summary_by_model_task_method(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not scores:
        return []
    frame = pd.DataFrame.from_records(scores)
    summaries: list[dict[str, Any]] = []
    for (model, task, uq_method), group_frame in frame.groupby(["model", "task", "uq_method"], sort=False):
        rows = group_frame.to_dict(orient="records")
        y_true = group_frame["y_true"].astype(int).tolist()
        y_pred = group_frame["y_pred"].astype(int).tolist()
        calibration_scores = calibration_probabilities(rows, str(task))
        summary: dict[str, Any] = {
            "model": model,
            "task": task,
            "uq_method": uq_method,
            "n": len(rows),
            "accuracy": task_accuracy(rows, str(task)),
            "brier": brier_score(y_true, calibration_scores),
            "ece": ece_score(y_true, calibration_scores),
            "error_detection_auroc": error_detection_auroc(rows, str(task)),
            "parse_failure_rate": float(
                group_frame["parse_failures"].astype(int).sum()
                / max(1, int(group_frame["total_n"].astype(int).sum()))
            ),
            "text_modality_accuracy": "",
            "text_modality_accuracy_all": "",
            "text_modality_parse_coverage": "",
            "label_text_consistency": "",
            "text_over_commitment": "",
            "text_under_commitment": "",
            "text_high_conf_overcommit_80": "",
            "text_high_conf_overcommit_90": "",
            "strengthening_recall": "",
            "false_preserve_rate": "",
            "evidence_phrase_source_rate": "",
            "monotonicity_strict_violations": "",
            "monotonicity_tolerance": "",
            "monotonicity_mean_max_increase": "",
            "monotonicity_max_increase": "",
            "unsupported_mandatory_acceptance_80": "",
            "unsupported_mandatory_acceptance_90": "",
            "high_conf_overcommit_all_80": "",
            "high_conf_overcommit_all_90": "",
            "high_conf_overcommit_overcommittable_80": "",
            "high_conf_overcommit_overcommittable_90": "",
            "weak_recall": "",
            "weak_strengthening_80": "",
            "weak_strengthening_90": "",
            "over_commitment_severity_all": "",
            "over_commitment_severity_given_overcommitment": "",
        }
        for threshold in HIGH_CONFIDENCE_THRESHOLDS:
            suffix = f"{int(threshold * 100):02d}"
            summary[f"high_conf_overcommit_{suffix}"] = high_confidence_overcommitment_rate(rows, str(task), threshold)
            if task == "task1":
                summary[f"unsupported_mandatory_acceptance_{suffix}"] = unsupported_mandatory_acceptance_rate(rows, threshold)
            elif task == "task2":
                summary[f"high_conf_overcommit_all_{suffix}"] = task2_high_confidence_overcommitment_rate(
                    rows,
                    threshold,
                    denominator="all",
                )
                summary[f"high_conf_overcommit_overcommittable_{suffix}"] = task2_high_confidence_overcommitment_rate(
                    rows,
                    threshold,
                    denominator="overcommittable",
                )
                summary[f"weak_strengthening_{suffix}"] = weak_strengthening_rate(rows, threshold)
        if task == "task1":
            p_yes = group_frame["p_yes"].astype(float).tolist()
            monotonicity_metrics = monotonicity_violation_diagnostics(rows, "p_yes")
            summary.update(
                {
                    "f1_or_macro_f1": binary_f1_score(y_true, y_pred),
                    "auroc": auroc_score(y_true, p_yes),
                    "spearman_modality_p_yes": spearman_corr(group_frame["numeric_strength"].astype(float).tolist(), p_yes),
                    "pearson_modality_p_yes": pearson_corr(group_frame["numeric_strength"].astype(float).tolist(), p_yes),
                    **monotonicity_metrics,
                    "over_commitment": "",
                    "under_commitment": "",
                    "over_commitment_severity": "",
                }
            )
        elif task == "task2":
            gold = group_frame["gold_modality"].astype(str).tolist()
            pred = group_frame["pred_modality"].astype(str).tolist()
            over_metrics = overcommitment_summary_metrics(rows)
            text_metrics = text_modality_summary_metrics(rows)
            summary.update(
                {
                    "f1_or_macro_f1": macro_f1_score(gold, pred, MODALITIES),
                    "auroc": "",
                    "spearman_modality_p_yes": "",
                    "pearson_modality_p_yes": "",
                    "monotonicity_violations": "",
                    "over_commitment": over_metrics["over_commitment"],
                    "under_commitment": over_metrics["under_commitment"],
                    # Backward-compatible alias: this is severity averaged over all valid outputs.
                    "over_commitment_severity": over_metrics["over_commitment_severity_all"],
                    "over_commitment_severity_all": over_metrics["over_commitment_severity_all"],
                    "over_commitment_severity_given_overcommitment": over_metrics[
                        "over_commitment_severity_given_overcommitment"
                    ],
                    "weak_recall": weak_modality_recall(rows),
                    **text_metrics,
                }
            )
        elif task == "task3":
            gold = group_frame["gold_relation"].astype(str).tolist()
            pred = group_frame["pred_relation"].astype(str).tolist()
            gold_labels = labels_present_in_rows(rows, "gold_relation", TASK3_RELATIONS)
            summary.update(
                {
                    "f1_or_macro_f1": macro_f1_score(gold, pred, gold_labels),
                    "auroc": "",
                    "spearman_modality_p_yes": "",
                    "pearson_modality_p_yes": "",
                    "monotonicity_violations": "",
                    "over_commitment": "",
                    "under_commitment": "",
                    "over_commitment_severity": "",
                    "strengthening_recall": task3_strengthening_recall(rows),
                    "false_preserve_rate": task3_false_preserve_rate(rows),
                    "evidence_phrase_source_rate": task3_evidence_phrase_source_rate(rows),
                }
            )
        summaries.append(summary)
    return sorted(summaries, key=lambda row: (row["model"], row["task"], row["uq_method"]))


def format_metric(value: Any, digits: int = 3) -> str:
    if value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "NA"
    return f"{number:.{digits}f}"


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    if not rows:
        return "_No rows._"
    frame = pd.DataFrame.from_records(rows)
    for field in fields:
        if field not in frame.columns:
            frame[field] = ""
    display = frame.loc[:, fields].copy()
    for field in fields:
        display[field] = display[field].map(format_metric)
    return display.to_markdown(index=False)


def uq_method_inventory_rows() -> list[dict[str, Any]]:
    return [
        {
            "uq_method": "verbalized_confidence",
            "survey_family": "self-verbalized UQ",
            "access_requirement": "black-box JSON output",
            "extra_inference_cost": "none beyond deterministic call",
            "headline": "yes",
            "notes": "Uses the model-reported confidence for the parsed decision or modality.",
        },
        {
            "uq_method": "label_self_consistency",
            "survey_family": "semantic-similarity / consistency UQ",
            "access_requirement": "black-box stochastic samples",
            "extra_inference_cost": "K stochastic Task 1 calls",
            "headline": "yes",
            "notes": "Task 1 majority frequency over yes/no decisions.",
        },
        {
            "uq_method": "modality_consistency",
            "survey_family": "semantic-similarity / consistency UQ",
            "access_requirement": "black-box stochastic samples",
            "extra_inference_cost": "K stochastic Task 2 calls",
            "headline": "yes",
            "notes": "Task 2 majority frequency over modality classes.",
        },
        {
            "uq_method": "relation_consistency",
            "survey_family": "semantic-similarity / consistency UQ",
            "access_requirement": "black-box stochastic verifier samples",
            "extra_inference_cost": "K stochastic Task 3 calls",
            "headline": "diagnostic",
            "notes": "Task 3 majority frequency over source-grounded relation labels.",
        },
        {
            "uq_method": "predictive_entropy",
            "survey_family": "semantic-similarity / distributional UQ",
            "access_requirement": "black-box stochastic samples",
            "extra_inference_cost": "reuses K stochastic calls",
            "headline": "yes",
            "notes": "Normalized entropy over the same stochastic label distribution; report as an uncertainty signal, not a separate prediction method.",
        },
        {
            "uq_method": "variation_ratio",
            "survey_family": "semantic-similarity / distributional UQ",
            "access_requirement": "black-box stochastic samples",
            "extra_inference_cost": "reuses K stochastic calls",
            "headline": "yes",
            "notes": "One minus the empirical probability of the majority label; report as an uncertainty signal, not a separate prediction method.",
        },
        {
            "uq_method": "model_ensemble_disagreement",
            "survey_family": "ensemble UQ",
            "access_requirement": "deterministic outputs from at least two local models",
            "extra_inference_cost": "none beyond planned multi-model run",
            "headline": "conditional",
            "notes": "Skipped automatically when fewer than two valid model outputs are available.",
        },
        {
            "uq_method": "token_logprob_confidence",
            "survey_family": "token-level UQ",
            "access_requirement": "OpenAI-compatible endpoint with logprobs/top_logprobs",
            "extra_inference_cost": "probe only unless endpoint support is confirmed",
            "headline": "no",
            "notes": "Optional robustness extension; not required for the main paper claim.",
        },
        {
            "uq_method": "mechanistic_interpretability",
            "survey_family": "mechanistic interpretability UQ",
            "access_requirement": "model internals",
            "extra_inference_cost": "out of scope",
            "headline": "no",
            "notes": "Explicitly excluded from this short communication because it requires internal activations or model modifications.",
        },
    ]


def write_uq_method_inventory(output_dir: str | Path, suffix: str = "") -> dict[str, Path]:
    rows = uq_method_inventory_rows()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / f"uq_method_inventory{suffix}.md"
    csv_path = output_dir / f"uq_method_inventory{suffix}.csv"
    fields = [
        "uq_method",
        "survey_family",
        "access_requirement",
        "extra_inference_cost",
        "headline",
        "notes",
    ]
    pd.DataFrame.from_records(rows).to_csv(csv_path, index=False)
    markdown_path.write_text(markdown_table(rows, fields) + "\n", encoding="utf-8")
    return {"markdown": markdown_path, "csv": csv_path}


def write_task1_modality_svg(scores: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records(scores)
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    if frame.empty or "task" not in frame.columns:
        ax.text(0.5, 0.5, "No Task 1 scores available.", ha="center", va="center")
        ax.axis("off")
    else:
        task1 = frame[(frame["task"] == "task1") & (frame["p_yes"] != "")].copy()
        if task1.empty:
            ax.text(0.5, 0.5, "No Task 1 scores available.", ha="center", va="center")
            ax.axis("off")
        else:
            task1["p_yes"] = pd.to_numeric(task1["p_yes"])
            summary = (
                task1.groupby(["model", "uq_method", "source_modality"], sort=False)["p_yes"]
                .mean()
                .reset_index()
            )
            x_positions = np.arange(len(MODALITIES))
            for (model, uq_method), group_frame in summary.groupby(["model", "uq_method"], sort=False):
                series = group_frame.set_index("source_modality").reindex(MODALITIES)["p_yes"]
                ax.plot(x_positions, series.to_numpy(dtype=float), marker="o", linewidth=2, label=f"{model} / {uq_method}")
            ax.set_title("Task 1 p_yes by source modality")
            ax.set_xlabel("Source modality")
            ax.set_ylabel("Mean p_yes")
            ax.set_xticks(x_positions)
            ax.set_xticklabels(MODALITIES, rotation=20, ha="right")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(axis="y", alpha=0.25)
            ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, format=path.suffix.lstrip(".") or "svg")
    plt.close(fig)
