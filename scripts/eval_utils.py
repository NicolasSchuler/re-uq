"""Shared utilities for the modality-conditioned UQ evaluation pipeline.

The module intentionally keeps the durable research contracts in one place:
dataset and benchmark construction, provider request planning, raw-output
parsing, UQ scoring, and compact paper-facing exports. Command-line entry
points and notebooks should stay thin wrappers around these helpers.
"""

from __future__ import annotations

import contextlib
import copy
import fcntl
import hashlib
import heapq
import json
import logging
import math
import os
import random
import re
import tempfile
import time
import uuid
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cache
from itertools import batched, pairwise
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

try:
    from scripts import structured_outputs as so
except ModuleNotFoundError:  # pragma: no cover
    import structured_outputs as so

_CACHE_DIR = Path(
    os.environ.get("RE_UQ_CACHE_DIR", Path(__file__).resolve().parents[1] / ".cache")
)
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
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score as sklearn_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODALITIES = ["mandatory", "recommended", "optional", "nice_to_have"]
TASK3_RELATIONS = ["preserves", "strengthens", "weakens", "content_changed"]
TASK3_AUDIT_MODES = ["blind", "declared_text", "declared_source"]
OFFICIAL_TASK3_AUDIT_MODE = "blind"
LEGACY_TASK3_AUDIT_MODE = "legacy_declared"
ACSE_PROXY_METHOD = "acse_semantic_entropy"
ACSE_PROXY_MEASURE = "acse_proxy_semantic_dispersion"
ACSE_PROXY_EMBEDDING_BACKEND = "tfidf_char_wb_3_5"
ACSE_SEMANTIC_MANIFEST_FILENAME = "acse_semantic_artifact_manifest.csv"
ACSE_MLX_EMBEDDING_BACKEND = "mlx"
ACSE_MLX_DEFAULT_MODEL = "mlx-community/Qwen3-Embedding-0.6B-8bit"
ACSE_EMBEDDING_BACKEND_ENV = "RE_UQ_ACSE_EMBEDDING_BACKEND"
ACSE_MLX_MODEL_ENV = "RE_UQ_ACSE_MLX_MODEL"
ACSE_MLX_MAX_LENGTH_ENV = "RE_UQ_ACSE_MLX_MAX_LENGTH"
ACSE_PROXY_DISTANCE_THRESHOLD = 0.35
ACSE_PROXY_INTERNAL_DISPERSION_WEIGHT = 0.20
ACSE_CALIBRATION_SEED = 20260526
ACSE_CALIBRATION_FRACTION = 0.5
ACSE_TARGET_ACCEPTED_ERROR_RATES = (0.10, 0.20)
ACSE_NORMALIZED_SCORE_FIELDS = [
    "run_id",
    "model",
    "task",
    "semantic_embedding_backend",
    "item_id",
    "seed_id",
    "source_modality",
    "gold_modality",
    "pred_modality",
    "valid_n",
    "total_n",
    "y_true",
    "y_pred",
    "prediction_error",
    "acse_raw_uncertainty_score",
    "acse_normalized_uncertainty_score",
    "acse_raw_group_min",
    "acse_raw_group_max",
    "acse_calibration_split",
    "semantic_cluster_count",
    "semantic_cluster_entropy",
    "semantic_cluster_variation_ratio",
    "semantic_dominant_cluster_share",
    "semantic_mean_pairwise_distance",
    "semantic_dominant_cluster_mean_distance",
]
ACSE_CALIBRATION_FIELDS = [
    "run_id",
    "model",
    "task",
    "semantic_embedding_backend",
    "target_accepted_error_rate",
    "selected_normalized_threshold",
    "calibration_n",
    "calibration_accepted_n",
    "calibration_coverage",
    "calibration_accepted_error_rate",
    "evaluation_mode",
    "evaluation_n",
    "evaluation_accepted_n",
    "evaluation_coverage",
    "evaluation_accepted_error_rate",
    "evaluation_deferred_error_rate",
    "all_n",
    "all_error_rate",
    "all_error_detection_auroc",
]
TEXT_MODALITY_BASES = [
    "weak_phrase",
    "explicit_modal",
    "heuristic_system_verb",
    "unknown",
]
STRICT_TEXT_MODALITY_BASES = {"weak_phrase", "explicit_modal"}
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
_MLX_EMBEDDING_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}
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
    "task2_text_modality",
    "task2_text_modality_basis",
    "task2_text_modality_parse_status",
    "task2_confidence",
    "task3_declared_relation",
    "task3_gold_relation",
    "task3_audit_mode",
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
CONFIDENCE_SCALE_0_100 = "0_100"
CONFIDENCE_SCALE_0_1 = so.CONFIDENCE_SCALE_0_1
CONFIDENCE_0_1_PROMPT_VERSIONS = {"v2-conf01", "v2-instructor-conf01"}
LOGPROB_PROBE_PROMPT = 'Return exactly this JSON object: {"decision":"yes","confidence":1.0,"brief_reason":"probe"}'
DATASET_NICE = "nice"
DATASET_MLM_TAPT = "mlm_tapt"
# `pure` is the document-context ablation cell (docs/context_ablation.md): PURE
# corpus requirements that carry an author-assigned mandatory/optional marker
# and their surrounding section and neighbours. It is never pooled into the
# paper's headline cells.
DATASET_PURE = "pure"
DATASET_IDS = {DATASET_NICE, DATASET_MLM_TAPT, DATASET_PURE}
DATASET_SUFFIXES = {
    DATASET_NICE: "",
    DATASET_MLM_TAPT: "_mlm_tapt",
    DATASET_PURE: "_pure",
}
SOURCE_DATASET_LABELS = {
    DATASET_NICE: "NICE",
    DATASET_MLM_TAPT: "mlm_tapt",
    DATASET_PURE: "PURE",
}
# Per-seed document context carried by the `pure` dataset (seed review rows and
# benchmark items alike). Rendered into the prompt only when a run sets
# `item_context: document`; see `document_context_text`.
PURE_CONTEXT_FIELDS = [
    "context_document",
    "context_requirement_id",
    "context_marker",
    "context_section",
    "context_before",
    "context_after",
    "context_legend",
]
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
TABLE_FIGURE_RE = re.compile(
    r"\b(table|figure|fig\.|annex|section|clause)\s+[A-Za-z0-9.-]+", re.I
)
LIST_MARKER_RE = re.compile(
    r"(^|\s)(\d+\.|\([a-z]\)|[a-z]\)|[ivx]+\.|[-*]\s|[•·]|\t)", re.I
)
SYMBOL_HEAVY_RE = re.compile(r"([=<>±×µ%]|\b0x[0-9a-f]+\b)", re.I)

DEFAULT_CONFIG: dict[str, Any] = {
    "project": {
        "seed": 20260518,
        "target_seed_count": 180,
        "pilot_seed_count": 20,
        "prompt_version": "v2-conf01",
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
        # PURE (Ferrari, Spagnolo & Gnesi, RE 2017; Zenodo 7118517, CC BY 4.0).
        # Only the two XML documents with a per-requirement M/O marker are used.
        "pure_xml_url": "https://zenodo.org/records/7118517/files/requirements-xml.zip?download=1",
        "pure_local_zip": "data/raw/pure_requirements_xml.zip",
        "pure_documents": [
            "XMLZIPFile/2007-eirene_fun_7-2.xml",
            "XMLZIPFile/2007-ertms.xml",
        ],
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
    "prompt_version",
    "temperature",
    "top_p",
    "config_sha",
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
    "batch_order",
    "item_context",
    # Run-quality diagnostics (see run_quality_counters in Section 9).
    "parse_status_histogram",
    "retry_total",
    "truncated_records",
    "latency_p50_s",
    "latency_p95_s",
    "usage_completion_tokens",
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
STRUCTURED_OUTPUT_MODES = {"none", "json_object", "json_schema", "instructor"}

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


# =============================================================================
# Section 1: Configuration loading, validation, and run-config normalization
# =============================================================================
# Helpers for reading config.example.json / run_configs/*.json, validating
# scalar and structured fields, normalizing provider profiles, and applying
# command-line overrides. The downstream pipeline consumes only normalized
# dictionaries produced here.


def project_root() -> Path:
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "AGENTS.md").exists() and (
            candidate / "docs" / "evaluation.md"
        ).exists():
            return candidate
    return Path(__file__).resolve().parents[1]


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge `overrides` into `base` and return an independently owned result.

    Both inputs are left untouched and no nested container is shared with
    either of them, so a caller that mutates the merged configuration cannot
    reach back into `DEFAULT_CONFIG` or into the next call's result.
    """
    result: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        current = result.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            result[key] = deep_update(current, value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path = "config.example.json") -> dict[str, Any]:
    """Load a config file merged over `DEFAULT_CONFIG` (never aliasing it)."""
    root = project_root()
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = root / config_path
    if not config_path.exists():
        return copy.deepcopy(DEFAULT_CONFIG)
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
        raise ValueError(
            f"{name} must be a non-negative integer, got {value!r}"
        ) from exc
    if resolved < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return resolved


def nonnegative_float(value: Any, name: str) -> float:
    try:
        resolved = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a non-negative number, got {value!r}"
        ) from exc
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


# --- Credential boundary -----------------------------------------------------
# The configuration contract is that a config carries the *name* of the
# environment variable holding a provider key (`api_key_env`), never the key.
# Provider-specific request bodies (`extra_body`) are free-form, though, and
# raw records, registry rows and exported Hydra profiles are durable artifacts
# that end up in the repository. `assert_no_credential_shaped_values` is the
# single fail-closed check every one of those exporters runs before writing.

ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Matched against the key with non-alphanumeric runs folded to "_", so
# "X-Api-Token" and "x_api_token" are treated alike. `max_tokens` is a request
# knob, not a credential, so bare "token" only matches a whole key or a
# `*_token` suffix.
CREDENTIAL_KEY_SUBSTRINGS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "cookie",
    "passwd",
    "password",
    "secret",
)
CREDENTIAL_KEY_NAMES = frozenset({"auth", "credential", "credentials", "key", "token"})


def _normalized_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower())


def is_credential_shaped_key(key: Any) -> bool:
    """True for keys whose value may carry a credential.

    Keys ending in `_env` name an environment variable rather than holding its
    value, so they are provenance and never match here; their *values* are
    validated separately as environment-variable names.
    """
    normalized = _normalized_key(key)
    if normalized.endswith("_env"):
        return False
    if normalized in CREDENTIAL_KEY_NAMES or normalized.endswith("_token"):
        return True
    return any(hint in normalized for hint in CREDENTIAL_KEY_SUBSTRINGS)


def _is_blank(value: Any) -> bool:
    return value is None or (
        isinstance(value, (str, Mapping, list, tuple)) and len(value) == 0
    )


def assert_no_credential_shaped_values(
    value: Any, *, where: str, _path: str = ""
) -> None:
    """Fail closed before `value` is written to a durable artifact.

    Raises `ValueError` when a credential-shaped key carries a value, or when
    an `*_env` key holds something that is not an environment-variable name
    (the shape a pasted token has). The message names the offending path and
    never echoes the value, so it is safe in logs and tracebacks.
    """
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{_path}.{key}" if _path else str(key)
            if _normalized_key(key).endswith("_env"):
                if not _is_blank(item) and not (
                    isinstance(item, str) and ENV_VAR_NAME_RE.match(item)
                ):
                    raise ValueError(
                        f"{where}: {path} must be the NAME of an environment "
                        "variable (matching ^[A-Za-z_][A-Za-z0-9_]*$), not a "
                        "credential value."
                    )
                continue
            if is_credential_shaped_key(key) and not _is_blank(item):
                raise ValueError(
                    f"{where}: {path} is credential-shaped; secrets must stay "
                    "in the environment and be referenced by an *_env key "
                    "holding the variable name."
                )
            assert_no_credential_shaped_values(item, where=where, _path=path)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_no_credential_shaped_values(
                item, where=where, _path=f"{_path}[{index}]"
            )


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


def _csv_frame(
    rows: list[dict[str, Any]] | pd.DataFrame, fieldnames: list[str] | None = None
) -> pd.DataFrame:
    frame = (
        rows.copy()
        if isinstance(rows, pd.DataFrame)
        else pd.DataFrame.from_records(rows)
    )
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
        return {
            "status": "overwritten" if existed_before else "written",
            "path": path,
            "candidate_path": "",
        }

    existing_text = path.read_text(encoding="utf-8")
    if existing_text == candidate_text:
        return {"status": "unchanged", "path": path, "candidate_path": ""}

    candidate = (
        Path(candidate_path)
        if candidate_path is not None
        else path.with_name(f"{path.stem}_candidate{path.suffix}")
    )
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(candidate_text, encoding="utf-8")
    return {"status": "candidate_written", "path": path, "candidate_path": candidate}


def normalize_dataset_id(dataset_id: str | None) -> str:
    normalized = str(dataset_id or DATASET_NICE).strip().lower().replace("-", "_")
    if normalized in {"", "main", "default", DATASET_NICE}:
        return DATASET_NICE
    if normalized in {DATASET_MLM_TAPT, "hf", "hf_requirements", "limsc_mlm_tapt"}:
        return DATASET_MLM_TAPT
    if normalized in {DATASET_PURE, "pure_context"}:
        return DATASET_PURE
    raise ValueError(f"Unknown dataset_id: {dataset_id}")


def dataset_suffix(dataset_id: str | None) -> str:
    return DATASET_SUFFIXES[normalize_dataset_id(dataset_id)]


def variant_suffix(variant: str | None) -> str:
    return "" if normalize_benchmark_variant(variant) == "must" else "_shall"


def dataset_variant_suffix(dataset_id: str | None, variant: str | None = None) -> str:
    return f"{dataset_suffix(dataset_id)}{variant_suffix(variant)}"


def artifact_path(
    path: str | Path, dataset_id: str | None = None, variant: str | None = None
) -> Path:
    path = Path(path)
    suffix = dataset_variant_suffix(dataset_id, variant)
    if not suffix:
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def task3_verification_items_path(
    root: str | Path,
    dataset_id: str | None,
    variant: str | None,
    source_run_id: str,
    model: str,
    audit_mode: str = OFFICIAL_TASK3_AUDIT_MODE,
    *,
    run_id: Any = None,
    smoke: bool | None = None,
) -> Path:
    """Path of the derived Task 3 audit items for one (source run, model, mode).

    ``run_id`` / ``smoke`` route smoke and fake runs into the parallel smoke tree
    (``data/processed/task3_verification_items/smoke/``) through
    :func:`resolve_run_artifact_path`, so a fake Task 3 run can never overwrite
    the paper-facing item file. Full runs keep the unchanged default path.
    """
    parts = [
        "task3_verification_items",
        normalize_dataset_id(dataset_id),
        normalize_benchmark_variant(variant),
        safe_identifier(source_run_id),
        safe_identifier(model),
        safe_identifier(normalize_task3_audit_mode(audit_mode)),
    ]
    return resolve_run_artifact_path(
        Path(root)
        / "data/processed/task3_verification_items"
        / ("_".join(parts) + ".csv"),
        run_id=run_id,
        smoke=smoke,
    )


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
    default_count = project_config.get(
        "target_seed_count", DEFAULT_CONFIG["project"]["target_seed_count"]
    )
    if dataset_id == DATASET_MLM_TAPT and isinstance(datasets_config, Mapping):
        return positive_int(
            datasets_config.get("mlm_tapt_target_seed_count", default_count),
            "datasets.mlm_tapt_target_seed_count",
        )
    return positive_int(default_count, "project.target_seed_count")


def seed_review_fields(dataset_id: str | None = None) -> list[str]:
    dataset_id = normalize_dataset_id(dataset_id)
    if dataset_id == DATASET_NICE:
        return [field for field in BASE_SEED_REVIEW_FIELDS if field != "source_corpus"]
    if dataset_id == DATASET_PURE:
        return [*BASE_SEED_REVIEW_FIELDS, *PURE_CONTEXT_FIELDS]
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
        elif normalized in {"task3", "modality_verification", "self_verification"}:
            candidates = ["task3"]
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


def selected_values(values: list[str], requested: str | None, name: str) -> list[str]:
    if not requested:
        return values
    normalized = (
        normalize_dataset_id(requested)
        if name == "dataset"
        else normalize_benchmark_variant(requested)
    )
    if normalized not in values:
        raise ValueError(
            f"Requested {name} {normalized!r} is not present in the run config."
        )
    return [normalized]


def logging_config_from_args(
    run_config: Mapping[str, Any], args: Any
) -> dict[str, Any]:
    return normalize_run_logging_config(
        run_config.get("logging"),
        overrides={
            "progress_every_records": args.progress_every_records,
            "progress_every_seconds": args.progress_every_seconds,
            "warn_after_records": args.warn_after_records,
            "warn_parse_failure_rate": args.warn_parse_failure_rate,
            "warn_request_error_rate": args.warn_request_error_rate,
            "write_progress_csv": False if args.no_progress_artifacts else None,
            "write_event_jsonl": False if args.no_progress_artifacts else None,
        },
    )


def normalize_task3_audit_mode(value: Any) -> str:
    normalized = (
        str(value or OFFICIAL_TASK3_AUDIT_MODE)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    aliases = {
        "": OFFICIAL_TASK3_AUDIT_MODE,
        "official": OFFICIAL_TASK3_AUDIT_MODE,
        "blind": "blind",
        "declared": "declared_text",
        "anchored": "declared_text",
        "declared_text": "declared_text",
        "declared_source": "declared_source",
        "legacy": LEGACY_TASK3_AUDIT_MODE,
        "legacy_declared": LEGACY_TASK3_AUDIT_MODE,
    }
    if normalized in aliases:
        return aliases[normalized]
    raise ValueError(f"Unknown Task 3 audit mode: {value}")


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
    return [item for item in values if item]


def normalize_run_logging_config(
    logging_config: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = dict(DEFAULT_RUN_LOGGING)
    if logging_config:
        config.update(dict(logging_config))
    if overrides:
        config.update(
            {key: value for key, value in overrides.items() if value is not None}
        )

    return {
        "progress_every_records": positive_int(
            config["progress_every_records"], "logging.progress_every_records"
        ),
        "progress_every_seconds": nonnegative_int(
            config["progress_every_seconds"], "logging.progress_every_seconds"
        ),
        "warn_after_records": nonnegative_int(
            config["warn_after_records"], "logging.warn_after_records"
        ),
        "warn_parse_failure_rate": nonnegative_float(
            config["warn_parse_failure_rate"], "logging.warn_parse_failure_rate"
        ),
        "warn_request_error_rate": nonnegative_float(
            config["warn_request_error_rate"], "logging.warn_request_error_rate"
        ),
        "write_progress_csv": bool_config(
            config["write_progress_csv"], "logging.write_progress_csv"
        ),
        "write_event_jsonl": bool_config(
            config["write_event_jsonl"], "logging.write_event_jsonl"
        ),
    }


def normalize_structured_output_mode(
    value: Any, *, json_mode: bool = False, json_schema: bool = False
) -> str:
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
        "instructor": "instructor",
        "pydantic": "instructor",
        "validated": "instructor",
    }
    if normalized not in aliases:
        raise ValueError(f"Unknown structured output mode: {value}")
    return aliases[normalized]


# Request-level reproducibility knobs. `seed` is sent to OpenAI-compatible
# providers so that greedy decoding is reproducible on servers that honour it;
# `batch_order` controls how jobs are packed into multi-item batch prompts.
DEFAULT_REQUEST_SEED = 20260518
DEFAULT_MAX_RETRIES = 3
BATCH_ORDER_GROUPED = "grouped"
BATCH_ORDER_SHUFFLED = "shuffled"
BATCH_ORDERS = (BATCH_ORDER_GROUPED, BATCH_ORDER_SHUFFLED)
DEFAULT_BATCH_ORDER = BATCH_ORDER_GROUPED


def normalize_batch_order(value: Any, field: str = "batch_order") -> str:
    """Normalize the batch-composition ablation knob.

    `grouped` keeps consecutive request indices together (all four modality
    variants of a seed land in one batch); `shuffled` deterministically shuffles
    job order inside each batch key group before chunking so batches mix seeds.
    """
    if value is None or value == "":
        return DEFAULT_BATCH_ORDER
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in BATCH_ORDERS:
        raise ValueError(
            f"Unknown {field}: {value!r} (expected one of {list(BATCH_ORDERS)})"
        )
    return normalized


# `item_context` controls whether a Task 2 item is shown bare (the paper
# condition) or inside its document context (docs/context_ablation.md).
ITEM_CONTEXT_BARE = "bare"
ITEM_CONTEXT_DOCUMENT = "document"
ITEM_CONTEXTS = (ITEM_CONTEXT_BARE, ITEM_CONTEXT_DOCUMENT)
DEFAULT_ITEM_CONTEXT = ITEM_CONTEXT_BARE


def normalize_item_context(value: Any, field: str = "item_context") -> str:
    """Normalize the document-context ablation knob (blank means bare)."""
    if value is None or value == "":
        return DEFAULT_ITEM_CONTEXT
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in ITEM_CONTEXTS:
        raise ValueError(
            f"Unknown {field}: {value!r} (expected one of {list(ITEM_CONTEXTS)})"
        )
    return normalized


def _profile_bool(
    profile: Mapping[str, Any], key: str, default: bool, profile_id: str
) -> bool:
    """Read a provider boolean, accepting the serialized string forms.

    A profile that survived a round trip through YAML/JSON/CLI can carry
    `"false"`, which is truthy under `bool()`; parsing it here means a
    disabled knob stays disabled.
    """
    value = profile.get(key, default)
    if value is None:
        return default
    return bool_config(value, f"profiles.{profile_id}.{key}")


def normalize_provider_profile(
    profile: Mapping[str, Any],
    *,
    default_seed: int = DEFAULT_REQUEST_SEED,
    default_batch_order: str = DEFAULT_BATCH_ORDER,
) -> dict[str, Any]:
    provider_id = str(
        profile.get("provider_id") or profile.get("provider") or ""
    ).strip()
    profile_id = str(
        profile.get("profile_id") or profile.get("id") or provider_id
    ).strip()
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
    api_key_env = str(profile.get("api_key_env") or "LOCAL_OPENAI_API_KEY").strip()
    if not ENV_VAR_NAME_RE.match(api_key_env):
        raise ValueError(
            f"Provider profile {profile_id} api_key_env must be the NAME of an "
            "environment variable (matching ^[A-Za-z_][A-Za-z0-9_]*$), not a "
            "credential value."
        )
    extra_body = profile.get("extra_body") or {}
    if not isinstance(extra_body, Mapping):
        raise ValueError(f"Provider profile {profile_id} extra_body must be an object.")
    # Free-form request bodies are the one place a token can enter a config;
    # reject it here so no downstream exporter ever sees it.
    assert_no_credential_shaped_values(
        extra_body, where=f"Provider profile {profile_id} extra_body"
    )
    json_mode = _profile_bool(profile, "json_mode", False, profile_id)
    structured_output = normalize_structured_output_mode(
        profile.get("structured_output"),
        json_mode=json_mode,
        json_schema=_profile_bool(profile, "json_schema", False, profile_id),
    )
    json_mode = json_mode or structured_output in {
        "json_object",
        "json_schema",
        "instructor",
    }
    response_format = profile.get("response_format")
    if structured_output == "instructor" and "response_format" in extra_body:
        extra_body = {
            key: value for key, value in extra_body.items() if key != "response_format"
        }
    if (
        response_format is None
        and structured_output == "json_object"
        and "response_format" not in extra_body
    ):
        response_format = {"type": "json_object"}
    if structured_output == "instructor":
        response_format = None
    if response_format is not None and not isinstance(response_format, Mapping):
        raise ValueError(
            f"Provider profile {profile_id} response_format must be an object."
        )
    return {
        "profile_id": profile_id,
        "provider_id": provider_id,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "models": models,
        "concurrency": positive_int(
            profile.get("concurrency", DEFAULT_CONFIG["llm"]["concurrency"]),
            f"profiles.{profile_id}.concurrency",
        ),
        "batch_size": positive_int(
            profile.get("batch_size", 1), f"profiles.{profile_id}.batch_size"
        ),
        "timeout_s": positive_int(
            profile.get("timeout_s", DEFAULT_CONFIG["llm"]["timeout_s"]),
            f"profiles.{profile_id}.timeout_s",
        ),
        "max_tokens": positive_int(
            profile.get("max_tokens", DEFAULT_CONFIG["llm"]["max_tokens"]),
            f"profiles.{profile_id}.max_tokens",
        ),
        "json_mode": json_mode,
        "structured_output": structured_output,
        "response_format": dict(response_format)
        if response_format is not None
        else None,
        "extra_body": dict(extra_body),
        "instructor_mode": normalize_instructor_mode_name(
            profile.get("instructor_mode") or "json"
        ),
        "validation_retries": nonnegative_int(
            profile.get("validation_retries", 2),
            f"profiles.{profile_id}.validation_retries",
        ),
        "fallback_batch_size": positive_int(
            profile.get("fallback_batch_size", 1),
            f"profiles.{profile_id}.fallback_batch_size",
        ),
        "seed": int(profile.get("seed", default_seed)),
        "send_seed": _profile_bool(profile, "send_seed", True, profile_id),
        "max_retries": nonnegative_int(
            profile.get("max_retries", DEFAULT_MAX_RETRIES),
            f"profiles.{profile_id}.max_retries",
        ),
        "batch_order": normalize_batch_order(
            profile.get("batch_order", default_batch_order),
            f"profiles.{profile_id}.batch_order",
        ),
        "requires_manual_server": _profile_bool(
            profile, "requires_manual_server", False, profile_id
        ),
        "notes": str(profile.get("notes", "")).strip(),
    }


def normalize_run_config(config: Mapping[str, Any]) -> dict[str, Any]:
    run_group_id = str(config.get("run_group_id") or "").strip()
    if not run_group_id:
        raise ValueError("Run config must define run_group_id.")
    profiles_value = config.get("profiles")
    if not isinstance(profiles_value, list) or not profiles_value:
        raise ValueError("Run config must define a non-empty profiles list.")
    project_config = (
        config.get("project", {}) if isinstance(config.get("project"), Mapping) else {}
    )
    llm_config = config.get("llm", {}) if isinstance(config.get("llm"), Mapping) else {}
    deterministic = (
        config.get("deterministic")
        or llm_config.get("deterministic")
        or DEFAULT_CONFIG["llm"]["deterministic"]
    )
    stochastic = (
        config.get("stochastic")
        or llm_config.get("stochastic")
        or DEFAULT_CONFIG["llm"]["stochastic"]
    )
    datasets = [
        normalize_dataset_id(value)
        for value in _string_list(config.get("datasets", [DATASET_NICE]), "datasets")
    ]
    variants = [
        normalize_benchmark_variant(value)
        for value in _string_list(
            config.get("benchmark_variants", ["must"]), "benchmark_variants"
        )
    ]
    tasks = normalize_task_filter(config.get("tasks", ["task1", "task2"]))
    run_seed = int(config.get("seed", llm_config.get("seed", DEFAULT_REQUEST_SEED)))
    run_batch_order = normalize_batch_order(
        config.get("batch_order", llm_config.get("batch_order")), "batch_order"
    )
    run_item_context = normalize_item_context(config.get("item_context"))

    def _acse_setting(value: Any) -> str | None:
        # A missing key must stay None, not become the string "None".
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    return {
        "acse_embedding_backend": _acse_setting(config.get("acse_embedding_backend")),
        "acse_embedding_mlx_model": _acse_setting(
            config.get("acse_embedding_mlx_model")
        ),
        "run_group_id": run_group_id,
        "seed": run_seed,
        "batch_order": run_batch_order,
        "item_context": run_item_context,
        "datasets": datasets,
        "benchmark_variants": variants,
        "tasks": tasks,
        "prompt_version": str(
            config.get("prompt_version")
            or project_config.get("prompt_version")
            or DEFAULT_CONFIG["project"]["prompt_version"]
        ),
        "deterministic": {
            "temperature": float(
                deterministic.get(
                    "temperature", DEFAULT_CONFIG["llm"]["deterministic"]["temperature"]
                )
            ),
            "top_p": float(
                deterministic.get(
                    "top_p", DEFAULT_CONFIG["llm"]["deterministic"]["top_p"]
                )
            ),
            "samples": positive_int(
                deterministic.get("samples", 1), "deterministic.samples"
            ),
        },
        "stochastic": {
            "temperature": float(
                stochastic.get(
                    "temperature", DEFAULT_CONFIG["llm"]["stochastic"]["temperature"]
                )
            ),
            "top_p": float(
                stochastic.get("top_p", DEFAULT_CONFIG["llm"]["stochastic"]["top_p"])
            ),
            "samples": max(
                0,
                int(
                    stochastic.get(
                        "samples", DEFAULT_CONFIG["llm"]["stochastic"]["samples"]
                    )
                ),
            ),
        },
        "logging": normalize_run_logging_config(
            config.get("logging")
            if isinstance(config.get("logging"), Mapping)
            else None
        ),
        "profiles": [
            normalize_provider_profile(
                profile, default_seed=run_seed, default_batch_order=run_batch_order
            )
            for profile in profiles_value
        ],
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
        profiles = [
            profile
            for profile in profiles
            if profile["profile_id"] == requested or profile["provider_id"] == requested
        ]
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


# =============================================================================
# Section 2: Artifact paths, manifests, and JSON/JSONL IO
# =============================================================================
# Path resolution for run registries, SHA-256 hashing for provenance, the
# benchmark manifest writer, and the JSONL/JSON readers and writers
# (append_jsonl appends; write_json overwrites).


SMOKE_RUN_PREFIX = "smoke-"
SMOKE_TREE_DIRNAME = "smoke"
SMOKE_TREE_ENV_VAR = "RE_UQ_SMOKE_TREE"


def is_smoke_run_id(run_id: Any) -> bool:
    """True for run ids produced by smoke/fake Task 1/2/3 runs.

    Task 3 inserts optional audit-mode and benchmark-variant components before
    ``smoke`` (for example ``task3-declared-text-shall-smoke-*``), so checking
    only the historical ``task3-smoke-*`` prefix misses valid smoke runs.
    """
    value = str(run_id or "").strip()
    if value.startswith(SMOKE_RUN_PREFIX):
        return True
    return bool(
        re.match(
            r"^task3(?:-declared-(?:text|source))?(?:-shall)?-smoke-",
            value,
        )
    )


def smoke_tree_env_enabled() -> bool:
    """Allow read-only consumers to opt into the smoke tree via the environment."""
    return str(os.environ.get(SMOKE_TREE_ENV_VAR, "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def smoke_tree_path(path: str | Path) -> Path:
    """Map ``data/processed/<name>`` to ``data/processed/smoke/<name>`` (idempotent)."""
    path = Path(path)
    if path.parent.name == SMOKE_TREE_DIRNAME:
        return path
    return path.parent / SMOKE_TREE_DIRNAME / path.name


def resolve_run_artifact_path(
    path: str | Path,
    *,
    run_id: Any = None,
    smoke: bool | None = None,
) -> Path:
    """Route run artifacts of smoke/fake runs into the parallel smoke tree.

    Paper-facing full runs keep writing into ``data/processed/``; smoke runs never
    touch those files so a fake run can never contaminate published numbers.
    """
    use_smoke = (
        bool(smoke)
        or is_smoke_run_id(run_id)
        or (smoke is None and run_id is None and smoke_tree_env_enabled())
    )
    return smoke_tree_path(path) if use_smoke else Path(path)


def run_registry_path(
    root: str | Path,
    dataset_id: str | None = None,
    variant: str | None = None,
    *,
    run_id: Any = None,
    smoke: bool | None = None,
) -> Path:
    return resolve_run_artifact_path(
        artifact_path(
            Path(root) / "data/processed/run_registry.csv", dataset_id, variant
        ),
        run_id=run_id,
        smoke=smoke,
    )


def model_outputs_raw_path(
    root: str | Path,
    dataset_id: str | None = None,
    variant: str | None = None,
    *,
    run_id: Any = None,
    smoke: bool | None = None,
) -> Path:
    return resolve_run_artifact_path(
        artifact_path(
            Path(root) / "data/processed/model_outputs_raw.jsonl", dataset_id, variant
        ),
        run_id=run_id,
        smoke=smoke,
    )


def task3_raw_path(
    root: str | Path,
    dataset_id: str | None = None,
    variant: str | None = None,
    *,
    run_id: Any = None,
    smoke: bool | None = None,
) -> Path:
    return resolve_run_artifact_path(
        artifact_path(
            Path(root) / "data/processed/model_outputs_raw_task3_verification.jsonl",
            dataset_id,
            variant,
        ),
        run_id=run_id,
        smoke=smoke,
    )


def task3_registry_path(
    root: str | Path,
    dataset_id: str | None = None,
    variant: str | None = None,
    *,
    run_id: Any = None,
    smoke: bool | None = None,
) -> Path:
    return resolve_run_artifact_path(
        artifact_path(
            Path(root) / "data/processed/run_registry_task3_verification.csv",
            dataset_id,
            variant,
        ),
        run_id=run_id,
        smoke=smoke,
    )


def task3_progress_path(
    root: str | Path,
    dataset_id: str | None = None,
    variant: str | None = None,
    *,
    run_id: Any = None,
    smoke: bool | None = None,
) -> Path:
    return resolve_run_artifact_path(
        artifact_path(
            Path(root) / "data/processed/run_progress_live_task3_verification.csv",
            dataset_id,
            variant,
        ),
        run_id=run_id,
        smoke=smoke,
    )


def task3_events_path(
    root: str | Path,
    dataset_id: str | None = None,
    variant: str | None = None,
    *,
    run_id: Any = None,
    smoke: bool | None = None,
) -> Path:
    return resolve_run_artifact_path(
        artifact_path(
            Path(root) / "data/processed/run_events_task3_verification.jsonl",
            dataset_id,
            variant,
        ),
        run_id=run_id,
        smoke=smoke,
    )


def run_log_path(root: str | Path, run_id: str) -> Path:
    return Path(root) / "data/processed/logs" / f"{safe_identifier(run_id)}.log"


def acse_semantic_cache_dir(analysis_dir: str | Path, backend_label: str) -> Path:
    return Path(analysis_dir) / f"acse_semantic_{safe_identifier(backend_label)}"


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes.

    `hashlib.file_digest` requires a freshly opened blocking binary file, which
    is exactly what this helper opens; the digest is identical to hashing the
    whole byte string.
    """
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def artifact_metadata(
    path: str | Path, root: str | Path | None = None
) -> dict[str, Any]:
    path = Path(path)
    root_path = Path(root).resolve() if root is not None else None
    resolved = path.resolve()
    relative = (
        str(resolved.relative_to(root_path))
        if root_path and resolved.is_relative_to(root_path)
        else str(path)
    )
    metadata: dict[str, Any] = {
        "path": relative,
        "exists": path.exists(),
    }
    if not path.exists():
        metadata.update({"sha256": "", "bytes": 0, "rows": ""})
        return metadata
    metadata["sha256"] = sha256_file(path)
    metadata["bytes"] = path.stat().st_size
    metadata["rows"] = (
        len(read_csv_frame(path)) if path.suffix.lower() == ".csv" else ""
    )
    return metadata


def write_benchmark_manifest(
    paths: list[str | Path],
    output_path: str | Path,
    root: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_path = Path(output_path)
    manifest = {
        "created_at_utc": utc_now_iso(),
        "metadata": metadata or {},
        "artifacts": [artifact_metadata(path, root=root) for path in paths],
    }
    write_json(output_path, manifest)
    return manifest


def _manifest_artifact_required(rel_path: str) -> bool:
    """Tracked artifacts whose absence invalidates the analysis (prompts + benchmark CSVs)."""
    normalized = rel_path.replace("\\", "/")
    if normalized.startswith("prompts/"):
        return True
    name = Path(normalized).name
    return (
        normalized.startswith("data/processed/")
        and name.startswith("benchmark_items")
        and name.endswith(".csv")
    )


def verify_benchmark_manifest(
    manifest_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    """Recompute sha256 for each artifact listed in the frozen benchmark manifest.

    Hard-fails (ValueError) on any hash mismatch, and on a missing required artifact
    (tracked prompts/ files or data/processed/benchmark_items*.csv). Missing gitignored
    data artifacts are reported but not fatal. Returns a summary dict.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise ValueError(f"Benchmark manifest not found: {manifest_path}")
    root = Path(project_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checked = 0
    missing: list[str] = []
    for entry in manifest.get("artifacts", []):
        rel_path = str(entry.get("path", "")).strip()
        expected = str(entry.get("sha256", "")).strip()
        if not rel_path:
            continue
        target = root / rel_path
        if not target.exists():
            if _manifest_artifact_required(rel_path):
                raise ValueError(
                    f"Required benchmark artifact missing: {rel_path} (listed in {manifest_path.name})."
                )
            missing.append(rel_path)
            continue
        if not expected:
            continue
        actual = sha256_file(target)
        if actual != expected:
            raise ValueError(
                f"Benchmark artifact hash mismatch for {rel_path}: "
                f"expected {expected}, got {actual} (listed in {manifest_path.name})."
            )
        checked += 1
    return {
        "manifest": str(manifest_path),
        "checked": checked,
        "missing": missing,
        "missing_count": len(missing),
    }


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


def latest_run_id(
    raw_rows: list[dict[str, Any]], prefix: str | Iterable[str] | None = None
) -> str | None:
    latest: str | None = None
    for row in raw_rows:
        run_id = str(row.get("run_id", "")).strip()
        if not run_id:
            continue
        if not run_id_matches_prefix(run_id, prefix):
            continue
        latest = run_id
    return latest


def select_run_rows(
    raw_rows: list[dict[str, Any]],
    run_id: str | None = None,
    prefix: str | Iterable[str] | None = None,
) -> tuple[str | None, list[dict[str, Any]]]:
    selected_run_id = (
        str(run_id).strip() if run_id else latest_run_id(raw_rows, prefix=prefix)
    )
    if not selected_run_id:
        return None, []
    if not run_id_matches_prefix(selected_run_id, prefix):
        return selected_run_id, []
    return selected_run_id, [
        row for row in raw_rows if row.get("run_id") == selected_run_id
    ]


def run_id_matches_prefix(
    run_id: Any, prefix: str | Iterable[str] | None = None
) -> bool:
    if not prefix:
        return True
    if not isinstance(prefix, str):
        prefixes = [str(value) for value in prefix]
        if not prefixes:
            return True
        return any(run_id_matches_prefix(run_id, value) for value in prefixes)
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


@contextlib.contextmanager
def file_lock(path: str | Path) -> Iterator[None]:
    """Advisory exclusive lock on a sidecar ``<path>.lock`` file.

    Concurrent runners share the per-dataset raw JSONL and registry CSVs, so
    every append/rewrite is serialized across processes. The lock is advisory:
    it only protects writers that go through these helpers.
    """
    lock_path = Path(str(path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield handle
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n"
    with file_lock(path), path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write ``text`` to ``path`` via a temp file + ``os.replace`` (never a torn file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Not a `with`: the temp file must stay named after it is closed so the
    # atomic rename below can move it into place.
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        Path(handle.name).replace(path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


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
    response = requests.get(
        url, timeout=timeout_s, headers={"User-Agent": "re-uq-evaluation/0.1"}
    )
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


SPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
NEGATION_RE = re.compile(
    r"\b(no|not|never|none|without|cannot|can't|won't|mustn't|shouldn't)\b", re.I
)
FORMULA_RE = re.compile(r"(<=|>=|==|!=|[<>]|%|\b\d+\s*[*/+-]\s*\d+\b)")
SENTENCE_END_RE = re.compile(r"[.!?]+")
OUTER_QUOTES_RE = re.compile(r"^[\"'“”‘’]+|[\"'“”‘’]+$")
STRANDED_PREPOSITION_RE = re.compile(
    r"^(?:with|to|from|for|of|in|on|at|by|about|into|onto|through|across|under|over|between|among)\b",
    re.I,
)


# =============================================================================
# Section 3: Text normalization and seed-candidate construction
# =============================================================================
# Low-level text helpers (quote stripping, capability extraction, modality cue
# regexes) and the dataset-specific seed candidate pipelines for NICE/PROMISE
# and the `limsc/mlm-tapt-requirements` Hugging Face dataset.


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
    text = re.sub(
        r"^(?:the\s+)?(?:system|software|application|app|product|platform|service|tool|data|table)\s+",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"^(?:shall|must|should|may|will|can|could)\s+(?:be\s+able\s+to\s+)?",
        "",
        text,
        flags=re.I,
    )
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
    if SENTENCE_END_RE.search(cleaned_requirement) or SENTENCE_END_RE.search(
        cleaned_capability
    ):
        reasons.append("multi_sentence")
    if NEGATION_RE.search(requirement):
        reasons.append("negation")
    if FORMULA_RE.search(requirement):
        reasons.append("formula_or_symbol")
    heavy_conjunctions = sum(
        requirement.lower().count(token) for token in [" and ", " or ", ";"]
    )
    if heavy_conjunctions > 1:
        reasons.append("possibly_multiple_capabilities")
    if not capability or word_count(capability) < 2:
        reasons.append("empty_or_too_short_capability")
    if CAPABILITY_MODAL_RE.search(capability):
        reasons.append("residual_modal_in_capability")
    if STRANDED_PREPOSITION_RE.search(cleaned_capability):
        reasons.append("stranded_preposition")
    return not reasons, ";".join(reasons)


def mlm_tapt_filter(
    requirement: str,
    capability: str,
    source_corpus: str = "",
    exclude_source_regex: str = "_PURE$",
) -> tuple[bool, str]:
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
    for preferred in [
        "RequirementText",
        "requirementtext",
        "requirement",
        "text",
        "Requirement",
    ]:
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
    source_dataset: str = SOURCE_DATASET_LABELS[DATASET_NICE],
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
            candidate["source_corpus"] = normalize_space(
                row.get(source_corpus_field, "")
            )
        candidates.append(candidate)
    return candidates


def load_mlm_tapt_rows(config: Mapping[str, Any]) -> list[dict[str, str]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Install the 'datasets' dependency to load limsc/mlm-tapt-requirements."
        ) from exc

    datasets_config = config.get("datasets", {}) if isinstance(config, Mapping) else {}
    repo = str(
        datasets_config.get(
            "mlm_tapt_repo", DEFAULT_CONFIG["datasets"]["mlm_tapt_repo"]
        )
    )
    config_name = str(
        datasets_config.get(
            "mlm_tapt_config", DEFAULT_CONFIG["datasets"]["mlm_tapt_config"]
        )
    )
    splits = datasets_config.get(
        "mlm_tapt_splits", DEFAULT_CONFIG["datasets"]["mlm_tapt_splits"]
    )
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
    eligible_indices = [
        index
        for index, row in enumerate(candidates)
        if is_truthy(row.get("auto_include", ""))
    ]
    if len(eligible_indices) < target_count:
        raise ValueError(
            f"Expected at least {target_count} eligible candidates, found {len(eligible_indices)}."
        )

    source_counts = Counter(
        str(candidates[index].get(source_field, "") or "unknown")
        for index in eligible_indices
    )
    rng = random.Random(seed)
    remaining = set(eligible_indices)
    selected: list[int] = []
    selected_by_source: Counter[str] = Counter()

    while len(selected) < target_count:
        allowed = [
            index
            for index in sorted(remaining)
            if selected_by_source[
                str(candidates[index].get(source_field, "") or "unknown")
            ]
            < source_cap
        ]
        if not allowed:
            raise ValueError(
                f"Could not sample {target_count} candidates with source_cap={source_cap}; selected {len(selected)}."
            )
        weights = [
            1.0
            / source_counts[str(candidates[index].get(source_field, "") or "unknown")]
            for index in allowed
        ]
        chosen = rng.choices(allowed, weights=weights, k=1)[0]
        selected.append(chosen)
        remaining.remove(chosen)
        selected_by_source[
            str(candidates[chosen].get(source_field, "") or "unknown")
        ] += 1
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

    selected_indices = set(
        weighted_sample_candidate_indices(
            candidates, target_count, seed=seed, source_cap=source_cap
        )
    )
    for index, candidate in enumerate(candidates):
        if index in selected_indices:
            candidate["include"] = "yes"
            candidate["exclusion_reason"] = ""
            candidate["capability_text_final"] = candidate["capability_text_auto"]
        elif is_truthy(candidate["auto_include"]):
            candidate["exclusion_reason"] = "not_sampled_weighted_pool"
    return candidates


# --- PURE corpus (document-context ablation) ---------------------------------
# Two PURE XML documents carry an author-assigned marker on every requirement:
# ERTMS FRS 5.0 as a `<modifier>M|O</modifier>` child, EIRENE FRS 7 as an inline
# `(M)`, `(O)` or `(I)` token at the end of the text body. Both define the
# legend in their introduction: (M) mandatory, (O) optional, (I) informative.

PURE_XML_NS = "{req_document.xsd}"
PURE_INLINE_MARKER_RE = re.compile(r"\((M|O|I)\)")
PURE_LEADING_ID_RE = re.compile(r"^\d+(?:\.\d+)*[a-z]?\s+")
PURE_MARKER_LEGEND = "(M) mandatory, (O) optional"
# EIRENE also marks informative paragraphs; the legend is what the document
# itself states in its introduction, so neighbours' markers stay explained.
PURE_DOCUMENT_LEGENDS = {
    "2007-eirene_fun_7-2": "(M) mandatory, (O) optional, (I) informative",
    "2007-ertms": PURE_MARKER_LEGEND,
}
PURE_IMPERSONAL_RE = re.compile(
    r"^it\s+(?:shall|should|may|must|will)\s+(?:not\s+)?be\s+possible\b", re.I
)
PURE_DOCUMENT_TITLES = {
    "2007-eirene_fun_7-2": "EIRENE Functional Requirements Specification, version 7",
    "2007-ertms": "ERTMS/ETCS Functional Requirements Specification, version 5.0",
}


def _pure_element_text(element: ET.Element | None) -> str:
    return normalize_space("".join(element.itertext())) if element is not None else ""


def _pure_document_id(member_name: str) -> str:
    return Path(member_name).stem


def _pure_requirement_fields(req: ET.Element) -> dict[str, Any]:
    """Split one `<req>` into its clean text and its author marker."""
    text = _pure_element_text(req.find(f"{PURE_XML_NS}text_body"))
    modifier = _pure_element_text(req.find(f"{PURE_XML_NS}modifier"))
    inline_markers = PURE_INLINE_MARKER_RE.findall(text)
    if modifier:
        marker, marker_count = modifier, 1
    elif inline_markers:
        marker, marker_count = inline_markers[-1], len(inline_markers)
    else:
        marker, marker_count = "", 0
    # EIRENE bodies start with the requirement id and may carry the next
    # sub-heading after the final marker; keep only the text before it.
    text = PURE_LEADING_ID_RE.sub("", text)
    marked_text = text
    if inline_markers:
        last = list(PURE_INLINE_MARKER_RE.finditer(text))[-1]
        marked_text = text[: last.end()]
        text = text[: last.start()]
    return {
        "text": normalize_space(PURE_INLINE_MARKER_RE.sub(" ", text)),
        # Verbatim body with its inline markers, for list-shaped neighbours.
        "marked_text": normalize_space(marked_text),
        "marker": marker,
        "marker_count": marker_count,
    }


def parse_pure_document(
    xml_text: str | bytes, document_id: str
) -> list[dict[str, Any]]:
    """Flatten one PURE XML document into requirement rows with their context.

    Each row carries the requirement id and marker, the section title path,
    and its previous/next requirement in document order (empty at the edges).
    """
    root = ET.fromstring(xml_text)
    document_title = PURE_DOCUMENT_TITLES.get(
        document_id, _pure_element_text(root.find(f"{PURE_XML_NS}title"))
    )
    legend = PURE_DOCUMENT_LEGENDS.get(document_id, PURE_MARKER_LEGEND)
    rows: list[dict[str, Any]] = []

    def walk(element: ET.Element, path: list[str]) -> None:
        for child in element:
            tag = child.tag.removeprefix(PURE_XML_NS)
            if tag == "p":
                title = _pure_element_text(child.find(f"{PURE_XML_NS}title"))
                walk(child, [*path, title] if title else path)
            elif tag == "req":
                fields = _pure_requirement_fields(child)
                rows.append(
                    {
                        "document_id": document_id,
                        "document_title": document_title,
                        "document_legend": legend,
                        "requirement_id": str(child.get("id", "")),
                        "section_path": " > ".join(path),
                        **fields,
                    }
                )

    walk(root, [])

    def neighbour(row: dict[str, Any] | None) -> str:
        if row is None or not row["text"]:
            return ""
        if row["marker_count"] > 1:
            # A list requirement marks each sub-item; show it as written.
            return f"{row['requirement_id']}: {row['marked_text']}"
        marker = f" ({row['marker']})" if row["marker"] else ""
        return f"{row['requirement_id']}{marker}: {row['text']}"

    for index, row in enumerate(rows):
        row["neighbour_before"] = neighbour(rows[index - 1] if index else None)
        row["neighbour_after"] = neighbour(
            rows[index + 1] if index + 1 < len(rows) else None
        )
    return rows


def load_pure_requirement_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Download (once) and parse the PURE XML documents named in the config."""
    datasets_config = config.get("datasets", {}) if isinstance(config, Mapping) else {}
    defaults = DEFAULT_CONFIG["datasets"]
    zip_path = project_root() / str(
        datasets_config.get("pure_local_zip", defaults["pure_local_zip"])
    )
    if not zip_path.exists():
        download_file(
            str(datasets_config.get("pure_xml_url", defaults["pure_xml_url"])),
            zip_path,
        )
    members = datasets_config.get("pure_documents", defaults["pure_documents"])
    if isinstance(members, str):
        members = [members]
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in members:
            rows.extend(
                parse_pure_document(
                    archive.read(str(member)), _pure_document_id(member)
                )
            )
    return rows


def pure_filter(
    requirement: str, capability: str, marker: str, marker_count: int
) -> tuple[bool, str]:
    """Seed filter for PURE rows: the mlm_tapt screen plus marker checks."""
    _, reason = mlm_tapt_filter(
        requirement, capability, source_corpus="", exclude_source_regex=""
    )
    reasons = [item for item in reason.split(";") if item]
    if marker_count == 0 or not marker:
        reasons.append("no_marker")
    elif marker_count > 1:
        reasons.append("multiple_markers")
    if marker == "I":
        reasons.append("informative_marker")
    if PURE_IMPERSONAL_RE.search(requirement):
        reasons.append("impersonal_construction")
    unique_reasons = list(dict.fromkeys(reasons))
    return not unique_reasons, ";".join(unique_reasons)


def make_pure_seed_candidates(
    document_rows: list[dict[str, Any]],
    target_count: int = DEFAULT_CONFIG["project"]["target_seed_count"],
    seed: int = DEFAULT_CONFIG["project"]["seed"],
) -> list[dict[str, Any]]:
    """Build the reviewable seed table for the `pure` dataset.

    Every eligible optional (O) requirement is included, because that stratum
    is small and is the one the context ablation reports on; the remainder is
    filled with mandatory (M) requirements sampled deterministically, in
    proportion to each document's eligible M pool.
    """
    seen: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for row in document_rows:
        original = normalize_space(str(row.get("text", "")))
        if not original:
            continue
        dedupe_key = original.lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        capability = auto_capability_text(original)
        marker = str(row.get("marker", ""))
        auto_include, reason = pure_filter(
            original, capability, marker, int(row.get("marker_count", 0))
        )
        candidates.append(
            {
                "seed_id": f"S{len(candidates) + 1:04d}",
                "source_dataset": SOURCE_DATASET_LABELS[DATASET_PURE],
                "source_corpus": str(row.get("document_id", "")),
                "original_requirement": original,
                "capability_text_auto": capability,
                "auto_include": "yes" if auto_include else "no",
                "auto_exclusion_reason": reason,
                "include": "no",
                "exclusion_reason": reason,
                "capability_text_final": "",
                "context_document": str(row.get("document_title", "")),
                "context_requirement_id": str(row.get("requirement_id", "")),
                "context_marker": marker,
                "context_section": str(row.get("section_path", "")),
                "context_before": str(row.get("neighbour_before", "")),
                "context_after": str(row.get("neighbour_after", "")),
                "context_legend": str(row.get("document_legend", PURE_MARKER_LEGEND)),
            }
        )

    eligible = [i for i, c in enumerate(candidates) if is_truthy(c["auto_include"])]
    optional = [i for i in eligible if candidates[i]["context_marker"] == "O"]
    mandatory_by_document: dict[str, list[int]] = {}
    for index in eligible:
        if candidates[index]["context_marker"] == "M":
            mandatory_by_document.setdefault(
                candidates[index]["source_corpus"], []
            ).append(index)
    fill = target_count - len(optional)
    mandatory_total = sum(len(pool) for pool in mandatory_by_document.values())
    if fill < 0 or mandatory_total < fill:
        raise ValueError(
            f"Cannot select {target_count} PURE seeds: {len(optional)} optional and "
            f"{mandatory_total} mandatory candidates are eligible."
        )
    rng = random.Random(seed)
    selected = set(optional)
    documents = sorted(mandatory_by_document)
    quotas = {
        doc: int(fill * len(mandatory_by_document[doc]) / mandatory_total)
        for doc in documents
    }
    # Largest-remainder rounding so the quotas sum to `fill` exactly.
    for doc in sorted(
        documents,
        key=lambda d: -(fill * len(mandatory_by_document[d]) / mandatory_total % 1),
    )[: fill - sum(quotas.values())]:
        quotas[doc] += 1
    for doc in documents:
        selected.update(rng.sample(sorted(mandatory_by_document[doc]), quotas[doc]))

    for index, candidate in enumerate(candidates):
        if index in selected:
            candidate["include"] = "yes"
            candidate["exclusion_reason"] = ""
            candidate["capability_text_final"] = candidate["capability_text_auto"]
        elif is_truthy(candidate["auto_include"]):
            candidate["exclusion_reason"] = "not_sampled_mandatory_pool"
    return candidates


def _review_compare_text(value: Any) -> str:
    return strip_final_punctuation(str(value)).lower()


def refresh_capability_suggestions(
    rows: list[dict[str, Any]], force: bool = False
) -> tuple[list[dict[str, Any]], int]:
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
        if (
            is_truthy(row.get("include", ""))
            and new_auto
            and (force or final_is_unedited)
        ):
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
        capability = strip_final_punctuation(
            row.get("capability_text_final") or row.get("capability_text_auto") or ""
        )
        if not capability:
            continue
        row = dict(row)
        row["capability_text_final"] = capability
        selected.append(row)
    if strict and len(selected) != target_count:
        raise ValueError(
            f"Expected exactly {target_count} included seeds, found {len(selected)}."
        )
    return selected


# =============================================================================
# Section 4: Capability review and modality-benchmark construction
# =============================================================================
# Manual-review tables for included capabilities, the controlled MUST /
# SHOULD / MAY / nice_to_have variant generators, weak-modality probe and
# Task 3 verifier item builders, and the benchmark-statement review export.


def included_capability_review_frame(path: str | Path) -> pd.DataFrame:
    frame = read_csv_frame(path)
    for column in [
        "seed_id",
        "original_requirement",
        "capability_text_final",
        "include",
    ]:
        if column not in frame.columns:
            raise ValueError(f"Missing required review column: {column}")
    included = frame[frame["include"].map(is_truthy)].copy()
    included["capability_text_final"] = included["capability_text_final"].map(
        strip_final_punctuation
    )
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


def write_included_capability_review(
    path: str | Path, output_dir: str | Path, suffix: str = ""
) -> dict[str, Path]:
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


def source_statement(
    capability: str, modality: str, mandatory_keyword: str = "MUST"
) -> str:
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
            source = template["source_template"].format(
                capability=capability_clause(capability)
            )
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
    audit_mode: str = OFFICIAL_TASK3_AUDIT_MODE,
) -> list[dict[str, Any]]:
    audit_mode = normalize_task3_audit_mode(audit_mode)
    benchmark_by_item = {row["item_id"]: row for row in benchmark_rows}
    # Append-only raw files can hold several rows per planned request; keep one.
    task2_raw_rows = dedupe_raw_rows(task2_raw_rows)
    task2_raw_rows = filter_raw_rows_to_current_benchmark(
        benchmark_rows, task2_raw_rows
    )
    items: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in task2_raw_rows:
        if raw.get("task") != "task2" or raw.get("sample_kind") != "deterministic":
            continue
        if raw.get("parse_status") != "ok" or not isinstance(
            raw.get("parsed_json"), dict
        ):
            continue
        source_item = benchmark_by_item.get(str(raw.get("item_id", "")))
        if not source_item:
            continue
        parsed = raw["parsed_json"]
        extracted_modality = normalize_modality(parsed.get("modality"))
        if extracted_modality is None:
            continue
        text_diagnostic = requirement_text_modality_diagnostic(
            parsed.get("requirement", "")
        )
        text_modality = normalize_modality(text_diagnostic["text_modality"])
        if text_modality is None:
            continue
        text_parse_status = "ok"
        model = str(raw.get("model", ""))
        source_item_id = str(source_item["item_id"])
        dedupe_key = (model, source_item_id)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        declared_relation = task3_gold_relation(
            source_item["source_modality"], extracted_modality
        )
        relation = task3_gold_relation(source_item["source_modality"], text_modality)
        confidence = confidence_probability(raw, parsed)
        items.append(
            {
                "item_id": f"{source_item_id}__task3__{safe_identifier(model)}__{safe_identifier(audit_mode)}",
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
                "task2_text_modality": text_modality,
                "task2_text_modality_basis": text_diagnostic["text_modality_basis"],
                "task2_text_modality_parse_status": text_parse_status,
                "task2_confidence": "" if confidence is None else confidence,
                "task3_declared_relation": declared_relation,
                "task3_gold_relation": relation,
                "task3_audit_mode": audit_mode,
                "ordinal_strength": int(source_item["ordinal_strength"]),
                "numeric_strength": float(source_item["numeric_strength"]),
            }
        )
    return items


def task3_items_from_raw_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    required_fields = {
        "item_id",
        "source_item_id",
        "seed_id",
        "source_modality",
        "source_statement",
        "task2_run_id",
        "task2_model",
        "task2_requirement",
        "task3_gold_relation",
        "task3_audit_mode",
        "ordinal_strength",
        "numeric_strength",
    }
    for raw in raw_rows:
        if raw.get("task") != "task3":
            continue
        item_id = str(raw.get("item_id", ""))
        if not item_id or item_id in seen:
            continue
        if any(str(raw.get(field, "")).strip() == "" for field in required_fields):
            continue
        seen.add(item_id)
        item = {field: raw.get(field, "") for field in TASK3_VERIFICATION_FIELDS}
        items.append(item)
    return items


def weak_modality_template_sanity_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for template in WEAK_MODALITY_PROBE_TEMPLATES:
        rows.append(
            {
                "template_id": template["template_id"],
                "source_statement_template": template["source_template"],
                "example_source_statement": template["source_template"].format(
                    capability="export reports"
                ),
                "intended_gold_modality": "nice_to_have",
                "weaker_than_should": "",
                "reviewer": "",
                "review_note": "",
            }
        )
    return rows


def write_weak_modality_template_sanity_check(
    output_dir: str | Path, suffix: str = ""
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"weak_modality_template_sanity_check{suffix}.csv"
    markdown_path = output_dir / f"weak_modality_template_sanity_check{suffix}.md"
    if csv_path.exists():
        rows = read_csv_rows(csv_path)
    else:
        rows = weak_modality_template_sanity_rows()
        write_csv_rows(csv_path, rows, fieldnames=WEAK_MODALITY_SANITY_FIELDS)
    markdown_path.write_text(
        markdown_table(rows, WEAK_MODALITY_SANITY_FIELDS) + "\n", encoding="utf-8"
    )
    return {"csv": csv_path, "markdown": markdown_path}


MAIN_MODALITY_TEMPLATE_INVENTORY_FIELDS = [
    "template_id",
    "condition",
    "variant",
    "source_statement_template",
    "example_source_statement",
    "intended_gold_modality",
    "note",
]
MODALITY_TEMPLATE_INVENTORY_EXAMPLE_CAPABILITY = "export reports"


def main_modality_template_rows(
    capability: str = MODALITY_TEMPLATE_INVENTORY_EXAMPLE_CAPABILITY,
) -> list[dict[str, Any]]:
    """Return the full modality-template inventory used to build benchmark sources.

    Covers the four main benchmark conditions (``MUST`` wording), the ``SHALL``
    robustness variant of the mandatory condition, and the four weak-intent
    phrasing-probe templates. Mirrors ``weak_modality_template_sanity_rows`` and
    is exported for the experimental-setup documentation.
    """
    cap = capability_clause(capability)
    rows: list[dict[str, Any]] = []
    main_conditions = [
        (
            "main_mandatory_must",
            "mandatory",
            "must",
            "Main benchmark mandatory condition.",
        ),
        (
            "main_recommended_should",
            "recommended",
            "must",
            "Main benchmark recommended condition.",
        ),
        ("main_optional_may", "optional", "must", "Main benchmark optional condition."),
        (
            "main_nice_to_have_useful_if",
            "nice_to_have",
            "must",
            "Main benchmark weak stakeholder-intent condition.",
        ),
    ]
    for template_id, condition, variant, note in main_conditions:
        rows.append(
            {
                "template_id": template_id,
                "condition": condition,
                "variant": variant,
                "source_statement_template": source_statement(
                    "{capability}", condition
                ),
                "example_source_statement": source_statement(cap, condition),
                "intended_gold_modality": condition,
                "note": note,
            }
        )
    rows.append(
        {
            "template_id": "shall_mandatory_shall",
            "condition": "mandatory",
            "variant": "shall",
            "source_statement_template": source_statement(
                "{capability}", "mandatory", mandatory_keyword="SHALL"
            ),
            "example_source_statement": source_statement(
                cap, "mandatory", mandatory_keyword="SHALL"
            ),
            "intended_gold_modality": "mandatory",
            "note": "SHALL robustness variant; swaps MUST in the mandatory condition only.",
        }
    )
    for template in WEAK_MODALITY_PROBE_TEMPLATES:
        rows.append(
            {
                "template_id": f"probe_{template['template_id']}",
                "condition": "nice_to_have",
                "variant": "weak_probe",
                "source_statement_template": template["source_template"],
                "example_source_statement": template["source_template"].format(
                    capability=cap
                ),
                "intended_gold_modality": "nice_to_have",
                "note": (
                    "Weak-intent phrasing probe; identical to the main nice_to_have template."
                    if template["source_template"]
                    == source_statement("{capability}", "nice_to_have")
                    else "Weak-intent phrasing probe."
                ),
            }
        )
    return rows


def write_main_modality_template_inventory(
    path: str | Path,
    capability: str = MODALITY_TEMPLATE_INVENTORY_EXAMPLE_CAPABILITY,
) -> dict[str, Path]:
    """Write the modality-template inventory as a CSV plus a sibling Markdown table."""
    csv_path = Path(path)
    markdown_path = csv_path.with_suffix(".md")
    rows = main_modality_template_rows(capability)
    write_csv_rows(csv_path, rows, fieldnames=MAIN_MODALITY_TEMPLATE_INVENTORY_FIELDS)
    markdown_path.write_text(
        markdown_table(rows, MAIN_MODALITY_TEMPLATE_INVENTORY_FIELDS) + "\n",
        encoding="utf-8",
    )
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
        answers = [
            _sanity_answer(value)
            for value in template_rows["weaker_than_should"].tolist()
        ]
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


def weak_modality_construct_review_rows(
    reviewer_ids: Iterable[str] = ("R1", "R2"),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for reviewer_id in reviewer_ids:
        for template in WEAK_MODALITY_PROBE_TEMPLATES:
            rows.append(
                {
                    "reviewer_id": reviewer_id,
                    "reviewer_role": "",
                    "template_id": template["template_id"],
                    "source_statement_template": template["source_template"],
                    "example_source_statement": template["source_template"].format(
                        capability="export reports"
                    ),
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

        answers = [
            _sanity_answer(value)
            for value in template_rows["weaker_than_should"].tolist()
        ]
        if "no" in answers:
            disagreeing.append(template_id)
        if len(answers) < expected_reviewers_per_template or any(
            answer != "yes" for answer in answers
        ):
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
    *,
    passthrough_fields: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Render seed x modality items; `passthrough_fields` copies extra seed columns."""
    numeric_strength = numeric_strength or NUMERIC_STRENGTH_DEFAULT
    passthrough = list(passthrough_fields)
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
                    "source_statement": source_statement(
                        capability, modality, mandatory_keyword=mandatory_keyword
                    ),
                    "candidate_requirement": candidate_requirement(
                        capability, mandatory_keyword=mandatory_keyword
                    ),
                    "mandatory_keyword": mandatory_keyword,
                    "task1_gold_decision": "yes" if modality == "mandatory" else "no",
                    "task1_gold_yes": 1 if modality == "mandatory" else 0,
                    "task2_gold_modality": modality,
                    "ordinal_strength": ORDINAL_STRENGTH[modality],
                    "numeric_strength": numeric_strength[modality],
                    **{field: seed.get(field, "") for field in passthrough},
                }
            )
    return items


def benchmark_statement_review_frame(
    items: list[dict[str, Any]] | pd.DataFrame | str | Path,
) -> pd.DataFrame:
    if isinstance(items, (str, Path)):
        frame = read_csv_frame(items)
    elif isinstance(items, pd.DataFrame):
        frame = items.copy()
    else:
        frame = pd.DataFrame.from_records(items)
    required = [
        "seed_id",
        "capability_text",
        "candidate_requirement",
        "source_modality",
        "source_statement",
    ]
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


# =============================================================================
# Section 5: Prompts, request planning, and JSON-schema response formats
# =============================================================================
# Frozen prompt loading and rendering, per-task JSON schemas, structured
# response-format objects for json_object / json_schema / instructor modes,
# and the planner that turns benchmark rows into provider completion jobs.

# Version of the compute_job_config_sha input schema. Bump whenever the set of
# hashed request parameters changes; recorded on every raw record so resume and
# provenance tooling can tell which fingerprint definition produced a row.
JOB_CONFIG_SHA_VERSION = 3


def load_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def render_prompt(template: str, **values: Any) -> str:
    return template.format(**values)


def document_context_text(item: Mapping[str, Any]) -> str:
    """Render the document context of a `pure` item for the `document` arm.

    One renderer serves both the single-item template (`context_block`) and the
    per-item `context` value of the batched wrapper, so the two never drift.
    The source statement itself is never repeated here.
    """
    marker = str(item.get("context_marker", "")).strip()
    legend = str(item.get("context_legend", "")).strip()
    document = str(item.get("context_document", "")).strip()
    lines = [f"Document: {document}" + (f" (markers: {legend})" if legend else "")]
    section = str(item.get("context_section", "")).strip()
    if section:
        lines.append(f"Section: {section}")
    before = str(item.get("context_before", "")).strip()
    if before:
        lines.append(f"Preceding requirement {before}")
    requirement_id = str(item.get("context_requirement_id", "")).strip()
    this_line = f"This requirement: {requirement_id}" if requirement_id else ""
    if marker:
        this_line = (this_line or "This requirement:") + f", marker ({marker})"
    if this_line:
        lines.append(this_line)
    after = str(item.get("context_after", "")).strip()
    if after:
        lines.append(f"Following requirement {after}")
    return "\n".join(lines)


def prompt_for_benchmark_task(
    task: str,
    item: Mapping[str, Any],
    task1_template: str,
    task2_template: str,
    *,
    item_context: str = DEFAULT_ITEM_CONTEXT,
    task2_context_template: str | None = None,
) -> str:
    if task == "task1":
        return render_prompt(
            task1_template,
            source_statement=item["source_statement"],
            candidate_requirement=item["candidate_requirement"],
        )
    if task == "task2":
        if normalize_item_context(item_context) == ITEM_CONTEXT_DOCUMENT:
            if not task2_context_template:
                raise ValueError(
                    "item_context=document needs a task2_context_template."
                )
            return render_prompt(
                task2_context_template,
                source_statement=item["source_statement"],
                context_block=document_context_text(item),
            )
        return render_prompt(task2_template, source_statement=item["source_statement"])
    raise ValueError(f"Unsupported benchmark task: {task}")


def _json_schema_object(
    properties: Mapping[str, Any], required: list[str]
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
    }


def task_response_schema(task: str, *, batched: bool = False) -> dict[str, Any]:
    confidence = {"type": "number", "minimum": 0, "maximum": 1}
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
            "modality": {"type": "string", "enum": list(MODALITIES)},
            "confidence": confidence,
        }
        required = ["requirement", "modality", "confidence"]
    elif task == "task3":
        properties = {
            "relation": {"type": "string", "enum": list(TASK3_RELATIONS)},
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
            {
                "results": {
                    "type": "array",
                    "items": _json_schema_object(properties, required),
                }
            },
            ["results"],
        )
    return _json_schema_object(properties, required)


def response_format_for_task(
    task: str, structured_output: str, *, batched: bool = False
) -> dict[str, Any] | None:
    mode = normalize_structured_output_mode(structured_output)
    if mode in {"none", "instructor"}:
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
    if mode == "instructor":
        if extra is not None and "response_format" in extra:
            extra = {
                key: value for key, value in extra.items() if key != "response_format"
            }
        return None, extra
    if mode == "none":
        return (dict(response_format) if response_format else None), extra

    resolved = response_format_for_task(task, mode, batched=batched)
    if extra is not None and "response_format" in extra:
        extra["response_format"] = resolved
        return None, extra
    return resolved, extra


def compute_job_config_sha(
    *,
    prompt: str,
    prompt_version: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    structured_output: str,
    json_mode: bool,
    extra_body: Mapping[str, Any] | None = None,
    response_format: Mapping[str, Any] | None = None,
    instructor_mode: str = "json",
    validation_retries: int = 2,
    seed: int | None = DEFAULT_REQUEST_SEED,
    task: str = "",
    batch_size: int = 1,
    batch_order: str = DEFAULT_BATCH_ORDER,
    fallback_batch_size: int = 1,
    item_context: str = DEFAULT_ITEM_CONTEXT,
) -> str:
    """Config fingerprint for a single completion job.

    Covers the request parameters that change model output but are not part of
    completion_record_key, so config-aware resume can detect stale cached rows.

    Version 2 (see JOB_CONFIG_SHA_VERSION) additionally hashes extra_body,
    response_format, instructor_mode, validation_retries, and seed.

    Version 3 additionally hashes the batching setup: batch_size, batch_order,
    fallback_batch_size, and -- only when batch_size > 1 -- the SHA of the batch
    prompt wrapper for `task` (see :func:`batch_prompt_wrapper_sha`). A batched
    request sees a different prompt envelope and neighbouring items than a
    single-item request, so a cached row produced under a different batching
    setup is not interchangeable with the planned one. Single-item plans
    (batch_size == 1) carry no wrapper hash, so editing the batch wrapper never
    invalidates an unbatched run.

    Resume impact: `pending_completion_jobs` (Section 9) reuses a cached row when
    it carries no job_config_sha at all (legacy cache, key-only match) but
    re-requests any row whose recorded sha differs from the planned one. Because
    v2 hashes more inputs, rows written under the v1 definition carry an OLD (not
    missing) sha and will therefore be re-requested on resume. That is intended:
    a v1 sha cannot prove the extra_body/response_format/seed of the cached row
    matches the current plan. To keep an old run resumable without re-requesting,
    replay it with the v1 fingerprint or accept the re-request cost. The same
    applies to v2 rows under the v3 definition.

    `item_context` (the document-context ablation) enters the canonical form
    only when it is not `bare`, so every fingerprint written before the knob
    existed stays valid and no version bump is needed; the `document` arm
    also hashes its own batch wrapper.
    """
    resolved_batch_size = positive_int(batch_size, "batch_size")
    resolved_item_context = normalize_item_context(item_context)
    canonical = compact_json(
        {
            "sha_version": JOB_CONFIG_SHA_VERSION,
            "prompt": prompt,
            "prompt_version": prompt_version,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
            "structured_output": normalize_structured_output_mode(
                structured_output, json_mode=json_mode
            ),
            "json_mode": bool(json_mode),
            "extra_body": dict(extra_body) if extra_body else None,
            "response_format": dict(response_format) if response_format else None,
            "instructor_mode": normalize_instructor_mode_name(instructor_mode),
            "validation_retries": nonnegative_int(
                validation_retries, "validation_retries"
            ),
            "seed": None if seed is None else int(seed),
            "batch_size": resolved_batch_size,
            "batch_order": normalize_batch_order(batch_order),
            "fallback_batch_size": positive_int(
                fallback_batch_size, "fallback_batch_size"
            ),
            # Only batched plans depend on the wrapper text, so single-item runs
            # keep a fingerprint that is stable across wrapper edits.
            **(
                {
                    "batch_wrapper_sha": batch_prompt_wrapper_sha(
                        task, resolved_item_context
                    )
                }
                if resolved_batch_size > 1
                else {}
            ),
            **(
                {"item_context": resolved_item_context}
                if resolved_item_context != DEFAULT_ITEM_CONTEXT
                else {}
            ),
        }
    )
    return sha256_text(canonical)


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
    instructor_mode: str = "json",
    validation_retries: int = 2,
    fallback_batch_size: int = 1,
    seed: int = DEFAULT_REQUEST_SEED,
    send_seed: bool = True,
    max_retries: int = DEFAULT_MAX_RETRIES,
    batch_order: str = DEFAULT_BATCH_ORDER,
    batch_size: int = 1,
    server_model_probe: Mapping[str, Any] | str | None = None,
    item_context: str = DEFAULT_ITEM_CONTEXT,
) -> dict[str, Any]:
    resolved_response_format, resolved_extra_body = resolve_response_format_args(
        task,
        structured_output=structured_output,
        json_mode=json_mode,
        response_format=response_format,
        extra_body=extra_body,
        batched=False,
    )
    resolved_seed = int(seed) if send_seed else None
    return {
        "request_index": request_index,
        "run_id": run_id,
        "job_config_sha": compute_job_config_sha(
            prompt=prompt,
            prompt_version=prompt_version,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            structured_output=structured_output,
            json_mode=json_mode,
            extra_body=resolved_extra_body,
            response_format=resolved_response_format,
            instructor_mode=instructor_mode,
            validation_retries=validation_retries,
            seed=resolved_seed,
            task=task,
            batch_size=batch_size,
            batch_order=batch_order,
            fallback_batch_size=fallback_batch_size,
            item_context=item_context,
        ),
        "job_config_sha_version": JOB_CONFIG_SHA_VERSION,
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
        "structured_output": normalize_structured_output_mode(
            structured_output, json_mode=json_mode
        ),
        "response_format": dict(resolved_response_format)
        if resolved_response_format
        else None,
        "extra_body": dict(resolved_extra_body) if resolved_extra_body else None,
        "instructor_mode": normalize_instructor_mode_name(instructor_mode),
        "validation_retries": nonnegative_int(validation_retries, "validation_retries"),
        "fallback_batch_size": positive_int(fallback_batch_size, "fallback_batch_size"),
        "seed": resolved_seed,
        "send_seed": bool(send_seed),
        "max_retries": nonnegative_int(max_retries, "max_retries"),
        "batch_order": normalize_batch_order(batch_order),
        "batch_size": positive_int(batch_size, "batch_size"),
        "server_model_probe": server_model_probe,
        "item_context": normalize_item_context(item_context),
    }


def _append_completion_jobs(
    jobs: list[dict[str, Any]],
    *,
    item: Mapping[str, Any],
    task: str,
    prompt: str,
    prompt_version: str,
    model: str,
    host: str,
    run_id: str,
    deterministic: Mapping[str, Any],
    stochastic: Mapping[str, Any],
    max_tokens: int,
    timeout_s: int,
    api_key_env: str,
    provider_id: str,
    profile_id: str,
    run_group_id: str,
    json_mode: bool,
    structured_output: str | None,
    response_format: Mapping[str, Any] | None,
    extra_body: Mapping[str, Any] | None,
    instructor_mode: str,
    validation_retries: int,
    fallback_batch_size: int,
    seed: int,
    send_seed: bool,
    max_retries: int,
    batch_order: str,
    batch_size: int,
    server_model_probe: Mapping[str, Any] | str | None,
    item_context: str = DEFAULT_ITEM_CONTEXT,
) -> None:
    """Append the deterministic job and any stochastic samples for one rendered prompt."""
    shared = {
        "item": item,
        "task": task,
        "model": model,
        "host": host,
        "run_id": run_id,
        "prompt": prompt,
        "prompt_version": prompt_version,
        "max_tokens": max_tokens,
        "timeout_s": timeout_s,
        "api_key_env": api_key_env,
        "provider_id": provider_id,
        "profile_id": profile_id,
        "run_group_id": run_group_id,
        "json_mode": json_mode,
        "structured_output": structured_output,
        "response_format": response_format,
        "extra_body": extra_body,
        "instructor_mode": instructor_mode,
        "validation_retries": validation_retries,
        "fallback_batch_size": fallback_batch_size,
        "seed": seed,
        "send_seed": send_seed,
        "max_retries": max_retries,
        "batch_order": batch_order,
        "batch_size": batch_size,
        "server_model_probe": server_model_probe,
        "item_context": item_context,
    }
    jobs.append(
        completion_request_job(
            sample_kind="deterministic",
            sample_index=0,
            temperature=float(deterministic["temperature"]),
            top_p=float(deterministic["top_p"]),
            request_index=len(jobs),
            **shared,
        )
    )
    for sample_index in range(max(0, int(stochastic.get("samples", 0)))):
        jobs.append(
            completion_request_job(
                sample_kind="stochastic",
                sample_index=sample_index,
                temperature=float(stochastic["temperature"]),
                top_p=float(stochastic["top_p"]),
                request_index=len(jobs),
                **shared,
            )
        )


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
    instructor_mode: str = "json",
    validation_retries: int = 2,
    fallback_batch_size: int = 1,
    seed: int = DEFAULT_REQUEST_SEED,
    send_seed: bool = True,
    max_retries: int = DEFAULT_MAX_RETRIES,
    batch_order: str = DEFAULT_BATCH_ORDER,
    batch_size: int = 1,
    server_model_probe: Mapping[str, Any] | str | None = None,
    item_context: str = DEFAULT_ITEM_CONTEXT,
    task2_context_template: str | None = None,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    task_list = normalize_task_filter(tasks)
    resolved_item_context = normalize_item_context(item_context)
    for item in benchmark_rows:
        for task in task_list:
            _append_completion_jobs(
                jobs,
                item=item,
                task=task,
                prompt=prompt_for_benchmark_task(
                    task,
                    item,
                    task1_template,
                    task2_template,
                    item_context=resolved_item_context,
                    task2_context_template=task2_context_template,
                ),
                prompt_version=prompt_version,
                model=model,
                host=host,
                run_id=run_id,
                deterministic=deterministic,
                stochastic=stochastic,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
                api_key_env=api_key_env,
                provider_id=provider_id,
                profile_id=profile_id,
                run_group_id=run_group_id,
                json_mode=json_mode,
                structured_output=structured_output,
                response_format=response_format,
                extra_body=extra_body,
                instructor_mode=instructor_mode,
                validation_retries=validation_retries,
                fallback_batch_size=fallback_batch_size,
                seed=seed,
                send_seed=send_seed,
                max_retries=max_retries,
                batch_order=batch_order,
                batch_size=batch_size,
                server_model_probe=server_model_probe,
                item_context=resolved_item_context,
            )
    return jobs


def planned_completion_jobs_for_items(
    items: list[dict[str, Any]],
    *,
    prompt_fn: Callable[[Mapping[str, Any]], str],
    prompt_version: str,
    model: str,
    host: str,
    run_id: str,
    deterministic: Mapping[str, Any],
    stochastic: Mapping[str, Any],
    max_tokens: int,
    timeout_s: int,
    api_key_env: str,
    task: str = "task3",
    provider_id: str = "",
    profile_id: str = "",
    run_group_id: str = "",
    json_mode: bool = False,
    structured_output: str | None = None,
    response_format: Mapping[str, Any] | None = None,
    extra_body: Mapping[str, Any] | None = None,
    instructor_mode: str = "json",
    validation_retries: int = 2,
    fallback_batch_size: int = 1,
    seed: int = DEFAULT_REQUEST_SEED,
    send_seed: bool = True,
    max_retries: int = DEFAULT_MAX_RETRIES,
    batch_order: str = DEFAULT_BATCH_ORDER,
    batch_size: int = 1,
    server_model_probe: Mapping[str, Any] | str | None = None,
) -> list[dict[str, Any]]:
    """Plan deterministic + stochastic jobs for pre-built items with a per-item prompt.

    The Task 3 runner renders one prompt per audit item (rather than per task as in
    planned_completion_jobs); prompt_fn maps each item to that rendered prompt.
    """
    jobs: list[dict[str, Any]] = []
    for item in items:
        _append_completion_jobs(
            jobs,
            item=item,
            task=task,
            prompt=prompt_fn(item),
            prompt_version=prompt_version,
            model=model,
            host=host,
            run_id=run_id,
            deterministic=deterministic,
            stochastic=stochastic,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            api_key_env=api_key_env,
            provider_id=provider_id,
            profile_id=profile_id,
            run_group_id=run_group_id,
            json_mode=json_mode,
            structured_output=structured_output,
            response_format=response_format,
            extra_body=extra_body,
            instructor_mode=instructor_mode,
            validation_retries=validation_retries,
            fallback_batch_size=fallback_batch_size,
            seed=seed,
            send_seed=send_seed,
            max_retries=max_retries,
            batch_order=batch_order,
            batch_size=batch_size,
            server_model_probe=server_model_probe,
        )
    return jobs


# =============================================================================
# Section 6: Response parsing, confidence handling, and modality normalization
# =============================================================================
# Tolerant JSON object extraction, confidence-scale detection and parsing for
# the v2 0-1 contract, decision/modality/relation normalization, gold-relation
# computation for Task 3, and the per-task response parser used downstream.


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


def normalize_confidence_scale(
    value: Any, default: str = CONFIDENCE_SCALE_0_100
) -> str:
    if value in {"", None}:
        return default
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "0_1": CONFIDENCE_SCALE_0_1,
        "01": CONFIDENCE_SCALE_0_1,
        "probability": CONFIDENCE_SCALE_0_1,
        "prob": CONFIDENCE_SCALE_0_1,
        "decimal": CONFIDENCE_SCALE_0_1,
        "0_100": CONFIDENCE_SCALE_0_100,
        "0100": CONFIDENCE_SCALE_0_100,
        "percent": CONFIDENCE_SCALE_0_100,
        "percentage": CONFIDENCE_SCALE_0_100,
        "legacy": CONFIDENCE_SCALE_0_100,
    }
    if normalized not in aliases:
        raise ValueError(f"Unknown confidence scale: {value}")
    return aliases[normalized]


def prompt_version_uses_confidence_0_1(prompt_version: Any) -> bool:
    normalized = str(prompt_version or "").strip().lower()
    return normalized in CONFIDENCE_0_1_PROMPT_VERSIONS or normalized.startswith("v2-")


def prompt_text_uses_confidence_0_1(prompt: Any) -> bool:
    text = normalize_space(str(prompt or "")).lower()
    if not text:
        return False
    probability_markers = (
        "0.0-1.0",
        "0.0 to 1.0",
        "0.0 and 1.0",
        "decimal probability",
    )
    return any(marker in text for marker in probability_markers)


def output_contract_uses_confidence_0_1(output_contract_version: Any) -> bool:
    return (
        str(output_contract_version or "") in so.CONFIDENCE_0_1_OUTPUT_CONTRACT_VERSIONS
    )


def confidence_scale_for_record(record: Mapping[str, Any]) -> str:
    explicit = str(record.get("confidence_scale", "")).strip()
    if explicit:
        return normalize_confidence_scale(explicit)
    if output_contract_uses_confidence_0_1(record.get("output_contract_version")):
        return CONFIDENCE_SCALE_0_1
    if prompt_version_uses_confidence_0_1(record.get("prompt_version")):
        return CONFIDENCE_SCALE_0_1
    if prompt_text_uses_confidence_0_1(record.get("prompt")):
        return CONFIDENCE_SCALE_0_1
    return CONFIDENCE_SCALE_0_100


def parse_confidence(
    value: Any, confidence_scale: str = CONFIDENCE_SCALE_0_100
) -> float | None:
    scale = normalize_confidence_scale(confidence_scale)
    if isinstance(value, bool) or (
        scale == CONFIDENCE_SCALE_0_1 and isinstance(value, str)
    ):
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if scale == CONFIDENCE_SCALE_0_1 and 0.0 <= confidence <= 1.0:
        return confidence
    if scale == CONFIDENCE_SCALE_0_100 and 0.0 <= confidence <= 100.0:
        return confidence
    return None


def row_uses_confidence_0_1(row: Mapping[str, Any]) -> bool:
    return confidence_scale_for_record(row) == CONFIDENCE_SCALE_0_1


def confidence_probability(
    row_or_parsed: Mapping[str, Any], parsed: Mapping[str, Any] | None = None
) -> float:
    if parsed is None and isinstance(row_or_parsed.get("parsed_json"), Mapping):
        parsed = row_or_parsed["parsed_json"]  # type: ignore[index]
    value_source = parsed if parsed is not None else row_or_parsed
    confidence = float(value_source.get("confidence", 0.0))
    return confidence if row_uses_confidence_0_1(row_or_parsed) else confidence / 100.0


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


def safe_identifier(value: Any, fallback: str = "value") -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_").lower()
    return text or fallback


def evidence_phrase_in_source(evidence_phrase: Any, source_statement_text: Any) -> bool:
    evidence = normalize_space(str(evidence_phrase or "")).lower()
    source = normalize_space(str(source_statement_text or "")).lower()
    return bool(evidence and evidence in source)


def raw_record_matches_benchmark_item(
    raw: Mapping[str, Any], item: Mapping[str, Any]
) -> bool:
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


def dedupe_raw_rows(raw_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Collapse duplicate completions of the same request to one row (latest ok wins).

    A raw JSONL file is append-only, so a resumed or retried run can hold several
    rows for one ``run_id`` + :func:`completion_record_key`. Scoring and progress
    must count each planned request exactly once; this keeps the LAST row in file
    order whose ``parse_status`` is ``ok`` (a later successful retry supersedes an
    earlier failure), or the last row of the group when none parsed. The relative
    order of the surviving rows follows the first appearance of each key so
    downstream ordering (and grouping) stays stable.
    """
    rows = [dict(row) for row in raw_rows]
    first_position: dict[tuple[Any, ...], int] = {}
    keep_index: dict[tuple[Any, ...], int] = {}
    for index, row in enumerate(rows):
        key = (str(row.get("run_id", "")), completion_record_key(row))
        first_position.setdefault(key, index)
        previous = keep_index.get(key)
        is_ok = str(row.get("parse_status", "")) == "ok"
        if (
            previous is None
            or is_ok
            or str(rows[previous].get("parse_status", "")) != "ok"
        ):
            keep_index[key] = index
    kept = sorted(keep_index.items(), key=lambda entry: first_position[entry[0]])
    return [rows[index] for _, index in kept]


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
    return [
        row for row in benchmark_rows if str(row.get("item_id", "")) in fresh_item_ids
    ]


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


# Surface modal cues mapped to the modality category they signal in generated
# requirement text. Used by ``requirement_text_modality_diagnostic``.
MODAL_CUE_CATEGORY = {
    "must": "mandatory",
    "shall": "mandatory",
    "required to": "mandatory",
    "should": "recommended",
    "recommended": "recommended",
    "may": "optional",
    "optional": "optional",
    "could": "optional",
    "can": "optional",
}
# Contractions that are themselves a negated modal cue, mapped to the base cue.
NEGATED_MODAL_CONTRACTIONS = {
    "cannot": "can",
    "can't": "can",
    "cant": "can",
    "mustn't": "must",
    "mustnt": "must",
    "shouldn't": "should",
    "shouldnt": "should",
    "shan't": "shall",
    "shant": "shall",
    "couldn't": "could",
    "couldnt": "could",
    "mayn't": "may",
}
NEGATION_TOKENS = {"not", "never"}
# How many preceding tokens are inspected for a negation cue before a modal.
MODAL_NEGATION_LOOKBACK = 3
# Priority order applied when several distinct modal categories co-occur.
MODAL_CATEGORY_PRIORITY = ["mandatory", "recommended", "optional"]


def _modal_cue_scan(text: str) -> tuple[list[str], list[str], list[str]]:
    """Return ``(cues_found, positive_cues, negated_cues)`` for lowercased ``text``.

    ``cues_found`` is the sorted set of surface modal cues seen anywhere in the
    text. A cue counts as negated when it is a negated contraction, when a
    negation token appears within ``MODAL_NEGATION_LOOKBACK`` preceding tokens,
    or when it is directly followed by ``not`` / ``n't``.
    """
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
    positive: list[str] = []
    negated: list[str] = []
    for index, token in enumerate(tokens):
        base_cue = NEGATED_MODAL_CONTRACTIONS.get(token)
        if base_cue is not None:
            negated.append(base_cue)
            continue
        if token not in MODAL_CUE_CATEGORY:
            continue
        window = tokens[max(0, index - MODAL_NEGATION_LOOKBACK) : index]
        following = tokens[index + 1] if index + 1 < len(tokens) else ""
        is_negated = bool(NEGATION_TOKENS.intersection(window)) or following in {
            "not",
            "n't",
            "never",
        }
        (negated if is_negated else positive).append(token)
    if re.search(r"\brequired\s+to\b", text):
        if re.search(r"\b(?:not|never)\s+required\s+to\b", text):
            negated.append("required to")
        else:
            positive.append("required to")
    cues = sorted(set(positive) | set(negated))
    return cues, positive, negated


def requirement_text_modality_diagnostic(requirement_text: Any) -> dict[str, Any]:
    """Classify the modal force expressed by generated requirement text.

    Returns the backward-compatible ``text_modality`` / ``text_modality_basis``
    pair plus ``text_modality_multi_modal`` and ``text_modality_modals_found``.

    Precedence, applied in this order:

    1. A weak phrase ("would be nice if", ...) wins outright (``nice_to_have``).
    2. A POSITIVE modal cue wins over any negated cue, ranked by
       ``MODAL_CATEGORY_PRIORITY`` (mandatory > recommended > optional). Text
       such as "The system must ensure that users cannot delete records." states
       a mandatory obligation whose *content* is a prohibition; scoring it as
       ``negated`` would drop a real strengthening case. Mixed cues still set
       ``text_modality_multi_modal`` when the categories differ.
    3. Only when there is NO positive cue does a negated modal ("must not",
       "cannot", "shouldn't", ...) resolve to ``text_modality = "negated"`` with
       basis ``negated_modal``, so an explicitly negated obligation is never
       scored as a positive (strengthening) modality.
    """
    text = normalize_space(str(requirement_text or "")).lower()
    if not text:
        return {
            "text_modality": "unknown",
            "text_modality_basis": "unknown",
            "text_modality_multi_modal": False,
            "text_modality_modals_found": [],
        }

    cues, positive_cues, negated_cues = _modal_cue_scan(text)
    categories = {MODAL_CUE_CATEGORY[cue] for cue in cues}
    result = {
        "text_modality_multi_modal": len(categories) > 1,
        "text_modality_modals_found": cues,
    }

    weak_patterns = [
        r"\bwould\s+be\s+nice\s+if\b",
        r"\bwould\s+be\s+useful\s+if\b",
        r"\blow[-\s]+priority\s+enhancement\b",
        r"\bfuture\s+enhancement\b",
        r"\bnice[-\s]+to[-\s]+have\b",
        r"\bwishlist\b",
    ]
    if any(re.search(pattern, text) for pattern in weak_patterns):
        return {
            "text_modality": "nice_to_have",
            "text_modality_basis": "weak_phrase",
            **result,
        }
    positive_categories = {MODAL_CUE_CATEGORY[cue] for cue in positive_cues}
    for category in MODAL_CATEGORY_PRIORITY:
        if category in positive_categories:
            return {
                "text_modality": category,
                "text_modality_basis": "explicit_modal",
                **result,
            }
    if negated_cues:
        return {
            "text_modality": "negated",
            "text_modality_basis": "negated_modal",
            **result,
        }
    if re.match(r"^(?:the\s+)?system\s+\w+", text):
        return {
            "text_modality": "mandatory",
            "text_modality_basis": "heuristic_system_verb",
            **result,
        }
    return {"text_modality": "unknown", "text_modality_basis": "unknown", **result}


def empty_text_modality_fields() -> dict[str, Any]:
    return {
        "text_modality": "",
        "text_modality_basis": "",
        "text_modality_multi_modal": "",
        "text_modality_modals_found": "",
        "text_modality_parse_status": "",
        "text_modality_correct": "",
        "label_text_consistent": "",
        "text_overcommit": "",
        "text_undercommit": "",
        "strict_text_overcommit": "",
        "text_high_conf_overcommit_80": "",
        "text_high_conf_overcommit_90": "",
        "strict_text_high_conf_overcommit_80": "",
        "strict_text_high_conf_overcommit_90": "",
        "label_correct_text_overcommit_80": "",
        "label_correct_text_overcommit_90": "",
    }


def text_modality_fields(
    requirement_text: Any,
    gold_modality: str,
    pred_modality: str,
    confidence: float,
) -> dict[str, Any]:
    diagnostic = requirement_text_modality_diagnostic(requirement_text)
    text_modality = diagnostic["text_modality"]
    text_modality_basis = diagnostic["text_modality_basis"]
    # "negated" and "unknown" are non-modality outcomes: they never count as a
    # parsed modality, never strengthen, and never enter the strict basis.
    parse_ok = text_modality in MODALITIES
    gold_strength = ORDINAL_STRENGTH.get(str(gold_modality))
    text_strength = ORDINAL_STRENGTH.get(text_modality)
    overcommit = bool(
        parse_ok
        and gold_strength is not None
        and text_strength is not None
        and text_strength > gold_strength
    )
    undercommit = bool(
        parse_ok
        and gold_strength is not None
        and text_strength is not None
        and text_strength < gold_strength
    )
    strict_overcommit = bool(
        overcommit and text_modality_basis in STRICT_TEXT_MODALITY_BASES
    )
    label_correct = normalize_modality(pred_modality) == normalize_modality(
        gold_modality
    )
    return {
        "text_modality": text_modality,
        "text_modality_basis": text_modality_basis,
        "text_modality_multi_modal": bool(
            diagnostic.get("text_modality_multi_modal", False)
        ),
        "text_modality_modals_found": "|".join(
            diagnostic.get("text_modality_modals_found", [])
        ),
        "text_modality_parse_status": "ok" if parse_ok else "unknown",
        "text_modality_correct": bool(parse_ok and text_modality == gold_modality),
        "label_text_consistent": bool(parse_ok and text_modality == pred_modality),
        "text_overcommit": overcommit,
        "text_undercommit": undercommit,
        "strict_text_overcommit": strict_overcommit,
        "text_high_conf_overcommit_80": bool(overcommit and confidence >= 0.80),
        "text_high_conf_overcommit_90": bool(overcommit and confidence >= 0.90),
        "strict_text_high_conf_overcommit_80": bool(
            strict_overcommit and confidence >= 0.80
        ),
        "strict_text_high_conf_overcommit_90": bool(
            strict_overcommit and confidence >= 0.90
        ),
        "label_correct_text_overcommit_80": bool(
            label_correct and overcommit and confidence >= 0.80
        ),
        "label_correct_text_overcommit_90": bool(
            label_correct and overcommit and confidence >= 0.90
        ),
    }


def parse_task_response(
    task: str,
    raw_text: str,
    confidence_scale: str = CONFIDENCE_SCALE_0_100,
) -> tuple[dict[str, Any] | None, str]:
    json_text = extract_json_object(raw_text)
    if json_text is None:
        return None, "invalid_json"
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        return None, "invalid_json"

    confidence = parse_confidence(parsed.get("confidence"), confidence_scale)
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


def parse_instructor_task_response(
    task: str, raw_text: str
) -> tuple[dict[str, Any] | None, str]:
    json_text = extract_json_object(raw_text)
    if json_text is None:
        return None, "invalid_json"
    try:
        return so.validated_json_for_task(task, json_text), "ok"
    except Exception:
        return None, "instructor_validation_error"


def batch_prompt_for_completion_jobs(jobs: list[Mapping[str, Any]]) -> str:
    if not jobs:
        raise ValueError("Cannot build a batch prompt for an empty job list.")
    task = str(jobs[0]["task"])
    if any(str(job["task"]) != task for job in jobs):
        raise ValueError("Batch prompts must contain jobs for exactly one task.")
    item_context = normalize_item_context(jobs[0].get("item_context"))
    if any(
        normalize_item_context(job.get("item_context")) != item_context for job in jobs
    ):
        raise ValueError(
            "Batch prompts must contain jobs for exactly one item_context."
        )

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
            "Use confidence as a decimal from 0.0 to 1.0 for confidence in the selected decision.\n"
            'Do not return percentages such as 95 or strings such as "95%".\n'
            "Return JSON only as this object:\n"
            '{"results":[{"request_index":0,"decision":"yes","confidence":0.95,"brief_reason":"<max 12 words>"}]}\n\n'
            "Items:\n"
            f"{json.dumps(items, ensure_ascii=False, indent=2)}"
        )

    if task == "task2":
        # The `document` arm (docs/context_ablation.md) adds one `context`
        # value per item and one instruction sentence; the bare rendering is
        # the paper condition and must stay byte-identical (its wrapper sha is
        # pinned in the tests).
        with_context = item_context == ITEM_CONTEXT_DOCUMENT
        items = [
            {
                "request_index": int(job["request_index"]),
                "source_statement": str(job["item"]["source_statement"]),
                **(
                    {"context": document_context_text(job["item"])}
                    if with_context
                    else {}
                ),
            }
            for job in jobs
        ]
        context_instruction = (
            "Each item's context shows where its source statement appears: the "
            "document, section, the author's requirement marker, and the "
            "neighbouring requirements. Extract from the source statement only.\n"
            if with_context
            else ""
        )
        return (
            "Extract exactly one requirement from each source statement.\n"
            "Preserve the modality of each source. Evaluate each item independently.\n"
            f"{context_instruction}\n"
            'Use one of: "mandatory", "recommended", "optional", "nice_to_have".\n'
            "Use confidence as a decimal from 0.0 to 1.0 for confidence in the selected modality.\n"
            'Do not return percentages such as 95 or strings such as "95%".\n'
            "Return JSON only as this object:\n"
            '{"results":[{"request_index":0,"requirement":"...","modality":"mandatory","confidence":0.95}]}\n\n'
            "Items:\n"
            f"{json.dumps(items, ensure_ascii=False, indent=2)}"
        )

    if task == "task3":
        items = []
        has_declared_modality = False
        for job in jobs:
            item = {
                "request_index": int(job["request_index"]),
                "source_statement": str(job["item"]["source_statement"]),
                "extracted_requirement": str(job["item"]["task2_requirement"]),
            }
            audit_mode = normalize_task3_audit_mode(
                job["item"].get("task3_audit_mode", OFFICIAL_TASK3_AUDIT_MODE)
            )
            if audit_mode == "declared_text":
                item["declared_extracted_modality"] = str(
                    job["item"].get("task2_text_modality", "")
                )
                has_declared_modality = True
            elif audit_mode == "declared_source":
                item["declared_extracted_modality"] = str(
                    job["item"].get("source_modality", "")
                )
                has_declared_modality = True
            items.append(item)
        declared_instruction = (
            "Use any declared_extracted_modality field only when it is present.\n"
            if has_declared_modality
            else ""
        )
        return (
            "Audit whether each extracted software requirement preserves the source statement.\n"
            "Evaluate each item independently and do not repair the extracted requirement.\n"
            f"{declared_instruction}\n"
            'Use one of: "preserves", "strengthens", "weakens", "content_changed".\n'
            "Use confidence as a decimal from 0.0 to 1.0 for confidence in the selected relation.\n"
            'Do not return percentages such as 95 or strings such as "95%".\n'
            "Return JSON only as this object:\n"
            '{"results":[{"request_index":0,"relation":"preserves","confidence":0.95,'
            '"evidence_phrase":"...","brief_reason":"<max 12 words>"}]}\n\n'
            "Items:\n"
            f"{json.dumps(items, ensure_ascii=False, indent=2)}"
        )

    raise ValueError(f"Unsupported benchmark task for batching: {task}")


# Fixed placeholder item used to render the batch wrapper for fingerprinting.
BATCH_WRAPPER_PROBE_ITEM = {
    "source_statement": "The system MUST export reports.",
    "candidate_requirement": "The system MUST export reports.",
    "task2_requirement": "The system must export reports.",
    # Fixed placeholders for the `document` item_context; the bare wrapper
    # ignores them, so the bare digest is unchanged.
    "context_document": "Probe document",
    "context_legend": PURE_MARKER_LEGEND,
    "context_section": "1 Probe section",
    "context_requirement_id": "1.1",
    "context_marker": "M",
    "context_before": "",
    "context_after": "1.2 (O): The system should archive reports.",
}


@cache
def batch_prompt_wrapper_sha(
    task: str, item_context: str = DEFAULT_ITEM_CONTEXT
) -> str:
    """SHA-256 of the batch prompt wrapper for ``task`` and ``item_context``.

    Renders :func:`batch_prompt_for_completion_jobs` for two fixed dummy jobs
    (request_index 0 and 1, fixed placeholder statements) so the digest depends
    only on the instruction envelope and item layout, not on real benchmark
    content. Hashed into ``compute_job_config_sha`` for batched plans (v3) so
    editing the wrapper invalidates cached batched rows on resume.
    """
    probe_jobs = [
        {
            "task": str(task),
            "request_index": index,
            "item": dict(BATCH_WRAPPER_PROBE_ITEM),
            "item_context": normalize_item_context(item_context),
        }
        for index in range(2)
    ]
    return sha256_text(batch_prompt_for_completion_jobs(probe_jobs))


def parse_batch_completion_results(
    raw_text: str,
) -> tuple[dict[int, dict[str, Any]], str]:
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
    duplicate_request_index = False
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
        if request_index_int in parsed:
            duplicate_request_index = True
            continue
        parsed[request_index_int] = dict(result)
    if duplicate_request_index:
        return parsed, "duplicate_request_index"
    return parsed, "ok"


# =============================================================================
# Section 7: Provider completion drivers (OpenAI-compatible, Instructor, logprobs)
# =============================================================================
# Wrappers over the OpenAI-compatible chat-completion path, the Instructor
# Pydantic-validated path, logprob capability probing via /v1/responses, and
# the run-ID and raw-record builders used during benchmark execution.

# Fields of the outgoing request that define request_payload_sha. Kept explicit
# (rather than hashing the whole kwargs dict) so the fingerprint stays stable when
# transport-only kwargs such as logprobs probing are added.
REQUEST_PAYLOAD_SHA_FIELDS = (
    "model",
    "messages",
    "temperature",
    "top_p",
    "max_tokens",
    "seed",
    "response_format",
    "extra_body",
)
RETRY_BASE_DELAY_S = 1.0
# Status codes that are worth retrying; 400/401/403 (e.g. ExceededBudget) must fail fast.
RETRYABLE_STATUS_CODES = frozenset({408, 429})
NON_RETRYABLE_STATUS_CODES = frozenset({400, 401, 403, 404, 422})


def request_payload_sha(request_kwargs: Mapping[str, Any]) -> str:
    """SHA-256 over the canonical JSON of the request payload actually sent."""
    payload = {
        key: request_kwargs.get(key)
        for key in REQUEST_PAYLOAD_SHA_FIELDS
        if key in request_kwargs
    }
    return sha256_text(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    )


def _openai_error_classes() -> tuple[type[BaseException], ...]:
    try:
        from openai import (
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        )
    except Exception:  # pragma: no cover - openai always present in this repo
        return ()
    return (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)


def is_transient_provider_error(exc: BaseException) -> bool:
    """True for timeouts, connection resets, 408/429 and 5xx responses.

    Deliberately excludes 400/401/403-class failures: a budget or auth rejection
    will never succeed on retry and must surface immediately.
    """
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        if status_code in NON_RETRYABLE_STATUS_CODES:
            return False
        return status_code in RETRYABLE_STATUS_CODES or status_code >= 500
    error_classes = _openai_error_classes()
    if error_classes and isinstance(exc, error_classes):
        return True
    return isinstance(exc, (TimeoutError, ConnectionError))


def retry_delay_seconds(
    attempt: int, base_delay_s: float = RETRY_BASE_DELAY_S
) -> float:
    """Exponential backoff with full jitter for the given zero-based attempt."""
    return random.uniform(0.0, base_delay_s * (2**attempt))


def call_with_retries(
    call: Callable[[], Any],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay_s: float = RETRY_BASE_DELAY_S,
) -> tuple[Any, int, BaseException | None]:
    """Run `call`, retrying transient provider failures with bounded backoff.

    Returns (result, retry_count, error). `max_retries` is the number of total
    attempts (3 means one initial call plus two retries); non-transient errors
    fail fast on the first attempt.
    """
    attempts = max(1, int(max_retries))
    retry_count = 0
    last_error: BaseException | None = None
    for attempt in range(attempts):
        try:
            return call(), retry_count, None
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= attempts or not is_transient_provider_error(exc):
                return None, retry_count, exc
            retry_count += 1
            time.sleep(retry_delay_seconds(attempt, base_delay_s))
    return None, retry_count, last_error


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
    seed: int | None = DEFAULT_REQUEST_SEED,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict[str, Any]:
    api_key = os.getenv(api_key_env, "EMPTY")
    # max_retries=0: the SDK's own retry loop is invisible to us, so it would
    # silently inflate wall-clock latency and hide attempts from retry_count.
    # call_with_retries is the single, recorded retry mechanism.
    client = OpenAI(
        base_url=host.rstrip("/") + "/",
        api_key=api_key,
        timeout=timeout_s,
        max_retries=0,
    )
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        request_kwargs["seed"] = int(seed)
    if logprobs:
        request_kwargs["logprobs"] = True
        if top_logprobs is not None:
            request_kwargs["top_logprobs"] = int(top_logprobs)
    if response_format:
        request_kwargs["response_format"] = dict(response_format)
    if extra_body:
        request_kwargs["extra_body"] = dict(extra_body)
    payload_sha = request_payload_sha(request_kwargs)

    start = time.perf_counter()
    response, retry_count, error = call_with_retries(
        lambda: client.chat.completions.create(**request_kwargs),
        max_retries=max_retries,
    )
    latency_s = time.perf_counter() - start
    if error is not None:
        return {
            "ok": False,
            "raw_text": "",
            "response_json": None,
            "latency_s": latency_s,
            "error": repr(error),
            "request_payload_sha": payload_sha,
            "request_seed": request_kwargs.get("seed"),
            "retry_count": retry_count,
        }
    try:
        raw_text = response.choices[0].message.content or ""
        response_json = response.model_dump(mode="json")
    except Exception as exc:  # malformed provider payload
        return {
            "ok": False,
            "raw_text": "",
            "response_json": None,
            "latency_s": latency_s,
            "error": repr(exc),
            "request_payload_sha": payload_sha,
            "request_seed": request_kwargs.get("seed"),
            "retry_count": retry_count,
        }
    return {
        "ok": True,
        "raw_text": raw_text,
        "response_json": response_json,
        "latency_s": latency_s,
        "error": "",
        "request_payload_sha": payload_sha,
        "request_seed": request_kwargs.get("seed"),
        "retry_count": retry_count,
    }


def normalize_instructor_mode_name(value: Any) -> str:
    normalized = str(value or "json").strip().lower().replace("-", "_")
    aliases = {
        "json": "json",
        "json_mode": "json",
        "mode_json": "json",
        "tools": "tools",
        "tool": "tools",
        "tools_strict": "tools_strict",
        "strict": "tools_strict",
        "md_json": "md_json",
        "markdown_json": "md_json",
    }
    if normalized not in aliases:
        raise ValueError(f"Unknown Instructor mode: {value}")
    return aliases[normalized]


def instructor_mode_value(value: Any) -> Any:
    import instructor

    mode = normalize_instructor_mode_name(value)
    return {
        "json": instructor.Mode.JSON,
        "tools": instructor.Mode.TOOLS,
        "tools_strict": instructor.Mode.TOOLS_STRICT,
        "md_json": instructor.Mode.MD_JSON,
    }[mode]


def instructor_extra_body(
    extra_body: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not extra_body:
        return None
    cleaned = {
        key: value for key, value in extra_body.items() if key != "response_format"
    }
    return cleaned or None


def instructor_completion(
    host: str,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int = 256,
    timeout_s: int = 120,
    api_key_env: str = "LOCAL_OPENAI_API_KEY",
    response_format: Mapping[str, Any] | None = None,
    extra_body: Mapping[str, Any] | None = None,
    task: str = "task1",
    batched: bool = False,
    instructor_mode: str = "json",
    validation_retries: int = 2,
    response_model: Any | None = None,
    seed: int | None = DEFAULT_REQUEST_SEED,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict[str, Any]:
    # response_format is accepted for call-signature parity with chat_completion;
    # Instructor derives the schema from response_model instead.
    import instructor

    api_key = os.getenv(api_key_env, "EMPTY")
    # max_retries=0 for the same reason as in chat_completion: retry_count must
    # be the whole retry story (Instructor's validation retries stay separate).
    base_client = OpenAI(
        base_url=host.rstrip("/") + "/",
        api_key=api_key,
        timeout=timeout_s,
        max_retries=0,
    )
    client = instructor.from_openai(
        base_client, mode=instructor_mode_value(instructor_mode)
    )
    model_type = response_model or so.response_model_for_task(task, batched=batched)

    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "response_model": model_type,
        # NOTE: Instructor's own `max_retries` is schema-VALIDATION retries, which is
        # a different knob from this function's transport-level `max_retries`.
        "max_retries": nonnegative_int(validation_retries, "validation_retries"),
    }
    if seed is not None:
        request_kwargs["seed"] = int(seed)
    cleaned_extra_body = instructor_extra_body(extra_body)
    if cleaned_extra_body:
        request_kwargs["extra_body"] = cleaned_extra_body
    payload_sha = request_payload_sha(request_kwargs)
    provenance = {
        "request_payload_sha": payload_sha,
        "request_seed": request_kwargs.get("seed"),
    }

    start = time.perf_counter()
    parsed, retry_count, error = call_with_retries(
        lambda: client.chat.completions.create(**request_kwargs),
        max_retries=max_retries,
    )
    latency_s = time.perf_counter() - start
    if error is None:
        payload = parsed.model_dump(mode="json")
        # Instructor exposes the underlying provider response on the validated model.
        raw_response = getattr(parsed, "_raw_response", None)
        raw_response_json = None
        if raw_response is not None:
            try:
                raw_response_json = raw_response.model_dump(mode="json")
            except Exception:
                raw_response_json = None
        response_json: dict[str, Any] = {"instructor_validated": payload}
        if isinstance(raw_response_json, Mapping):
            response_json.update(dict(raw_response_json))
        return {
            "ok": True,
            "raw_text": json.dumps(payload, ensure_ascii=False),
            "response_json": response_json,
            "latency_s": latency_s,
            "error": "",
            "retry_count": retry_count,
            **provenance,
        }

    error_name = error.__class__.__name__
    parse_status_override = (
        "instructor_validation_error"
        if "Retry" in error_name
        or "Validation" in error_name
        or "Instructor" in error_name
        else ""
    )
    raw_text = str(getattr(error, "last_completion", "") or "")
    return {
        "ok": False,
        "raw_text": raw_text,
        "response_json": None,
        "latency_s": latency_s,
        "error": repr(error),
        "parse_status_override": parse_status_override,
        "retry_count": retry_count,
        **provenance,
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


def response_logprob_tokens(
    response_json: dict[str, Any] | None,
) -> list[dict[str, Any]]:
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
                        tokens.extend(
                            normalize_logprob_tokens(content_item.get("logprobs"))
                        )
        if tokens:
            return tokens

    return normalize_logprob_tokens(response_json.get("logprobs"))


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
                    if isinstance(content_item, dict) and isinstance(
                        content_item.get("text"), str
                    ):
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
        response = requests.post(
            endpoint, headers=headers, json=payload, timeout=timeout_s
        )
        latency_s = time.perf_counter() - start
        try:
            response_json = response.json()
        except ValueError:
            response_json = None
        raw_text = responses_output_text(response_json) or response.text[:500]
        error = (
            "" if response.ok else f"HTTP {response.status_code}: {response.text[:500]}"
        )
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


EMPTY_RESPONSE_FIELDS: dict[str, Any] = {
    "finish_reason": "",
    "usage_prompt_tokens": None,
    "usage_completion_tokens": None,
    "usage_total_tokens": None,
    "served_model": "",
    "system_fingerprint": "",
    "response_id": "",
}


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_response_fields(response_json: Any) -> dict[str, Any]:
    """Pull provider-response provenance out of a raw chat-completion payload.

    Missing values stay at their empty defaults ("" for strings, None for token
    counts) so downstream tables have a stable schema regardless of provider.
    """
    fields = dict(EMPTY_RESPONSE_FIELDS)
    if not isinstance(response_json, Mapping):
        return fields

    choices = response_json.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        finish_reason = choices[0].get("finish_reason")
        fields["finish_reason"] = str(finish_reason) if finish_reason else ""

    usage = response_json.get("usage")
    if isinstance(usage, Mapping):
        fields["usage_prompt_tokens"] = _optional_int(
            usage.get("prompt_tokens", usage.get("input_tokens"))
        )
        fields["usage_completion_tokens"] = _optional_int(
            usage.get("completion_tokens", usage.get("output_tokens"))
        )
        fields["usage_total_tokens"] = _optional_int(usage.get("total_tokens"))

    served_model = response_json.get("model")
    fields["served_model"] = str(served_model) if served_model else ""
    system_fingerprint = response_json.get("system_fingerprint")
    fields["system_fingerprint"] = str(system_fingerprint) if system_fingerprint else ""
    response_id = response_json.get("id")
    fields["response_id"] = str(response_id) if response_id else ""
    return fields


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
    output_contract_version: str = "",
    confidence_scale: str = "",
    job_config_sha: str = "",
    job_config_sha_version: int = JOB_CONFIG_SHA_VERSION,
    request_seed: int | None = None,
    max_tokens: int | None = None,
    batch_order: str = DEFAULT_BATCH_ORDER,
    batch_seed_ids: Iterable[str] | None = None,
    batch_variant_mix: int | str = "",
    item_context: str = DEFAULT_ITEM_CONTEXT,
) -> dict[str, Any]:
    record_contract_version = output_contract_version
    record_confidence_scale = confidence_scale
    if not record_contract_version and (
        prompt_version_uses_confidence_0_1(prompt_version)
        or prompt_text_uses_confidence_0_1(prompt)
    ):
        record_contract_version = so.PROMPT_OUTPUT_CONTRACT_VERSION
    if not record_confidence_scale:
        record_confidence_scale = confidence_scale_for_record(
            {
                "prompt_version": prompt_version,
                "output_contract_version": record_contract_version,
                "prompt": prompt,
            }
        )
    if output_contract_version == so.INSTRUCTOR_OUTPUT_CONTRACT_VERSION:
        parsed_json, parse_status = parse_instructor_task_response(
            task, completion.get("raw_text", "")
        )
    else:
        parsed_json, parse_status = parse_task_response(
            task,
            completion.get("raw_text", ""),
            confidence_scale=record_confidence_scale,
        )
    parse_status_override = str(completion.get("parse_status_override", "")).strip()
    if parse_status_override:
        parsed_json = None
        parse_status = parse_status_override
    elif not completion.get("ok"):
        parse_status = "request_error"

    response_fields = extract_response_fields(completion.get("response_json"))
    # A response cut off at the token limit is a budget problem, not malformed
    # model output: give it its own status so it is not silently filed as
    # invalid_json. It remains a parse failure (parse_status != "ok") everywhere.
    if response_fields["finish_reason"] == "length" and parse_status not in {
        "ok",
        "request_error",
    }:
        parse_status = "truncated"

    raw_text = completion.get("raw_text", "")
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
        "raw_text": raw_text,
        "parsed_json": parsed_json,
        "parse_status": parse_status,
        "latency_s": completion.get("latency_s", ""),
        "error": completion.get("error", ""),
        # Provider-response provenance (see extract_response_fields).
        **response_fields,
        "response_chars": len(raw_text or ""),
        "retry_count": int(completion.get("retry_count", 0) or 0),
        # We send exactly one user message and no system prompt; recording both
        # makes that auditable from the raw table alone.
        "system_prompt": "",
        "request_messages_role_layout": "user_only",
        "request_seed": completion.get("request_seed", request_seed),
        "job_config_sha_version": int(job_config_sha_version),
        "batch_order": normalize_batch_order(batch_order),
        "item_context": normalize_item_context(item_context),
    }
    if (
        task == "task2"
        and isinstance(parsed_json, Mapping)
        and parsed_json.get("requirement")
    ):
        record["requirement_word_count"] = len(str(parsed_json["requirement"]).split())
    record["request_payload_sha"] = str(
        completion.get("request_payload_sha")
        or request_payload_sha(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "seed": request_seed,
                "response_format": dict(response_format) if response_format else None,
                "extra_body": dict(request_extra_body) if request_extra_body else None,
            }
        )
    )
    if batch_seed_ids is not None:
        record["batch_seed_ids"] = sorted({str(value) for value in batch_seed_ids})
    else:
        record["batch_seed_ids"] = [str(item.get("seed_id", ""))]
    if batch_variant_mix != "":
        record["batch_variant_mix"] = int(batch_variant_mix)
    else:
        record["batch_variant_mix"] = 1
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
    resolved_structured_output = normalize_structured_output_mode(
        structured_output, json_mode=json_mode
    )
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
    if batch_id:
        # Usage/finish_reason on a batched row describe the whole batch request,
        # not this single item; mirror completion tokens under an explicit
        # batch-scoped name so per-item consumers cannot misread the copy.
        record["batch_usage_completion_tokens"] = response_fields[
            "usage_completion_tokens"
        ]
    if job_config_sha:
        record["job_config_sha"] = job_config_sha
    if record_contract_version:
        record["output_contract_version"] = record_contract_version
    if record_confidence_scale != CONFIDENCE_SCALE_0_100:
        record["confidence_scale"] = record_confidence_scale
    if "template_id" in item:
        record["template_id"] = item["template_id"]
    for key in [
        "source_dataset",
        "original_requirement",
        "capability_text",
        "source_statement",
        "source_item_id",
        "task2_run_id",
        "task2_model",
        "task2_requirement",
        "task2_modality",
        "task2_text_modality",
        "task2_text_modality_basis",
        "task2_text_modality_parse_status",
        "task2_confidence",
        "task3_declared_relation",
        "task3_gold_relation",
        "task3_audit_mode",
        "ordinal_strength",
        "numeric_strength",
        "context_marker",
        "context_requirement_id",
    ]:
        if key in item:
            record[key] = item[key]
    if request_index is not None:
        record["request_index"] = request_index
    # Raw JSONL is the most durable artifact in the repository. Check the
    # configuration-derived provenance before it is handed to the writer;
    # `parsed_json` is model output, not provider configuration, so a model
    # that happens to emit a credential-shaped key must not abort a run.
    assert_no_credential_shaped_values(
        {key: value for key, value in record.items() if key != "parsed_json"},
        where="raw record",
    )
    return record


def batch_composition(jobs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Auditable summary of which items share one batched provider request."""
    items = [dict(job.get("item", {})) for job in jobs]
    return {
        "batch_seed_ids": sorted({str(item.get("seed_id", "")) for item in items}),
        "batch_variant_mix": len(
            {str(item.get("source_modality", "")) for item in items}
        ),
    }


def _job_record(
    job: Mapping[str, Any],
    *,
    completion: dict[str, Any],
    request_index: int | None,
    response_format: Mapping[str, Any] | None,
    request_extra_body: Mapping[str, Any] | None,
    batch_id: str = "",
    batch_size: int | str = "",
    batch_item_count: int | str = "",
    batch_prompt_hash: str = "",
    output_contract_version: str = "",
    confidence_scale: str = "",
    batch_jobs: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Map a planned job dict and its completion onto a raw output record.

    Centralizes the ~20 job-derived fields shared by the single, batch, and
    instructor-batch completion paths; callers supply only the fields that
    genuinely differ between paths.
    """
    composition = batch_composition(batch_jobs if batch_jobs is not None else [job])
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
        request_index=request_index,
        provider_id=str(job.get("provider_id", "")),
        profile_id=str(job.get("profile_id", "")),
        run_group_id=str(job.get("run_group_id", "")),
        base_url=str(job.get("base_url", job.get("host", ""))),
        api_key_env=str(job.get("api_key_env", "")),
        json_mode=bool(job.get("json_mode", False)),
        structured_output=str(job.get("structured_output", "none")),
        response_format=response_format,
        request_extra_body=request_extra_body,
        server_model_probe=job.get("server_model_probe"),
        batch_id=batch_id,
        batch_size=batch_size,
        batch_item_count=batch_item_count,
        batch_prompt_hash=batch_prompt_hash,
        output_contract_version=output_contract_version,
        confidence_scale=confidence_scale,
        job_config_sha=str(job.get("job_config_sha", "")),
        job_config_sha_version=int(
            job.get("job_config_sha_version", JOB_CONFIG_SHA_VERSION)
        ),
        request_seed=job.get("seed"),
        max_tokens=int(job.get("max_tokens", 256)),
        batch_order=str(job.get("batch_order", DEFAULT_BATCH_ORDER)),
        batch_seed_ids=composition["batch_seed_ids"],
        batch_variant_mix=composition["batch_variant_mix"],
        item_context=str(job.get("item_context", DEFAULT_ITEM_CONTEXT)),
    )


def job_uses_instructor(job: Mapping[str, Any]) -> bool:
    return (
        normalize_structured_output_mode(job.get("structured_output")) == "instructor"
    )


def output_contract_version_for_job(job: Mapping[str, Any]) -> str:
    if job_uses_instructor(job):
        return so.INSTRUCTOR_OUTPUT_CONTRACT_VERSION
    if prompt_version_uses_confidence_0_1(job.get("prompt_version")):
        return so.PROMPT_OUTPUT_CONTRACT_VERSION
    return ""


def confidence_scale_for_job(job: Mapping[str, Any]) -> str:
    if job_uses_instructor(job) or prompt_version_uses_confidence_0_1(
        job.get("prompt_version")
    ):
        return CONFIDENCE_SCALE_0_1
    return ""


def completion_runner_for_job(
    job: Mapping[str, Any],
    completion_fn: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    if job_uses_instructor(job) and completion_fn is chat_completion:
        return instructor_completion
    return completion_fn


def completion_kwargs_for_job(
    job: Mapping[str, Any],
    *,
    prompt: str,
    max_tokens: int,
    response_format: Mapping[str, Any] | None,
    extra_body: Mapping[str, Any] | None,
    batched: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "host": str(job["host"]),
        "model": str(job["model"]),
        "prompt": prompt,
        "temperature": float(job["temperature"]),
        "top_p": float(job["top_p"]),
        "max_tokens": int(max_tokens),
        "timeout_s": int(job.get("timeout_s", 120)),
        "api_key_env": str(job.get("api_key_env", "LOCAL_OPENAI_API_KEY")),
        "response_format": response_format,
        "extra_body": extra_body,
        "seed": job.get("seed"),
        "max_retries": int(job.get("max_retries", DEFAULT_MAX_RETRIES)),
    }
    if job_uses_instructor(job):
        task = str(job["task"])
        kwargs.update(
            {
                "task": task,
                "batched": batched,
                "instructor_mode": str(job.get("instructor_mode", "json")),
                "validation_retries": int(job.get("validation_retries", 2)),
                "response_model": so.response_model_for_task(task, batched=batched),
            }
        )
    return kwargs


def run_completion_job(
    job: Mapping[str, Any],
    completion_fn: Callable[..., dict[str, Any]] = chat_completion,
) -> dict[str, Any]:
    runner = completion_runner_for_job(job, completion_fn)
    completion = runner(
        **completion_kwargs_for_job(
            job,
            prompt=str(job["prompt"]),
            max_tokens=int(job.get("max_tokens", 256)),
            response_format=job.get("response_format"),
            extra_body=job.get("extra_body"),
            batched=False,
        )
    )
    request_index = job.get("request_index")
    return _job_record(
        job,
        completion=completion,
        request_index=int(request_index) if request_index is not None else None,
        response_format=job.get("response_format"),
        request_extra_body=instructor_extra_body(job.get("extra_body"))
        if job_uses_instructor(job)
        else job.get("extra_body"),
        output_contract_version=output_contract_version_for_job(job),
        confidence_scale=confidence_scale_for_job(job),
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
        str(job.get("instructor_mode", "")),
        int(job.get("validation_retries", 0)),
        int(job.get("fallback_batch_size", 1)),
        compact_json(job.get("seed")),
        bool(job.get("send_seed", True)),
        str(job.get("batch_order", DEFAULT_BATCH_ORDER)),
    )


def _job_seed_id(job: Mapping[str, Any]) -> str:
    item = job.get("item")
    return str(item.get("seed_id", "")) if isinstance(item, Mapping) else ""


def _shuffled_group_jobs(
    ordered: list[dict[str, Any]],
    group_key: tuple[Any, ...],
    seed: Any,
    batch_size: int = 1,
) -> list[list[dict[str, Any]]]:
    """Deterministic seed-disjoint batches for one batch-key group.

    Seeding the RNG from a SHA-256 of ``(seed, batch key)`` keeps the permutation
    stable across processes and independent of dict iteration order, so a
    `shuffled` run is exactly reproducible from its recorded seed.

    The permutation is applied to the SEED GROUPS (and to the variants within a
    seed), not to the flat job list: the jobs of one seed are then handed out to
    distinct batches, always filling the currently emptiest one. So a shuffled
    batch holds ``batch_size`` DIFFERENT seeds -- the point of the ablation is to
    break the grouped default's "4 variants of one seed in one prompt" confound,
    and a plain shuffle would still collide two variants of a seed in the same
    request by chance. When a group has fewer distinct seeds than ``batch_size``
    (or a seed has more jobs than there are batches) a collision is unavoidable;
    the batches are still filled, and the shortfall is logged as a warning rather
    than passing silently.
    """
    digest = sha256_text(f"{seed}|{compact_json([str(part) for part in group_key])}")
    rng = random.Random(int(digest[:16], 16))
    resolved_batch_size = max(1, int(batch_size))

    by_seed: dict[str, list[dict[str, Any]]] = {}
    for job in ordered:
        by_seed.setdefault(_job_seed_id(job), []).append(job)
    seed_ids = list(by_seed)
    rng.shuffle(seed_ids)
    for seed_id in seed_ids:
        rng.shuffle(by_seed[seed_id])

    batch_count = -(-len(ordered) // resolved_batch_size)
    batches: list[list[dict[str, Any]]] = [[] for _ in range(batch_count)]
    seeds_in_batch: list[set[str]] = [set() for _ in range(batch_count)]
    # Min-heap of (current size, batch index) over the batches that still have
    # room, so every job lands in the emptiest batch that does not hold its seed.
    open_batches = [(0, index) for index in range(batch_count)]
    heapq.heapify(open_batches)
    collisions = 0
    # Seeds with the most jobs are the hardest to spread; place them first.
    for seed_id in sorted(seed_ids, key=lambda value: -len(by_seed[value])):
        for job in by_seed[seed_id]:
            skipped: list[tuple[int, int]] = []
            chosen: tuple[int, int] | None = None
            while open_batches:
                candidate = heapq.heappop(open_batches)
                if seed_id not in seeds_in_batch[candidate[1]]:
                    chosen = candidate
                    break
                skipped.append(candidate)
            if chosen is None:
                if not skipped:  # pragma: no cover - capacity guarantees a slot
                    raise RuntimeError("No batch slot left for a planned job.")
                collisions += 1
                chosen = skipped.pop(0)
            for entry in skipped:
                heapq.heappush(open_batches, entry)
            size, index = chosen
            batches[index].append(job)
            seeds_in_batch[index].add(seed_id)
            if size + 1 < resolved_batch_size:
                heapq.heappush(open_batches, (size + 1, index))
    if collisions:
        logger.warning(
            "shuffled batching: %d job(s) share a batch with another variant of the same seed "
            "(group has %d distinct seed(s) for batch_size %d)",
            collisions,
            len(seed_ids),
            resolved_batch_size,
        )
    return batches


def completion_job_batches(
    jobs: Iterable[Mapping[str, Any]],
    batch_size: int,
    batch_order: str | None = None,
    seed: Any = None,
    *,
    planned_jobs: Iterable[Mapping[str, Any]] | None = None,
) -> list[list[dict[str, Any]]]:
    """Chunk planned jobs into provider batches.

    `grouped` (default) keeps consecutive request indices together, so all four
    modality variants of a seed share a batch. `shuffled` deterministically
    permutes each batch-key group and then spreads the jobs of one seed across
    DIFFERENT batches, so no batch contains two variants of the same seed
    whenever that is feasible; it is an ablation for batch-composition effects,
    not the default. Both knobs fall back to the values carried on the jobs
    themselves.

    `planned_jobs` makes resume batch-stable: pass the FULL plan there and the
    already-completed subset in `jobs`, and the batches are computed over the
    plan and then filtered (by :func:`completion_record_key`) to the jobs that
    still have to run, dropping batches that are left empty. Without it a resumed
    run would re-shuffle the pending subset and send different batch neighbours
    than the first attempt.
    """
    job_list = [dict(job) for job in jobs]
    if not job_list:
        return []
    if planned_jobs is not None:
        pending_by_key = {completion_record_key(job): job for job in job_list}
        planned_batches = completion_job_batches(
            planned_jobs,
            batch_size,
            batch_order=batch_order,
            seed=seed,
        )
        filtered: list[list[dict[str, Any]]] = []
        for batch in planned_batches:
            kept = [
                pending_by_key[key]
                for key in (completion_record_key(job) for job in batch)
                if key in pending_by_key
            ]
            if kept:
                filtered.append(kept)
        return filtered

    resolved_batch_size = positive_int(batch_size, "batch_size")
    resolved_order = normalize_batch_order(
        batch_order if batch_order is not None else job_list[0].get("batch_order")
    )
    resolved_seed = (
        seed if seed is not None else job_list[0].get("seed", DEFAULT_REQUEST_SEED)
    )
    if resolved_batch_size <= 1:
        return [[job] for job in job_list]

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for job in job_list:
        grouped.setdefault(completion_batch_key(job), []).append(job)

    batches: list[list[dict[str, Any]]] = []
    sorted_groups = sorted(
        grouped.items(),
        key=lambda entry: min(int(row.get("request_index", 0)) for row in entry[1]),
    )
    for group_key, group_jobs in sorted_groups:
        ordered = sorted(group_jobs, key=lambda row: int(row.get("request_index", 0)))
        if resolved_order == BATCH_ORDER_SHUFFLED:
            batches.extend(
                _shuffled_group_jobs(
                    ordered, group_key, resolved_seed, resolved_batch_size
                )
            )
            continue
        batches.extend(
            # A short final batch is expected, so strict= stays off.
            list(batch)
            for batch in batched(ordered, resolved_batch_size, strict=False)
        )
    return batches


# Driver-level provenance of the single provider request behind a batch. Copied
# onto every per-item completion built from that request so a batched raw row
# reports the seed, payload sha, and retry count that actually produced it.
BATCH_DRIVER_PROVENANCE_FIELDS = ("request_payload_sha", "request_seed", "retry_count")


def batch_driver_provenance(completion: Mapping[str, Any]) -> dict[str, Any]:
    """Return the driver provenance fields present on a batch ``completion``.

    Absent keys are omitted rather than defaulted, so build_raw_record keeps its
    fallbacks (job seed, locally recomputed payload sha) for drivers that do not
    report them.
    """
    return {
        key: completion[key]
        for key in BATCH_DRIVER_PROVENANCE_FIELDS
        if key in completion
    }


def valid_instructor_batch_results(
    task: str,
    parsed_results: Mapping[int, Mapping[str, Any]],
    jobs: list[Mapping[str, Any]],
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    expected = {int(job["request_index"]) for job in jobs}
    valid: dict[int, dict[str, Any]] = {}
    fallback_jobs: list[dict[str, Any]] = []
    for job in jobs:
        request_index = int(job["request_index"])
        result = parsed_results.get(request_index)
        if result is None:
            fallback_jobs.append(dict(job))
            continue
        item_result = {
            key: value for key, value in result.items() if key != "request_index"
        }
        _, parse_status = parse_instructor_task_response(
            task, json.dumps(item_result, ensure_ascii=False)
        )
        if parse_status == "ok" and set(parsed_results).issubset(expected):
            valid[request_index] = item_result
        else:
            fallback_jobs.append(dict(job))
    return valid, fallback_jobs


def run_instructor_completion_batch(
    jobs: list[Mapping[str, Any]],
    completion_fn: Callable[..., dict[str, Any]] = chat_completion,
) -> list[dict[str, Any]]:
    if not jobs:
        return []
    if len(jobs) == 1:
        return [run_completion_job(jobs[0], completion_fn=completion_fn)]

    first = jobs[0]
    task = str(first["task"])
    batch_prompt = batch_prompt_for_completion_jobs(jobs)
    batch_prompt_hash = sha256_text(batch_prompt)
    batch_id = (
        f"{first['run_id']}:{first['model']}:{first['task']}:"
        f"{first['sample_kind']}:{first['sample_index']}:"
        f"{min(int(job.get('request_index', 0)) for job in jobs)}-"
        f"{max(int(job.get('request_index', 0)) for job in jobs)}"
    )
    runner = completion_runner_for_job(first, completion_fn)
    completion = runner(
        **completion_kwargs_for_job(
            first,
            prompt=batch_prompt,
            max_tokens=int(first.get("max_tokens", 256)) * len(jobs),
            response_format=None,
            extra_body=first.get("extra_body"),
            batched=True,
        )
    )

    records_by_request_index: dict[int, dict[str, Any]] = {}
    fallback_jobs = [dict(job) for job in jobs]
    if completion.get("ok"):
        parsed_results, batch_parse_status = parse_batch_completion_results(
            completion.get("raw_text", "")
        )
        if batch_parse_status == "ok":
            valid_results, fallback_jobs = valid_instructor_batch_results(
                task, parsed_results, jobs
            )
            for job in jobs:
                request_index = int(job["request_index"])
                result = valid_results.get(request_index)
                if result is None:
                    continue
                item_completion = {
                    "ok": True,
                    "raw_text": json.dumps(result, ensure_ascii=False),
                    "response_json": completion.get("response_json"),
                    "latency_s": completion.get("latency_s", ""),
                    "error": "",
                    # The driver-level provenance of the one request that
                    # produced this item; without it a batched row would fall
                    # back to a locally recomputed single-item payload sha and
                    # lose the batch's seed and retry count.
                    **batch_driver_provenance(completion),
                }
                records_by_request_index[request_index] = _job_record(
                    job,
                    completion=item_completion,
                    request_index=request_index,
                    response_format=None,
                    request_extra_body=instructor_extra_body(job.get("extra_body")),
                    batch_id=batch_id,
                    batch_size=len(jobs),
                    batch_item_count=len(jobs),
                    batch_prompt_hash=batch_prompt_hash,
                    output_contract_version=output_contract_version_for_job(job),
                    confidence_scale=confidence_scale_for_job(job),
                    batch_jobs=jobs,
                )

    for fallback_job in fallback_jobs:
        fallback_record = run_completion_job(fallback_job, completion_fn=completion_fn)
        records_by_request_index[int(fallback_job["request_index"])] = fallback_record
    return [records_by_request_index[int(job["request_index"])] for job in jobs]


def run_completion_batch(
    jobs: list[Mapping[str, Any]],
    completion_fn: Callable[..., dict[str, Any]] = chat_completion,
) -> list[dict[str, Any]]:
    if not jobs:
        return []
    if len(jobs) == 1:
        return [run_completion_job(jobs[0], completion_fn=completion_fn)]
    if job_uses_instructor(jobs[0]):
        return run_instructor_completion_batch(jobs, completion_fn=completion_fn)

    first = jobs[0]
    batch_prompt = batch_prompt_for_completion_jobs(jobs)
    batch_prompt_hash = sha256_text(batch_prompt)
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
        seed=first.get("seed"),
        max_retries=int(first.get("max_retries", DEFAULT_MAX_RETRIES)),
    )
    batch_id = (
        f"{first['run_id']}:{first['model']}:{first['task']}:"
        f"{first['sample_kind']}:{first['sample_index']}:"
        f"{min(int(job.get('request_index', 0)) for job in jobs)}-"
        f"{max(int(job.get('request_index', 0)) for job in jobs)}"
    )
    parsed_results, batch_parse_status = parse_batch_completion_results(
        completion.get("raw_text", "")
    )
    if batch_parse_status != "ok":
        parsed_results = {}
    records_by_request_index: dict[int, dict[str, Any]] = {}
    # (job, why the batch could not serve it) for every item that has to be
    # re-sent on its own, mirroring the Instructor batch path.
    fallback_jobs: list[tuple[Mapping[str, Any], str]] = []
    for job in jobs:
        request_index = int(job["request_index"])
        result = parsed_results.get(request_index)
        if not completion.get("ok"):
            fallback_jobs.append(
                (job, str(completion.get("error", "")) or "batch_request_failed")
            )
            continue
        if result is None:
            fallback_jobs.append(
                (
                    job,
                    f"batch_parse_status={batch_parse_status}; missing request_index={request_index}",
                )
            )
            continue
        item_completion = {
            "ok": True,
            "raw_text": json.dumps(result, ensure_ascii=False),
            "response_json": completion.get("response_json"),
            "latency_s": completion.get("latency_s", ""),
            "error": "",
            # Provenance of the single batched request that produced this item.
            **batch_driver_provenance(completion),
        }
        records_by_request_index[request_index] = _job_record(
            job,
            completion=item_completion,
            request_index=request_index,
            response_format=batch_response_format,
            request_extra_body=batch_extra_body,
            batch_id=batch_id,
            batch_size=len(jobs),
            batch_item_count=len(jobs),
            batch_prompt_hash=batch_prompt_hash,
            batch_jobs=jobs,
        )

    for fallback_job, batch_error in fallback_jobs:
        records_by_request_index[int(fallback_job["request_index"])] = (
            _batch_fallback_record(
                fallback_job,
                completion_fn=completion_fn,
                batch_error=batch_error,
            )
        )
    return [records_by_request_index[int(job["request_index"])] for job in jobs]


def _batch_fallback_record(
    job: Mapping[str, Any],
    *,
    completion_fn: Callable[..., dict[str, Any]],
    batch_error: str,
) -> dict[str, Any]:
    """Re-send one job of a failed batch as a single-item request.

    The record is the fallback's own record (its status, its payload sha), tagged
    with ``batch_size = 1`` so a batched run stays auditable down to the rows that
    were actually sent alone. Only when the fallback itself fails to parse does
    the failure stand; the original batch error is then kept in ``error`` behind a
    ``batch_fallback:`` prefix so the batch-level cause is not lost.
    """
    record = run_completion_job(job, completion_fn=completion_fn)
    record["batch_size"] = 1
    if str(record.get("parse_status", "")) != "ok":
        own_error = str(record.get("error", "") or "")
        record["error"] = f"batch_fallback:{batch_error}" + (
            f"; {own_error}" if own_error else ""
        )
    return record


def run_completion_jobs(
    jobs: Iterable[Mapping[str, Any]],
    max_workers: int,
    completion_fn: Callable[..., dict[str, Any]] = chat_completion,
    batch_size: int = 1,
    batch_order: str | None = None,
    seed: Any = None,
    *,
    planned_jobs: Iterable[Mapping[str, Any]] | None = None,
) -> Iterable[dict[str, Any]]:
    """Run planned jobs, batched per :func:`completion_job_batches`.

    ``planned_jobs`` carries the FULL plan on a resumed run so the pending subset
    keeps the batch composition of the first attempt instead of being re-batched
    on its own; see :func:`completion_job_batches`.
    """
    batches = completion_job_batches(
        jobs,
        batch_size=batch_size,
        batch_order=batch_order,
        seed=seed,
        planned_jobs=planned_jobs,
    )
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


# =============================================================================
# Section 8: Metrics and UQ scoring
# =============================================================================
# Classification metrics (accuracy, F1, macro-F1, Brier, ECE, AUROC),
# rank correlations, distributional UQ (predictive entropy, variation ratio,
# self-consistency, ensemble disagreement), monotonicity diagnostics, and
# bootstrap confidence intervals used by the paper-facing summaries.


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
    return float(
        f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )


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
    for bin_index, (low, high) in enumerate(pairwise(bin_edges)):
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
        return dict.fromkeys(label_order, 0.0)
    counts = Counter(labels)
    total = sum(counts.values())
    return {label: counts.get(label, 0) / total for label in label_order}


def label_distribution_from_rows(
    task: str, rows: list[dict[str, Any]]
) -> dict[str, float]:
    labels = [label_from_parsed(task, row["parsed_json"]) for row in rows]
    return label_distribution(labels, class_order_for_task(task))


def label_distribution_json(distribution: dict[str, float]) -> str:
    normalized = {
        label: round(float(value), 12) for label, value in distribution.items()
    }
    return json.dumps(
        normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )


def normalized_predictive_entropy(distribution: dict[str, float]) -> float:
    if len(distribution) < 2:
        return 0.0
    probabilities = np.asarray(
        [value for value in distribution.values() if value > 0.0], dtype=float
    )
    if probabilities.size == 0:
        return math.nan
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / math.log(len(distribution))


def variation_ratio(distribution: dict[str, float]) -> float:
    if not distribution:
        return math.nan
    return 1.0 - max(float(value) for value in distribution.values())


def semantic_response_text(task: str, parsed: Mapping[str, Any]) -> str:
    if task == "task1":
        decision = (
            normalize_decision(parsed.get("decision"))
            or str(parsed.get("decision", "")).strip()
        )
        parts = [
            f"decision: {decision}" if decision else "",
            f"reason: {parsed.get('brief_reason', '')}",
        ]
    elif task == "task2":
        modality = (
            normalize_modality(parsed.get("modality"))
            or str(parsed.get("modality", "")).strip()
        )
        parts = [
            f"modality: {modality}" if modality else "",
            f"requirement: {parsed.get('requirement', '')}",
        ]
    elif task == "task3":
        relation = (
            normalize_relation(parsed.get("relation"))
            or str(parsed.get("relation", "")).strip()
        )
        parts = [
            f"relation: {relation}" if relation else "",
            f"evidence: {parsed.get('evidence_phrase', '')}",
            f"reason: {parsed.get('brief_reason', '')}",
        ]
    else:
        parts = [json.dumps(parsed, ensure_ascii=True, sort_keys=True)]
    text = "\n".join(str(part).strip() for part in parts if str(part).strip())
    return re.sub(r"\s+", " ", text).strip()


def semantic_texts_from_rows(task: str, rows: list[dict[str, Any]]) -> list[str]:
    texts: list[str] = []
    for row in rows:
        parsed = row.get("parsed_json")
        if isinstance(parsed, Mapping):
            texts.append(semantic_response_text(task, parsed))
    return texts


def normalize_embedding_rows(embeddings: np.ndarray) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0.0)


def _tfidf_text_embedding_matrix(texts: list[str]) -> np.ndarray:
    vectorizer_input = [text if text else "<empty response>" for text in texts]
    try:
        embeddings = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), lowercase=True
        ).fit_transform(vectorizer_input)
        return np.asarray(embeddings.toarray(), dtype=float)
    except ValueError:
        return np.zeros((len(texts), 1), dtype=float)


def _mlx_text_embedding_matrix(
    texts: list[str], model_name: str, max_length: int
) -> np.ndarray:
    try:
        import mlx.core as mx
        from mlx_embeddings.utils import load as mlx_embedding_load
    except (
        ModuleNotFoundError
    ) as exc:  # pragma: no cover - exercised only with optional MLX dependency absent
        raise RuntimeError(
            "MLX semantic embeddings require the optional `mlx-embeddings` package. "
            "Install it in the local environment before setting RE_UQ_ACSE_EMBEDDING_BACKEND=mlx."
        ) from exc

    if model_name not in _MLX_EMBEDDING_MODEL_CACHE:
        _MLX_EMBEDDING_MODEL_CACHE[model_name] = mlx_embedding_load(model_name)
    model, tokenizer = _MLX_EMBEDDING_MODEL_CACHE[model_name]
    inputs = tokenizer.batch_encode_plus(
        [text if text else "<empty response>" for text in texts],
        return_tensors="mlx",
        padding=True,
        truncation=True,
        max_length=max(1, int(max_length)),
    )
    outputs = model(inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
    text_embeds = getattr(outputs, "text_embeds", None)
    if text_embeds is None and isinstance(outputs, Mapping):
        text_embeds = outputs.get("text_embeds")
    if text_embeds is None:
        raise RuntimeError(
            f"MLX embedding model {model_name!r} did not return `text_embeds`; "
            "use a model supported by `mlx-embeddings` for text embedding."
        )
    mx.eval(text_embeds)
    return np.asarray(text_embeds.tolist(), dtype=float)


def apply_acse_embedding_env(config: Mapping[str, Any]) -> None:
    """Export the run config's ACSE embedding choice as environment variables.

    Uses ``setdefault``, so explicitly exported environment variables still
    take precedence (for example a manual ``RE_UQ_ACSE_MLX_MODEL`` override).
    """
    backend = str(config.get("acse_embedding_backend") or "").strip()
    model_name = str(config.get("acse_embedding_mlx_model") or "").strip()
    if backend:
        os.environ.setdefault(ACSE_EMBEDDING_BACKEND_ENV, backend)
    if model_name:
        os.environ.setdefault(ACSE_MLX_MODEL_ENV, model_name)


def semantic_embedding_backend_label(
    embedding_backend: str | None = None,
    mlx_model_name: str | None = None,
) -> tuple[str, str | None]:
    backend = str(
        embedding_backend
        or os.environ.get(ACSE_EMBEDDING_BACKEND_ENV, ACSE_PROXY_EMBEDDING_BACKEND)
    ).strip()
    backend = backend or ACSE_PROXY_EMBEDDING_BACKEND
    if backend in {"tfidf", ACSE_PROXY_EMBEDDING_BACKEND}:
        return ACSE_PROXY_EMBEDDING_BACKEND, None
    if backend == ACSE_MLX_EMBEDDING_BACKEND:
        model_name = str(
            mlx_model_name or os.environ.get(ACSE_MLX_MODEL_ENV, ACSE_MLX_DEFAULT_MODEL)
        ).strip()
        return f"{ACSE_MLX_EMBEDDING_BACKEND}:{model_name}", model_name
    raise ValueError(
        f"Unknown ACSE semantic embedding backend {backend!r}; "
        f"use {ACSE_PROXY_EMBEDDING_BACKEND!r} or {ACSE_MLX_EMBEDDING_BACKEND!r}."
    )


def semantic_embedding_backend_args(
    backend_label: str | None,
) -> tuple[str | None, str | None]:
    """Recover backend arguments from a persisted backend label.

    Raw run rows and analysis manifests store the resolved label so a later
    process can reproduce the run's embedding choice without inheriting the
    runner's environment. An empty label retains the legacy environment/default
    lookup used by runs created before this provenance field existed.
    """
    label = str(backend_label or "").strip()
    if not label:
        return None, None
    if label in {"tfidf", ACSE_PROXY_EMBEDDING_BACKEND}:
        return ACSE_PROXY_EMBEDDING_BACKEND, None
    prefix = f"{ACSE_MLX_EMBEDDING_BACKEND}:"
    if label.startswith(prefix) and label[len(prefix) :].strip():
        return ACSE_MLX_EMBEDDING_BACKEND, label[len(prefix) :].strip()
    raise ValueError(
        f"Invalid persisted ACSE semantic embedding backend label: {label!r}."
    )


def recorded_semantic_embedding_backend(rows: Iterable[Mapping[str, Any]]) -> str:
    """Return the one resolved embedding backend recorded for a run.

    Missing values are allowed for legacy rows, but conflicting non-empty
    values indicate that rows from incompatible embedding configurations were
    mixed under one analysis request and therefore fail closed.
    """
    labels = {
        str(row.get("semantic_embedding_backend", "")).strip()
        for row in rows
        if str(row.get("semantic_embedding_backend", "")).strip()
    }
    if len(labels) > 1:
        raise ValueError(
            "Raw rows contain multiple semantic embedding backends: "
            f"{sorted(labels)!r}. Analyze each configured run separately."
        )
    if labels:
        return next(iter(labels))
    return semantic_embedding_backend_label()[0]


def semantic_embedding_matrix(
    texts: list[str],
    embedding_backend: str | None = None,
    mlx_model_name: str | None = None,
) -> tuple[np.ndarray, str]:
    backend_label, model_name = semantic_embedding_backend_label(
        embedding_backend, mlx_model_name
    )
    if model_name is None:
        return normalize_embedding_rows(
            _tfidf_text_embedding_matrix(texts)
        ), backend_label
    if backend_label.startswith(f"{ACSE_MLX_EMBEDDING_BACKEND}:"):
        max_length = int(os.environ.get(ACSE_MLX_MAX_LENGTH_ENV, "512"))
        embeddings = _mlx_text_embedding_matrix(
            texts, model_name, max_length=max_length
        )
        return normalize_embedding_rows(embeddings), backend_label
    raise AssertionError(f"Unhandled embedding backend label: {backend_label}")


def acse_cluster_labels_for_embeddings(
    embeddings: np.ndarray,
    distance_threshold: float = ACSE_PROXY_DISTANCE_THRESHOLD,
) -> list[str]:
    matrix = normalize_embedding_rows(embeddings)
    sample_count = int(matrix.shape[0])
    if sample_count <= 1:
        return ["cluster_0"] * sample_count
    distance_matrix = np.clip(1.0 - cosine_similarity(matrix), 0.0, 1.0)
    distance_matrix[np.abs(distance_matrix) < 1e-12] = 0.0
    np.fill_diagonal(distance_matrix, 0.0)
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=float(distance_threshold),
    )
    raw_labels = [int(label) for label in clusterer.fit_predict(distance_matrix)]
    counts = Counter(raw_labels)
    ordered = sorted(counts, key=lambda label: (-counts[label], label))
    remap = {label: f"cluster_{rank}" for rank, label in enumerate(ordered)}
    return [remap[label] for label in raw_labels]


def acse_semantic_diagnostics_from_embeddings(
    embeddings: np.ndarray,
    semantic_embedding_backend: str,
    distance_threshold: float = ACSE_PROXY_DISTANCE_THRESHOLD,
) -> dict[str, Any]:
    matrix = normalize_embedding_rows(embeddings)
    sample_count = int(matrix.shape[0])
    if sample_count == 0:
        return {
            "semantic_embedding_backend": semantic_embedding_backend,
            "semantic_distance_threshold": float(distance_threshold),
            "semantic_cluster_count": 0,
            "semantic_cluster_distribution": "",
            "semantic_cluster_entropy": math.nan,
            "semantic_cluster_variation_ratio": math.nan,
            "semantic_dominant_cluster_share": math.nan,
            "semantic_mean_pairwise_distance": math.nan,
            "semantic_dominant_cluster_mean_distance": math.nan,
            "semantic_uncertainty_score": math.nan,
        }

    if sample_count == 1:
        distribution = {"cluster_0": 1.0}
        return {
            "semantic_embedding_backend": semantic_embedding_backend,
            "semantic_distance_threshold": float(distance_threshold),
            "semantic_cluster_count": 1,
            "semantic_cluster_distribution": label_distribution_json(distribution),
            "semantic_cluster_entropy": 0.0,
            "semantic_cluster_variation_ratio": 0.0,
            "semantic_dominant_cluster_share": 1.0,
            "semantic_mean_pairwise_distance": 0.0,
            "semantic_dominant_cluster_mean_distance": 0.0,
            "semantic_uncertainty_score": 0.0,
        }

    distance_matrix = np.clip(1.0 - cosine_similarity(matrix), 0.0, 1.0)
    distance_matrix[np.abs(distance_matrix) < 1e-12] = 0.0
    np.fill_diagonal(distance_matrix, 0.0)

    cluster_labels = acse_cluster_labels_for_embeddings(matrix, distance_threshold)
    counts = Counter(cluster_labels)
    ordered_counts = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    distribution = {label: count / sample_count for label, count in ordered_counts}
    cluster_entropy = normalized_predictive_entropy(distribution)
    cluster_variation = variation_ratio(distribution)
    dominant_label, dominant_count = ordered_counts[0]

    pairwise = distance_matrix[np.triu_indices(sample_count, k=1)]
    mean_pairwise_distance = float(np.mean(pairwise)) if pairwise.size else 0.0
    dominant_indices = [
        index for index, label in enumerate(cluster_labels) if label == dominant_label
    ]
    if len(dominant_indices) > 1:
        dominant_distances = distance_matrix[np.ix_(dominant_indices, dominant_indices)]
        dominant_pairwise = dominant_distances[
            np.triu_indices(len(dominant_indices), k=1)
        ]
        dominant_mean_distance = (
            float(np.mean(dominant_pairwise)) if dominant_pairwise.size else 0.0
        )
    else:
        dominant_mean_distance = 0.0

    dispersion_denominator = max(float(distance_threshold), 1e-12)
    dispersion_component = min(
        1.0,
        max(mean_pairwise_distance, dominant_mean_distance) / dispersion_denominator,
    )
    entropy_weight = 1.0 - ACSE_PROXY_INTERNAL_DISPERSION_WEIGHT
    uncertainty_score = min(
        1.0,
        entropy_weight * float(cluster_entropy)
        + ACSE_PROXY_INTERNAL_DISPERSION_WEIGHT * dispersion_component,
    )
    if abs(uncertainty_score) < 1e-12:
        uncertainty_score = 0.0
    return {
        "semantic_embedding_backend": semantic_embedding_backend,
        "semantic_distance_threshold": float(distance_threshold),
        "semantic_cluster_count": len(counts),
        "semantic_cluster_distribution": label_distribution_json(distribution),
        "semantic_cluster_entropy": float(cluster_entropy),
        "semantic_cluster_variation_ratio": float(cluster_variation),
        "semantic_dominant_cluster_share": dominant_count / sample_count,
        "semantic_mean_pairwise_distance": mean_pairwise_distance,
        "semantic_dominant_cluster_mean_distance": dominant_mean_distance,
        "semantic_uncertainty_score": float(uncertainty_score),
    }


def acse_semantic_proxy_diagnostics(
    texts: list[str],
    distance_threshold: float = ACSE_PROXY_DISTANCE_THRESHOLD,
    embedding_backend: str | None = None,
    mlx_model_name: str | None = None,
) -> dict[str, Any]:
    cleaned = [re.sub(r"\s+", " ", str(text or "").strip()) for text in texts]
    backend_label, _ = semantic_embedding_backend_label(
        embedding_backend, mlx_model_name
    )
    sample_count = len(cleaned)
    if sample_count == 0:
        return {
            "semantic_embedding_backend": backend_label,
            "semantic_distance_threshold": float(distance_threshold),
            "semantic_cluster_count": 0,
            "semantic_cluster_distribution": "",
            "semantic_cluster_entropy": math.nan,
            "semantic_cluster_variation_ratio": math.nan,
            "semantic_dominant_cluster_share": math.nan,
            "semantic_mean_pairwise_distance": math.nan,
            "semantic_dominant_cluster_mean_distance": math.nan,
            "semantic_uncertainty_score": math.nan,
        }

    if sample_count == 1:
        distribution = {"cluster_0": 1.0}
        return {
            "semantic_embedding_backend": backend_label,
            "semantic_distance_threshold": float(distance_threshold),
            "semantic_cluster_count": 1,
            "semantic_cluster_distribution": label_distribution_json(distribution),
            "semantic_cluster_entropy": 0.0,
            "semantic_cluster_variation_ratio": 0.0,
            "semantic_dominant_cluster_share": 1.0,
            "semantic_mean_pairwise_distance": 0.0,
            "semantic_dominant_cluster_mean_distance": 0.0,
            "semantic_uncertainty_score": 0.0,
        }

    embeddings, backend_label = semantic_embedding_matrix(
        cleaned,
        embedding_backend=embedding_backend,
        mlx_model_name=mlx_model_name,
    )
    return acse_semantic_diagnostics_from_embeddings(
        embeddings, backend_label, distance_threshold
    )


def majority_label(distribution: dict[str, float], label_order: list[str]) -> str:
    if not distribution:
        raise ValueError("Cannot choose a majority label from an empty distribution")
    # Primary sort is highest probability. Constraint: ties must not inflate
    # over-commitment, so break them toward the WEAKEST label (last in
    # label_order); unknown labels sort last and never win a tie.
    order_index = {label: index for index, label in enumerate(label_order)}
    return sorted(
        distribution.items(),
        key=lambda pair: (-float(pair[1]), -order_index.get(pair[0], -1)),
    )[0][0]


def one_hot_distribution(label: str, label_order: list[str]) -> dict[str, float]:
    return {candidate: 1.0 if candidate == label else 0.0 for candidate in label_order}


def task3_score_fields(
    item: dict[str, Any], pred_relation: str, evidence_phrase: Any = ""
) -> dict[str, Any]:
    return {
        "source_item_id": item.get("source_item_id", item.get("item_id", "")),
        "gold_relation": item.get("task3_gold_relation", ""),
        "pred_relation": pred_relation,
        "task2_modality": item.get("task2_modality", ""),
        "task2_text_modality": item.get("task2_text_modality", ""),
        "task2_text_modality_basis": item.get("task2_text_modality_basis", ""),
        "task2_text_modality_parse_status": item.get(
            "task2_text_modality_parse_status", ""
        ),
        "task3_declared_relation": item.get("task3_declared_relation", ""),
        "task3_audit_mode": item.get("task3_audit_mode", ""),
        "task2_requirement": item.get("task2_requirement", ""),
        "evidence_phrase": str(evidence_phrase or ""),
        "evidence_phrase_in_source": evidence_phrase_in_source(
            evidence_phrase, item.get("source_statement", "")
        ),
    }


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
    frame["_modality_sort"] = frame["source_modality"].map(
        lambda value: -ORDINAL_STRENGTH[str(value)]
    )
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
    tolerant_violations = sum(
        1 for value in max_increases if value > float(tolerance) + 1e-12
    )
    return {
        "monotonicity_violations": tolerant_violations / checked,
        "monotonicity_strict_violations": strict_violations / checked,
        "monotonicity_tolerance": float(tolerance),
        "monotonicity_mean_max_increase": float(np.mean(max_increases)),
        "monotonicity_max_increase": float(np.max(max_increases)),
    }


def answer_length_fields(
    raw: Mapping[str, Any], item: Mapping[str, Any]
) -> dict[str, Any]:
    """Answer-length / bloat fields for one raw completion record.

    ``requirement_word_count`` is read from the raw record when available;
    legacy rows fall back to counting words in ``parsed_json.requirement``.
    ``completion_tokens`` is populated only for single-item requests because a
    provider's batched usage is not attributable to an individual answer.
    ``source_word_count`` always comes from the benchmark ``source_statement``.
    """
    parsed = raw.get("parsed_json") if isinstance(raw, Mapping) else None
    requirement_text = ""
    if isinstance(parsed, Mapping):
        requirement_text = str(parsed.get("requirement", "") or "")
    raw_words = raw.get("requirement_word_count") if isinstance(raw, Mapping) else None
    try:
        requirement_words = int(raw_words) if raw_words not in {None, ""} else None
    except (TypeError, ValueError):
        requirement_words = None
    if requirement_words is None:
        requirement_words = word_count(requirement_text) if requirement_text else None
    source_words = word_count(str(item.get("source_statement", "") or "")) or None
    batch_id = str(raw.get("batch_id", "") or "") if isinstance(raw, Mapping) else ""
    try:
        batch_item_count = (
            int(raw.get("batch_item_count", 0) or 0) if isinstance(raw, Mapping) else 0
        )
    except (TypeError, ValueError):
        batch_item_count = 0
    # Provider token usage for a multi-item response is request-level. Do not
    # report that whole-batch total as the length of every individual answer.
    completion_tokens = (
        ""
        if batch_item_count > 1 or (batch_id and batch_item_count != 1)
        else raw.get("usage_completion_tokens", "")
        if isinstance(raw, Mapping)
        else ""
    )
    if completion_tokens is None:
        completion_tokens = ""
    return {
        "requirement_word_count": requirement_words
        if requirement_words is not None
        else "",
        "source_word_count": source_words if source_words is not None else "",
        "length_ratio": (
            requirement_words / source_words
            if requirement_words is not None and source_words
            else ""
        ),
        "completion_tokens": completion_tokens,
    }


def empty_answer_length_fields() -> dict[str, Any]:
    return {
        "requirement_word_count": "",
        "source_word_count": "",
        "length_ratio": "",
        "completion_tokens": "",
    }


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


def bootstrap_seed_metric_delta(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    metric: Callable[[list[dict[str, Any]]], float],
    seed_field: str = "seed_id",
    iterations: int = 1000,
    seed: int = 20260518,
) -> tuple[float, float, float]:
    """Paired seed-clustered bootstrap for ``metric(rows_b) - metric(rows_a)``.

    The two arms score the same benchmark items under different conditions
    (for example `item_context` bare vs document), so their rows are paired by
    seed. Each iteration draws one resample of the union of seed ids and
    evaluates the metric on *both* arms restricted to those seeds before
    differencing; resampling the arms independently would treat paired
    observations as unrelated and overstate the interval. Returns
    ``(delta_point, ci_low, ci_high)`` at the 2.5/97.5 percentiles; NaN when
    either arm is empty.
    """
    if not rows_a or not rows_b:
        return math.nan, math.nan, math.nan

    def by_seed(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(str(row.get(seed_field, "")), []).append(row)
        return groups

    groups_a, groups_b = by_seed(rows_a), by_seed(rows_b)
    seed_keys = np.asarray(sorted(set(groups_a) | set(groups_b)), dtype=object)
    point = metric(rows_b) - metric(rows_a)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(iterations):
        sample_a: list[dict[str, Any]] = []
        sample_b: list[dict[str, Any]] = []
        for seed_key in rng.choice(seed_keys, size=seed_keys.size, replace=True):
            sample_a.extend(groups_a.get(str(seed_key), []))
            sample_b.extend(groups_b.get(str(seed_key), []))
        samples.append(metric(sample_b) - metric(sample_a))
    sample_array = np.asarray([x for x in samples if not math.isnan(x)], dtype=float)
    if sample_array.size == 0:
        return float(point), math.nan, math.nan
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
    sample_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    entropy = normalized_predictive_entropy(distribution)
    vr = variation_ratio(distribution)
    rows = [
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
    if sample_rows is not None:
        embedding_backend, mlx_model_name = semantic_embedding_backend_args(
            raw.get("semantic_embedding_backend")
        )
        diagnostics = acse_semantic_proxy_diagnostics(
            semantic_texts_from_rows(str(raw.get("task", "")), sample_rows),
            embedding_backend=embedding_backend,
            mlx_model_name=mlx_model_name,
        )
        acse_row = score_from_distribution(
            raw,
            item,
            ACSE_PROXY_METHOD,
            distribution,
            valid_n,
            total_n,
            ACSE_PROXY_MEASURE,
            diagnostics["semantic_uncertainty_score"],
            model_name=model_name,
        )
        acse_row.update(diagnostics)
        rows.append(acse_row)
    return rows


def _build_ensemble_disagreement_scores(
    benchmark_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    *,
    group_columns: list[str],
    required_columns: set[str],
    member_key: Callable[[Mapping[str, Any]], str],
    uq_method: str,
    model_name_builder: Callable[[Any, int], str],
    run_group_id: str | None = None,
) -> list[dict[str, Any]]:
    """Score cross-member label disagreement over deterministic outputs.

    Shared by the per-run-id and per-run-group ensemble views; callers supply the
    grouping columns, how an ensemble member is keyed, and the score labels.
    """
    benchmark_by_item = {row["item_id"]: row for row in benchmark_rows}
    raw_rows = filter_raw_rows_to_current_benchmark(benchmark_rows, raw_rows)
    raw_frame = pd.DataFrame.from_records(raw_rows)
    if raw_frame.empty or not required_columns.issubset(raw_frame.columns):
        return []
    deterministic_frame = raw_frame[raw_frame["sample_kind"] == "deterministic"]
    if run_group_id:
        deterministic_frame = deterministic_frame[
            deterministic_frame["run_group_id"].astype(str) == str(run_group_id)
        ]
    if deterministic_frame.empty:
        return []

    scores: list[dict[str, Any]] = []
    for group_values, group_frame in deterministic_frame.groupby(
        group_columns, sort=False
    ):
        group_id, task, item_id = group_values
        item = benchmark_by_item.get(item_id)
        if not item:
            continue
        records = group_frame.to_dict(orient="records")
        total_members = len({member_key(row) for row in records})
        valid_by_member: dict[str, dict[str, Any]] = {}
        for row in records:
            if row.get("parse_status") != "ok" or not isinstance(
                row.get("parsed_json"), dict
            ):
                continue
            valid_by_member.setdefault(member_key(row), row)
        if len(valid_by_member) < 2:
            continue
        valid_rows = list(valid_by_member.values())
        distribution = label_distribution_from_rows(str(task), valid_rows)
        scores.append(
            score_from_distribution(
                valid_rows[0],
                item,
                uq_method,
                distribution,
                valid_n=len(valid_by_member),
                total_n=total_members,
                uncertainty_measure="variation_ratio",
                uncertainty_score=variation_ratio(distribution),
                model_name=model_name_builder(group_id, len(valid_by_member)),
            )
        )
    return scores


def build_ensemble_disagreement_scores(
    benchmark_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _build_ensemble_disagreement_scores(
        benchmark_rows,
        raw_rows,
        group_columns=["run_id", "task", "item_id"],
        required_columns={"sample_kind", "run_id", "task", "item_id", "model"},
        member_key=lambda row: str(row.get("model", "")),
        uq_method="model_ensemble_disagreement",
        model_name_builder=lambda group_id, n_members: (
            f"{ENSEMBLE_MODEL_PREFIX}:{n_members}_models"
        ),
    )


def build_run_group_ensemble_disagreement_scores(
    benchmark_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    run_group_id: str | None = None,
) -> list[dict[str, Any]]:
    return _build_ensemble_disagreement_scores(
        benchmark_rows,
        raw_rows,
        group_columns=["run_group_id", "task", "item_id"],
        required_columns={"sample_kind", "run_group_id", "task", "item_id", "model"},
        member_key=lambda row: (
            f"{row.get('provider_id', '')}:{row.get('model', '')}:{row.get('run_id', '')}"
        ),
        uq_method="model_ensemble_disagreement_run_group",
        model_name_builder=lambda group_id, n_members: (
            f"{ENSEMBLE_MODEL_PREFIX}:run_group:{group_id}:{n_members}_models"
        ),
        run_group_id=run_group_id,
    )


def build_uq_scores(
    benchmark_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    min_valid_samples: int = 1,
    expected_stochastic_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Build per-item UQ score rows.

    ``min_valid_samples`` drops stochastic groups with fewer than that many
    successfully parsed samples. Every stochastic row carries
    ``stochastic_complete`` (``valid_n == total_n``); agreement and unanimity
    summaries must be restricted to complete rows, see
    :func:`repeated_sample_agreement_metrics`.

    ``expected_stochastic_samples`` is the number of repeated samples the run
    PLANNED per item. Without it ``total_n`` is only the number of rows that were
    written, so a sample the run never wrote (crash, budget stop) is invisible and
    the group is reported as complete. When given, ``total_n`` is
    ``max(len(group), expected_stochastic_samples)`` and a missing sample counts
    as a parse failure, i.e. ``stochastic_complete`` is False.

    Duplicate rows for one ``run_id`` + :func:`completion_record_key` (append-only
    raw files, resumed runs) are collapsed by :func:`dedupe_raw_rows` first, so a
    re-requested item cannot be double-counted or scored from its stale attempt.
    """
    benchmark_by_item = {row["item_id"]: row for row in benchmark_rows}
    raw_rows = dedupe_raw_rows(raw_rows)
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
        confidence = confidence_probability(raw, parsed)
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

    for (_, task, item_id, _), group_frame in stochastic_frame.groupby(
        ["model", "task", "item_id", "run_id"], sort=False
    ):
        group = group_frame.to_dict(orient="records")
        item = benchmark_by_item.get(item_id)
        if not item:
            continue
        valid = [
            row
            for row in group
            if row.get("parse_status") == "ok"
            and isinstance(row.get("parsed_json"), dict)
        ]
        if len(valid) < max(1, int(min_valid_samples)):
            continue
        distribution = label_distribution_from_rows(str(task), valid)
        consistency_method = (
            "label_self_consistency" if task == "task1" else "modality_consistency"
        )
        # A sample that was never written is missing, not absent-by-design.
        total_n = max(len(group), int(expected_stochastic_samples or 0))
        scores.extend(
            distribution_score_rows(
                valid[0],
                item,
                distribution,
                len(valid),
                total_n,
                consistency_method,
                sample_rows=valid,
            )
        )

    scores.extend(build_ensemble_disagreement_scores(benchmark_rows, raw_rows))
    return scores


def build_task3_scores(
    task3_items: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    min_valid_samples: int = 1,
) -> list[dict[str, Any]]:
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
        confidence = confidence_probability(raw, parsed)
        gold_relation = item["task3_gold_relation"]
        if gold_relation not in TASK3_RELATIONS:
            continue
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
                **task3_score_fields(
                    item, pred_relation, parsed.get("evidence_phrase", "")
                ),
            }
        )

    raw_frame = pd.DataFrame.from_records(raw_rows)
    if raw_frame.empty or "sample_kind" not in raw_frame.columns:
        return scores
    stochastic_frame = raw_frame[
        (raw_frame["sample_kind"] == "stochastic") & (raw_frame["task"] == "task3")
    ]
    if stochastic_frame.empty:
        return scores

    for (_, item_id, _), group_frame in stochastic_frame.groupby(
        ["model", "item_id", "run_id"], sort=False
    ):
        group = group_frame.to_dict(orient="records")
        item = item_by_id.get(str(item_id))
        if not item:
            continue
        valid = [
            row
            for row in group
            if row.get("parse_status") == "ok"
            and isinstance(row.get("parsed_json"), dict)
        ]
        if len(valid) < max(1, int(min_valid_samples)):
            continue
        if item.get("task3_gold_relation", "") not in TASK3_RELATIONS:
            continue
        distribution = label_distribution_from_rows("task3", valid)
        scores.extend(
            distribution_score_rows(
                valid[0],
                item,
                distribution,
                len(valid),
                len(group),
                "relation_consistency",
                sample_rows=valid,
            )
        )

    return scores


# =============================================================================
# Section 9: Run status, registries, progress, and provider preflight
# =============================================================================
# Bookkeeping for resumable runs: progress summaries, registry upserts, live
# progress CSV/event JSONL writers, warning event derivation, and the
# provider preflight probe.

LOGGER_NAME = "re_uq"
logger = logging.getLogger(LOGGER_NAME)

# Every parse_status the runner can emit. "ok" is the only success value; every
# other status (including "truncated", emitted when a completion hit the token
# limit) counts as a parse failure.
PARSE_STATUS_OK = "ok"
PARSE_STATUS_CATEGORIES = (
    "invalid_json",
    "truncated",
    "invalid_confidence",
    "invalid_label",
    "missing_fields",
    "request_error",
    "missing_batch_result",
)


def configure_run_logging(
    level: str | int = "INFO",
    *,
    log_path: str | Path | None = None,
) -> logging.Logger:
    """Configure the shared ``re_uq`` logger for a CLI run.

    Always logs to stderr; when ``log_path`` is given the same records are also
    appended to that file so a long provider run leaves a durable trace.
    """
    resolved_level = (
        level
        if isinstance(level, int)
        else getattr(logging, str(level).upper(), logging.INFO)
    )
    logger.setLevel(resolved_level)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    # A process may execute several Hydra cells/models. Keep the shared stderr
    # handler, but never retain a previous run's file handler: otherwise later
    # records are copied into every earlier log and file descriptors accumulate.
    for handler in list(logger.handlers):
        target = getattr(handler, "_re_uq_target", None)
        if target not in {None, "stream"}:
            logger.removeHandler(handler)
            handler.close()

    existing_targets = {
        getattr(handler, "_re_uq_target", None) for handler in logger.handlers
    }
    if "stream" not in existing_targets:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler._re_uq_target = "stream"  # type: ignore[attr-defined]
        logger.addHandler(stream_handler)
    if log_path is not None:
        resolved_path = Path(log_path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(resolved_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler._re_uq_target = str(resolved_path)  # type: ignore[attr-defined]
        logger.addHandler(file_handler)
    for handler in logger.handlers:
        handler.setLevel(resolved_level)
    return logger


def is_parse_failure_status(status: Any) -> bool:
    """Any status other than ``ok`` (including ``truncated``) is a parse failure."""
    return str(status or "").strip() != PARSE_STATUS_OK


def parse_status_histogram(raw_rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Categorized parse-status counts with every known category always present."""
    histogram = dict.fromkeys(PARSE_STATUS_CATEGORIES, 0)
    histogram["other"] = 0
    for row in raw_rows:
        status = str(row.get("parse_status", "")).strip()
        if status == PARSE_STATUS_OK:
            continue
        if status in histogram:
            histogram[status] += 1
        else:
            histogram["other"] += 1
    return histogram


def _percentile(values: list[float], fraction: float) -> float | str:
    if not values:
        return ""
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[index])


def run_quality_counters(raw_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Diagnostics shared by the registry row, the finish event, and the summary line.

    Covers the categorized parse-status histogram, retry volume, truncation
    counts, latency percentiles, and provider-reported completion tokens.
    Request-level fields are deduplicated by ``batch_id``; parse and truncation
    statuses remain per output record.
    """
    rows = list(raw_rows)
    histogram = parse_status_histogram(rows)
    retry_total = 0
    truncated_records = 0
    usage_completion_tokens = 0
    usage_seen = False
    latencies: list[float] = []
    # Provider usage, retries, and latency describe requests. Batched responses
    # copy that provenance to each item row, so select one representative per
    # batch while retaining every unbatched row as its own request.
    request_rows: list[Mapping[str, Any]] = []
    seen_batch_ids: set[str] = set()
    for row in rows:
        batch_id = str(row.get("batch_id", "") or "")
        if batch_id:
            if batch_id in seen_batch_ids:
                continue
            seen_batch_ids.add(batch_id)
        request_rows.append(row)

    # try/except rather than contextlib.suppress: this walks every raw row of a
    # run (~70k), where suppress measured ~5.6x slower than the bare handler.
    for row in request_rows:
        try:  # noqa: SIM105
            retry_total += int(row.get("retry_count", 0) or 0)
        except (TypeError, ValueError):
            pass
        usage_value = row.get("usage_completion_tokens", "")
        if usage_value not in ("", None):
            try:
                usage_completion_tokens += int(usage_value)
                usage_seen = True
            except (TypeError, ValueError):
                pass
        latency = row.get("latency_s", "")
        if latency not in ("", None):
            try:  # noqa: SIM105
                latencies.append(float(latency))
            except (TypeError, ValueError):
                pass
    for row in rows:
        if (
            str(row.get("parse_status", "")).strip() == "truncated"
            or str(row.get("finish_reason", "")).strip() == "length"
        ):
            truncated_records += 1
    return {
        "parse_status_histogram": histogram,
        "retry_total": retry_total,
        "truncated_records": truncated_records,
        "latency_p50_s": _percentile(latencies, 0.50),
        "latency_p95_s": _percentile(latencies, 0.95),
        "usage_completion_tokens": usage_completion_tokens if usage_seen else "",
    }


def format_run_quality_line(run_id: str, quality: Mapping[str, Any]) -> str:
    histogram = dict(quality.get("parse_status_histogram", {}) or {})
    nonzero = {key: value for key, value in histogram.items() if value}
    latency_p50 = quality.get("latency_p50_s", "")
    latency_p95 = quality.get("latency_p95_s", "")
    latency_label = (
        f"latency_p50 {float(latency_p50):.2f}s, latency_p95 {float(latency_p95):.2f}s"
        if latency_p50 != "" and latency_p95 != ""
        else "latency_p50 unknown, latency_p95 unknown"
    )
    return (
        f"{run_id}: parse_status {nonzero or 'all ok'}, "
        f"retries {int(quality.get('retry_total', 0) or 0)}, "
        f"truncated {int(quality.get('truncated_records', 0) or 0)}, "
        f"{latency_label}, "
        f"completion_tokens {quality.get('usage_completion_tokens', '') or 'unknown'}"
    )


def run_progress_summary(
    benchmark_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    expected_stochastic_samples: int = 5,
) -> list[dict[str, Any]]:
    """Per (run, model, task) coverage and parse-rate summary.

    Duplicate rows for one ``run_id`` + :func:`completion_record_key` are
    collapsed by :func:`dedupe_raw_rows` first, so a resumed run cannot report a
    record_completion_rate above 1.0 (or hide a failed attempt behind its retry).
    """
    if not raw_rows:
        return []
    raw_rows = dedupe_raw_rows(raw_rows)
    frame = pd.DataFrame.from_records(raw_rows)
    required = {"run_id", "model", "task", "item_id", "sample_kind", "parse_status"}
    if frame.empty or not required.issubset(frame.columns):
        return []
    benchmark_item_ids = {str(row["item_id"]) for row in benchmark_rows}
    benchmark_item_count = len(benchmark_item_ids)
    expected_stochastic_samples = max(0, int(expected_stochastic_samples))
    rows: list[dict[str, Any]] = []
    for (run_id, model, task), group_frame in frame.groupby(
        ["run_id", "model", "task"], sort=False
    ):
        group_frame = group_frame[
            group_frame["item_id"].astype(str).isin(benchmark_item_ids)
        ]
        if group_frame.empty:
            continue
        deterministic = group_frame[group_frame["sample_kind"] == "deterministic"]
        stochastic = group_frame[group_frame["sample_kind"] == "stochastic"]
        ok_count = int((group_frame["parse_status"] == "ok").sum())
        deterministic_items = deterministic["item_id"].astype(str).nunique()
        stochastic_items = stochastic["item_id"].astype(str).nunique()
        if expected_stochastic_samples:
            stochastic_counts = stochastic.groupby("item_id", sort=False).size()
            stochastic_complete_items = int(
                (stochastic_counts >= expected_stochastic_samples).sum()
            )
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
                "record_completion_rate": observed_records / expected_records
                if expected_records
                else math.nan,
                "parse_success_rate": ok_count / observed_records
                if observed_records
                else math.nan,
                "deterministic_records": len(deterministic),
                "deterministic_ok": int((deterministic["parse_status"] == "ok").sum()),
                "deterministic_item_coverage": deterministic_items
                / benchmark_item_count
                if benchmark_item_count
                else math.nan,
                "stochastic_records": len(stochastic),
                "stochastic_ok": int((stochastic["parse_status"] == "ok").sum()),
                "stochastic_item_coverage": stochastic_items / benchmark_item_count
                if benchmark_item_count
                else math.nan,
                "stochastic_complete_item_rate": stochastic_complete_items
                / benchmark_item_count
                if benchmark_item_count
                else math.nan,
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
        tasks = (
            set(group_frame["task"].astype(str).tolist())
            if "task" in group_frame.columns
            else set()
        )
        if not expected_task_set.issubset(tasks):
            continue
        completion_columns = [
            "record_completion_rate",
            "deterministic_item_coverage",
            "stochastic_complete_item_rate",
        ]
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


def pending_completion_jobs(
    jobs: Iterable[Mapping[str, Any]],
    raw_rows: Iterable[Mapping[str, Any]],
    run_id: str,
) -> list[dict[str, Any]]:
    # Map each completed record to the job_config_sha it was produced under so
    # resume can re-run rows whose planned config has since changed. Legacy rows
    # predate the field and fall back to the old key-only match.
    completed: dict[tuple[str, str, str, str, int], str] = {}
    for row in raw_rows:
        if (
            str(row.get("run_id", "")) == str(run_id)
            and str(row.get("parse_status", "")) == "ok"
        ):
            completed[completion_record_key(row)] = str(row.get("job_config_sha", ""))

    pending: list[dict[str, Any]] = []
    legacy_reuse_count = 0
    for job in jobs:
        key = completion_record_key(job)
        if key not in completed:
            pending.append(dict(job))
            continue
        existing_sha = completed[key]
        if not existing_sha:
            legacy_reuse_count += 1
            continue
        if existing_sha != str(job.get("job_config_sha", "")):
            pending.append(dict(job))
    if legacy_reuse_count:
        logger.warning(
            "%s: reused %d cached record(s) without job_config_sha; "
            "resuming without config comparison (legacy cache)",
            run_id,
            legacy_reuse_count,
        )
    return pending


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
    batch_order: str = DEFAULT_BATCH_ORDER,
    item_context: str = DEFAULT_ITEM_CONTEXT,
    notes: str = "",
) -> dict[str, Any]:
    # `request_extra_body` is flattened into a JSON string below, which would
    # hide a credential from a check on the finished row: validate the provider
    # provenance while it is still structured.
    assert_no_credential_shaped_values(
        {
            "api_key_env": api_key_env,
            "base_url": base_url,
            "request_extra_body": request_extra_body,
            "server_model_probe": server_model_probe,
        },
        where="run registry row",
    )
    task_list = normalize_task_filter(tasks)
    run_rows = [
        row
        for row in raw_rows
        if str(row.get("run_id", "")) == str(run_id)
        and str(row.get("model", "")) == str(model)
        and str(row.get("task", "")) in set(task_list)
    ]
    progress = run_progress_summary(
        benchmark_rows,
        run_rows,
        expected_stochastic_samples=expected_stochastic_samples,
    )
    task_progress = [
        row for row in progress if str(row.get("task", "")) in set(task_list)
    ]
    benchmark_item_count = len({str(row["item_id"]) for row in benchmark_rows})
    expected_records = (
        benchmark_item_count
        * len(task_list)
        * (1 + max(0, int(expected_stochastic_samples)))
    )
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
    deterministic_coverages = [
        float(row.get("deterministic_item_coverage", 0) or 0) for row in task_progress
    ]
    stochastic_coverages = [
        float(row.get("stochastic_complete_item_rate", 0) or 0) for row in task_progress
    ]
    complete = (
        observed_records >= expected_records
        and len(task_progress) >= len(task_list)
        and all(value >= 1.0 for value in deterministic_coverages)
        and all(value >= 1.0 for value in stochastic_coverages)
    )
    resolved_status = status or ("complete" if complete else "partial")
    config_row = next(
        (row for row in run_rows if str(row.get("sample_kind", "")) == "deterministic"),
        run_rows[0] if run_rows else {},
    )
    run_config_shas = sorted(
        {
            str(row.get("job_config_sha", ""))
            for row in run_rows
            if row.get("job_config_sha")
        }
    )
    config_sha = sha256_text("\n".join(run_config_shas)) if run_config_shas else ""
    quality = run_quality_counters(run_rows)
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
        "prompt_version": str(config_row.get("prompt_version", "")),
        "temperature": config_row.get("temperature", ""),
        "top_p": config_row.get("top_p", ""),
        "config_sha": config_sha,
        "expected_records": expected_records,
        "observed_records": observed_records,
        "parse_success_rate": ok_records / observed_records if observed_records else "",
        "deterministic_item_coverage": min(deterministic_coverages)
        if deterministic_coverages
        else "",
        "stochastic_complete_item_rate": min(stochastic_coverages)
        if stochastic_coverages
        else "",
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
        "structured_output": normalize_structured_output_mode(
            structured_output, json_mode=json_mode
        ),
        "request_extra_body": compact_json(request_extra_body),
        "server_model_probe": compact_json(server_model_probe),
        "batch_order": normalize_batch_order(batch_order),
        "item_context": normalize_item_context(item_context),
        "parse_status_histogram": compact_json(
            {
                key: value
                for key, value in quality["parse_status_histogram"].items()
                if value
            }
        ),
        "retry_total": quality["retry_total"],
        "truncated_records": quality["truncated_records"],
        "latency_p50_s": quality["latency_p50_s"],
        "latency_p95_s": quality["latency_p95_s"],
        "usage_completion_tokens": quality["usage_completion_tokens"],
        "notes": notes,
    }


def upsert_run_registry_row(
    path: str | Path,
    row: Mapping[str, Any],
    fieldnames: list[str] | None = None,
) -> None:
    """Insert-or-replace one registry row under an advisory lock, atomically.

    Concurrent runners share one registry CSV per dataset/variant, so the
    read-modify-write cycle is serialized with ``file_lock`` and published with
    a temp file + ``os.replace`` so readers never observe a half-written CSV.
    """
    path = Path(path)
    key_fields = ["run_id", "profile_id", "model", "dataset_id", "benchmark_variant"]
    row_key = tuple(str(row.get(field, "")) for field in key_fields)
    with file_lock(path):
        rows = read_csv_rows(path) if path.exists() else []
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
        text = _csv_frame(
            output_rows, fieldnames=fieldnames or RUN_REGISTRY_FIELDS
        ).to_csv(index=False)
        atomic_write_text(path, text)


# Task 3 writes into its own registry file but shares the row contract and the
# same concurrency guarantees.
upsert_task3_registry_row = upsert_run_registry_row


def registry_row_compatibility_issues(
    row: Mapping[str, Any],
    *,
    run_group_id: str,
    benchmark_item_count: int,
    expected_stochastic_samples: int,
    required_tasks: Iterable[str] = ("task2",),
    exact_tasks: Iterable[str] | None = None,
    expected_batch_order: str | None = None,
    expected_batch_size: int | None = None,
    allow_partial_benchmark: bool = False,
    allow_missing_batch_order: bool = False,
) -> list[str]:
    """Explain why a registry row does not match an evaluation plan.

    ``expected_records`` encodes both the number of planned benchmark items and
    stochastic samples. Validating it prevents deterministic-only or partial
    ablations from being selected merely because they are newer and marked
    complete.
    """
    issues: list[str] = []
    if str(row.get("status", "")) != "complete":
        issues.append("status is not complete")
    if str(row.get("run_group_id", "")) != str(run_group_id):
        issues.append(f"run_group_id is not {run_group_id!r}")

    tasks = [
        value.strip() for value in str(row.get("tasks", "")).split(",") if value.strip()
    ]
    task_set = set(tasks)
    missing_tasks = set(required_tasks) - task_set
    if missing_tasks:
        issues.append(f"missing tasks: {','.join(sorted(missing_tasks))}")
    if exact_tasks is not None and task_set != set(exact_tasks):
        issues.append(f"tasks are {','.join(tasks) or 'empty'}")

    samples_per_item = 1 + max(0, int(expected_stochastic_samples))
    records_per_item = len(tasks) * samples_per_item
    try:
        expected_records = int(row.get("expected_records", 0) or 0)
    except (TypeError, ValueError):
        expected_records = 0
    if (
        not records_per_item
        or expected_records <= 0
        or expected_records % records_per_item
    ):
        issues.append(
            "expected_records is incompatible with tasks and stochastic samples"
        )
    else:
        planned_items = expected_records // records_per_item
        if allow_partial_benchmark:
            if not 0 < planned_items <= int(benchmark_item_count):
                issues.append(
                    "planned benchmark size is outside the available benchmark"
                )
        elif planned_items != int(benchmark_item_count):
            issues.append(
                f"planned benchmark size is {planned_items}, expected {int(benchmark_item_count)}"
            )

    try:
        observed_records = int(row.get("observed_records", 0) or 0)
    except (TypeError, ValueError):
        observed_records = 0
    if expected_records > 0 and observed_records < expected_records:
        issues.append("observed_records is below expected_records")

    def coverage_at_least_one(field: str) -> bool:
        try:
            return float(row.get(field, 0) or 0) >= 1.0
        except (TypeError, ValueError):
            return False

    if not coverage_at_least_one("deterministic_item_coverage"):
        issues.append("deterministic coverage is incomplete")
    if expected_stochastic_samples > 0 and not coverage_at_least_one(
        "stochastic_complete_item_rate"
    ):
        issues.append("stochastic coverage is incomplete")

    if expected_batch_order is not None:
        recorded_order = str(row.get("batch_order", "") or "")
        if not recorded_order:
            if not allow_missing_batch_order:
                issues.append("batch_order is missing")
        elif normalize_batch_order(recorded_order) != normalize_batch_order(
            expected_batch_order
        ):
            issues.append(f"batch_order is {recorded_order!r}")
    if expected_batch_size is not None:
        try:
            recorded_batch_size = int(row.get("batch_size", 0) or 0)
        except (TypeError, ValueError):
            recorded_batch_size = 0
        if recorded_batch_size != int(expected_batch_size):
            issues.append(
                f"batch_size is {recorded_batch_size}, expected {int(expected_batch_size)}"
            )
    return issues


def run_events_path(
    root: str | Path,
    dataset_id: str | None = None,
    variant: str | None = None,
    *,
    run_id: Any = None,
    smoke: bool | None = None,
) -> Path:
    return resolve_run_artifact_path(
        artifact_path(
            Path(root) / "data/processed/run_events.jsonl", dataset_id, variant
        ),
        run_id=run_id,
        smoke=smoke,
    )


def run_progress_live_path(
    root: str | Path,
    dataset_id: str | None = None,
    variant: str | None = None,
    *,
    run_id: Any = None,
    smoke: bool | None = None,
) -> Path:
    return resolve_run_artifact_path(
        artifact_path(
            Path(root) / "data/processed/run_progress_live.csv", dataset_id, variant
        ),
        run_id=run_id,
        smoke=smoke,
    )


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
    request_error_records = sum(
        1 for row in raw_rows if str(row.get("parse_status", "")) == "request_error"
    )
    observed_api_calls = len(
        {
            str(row.get("batch_id") or f"single:{row.get('request_index', index)}")
            for index, row in enumerate(raw_rows)
        }
    )
    elapsed_s = 0.0
    if started_monotonic is not None:
        elapsed_s = max(
            0.0,
            float(now_monotonic if now_monotonic is not None else time.monotonic())
            - float(started_monotonic),
        )
    records_per_s = observed_records / elapsed_s if elapsed_s > 0 else 0.0
    remaining_records = max(0, int(expected_records) - observed_records)
    if remaining_records == 0:
        # A finished run has no ETA to guess at; report 0 instead of "unknown".
        eta_s: float | str = 0.0
    else:
        eta_s = remaining_records / records_per_s if records_per_s > 0 else ""
    parse_failure_records = observed_records - ok_records
    return {
        **run_quality_counters(raw_rows),
        "expected_records": int(expected_records),
        "observed_records": observed_records,
        "record_completion_rate": observed_records / expected_records
        if expected_records
        else math.nan,
        "ok_records": ok_records,
        "parse_failure_records": parse_failure_records,
        "parse_success_rate": ok_records / observed_records
        if observed_records
        else math.nan,
        "parse_failure_rate": parse_failure_records / observed_records
        if observed_records
        else 0.0,
        "request_error_records": request_error_records,
        "request_error_rate": request_error_records / observed_records
        if observed_records
        else 0.0,
        "expected_api_calls": int(expected_api_calls),
        "observed_api_calls": observed_api_calls,
        "api_call_completion_rate": observed_api_calls / expected_api_calls
        if expected_api_calls
        else math.nan,
        "elapsed_s": elapsed_s,
        "records_per_s": records_per_s,
        "eta_s": eta_s,
    }


def _duration_label(seconds: Any) -> str:
    if (
        seconds == ""
        or seconds is None
        or (isinstance(seconds, float) and math.isnan(seconds))
    ):
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
    append_jsonl(path, {"created_at_utc": utc_now_iso(), **dict(event)})


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
    warn_after = int(
        logging_config.get(
            "warn_after_records", DEFAULT_RUN_LOGGING["warn_after_records"]
        )
    )
    if observed_records < warn_after:
        return []

    events: list[dict[str, Any]] = []
    parse_failure_rate = float(counters.get("parse_failure_rate", 0.0) or 0.0)
    parse_threshold = float(
        logging_config.get(
            "warn_parse_failure_rate", DEFAULT_RUN_LOGGING["warn_parse_failure_rate"]
        )
    )
    if (
        "parse_failure_rate" not in emitted_warning_types
        and parse_failure_rate > parse_threshold
    ):
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
    request_threshold = float(
        logging_config.get(
            "warn_request_error_rate", DEFAULT_RUN_LOGGING["warn_request_error_rate"]
        )
    )
    if (
        "request_error_rate" not in emitted_warning_types
        and request_error_rate > request_threshold
    ):
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
    prompt_version: str = "v1",
    json_mode: bool = False,
    structured_output: str = "none",
    response_format: Mapping[str, Any] | None = None,
    extra_body: Mapping[str, Any] | None = None,
    instructor_mode: str = "json",
    validation_retries: int = 2,
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
    if (
        resolved_response_format is None
        and json_mode
        and structured_output == "none"
        and not (
            isinstance(resolved_extra_body, Mapping)
            and "response_format" in resolved_extra_body
        )
    ):
        resolved_response_format = {"type": "json_object"}
    preflight_job = {
        "host": host,
        "model": model,
        "task": "task1",
        "temperature": 0.0,
        "top_p": 1.0,
        "timeout_s": timeout_s,
        "api_key_env": api_key_env,
        "structured_output": structured_output,
        "instructor_mode": instructor_mode,
        "validation_retries": validation_retries,
    }
    runner = completion_runner_for_job(preflight_job, completion_fn)
    completion = runner(
        **completion_kwargs_for_job(
            preflight_job,
            prompt=LOGPROB_PROBE_PROMPT,
            max_tokens=48,
            response_format=resolved_response_format,
            extra_body=resolved_extra_body,
            batched=False,
        )
    )
    if (
        normalize_structured_output_mode(structured_output, json_mode=json_mode)
        == "instructor"
    ):
        _, parse_status = parse_instructor_task_response(
            "task1", completion.get("raw_text", "")
        )
    else:
        confidence_scale = confidence_scale_for_record(
            {"prompt_version": prompt_version}
        )
        _, parse_status = parse_task_response(
            "task1",
            completion.get("raw_text", ""),
            confidence_scale=confidence_scale,
        )
    return {
        "ok": bool(completion.get("ok")) and parse_status == "ok",
        "parse_status": parse_status,
        "error": completion.get("error", ""),
        "latency_s": completion.get("latency_s", ""),
        "raw_text": str(completion.get("raw_text", ""))[:240],
    }


def select_model_run_rows(
    rows: list[dict[str, Any]], run_id: str, model: str, tasks: Iterable[str]
) -> list[dict[str, Any]]:
    task_set = {str(task) for task in tasks}
    return [
        row
        for row in rows
        if str(row.get("run_id", "")) == str(run_id)
        and str(row.get("model", "")) == str(model)
        and str(row.get("task", "")) in task_set
    ]


def preflight_profile(
    profile: Mapping[str, Any],
    *,
    model: str,
    prompt_version: str,
    completion_fn: Callable[..., dict[str, Any]] = chat_completion,
) -> dict[str, Any]:
    preflight = provider_preflight(
        host=profile["base_url"],
        model=model,
        api_key_env=profile["api_key_env"],
        timeout_s=int(profile["timeout_s"]),
        prompt_version=prompt_version,
        json_mode=bool(profile["json_mode"]),
        structured_output=str(profile.get("structured_output", "none")),
        response_format=profile.get("response_format"),
        extra_body=profile.get("extra_body"),
        instructor_mode=str(profile.get("instructor_mode", "json")),
        validation_retries=int(profile.get("validation_retries", 2)),
        completion_fn=completion_fn,
    )
    if not preflight["ok"]:
        raise RuntimeError(
            f"Provider preflight failed for {profile['profile_id']} / {model}: {preflight}"
        )
    return preflight


def emit_warning_events(
    counters: Mapping[str, Any],
    *,
    logging_config: Mapping[str, Any],
    emitted_warning_types: set[str],
    context: Mapping[str, Any],
    events_path: str | Path,
) -> None:
    run_id = str(context.get("run_id", ""))
    for warning_event in warning_events_for_counters(
        counters, logging_config, emitted_warning_types
    ):
        emitted_warning_types.add(str(warning_event["warning_type"]))
        warning_event.update(context)
        if logging_config["write_event_jsonl"]:
            append_run_event(events_path, warning_event)
        logger.warning("%s: %s", run_id, warning_event["message"])


# =============================================================================
# Section 10: Paper-facing exports, summaries, and figures
# =============================================================================
# Preliminary and final result snapshots, prompt-sensitivity summaries, weak
# modality probe summaries, over-commitment metrics at the 0.80 / 0.90
# thresholds, calibration / selective-deferral metrics, qualitative example
# extraction, the UQ method inventory, and figure rendering.


def preliminary_result_paths(
    root: str | Path, variant: str | None = None, dataset_id: str | None = None
) -> dict[str, Path]:
    root = Path(root)
    return {
        "scores": artifact_path(
            root / "data/processed/uq_scores_preliminary.csv", dataset_id, variant
        ),
        "summary": artifact_path(
            root / "data/processed/metrics_summary_preliminary.csv", dataset_id, variant
        ),
        "progress": artifact_path(
            root / "data/processed/run_progress_preliminary.csv", dataset_id, variant
        ),
        "table": artifact_path(
            root / "outputs/preliminary_results_table.md", dataset_id, variant
        ),
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
    scored_benchmark_rows = benchmark_rows_with_current_raw_outputs(
        benchmark_rows, raw_rows
    )
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
        "stochastic_complete",
        "requirement_word_count",
        "source_word_count",
        "length_ratio",
        "completion_tokens",
        "y_true",
        "y_pred",
        "p_yes",
        "confidence",
        "uncertainty_score",
        "uncertainty_measure",
        "label_distribution",
        "semantic_embedding_backend",
        "semantic_distance_threshold",
        "semantic_cluster_count",
        "semantic_cluster_distribution",
        "semantic_cluster_entropy",
        "semantic_cluster_variation_ratio",
        "semantic_dominant_cluster_share",
        "semantic_mean_pairwise_distance",
        "semantic_dominant_cluster_mean_distance",
        "semantic_uncertainty_score",
        "gold_modality",
        "pred_modality",
        "text_modality",
        "text_modality_basis",
        "text_modality_multi_modal",
        "text_modality_modals_found",
        "text_modality_parse_status",
        "text_modality_correct",
        "label_text_consistent",
        "text_overcommit",
        "text_undercommit",
        "strict_text_overcommit",
        "text_high_conf_overcommit_80",
        "text_high_conf_overcommit_90",
        "strict_text_high_conf_overcommit_80",
        "strict_text_high_conf_overcommit_90",
        "label_correct_text_overcommit_80",
        "label_correct_text_overcommit_90",
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
        "heuristic_text_modality_rate",
        "label_text_consistency",
        "text_over_commitment",
        "text_over_commitment_n_numerator",
        "text_over_commitment_n_denominator",
        "text_over_commitment_n_unknown_excluded",
        "text_over_commitment_lower_bound",
        "text_over_commitment_upper_bound",
        "strict_text_over_commitment",
        "strict_text_over_commitment_n_numerator",
        "strict_text_over_commitment_n_denominator",
        "strict_text_over_commitment_n_unknown_excluded",
        "strict_text_over_commitment_lower_bound",
        "strict_text_over_commitment_upper_bound",
        "text_modality_negated_rate",
        "text_modality_multi_modal_rate",
        "text_under_commitment",
        "text_high_conf_overcommit_80",
        "text_high_conf_overcommit_90",
        "label_correct_text_overcommit_80",
        "label_correct_text_overcommit_90",
        "mean_requirement_word_count",
        "mean_length_ratio",
        "strengthening_rate_by_length_tercile",
        "mean_requirement_word_count_by_source_modality",
        "repeated_sample_unanimity",
        "agreement_n_complete",
        "agreement_n_incomplete_excluded",
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


def score_base(
    raw: dict[str, Any],
    item: dict[str, Any],
    uq_method: str,
    valid_n: int,
    total_n: int,
) -> dict[str, Any]:
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
        # True when every requested repeated sample parsed. Published
        # agreement / unanimity metrics are restricted to these rows.
        "stochastic_complete": bool(total_n > 0 and valid_n == total_n),
        **answer_length_fields(raw, item),
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
        "stochastic_complete": True,
        **empty_answer_length_fields(),
    }


def build_rule_baseline_scores(
    benchmark_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    for item in benchmark_rows:
        predicted_modality = rule_based_source_modality(item["source_statement"])
        if predicted_modality is None:
            raise ValueError(
                f"Could not parse source modality for {item.get('item_id')}: {item.get('source_statement')}"
            )

        pred_yes = 1 if predicted_modality == "mandatory" else 0
        task1_distribution = one_hot_distribution(
            "yes" if pred_yes else "no", class_order_for_task("task1")
        )
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
        task2_distribution = one_hot_distribution(
            predicted_modality, class_order_for_task("task2")
        )
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


def prompt_sensitivity_summary(
    benchmark_rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    scores = build_uq_scores(benchmark_rows, raw_rows)
    frame = pd.DataFrame.from_records(scores)
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for (model, run_id), group_frame in frame.groupby(["model", "run_id"], sort=False):
        group_rows = group_frame.to_dict(orient="records")
        task1 = group_frame[group_frame["task"] == "task1"]
        weak = task1[task1["y_true"].astype(int) == 0].copy()
        rows.append(
            {
                "model": model,
                "prompt_run_id": run_id,
                "n": len(task1),
                "accuracy": accuracy_score(
                    task1["y_true"].astype(int).tolist(),
                    task1["y_pred"].astype(int).tolist(),
                )
                if not task1.empty
                else math.nan,
                "weak_source_high_p_yes_80": high_confidence_overcommitment_rate(
                    group_rows, "task1", 0.80
                ),
                "weak_source_high_p_yes_90": high_confidence_overcommitment_rate(
                    group_rows, "task1", 0.90
                ),
                "mean_weak_p_yes": float(pd.to_numeric(weak["p_yes"]).mean())
                if not weak.empty
                else math.nan,
            }
        )
    return rows


def task2_prompt_sensitivity_summary(
    benchmark_rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
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
            if (
                raw.get("task") != "task2"
                or not item
                or raw.get("parse_status") != "ok"
                or not isinstance(parsed, dict)
            ):
                continue
            pred_modality = normalize_modality(parsed.get("modality"))
            if pred_modality is None:
                continue
            confidence = confidence_probability(raw, parsed)
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
        nice_rows = [
            row for row in score_rows if row["gold_modality"] == "nice_to_have"
        ]
        nice_to_recommended = [
            row for row in nice_rows if row["pred_modality"] == "recommended"
        ]
        rows.append(
            {
                "model": model,
                "prompt_run_id": run_id,
                "n": len(group_frame),
                "valid_n": len(score_rows),
                "parse_success_rate": len(score_rows) / len(group_frame)
                if len(group_frame)
                else math.nan,
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
                "nice_to_have_to_recommended_rate": len(nice_to_recommended)
                / len(nice_rows)
                if nice_rows
                else math.nan,
                "over_commitment": overcommitment_summary_metrics(score_rows)[
                    "over_commitment"
                ]
                if score_rows
                else math.nan,
                "high_conf_overcommit_80": high_confidence_overcommitment_rate(
                    score_rows, "task2", 0.80
                ),
                "high_conf_overcommit_90": high_confidence_overcommitment_rate(
                    score_rows, "task2", 0.90
                ),
            }
        )
    return rows


def weak_modality_probe_summary(
    probe_items: list[dict[str, Any]], raw_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
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
            if (
                raw.get("task") != "task2"
                or not item
                or raw.get("parse_status") != "ok"
                or not isinstance(parsed, dict)
            ):
                continue
            pred_modality = normalize_modality(parsed.get("modality"))
            if pred_modality is None:
                continue
            confidence = confidence_probability(raw, parsed)
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
            if ORDINAL_STRENGTH[row["pred_modality"]]
            > ORDINAL_STRENGTH[row["gold_modality"]]
        ]
        row = {
            "model": model,
            "run_id": run_id,
            "template_id": template_id,
            "sample_kind": sample_kind,
            "n": len(group_frame),
            "valid_n": valid_n,
            "parse_success_rate": valid_n / len(group_frame)
            if len(group_frame)
            else math.nan,
            "accuracy": (pred_counts.get("nice_to_have", 0) / valid_n)
            if valid_n
            else math.nan,
            "to_recommended_rate": (pred_counts.get("recommended", 0) / valid_n)
            if valid_n
            else math.nan,
            "over_commitment": len(over) / valid_n if valid_n else math.nan,
            "high_conf_overcommit_80": (
                len([row for row in over if row["confidence"] >= 0.80]) / valid_n
                if valid_n
                else math.nan
            ),
            "high_conf_overcommit_90": (
                len([row for row in over if row["confidence"] >= 0.90]) / valid_n
                if valid_n
                else math.nan
            ),
            "pred_mandatory_rate": pred_counts.get("mandatory", 0) / valid_n
            if valid_n
            else math.nan,
            "pred_recommended_rate": pred_counts.get("recommended", 0) / valid_n
            if valid_n
            else math.nan,
            "pred_optional_rate": pred_counts.get("optional", 0) / valid_n
            if valid_n
            else math.nan,
            "pred_nice_to_have_rate": pred_counts.get("nice_to_have", 0) / valid_n
            if valid_n
            else math.nan,
            "mean_confidence": float(
                np.mean([row["confidence"] for row in valid_predictions])
            )
            if valid_predictions
            else math.nan,
        }
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            str(row["model"]),
            str(row["run_id"]),
            str(row["sample_kind"]),
            str(row["template_id"]),
        ),
    )


def write_weak_modality_probe_summary(
    summary_rows: list[dict[str, Any]], output_dir: str | Path, suffix: str = ""
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"weak_modality_probe_summary{suffix}.csv"
    markdown_path = output_dir / f"weak_modality_probe_summary{suffix}.md"
    write_csv_rows(
        csv_path, summary_rows, fieldnames=WEAK_MODALITY_PROBE_SUMMARY_FIELDS
    )
    markdown_path.write_text(
        markdown_table(summary_rows, WEAK_MODALITY_PROBE_SUMMARY_FIELDS) + "\n",
        encoding="utf-8",
    )
    return {"csv": csv_path, "markdown": markdown_path}


def grouped(
    rows: Iterable[dict[str, Any]], keys: list[str]
) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
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
        "over_commitment_severity_given_overcommitment": (severity_sum / over)
        if over
        else math.nan,
    }


def unsupported_mandatory_acceptance_rate(
    rows: list[dict[str, Any]], threshold: float
) -> float:
    weak_rows = [
        row
        for row in rows
        if str(row.get("task", "")) == "task1" and int(row["y_true"]) == 0
    ]
    if not weak_rows:
        return math.nan
    accepted = [
        row
        for row in weak_rows
        if row.get("p_yes") != "" and float(row["p_yes"]) >= threshold
    ]
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
            if ORDINAL_STRENGTH[str(row["gold_modality"])]
            < max(ORDINAL_STRENGTH.values())
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


def weak_strengthening_rate(
    rows: list[dict[str, Any]], threshold: float | None = None
) -> float:
    weak_rows = [
        row
        for row in rows
        if str(row.get("task", "")) == "task2"
        and str(row.get("gold_modality", "")) == "nice_to_have"
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


def label_correct_text_overcommit_rate(
    rows: list[dict[str, Any]], threshold: float | None = None
) -> float:
    label_correct_rows = [
        row
        for row in rows
        if str(row.get("task", "")) == "task2" and _truthy(row.get("y_true"))
    ]
    if not label_correct_rows:
        return math.nan
    over = []
    for row in label_correct_rows:
        if not _truthy(row.get("text_overcommit")):
            continue
        if threshold is not None and float(row.get("confidence", 0.0)) < threshold:
            continue
        over.append(row)
    return len(over) / len(label_correct_rows)


def weak_modality_recall(rows: list[dict[str, Any]]) -> float:
    weak_rows = [
        row
        for row in rows
        if str(row.get("task", "")) == "task2"
        and str(row.get("gold_modality", "")) == "nice_to_have"
    ]
    if not weak_rows:
        return math.nan
    preserved = [
        row for row in weak_rows if str(row.get("pred_modality", "")) == "nice_to_have"
    ]
    return len(preserved) / len(weak_rows)


def high_confidence_overcommitment_rate(
    rows: list[dict[str, Any]], task: str, threshold: float
) -> float:
    task_rows = [row for row in rows if str(row.get("task", "")) == task]
    if not task_rows:
        return math.nan
    if task == "task1":
        return unsupported_mandatory_acceptance_rate(task_rows, threshold)
    if task == "task2":
        return task2_high_confidence_overcommitment_rate(
            task_rows, threshold, denominator="all"
        )
    if task == "task3":
        return math.nan
    return math.nan


def is_truthy_strict(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


# Internal short alias retained for the strict-truthy call sites in this module.
_truthy = is_truthy_strict


TEXT_MODALITY_COVERAGE_METRICS = ("text_over_commitment", "strict_text_over_commitment")
TEXT_MODALITY_COVERAGE_SUFFIXES = (
    "_n_numerator",
    "_n_denominator",
    "_n_unknown_excluded",
    "_lower_bound",
    "_upper_bound",
)
LENGTH_TERCILE_COUNT = 3


def empty_text_modality_summary_metrics() -> dict[str, float | str]:
    keys = [
        "text_modality_accuracy",
        "text_modality_accuracy_all",
        "text_modality_parse_coverage",
        "heuristic_text_modality_rate",
        "label_text_consistency",
        "text_over_commitment",
        "strict_text_over_commitment",
        "text_under_commitment",
        "text_high_conf_overcommit_80",
        "text_high_conf_overcommit_90",
        "label_correct_text_overcommit_80",
        "label_correct_text_overcommit_90",
        "text_modality_multi_modal_rate",
        "text_modality_negated_rate",
    ]
    keys += [
        f"{metric}{suffix}"
        for metric in TEXT_MODALITY_COVERAGE_METRICS
        for suffix in TEXT_MODALITY_COVERAGE_SUFFIXES
    ]
    return dict.fromkeys(keys, "")


def _coverage_adjusted_bounds(
    metric: str,
    numerator: int,
    denominator: int,
    unknown_excluded: int,
) -> dict[str, float | str]:
    """Worst/best-case bounds for a rate whose denominator drops unknown text.

    The published rate uses ``denominator`` (rows whose text modality parsed).
    ``*_lower_bound`` charges every unknown row as non-strengthening and
    ``*_upper_bound`` charges every unknown row as strengthening, both over the
    coverage-complete denominator ``denominator + unknown_excluded``.
    """
    full = denominator + unknown_excluded
    return {
        f"{metric}_n_numerator": numerator,
        f"{metric}_n_denominator": denominator,
        f"{metric}_n_unknown_excluded": unknown_excluded,
        f"{metric}_lower_bound": (numerator / full) if full else "",
        f"{metric}_upper_bound": ((numerator + unknown_excluded) / full)
        if full
        else "",
    }


def text_modality_summary_metrics(rows: list[dict[str, Any]]) -> dict[str, float | str]:
    """Text-modality drift metrics with coverage-adjusted bounds.

    The headline ``text_over_commitment`` / ``strict_text_over_commitment``
    rates are computed over rows whose generated text yielded a modality
    (``text_modality_parse_status == "ok"``). Rows whose text modality is
    ``unknown`` or ``negated`` are excluded from that denominator, so each
    metric is also reported with its numerator, denominator, the number of
    excluded unknown rows, and the worst/best-case bounds that put every
    excluded row on one side.
    """
    diagnostic_rows = [
        row
        for row in rows
        if str(row.get("text_modality_parse_status", "")) in {"ok", "unknown"}
    ]
    if not diagnostic_rows:
        return empty_text_modality_summary_metrics()
    total_rows = len(diagnostic_rows)
    text_rows = [
        row
        for row in diagnostic_rows
        if str(row.get("text_modality_parse_status", "")) == "ok"
    ]
    unknown_excluded = total_rows - len(text_rows)
    coverage = len(text_rows) / total_rows
    correct_over_all = (
        sum(1 for row in diagnostic_rows if _truthy(row.get("text_modality_correct")))
        / total_rows
    )
    negated_rate = (
        sum(
            1
            for row in diagnostic_rows
            if str(row.get("text_modality", "")) == "negated"
        )
        / total_rows
    )
    multi_modal_rate = (
        sum(
            1
            for row in diagnostic_rows
            if _truthy(row.get("text_modality_multi_modal"))
        )
        / total_rows
    )
    if not text_rows:
        return {
            **empty_text_modality_summary_metrics(),
            "text_modality_accuracy_all": correct_over_all,
            "text_modality_parse_coverage": coverage,
            "text_modality_negated_rate": negated_rate,
            "text_modality_multi_modal_rate": multi_modal_rate,
            "label_correct_text_overcommit_80": label_correct_text_overcommit_rate(
                diagnostic_rows, 0.80
            ),
            "label_correct_text_overcommit_90": label_correct_text_overcommit_rate(
                diagnostic_rows, 0.90
            ),
            **_coverage_adjusted_bounds("text_over_commitment", 0, 0, unknown_excluded),
            **_coverage_adjusted_bounds(
                "strict_text_over_commitment", 0, 0, unknown_excluded
            ),
        }
    total = len(text_rows)
    broad_numerator = sum(1 for row in text_rows if _truthy(row.get("text_overcommit")))
    strict_numerator = sum(
        1 for row in text_rows if _truthy(row.get("strict_text_overcommit"))
    )
    return {
        "text_modality_accuracy": sum(
            1 for row in text_rows if _truthy(row.get("text_modality_correct"))
        )
        / total,
        "text_modality_accuracy_all": correct_over_all,
        "text_modality_parse_coverage": coverage,
        "heuristic_text_modality_rate": sum(
            1
            for row in text_rows
            if str(row.get("text_modality_basis", "")) == "heuristic_system_verb"
        )
        / total,
        "label_text_consistency": sum(
            1 for row in text_rows if _truthy(row.get("label_text_consistent"))
        )
        / total,
        "text_over_commitment": broad_numerator / total,
        "strict_text_over_commitment": strict_numerator / total,
        "text_under_commitment": sum(
            1 for row in text_rows if _truthy(row.get("text_undercommit"))
        )
        / total,
        "text_high_conf_overcommit_80": sum(
            1 for row in text_rows if _truthy(row.get("text_high_conf_overcommit_80"))
        )
        / total,
        "text_high_conf_overcommit_90": sum(
            1 for row in text_rows if _truthy(row.get("text_high_conf_overcommit_90"))
        )
        / total,
        "label_correct_text_overcommit_80": label_correct_text_overcommit_rate(
            diagnostic_rows, 0.80
        ),
        "label_correct_text_overcommit_90": label_correct_text_overcommit_rate(
            diagnostic_rows, 0.90
        ),
        "text_modality_negated_rate": negated_rate,
        "text_modality_multi_modal_rate": multi_modal_rate,
        **_coverage_adjusted_bounds(
            "text_over_commitment", broad_numerator, total, unknown_excluded
        ),
        **_coverage_adjusted_bounds(
            "strict_text_over_commitment", strict_numerator, total, unknown_excluded
        ),
    }


def _row_variation_ratio(row: Mapping[str, Any]) -> float | None:
    """Variation ratio (1 - p_majority) for one repeated-sample score row."""
    if str(row.get("uncertainty_measure", "")) == "variation_ratio":
        try:
            return float(row["uncertainty_score"])
        except (KeyError, TypeError, ValueError):
            pass
    raw_distribution = row.get("label_distribution", "")
    if not raw_distribution:
        return None
    try:
        distribution = (
            json.loads(raw_distribution)
            if isinstance(raw_distribution, str)
            else dict(raw_distribution)
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    values = [float(value) for value in distribution.values()] if distribution else []
    return 1.0 - max(values) if values else None


AGREEMENT_CONSISTENCY_METHODS = {
    "label_self_consistency",
    "modality_consistency",
    "relation_consistency",
}


def repeated_sample_agreement_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Repeated-sample agreement restricted to coverage-complete rows.

    A published agreement / unanimity number may only be computed over items
    where every requested stochastic sample parsed (``stochastic_complete``).
    Items with ``valid_n < total_n`` are excluded and counted in
    ``agreement_n_incomplete_excluded`` so the reported share is auditable.
    """
    repeated: list[dict[str, Any]] = []
    for row in rows:
        try:
            total_n = int(float(row.get("total_n", 0) or 0))
        except (TypeError, ValueError):
            continue
        if total_n <= 1:
            continue
        repeated.append(row)
    consistency_rows = [
        row
        for row in repeated
        if str(row.get("uq_method", "")) in AGREEMENT_CONSISTENCY_METHODS
    ]
    if consistency_rows:
        repeated = consistency_rows
    complete = [
        row for row in repeated if is_truthy_strict(row.get("stochastic_complete"))
    ]
    excluded = len(repeated) - len(complete)
    if not complete:
        return {
            "repeated_sample_unanimity": "",
            "mean_repeated_sample_agreement": "",
            "agreement_n_complete": 0,
            "agreement_n_incomplete_excluded": excluded,
        }
    variation_ratios = [_row_variation_ratio(row) for row in complete]
    usable = [value for value in variation_ratios if value is not None]
    if not usable:
        return {
            "repeated_sample_unanimity": "",
            "mean_repeated_sample_agreement": "",
            "agreement_n_complete": len(complete),
            "agreement_n_incomplete_excluded": excluded,
        }
    return {
        "repeated_sample_unanimity": sum(1 for value in usable if value <= 1e-12)
        / len(usable),
        "mean_repeated_sample_agreement": sum(1.0 - value for value in usable)
        / len(usable),
        "agreement_n_complete": len(complete),
        "agreement_n_incomplete_excluded": excluded,
    }


def _float_or_none(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def empty_length_bloat_metrics() -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "mean_requirement_word_count": "",
        "mean_source_word_count": "",
        "mean_length_ratio": "",
        "mean_completion_tokens": "",
        "length_tercile_bounds": "",
        "strengthening_rate_by_length_tercile": "",
    }
    for index in range(1, LENGTH_TERCILE_COUNT + 1):
        metrics[f"length_tercile_{index}_n"] = ""
        metrics[f"length_tercile_{index}_mean_length_ratio"] = ""
        metrics[f"length_tercile_{index}_text_over_commitment"] = ""
        metrics[f"length_tercile_{index}_strict_text_over_commitment"] = ""
    for modality in MODALITIES:
        metrics[f"mean_requirement_word_count_{modality}"] = ""
    metrics["mean_requirement_word_count_by_source_modality"] = ""
    return metrics


def length_bloat_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Answer-length / bloat metrics and the length-vs-strengthening breakdown.

    Rows are bucketed into ``LENGTH_TERCILE_COUNT`` equal-count terciles of
    ``length_ratio`` (generated requirement words / source statement words) and
    the broad and strict text-strengthening rates are reported per bucket, so a
    bloat/strengthening interaction is visible rather than pooled away.
    """
    metrics = empty_length_bloat_metrics()
    word_counts = [_float_or_none(row.get("requirement_word_count")) for row in rows]
    word_counts = [value for value in word_counts if value is not None]
    source_counts = [_float_or_none(row.get("source_word_count")) for row in rows]
    source_counts = [value for value in source_counts if value is not None]
    completion_tokens = [_float_or_none(row.get("completion_tokens")) for row in rows]
    completion_tokens = [value for value in completion_tokens if value is not None]
    if word_counts:
        metrics["mean_requirement_word_count"] = sum(word_counts) / len(word_counts)
    if source_counts:
        metrics["mean_source_word_count"] = sum(source_counts) / len(source_counts)
    if completion_tokens:
        metrics["mean_completion_tokens"] = sum(completion_tokens) / len(
            completion_tokens
        )

    by_modality: dict[str, list[float]] = {modality: [] for modality in MODALITIES}
    for row in rows:
        value = _float_or_none(row.get("requirement_word_count"))
        modality = str(row.get("source_modality", ""))
        if value is not None and modality in by_modality:
            by_modality[modality].append(value)
    parts = []
    for modality, values in by_modality.items():
        if values:
            mean_value = sum(values) / len(values)
            metrics[f"mean_requirement_word_count_{modality}"] = mean_value
            parts.append(f"{modality}={mean_value:.2f}")
    metrics["mean_requirement_word_count_by_source_modality"] = "|".join(parts)

    ratio_rows = [(row, _float_or_none(row.get("length_ratio"))) for row in rows]
    ratio_rows = [(row, ratio) for row, ratio in ratio_rows if ratio is not None]
    if not ratio_rows:
        return metrics
    ratios = [ratio for _, ratio in ratio_rows]
    metrics["mean_length_ratio"] = sum(ratios) / len(ratios)
    ratio_rows.sort(key=lambda pair: pair[1])
    size = len(ratio_rows)
    bucket_parts = []
    bounds = []
    for index in range(LENGTH_TERCILE_COUNT):
        low = index * size // LENGTH_TERCILE_COUNT
        high = (index + 1) * size // LENGTH_TERCILE_COUNT
        bucket = ratio_rows[low:high]
        label = f"t{index + 1}"
        if not bucket:
            bucket_parts.append(f"{label}:n=0")
            continue
        bucket_ratios = [ratio for _, ratio in bucket]
        bounds.append(f"{label}:[{min(bucket_ratios):.3f},{max(bucket_ratios):.3f}]")
        parsed = [
            row
            for row, _ in bucket
            if str(row.get("text_modality_parse_status", "")) == "ok"
        ]
        denominator = len(parsed)
        broad = sum(1 for row in parsed if _truthy(row.get("text_overcommit")))
        strict = sum(1 for row in parsed if _truthy(row.get("strict_text_overcommit")))
        metrics[f"length_tercile_{index + 1}_n"] = denominator
        metrics[f"length_tercile_{index + 1}_mean_length_ratio"] = sum(
            bucket_ratios
        ) / len(bucket_ratios)
        broad_rate = (broad / denominator) if denominator else ""
        strict_rate = (strict / denominator) if denominator else ""
        metrics[f"length_tercile_{index + 1}_text_over_commitment"] = broad_rate
        metrics[f"length_tercile_{index + 1}_strict_text_over_commitment"] = strict_rate
        broad_text = f"{broad_rate:.4f}" if broad_rate != "" else ""
        strict_text = f"{strict_rate:.4f}" if strict_rate != "" else ""
        bucket_parts.append(
            f"{label}:n={denominator},broad={broad_text},strict={strict_text}"
        )
    metrics["length_tercile_bounds"] = "|".join(bounds)
    metrics["strengthening_rate_by_length_tercile"] = "|".join(bucket_parts)
    return metrics


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
    return sorted(
        examples,
        key=lambda row: (-float(row["risk_score"]), row["model"], row["seed_id"]),
    )[:limit]


def write_qualitative_overcommitment_examples(
    scores: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
    output_dir: str | Path,
    suffix: str = "",
    limit: int = 5,
    threshold: float = 0.80,
) -> dict[str, Path]:
    examples = qualitative_overcommitment_examples(
        scores, benchmark_rows, limit=limit, threshold=threshold
    )
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
    markdown_path.write_text(
        frame.to_markdown(index=False) + "\n"
        if not frame.empty
        else "_No high-confidence over-commitment examples._\n",
        encoding="utf-8",
    )
    return {"csv": csv_path, "markdown": markdown_path}


def calibration_probabilities(rows: list[dict[str, Any]], task: str) -> list[float]:
    if task == "task1":
        return [float(row["p_yes"]) for row in rows]
    return [float(row["confidence"]) for row in rows]


def _error_label(task: str, y_true: int, y_pred: int) -> int:
    if task == "task1":
        return 1 if y_true != y_pred else 0
    if task in {"task2", "task3"}:
        return 1 - y_true
    raise ValueError(f"Unknown task: {task}")


def prediction_error_labels(rows: list[dict[str, Any]], task: str) -> list[int]:
    labels: list[int] = []
    for row in rows:
        if task == "task1":
            labels.append(_error_label(task, int(row["y_true"]), int(row["y_pred"])))
        elif task in {"task2", "task3"}:
            labels.append(_error_label(task, int(row["y_true"]), 0))
        else:
            raise ValueError(f"Unknown task: {task}")
    return labels


def error_detection_auroc(rows: list[dict[str, Any]], task: str) -> float:
    errors: list[int] = []
    uncertainty_scores: list[float] = []
    for row, error in zip(rows, prediction_error_labels(rows, task), strict=True):
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


def _float_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if not math.isnan(number) else math.nan


def _error_label_for_score_row(row: Mapping[str, Any]) -> int:
    task = str(row.get("task", ""))
    if task == "task1":
        return _error_label(task, int(row.get("y_true", 0)), int(row.get("y_pred", 0)))
    return _error_label(task, int(row.get("y_true", 0)), 0)


def _seed_split(
    seed_ids: Iterable[Any], calibration_fraction: float = ACSE_CALIBRATION_FRACTION
) -> set[str]:
    unique = sorted({str(seed_id) for seed_id in seed_ids if str(seed_id)})
    if not unique:
        return set()
    rng = random.Random(ACSE_CALIBRATION_SEED)
    shuffled = list(unique)
    rng.shuffle(shuffled)
    if len(shuffled) == 1:
        return {shuffled[0]}
    calibration_n = max(
        1,
        min(
            len(shuffled) - 1,
            math.ceil(len(shuffled) * float(calibration_fraction)),
        ),
    )
    return set(shuffled[:calibration_n])


def acse_normalized_score_rows(
    scores: list[dict[str, Any]],
    calibration_fraction: float = ACSE_CALIBRATION_FRACTION,
) -> list[dict[str, Any]]:
    acse_rows = [
        row
        for row in scores
        if str(row.get("uq_method", "")) == ACSE_PROXY_METHOD
        and not math.isnan(_float_or_nan(row.get("uncertainty_score", "")))
    ]
    normalized_rows: list[dict[str, Any]] = []
    group_keys = ["run_id", "model", "task", "semantic_embedding_backend"]
    for key, group_rows in grouped(acse_rows, group_keys).items():
        raw_scores = [
            _float_or_nan(row.get("uncertainty_score", "")) for row in group_rows
        ]
        valid_scores = [score for score in raw_scores if not math.isnan(score)]
        if not valid_scores:
            continue
        raw_min = min(valid_scores)
        raw_max = max(valid_scores)
        span = raw_max - raw_min
        calibration_seed_ids = _seed_split(
            [row.get("seed_id", "") for row in group_rows],
            calibration_fraction=calibration_fraction,
        )
        for row in group_rows:
            raw_score = _float_or_nan(row.get("uncertainty_score", ""))
            normalized = (
                0.0
                if span <= 1e-12
                else min(1.0, max(0.0, (raw_score - raw_min) / span))
            )
            split = (
                "calibration"
                if str(row.get("seed_id", "")) in calibration_seed_ids
                else "evaluation"
            )
            if not calibration_seed_ids:
                split = "calibration"
            normalized_rows.append(
                {
                    "run_id": key[0],
                    "model": key[1],
                    "task": key[2],
                    "semantic_embedding_backend": key[3],
                    "item_id": row.get("item_id", ""),
                    "seed_id": row.get("seed_id", ""),
                    "source_modality": row.get("source_modality", ""),
                    "gold_modality": row.get("gold_modality", ""),
                    "pred_modality": row.get("pred_modality", ""),
                    "valid_n": row.get("valid_n", ""),
                    "total_n": row.get("total_n", ""),
                    "y_true": row.get("y_true", ""),
                    "y_pred": row.get("y_pred", ""),
                    "prediction_error": _error_label_for_score_row(row),
                    "acse_raw_uncertainty_score": raw_score,
                    "acse_normalized_uncertainty_score": normalized,
                    "acse_raw_group_min": raw_min,
                    "acse_raw_group_max": raw_max,
                    "acse_calibration_split": split,
                    "semantic_cluster_count": row.get("semantic_cluster_count", ""),
                    "semantic_cluster_entropy": row.get("semantic_cluster_entropy", ""),
                    "semantic_cluster_variation_ratio": row.get(
                        "semantic_cluster_variation_ratio", ""
                    ),
                    "semantic_dominant_cluster_share": row.get(
                        "semantic_dominant_cluster_share", ""
                    ),
                    "semantic_mean_pairwise_distance": row.get(
                        "semantic_mean_pairwise_distance", ""
                    ),
                    "semantic_dominant_cluster_mean_distance": row.get(
                        "semantic_dominant_cluster_mean_distance", ""
                    ),
                }
            )
    return sorted(
        normalized_rows,
        key=lambda row: (
            str(row["model"]),
            str(row["task"]),
            str(row["semantic_embedding_backend"]),
            str(row["item_id"]),
        ),
    )


def _accepted_error_rate(
    rows: list[dict[str, Any]], threshold: float
) -> tuple[int, float, float]:
    accepted = [
        row
        for row in rows
        if _float_or_nan(row.get("acse_normalized_uncertainty_score", "")) <= threshold
    ]
    if not rows:
        return 0, math.nan, math.nan
    coverage = len(accepted) / len(rows)
    if not accepted:
        return 0, coverage, math.nan
    error_rate = sum(int(row["prediction_error"]) for row in accepted) / len(accepted)
    return len(accepted), coverage, error_rate


def _deferred_error_rate(rows: list[dict[str, Any]], threshold: float) -> float:
    deferred = [
        row
        for row in rows
        if _float_or_nan(row.get("acse_normalized_uncertainty_score", "")) > threshold
    ]
    if not deferred:
        return math.nan
    return sum(int(row["prediction_error"]) for row in deferred) / len(deferred)


def _select_threshold_for_error_target(
    rows: list[dict[str, Any]],
    target_error_rate: float,
) -> float | None:
    if not rows:
        return None
    candidates = sorted(
        {
            _float_or_nan(row.get("acse_normalized_uncertainty_score", ""))
            for row in rows
            if not math.isnan(
                _float_or_nan(row.get("acse_normalized_uncertainty_score", ""))
            )
        }
    )
    selected: float | None = None
    for threshold in candidates:
        accepted_n, _, error_rate = _accepted_error_rate(rows, threshold)
        if (
            accepted_n
            and not math.isnan(error_rate)
            and error_rate <= float(target_error_rate) + 1e-12
        ):
            selected = threshold
    return selected


def acse_calibration_diagnostic_rows(
    normalized_rows: list[dict[str, Any]],
    target_error_rates: Iterable[float] = ACSE_TARGET_ACCEPTED_ERROR_RATES,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    group_keys = ["run_id", "model", "task", "semantic_embedding_backend"]
    for key, group_rows in grouped(normalized_rows, group_keys).items():
        calibration_rows = [
            row
            for row in group_rows
            if str(row.get("acse_calibration_split", "")) == "calibration"
        ]
        evaluation_rows = [
            row
            for row in group_rows
            if str(row.get("acse_calibration_split", "")) == "evaluation"
        ]
        evaluation_mode = (
            "heldout_seed_split" if evaluation_rows else "resubstitution_no_heldout"
        )
        evaluation_source_rows = evaluation_rows or calibration_rows
        all_errors = [int(row["prediction_error"]) for row in group_rows]
        all_scores = [
            _float_or_nan(row.get("acse_normalized_uncertainty_score", ""))
            for row in group_rows
        ]
        error_detection = auroc_score(all_errors, all_scores)
        for target in target_error_rates:
            threshold = _select_threshold_for_error_target(
                calibration_rows, float(target)
            )
            if threshold is None:
                calibration_accepted_n, calibration_coverage, calibration_error_rate = (
                    0,
                    0.0,
                    math.nan,
                )
                evaluation_accepted_n, evaluation_coverage, evaluation_error_rate = (
                    0,
                    0.0,
                    math.nan,
                )
                evaluation_deferred_error_rate = math.nan
            else:
                calibration_accepted_n, calibration_coverage, calibration_error_rate = (
                    _accepted_error_rate(
                        calibration_rows,
                        threshold,
                    )
                )
                evaluation_accepted_n, evaluation_coverage, evaluation_error_rate = (
                    _accepted_error_rate(
                        evaluation_source_rows,
                        threshold,
                    )
                )
                evaluation_deferred_error_rate = _deferred_error_rate(
                    evaluation_source_rows, threshold
                )
            rows.append(
                {
                    "run_id": key[0],
                    "model": key[1],
                    "task": key[2],
                    "semantic_embedding_backend": key[3],
                    "target_accepted_error_rate": float(target),
                    "selected_normalized_threshold": ""
                    if threshold is None
                    else threshold,
                    "calibration_n": len(calibration_rows),
                    "calibration_accepted_n": calibration_accepted_n,
                    "calibration_coverage": calibration_coverage,
                    "calibration_accepted_error_rate": calibration_error_rate,
                    "evaluation_mode": evaluation_mode,
                    "evaluation_n": len(evaluation_source_rows),
                    "evaluation_accepted_n": evaluation_accepted_n,
                    "evaluation_coverage": evaluation_coverage,
                    "evaluation_accepted_error_rate": evaluation_error_rate,
                    "evaluation_deferred_error_rate": evaluation_deferred_error_rate,
                    "all_n": len(group_rows),
                    "all_error_rate": sum(all_errors) / len(all_errors)
                    if all_errors
                    else math.nan,
                    "all_error_detection_auroc": error_detection,
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            str(row["model"]),
            str(row["task"]),
            str(row["semantic_embedding_backend"]),
            float(row["target_accepted_error_rate"]),
        ),
    )


def selective_deferral_metrics(
    rows: list[dict[str, Any]],
    task: str,
    defer_fractions: Iterable[float] = (0.10, 0.20),
) -> dict[str, float]:
    pairs: list[tuple[float, int]] = []
    for row, error in zip(rows, prediction_error_labels(rows, task), strict=True):
        value = row.get("uncertainty_score", "")
        if value in {"", None}:
            continue
        try:
            uncertainty = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(uncertainty):
            continue
        pairs.append((uncertainty, error))

    metrics: dict[str, float] = {}
    pairs = sorted(pairs, key=lambda pair: pair[0], reverse=True)
    total = len(pairs)
    for fraction in defer_fractions:
        suffix = f"{round(float(fraction) * 100):02d}"
        if total == 0:
            metrics[f"selective_coverage_defer_{suffix}"] = math.nan
            metrics[f"selective_error_defer_{suffix}"] = math.nan
            continue
        defer_n = min(total, math.ceil(total * float(fraction)))
        retained = pairs[defer_n:]
        metrics[f"selective_coverage_defer_{suffix}"] = len(retained) / total
        metrics[f"selective_error_defer_{suffix}"] = (
            sum(error for _, error in retained) / len(retained)
            if retained
            else math.nan
        )
    return metrics


def headline_risk_ci_fields(
    rows: list[dict[str, Any]],
    task: str,
    iterations: int = 1000,
) -> dict[str, float | str]:
    fields: dict[str, float | str] = {}
    for threshold in HIGH_CONFIDENCE_THRESHOLDS:
        suffix = f"{int(threshold * 100):02d}"
        if task == "task1":
            metric_name = f"unsupported_mandatory_acceptance_{suffix}"

            def metric(
                sample_rows: list[dict[str, Any]], threshold: float = threshold
            ) -> float:
                return unsupported_mandatory_acceptance_rate(sample_rows, threshold)

            _, low, high = bootstrap_seed_metric(rows, metric, iterations=iterations)
            fields[f"{metric_name}_ci_low"] = low
            fields[f"{metric_name}_ci_high"] = high
        elif task == "task2":
            metric_specs = {
                f"high_conf_overcommit_overcommittable_{suffix}": (
                    lambda sample_rows, threshold=threshold: (
                        task2_high_confidence_overcommitment_rate(
                            sample_rows,
                            threshold,
                            denominator="overcommittable",
                        )
                    )
                ),
                f"weak_strengthening_{suffix}": (
                    lambda sample_rows, threshold=threshold: weak_strengthening_rate(
                        sample_rows, threshold
                    )
                ),
                f"label_correct_text_overcommit_{suffix}": (
                    lambda sample_rows, threshold=threshold: (
                        label_correct_text_overcommit_rate(
                            sample_rows,
                            threshold,
                        )
                    )
                ),
            }
            for metric_name, metric in metric_specs.items():
                _, low, high = bootstrap_seed_metric(
                    rows, metric, iterations=iterations
                )
                fields[f"{metric_name}_ci_low"] = low
                fields[f"{metric_name}_ci_high"] = high
    return fields


def text_strengthening_rate(rows: list[dict[str, Any]], strict: bool = False) -> float:
    """Broad (or strict) generated-text strengthening rate over parsed rows."""
    text_rows = [
        row for row in rows if str(row.get("text_modality_parse_status", "")) == "ok"
    ]
    if not text_rows:
        return math.nan
    key = "strict_text_overcommit" if strict else "text_overcommit"
    return sum(1 for row in text_rows if _truthy(row.get(key))) / len(text_rows)


def text_over_commitment_ci_fields(
    rows: list[dict[str, Any]],
    iterations: int = 1000,
    seed: int = 20260518,
) -> dict[str, float | str]:
    """Point estimate, counts, and seed-clustered bootstrap CI for text drift.

    The bootstrap resamples benchmark *seeds* with replacement (via
    :func:`bootstrap_seed_metric`) because the four source-modality variants of
    one seed share a capability and are not independent.
    """
    fields: dict[str, float | str] = {}
    task2_rows = [row for row in rows if str(row.get("task", "")) == "task2"]
    text_rows = [
        row
        for row in task2_rows
        if str(row.get("text_modality_parse_status", "")) == "ok"
    ]
    for metric_name, strict in (
        ("text_over_commitment", False),
        ("strict_text_over_commitment", True),
    ):
        key = "strict_text_overcommit" if strict else "text_overcommit"
        numerator = sum(1 for row in text_rows if _truthy(row.get(key)))
        fields[f"{metric_name}_n_numerator"] = numerator
        fields[f"{metric_name}_n_denominator"] = len(text_rows)
        if not task2_rows or iterations <= 0:
            fields[metric_name] = (
                text_strengthening_rate(task2_rows, strict=strict) if task2_rows else ""
            )
            fields[f"{metric_name}_ci_low"] = ""
            fields[f"{metric_name}_ci_high"] = ""
            continue

        def metric(sample_rows: list[dict[str, Any]], strict: bool = strict) -> float:
            return text_strengthening_rate(sample_rows, strict=strict)

        point, low, high = bootstrap_seed_metric(
            task2_rows, metric, iterations=iterations, seed=seed
        )
        fields[metric_name] = point
        fields[f"{metric_name}_ci_low"] = low
        fields[f"{metric_name}_ci_high"] = high
    return fields


def task3_strengthening_recall(rows: list[dict[str, Any]]) -> float:
    strengthened = [
        row for row in rows if str(row.get("gold_relation", "")) == "strengthens"
    ]
    if not strengthened:
        return math.nan
    detected = [
        row
        for row in strengthened
        if str(row.get("pred_relation", "")) == "strengthens"
    ]
    return len(detected) / len(strengthened)


def task3_false_preserve_rate(rows: list[dict[str, Any]]) -> float:
    strengthened = [
        row for row in rows if str(row.get("gold_relation", "")) == "strengthens"
    ]
    if not strengthened:
        return math.nan
    false_preserve = [
        row for row in strengthened if str(row.get("pred_relation", "")) == "preserves"
    ]
    return len(false_preserve) / len(strengthened)


def task3_evidence_phrase_source_rate(rows: list[dict[str, Any]]) -> float | str:
    evidence_rows = [row for row in rows if str(row.get("evidence_phrase", "")).strip()]
    if not evidence_rows:
        return ""
    return sum(
        1 for row in evidence_rows if _truthy(row.get("evidence_phrase_in_source"))
    ) / len(evidence_rows)


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


def labels_present_in_rows(
    rows: list[dict[str, Any]], field: str, label_order: list[str]
) -> list[str]:
    present = {str(row.get(field, "")) for row in rows}
    labels = [label for label in label_order if label in present]
    return labels or list(label_order)


def metric_summary_by_model_task_method(
    scores: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not scores:
        return []
    frame = pd.DataFrame.from_records(scores)
    summaries: list[dict[str, Any]] = []
    for (model, task, uq_method), group_frame in frame.groupby(
        ["model", "task", "uq_method"], sort=False
    ):
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
            **selective_deferral_metrics(rows, str(task)),
            "parse_failure_rate": float(
                group_frame["parse_failures"].astype(int).sum()
                / max(1, int(group_frame["total_n"].astype(int).sum()))
            ),
            **empty_text_modality_summary_metrics(),
            **empty_length_bloat_metrics(),
            **repeated_sample_agreement_metrics(rows),
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
            summary[f"high_conf_overcommit_{suffix}"] = (
                high_confidence_overcommitment_rate(rows, str(task), threshold)
            )
            if task == "task1":
                summary[f"unsupported_mandatory_acceptance_{suffix}"] = (
                    unsupported_mandatory_acceptance_rate(rows, threshold)
                )
            elif task == "task2":
                summary[f"high_conf_overcommit_all_{suffix}"] = (
                    task2_high_confidence_overcommitment_rate(
                        rows,
                        threshold,
                        denominator="all",
                    )
                )
                summary[f"high_conf_overcommit_overcommittable_{suffix}"] = (
                    task2_high_confidence_overcommitment_rate(
                        rows,
                        threshold,
                        denominator="overcommittable",
                    )
                )
                summary[f"weak_strengthening_{suffix}"] = weak_strengthening_rate(
                    rows, threshold
                )
        if task == "task1":
            p_yes = group_frame["p_yes"].astype(float).tolist()
            monotonicity_metrics = monotonicity_violation_diagnostics(rows, "p_yes")
            summary.update(
                {
                    "f1_or_macro_f1": binary_f1_score(y_true, y_pred),
                    "auroc": auroc_score(y_true, p_yes),
                    "spearman_modality_p_yes": spearman_corr(
                        group_frame["numeric_strength"].astype(float).tolist(), p_yes
                    ),
                    "pearson_modality_p_yes": pearson_corr(
                        group_frame["numeric_strength"].astype(float).tolist(), p_yes
                    ),
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
                    "over_commitment_severity": over_metrics[
                        "over_commitment_severity_all"
                    ],
                    "over_commitment_severity_all": over_metrics[
                        "over_commitment_severity_all"
                    ],
                    "over_commitment_severity_given_overcommitment": over_metrics[
                        "over_commitment_severity_given_overcommitment"
                    ],
                    "weak_recall": weak_modality_recall(rows),
                    **text_metrics,
                    **length_bloat_metrics(rows),
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
                    "evidence_phrase_source_rate": task3_evidence_phrase_source_rate(
                        rows
                    ),
                }
            )
        summaries.append(summary)
    return sorted(
        summaries, key=lambda row: (row["model"], row["task"], row["uq_method"])
    )


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
            "uq_method": ACSE_PROXY_METHOD,
            "survey_family": "semantic clustering UQ",
            "access_requirement": "black-box stochastic answer texts",
            "extra_inference_cost": "reuses K stochastic calls; TF-IDF fallback or optional local MLX embedding pass",
            "headline": "diagnostic",
            "notes": "ACSE-inspired 5-sample semantic dispersion score for ranking and triage; do not treat it as a conformal accept/abstain rule without held-out calibration.",
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


def write_uq_method_inventory(
    output_dir: str | Path, suffix: str = ""
) -> dict[str, Path]:
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
                task1.groupby(["model", "uq_method", "source_modality"], sort=False)[
                    "p_yes"
                ]
                .mean()
                .reset_index()
            )
            x_positions = np.arange(len(MODALITIES))
            for (model, uq_method), group_frame in summary.groupby(
                ["model", "uq_method"], sort=False
            ):
                series = group_frame.set_index("source_modality").reindex(MODALITIES)[
                    "p_yes"
                ]
                ax.plot(
                    x_positions,
                    series.to_numpy(dtype=float),
                    marker="o",
                    linewidth=2,
                    label=f"{model} / {uq_method}",
                )
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
