"""Shared utilities for the modality-conditioned UQ evaluation pipeline.

The module intentionally keeps the durable research contracts in one place:
dataset and benchmark construction, provider request planning, raw-output
parsing, UQ scoring, and compact paper-facing exports. Command-line entry
points should stay thin wrappers around these helpers.
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
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field as dataclass_field
from functools import cache
from itertools import batched, pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, TextIO
from xml.etree import ElementTree as ET

if TYPE_CHECKING:
    import pyarrow as pa

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
    "unresolved_n",
    "no_readable_cue_n",
    *[
        f"{metric}{suffix}"
        for metric in ("strict_text_over_commitment", "text_over_commitment")
        for suffix in (
            "",
            "_n_numerator",
            "_n_denominator",
            "_ci_low",
            "_ci_high",
            "_seed_ci_low",
            "_seed_ci_high",
        )
    ],
    "bootstrap_ci_cluster_field",
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
    "task3_reference_status",
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
# Bootstrap resampling unit. Batched runs sent 16 benchmark items per provider
# request (four whole seeds x four source conditions) and model behaviour is
# decided per request, so the request nests the seed and is the conservative
# cluster. Rows that carry no request id are resolved in two steps: under the
# single-item protocol the request *is* the item, so rows that all carry an
# ``item_id`` cluster on it; rows without one (legacy score rows, synthesised
# slices) fall back to the seed. Clustering single-item rows on the seed would
# put the four source-modality variants of one capability into one cluster,
# and a metric that is constant within every seed (strict strengthening is:
# exactly the nice_to_have quarter) then resamples to a zero-width interval.
DEFAULT_BOOTSTRAP_CLUSTER_FIELD = "batch_id"
BOOTSTRAP_ITEM_CLUSTER_FIELD = "item_id"
BOOTSTRAP_CLUSTER_FALLBACK_FIELD = "seed_id"
# Which of the strict-strengthening rows changed the modal itself. A rewrite
# of "It would be useful if the system could X" to "The system could X" keeps
# the weak modal and only drops the wish frame; the scorer still reads it as
# optional > nice_to_have. A rewrite to "should"/"shall"/"must" is an
# escalation of the obligation. The two are reported apart.
STRICT_OVERCOMMIT_ESCALATION = "modal_escalation"
STRICT_OVERCOMMIT_FRAME_ONLY = "frame_only"
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
    # What the run was actually executed with. `expected_records` folds this in
    # with the item count, so without it a later comparison has to guess the
    # denominator a run was scored against.
    "expected_stochastic_samples",
    "expected_records",
    # `observed_records` counts logical observations, `observed_attempts` the
    # physical raw rows behind them; see AttemptLedger in Section 9.
    "observed_records",
    "observed_attempts",
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
    "parse_repairs",
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
    "write_request_transcripts": True,
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
    """Return the repository root used to resolve relative paths.

    The first directory from the current working directory upwards that holds
    both `AGENTS.md` and `docs/evaluation.md` wins, so commands work from any
    subdirectory. Otherwise the parent of this module's `scripts/` directory is
    used.

    Returns:
        The resolved repository root.
    """
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

    Args:
        base: Configuration to start from.
        overrides: Values that win over `base`. Nested dictionaries are merged
            key by key; any other value replaces the base value outright.

    Returns:
        A deep copy of `base` with `overrides` merged in.
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
    """Load a config file merged over `DEFAULT_CONFIG` (never aliasing it).

    Args:
        path: Config file; a relative path is resolved against `project_root`.
            A missing file yields a copy of `DEFAULT_CONFIG`.

    Returns:
        The merged configuration dictionary.
    """
    root = project_root()
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = root / config_path
    if not config_path.exists():
        return copy.deepcopy(DEFAULT_CONFIG)
    with config_path.open("r", encoding="utf-8") as handle:
        return deep_update(DEFAULT_CONFIG, json.load(handle))


def positive_int(value: Any, name: str) -> int:
    """Parse a setting that must be an integer of at least 1.

    Args:
        value: Raw config or command-line value; it is stripped as a string
            before parsing, so `" 4 "` is accepted.
        name: Setting name quoted in the error message.

    Returns:
        The parsed integer.

    Raises:
        ValueError: If the value is not an integer or is below 1.
    """
    try:
        resolved = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if resolved < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return resolved


def nonnegative_int(value: Any, name: str) -> int:
    """Parse a setting that must be an integer of at least 0.

    Args:
        value: Raw config or command-line value; it is stripped as a string
            before parsing.
        name: Setting name quoted in the error message.

    Returns:
        The parsed integer.

    Raises:
        ValueError: If the value is not an integer or is negative.
    """
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
    """Parse a setting that must be a number of at least 0.

    Args:
        value: Raw config or command-line value; it is stripped as a string
            before parsing.
        name: Setting name quoted in the error message.

    Returns:
        The parsed number.

    Raises:
        ValueError: If the value is not a number or is negative.
    """
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
    """Parse a boolean setting from a bool or a yes/no style string.

    Args:
        value: A bool, or a string such as `"true"`, `"1"`, `"yes"`, `"on"`
            (true) or `"false"`, `"0"`, `"no"`, `"off"` (false), case-insensitive.
        name: Setting name quoted in the error message.

    Returns:
        The parsed boolean.

    Raises:
        ValueError: If the value is not one of the accepted spellings.
    """
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
    """Return True for keys whose value may carry a credential.

    Keys ending in `_env` name an environment variable rather than holding its
    value, so they are provenance and never match here; their *values* are
    validated separately as environment-variable names.

    Args:
        key: Mapping key to classify; non-alphanumeric runs are folded to `_`
            and the comparison is case-insensitive.

    Returns:
        True when the key matches a credential name or substring.
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

    Args:
        value: Structure to check; mappings, lists and tuples are walked
            recursively.
        where: Artifact or context name that prefixes the error message.
        _path: Dotted path of `value` inside the top-level structure, built up
            by the recursion; callers leave it empty.

    Raises:
        ValueError: If a credential-shaped key holds a non-blank value or an
            `*_env` key holds something other than a variable name.
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
    """Return the number of concurrent provider requests to use.

    Args:
        config: Loaded configuration; `llm.concurrency` is used when the
            environment does not override it, falling back to the default config.
        env: Environment to read; defaults to `os.environ`.
        env_var: Environment variable whose non-empty value overrides the config.

    Returns:
        The concurrency as a positive integer.
    """
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
    """Create the standard data, output and prompt directories if they are missing.

    Args:
        root: Repository root; defaults to `project_root`.
    """
    root = root or project_root()
    for rel in [
        "data/raw",
        "data/processed",
        "outputs",
        "prompts",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def read_csv_frame(path: str | Path) -> pd.DataFrame:
    """Read a CSV file with every column as a string and empty cells kept as `""`.

    Args:
        path: CSV file to read.

    Returns:
        The file as a data frame of strings.
    """
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV file as a list of string-valued row dictionaries.

    Args:
        path: CSV file to read; values are read as by `read_csv_frame`.

    Returns:
        One dictionary per row, keyed by column name.
    """
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
    """Write rows to a CSV file, creating parent directories as needed.

    Args:
        path: Destination file; overwritten if it exists.
        rows: Row dictionaries or a data frame.
        fieldnames: Column order to write; missing columns are filled with `""`
            and extra columns are dropped. `None` keeps the rows' own columns.
    """
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
    """Write a CSV file without silently replacing different existing content.

    A new file is written directly. When the file exists with identical text,
    nothing is written. When it exists with different text, the new content
    goes to a candidate file next to it so a reviewed file is not overwritten.

    Args:
        path: Destination file.
        rows: Row dictionaries or a data frame.
        fieldnames: Column order, applied as in `write_csv_rows`.
        candidate_path: Where to write differing content; defaults to
            `<stem>_candidate<suffix>` next to `path`.
        overwrite: Replace an existing file even if its content differs.

    Returns:
        A dictionary with `status` (`"written"`, `"overwritten"`, `"unchanged"`
        or `"candidate_written"`), `path`, and `candidate_path` (empty unless a
        candidate file was written).
    """
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
    """Map a dataset name or alias to its canonical dataset id.

    Args:
        dataset_id: Dataset name; empty, `None`, `"main"` and `"default"` mean
            the NICE dataset, and aliases such as `"hf"` or `"pure_context"` are
            accepted. Case and hyphens are ignored.

    Returns:
        One of `DATASET_NICE`, `DATASET_MLM_TAPT` or `DATASET_PURE`.

    Raises:
        ValueError: If the name is not a known dataset or alias.
    """
    normalized = str(dataset_id or DATASET_NICE).strip().lower().replace("-", "_")
    if normalized in {"", "main", "default", DATASET_NICE}:
        return DATASET_NICE
    if normalized in {DATASET_MLM_TAPT, "hf", "hf_requirements", "limsc_mlm_tapt"}:
        return DATASET_MLM_TAPT
    if normalized in {DATASET_PURE, "pure_context"}:
        return DATASET_PURE
    raise ValueError(f"Unknown dataset_id: {dataset_id}")


def dataset_suffix(dataset_id: str | None) -> str:
    """Return the artifact file-name suffix for a dataset.

    Args:
        dataset_id: Dataset name or alias, as accepted by `normalize_dataset_id`.

    Returns:
        The suffix, empty for the NICE dataset.
    """
    return DATASET_SUFFIXES[normalize_dataset_id(dataset_id)]


def variant_suffix(variant: str | None) -> str:
    """Return the artifact file-name suffix for a benchmark variant.

    Args:
        variant: Benchmark variant, as accepted by `normalize_benchmark_variant`.

    Returns:
        An empty string for the `must` variant and `"_shall"` for `shall`.
    """
    return "" if normalize_benchmark_variant(variant) == "must" else "_shall"


def dataset_variant_suffix(dataset_id: str | None, variant: str | None = None) -> str:
    """Return the combined dataset and variant suffix for artifact file names.

    Args:
        dataset_id: Dataset name or alias.
        variant: Benchmark variant; `None` means `must`.

    Returns:
        The dataset suffix followed by the variant suffix.
    """
    return f"{dataset_suffix(dataset_id)}{variant_suffix(variant)}"


def artifact_path(
    path: str | Path, dataset_id: str | None = None, variant: str | None = None
) -> Path:
    """Insert the dataset and variant suffix into a file name before its extension.

    Args:
        path: Base artifact path, as used for the NICE `must` benchmark.
        dataset_id: Dataset name or alias.
        variant: Benchmark variant; `None` means `must`.

    Returns:
        The path with the suffix added, or `path` unchanged when the suffix is
        empty.
    """
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
    """Return the derived Task 3 audit item path for one (source run, model, mode).

    `run_id` / `smoke` route smoke and fake runs into the parallel smoke tree
    (`data/processed/task3_verification_items/smoke/`) through
    `resolve_run_artifact_path`, so a fake Task 3 run can never overwrite
    the paper-facing item file. Full runs keep the unchanged default path.

    Args:
        root: Repository root.
        dataset_id: Dataset name or alias.
        variant: Benchmark variant.
        source_run_id: Run whose answers the verification items audit.
        model: Model whose answers are audited.
        audit_mode: Task 3 audit mode, normalized by `normalize_task3_audit_mode`.
        run_id: Run id used to detect smoke runs.
        smoke: Explicit smoke flag; `None` infers it from `run_id`.

    Returns:
        The CSV path of the verification items.
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
    """Return the `<stem>_candidate<suffix>` path next to a file.

    Args:
        path: File the candidate stands in for.

    Returns:
        The candidate file path.
    """
    path = Path(path)
    return path.with_name(f"{path.stem}_candidate{path.suffix}")


def auto_candidates_path(path: str | Path) -> Path:
    """Return the `<stem>_candidates_auto<suffix>` path next to a file.

    Args:
        path: File the automatic candidates belong to.

    Returns:
        The automatic candidates file path.
    """
    path = Path(path)
    return path.with_name(f"{path.stem}_candidates_auto{path.suffix}")


def dataset_target_seed_count(config: Mapping[str, Any], dataset_id: str | None) -> int:
    """Return how many seed requirements to select for a dataset.

    Args:
        config: Loaded configuration; `project.target_seed_count` applies to
            every dataset except `mlm_tapt`, which uses
            `datasets.mlm_tapt_target_seed_count` when set.
        dataset_id: Dataset name or alias.

    Returns:
        The target seed count.
    """
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
    """Return the column order of the seed review file for a dataset.

    Args:
        dataset_id: Dataset name or alias. NICE omits `source_corpus` and PURE
            adds the document-context columns.

    Returns:
        The review column names.
    """
    dataset_id = normalize_dataset_id(dataset_id)
    if dataset_id == DATASET_NICE:
        return [field for field in BASE_SEED_REVIEW_FIELDS if field != "source_corpus"]
    if dataset_id == DATASET_PURE:
        return [*BASE_SEED_REVIEW_FIELDS, *PURE_CONTEXT_FIELDS]
    return list(BASE_SEED_REVIEW_FIELDS)


def normalize_task_filter(task: str | Iterable[str] | None = None) -> list[str]:
    """Resolve a task selection into canonical task names.

    Args:
        task: One task name, several, or `None`. `"both"`, `"all"` and empty
            select Task 1 and Task 2; descriptive aliases such as
            `"mandatory_entailment"`, `"modality_extraction"` and
            `"modality_verification"` are accepted. Task 3 is only selected
            explicitly.

    Returns:
        Unique task names (`"task1"`, `"task2"`, `"task3"`) in request order,
        defaulting to Task 1 and Task 2.

    Raises:
        ValueError: If a task name is not recognised.
    """
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
    """Map a benchmark variant name to `must` or `shall`.

    Args:
        variant: Variant name; empty, `None` and `"main"` mean `must`.

    Returns:
        `"must"` or `"shall"`.

    Raises:
        ValueError: If the name is neither variant.
    """
    normalized = str(variant or "must").strip().lower()
    if normalized in {"", "main", "must"}:
        return "must"
    if normalized == "shall":
        return "shall"
    raise ValueError(f"Unknown benchmark variant: {variant}")


def selected_values(values: list[str], requested: str | None, name: str) -> list[str]:
    """Restrict configured datasets or variants to the one requested by the user.

    Args:
        values: Canonical values configured for the run.
        requested: Requested value; empty or `None` keeps all `values`.
        name: `"dataset"` normalizes `requested` as a dataset id; any other value
            normalizes it as a benchmark variant and names it in the error.

    Returns:
        `values` unchanged, or a one-element list with the requested value.

    Raises:
        ValueError: If the requested value is not among `values`.
    """
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
    """Build the run logging config with command-line overrides applied.

    Args:
        run_config: Run configuration whose `logging` block is the base.
        args: Parsed command-line arguments carrying the progress and warning
            thresholds; `no_progress_artifacts` and `no_request_transcripts`
            switch the corresponding artifacts off.

    Returns:
        The normalized logging config from `normalize_run_logging_config`.
    """
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
            "write_request_transcripts": (
                False if getattr(args, "no_request_transcripts", False) else None
            ),
        },
    )


def normalize_task3_audit_mode(value: Any) -> str:
    """Map a Task 3 audit mode name or alias to its canonical name.

    Args:
        value: Mode name; empty or `"official"` means the official mode, and
            hyphens, spaces and case are ignored.

    Returns:
        The canonical audit mode, such as `"blind"` or `"declared_text"`.

    Raises:
        ValueError: If the mode is not recognised.
    """
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
    """Return a provider base URL stripped of whitespace and trailing slashes.

    Args:
        value: Base URL; `None` becomes an empty string.

    Returns:
        The normalized URL.
    """
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
    """Validate the run logging settings over `DEFAULT_RUN_LOGGING`.

    Args:
        logging_config: Logging block from the run config; its keys replace the
            defaults.
        overrides: Command-line values applied last; `None` values are ignored.

    Returns:
        The logging settings with every value validated and typed.
    """
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
        "write_request_transcripts": bool_config(
            config["write_request_transcripts"], "logging.write_request_transcripts"
        ),
    }


def normalize_structured_output_mode(
    value: Any, *, json_mode: bool = False, json_schema: bool = False
) -> str:
    """Map a structured-output setting or alias to its canonical mode.

    Args:
        value: Mode name or boolean. `True` and aliases such as `"json"` mean
            `json_object`; `"schema"` and `"strict"` mean `json_schema`;
            `"pydantic"` and `"validated"` mean `instructor`; `False` and
            `"off"` mean `none`.
        json_mode: Legacy switch used only when `value` is `None`; selects
            `json_object`.
        json_schema: Legacy switch used only when `value` is `None`; selects
            `json_schema` and wins over `json_mode`.

    Returns:
        One of `"none"`, `"json_object"`, `"json_schema"` or `"instructor"`.

    Raises:
        ValueError: If the mode is not recognised.
    """
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


def batching_arm_name(batch_size: int, batch_order: str) -> str:
    """Return a stable request-composition identity, retaining archived 16-item names.

    Args:
        batch_size: Items per provider request.
        batch_order: Batch composition, `grouped` or `shuffled`.

    Returns:
        `"single"` for one item per request, the bare order name for 16-item
        batches, and `<order>_<size>` otherwise.
    """
    size = positive_int(batch_size, "batch_size")
    order = normalize_batch_order(batch_order)
    if size == 1:
        return "single"
    return order if size == 16 else f"{order}_{size}"


def normalize_batch_order(value: Any, field: str = "batch_order") -> str:
    """Normalize the batch-composition ablation knob.

    `grouped` keeps consecutive request indices together (all four modality
    variants of a seed land in one batch); `shuffled` deterministically shuffles
    job order inside each batch key group before chunking so batches mix seeds.

    Args:
        value: Configured order; `None` or empty means `DEFAULT_BATCH_ORDER`.
            Case and hyphens are ignored.
        field: Setting name quoted in the error message.

    Returns:
        `"grouped"` or `"shuffled"`.

    Raises:
        ValueError: If the order is not recognised.
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
    """Normalize the document-context ablation knob (blank means bare).

    Args:
        value: Configured context, `bare` or `document`; case and hyphens are
            ignored.
        field: Setting name quoted in the error message.

    Returns:
        `"bare"` or `"document"`.

    Raises:
        ValueError: If the context is not recognised.
    """
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

    Args:
        profile: Provider profile to read from.
        key: Profile key holding the boolean.
        default: Value used when the key is missing or `None`.
        profile_id: Profile id quoted in the error message.

    Returns:
        The parsed boolean.
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
    """Validate one provider profile and fill in its request defaults.

    The profile names the endpoint, models and request settings of one
    provider. Structured-output settings are reconciled here: `json_mode` is
    switched on by any JSON-producing mode, `json_object` gets a default
    `response_format` unless one is configured, and `instructor` drops any
    `response_format`.

    Args:
        profile: Raw profile from the run config. `provider` and `host` are
            accepted for `provider_id` and `base_url`, and `id` for `profile_id`.
        default_seed: Request seed used when the profile sets none.
        default_batch_order: Batch order used when the profile sets none.

    Returns:
        The normalized profile with every request setting present and typed.

    Raises:
        ValueError: If the provider id, profile id, base URL or models are
            missing, if `api_key_env` is not an environment-variable name, or if
            `extra_body` or `response_format` is not an object.
    """
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
    """Validate a run config and fill in the defaults the pipeline relies on.

    Sampling settings may sit at the top level or under `llm`; the top level
    wins, then the `llm` block, then `DEFAULT_CONFIG`. The run-level seed and
    batch order become the defaults of every provider profile.

    Args:
        config: Raw run config as loaded from `run_configs/*.json`.

    Returns:
        The normalized run config with datasets, variants, tasks, sampling,
        logging and provider profiles resolved.

    Raises:
        ValueError: If `run_group_id` is missing or `profiles` is not a
            non-empty list.
    """
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
    """Remove commas that directly precede a closing bracket or brace.

    Commas inside string literals, including escaped quotes, are left alone.

    Args:
        text: JSON text that may contain trailing commas.

    Returns:
        The text with trailing commas removed.
    """
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
    """Load a JSON run config, tolerating trailing commas, and normalize it.

    Args:
        path: Run config file.

    Returns:
        The run config as normalized by `normalize_run_config`.

    Raises:
        ValueError: If the file is not valid JSON even after trailing commas are
            removed; the message gives the line and column.
    """
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


SAMPLING_PLAN_SOURCE_PLANNED = "planned"
SAMPLING_PLAN_SOURCE_INFERRED = "inferred"
SAMPLING_PLAN_SOURCES = (SAMPLING_PLAN_SOURCE_PLANNED, SAMPLING_PLAN_SOURCE_INFERRED)


def _optional_float(value: Any) -> float | None:
    """Return a registry cell as a float, or `None` when it is blank or unparseable.

    Args:
        value: Registry cell value.

    Returns:
        The parsed float or `None`.
    """
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class SamplingPlan:
    """The repeated-sampling plan a run was executed under.

    `stochastic_samples` is the number of stochastic samples the run PLANNED
    per item. It is what makes a sample the run never wrote (crash, budget stop,
    interrupted resume) visible as a missing observation rather than as absent
    by design: three observed samples of a planned five are 3/5, not 3/3.
    `temperature` and `top_p` are carried when the caller holds the run
    config, so the sampling regime behind a score row stays reconstructable.

    `source` is `"planned"` for a declared plan and `"inferred"` for the
    explicitly requested exploratory mode that reconstructs the denominator from
    whatever rows happen to exist. It is stamped on every score row and every
    manifest the plan feeds, so an inferred denominator can never be read
    downstream as a declared one.

    Attributes:
        stochastic_samples: Planned stochastic samples per item; 0 for
            deterministic-only runs and for the inferred plan.
        temperature: Stochastic sampling temperature in [0, 2], or `None` when
            not known.
        top_p: Stochastic nucleus-sampling threshold in (0, 1], or `None` when
            not known.
        source: `"planned"` or `"inferred"`, as described above.

    Raises:
        ValueError: On construction, if `stochastic_samples` is not a
            non-negative int, `temperature` or `top_p` is out of range, or
            `source` is unknown.
    """

    stochastic_samples: int
    temperature: float | None = None
    top_p: float | None = None
    source: str = SAMPLING_PLAN_SOURCE_PLANNED

    def __post_init__(self) -> None:
        if isinstance(self.stochastic_samples, bool) or not isinstance(
            self.stochastic_samples, int
        ):
            raise ValueError(
                "SamplingPlan.stochastic_samples must be an int, got "
                f"{self.stochastic_samples!r}."
            )
        if self.stochastic_samples < 0:
            raise ValueError(
                "SamplingPlan.stochastic_samples must not be negative, got "
                f"{self.stochastic_samples}."
            )
        if self.temperature is not None and not 0.0 <= float(self.temperature) <= 2.0:
            raise ValueError(
                f"SamplingPlan.temperature must be in [0, 2], got {self.temperature!r}."
            )
        if self.top_p is not None and not 0.0 < float(self.top_p) <= 1.0:
            raise ValueError(
                f"SamplingPlan.top_p must be in (0, 1], got {self.top_p!r}."
            )
        if self.source not in SAMPLING_PLAN_SOURCES:
            raise ValueError(
                f"SamplingPlan.source must be one of {SAMPLING_PLAN_SOURCES}, got "
                f"{self.source!r}."
            )

    @classmethod
    def from_run_config(cls, run_config: Mapping[str, Any]) -> SamplingPlan:
        """Read the plan from a run config or from a project config `llm` block.

        Args:
            run_config: Run config with a top-level `stochastic` block, or a project
                config whose `llm.stochastic` block is used instead.

        Returns:
            A planned sampling plan with the configured sample count, temperature
            and top-p.

        Raises:
            ValueError: If neither location holds a stochastic sampling block.
        """
        stochastic = run_config.get("stochastic")
        if not isinstance(stochastic, Mapping):
            llm_config = run_config.get("llm")
            stochastic = (
                llm_config.get("stochastic")
                if isinstance(llm_config, Mapping)
                else None
            )
        if not isinstance(stochastic, Mapping):
            raise ValueError(
                "Run config declares no stochastic sampling block; a SamplingPlan "
                "cannot be derived from it."
            )
        temperature = stochastic.get("temperature")
        top_p = stochastic.get("top_p")
        return cls(
            stochastic_samples=int(stochastic.get("samples", 0) or 0),
            temperature=None if temperature is None else float(temperature),
            top_p=None if top_p is None else float(top_p),
        )

    @classmethod
    def from_registry_row(cls, row: Mapping[str, Any]) -> SamplingPlan | None:
        """Return the plan a completed run recorded, or `None` if it recorded none.

        Rows written before `expected_stochastic_samples` became a registry
        column leave it blank. `None` says the run's own plan is unknown, so a
        caller can decide between failing and falling back rather than silently
        reading a blank as zero.

        Args:
            row: Run registry row; `expected_stochastic_samples`, `temperature` and
                `top_p` are read.

        Returns:
            The recorded plan, or `None` when the sample count is blank or
            unparseable.
        """
        raw = str(row.get("expected_stochastic_samples", "")).strip()
        if not raw:
            return None
        try:
            samples = int(float(raw))
        except ValueError:
            return None
        temperature = _optional_float(row.get("temperature"))
        top_p = _optional_float(row.get("top_p"))
        return cls(stochastic_samples=samples, temperature=temperature, top_p=top_p)

    @classmethod
    def inferred_from_observations(cls) -> SamplingPlan:
        """Return the exploratory plan whose denominator is whatever rows were written.

        Returns:
            A plan with no planned samples and `source` set to `"inferred"`.
        """
        return cls(stochastic_samples=0, source=SAMPLING_PLAN_SOURCE_INFERRED)

    def total_samples(self, observed: int) -> int:
        """Return the denominator for one stochastic group of `observed` written rows.

        Args:
            observed: Stochastic rows written for the group.

        Returns:
            The larger of `observed` and the planned sample count.
        """
        return max(int(observed), self.stochastic_samples)


# Prompt-sensitivity and other deterministic-only probes plan no repeated
# samples at all, so their denominator is the row they wrote, by design.
DETERMINISTIC_ONLY_SAMPLING_PLAN = SamplingPlan(stochastic_samples=0)


def resolve_sampling_plan(
    sampling_plan: SamplingPlan | None,
    infer_plan_from_observations: bool,
    *,
    caller: str,
) -> SamplingPlan:
    """Return the plan a scoring call must use, or refuse to guess a denominator.

    Args:
        sampling_plan: Declared plan of the run being scored.
        infer_plan_from_observations: Opt into the exploratory inferred plan;
            only allowed when `sampling_plan` is `None`.
        caller: Name of the scoring function, quoted in error messages.

    Returns:
        `sampling_plan`, or the inferred plan when exploration was requested.

    Raises:
        ValueError: If both or neither of `sampling_plan` and
            `infer_plan_from_observations` are given, or if `sampling_plan` is
            not a `SamplingPlan`.
    """
    if sampling_plan is not None and infer_plan_from_observations:
        raise ValueError(
            f"{caller}: pass either sampling_plan or "
            "infer_plan_from_observations=True, not both."
        )
    if sampling_plan is not None:
        if not isinstance(sampling_plan, SamplingPlan):
            raise ValueError(
                f"{caller}: sampling_plan must be a SamplingPlan, got "
                f"{type(sampling_plan).__name__}."
            )
        return sampling_plan
    if infer_plan_from_observations:
        return SamplingPlan.inferred_from_observations()
    raise ValueError(
        f"{caller} requires an explicit SamplingPlan carrying the number of "
        "stochastic samples the run planned per item. Pass "
        "infer_plan_from_observations=True only for exploratory work where "
        "reconstructing the denominator from the rows that happen to exist is "
        "acceptable; those rows are stamped sampling_plan_source='inferred'."
    )


def filter_run_profiles(
    run_config: Mapping[str, Any],
    profile_id: str | None = None,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """Select the provider profiles and models a command should run.

    Args:
        run_config: Normalized run config.
        profile_id: Keep only profiles whose profile id or provider id equals
            this value; empty keeps all.
        model: Keep only profiles that list this model, narrowed to that one
            model; empty keeps all models.

    Returns:
        Copies of the matching profiles.

    Raises:
        ValueError: If no profile matches the filters.
    """
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
    """Check that a profile served by a manually restarted server runs one model.

    Args:
        profile: Normalized provider profile.

    Raises:
        ValueError: If the profile sets `requires_manual_server` and lists
            other than exactly one model.
    """
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
    """Return True for run ids produced by smoke/fake Task 1/2/3 runs.

    Runners other than Task 1/2 put their own components before `smoke`:
    Task 3 an optional audit mode and benchmark variant
    (`task3-declared-text-shall-smoke-*`), the weak-phrasing probe its name
    (`weak-modality-probe-shall-smoke-*`). Checking only the historical
    `task3-smoke-*` prefix would route their smoke runs into the real tree.

    Args:
        run_id: Run id to classify; `None` is not a smoke run.

    Returns:
        True when the id starts with `smoke-` or has one of the runner-specific
        smoke shapes.
    """
    value = str(run_id or "").strip()
    if value.startswith(SMOKE_RUN_PREFIX):
        return True
    return bool(
        re.match(
            r"^(?:task3(?:-declared-(?:text|source))?|weak-modality-probe)"
            r"(?:-shall)?-smoke-",
            value,
        )
    )


def smoke_tree_env_enabled() -> bool:
    """Allow read-only consumers to opt into the smoke tree via the environment.

    Returns:
        True when `RE_UQ_SMOKE_TREE` is set to `1`, `true`, `yes` or `on`.
    """
    return str(os.environ.get(SMOKE_TREE_ENV_VAR, "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def smoke_tree_path(path: str | Path) -> Path:
    """Map `data/processed/<name>` to `data/processed/smoke/<name>` (idempotent).

    Args:
        path: Artifact path, already in the smoke tree or not.

    Returns:
        The path inside the smoke directory next to the original file.
    """
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

    Paper-facing full runs keep writing into `data/processed/`; smoke runs never
    touch those files so a fake run can never contaminate published numbers.

    Args:
        path: Default artifact path in the paper-facing tree.
        run_id: Run id; a smoke run id selects the smoke tree.
        smoke: Force the smoke tree when true. When both `smoke` and `run_id`
            are `None`, `RE_UQ_SMOKE_TREE` decides.

    Returns:
        The smoke-tree path for smoke runs, otherwise `path`.
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
    """Return the Task 1/2 run registry CSV for a dataset and variant.

    Args:
        root: Repository root.
        dataset_id: Dataset name or alias; selects the file-name suffix.
        variant: Benchmark variant; selects the file-name suffix.
        run_id: Run id; smoke runs are routed into the smoke tree.
        smoke: Force the smoke tree when true; `None` infers it from `run_id`.

    Returns:
        The registry path, in the smoke tree for smoke runs.
    """
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
    """Return the Task 1/2 raw model output JSONL for a dataset and variant.

    Args:
        root: Repository root.
        dataset_id: Dataset name or alias; selects the file-name suffix.
        variant: Benchmark variant; selects the file-name suffix.
        run_id: Run id; smoke runs are routed into the smoke tree.
        smoke: Force the smoke tree when true; `None` infers it from `run_id`.

    Returns:
        The raw output path, in the smoke tree for smoke runs.
    """
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
    """Return the Task 3 verification raw model output JSONL for a dataset and variant.

    Args:
        root: Repository root.
        dataset_id: Dataset name or alias; selects the file-name suffix.
        variant: Benchmark variant; selects the file-name suffix.
        run_id: Run id; smoke runs are routed into the smoke tree.
        smoke: Force the smoke tree when true; `None` infers it from `run_id`.

    Returns:
        The raw output path, in the smoke tree for smoke runs.
    """
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
    """Return the Task 3 verification run registry CSV for a dataset and variant.

    Args:
        root: Repository root.
        dataset_id: Dataset name or alias; selects the file-name suffix.
        variant: Benchmark variant; selects the file-name suffix.
        run_id: Run id; smoke runs are routed into the smoke tree.
        smoke: Force the smoke tree when true; `None` infers it from `run_id`.

    Returns:
        The registry path, in the smoke tree for smoke runs.
    """
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
    """Return the Task 3 verification live progress CSV for a dataset and variant.

    Args:
        root: Repository root.
        dataset_id: Dataset name or alias; selects the file-name suffix.
        variant: Benchmark variant; selects the file-name suffix.
        run_id: Run id; smoke runs are routed into the smoke tree.
        smoke: Force the smoke tree when true; `None` infers it from `run_id`.

    Returns:
        The progress file path, in the smoke tree for smoke runs.
    """
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
    """Return the Task 3 verification event log JSONL for a dataset and variant.

    Args:
        root: Repository root.
        dataset_id: Dataset name or alias; selects the file-name suffix.
        variant: Benchmark variant; selects the file-name suffix.
        run_id: Run id; smoke runs are routed into the smoke tree.
        smoke: Force the smoke tree when true; `None` infers it from `run_id`.

    Returns:
        The event log path, in the smoke tree for smoke runs.
    """
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
    """Return the text log file of one run.

    Args:
        root: Repository root.
        run_id: Run id; made safe for use as a file name.

    Returns:
        The log path under `data/processed/logs/`.
    """
    return Path(root) / "data/processed/logs" / f"{safe_identifier(run_id)}.log"


def acse_semantic_cache_dir(
    analysis_dir: str | Path, backend_label: str, model: str
) -> Path:
    """Return the cache directory for one model's artifacts under one embedding backend.

    The model is part of the path because one analysis dir can hold several
    scored models: an empty `model_filter` expands to all of them, and without
    the model component they would all write the same directory.

    Args:
        analysis_dir: Analysis output directory that holds the cache.
        backend_label: Label of the embedding backend.
        model: Scored model the artifacts belong to.

    Returns:
        The cache directory path; it is not created.
    """
    return Path(analysis_dir) / (
        f"acse_semantic_{safe_identifier(backend_label)}__{safe_identifier(model)}"
    )


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string with a `Z` suffix.

    Returns:
        The timestamp at second resolution.
    """
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of a file's bytes.

    `hashlib.file_digest` requires a freshly opened blocking binary file, which
    is exactly what this helper opens; the digest is identical to hashing the
    whole byte string.

    Args:
        path: File to hash.

    Returns:
        The hexadecimal digest.
    """
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def sha256_text(text: str) -> str:
    """Return the SHA-256 of a string encoded as UTF-8.

    Args:
        text: Text to hash.

    Returns:
        The hexadecimal digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def artifact_metadata(
    path: str | Path, root: str | Path | None = None
) -> dict[str, Any]:
    """Describe one artifact for a manifest: path, existence, hash, size and rows.

    Args:
        path: Artifact file.
        root: Directory the recorded path is made relative to; paths outside it,
            or any path when `root` is `None`, are recorded as given.

    Returns:
        A dictionary with `path`, `exists`, `sha256`, `bytes` and `rows`. A
        missing file has an empty hash and zero bytes; `rows` is the data row
        count for CSV files and empty otherwise.
    """
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
    """Write a JSON manifest that records the hash of every benchmark artifact.

    Args:
        paths: Artifacts to record.
        output_path: Manifest file to write.
        root: Directory the recorded artifact paths are made relative to.
        metadata: Free-form metadata stored under `metadata`.

    Returns:
        The manifest as written, with a creation timestamp and one
        `artifact_metadata` entry per path.
    """
    output_path = Path(output_path)
    manifest = {
        "created_at_utc": utc_now_iso(),
        "metadata": metadata or {},
        "artifacts": [artifact_metadata(path, root=root) for path in paths],
    }
    write_json(output_path, manifest)
    return manifest


def _manifest_artifact_required(rel_path: str) -> bool:
    """Return True for tracked artifacts whose absence invalidates the analysis.

    These are the prompts and the benchmark CSVs: `prompts/` files and
    `data/processed/benchmark_items*.csv`.

    Args:
        rel_path: Artifact path relative to the project root.

    Returns:
        True when the artifact must exist.
    """
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

    Hard-fails (`ValueError`) on any hash mismatch, and on a missing required
    artifact (tracked `prompts/` files or `data/processed/benchmark_items*.csv`).
    Missing gitignored data artifacts are reported but not fatal.

    Args:
        manifest_path: Manifest written by `write_benchmark_manifest`.
        project_root: Root the manifest's relative paths are resolved against.

    Returns:
        A summary dict with the manifest path, the number of hashes `checked`,
        the `missing` optional artifacts and their `missing_count`.

    Raises:
        ValueError: If the manifest is missing, a required artifact is missing,
            or an artifact's hash differs from the recorded one.
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


# ---------------------------------------------------------------------------
# Raw row store: append-only JSONL tail + compacted Parquet sibling
# ---------------------------------------------------------------------------
# Every raw file (`model_outputs_raw*.jsonl`, transcripts, events) is written
# one JSON line at a time under `file_lock`, which is what makes concurrent
# runners and crashes safe. Those lines repeat the prompt and dozens of constant
# metadata fields per row, so a finished run costs ~40x more disk as JSONL than
# as zstd Parquet. `compact_jsonl` moves the accumulated lines into the sibling
# `<stem>.parquet`; readers transparently return Parquet rows followed by the
# JSONL tail, so nothing downstream needs to know which file holds a row.
#
# The Parquet file is a faithful row store, not a typed table: nested values
# and columns of mixed Python types are JSON-encoded, and keys absent from a
# row stay absent after a round trip (see `RAW_STORE_DENSE_COLUMNS_KEY`).
# Compaction verifies the round trip before it truncates the JSONL tail.

RAW_STORE_SUFFIX = ".parquet"
RAW_STORE_FORMAT = "re-uq-raw-store/1"
RAW_STORE_FORMAT_KEY = b"re_uq.raw_store.format"
RAW_STORE_JSON_COLUMNS_KEY = b"re_uq.raw_store.json_columns"
RAW_STORE_DENSE_COLUMNS_KEY = b"re_uq.raw_store.dense_columns"
# Per-row list of sparse keys that were present with a JSON null value, which a
# nullable Arrow column cannot tell apart from an absent key.
RAW_STORE_NULL_KEYS_COLUMN = "_re_uq_null_keys"
RAW_STORE_COMPRESSION = "zstd"
RAW_STORE_COMPRESSION_LEVEL = 3
_RAW_STORE_SCALAR_TYPES = (str, int, float, bool)


def raw_store_path(path: str | Path) -> Path:
    """Return the compacted Parquet sibling of a raw path (`x.jsonl` -> `x.parquet`).

    Args:
        path: Raw JSONL path.

    Returns:
        The Parquet path next to it.

    Raises:
        ValueError: If `path` does not end in `.jsonl`.
    """
    path = Path(path)
    if path.suffix != ".jsonl":
        raise ValueError(f"raw store paths end with .jsonl, got {path}")
    return path.with_suffix(RAW_STORE_SUFFIX)


def raw_store_exists(path: str | Path) -> bool:
    """Return whether either half of the store exists for this raw path.

    Only existence is checked, so an empty JSONL tail also counts.

    Args:
        path: Raw JSONL path.

    Returns:
        True when the JSONL tail or its Parquet sibling exists as a file.
    """
    path = Path(path)
    return path.exists() or raw_store_path(path).exists()


def raw_store_files(path: str | Path) -> list[Path]:
    """Return the existing files backing a raw path, Parquet first.

    Args:
        path: Raw JSONL path.

    Returns:
        The Parquet sibling and the JSONL tail, each only if it exists.
    """
    path = Path(path)
    store = raw_store_path(path)
    return [candidate for candidate in (store, path) if candidate.exists()]


def _raw_store_json_columns(rows: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    """Return columns that need JSON encoding, and columns present in every row.

    A column needs JSON encoding when its non-null values have more than one
    Python type or any non-scalar type.

    Args:
        rows: Raw rows about to be stored.

    Returns:
        The set of JSON-encoded columns and the set of dense columns.
    """
    types: dict[str, set[type]] = defaultdict(set)
    counts: Counter[str] = Counter()
    for row in rows:
        for key, value in row.items():
            counts[key] += 1
            if value is not None:
                types[key].add(type(value))
    json_columns = {
        key
        for key, seen in types.items()
        if len(seen) > 1 or any(kind not in _RAW_STORE_SCALAR_TYPES for kind in seen)
    }
    dense = {key for key, count in counts.items() if count == len(rows)}
    return json_columns, dense


def raw_rows_to_table(rows: list[dict[str, Any]]) -> pa.Table:
    """Build the Arrow table for `rows`, recording the encoding in the schema metadata.

    Args:
        rows: Raw rows to store; keys may differ between rows.

    Returns:
        The table, with the format tag and the JSON-encoded and dense column
        lists in its schema metadata.

    Raises:
        ValueError: If a row uses the reserved null-keys column name.
    """
    import pyarrow as pa

    json_columns, dense = _raw_store_json_columns(rows)
    columns = sorted({key for row in rows for key in row})
    if RAW_STORE_NULL_KEYS_COLUMN in columns:
        raise ValueError(f"{RAW_STORE_NULL_KEYS_COLUMN} is reserved by the raw store")
    arrays = {}
    null_keys = [
        sorted(key for key, value in row.items() if value is None and key not in dense)
        for row in rows
    ]
    if any(null_keys):
        arrays[RAW_STORE_NULL_KEYS_COLUMN] = pa.array(
            [json.dumps(keys) if keys else None for keys in null_keys], type=pa.string()
        )
    for key in columns:
        if key in json_columns:
            values = [
                None
                if key not in row or row[key] is None
                else json.dumps(row[key], ensure_ascii=True, sort_keys=True)
                for row in rows
            ]
            arrays[key] = pa.array(values, type=pa.string())
        else:
            arrays[key] = pa.array([row.get(key) for row in rows])
    table = pa.table(arrays) if arrays else pa.table({})
    metadata = {
        RAW_STORE_FORMAT_KEY: RAW_STORE_FORMAT.encode(),
        RAW_STORE_JSON_COLUMNS_KEY: json.dumps(sorted(json_columns)).encode(),
        RAW_STORE_DENSE_COLUMNS_KEY: json.dumps(sorted(dense)).encode(),
    }
    return table.replace_schema_metadata({**(table.schema.metadata or {}), **metadata})


def raw_table_to_rows(table: pa.Table) -> list[dict[str, Any]]:
    """Return the rows of a store table, decoding what `raw_rows_to_table` encoded.

    Args:
        table: Table read from a raw store Parquet file.

    Returns:
        The rows, with JSON columns decoded and keys that were absent from a row
        left absent.

    Raises:
        ValueError: If the table lacks the raw store format metadata.
    """
    metadata = table.schema.metadata or {}
    if metadata.get(RAW_STORE_FORMAT_KEY, b"").decode() != RAW_STORE_FORMAT:
        raise ValueError("not a re-uq raw store Parquet file (missing format metadata)")
    json_columns = set(json.loads(metadata.get(RAW_STORE_JSON_COLUMNS_KEY, b"[]")))
    dense = set(json.loads(metadata.get(RAW_STORE_DENSE_COLUMNS_KEY, b"[]")))
    rows: list[dict[str, Any]] = []
    for record in table.to_pylist():
        row: dict[str, Any] = {}
        null_keys = record.pop(RAW_STORE_NULL_KEYS_COLUMN, None)
        for key, value in record.items():
            if value is None:
                if key in dense:
                    row[key] = None
                continue
            row[key] = json.loads(value) if key in json_columns else value
        if null_keys:
            for key in json.loads(null_keys):
                row[key] = None
        rows.append(row)
    return rows


def read_raw_store(
    path: str | Path,
    *,
    columns: Iterable[str] | None = None,
    run_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Read the rows of the Parquet half of a raw path, optionally a column/run subset.

    Column projection and run-id filtering happen inside Parquet, which is what
    makes resume and the paper exporter cheap on a multi-GB corpus. A partial
    projection returns rows holding only those columns.

    Args:
        path: Raw JSONL path whose Parquet sibling is read.
        columns: Columns to read; names absent from the file are skipped. `None`
            reads all columns.
        run_ids: Keep only rows of these runs; an empty iterable returns no rows.
            `None` keeps every run.

    Returns:
        The decoded rows, or an empty list when no Parquet file exists.
    """
    import pyarrow.parquet as pq

    store = raw_store_path(path)
    if not store.exists():
        return []
    filters = None
    if run_ids is not None:
        selected = sorted({str(run_id) for run_id in run_ids})
        if not selected:
            return []
        filters = [("run_id", "in", selected)]
    requested = None if columns is None else list(columns)
    if requested is not None:
        schema = pq.read_schema(store)
        requested = [name for name in requested if name in schema.names]
        if RAW_STORE_NULL_KEYS_COLUMN in schema.names:
            requested.append(RAW_STORE_NULL_KEYS_COLUMN)
    table = pq.read_table(store, columns=requested, filters=filters)
    return raw_table_to_rows(table)


def _read_jsonl_lines(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read every row of a raw path: the compacted Parquet rows, then the JSONL tail.

    Reads under a shared lock so a concurrent `compact_jsonl` cannot hand back
    a row twice (once from the new Parquet file, once from the not-yet-truncated
    tail).

    Args:
        path: Raw JSONL path.

    Returns:
        All rows in storage order, or an empty list when neither file exists.
    """
    path = Path(path)
    if not raw_store_exists(path):
        return []
    with file_lock(path, shared=True):
        rows = read_raw_store(path)
        if path.exists():
            rows.extend(_read_jsonl_lines(path))
    return rows


def _raw_rows_canonical(rows: list[dict[str, Any]]) -> list[str]:
    return [json.dumps(row, ensure_ascii=True, sort_keys=True) for row in rows]


def compact_jsonl(path: str | Path, *, verify: bool = True) -> dict[str, Any]:
    """Move the JSONL tail of a raw path into its Parquet sibling.

    Holds the exclusive lock for the whole operation: read both halves, write
    the merged Parquet file atomically, then truncate the tail to zero bytes
    (kept, not deleted, so `path.exists()` checks and appenders keep working).
    With `verify`, the merged file is read back and compared row for row
    before the tail is touched; a mismatch leaves both files as they were and
    raises, because losing raw rows is the one thing this must never do.

    Args:
        path: Raw JSONL path to compact.
        verify: Read the merged Parquet file back and compare it with the input
            rows before replacing the store and truncating the tail.

    Returns:
        A summary with the paths, row counts before and after, the tail and
        store sizes in bytes, and whether anything was `compacted`.

    Raises:
        RuntimeError: If verification finds that the merged file does not
            round-trip the rows.
    """
    import pyarrow.parquet as pq

    path = Path(path)
    store = raw_store_path(path)
    summary = {
        "path": str(path),
        "store": str(store),
        "tail_rows": 0,
        "store_rows_before": 0,
        "store_rows_after": 0,
        "tail_bytes_before": 0,
        "store_bytes_after": 0,
        "compacted": False,
    }
    if not path.exists() or path.stat().st_size == 0:
        if store.exists():
            summary["store_rows_after"] = pq.read_metadata(store).num_rows
            summary["store_bytes_after"] = store.stat().st_size
        return summary
    with file_lock(path):
        summary["tail_bytes_before"] = path.stat().st_size
        existing = read_raw_store(path)
        tail = _read_jsonl_lines(path)
        summary["store_rows_before"] = len(existing)
        summary["tail_rows"] = len(tail)
        if not tail:
            return summary
        merged = existing + tail
        table = raw_rows_to_table(merged)
        temp = store.with_name(store.name + f".tmp-{os.getpid()}")
        pq.write_table(
            table,
            temp,
            compression=RAW_STORE_COMPRESSION,
            compression_level=RAW_STORE_COMPRESSION_LEVEL,
        )
        if verify:
            written = raw_table_to_rows(pq.read_table(temp))
            if _raw_rows_canonical(written) != _raw_rows_canonical(merged):
                temp.unlink(missing_ok=True)
                raise RuntimeError(
                    f"raw store round trip mismatch for {path}; JSONL left untouched"
                )
        temp.replace(store)
        with path.open("w", encoding="utf-8") as handle:
            handle.truncate(0)
            handle.flush()
            os.fsync(handle.fileno())
        summary["store_rows_after"] = len(merged)
        summary["store_bytes_after"] = store.stat().st_size
        summary["compacted"] = True
    return summary


def latest_run_id(
    raw_rows: list[dict[str, Any]], prefix: str | Iterable[str] | None = None
) -> str | None:
    """Return the run id of the last row whose run id matches a prefix.

    "Latest" means last in row order, which for append-only raw files is the
    most recently written run.

    Args:
        raw_rows: Raw output rows in storage order.
        prefix: Run-id prefix or prefixes, matched as in `run_id_matches_prefix`;
            empty matches every run.

    Returns:
        The run id, or `None` when no row matches.
    """
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
    """Select the rows of one run: the requested run or the latest matching one.

    Args:
        raw_rows: Raw output rows in storage order.
        run_id: Run to select; empty selects the latest run matching `prefix`.
        prefix: Run-id prefix or prefixes the selected run must match.

    Returns:
        The selected run id (or `None`) and its rows. An explicit `run_id` that
        does not match `prefix` is returned with no rows.
    """
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
    """Return whether a run id belongs to a run prefix.

    The run id must start with `<prefix>-`. Benchmark variants are encoded
    after the prefix, so `full` does not match `full-shall-*`.

    Args:
        run_id: Run id to test; an empty id matches only an empty prefix.
        prefix: One prefix or several, any of which may match; `None`, an empty
            string or an empty list matches every run id.

    Returns:
        True when the run id matches.
    """
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
def file_lock(path: str | Path, *, shared: bool = False) -> Iterator[TextIO]:
    """Hold an advisory lock on a sidecar `<path>.lock` file (exclusive by default).

    Concurrent runners share the per-dataset raw JSONL and registry CSVs, so
    every append/rewrite is serialized across processes. Readers of the raw
    store take the lock `shared` so they never observe a half-compacted
    file. The lock is advisory: it only protects callers that go through
    these helpers.

    Args:
        path: File to protect; the lock file is created next to it.
        shared: Take a shared (reader) lock instead of an exclusive one.

    Yields:
        The open lock-file handle; callers use the context only for its lock.
    """
    lock_path = Path(str(path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        try:
            yield handle
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    """Append one row as a JSON line under the file lock and sync it to disk.

    Args:
        path: JSONL file; created with its parent directories if missing.
        row: Row to append, written ASCII-escaped with sorted keys.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n"
    with file_lock(path), path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write `text` to `path` via a temp file + `os.replace` (never a torn file).

    Args:
        path: Destination file; parent directories are created.
        text: Content to write as UTF-8.
    """
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
    """Write a value as indented, key-sorted JSON with a trailing newline.

    Args:
        path: Destination file; overwritten, with parent directories created.
        value: JSON-serializable value; non-ASCII characters are escaped.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def compact_json(value: Any) -> str:
    """Serialize a value as compact, key-sorted JSON for a single CSV cell.

    Args:
        value: JSON-serializable value.

    Returns:
        The JSON text, or an empty string for `None`, `""`, `{}` and `[]`.
    """
    if value in (None, "", {}, []):
        return ""
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def download_file(url: str, dest: str | Path, timeout_s: int = 120) -> Path:
    """Download a URL to a local file.

    An HTTP error status raises through `requests.Response.raise_for_status`.

    Args:
        url: URL to fetch.
        dest: Destination file; overwritten, with parent directories created.
        timeout_s: Request timeout in seconds.

    Returns:
        The destination path.
    """
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
    """Collapse whitespace runs to single spaces and trim the ends.

    Args:
        text: Text to normalize; `None` or empty yields `""`.

    Returns:
        The normalized text.
    """
    return SPACE_RE.sub(" ", (text or "").strip())


def unwrap_outer_quotes(text: str) -> str:
    """Strip straight and typographic quotes from both ends, repeatedly.

    Args:
        text: Text to unwrap; whitespace is normalized first.

    Returns:
        The text without surrounding quotes.
    """
    previous = normalize_space(text)
    while previous:
        current = OUTER_QUOTES_RE.sub("", previous).strip()
        if current == previous:
            return current
        previous = current
    return previous


def strip_final_punctuation(text: str) -> str:
    """Remove outer quotes and trailing sentence punctuation from a text.

    Args:
        text: Requirement or capability text.

    Returns:
        The text without surrounding quotes and without trailing spaces, periods,
        semicolons, colons or quotes.
    """
    return unwrap_outer_quotes(normalize_space(text)).rstrip(" .;:'\"“”‘’")


def word_count(text: str) -> int:
    """Count alphanumeric words, treating hyphenated and apostrophe forms as one.

    Args:
        text: Text to count; `None` counts as empty.

    Returns:
        The number of words.
    """
    return len(WORD_RE.findall(text or ""))


def lower_initial(text: str) -> str:
    """Lowercase the first letter of a text so it can continue a sentence.

    Text that starts with two uppercase characters, such as an acronym, keeps
    its case. Outer quotes and final punctuation are stripped first.

    Args:
        text: Capability or requirement text.

    Returns:
        The adjusted text.
    """
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


def auto_capability_text(requirement: str, *, preserve_subject: bool = False) -> str:
    """Derive the capability phrase of a requirement by stripping subject and modal.

    List markers, a leading actor such as "The system" or "Users", the modal
    verb and "be able to" are removed, and the first letter is lowercased, so
    "The system shall be able to export reports." becomes "export reports".

    Args:
        requirement: Original requirement sentence.
        preserve_subject: Keep the whole sentence (without final punctuation)
            unless it starts with "the system" plus a modal; used for PURE, whose
            requirements have varied actors and leading conditions.

    Returns:
        The automatic capability text, meant for manual review.
    """
    text = strip_final_punctuation(requirement)
    text = re.sub(r"^[\-\*\d.)\s]+", "", text)
    if preserve_subject:
        # PURE has heterogeneous actors and leading conditions. The generic
        # prefix stripper loses both. Preserve the sentence for manual review
        # unless the actor is already the benchmark's generic system.
        prefix = re.match(
            r"^(?:the\s+)?system\s+(?:shall|must|should|may|will|can|could)\s+",
            text,
            flags=re.I,
        )
        if prefix is None:
            return text
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
    """Apply the automatic seed eligibility checks to a requirement and capability.

    The checks reject requirements under 5 or over 35 words, multiple
    sentences, negation, formulas or symbols, more than one "and"/"or"/";",
    a missing or one-word capability, a modal left in the capability, and a
    capability that starts with a stranded preposition.

    Args:
        requirement: Original requirement sentence.
        capability: Automatic capability text derived from it.

    Returns:
        Whether the pair passes, and the failed check names joined by `;`.
    """
    reasons: list[str] = []
    requirement_word_count = word_count(requirement)
    cleaned_requirement = strip_final_punctuation(requirement)
    cleaned_capability = strip_final_punctuation(capability)
    if requirement_word_count < 5:
        reasons.append("too_short")
    if requirement_word_count > 35:
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
    """Apply the stricter seed checks for the Hugging Face requirements corpus.

    On top of `automatic_filter`, this rejects excluded source corpora,
    requirements without a requirement cue, table or figure references, colon
    structures, list or heading markers, notes and symbol-heavy text.

    Args:
        requirement: Original requirement sentence.
        capability: Automatic capability text derived from it.
        source_corpus: Corpus the requirement comes from.
        exclude_source_regex: Pattern of source corpora to exclude; the default
            drops PURE documents, which have their own dataset. Empty disables
            the check.

    Returns:
        Whether the pair passes, and the unique failed check names joined by `;`.
    """
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
    """Find the column that holds the requirement text in a source dataset.

    Known names such as `RequirementText` are tried first (case-insensitive),
    then any column containing "requirement" or named `text` or `sentence`.

    Args:
        rows: Source dataset rows; the first row's keys are inspected.

    Returns:
        The column name.

    Raises:
        ValueError: If there are no rows or no matching column.
    """
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
    """Build seed candidate rows from a requirements dataset for manual review.

    Requirements are whitespace-normalized and deduplicated case-insensitively.
    Each one gets a sequential seed id, an automatic capability text and the
    `automatic_filter` verdict; the first `target_count` eligible rows are
    marked for inclusion.

    Args:
        dataset_rows: Source dataset rows.
        target_count: Number of eligible rows to mark `include = yes`.
        source_dataset: Dataset label stored in every candidate.
        text_column: Column holding the requirement; found automatically when
            `None`.
        source_corpus_field: Column whose value is stored as `source_corpus`;
            `None` omits that field.

    Returns:
        One candidate row per distinct requirement, in dataset order.
    """
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
    """Load the `limsc/mlm-tapt-requirements` dataset from Hugging Face.

    Args:
        config: Loaded configuration; the repository, dataset configuration and
            splits are read from `datasets` with `DEFAULT_CONFIG` fallbacks.

    Returns:
        One row per requirement with `source`, `reqs` and `split`.

    Raises:
        ImportError: If the `datasets` package is not installed.
    """
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
    """Sample eligible candidates so that no single source corpus dominates.

    Only rows with a truthy `auto_include` are eligible. Each draw weights a
    candidate by the inverse of its source's eligible count and skips sources
    that already reached `source_cap`, so the result is spread across sources
    and reproducible for a given seed.

    Args:
        candidates: Candidate rows to sample from.
        target_count: Number of candidates to select.
        seed: Random seed for the draws.
        source_cap: Maximum selections from one source.
        source_field: Column naming the source; blank values count as
            `"unknown"`.

    Returns:
        Indices into `candidates`, in draw order.

    Raises:
        ValueError: If fewer than `target_count` candidates are eligible, or the
            source cap leaves no candidate before the target is reached.
    """
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
    """Build seed candidate rows from the Hugging Face requirements corpus.

    Requirements are deduplicated case-insensitively and screened with
    `mlm_tapt_filter`. The included seeds are then drawn from the eligible pool
    by `weighted_sample_candidate_indices`; eligible rows that were not drawn
    are marked `not_sampled_weighted_pool`.

    Args:
        dataset_rows: Rows from `load_mlm_tapt_rows`, with `source` and `reqs`.
        target_count: Number of seeds to include.
        seed: Random seed for the weighted draw.
        exclude_source_regex: Pattern of source corpora to exclude.
        source_cap: Maximum included seeds from one source corpus.

    Returns:
        One candidate row per distinct requirement, in dataset order.
    """
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
    """Split one `<req>` into its clean text and its author marker.

    Args:
        req: The `<req>` element of a PURE XML document.

    Returns:
        The marker-free `text`, the `marked_text` up to the last inline marker,
        the `marker` (`M`, `O`, `I` or empty) and the `marker_count`.
    """
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

    Args:
        xml_text: Content of the XML document.
        document_id: Document id, the archive member's file stem; selects the
            known title and marker legend.

    Returns:
        One row per requirement in document order.
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
    """Download (once) and parse the PURE XML documents named in the config.

    Args:
        config: Loaded configuration; the archive URL, local zip path and
            document members are read from `datasets` with `DEFAULT_CONFIG`
            fallbacks.

    Returns:
        The requirement rows of all configured documents, in document order.
    """
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
    """Apply the seed filter for PURE rows: the `mlm_tapt` screen plus marker checks.

    Args:
        requirement: Requirement text without its marker.
        capability: Automatic capability text derived from it.
        marker: Author marker, `M`, `O` or `I`; `I` (informative) is rejected.
        marker_count: Number of markers on the requirement; zero or several are
            rejected.

    Returns:
        Whether the row passes, and the unique failed check names joined by `;`.
    """
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

    Args:
        document_rows: Requirement rows from `load_pure_requirement_rows`.
        target_count: Total number of seeds to include.
        seed: Random seed for the mandatory draw.

    Returns:
        One candidate row per distinct requirement, with its document context.
        Included rows have an empty final capability text, to be written in
        review.

    Raises:
        ValueError: If the optional requirements exceed `target_count` or too
            few mandatory requirements are eligible to fill it.
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
        capability = auto_capability_text(original, preserve_subject=True)
        marker = str(row.get("marker", ""))
        auto_include, reason = pure_filter(
            # Keep the sampling pool stable. This screen selects sources, not
            # semantically approved clauses; subject-preserving suggestions
            # can still contain a modal and require manual rewriting.
            original,
            auto_capability_text(original),
            marker,
            int(row.get("marker_count", 0)),
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
            # The PURE builder rejects residual modalities before publishing.
            candidate["capability_text_final"] = ""
        elif is_truthy(candidate["auto_include"]):
            candidate["exclusion_reason"] = "not_sampled_mandatory_pool"
    return candidates


def _review_compare_text(value: Any) -> str:
    return strip_final_punctuation(str(value)).lower()


def refresh_capability_suggestions(
    rows: list[dict[str, Any]], force: bool = False
) -> tuple[list[dict[str, Any]], int]:
    """Recompute automatic capability texts and update unedited final texts.

    A final capability text counts as unedited when it is empty or equals the
    old automatic text or the original requirement, ignoring case and final
    punctuation. Reviewer edits are kept unless `force` is set.

    Args:
        rows: Seed review rows; they are copied, not modified.
        force: Overwrite the final text of every included row, even if edited.

    Returns:
        The refreshed rows and the number of included rows whose final text
        changed.
    """
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
    """Apply `refresh_capability_suggestions` to a seed review CSV in place.

    Args:
        path: Seed review CSV; rewritten with its original column order.
        force: Overwrite reviewer-edited final texts as well.

    Returns:
        The number of included rows whose final text changed.
    """
    rows = read_csv_rows(path)
    refreshed, updated = refresh_capability_suggestions(rows, force=force)
    write_csv_rows(path, refreshed, fieldnames=list(read_csv_frame(path).columns))
    return updated


def is_truthy(value: Any) -> bool:
    """Return whether a review cell means yes.

    Args:
        value: Cell value; `1`, `true`, `yes`, `y` and `include` are truthy,
            case-insensitive.

    Returns:
        True for a truthy value.
    """
    return str(value).strip().lower() in {"1", "true", "yes", "y", "include"}


def load_reviewed_seeds(
    path: str | Path,
    target_count: int = DEFAULT_CONFIG["project"]["target_seed_count"],
    strict: bool = True,
) -> list[dict[str, Any]]:
    """Load the included seeds with a final capability text from a review CSV.

    Rows marked for inclusion fall back to the automatic capability text when
    the final one is empty, and rows with neither are skipped.

    Args:
        path: Seed review CSV.
        target_count: Number of included seeds expected.
        strict: Require exactly `target_count` included seeds.

    Returns:
        Copies of the included rows with a cleaned `capability_text_final`.

    Raises:
        ValueError: If `strict` is set and the count differs from
            `target_count`.
    """
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
    """Build the reader-facing table of included seeds and their capability texts.

    Args:
        path: Seed review CSV with `seed_id`, `original_requirement`,
            `capability_text_final` and `include`.

    Returns:
        The included rows with display column names; `Source corpus` is present
        only when the review has a `source_corpus` column.

    Raises:
        ValueError: If a required review column is missing.
    """
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
    """Write the included-capability review table as Markdown and CSV.

    Args:
        path: Seed review CSV.
        output_dir: Directory for the two files; created if missing.
        suffix: Dataset and variant suffix appended to the file names.

    Returns:
        The written paths under `markdown` and `csv`.
    """
    frame = included_capability_review_frame(path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / f"included_capabilities_review{suffix}.md"
    csv_path = output_dir / f"included_capabilities_review{suffix}.csv"
    markdown_path.write_text(frame.to_markdown(index=False) + "\n", encoding="utf-8")
    frame.to_csv(csv_path, index=False)
    return {"markdown": markdown_path, "csv": csv_path}


def capability_clause(capability: str) -> str:
    """Return a capability text without outer quotes or final punctuation.

    Args:
        capability: Capability text to embed in a statement.

    Returns:
        The cleaned clause.
    """
    return strip_final_punctuation(capability)


def source_statement(
    capability: str, modality: str, mandatory_keyword: str = "MUST"
) -> str:
    """Render the controlled source statement of a capability for one modality.

    Mandatory uses the given keyword, recommended `SHOULD`, optional `MAY`, and
    nice-to-have the wish frame "It would be useful if the system could ...".

    Args:
        capability: Capability text.
        modality: `mandatory`, `recommended`, `optional` or `nice_to_have`.
        mandatory_keyword: Keyword of the mandatory variant, `MUST` or `SHALL`;
            uppercased.

    Returns:
        The statement, ending with a period.

    Raises:
        ValueError: If the modality is unknown.
    """
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
    """Look up a weak-phrasing probe template by its id.

    Args:
        template_id: Template id from `WEAK_MODALITY_PROBE_TEMPLATES`.

    Returns:
        The template with `template_id` and `source_template`.

    Raises:
        ValueError: If no template has this id.
    """
    for template in WEAK_MODALITY_PROBE_TEMPLATES:
        if template["template_id"] == template_id:
            return template
    raise ValueError(f"Unknown weak-modality probe template: {template_id}")


def build_weak_modality_probe_items(
    seed_rows: list[dict[str, Any]],
    templates: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Build the weak-phrasing probe items: one per seed and wish template.

    Every item is a nice-to-have statement, so its Task 2 gold modality is
    `nice_to_have` whichever template phrased it.

    Args:
        seed_rows: Reviewed seeds with `seed_id` and `capability_text_final`.
        templates: Templates to render; defaults to
            `WEAK_MODALITY_PROBE_TEMPLATES`.

    Returns:
        The probe items, seed by seed in template order.
    """
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
    """Audit nonempty valid extractions, even without an automatic reference.

    Text-modality recognition determines scoring eligibility, not whether the
    model can audit the output. Unknown and negated wording has no ordinal
    reference relation; it remains available for review and coverage reporting.

    Args:
        benchmark_rows: Benchmark items the Task 2 answers refer to.
        task2_raw_rows: Raw Task 2 output rows. They are deduplicated and
            restricted to the current benchmark; only deterministic, parsed
            answers with a non-empty requirement and a recognised modality
            become items.
        audit_mode: Task 3 audit mode, stored in every item and its id.

    Returns:
        One verification item per (model, source item), carrying the relation
        implied by the declared modality and, when the extracted wording is
        recognised, the reference relation implied by the text.
    """
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
        requirement = str(parsed.get("requirement", "") or "").strip()
        if not requirement:
            continue
        extracted_modality = normalize_modality(parsed.get("modality"))
        if extracted_modality is None:
            continue
        text_diagnostic = requirement_text_modality_diagnostic(requirement)
        text_modality = normalize_modality(text_diagnostic["text_modality"])
        reference_status = (
            "available"
            if text_modality is not None
            else f"{text_diagnostic['text_modality']}_text_modality"
        )
        text_parse_status = "ok" if text_modality is not None else "unknown"
        model = str(raw.get("model", ""))
        source_item_id = str(source_item["item_id"])
        dedupe_key = (model, source_item_id)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        declared_relation = task3_gold_relation(
            source_item["source_modality"], extracted_modality
        )
        relation = (
            task3_gold_relation(source_item["source_modality"], text_modality)
            if text_modality is not None
            else ""
        )
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
                "task2_text_modality": text_diagnostic["text_modality"],
                "task2_text_modality_basis": text_diagnostic["text_modality_basis"],
                "task2_text_modality_parse_status": text_parse_status,
                "task2_confidence": "" if confidence is None else confidence,
                "task3_declared_relation": declared_relation,
                "task3_gold_relation": relation,
                "task3_reference_status": reference_status,
                "task3_audit_mode": audit_mode,
                "ordinal_strength": int(source_item["ordinal_strength"]),
                "numeric_strength": float(source_item["numeric_strength"]),
            }
        )
    return items


def task3_items_from_raw_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recover the Task 3 verification items from Task 3 raw output rows.

    Rows that lack a required item field are skipped, and so are rows with no
    valid reference relation unless they are explicitly unscored because the
    extracted wording is unknown or negated.

    Args:
        raw_rows: Raw output rows; rows of other tasks are ignored.

    Returns:
        One item per distinct item id, restricted to
        `TASK3_VERIFICATION_FIELDS`.
    """
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
        if raw.get("task3_gold_relation") not in TASK3_RELATIONS and raw.get(
            "task3_reference_status"
        ) not in {"unknown_text_modality", "negated_text_modality"}:
            # A missing legacy reference is not an explicitly unscored item.
            continue
        seen.add(item_id)
        item = {field: raw.get(field, "") for field in TASK3_VERIFICATION_FIELDS}
        items.append(item)
    return items


def weak_modality_template_sanity_rows() -> list[dict[str, Any]]:
    """Return blank review rows for checking the weak-phrasing probe templates.

    Each row shows a template with an example statement, so a reviewer can
    confirm that it reads weaker than SHOULD; the review columns are empty.

    Returns:
        One row per template in `WEAK_MODALITY_PROBE_TEMPLATES`.
    """
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
    """Write the weak-phrasing template review sheet as CSV and Markdown.

    An existing CSV is kept so reviewer answers survive; the Markdown table is
    always regenerated from the CSV.

    Args:
        output_dir: Directory for both files; created if missing.
        suffix: Dataset and variant suffix appended to the file names.

    Returns:
        The paths under `csv` and `markdown`.
    """
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

    Covers the four main benchmark conditions (`MUST` wording), the `SHALL`
    robustness variant of the mandatory condition, and the four weak-intent
    phrasing-probe templates. Mirrors `weak_modality_template_sanity_rows` and
    is exported for the experimental-setup documentation.

    Args:
        capability: Capability used to render the example statements.

    Returns:
        One row per template with its condition, variant, template text,
        example statement, intended gold modality and a note.
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
    """Write the modality-template inventory as a CSV plus a sibling Markdown table.

    Args:
        path: CSV path; the Markdown file gets the same name with `.md`.
        capability: Capability used to render the example statements.

    Returns:
        The paths under `csv` and `markdown`.
    """
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
    """Check that every weak-phrasing template was confirmed weaker than SHOULD.

    Args:
        rows: Rows of the template review sheet; yes/no answers are read from
            `weaker_than_should`.

    Returns:
        `valid` plus the template ids that are `missing`, `incomplete` (some
        answer is not yes) or `disagreeing` (some answer is no).
    """
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
    """Return blank construct-review rows: one per reviewer and weak-phrasing template.

    Args:
        reviewer_ids: Reviewer ids to prepare rows for.

    Returns:
        Rows with the template text and an example statement; the reviewer role,
        answer, ordinal rank and note are left empty.
    """
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
    """Check that enough reviewers confirmed every weak-phrasing template.

    Args:
        rows: Rows of the construct review sheet; yes/no answers are read from
            `weaker_than_should` and reviewers from `reviewer_id`.
        expected_reviewers_per_template: Distinct reviewers and answers required
            per template.

    Returns:
        `valid`, the expected reviewer count, `reviewer_counts` per template,
        and the template ids that are `missing`, `insufficient` (too few
        reviewers), `incomplete` (too few or non-yes answers) or `disagreeing`
        (some answer is no).
    """
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
    """Render the mandatory candidate requirement that Task 1 asks about.

    Args:
        capability: Capability text.
        mandatory_keyword: Mandatory keyword, `MUST` or `SHALL`; uppercased.

    Returns:
        "The system MUST <capability>." with the given keyword.
    """
    return f"The system {mandatory_keyword.upper()} {capability_clause(capability)}."


def build_benchmark_items(
    seed_rows: list[dict[str, Any]],
    numeric_strength: dict[str, float] | None = None,
    mandatory_keyword: str = "MUST",
    *,
    passthrough_fields: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Render seed x modality items; `passthrough_fields` copies extra seed columns.

    Each seed yields one item per modality in `MODALITIES`, with its source
    statement, the mandatory candidate requirement, the Task 1 and Task 2 gold
    labels, and the ordinal and numeric strength of the source modality.

    Args:
        seed_rows: Reviewed seeds with `seed_id` and `capability_text_final`.
        numeric_strength: Numeric strength per modality; defaults to
            `NUMERIC_STRENGTH_DEFAULT`.
        mandatory_keyword: Keyword of the mandatory variant, `MUST` or `SHALL`
            (case-insensitive).
        passthrough_fields: Seed columns copied unchanged into every item, such
            as the PURE document context.

    Returns:
        The benchmark items, seed by seed in modality order.

    Raises:
        ValueError: If the mandatory keyword is neither `MUST` nor `SHALL`.
    """
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
    """Build a review table with one row per seed and one column per source statement.

    Args:
        items: Benchmark items as rows, a data frame, or a CSV path.

    Returns:
        One row per seed with the capability, the four source statements and
        the mandatory candidate, under display column names.

    Raises:
        ValueError: If a required benchmark column is missing.
    """
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
    """Write the benchmark statement review table as Markdown and CSV.

    Args:
        items: Benchmark items as rows, a data frame, or a CSV path.
        output_dir: Directory for the two files; created if missing.
        suffix: Dataset and variant suffix appended to the file names.

    Returns:
        The written paths under `markdown` and `csv`.
    """
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
    """Read a frozen prompt template.

    Args:
        path: Prompt file under `prompts/`.

    Returns:
        The template text.
    """
    return Path(path).read_text(encoding="utf-8")


def render_prompt(template: str, **values: Any) -> str:
    """Fill a prompt template's `str.format` placeholders.

    Args:
        template: Prompt template text.
        **values: Placeholder values, such as `source_statement`.

    Returns:
        The rendered prompt.
    """
    return template.format(**values)


def document_context_text(item: Mapping[str, Any]) -> str:
    """Render the document context of a `pure` item for the `document` arm.

    One renderer serves both the single-item template (`context_block`) and the
    per-item `context` value of the batched wrapper, so the two never drift.
    The source statement itself is never repeated here.

    Args:
        item: Benchmark item carrying the `context_*` fields; empty fields are
            left out.

    Returns:
        Newline-separated lines naming the document and its marker legend, the
        section, the preceding requirement, this requirement's id and marker,
        and the following requirement.
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
    """Render the single-item prompt of a benchmark item for Task 1 or Task 2.

    Args:
        task: `"task1"` (mandatory entailment) or `"task2"` (modality
            extraction).
        item: Benchmark item with `source_statement` and, for Task 1,
            `candidate_requirement`.
        task1_template: Task 1 prompt template.
        task2_template: Task 2 prompt template for bare items.
        item_context: `"bare"` or `"document"`; the document arm applies to
            Task 2 only.
        task2_context_template: Task 2 template with a `context_block`
            placeholder, required for the document arm.

    Returns:
        The rendered prompt.

    Raises:
        ValueError: If the task is unsupported, or the document arm is selected
            without a context template.
    """
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


def task_response_schema(task: str, *, batched: bool = False) -> dict[str, Any]:
    """Build the strict provider JSON Schema for `task` from its Pydantic model.

    Derived rather than written out here so the schema a request carries and
    the model Instructor validates against cannot drift apart. The derivation
    reproduces the handwritten schema byte for byte; those bytes are pinned in
    `tests/test_structured_outputs.py` because they are part of the request
    fingerprint every archived run resumes on.

    Args:
        task: Task name with a response model in `structured_outputs`.
        batched: Return the schema of the multi-item batch answer.

    Returns:
        The JSON Schema as a dictionary.

    Raises:
        ValueError: If the task has no response model.
    """
    if task not in so.TASK_RESPONSE_MODELS:
        raise ValueError(f"Unsupported task for JSON Schema response format: {task}")
    return so.provider_schema_for_task(task, batched=batched)


def response_format_for_task(
    task: str, structured_output: str, *, batched: bool = False
) -> dict[str, Any] | None:
    """Build the `response_format` request argument for a structured-output mode.

    Args:
        task: Task whose answer schema is used for `json_schema`.
        structured_output: Structured-output mode or alias.
        batched: Use the multi-item batch schema.

    Returns:
        `None` for `none` and `instructor`, `{"type": "json_object"}` for
        `json_object`, and a strict named JSON Schema format for `json_schema`.
    """
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
    """Decide where a request's response format goes: its own argument or `extra_body`.

    For `instructor` no response format is sent and any `response_format` is
    removed from `extra_body`. For `none` the profile's `response_format` is
    passed through unchanged. Otherwise the task's format is computed; a
    provider that already expects `response_format` inside `extra_body` gets it
    there instead of as the request argument.

    Args:
        task: Task whose answer schema is used.
        structured_output: Structured-output mode or alias; `None` falls back to
            `json_mode`.
        json_mode: Legacy switch used when `structured_output` is `None`.
        response_format: Profile-level response format, used only in `none`
            mode.
        extra_body: Provider-specific request body; copied, not modified.
        batched: Use the multi-item batch schema.

    Returns:
        The `response_format` argument (or `None`) and the `extra_body` to send
        (or `None`).
    """
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
    """Return the config fingerprint for a single completion job.

    Covers the request parameters that change model output but are not part of
    completion_record_key, so config-aware resume can detect stale cached rows.

    Version 2 (see JOB_CONFIG_SHA_VERSION) additionally hashes extra_body,
    response_format, instructor_mode, validation_retries, and seed.

    Version 3 additionally hashes the batching setup: batch_size, batch_order,
    fallback_batch_size, and -- only when batch_size > 1 -- the SHA of the batch
    prompt wrapper for `task` (see `batch_prompt_wrapper_sha`). A batched
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

    Args:
        prompt: Rendered prompt text of the job.
        prompt_version: Prompt version of the run.
        temperature: Sampling temperature.
        top_p: Nucleus-sampling threshold.
        max_tokens: Completion token limit.
        structured_output: Structured-output mode or alias, normalized together
            with `json_mode`.
        json_mode: JSON-mode switch, hashed as given.
        extra_body: Provider-specific request body as it will be sent.
        response_format: Response format argument as it will be sent.
        instructor_mode: Instructor library mode name.
        validation_retries: Instructor validation retries.
        seed: Request seed, or `None` when no seed is sent.
        task: Task whose batch prompt wrapper is hashed for batched plans.
        batch_size: Items per provider request.
        batch_order: Batch composition, `grouped` or `shuffled`.
        fallback_batch_size: Profile setting recorded with the batching setup.
        item_context: `bare` or `document`.

    Returns:
        The hexadecimal SHA-256 of the canonical JSON of the hashed parameters.
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
    """Build one planned single-item completion job with its config fingerprint.

    The response format and `extra_body` are resolved with
    `resolve_response_format_args`, and the fingerprint is computed from the
    resolved values, so it describes the request that is actually sent.

    Args:
        item: Benchmark item the prompt was rendered from; copied into the job.
        task: Task name, such as `"task1"` or `"task2"`.
        model: Model id requested from the provider.
        host: Provider base URL.
        run_id: Run the job belongs to.
        sample_kind: `"deterministic"` or `"stochastic"`.
        sample_index: Index of the sample within its kind.
        temperature: Sampling temperature.
        top_p: Nucleus-sampling threshold.
        prompt: Rendered prompt text.
        prompt_version: Prompt version of the run.
        max_tokens: Completion token limit.
        timeout_s: Request timeout in seconds.
        api_key_env: Name of the environment variable holding the provider key.
        request_index: Position of the job in the run's plan.
        provider_id: Provider id of the profile.
        profile_id: Profile id within the run config.
        run_group_id: Run group of the run config.
        json_mode: JSON-mode switch of the profile.
        structured_output: Structured-output mode or alias.
        response_format: Profile-level response format.
        extra_body: Provider-specific request body.
        instructor_mode: Instructor library mode name.
        validation_retries: Instructor validation retries.
        fallback_batch_size: Profile setting recorded on the job and in its
            fingerprint.
        seed: Request seed.
        send_seed: Send `seed` with the request; when false the job records no
            seed.
        max_retries: Total number of attempts for a request, as in
            `call_with_retries` (3 means one call plus two retries).
        batch_order: Batch composition, `grouped` or `shuffled`.
        batch_size: Items per provider request.
        server_model_probe: Result of the server preflight check, recorded on the
            job.
        item_context: `bare` or `document`.

    Returns:
        The job dictionary, including `job_config_sha` and its version.
    """
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
    """Append the deterministic job and any stochastic samples for one rendered prompt.

    The deterministic job uses `seed`; stochastic sample `i` uses `seed + i + 1`.
    Each job's request index is its position in `jobs`.

    Args:
        jobs: Plan to extend in place.
        item: Benchmark item the prompt was rendered from.
        task: Task name.
        prompt: Rendered prompt text.
        prompt_version: Prompt version of the run.
        model: Model id requested from the provider.
        host: Provider base URL.
        run_id: Run the jobs belong to.
        deterministic: Deterministic sampling settings with `temperature` and
            `top_p`.
        stochastic: Stochastic sampling settings with `temperature`, `top_p` and
            the number of `samples`.
        max_tokens: Completion token limit.
        timeout_s: Request timeout in seconds.
        api_key_env: Name of the environment variable holding the provider key.
        provider_id: Provider id of the profile.
        profile_id: Profile id within the run config.
        run_group_id: Run group of the run config.
        json_mode: JSON-mode switch of the profile.
        structured_output: Structured-output mode or alias.
        response_format: Profile-level response format.
        extra_body: Provider-specific request body.
        instructor_mode: Instructor library mode name.
        validation_retries: Instructor validation retries.
        fallback_batch_size: Profile setting recorded on each job.
        seed: Base request seed.
        send_seed: Send seeds with the requests.
        max_retries: Total number of attempts for a request, as in
            `call_with_retries` (3 means one call plus two retries).
        batch_order: Batch composition, `grouped` or `shuffled`.
        batch_size: Items per provider request.
        server_model_probe: Result of the server preflight check.
        item_context: `bare` or `document`.
    """
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
            seed=seed,
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
                seed=seed + sample_index + 1,
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
    """Plan the deterministic and stochastic completion jobs for benchmark items.

    Each item is rendered once per requested task with `prompt_for_benchmark_task`
    and expanded by `_append_completion_jobs` into one deterministic job plus the
    planned stochastic samples, so request indices follow item order and, within
    an item, task order.

    Args:
        benchmark_rows: Benchmark items to plan.
        tasks: Tasks to plan for every item, resolved with `normalize_task_filter`.
        model: Model id requested from the provider.
        host: Provider base URL.
        run_id: Run the jobs belong to.
        prompt_version: Prompt version of the run.
        task1_template: Task 1 prompt template.
        task2_template: Task 2 prompt template for bare items.
        deterministic: Deterministic sampling settings with `temperature` and
            `top_p`.
        stochastic: Stochastic sampling settings with `temperature`, `top_p` and
            the number of `samples`.
        max_tokens: Completion token limit.
        timeout_s: Request timeout in seconds.
        api_key_env: Name of the environment variable holding the provider key.
        provider_id: Provider id of the profile.
        profile_id: Profile id within the run config.
        run_group_id: Run group of the run config.
        json_mode: JSON-mode switch of the profile.
        structured_output: Structured-output mode or alias.
        response_format: Profile-level response format.
        extra_body: Provider-specific request body.
        instructor_mode: Instructor library mode name.
        validation_retries: Instructor validation retries.
        fallback_batch_size: Profile setting recorded on each job.
        seed: Base request seed; stochastic sample `i` uses `seed + i + 1`.
        send_seed: Send seeds with the requests.
        max_retries: Total number of attempts for a request, as in
            `call_with_retries` (3 means one call plus two retries).
        batch_order: Batch composition, `grouped` or `shuffled`.
        batch_size: Items per provider request.
        server_model_probe: Result of the server preflight check.
        item_context: `bare` or `document`; the document arm applies to Task 2
            only.
        task2_context_template: Task 2 template with a `context_block`
            placeholder, required for the document arm.

    Returns:
        The planned jobs in request order.
    """
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
    `planned_completion_jobs`); `prompt_fn` maps each item to that rendered prompt.

    Args:
        items: Pre-built items, such as Task 3 audit items.
        prompt_fn: Renders the prompt of one item.
        prompt_version: Prompt version of the run.
        model: Model id requested from the provider.
        host: Provider base URL.
        run_id: Run the jobs belong to.
        deterministic: Deterministic sampling settings with `temperature` and
            `top_p`.
        stochastic: Stochastic sampling settings with `temperature`, `top_p` and
            the number of `samples`.
        max_tokens: Completion token limit.
        timeout_s: Request timeout in seconds.
        api_key_env: Name of the environment variable holding the provider key.
        task: Task name stamped on every job.
        provider_id: Provider id of the profile.
        profile_id: Profile id within the run config.
        run_group_id: Run group of the run config.
        json_mode: JSON-mode switch of the profile.
        structured_output: Structured-output mode or alias.
        response_format: Profile-level response format.
        extra_body: Provider-specific request body.
        instructor_mode: Instructor library mode name.
        validation_retries: Instructor validation retries.
        fallback_batch_size: Profile setting recorded on each job.
        seed: Base request seed; stochastic sample `i` uses `seed + i + 1`.
        send_seed: Send seeds with the requests.
        max_retries: Total number of attempts for a request, as in
            `call_with_retries` (3 means one call plus two retries).
        batch_order: Batch composition, `grouped` or `shuffled`.
        batch_size: Items per provider request.
        server_model_probe: Result of the server preflight check.

    Returns:
        The planned jobs in request order.
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
    """Return the first brace-balanced JSON object embedded in a model response.

    The scan starts at the first `{` and tracks brace depth outside string
    literals, so prose or code fences around the object are ignored. The returned
    text is not validated as JSON.

    Args:
        text: Raw response text.

    Returns:
        The substring from the first `{` to its matching `}`, or `None` when the
        text is empty, has no `{`, or the braces never balance.
    """
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
    """Decode the first JSON object or array that parses anywhere in a response.

    Each `{` or `[` position is tried in turn with `json.JSONDecoder.raw_decode`,
    so leading prose and trailing text are tolerated.

    Args:
        text: Raw response text.

    Returns:
        The decoded value, or `None` when the text is empty or nothing parses.
    """
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
    """Map a confidence-scale name or alias to its canonical constant.

    Args:
        value: Scale name such as `0_1`, `probability`, `percent` or `legacy`;
            case is ignored and hyphens count as underscores.
        default: Scale returned for an empty or `None` value.

    Returns:
        `CONFIDENCE_SCALE_0_1` or `CONFIDENCE_SCALE_0_100`.

    Raises:
        ValueError: If the value is not a known alias.
    """
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
    """Return whether a prompt version asks for confidence on the 0-1 scale.

    Args:
        prompt_version: Prompt version label. Versions listed in
            `CONFIDENCE_0_1_PROMPT_VERSIONS` or starting with `v2-` use the 0-1
            scale.

    Returns:
        True for prompt versions with the 0-1 confidence contract.
    """
    normalized = str(prompt_version or "").strip().lower()
    return normalized in CONFIDENCE_0_1_PROMPT_VERSIONS or normalized.startswith("v2-")


def prompt_text_uses_confidence_0_1(prompt: Any) -> bool:
    """Return whether a rendered prompt asks for confidence as a decimal probability.

    Looks for phrases such as `0.0-1.0` or `decimal probability` in the
    whitespace-normalized, lower-cased text.

    Args:
        prompt: Rendered prompt text; `None` or empty counts as no.

    Returns:
        True when the prompt requests confidence on the 0-1 scale.
    """
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
    """Return whether an output contract version reports confidence on the 0-1 scale.

    Args:
        output_contract_version: Contract version recorded on a raw row.

    Returns:
        True when the version is listed in
        `structured_outputs.CONFIDENCE_0_1_OUTPUT_CONTRACT_VERSIONS`.
    """
    return (
        str(output_contract_version or "") in so.CONFIDENCE_0_1_OUTPUT_CONTRACT_VERSIONS
    )


def confidence_scale_for_record(record: Mapping[str, Any]) -> str:
    """Infer the confidence scale a raw record was answered on.

    An explicit `confidence_scale` field wins. Otherwise the output contract
    version, the prompt version and finally the prompt text are checked for the
    0-1 contract; a record with none of these signals is on the 0-100 scale.

    Args:
        record: Raw record or scored row.

    Returns:
        `CONFIDENCE_SCALE_0_1` or `CONFIDENCE_SCALE_0_100`.
    """
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
    """Parse a reported confidence and check that it lies within its scale.

    Booleans are always rejected, and strings are rejected on the 0-1 scale.

    Args:
        value: Confidence value from the parsed response.
        confidence_scale: Scale the value is reported on, in any form accepted by
            `normalize_confidence_scale`.

    Returns:
        The confidence on its own scale, or `None` when it is missing,
        non-numeric or out of range.
    """
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
    """Return whether a row reports confidence on the 0-1 scale.

    Args:
        row: Raw record or scored row, read with `confidence_scale_for_record`.

    Returns:
        True for the 0-1 scale.
    """
    return confidence_scale_for_record(row) == CONFIDENCE_SCALE_0_1


def confidence_probability(
    row_or_parsed: Mapping[str, Any], parsed: Mapping[str, Any] | None = None
) -> float:
    """Return the reported confidence as a probability between 0 and 1.

    Confidences on the 0-100 scale are divided by 100; a missing confidence
    counts as 0.

    Args:
        row_or_parsed: Raw row or parsed response. Its scale fields decide the
            scale, and its `parsed_json`, when present, supplies the confidence
            unless `parsed` is given.
        parsed: Parsed response to read `confidence` from.

    Returns:
        The confidence as a probability.
    """
    if parsed is None and isinstance(row_or_parsed.get("parsed_json"), Mapping):
        parsed = row_or_parsed["parsed_json"]  # type: ignore[index]
    value_source = parsed if parsed is not None else row_or_parsed
    confidence = float(value_source.get("confidence", 0.0))
    return confidence if row_uses_confidence_0_1(row_or_parsed) else confidence / 100.0


def normalize_decision(value: Any) -> str | None:
    """Map a reported yes/no decision to `"yes"` or `"no"`.

    Args:
        value: Decision as reported. `y`, `true` and `1` count as yes, `n`,
            `false` and `0` as no, case-insensitively.

    Returns:
        `"yes"`, `"no"`, or `None` for anything else.
    """
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1"}:
        return "yes"
    if text in {"no", "n", "false", "0"}:
        return "no"
    return None


def normalize_modality(value: Any) -> str | None:
    """Map a modality label or modal verb to its canonical modality category.

    Args:
        value: Label such as `must`, `should`, `may` or `nice to have`; case is
            ignored and hyphens and spaces count as underscores.

    Returns:
        `mandatory`, `recommended`, `optional` or `nice_to_have`, or `None` when
        the label is unknown.
    """
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
    """Map a Task 3 relation label or synonym to its canonical relation.

    Args:
        value: Label such as `preserved`, `upgrade` or `content mismatch`; case is
            ignored and hyphens and spaces count as underscores.

    Returns:
        `preserves`, `strengthens`, `weakens` or `content_changed`, or `None` when
        the label is unknown.
    """
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
    """Derive the gold Task 3 relation from the source and extracted modalities.

    The relation compares the ordinal strengths in `ORDINAL_STRENGTH`, so it is
    never `content_changed`.

    Args:
        source_modality: Modality of the source statement.
        extracted_modality: Modality of the extracted requirement.

    Returns:
        `strengthens`, `weakens` or `preserves`.

    Raises:
        ValueError: If either modality is not a recognized label.
    """
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
    """Turn a value into a lower-case identifier for file names and item ids.

    Runs of characters other than ASCII letters and digits become one underscore,
    and leading and trailing underscores are stripped.

    Args:
        value: Value to convert, such as a run id or model id.
        fallback: Identifier returned when nothing remains.

    Returns:
        The identifier, or `fallback`.
    """
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_").lower()
    return text or fallback


def evidence_phrase_in_source(evidence_phrase: Any, source_statement_text: Any) -> bool:
    """Return whether an evidence phrase occurs verbatim in the source statement.

    The comparison is case-insensitive and ignores whitespace differences.

    Args:
        evidence_phrase: Phrase the model quoted as evidence.
        source_statement_text: Source statement of the item.

    Returns:
        True when the phrase is non-empty and a substring of the source.
    """
    evidence = normalize_space(str(evidence_phrase or "")).lower()
    source = normalize_space(str(source_statement_text or "")).lower()
    return bool(evidence and evidence in source)


def raw_record_matches_benchmark_item(
    raw: Mapping[str, Any], item: Mapping[str, Any]
) -> bool:
    """Return whether a raw record was generated from the item's current source text.

    A record matches when its prompt contains the item's whitespace-normalized
    `source_statement`. Records without a prompt and items without a source
    statement match by default.

    Args:
        raw: Raw record with its rendered `prompt`.
        item: Current benchmark item.

    Returns:
        False only when the prompt lacks the current source statement.
    """
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
    """Keep the raw rows that belong to an item of the current benchmark.

    A row is kept when its `item_id` is in the benchmark and
    `raw_record_matches_benchmark_item` accepts it, which drops rows generated
    from an earlier version of the item text.

    Args:
        benchmark_rows: Current benchmark items.
        raw_rows: Raw records to filter.

    Returns:
        The matching raw rows in input order.
    """
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
    rows for one `run_id` + `completion_record_key`. Scoring and progress must
    count each planned request exactly once; this keeps the LAST row in file order
    whose `parse_status` is `ok` (a later successful retry supersedes an earlier
    failure), or the last row of the group when none parsed. The relative order of
    the surviving rows follows the first appearance of each key so downstream
    ordering (and grouping) stays stable.

    This is the implementation of `AttemptLedger.latest_logical_observations`
    (Section 9); prefer the ledger when a caller needs to say which of the two
    views of a raw file it is consuming.

    Args:
        raw_rows: Raw rows in file order.

    Returns:
        Copies of the surviving rows, one per request.
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
    """Keep the benchmark items that have at least one current raw output.

    A raw row counts when `raw_record_matches_benchmark_item` accepts it for its
    item. With no raw rows at all, every item is kept.

    Args:
        benchmark_rows: Current benchmark items.
        raw_rows: Raw records of the run.

    Returns:
        The items with a matching raw row, in benchmark order.
    """
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
    """Classify a source statement's modality from its template opening.

    Recognizes the openings `The system must`, `The system shall`,
    `The system should`, `The system may` and
    `It would be useful if the system could`, case-insensitively and ignoring
    whitespace differences.

    Args:
        source_statement_text: Source statement of a benchmark item.

    Returns:
        `mandatory`, `recommended`, `optional` or `nice_to_have`, or `None` when
        no template opening matches.
    """
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
    """Return `(cues_found, positive_cues, negated_cues)` for lowercased `text`.

    `cues_found` is the sorted set of surface modal cues seen anywhere in the
    text. A cue counts as negated when it is a negated contraction, when a
    negation token (`not` or `never`) appears within `MODAL_NEGATION_LOOKBACK`
    preceding tokens, or when it is directly followed by `not`, `n't` or `never`.
    The phrase `required to` counts as negated only when `not` or `never`
    directly precedes it.

    Args:
        text: Lower-cased, whitespace-normalized requirement text.

    Returns:
        The sorted unique cues, then the positive and the negated cue
        occurrences, each listed once per occurrence.
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

    Returns the backward-compatible `text_modality` / `text_modality_basis`
    pair plus `text_modality_multi_modal` and `text_modality_modals_found`.

    Precedence, applied in this order:

    1. A weak phrase ("would be nice if", ...) takes precedence over optional
       cues in its wish frame, but never hides a positive obligation or
       recommendation. Mixed weak/strong wording is flagged for review.
    2. A POSITIVE modal cue wins over any negated cue, ranked by
       `MODAL_CATEGORY_PRIORITY` (mandatory > recommended > optional). Text
       such as "The system must ensure that users cannot delete records." states
       a mandatory obligation whose *content* is a prohibition; scoring it as
       `negated` would drop a real strengthening case. Mixed cues still set
       `text_modality_multi_modal` when the categories differ.
    3. Only when there is NO positive cue does a negated modal ("must not",
       "cannot", "shouldn't", ...) resolve to `text_modality = "negated"` with
       basis `negated_modal`, so an explicitly negated obligation is never
       scored as a positive (strengthening) modality.

    Args:
        requirement_text: Generated requirement text; `None` or empty text is
            classified as `unknown`.

    Returns:
        A dict with `text_modality` (a modality category, `negated` or
        `unknown`), `text_modality_basis` (`weak_phrase`, `explicit_modal`,
        `negated_modal`, `heuristic_system_verb` or `unknown`),
        `text_modality_multi_modal` and the sorted cue list
        `text_modality_modals_found`.
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
        # Paraphrases of the weak template that keep the hedge but drop the
        # "if ... could" frame: "It would be useful for the system to ...",
        # "It is desirable that the system ...", "... would be useful." (a
        # trailing hedge on a gerund clause). Seen in glm-5.3, glm-5.3-flash,
        # glm-5.1 and qwen3.8-27b outputs (2026-09-11); they preserve the
        # nice-to-have intent and must not fall through to the unknown or
        # system-verb branches.
        (
            r"^it\s+(?:would|will|could|might)\s+be\s+"
            r"(?:useful|nice|desirable|beneficial|helpful)\s+(?:if|for|to)\b"
        ),
        (
            r"^it\s+is\s+(?:desirable|useful|nice|beneficial|helpful)\s+"
            r"(?:if|for|that|to)\b"
        ),
        r"\blow[-\s]+priority\s+enhancement\b",
        r"\bfuture\s+enhancement\b",
        r"\bnice[-\s]+to[-\s]+have\b",
        r"\bwishlist\b",
    ]
    # A trailing utility remark is a weak frame only without an explicit
    # modal: "exporting reports is useful", not "may export ... is useful".
    trailing_weak = not cues and re.search(
        r"\b(?:would\s+be|is)\s+(?:useful|nice|desirable|beneficial|helpful)\.?$",
        text,
    )
    weak_phrase = bool(trailing_weak) or any(
        re.search(pattern, text) for pattern in weak_patterns
    )
    positive_categories = {MODAL_CUE_CATEGORY[cue] for cue in positive_cues}
    strong_categories = positive_categories & {"mandatory", "recommended"}
    if weak_phrase and strong_categories:
        result["text_modality_multi_modal"] = True
    if weak_phrase and not strong_categories:
        return {
            "text_modality": "nice_to_have",
            "text_modality_basis": "weak_phrase",
            **result,
        }
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
    """Return the text-modality columns with blank values.

    Rows that are not scored for text modality carry these blanks, so every
    scored row has the columns that `text_modality_fields` fills.

    Returns:
        A dict mapping each text-modality column to an empty string.
    """
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
        "strict_text_overcommit_kind": "",
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
    """Score the modality expressed by generated requirement text against the gold.

    The text modality comes from `requirement_text_modality_diagnostic`. Only a
    parsed modality category can over- or under-commit; `negated` and `unknown`
    never do. A strict over-commitment also needs a basis in
    `STRICT_TEXT_MODALITY_BASES`, and its kind records whether the modal itself
    escalated or only the wish frame of a nice-to-have source was dropped.

    Args:
        requirement_text: Generated requirement text.
        gold_modality: Gold modality of the source statement.
        pred_modality: Modality label the model reported; decides
            `label_text_consistent` and the label-correct columns.
        confidence: Reported confidence as a probability, compared with the 0.80
            and 0.90 thresholds.

    Returns:
        The text-modality columns: modality, basis, multi-modal flag and
        `|`-joined cues, parse status, correctness, label consistency, over- and
        under-commitment, strict over-commitment and its kind, and the
        high-confidence variants at 0.80 and 0.90.
    """
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
    # Which strict strengthening this is. The generated text expresses an
    # obligation ("should"/"shall"/"must", strength >= recommended) only when
    # the modal itself moved; a strict-strengthened text that still reads as
    # optional came from a nice_to_have source whose wish frame was dropped
    # while its weak modal ("could"/"may") survived.
    strict_overcommit_kind = ""
    if strict_overcommit:
        strict_overcommit_kind = (
            STRICT_OVERCOMMIT_ESCALATION
            if (text_strength or 0) >= ORDINAL_STRENGTH["recommended"]
            else STRICT_OVERCOMMIT_FRAME_ONLY
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
        "strict_text_overcommit_kind": strict_overcommit_kind,
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


# Names for the concessions the tolerant parser makes relative to the strict
# JSON Schema and Instructor paths. They are recorded under PARSE_REPAIRS_FIELD
# inside the parsed payload, so a raw row tells apart a response the provider
# contract would have accepted as sent (empty list) from one that only parsed
# because we repaired it. Rows written before the field existed carry no key at
# all, which `parse_repairs_for_record` reads as "unknown, treat as none".
PARSE_REPAIRS_FIELD = "parse_repairs"
PARSE_REPAIR_PROSE_WRAPPER = "prose_wrapper"
PARSE_REPAIR_LABEL_ALIAS = "label_alias"
PARSE_REPAIR_UNEXPECTED_FIELDS = "unexpected_fields"

# Fields each task's strict contract allows, plus the batch key that a batched
# item carries. Derived from the response models so "unexpected" always means
# "a key `additionalProperties: false` would have rejected".
TASK_CONTRACT_FIELDS: dict[str, frozenset[str]] = {
    task: frozenset(model.model_fields) | {"request_index"}
    for task, model in so.TASK_RESPONSE_MODELS.items()
}


def _with_parse_repairs(parsed: dict[str, Any], repairs: list[str]) -> dict[str, Any]:
    """Attach the repair list, deduplicated, in the order the repairs applied.

    Args:
        parsed: Parsed payload, updated in place.
        repairs: Repair names in the order they were applied.

    Returns:
        `parsed` with `PARSE_REPAIRS_FIELD` set.
    """
    parsed[PARSE_REPAIRS_FIELD] = list(dict.fromkeys(repairs))
    return parsed


def _repaired_contract_text(
    parsed: dict[str, Any],
    field: str,
    repairs: list[str],
    max_length: int | None = None,
) -> str:
    """Force `field` to the contract's string type and length, naming repairs.

    The strict paths reject an absent, non-string, or over-long value outright;
    here each of those is fixed up and recorded so the row stays auditable.

    Args:
        parsed: Parsed payload, updated in place.
        field: Contract field to repair.
        repairs: Repair list that receives `coerced_<field>`,
            `defaulted_<field>` or `truncated_<field>`.
        max_length: Contract length limit; `None` means no limit.

    Returns:
        The repaired value.
    """
    if field in parsed:
        value = parsed[field]
        if not isinstance(value, str):
            repairs.append(f"coerced_{field}")
            value = str(value)
    else:
        repairs.append(f"defaulted_{field}")
        value = ""
    if max_length is not None and len(value) > max_length:
        repairs.append(f"truncated_{field}")
        value = value[:max_length]
    parsed[field] = value
    return value


def parse_task_response(
    task: str,
    raw_text: str,
    confidence_scale: str = CONFIDENCE_SCALE_0_100,
) -> tuple[dict[str, Any] | None, str]:
    """Parse one response tolerantly, naming every repair applied.

    Deliberately accepts responses the strict JSON Schema and Instructor paths
    reject -- prose around the JSON object, label aliases, a missing or
    over-long free-text field, extra keys -- so a run is not lost to a
    formatting slip. Each concession is listed under `parse_repairs` in the
    returned payload; an empty list means the response conformed as sent.

    Args:
        task: `task1`, `task2` or `task3`; selects the label field and the
            free-text fields.
        raw_text: Raw response text.
        confidence_scale: Scale the confidence is reported on.

    Returns:
        The parsed payload, or `None` when no JSON object decodes, and the parse
        status: `ok`, `invalid_json`, `invalid_confidence`, `invalid_label` or
        `missing_fields`.

    Raises:
        ValueError: If the task is unknown.
    """
    json_text = extract_json_object(raw_text)
    if json_text is None:
        return None, "invalid_json"
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        return None, "invalid_json"

    repairs: list[str] = []
    if json_text.strip() != str(raw_text or "").strip():
        repairs.append(PARSE_REPAIR_PROSE_WRAPPER)
    contract_fields = TASK_CONTRACT_FIELDS.get(task)
    if contract_fields is not None and set(parsed) - contract_fields:
        repairs.append(PARSE_REPAIR_UNEXPECTED_FIELDS)

    confidence = parse_confidence(parsed.get("confidence"), confidence_scale)
    if confidence is None:
        return _with_parse_repairs(parsed, repairs), "invalid_confidence"
    parsed["confidence"] = confidence

    if task == "task1":
        decision = normalize_decision(parsed.get("decision"))
        if decision is None:
            return _with_parse_repairs(parsed, repairs), "invalid_label"
        if parsed.get("decision") != decision:
            repairs.append(PARSE_REPAIR_LABEL_ALIAS)
        parsed["decision"] = decision
        _repaired_contract_text(parsed, "brief_reason", repairs, max_length=200)
        return _with_parse_repairs(parsed, repairs), "ok"

    if task == "task2":
        modality = normalize_modality(parsed.get("modality"))
        if modality is None:
            return _with_parse_repairs(parsed, repairs), "invalid_label"
        if "requirement" not in parsed:
            return _with_parse_repairs(parsed, repairs), "missing_fields"
        if parsed.get("modality") != modality:
            repairs.append(PARSE_REPAIR_LABEL_ALIAS)
        parsed["modality"] = modality
        requirement = _repaired_contract_text(parsed, "requirement", repairs)
        if not requirement.strip():
            # A blank extraction is a failed extraction, not an empty one, and
            # Task2Response rejects it too. No archived raw row is affected.
            return _with_parse_repairs(parsed, repairs), "missing_fields"
        return _with_parse_repairs(parsed, repairs), "ok"

    if task == "task3":
        relation = normalize_relation(parsed.get("relation"))
        if relation is None:
            return _with_parse_repairs(parsed, repairs), "invalid_label"
        if "evidence_phrase" not in parsed:
            return _with_parse_repairs(parsed, repairs), "missing_fields"
        if parsed.get("relation") != relation:
            repairs.append(PARSE_REPAIR_LABEL_ALIAS)
        parsed["relation"] = relation
        _repaired_contract_text(parsed, "evidence_phrase", repairs, max_length=240)
        _repaired_contract_text(parsed, "brief_reason", repairs, max_length=240)
        return _with_parse_repairs(parsed, repairs), "ok"

    raise ValueError(f"Unknown task: {task}")


def parse_instructor_task_response(
    task: str, raw_text: str
) -> tuple[dict[str, Any] | None, str]:
    """Parse one response against the strict per-task response model.

    Validation uses `structured_outputs.validated_json_for_task`, so nothing is
    repaired except lifting the JSON object out of surrounding prose, which is
    recorded as a prose-wrapper repair.

    Args:
        task: Task whose response model validates the object.
        raw_text: Raw response text.

    Returns:
        The validated payload, or `None`, and the parse status: `ok`,
        `invalid_json` or `instructor_validation_error`.
    """
    json_text = extract_json_object(raw_text)
    if json_text is None:
        return None, "invalid_json"
    try:
        parsed = so.validated_json_for_task(task, json_text)
    except Exception:
        return None, "instructor_validation_error"
    # The strict path repairs nothing by construction; its one concession is
    # lifting the object out of surrounding prose before validating it.
    prose = json_text.strip() != str(raw_text or "").strip()
    repairs = [PARSE_REPAIR_PROSE_WRAPPER] if prose else []
    return _with_parse_repairs(parsed, repairs), "ok"


def parse_repairs_for_record(row: Mapping[str, Any]) -> list[str]:
    """Return a raw row's recorded repairs; empty for conforming and pre-field rows.

    Args:
        row: Raw row whose `parsed_json` may carry the repair list.

    Returns:
        The repair names as strings.
    """
    parsed = row.get("parsed_json")
    if not isinstance(parsed, Mapping):
        return []
    repairs = parsed.get(PARSE_REPAIRS_FIELD)
    if not isinstance(repairs, list):
        return []
    return [str(repair) for repair in repairs]


def parse_repair_counts(raw_rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Count parse repairs over raw rows, for run-quality reporting.

    `repaired_records` counts rows that needed at least one repair; the other
    keys count how often each individual repair was applied. A run whose rows
    all parsed as sent reports `{"repaired_records": 0}`.

    Args:
        raw_rows: Raw rows of a run.

    Returns:
        The repair histogram.
    """
    counts: dict[str, int] = {"repaired_records": 0}
    for row in raw_rows:
        repairs = parse_repairs_for_record(row)
        if not repairs:
            continue
        counts["repaired_records"] += 1
        for repair in repairs:
            counts[repair] = counts.get(repair, 0) + 1
    return counts


def batch_prompt_for_completion_jobs(jobs: list[Mapping[str, Any]]) -> str:
    """Render the batched prompt that asks for one result per job.

    Every item is listed as JSON with its `request_index`, under task-specific
    instructions that ask for a `results` array keyed by `request_index` and for
    confidence as a decimal between 0.0 and 1.0. Task 2 items carry their
    document context in the document arm, and Task 3 items carry a declared
    extracted modality in the declared audit modes.

    Args:
        jobs: Planned completion jobs of one batch, in prompt order; all must
            share one task and one item context.

    Returns:
        The rendered batch prompt.

    Raises:
        ValueError: If `jobs` is empty, mixes tasks or item contexts, or its task
            cannot be batched.
    """
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
    """Return the SHA-256 of the batch prompt wrapper for `task` and `item_context`.

    Renders `batch_prompt_for_completion_jobs` for two fixed dummy jobs
    (request_index 0 and 1, fixed placeholder statements) so the digest depends
    only on the instruction envelope and item layout, not on real benchmark
    content. Hashed into `compute_job_config_sha` for batched plans
    (`JOB_CONFIG_SHA_VERSION` 3) so editing the wrapper invalidates cached
    batched rows on resume.

    Args:
        task: Task whose wrapper is fingerprinted.
        item_context: `bare` or `document`.

    Returns:
        The hex digest, cached per argument pair.
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
    """Parse a batched response into per-item results keyed by request index.

    Accepts a `{"results": [...]}` object or a bare array. Each result is keyed by
    its `request_index`, or by `id` when that is absent; a result without a
    usable index gets the negative key `-(position + 1)`, so it matches no
    planned job. Entries that are not objects are skipped, and only the first
    result per index is kept.

    Args:
        raw_text: Raw response text.

    Returns:
        The results by request index and the parse status: `ok`, `invalid_json`,
        `missing_results` or `duplicate_request_index`.
    """
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
    """Return the SHA-256 over the canonical JSON of the request payload actually sent.

    Args:
        request_kwargs: Keyword arguments of the provider call; only the fields in
            `REQUEST_PAYLOAD_SHA_FIELDS` are hashed.

    Returns:
        The hex digest.
    """
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


def transcript_request_payload(request_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy of the payload actually sent, for the run transcript.

    Everything a provider request carries is already JSON except Instructor's
    `response_model`, which is a Pydantic class; it is rendered as its name so
    the transcript stays a plain JSONL file. The API key is never here -- it
    lives on the client, not in the payload.

    Args:
        request_kwargs: Keyword arguments of the provider call.

    Returns:
        The payload, with values that are not JSON-serializable replaced by their
        `__name__` or `repr`.
    """
    payload: dict[str, Any] = {}
    for key, value in request_kwargs.items():
        try:
            json.dumps(value)
        except TypeError:
            payload[key] = getattr(value, "__name__", None) or repr(value)
        else:
            payload[key] = value
    return payload


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


def is_model_output_format_error(exc: BaseException) -> bool:
    """Recognize the server's explicit generated-output rejection, not HTTP alone.

    Re-generating this answer would replace a failed sample with a new draw.
    Match narrowly: other 5xx errors still use the ordinary transport policy.
    Some validation clients wrap the original exception in their cause chain.

    Args:
        exc: Exception raised by a provider call; its cause and context chain is
            searched as well.

    Returns:
        True when the message or body of an exception in the chain says the model
        output does not match the expected PEG-native format.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = str(current).lower()
        body = str(getattr(current, "body", "") or "").lower()
        if (
            "model produced output that does not match the expected peg-native format"
            in (message + " " + body)
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def is_transient_provider_error(exc: BaseException) -> bool:
    """Return True for transport or service failures, not generated-output rejections.

    Deliberately excludes 400/401/403-class failures: a budget or auth rejection
    will never succeed on retry and must surface immediately.

    Args:
        exc: Exception raised by a provider call.

    Returns:
        True for status codes 408, 429 and 5xx and, when no status code is
        available, for OpenAI timeout, connection, rate-limit and server errors
        and builtin timeout and connection errors. False for 400, 401, 403, 404
        and 422 and for generated-output rejections.
    """
    if is_model_output_format_error(exc):
        return False
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
    """Return an exponential backoff delay with full jitter for a zero-based attempt.

    Args:
        attempt: Zero-based attempt number.
        base_delay_s: Delay scale in seconds.

    Returns:
        A delay drawn uniformly between 0 and `base_delay_s * 2**attempt`.
    """
    return random.uniform(0.0, base_delay_s * (2**attempt))


def call_with_retries(
    call: Callable[[], Any],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay_s: float = RETRY_BASE_DELAY_S,
    on_attempt_error: Callable[[int, BaseException], None] | None = None,
) -> tuple[Any, int, BaseException | None]:
    """Run `call`, retrying transient provider failures with bounded backoff.

    `max_retries` is the number of total attempts (3 means one initial call plus
    two retries); non-transient errors fail fast on the first attempt.

    `on_attempt_error(attempt, exc)` is called for every attempt that is
    retried, so a transcript can keep the attempts that only survive here as a
    count. The final failure is returned rather than reported through it.

    Args:
        call: Provider call without arguments.
        max_retries: Total number of attempts; at least one is made.
        base_delay_s: Backoff scale passed to `retry_delay_seconds`.
        on_attempt_error: Callback for each failed attempt that is retried.

    Returns:
        `(result, retry_count, error)`: the call's result or `None`, the number of
        retries made, and the final exception or `None` on success.
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
            if on_attempt_error is not None:
                on_attempt_error(attempt, exc)
            retry_count += 1
            time.sleep(retry_delay_seconds(attempt, base_delay_s))
    return None, retry_count, last_error


def provider_error_record(error: BaseException) -> dict[str, Any]:
    """Retain an HTTP failure's body without copying request headers or credentials.

    Args:
        error: Exception raised by a provider call.

    Returns:
        A dict with `error` (the exception's repr), `error_kind`
        (`model_output_format` or `provider_error`), `status_code`,
        `response_json` (the decoded response or the error body) and `raw_text`.
    """
    response = getattr(error, "response", None)
    body = getattr(error, "body", None)
    raw_text = ""
    if response is not None:
        try:
            raw_text = response.text
            body = response.json()
        except (ValueError, AttributeError):
            pass
    return {
        "error": repr(error),
        "error_kind": (
            "model_output_format"
            if is_model_output_format_error(error)
            else "provider_error"
        ),
        "status_code": getattr(
            error, "status_code", getattr(response, "status_code", None)
        ),
        "response_json": body,
        "raw_text": raw_text,
    }


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
    """Send one chat-completion request to an OpenAI-compatible endpoint.

    The SDK's own retries are disabled, so `call_with_retries` is the only retry
    mechanism and every retried attempt is recorded. Every result carries the
    transcript provenance: the JSON-safe request payload, its
    `request_payload_sha`, the request seed, the errors of retried attempts and
    the start time.

    Args:
        host: Provider base URL.
        model: Model id requested from the provider.
        prompt: Prompt sent as the single user message.
        temperature: Sampling temperature.
        top_p: Nucleus-sampling threshold.
        max_tokens: Completion token limit.
        timeout_s: Request timeout in seconds.
        api_key_env: Name of the environment variable holding the provider key;
            `EMPTY` is sent when it is unset.
        logprobs: Request token log probabilities.
        top_logprobs: Alternatives per token, sent only with `logprobs`.
        response_format: Response format sent with the request.
        extra_body: Provider-specific request body.
        seed: Request seed; `None` sends no seed.
        max_retries: Total attempts allowed by `call_with_retries`.

    Returns:
        A result with `ok`, `raw_text`, `response_json`, `latency_s`, `error` and
        `retry_count` plus the provenance fields; a failed request also carries
        the fields of `provider_error_record`.
    """
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
    # Carried on every return so the run transcript can record what was sent,
    # including the attempts call_with_retries would otherwise only count.
    attempt_errors: list[dict[str, Any]] = []
    provenance = {
        "request_payload": transcript_request_payload(request_kwargs),
        "request_payload_sha": payload_sha,
        "request_seed": request_kwargs.get("seed"),
        "attempt_errors": attempt_errors,
        "started_at_utc": utc_now_iso(),
    }

    start = time.perf_counter()
    response, retry_count, error = call_with_retries(
        lambda: client.chat.completions.create(**request_kwargs),
        max_retries=max_retries,
        on_attempt_error=lambda _attempt, exc: attempt_errors.append(
            provider_error_record(exc)
        ),
    )
    latency_s = time.perf_counter() - start
    if error is not None:
        return {
            "ok": False,
            **provider_error_record(error),
            "latency_s": latency_s,
            "error": repr(error),
            "retry_count": retry_count,
            **provenance,
        }
    response_json = None
    try:
        response_json = response.model_dump(mode="json")
        raw_text = response.choices[0].message.content or ""
    except Exception as exc:  # malformed provider payload
        return {
            "ok": False,
            "raw_text": "",
            "response_json": response_json,
            "latency_s": latency_s,
            "error": repr(exc),
            "retry_count": retry_count,
            **provenance,
        }
    return {
        "ok": True,
        "raw_text": raw_text,
        "response_json": response_json,
        "latency_s": latency_s,
        "error": "",
        "retry_count": retry_count,
        **provenance,
    }


def normalize_instructor_mode_name(value: Any) -> str:
    """Map an Instructor mode name or alias to its canonical name.

    Args:
        value: Mode name such as `json`, `tools`, `strict` or `markdown_json`;
            empty means `json`, case is ignored and hyphens count as underscores.

    Returns:
        `json`, `tools`, `tools_strict` or `md_json`.

    Raises:
        ValueError: If the name is not a known alias.
    """
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
    """Return the `instructor.Mode` member for an Instructor mode name.

    Args:
        value: Mode name or alias accepted by `normalize_instructor_mode_name`.

    Returns:
        The matching `instructor.Mode` member.
    """
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
    """Return `extra_body` without `response_format`, for the Instructor path.

    Instructor derives the response schema from the response model, so a
    `response_format` entry is dropped.

    Args:
        extra_body: Provider-specific request body of the profile.

    Returns:
        The remaining entries, or `None` when none remain.
    """
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
    """Send one request through Instructor, validating the reply with a response model.

    `response_format` is accepted for call-signature parity with
    `chat_completion` and ignored, because Instructor derives the schema from the
    response model. Two retry loops stay separate: Instructor's own loop retries
    failed attempts, including schema-validation failures, but never the server's
    generated-output rejection, while `call_with_retries` retries transport
    failures and records them.

    Args:
        host: Provider base URL.
        model: Model id requested from the provider.
        prompt: Prompt sent as the single user message.
        temperature: Sampling temperature.
        top_p: Nucleus-sampling threshold.
        max_tokens: Completion token limit.
        timeout_s: Request timeout in seconds.
        api_key_env: Name of the environment variable holding the provider key;
            `EMPTY` is sent when it is unset.
        response_format: Ignored; see above.
        extra_body: Provider-specific request body, cleaned with
            `instructor_extra_body`.
        task: Task whose response model is used when `response_model` is not
            given.
        batched: Use the batched response model of the task.
        instructor_mode: Instructor mode name or alias.
        validation_retries: Attempts Instructor makes to obtain a schema-valid
            reply; at least one is made.
        response_model: Pydantic model that overrides the task's response model.
        seed: Request seed; `None` sends no seed.
        max_retries: Total transport attempts allowed by `call_with_retries`.

    Returns:
        A result with `ok`, `raw_text`, `response_json`, `latency_s`, `error`,
        `retry_count` and the request payload, its SHA, seed and attempt errors.
        On success `raw_text` is the validated payload as JSON and
        `response_json` holds it under `instructor_validated`, merged with the
        provider response. On failure the result also carries the fields of
        `provider_error_record`, `raw_text` holds the rejected output or the last
        completion, and `parse_status_override` is `instructor_validation_error`
        for validation and retry errors.
    """
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
    attempt_errors: list[dict[str, Any]] = []
    provenance = {
        "request_payload": transcript_request_payload(request_kwargs),
        "request_payload_sha": payload_sha,
        "request_seed": request_kwargs.get("seed"),
        "attempt_errors": attempt_errors,
    }

    start = time.perf_counter()
    # Instructor's default retry object retries all exceptions, not only schema
    # validation failures. Exclude native generation rejections there as well.
    from tenacity import Retrying, retry_if_exception, stop_after_attempt

    def validated_attempt() -> Any:
        return client.chat.completions.create(
            **{
                **request_kwargs,
                "max_retries": Retrying(
                    stop=stop_after_attempt(max(1, int(request_kwargs["max_retries"]))),
                    retry=retry_if_exception(
                        lambda exc: not is_model_output_format_error(exc)
                    ),
                ),
            }
        )

    parsed, retry_count, error = call_with_retries(
        validated_attempt,
        max_retries=max_retries,
        on_attempt_error=lambda _attempt, exc: attempt_errors.append(
            provider_error_record(exc)
        ),
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
    error_record = provider_error_record(error)
    raw_text = (
        error_record["raw_text"]
        if is_model_output_format_error(error)
        else str(getattr(error, "last_completion", "") or "")
    )
    return {
        "ok": False,
        **error_record,
        "raw_text": raw_text,
        "response_json": error_record["response_json"],
        "latency_s": latency_s,
        "error": repr(error),
        "parse_status_override": (
            "" if is_model_output_format_error(error) else parse_status_override
        ),
        "retry_count": retry_count,
        **provenance,
    }


def normalize_logprob_tokens(content: Any) -> list[dict[str, Any]]:
    """Normalize a provider's token log-probability entries to one shape.

    Entries that are not dicts or whose `logprob` is not numeric are skipped.

    Args:
        content: Token entries as returned by the provider, each with `token`,
            `logprob` and optionally `top_logprobs`.

    Returns:
        Dicts with the `token` as a string, the `logprob` as a float and the
        `top_logprobs` as given; empty when `content` is not a list.
    """
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
    """Extract token log probabilities from a chat-completion or Responses API payload.

    Looks in the first choice's `logprobs.content` (chat completions), then in
    the `logprobs` of the `output` items and their `content` parts (Responses
    API), and finally in a top-level `logprobs` list; the first location that
    yields tokens wins.

    Args:
        response_json: Provider response payload.

    Returns:
        The normalized token entries, empty when none are found.
    """
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
    """Return the Responses API endpoint for a provider base URL.

    Args:
        host: Provider base URL, with or without `/v1` or `/responses` at the end.

    Returns:
        The URL ending in `/v1/responses`, or the host itself when it already
        ends in `/responses`.
    """
    base = host.rstrip("/")
    if base.endswith("/responses"):
        return base
    if base.endswith("/v1"):
        return f"{base}/responses"
    return f"{base}/v1/responses"


def responses_output_text(response_json: dict[str, Any] | None) -> str:
    """Return the generated text of a Responses API payload.

    Uses `output_text` when present; otherwise concatenates the `text` of each
    `output` item and of its `content` parts.

    Args:
        response_json: Responses API payload.

    Returns:
        The text, empty when the payload is not a dict or carries none.
    """
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
    """Probe whether an endpoint returns token log probabilities via the Responses API.

    Posts `LOGPROB_PROBE_PROMPT` to the Responses endpoint and asks for
    output-text log probabilities with five alternatives per token. Request
    failures are recorded in the result rather than raised.

    Args:
        host: Provider base URL.
        model: Model id to probe.
        api_key_env: Name of the environment variable holding the provider key;
            no `Authorization` header is sent when it is unset.
        timeout_s: Request timeout in seconds.
        max_tokens: Output token limit of the probe.

    Returns:
        The probe result with `model`, `host`, `endpoint`, `supported` (the
        request succeeded and returned tokens), `token_count`, the requested
        `include` and `top_logprobs`, `raw_text` (the output text or the first 500
        characters of the body), `error` and `latency_s`.
    """
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
    """Return a new run id built from a prefix, the local time and a random suffix.

    Args:
        prefix: Leading part of the id.

    Returns:
        An id of the form `<prefix>-YYYYmmdd-HHMMSS-<8 hex digits>`.
    """
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

    Args:
        response_json: Provider response payload; anything but a mapping yields
            the defaults.

    Returns:
        `finish_reason`, the prompt, completion and total token counts (the
        Responses API names `input_tokens` and `output_tokens` are accepted),
        `served_model`, `system_fingerprint` and `response_id`.
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
    """Build the raw record of one item from a completed provider request.

    The response is parsed with `parse_instructor_task_response` for the
    Instructor output contract and with `parse_task_response` otherwise. A
    completion's `parse_status_override` replaces the parse status and drops the
    parsed payload; a failed request becomes `model_output_error` or
    `request_error`, and a parse failure cut off at the token limit becomes
    `truncated`. Optional provenance is written only when set, selected item
    fields are copied, and every field except `parsed_json` is checked with
    `assert_no_credential_shaped_values` before the record is returned.

    Args:
        run_id: Run the record belongs to.
        model: Model id requested from the provider.
        host: Provider base URL.
        task: Task name; selects the response parser.
        item: Benchmark or audit item, with at least `item_id`, `seed_id` and
            `source_modality`.
        sample_index: Index of the sample within its kind.
        sample_kind: `deterministic` or `stochastic`.
        temperature: Sampling temperature.
        top_p: Nucleus-sampling threshold.
        prompt_version: Prompt version of the run.
        prompt: Single-item prompt of the job.
        completion: Result of `chat_completion` or `instructor_completion`.
        request_index: Position of the job in the run's plan.
        provider_id: Provider id of the profile.
        profile_id: Profile id within the run config.
        run_group_id: Run group of the run config.
        base_url: Provider base URL recorded as `base_url`.
        api_key_env: Name of the environment variable holding the provider key.
        json_mode: JSON-mode switch of the profile.
        structured_output: Structured-output mode or alias; `none` is not
            recorded.
        response_format: Response format sent with the request.
        request_extra_body: Provider-specific request body that was sent.
        server_model_probe: Result of the server preflight check.
        batch_id: Id of the batched request; empty for single-item requests.
        batch_size: Batch size recorded on the row.
        batch_item_count: Number of items the batched request carried.
        batch_prompt_hash: Hash of the batch prompt.
        output_contract_version: Output contract of the request; when empty and
            the prompt version or text asks for 0-1 confidence, the prompt
            contract is recorded.
        confidence_scale: Confidence scale of the response; inferred with
            `confidence_scale_for_record` when empty.
        job_config_sha: Config fingerprint of the job.
        job_config_sha_version: Fingerprint version recorded on the row.
        request_seed: Seed sent with the request; recorded when the completion
            carries none.
        max_tokens: Completion token limit, used with the sampling settings,
            `request_seed`, `response_format` and `request_extra_body` to
            recompute the payload SHA when the completion carries none.
        batch_order: Batch composition, `grouped` or `shuffled`.
        batch_seed_ids: Seed ids sharing the batched request; defaults to the
            item's own seed id.
        batch_variant_mix: Number of distinct source modalities in the batch;
            defaults to 1.
        item_context: `bare` or `document`.

    Returns:
        The raw record.
    """
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
        parse_status = (
            "model_output_error"
            if completion.get("error_kind") == "model_output_format"
            else "request_error"
        )

    response_fields = extract_response_fields(completion.get("response_json"))
    # A response cut off at the token limit is a budget problem, not malformed
    # model output: give it its own status so it is not silently filed as
    # invalid_json. It remains a parse failure (parse_status != "ok") everywhere.
    if response_fields["finish_reason"] == "length" and parse_status not in {
        "ok",
        "request_error",
        "model_output_error",
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
        "error_kind": completion.get("error_kind", ""),
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
        "task3_reference_status",
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
    """Summarize which items share one batched provider request, for auditing.

    Args:
        jobs: Jobs of one batched request.

    Returns:
        `batch_seed_ids`, the sorted distinct seed ids, and `batch_variant_mix`,
        the number of distinct source modalities.
    """
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

    Args:
        job: Planned job; supplies the run, model, sampling, prompt and
            provenance fields.
        completion: Result of the completion call.
        request_index: Position of the job in the run's plan.
        response_format: Response format sent with the request.
        request_extra_body: Provider-specific request body that was sent.
        batch_id: Id of the batched request; empty for single-item requests.
        batch_size: Batch size recorded on the row.
        batch_item_count: Number of items the batched request carried.
        batch_prompt_hash: Hash of the batch prompt.
        output_contract_version: Output contract of the request.
        confidence_scale: Confidence scale of the response; inferred when empty.
        batch_jobs: Jobs that shared the request, summarized with
            `batch_composition`; defaults to `job` alone.

    Returns:
        The raw record built by `build_raw_record`.
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
    """Return whether a job is answered through Instructor.

    Args:
        job: Planned completion job; its `structured_output` mode decides.

    Returns:
        True when the structured-output mode is `instructor`.
    """
    return (
        normalize_structured_output_mode(job.get("structured_output")) == "instructor"
    )


def output_contract_version_for_job(job: Mapping[str, Any]) -> str:
    """Return the output contract version a job's responses follow.

    Args:
        job: Planned completion job.

    Returns:
        `structured_outputs.INSTRUCTOR_OUTPUT_CONTRACT_VERSION` for Instructor
        jobs, `structured_outputs.PROMPT_OUTPUT_CONTRACT_VERSION` for prompt
        versions with 0-1 confidence, and an empty string otherwise.
    """
    if job_uses_instructor(job):
        return so.INSTRUCTOR_OUTPUT_CONTRACT_VERSION
    if prompt_version_uses_confidence_0_1(job.get("prompt_version")):
        return so.PROMPT_OUTPUT_CONTRACT_VERSION
    return ""


def confidence_scale_for_job(job: Mapping[str, Any]) -> str:
    """Return the confidence scale a job's responses are known to use.

    Args:
        job: Planned completion job.

    Returns:
        `CONFIDENCE_SCALE_0_1` for Instructor jobs and prompt versions with 0-1
        confidence; otherwise an empty string, which leaves the scale to be
        inferred from the record.
    """
    if job_uses_instructor(job) or prompt_version_uses_confidence_0_1(
        job.get("prompt_version")
    ):
        return CONFIDENCE_SCALE_0_1
    return ""


def completion_runner_for_job(
    job: Mapping[str, Any],
    completion_fn: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    """Return the completion function that answers a job.

    Instructor jobs are routed to `instructor_completion` when the caller passed
    the default `chat_completion`; any other function, such as a test fake, is
    used as given.

    Args:
        job: Planned completion job.
        completion_fn: Completion function chosen by the caller.

    Returns:
        `instructor_completion` or `completion_fn`.
    """
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
    """Build the keyword arguments of the completion call for a job.

    Instructor jobs also get the task, the batching flag, the Instructor mode, the
    validation retries and the task's response model.

    Args:
        job: Planned completion job; supplies host, model, sampling, timeout, key,
            seed and retry settings.
        prompt: Prompt to send, the job's own or a batch prompt.
        max_tokens: Completion token limit of the request.
        response_format: Response format to send.
        extra_body: Provider-specific request body to send.
        batched: The request answers several jobs; selects the batched response
            model.

    Returns:
        Keyword arguments for `chat_completion` or `instructor_completion`.
    """
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


class TranscriptSink(Protocol):
    """What the job runners need from `run_transcripts.TranscriptWriter`.

    Structural, so `eval_utils` keeps depending on nothing above it: the writer
    imports this module, not the other way round.
    """

    def record(
        self,
        job: Any,
        completion: Any,
        *,
        batch_id: str = "",
        batch_size: int = 1,
        request_indices: Any = (),
    ) -> None:
        """Write the attempts behind one provider call.

        Args:
            job: Any planned completion job of the call (the first one, for a
                batch); supplies the run identity.
            completion: The dict the completion function returned.
            batch_id: Batch wrapper identifier; empty for a single-item call.
            batch_size: Number of items the call carried.
            request_indices: Request indices of the items the call covered.
        """
        ...


def run_completion_job(
    job: Mapping[str, Any],
    completion_fn: Callable[..., dict[str, Any]] = chat_completion,
    *,
    transcript: TranscriptSink | None = None,
) -> dict[str, Any]:
    """Run one single-item completion job and build its raw record.

    Args:
        job: Planned completion job.
        completion_fn: Completion function; Instructor jobs are routed to
            `instructor_completion` when this is `chat_completion`.
        transcript: Sink that records the request and its completion.

    Returns:
        The raw record of the job.
    """
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
    if transcript is not None:
        transcript.record(
            job,
            completion,
            batch_size=1,
            request_indices=() if request_index is None else (int(request_index),),
        )
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
    """Return the request settings that jobs must share to be batched together.

    The key covers endpoint, model, run, task, sample kind and index, sampling,
    prompt version, limits, profile, structured-output and seed settings and the
    batch order; the items themselves are not part of it.

    Args:
        job: Planned completion job.

    Returns:
        The batching key.
    """
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
    """Build deterministic seed-disjoint batches for one batch-key group.

    Seeding the RNG from a SHA-256 of `(seed, batch key)` keeps the permutation
    stable across processes and independent of dict iteration order, so a
    `shuffled` run is exactly reproducible from its recorded seed.

    The permutation is applied to the SEED GROUPS (and to the variants within a
    seed), not to the flat job list: the jobs of one seed are then handed out to
    distinct batches, always filling the currently emptiest one. So a shuffled
    batch holds `batch_size` DIFFERENT seeds -- the point of the ablation is to
    break the grouped default's "4 variants of one seed in one prompt" confound,
    and a plain shuffle would still collide two variants of a seed in the same
    request by chance. When a group has fewer distinct seeds than `batch_size`
    (or a seed has more jobs than there are batches) a collision is unavoidable;
    the batches are still filled, and the shortfall is logged as a warning rather
    than passing silently.

    Args:
        ordered: Jobs of one batch-key group in request order.
        group_key: Batch key of the group, mixed into the RNG seed.
        seed: Recorded seed of the run.
        batch_size: Jobs per batch; values below 1 count as 1.

    Returns:
        The batches, as many as needed to hold all jobs at `batch_size` per
        batch.

    Raises:
        RuntimeError: If no batch has room for a job, which the batch count rules
            out.
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
    jobs that still have to run in `jobs`, and the batches are computed over the
    plan and then filtered (by `completion_record_key`) to those pending jobs,
    dropping batches that are left empty. Without it a resumed run would
    re-shuffle the pending subset and send different batch neighbours than the
    first attempt.

    Args:
        jobs: Jobs to batch; with `planned_jobs`, the jobs that still have to
            run.
        batch_size: Jobs per provider request; 1 gives one batch per job.
        batch_order: `grouped` or `shuffled`; defaults to the first job's
            `batch_order`.
        seed: Seed of the shuffled order; defaults to the first job's `seed`.
        planned_jobs: Full plan of the run, for batch-stable resume.

    Returns:
        The batches; batch-key groups are ordered by their smallest request
        index, and an empty `jobs` gives no batches.
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
    """Return the driver provenance fields present on a batch `completion`.

    Absent keys are omitted rather than defaulted, so `build_raw_record` keeps its
    fallbacks (job seed, locally recomputed payload sha) for drivers that do not
    report them.

    Args:
        completion: Result of the batch's provider call.

    Returns:
        The fields of `BATCH_DRIVER_PROVENANCE_FIELDS` that the completion carries.
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
    """Split a batched Instructor response into valid results and jobs to re-send.

    A job's result, without its `request_index`, must pass
    `parse_instructor_task_response`, and the response must not carry results for
    request indices outside the batch; otherwise the job falls back to a
    single-item request.

    Args:
        task: Task whose response model validates each result.
        parsed_results: Results of the batched response keyed by request index,
            as returned by `parse_batch_completion_results`.
        jobs: Jobs of the batch.

    Returns:
        The valid results by request index and copies of the jobs to re-send.
    """
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
    *,
    transcript: TranscriptSink | None = None,
) -> list[dict[str, Any]]:
    """Run a batch of Instructor jobs in one request, re-sending failed items alone.

    A batch of one job runs as a single-item job. Otherwise the batch prompt is
    sent with the batched response model and a token limit scaled by the number of
    jobs. A generated-output rejection becomes the record of every job in the
    batch. Otherwise each item whose result validates gets a record carrying the
    batch request's provenance, and every other job is re-run on its own.

    Args:
        jobs: Jobs of one batch, sharing a batch key.
        completion_fn: Completion function; routed to `instructor_completion`
            when this is `chat_completion`.
        transcript: Sink that records every provider call.

    Returns:
        The raw records in job order.
    """
    if not jobs:
        return []
    if len(jobs) == 1:
        return [
            run_completion_job(
                jobs[0], completion_fn=completion_fn, transcript=transcript
            )
        ]

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
    if transcript is not None:
        transcript.record(
            first,
            completion,
            batch_id=batch_id,
            batch_size=len(jobs),
            request_indices=[int(job["request_index"]) for job in jobs],
        )

    if completion.get("error_kind") == "model_output_format":
        return [
            _job_record(
                job,
                completion=completion,
                request_index=int(job["request_index"]),
                response_format=None,
                request_extra_body=instructor_extra_body(job.get("extra_body")),
                batch_id=batch_id,
                batch_size=len(jobs),
                batch_item_count=len(jobs),
                batch_prompt_hash=batch_prompt_hash,
                batch_jobs=jobs,
                output_contract_version=output_contract_version_for_job(job),
                confidence_scale=confidence_scale_for_job(job),
            )
            for job in jobs
        ]
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
        fallback_record = run_completion_job(
            fallback_job, completion_fn=completion_fn, transcript=transcript
        )
        records_by_request_index[int(fallback_job["request_index"])] = fallback_record
    return [records_by_request_index[int(job["request_index"])] for job in jobs]


def run_completion_batch(
    jobs: list[Mapping[str, Any]],
    completion_fn: Callable[..., dict[str, Any]] = chat_completion,
    *,
    transcript: TranscriptSink | None = None,
) -> list[dict[str, Any]]:
    """Run a batch of jobs in one provider request, re-sending failed items alone.

    A batch of one job runs as a single-item job, and Instructor batches go to
    `run_instructor_completion_batch`. Otherwise the batch prompt is sent with the
    batched response format and a token limit scaled by the number of jobs. A
    generated-output rejection becomes the record of every job in the batch,
    because it is already that sample's result. When the request fails, every
    job is re-sent alone through `_batch_fallback_record`, and so is each job
    whose result is missing from the response; the other jobs get a record
    carrying the batch request's provenance.

    Args:
        jobs: Jobs of one batch, sharing a batch key.
        completion_fn: Completion function for the batch request and the
            single-item re-sends.
        transcript: Sink that records every provider call.

    Returns:
        The raw records in job order.
    """
    if not jobs:
        return []
    if len(jobs) == 1:
        return [
            run_completion_job(
                jobs[0], completion_fn=completion_fn, transcript=transcript
            )
        ]
    if job_uses_instructor(jobs[0]):
        return run_instructor_completion_batch(
            jobs, completion_fn=completion_fn, transcript=transcript
        )

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
    if transcript is not None:
        transcript.record(
            first,
            completion,
            batch_id=batch_id,
            batch_size=len(jobs),
            request_indices=[int(job["request_index"]) for job in jobs],
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
        if completion.get("error_kind") == "model_output_format":
            # A native generation rejection is already this sample's result.
            # Do not replace it with a fresh single-item generation.
            records_by_request_index[request_index] = _job_record(
                job,
                completion=completion,
                request_index=request_index,
                response_format=batch_response_format,
                request_extra_body=batch_extra_body,
                batch_id=batch_id,
                batch_size=len(jobs),
                batch_item_count=len(jobs),
                batch_prompt_hash=batch_prompt_hash,
                batch_jobs=jobs,
            )
            continue
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
                transcript=transcript,
            )
        )
    return [records_by_request_index[int(job["request_index"])] for job in jobs]


def _batch_fallback_record(
    job: Mapping[str, Any],
    *,
    completion_fn: Callable[..., dict[str, Any]],
    batch_error: str,
    transcript: TranscriptSink | None = None,
) -> dict[str, Any]:
    """Re-send one job of a failed batch as a single-item request.

    The record is the fallback's own record (its status, its payload sha), tagged
    with `batch_size = 1` so a batched run stays auditable down to the rows that
    were actually sent alone. Only when the fallback itself fails to parse does
    the failure stand; the original batch error is then kept in `error` behind a
    `batch_fallback:` prefix so the batch-level cause is not lost.

    Args:
        job: Job of the failed batch.
        completion_fn: Completion function for the re-send.
        batch_error: Why the batch could not serve the job.
        transcript: Sink that records the re-send.

    Returns:
        The raw record of the re-sent job.
    """
    record = run_completion_job(job, completion_fn=completion_fn, transcript=transcript)
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
    transcript: TranscriptSink | None = None,
) -> Iterable[dict[str, Any]]:
    """Run planned jobs, batched per `completion_job_batches`.

    `planned_jobs` carries the FULL plan on a resumed run so the pending subset
    keeps the batch composition of the first attempt instead of being re-batched
    on its own; see `completion_job_batches`.

    `transcript` receives one row per provider call, including the single-item
    re-sends a failed batch falls back to; None turns transcripts off.

    Args:
        jobs: Jobs to run.
        max_workers: Upper bound on batches run concurrently; must be positive.
        completion_fn: Completion function passed to every batch.
        batch_size: Jobs per provider request.
        batch_order: `grouped` or `shuffled`; defaults to the jobs' own setting.
        seed: Seed of the shuffled order; defaults to the jobs' own seed.
        planned_jobs: Full plan of a resumed run.
        transcript: Sink for the provider calls.

    Yields:
        Raw records, batch by batch: in batch order with one worker, and in the
        order the batches finish with several workers.
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
            yield from run_completion_batch(
                batch, completion_fn=completion_fn, transcript=transcript
            )
        return

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                run_completion_batch,
                batch,
                completion_fn=completion_fn,
                transcript=transcript,
            )
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
    """Return the share of predictions that equal the gold labels.

    Args:
        y_true: Gold labels.
        y_pred: Predicted labels, aligned with `y_true`.

    Returns:
        The accuracy, or NaN when there are no labels.
    """
    if not y_true:
        return math.nan
    return float(sklearn_accuracy_score(y_true, y_pred))


def binary_f1_score(y_true: list[int], y_pred: list[int]) -> float:
    """Return the F1 score of the positive class 1.

    An undefined precision or recall counts as 0.

    Args:
        y_true: Gold labels, 0 or 1.
        y_pred: Predicted labels, 0 or 1.

    Returns:
        The F1 score, or NaN when there are no labels.
    """
    if not y_true:
        return math.nan
    return float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))


def macro_f1_score(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    """Return the unweighted mean of the per-class F1 scores.

    An undefined per-class score counts as 0.

    Args:
        y_true: Gold labels.
        y_pred: Predicted labels.
        labels: Classes to average over, including classes that never occur.

    Returns:
        The macro F1 score, or NaN when there are no labels.
    """
    if not y_true:
        return math.nan
    return float(
        f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )


def brier_score(y_true: list[int], probabilities: list[float]) -> float:
    """Return the mean squared gap between predicted probabilities and binary outcomes.

    Args:
        y_true: Binary outcomes.
        probabilities: Predicted probabilities of outcome 1.

    Returns:
        The Brier score, or NaN when there are no outcomes.
    """
    if not y_true:
        return math.nan
    return float(brier_score_loss(y_true, probabilities))


def ece_score(y_true: list[int], probabilities: list[float], bins: int = 10) -> float:
    """Return the expected calibration error over equal-width probability bins.

    Each non-empty bin contributes its share of the samples times the absolute
    gap between its outcome rate and its mean probability. The last bin includes
    1.0.

    Args:
        y_true: Binary outcomes.
        probabilities: Predicted probabilities of outcome 1, between 0 and 1.
        bins: Number of equal-width bins.

    Returns:
        The calibration error, or NaN when there are no outcomes.
    """
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
    """Return the Spearman rank correlation of two paired samples.

    Pairs with a NaN are omitted.

    Args:
        xs: First sample.
        ys: Second sample, paired with `xs`.

    Returns:
        The correlation, or NaN for fewer than two pairs, unequal lengths or an
        undefined result.
    """
    if len(xs) < 2 or len(xs) != len(ys):
        return math.nan
    statistic = spearmanr(xs, ys, nan_policy="omit").statistic
    return float(statistic) if not math.isnan(statistic) else math.nan


def pearson_corr(xs: list[float], ys: list[float]) -> float:
    """Return the Pearson correlation of two paired samples, ignoring pairs with a NaN.

    Args:
        xs: First sample.
        ys: Second sample, paired with `xs`.

    Returns:
        The correlation, or NaN for unequal lengths, fewer than two complete pairs
        or a constant sample.
    """
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
    """Return the area under the ROC curve of probabilities against binary outcomes.

    Args:
        y_true: Binary outcomes.
        probabilities: Predicted probabilities of outcome 1.

    Returns:
        The AUROC, or NaN when there are no outcomes, only one class occurs or
        scikit-learn rejects the input.
    """
    if not y_true or len(set(y_true)) < 2:
        return math.nan
    try:
        return float(roc_auc_score(y_true, probabilities))
    except ValueError:
        return math.nan


def class_order_for_task(task: str) -> list[str]:
    """Return the label set of a task in its canonical order.

    Args:
        task: `task1`, `task2` or `task3`.

    Returns:
        `yes` and `no`, the modalities of `MODALITIES` or the relations of
        `TASK3_RELATIONS`.

    Raises:
        ValueError: If the task is unknown.
    """
    if task == "task1":
        return ["yes", "no"]
    if task == "task2":
        return list(MODALITIES)
    if task == "task3":
        return list(TASK3_RELATIONS)
    raise ValueError(f"Unknown task: {task}")


def label_from_parsed(task: str, parsed: dict[str, Any]) -> str:
    """Return the canonical label of a parsed response.

    Args:
        task: `task1`, `task2` or `task3`; selects the `decision`, `modality` or
            `relation` field.
        parsed: Parsed response payload.

    Returns:
        The normalized label.

    Raises:
        ValueError: If the label is missing or unrecognized, or the task is
            unknown.
    """
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
    """Return the relative frequency of each label.

    Args:
        labels: Observed labels.
        label_order: Labels to report, in order; labels outside it still count
            towards the total.

    Returns:
        The share of each label in `label_order`, all 0.0 when `labels` is empty.
    """
    if not labels:
        return dict.fromkeys(label_order, 0.0)
    counts = Counter(labels)
    total = sum(counts.values())
    return {label: counts.get(label, 0) / total for label in label_order}


def label_distribution_from_rows(
    task: str, rows: list[dict[str, Any]]
) -> dict[str, float]:
    """Return the label distribution of a task's parsed raw rows.

    Args:
        task: Task of the rows; fixes the labels and their order.
        rows: Raw rows with a parsed `parsed_json`.

    Returns:
        The share of each label of the task, in canonical order.
    """
    labels = [label_from_parsed(task, row["parsed_json"]) for row in rows]
    return label_distribution(labels, class_order_for_task(task))


def label_distribution_json(distribution: dict[str, float]) -> str:
    """Serialize a label distribution as compact JSON with sorted keys.

    Values are rounded to 12 decimal places.

    Args:
        distribution: Share per label.

    Returns:
        The JSON text.
    """
    normalized = {
        label: round(float(value), 12) for label, value in distribution.items()
    }
    return json.dumps(
        normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )


def normalized_predictive_entropy(distribution: dict[str, float]) -> float:
    """Return the entropy of a label distribution divided by its maximum.

    The maximum is the log of the number of labels, so the result lies between 0
    (one label carries all mass) and 1 (uniform).

    Args:
        distribution: Share per label, including labels with zero share.

    Returns:
        The normalized entropy; 0.0 for fewer than two labels and NaN when no
        label has a positive share.
    """
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
    """Return one minus the share of the most frequent label.

    Args:
        distribution: Share per label.

    Returns:
        The variation ratio, or NaN for an empty distribution.
    """
    if not distribution:
        return math.nan
    return 1.0 - max(float(value) for value in distribution.values())


def semantic_response_text(task: str, parsed: Mapping[str, Any]) -> str:
    """Render the meaning-bearing fields of a parsed response as one line of text.

    This is the text the ACSE semantic embedding sees. The label comes first,
    normalized when possible and otherwise the raw stripped value, followed by the
    task's free-text fields: the reason for Task 1, the rewritten requirement for
    Task 2, the evidence phrase and reason for Task 3. Empty parts are dropped and
    all whitespace is collapsed to single spaces.

    Args:
        task: `task1`, `task2` or `task3`; any other value serializes the whole
            payload as sorted JSON.
        parsed: Parsed response payload.

    Returns:
        The single-line text, empty when no field has content.
    """
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
    """Return the semantic response text of every raw row with a parsed payload.

    Rows whose `parsed_json` is not a mapping (failed parses) are skipped, so the
    result can be shorter than `rows`.

    Args:
        task: Task of the rows; selects the fields `semantic_response_text` uses.
        rows: Raw rows of one item.

    Returns:
        One text per parsed row, in row order.
    """
    texts: list[str] = []
    for row in rows:
        parsed = row.get("parsed_json")
        if isinstance(parsed, Mapping):
            texts.append(semantic_response_text(task, parsed))
    return texts


def normalize_embedding_rows(embeddings: np.ndarray) -> np.ndarray:
    """Scale every row of an embedding matrix to unit Euclidean length.

    All-zero rows stay all zero instead of dividing by zero.

    Args:
        embeddings: One embedding per row.

    Returns:
        The row-normalized float matrix.

    Raises:
        ValueError: If the input is not two-dimensional.
    """
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

    Uses `setdefault`, so explicitly exported environment variables still
    take precedence (for example a manual `RE_UQ_ACSE_MLX_MODEL` override).

    Args:
        config: Run config; `acse_embedding_backend` and
            `acse_embedding_mlx_model` are read, and blank values export nothing.
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
    """Resolve the ACSE embedding backend and return its persisted label.

    Explicit arguments win over the `RE_UQ_ACSE_EMBEDDING_BACKEND` and
    `RE_UQ_ACSE_MLX_MODEL` environment variables, which win over the defaults
    (the character TF-IDF proxy and `ACSE_MLX_DEFAULT_MODEL`).

    Args:
        embedding_backend: `tfidf`, the TF-IDF proxy label or `mlx`; `None` or
            empty falls back to the environment.
        mlx_model_name: MLX embedding model; used only for the MLX backend.

    Returns:
        The label (`tfidf_char_wb_3_5` or `mlx:<model>`) and the MLX model name,
        which is `None` for the TF-IDF proxy.

    Raises:
        ValueError: If the backend is unknown.
    """
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

    Args:
        backend_label: Label written by `semantic_embedding_backend_label`, or
            empty for legacy runs.

    Returns:
        The backend and MLX model name to pass back to
        `semantic_embedding_matrix`; `(None, None)` for an empty label and a
        `None` model for the TF-IDF proxy.

    Raises:
        ValueError: If the label is neither the TF-IDF proxy nor `mlx:<model>`.
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

    Args:
        rows: Raw rows of the run; `semantic_embedding_backend` is read.

    Returns:
        The recorded label, or the currently configured label when no row
        records one.

    Raises:
        ValueError: If the rows record more than one backend.
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
    """Embed texts with the configured ACSE backend.

    The TF-IDF proxy fits a character n-gram vectorizer on exactly these texts;
    the MLX backend loads (and caches) the embedding model and truncates inputs to
    `RE_UQ_ACSE_MLX_MAX_LENGTH` tokens, 512 by default.

    Args:
        texts: Texts to embed, one per row of the result.
        embedding_backend: Backend override; see `semantic_embedding_backend_label`.
        mlx_model_name: MLX model override; see `semantic_embedding_backend_label`.

    Returns:
        The row-normalized embedding matrix and the resolved backend label.

    Raises:
        AssertionError: If the resolved label matches no backend.
    """
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


class AcseClusterAnalysis(NamedTuple):
    """One ACSE clustering pass: the diagnostics row and the labels behind it.

    `cluster_labels` are exactly the labels the diagnostics were computed
    from, so callers that persist per-sample cluster assignments never have to
    re-run the clustering and risk disagreeing with the score.

    Attributes:
        diagnostics: The `semantic_*` diagnostic fields of the item.
        cluster_labels: Size-ranked cluster label of each sample, in sample order.
    """

    diagnostics: dict[str, Any]
    cluster_labels: list[str]


def _acse_diagnostics_without_samples(
    semantic_embedding_backend: str, distance_threshold: float
) -> dict[str, Any]:
    """Return the diagnostics for an item with no parsable samples: every metric is NaN.

    Args:
        semantic_embedding_backend: Backend label to record.
        distance_threshold: Clustering threshold to record.

    Returns:
        The diagnostics row with zero clusters.
    """
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


def _acse_diagnostics_for_single_sample(
    semantic_embedding_backend: str, distance_threshold: float
) -> dict[str, Any]:
    """Return diagnostics for a lone sample: one cluster, zero dispersion, zero score.

    Args:
        semantic_embedding_backend: Backend label to record.
        distance_threshold: Clustering threshold to record.

    Returns:
        The diagnostics row of a single cluster.
    """
    return {
        "semantic_embedding_backend": semantic_embedding_backend,
        "semantic_distance_threshold": float(distance_threshold),
        "semantic_cluster_count": 1,
        "semantic_cluster_distribution": label_distribution_json({"cluster_0": 1.0}),
        "semantic_cluster_entropy": 0.0,
        "semantic_cluster_variation_ratio": 0.0,
        "semantic_dominant_cluster_share": 1.0,
        "semantic_mean_pairwise_distance": 0.0,
        "semantic_dominant_cluster_mean_distance": 0.0,
        "semantic_uncertainty_score": 0.0,
    }


def acse_semantic_cluster_analysis(
    embeddings: np.ndarray,
    semantic_embedding_backend: str,
    distance_threshold: float = ACSE_PROXY_DISTANCE_THRESHOLD,
    dispersion_weight: float = ACSE_PROXY_INTERNAL_DISPERSION_WEIGHT,
) -> AcseClusterAnalysis:
    """Cluster one item's sample embeddings and report the ACSE diagnostics.

    This is the single place ACSE clustering happens. The rows are normalized
    once, the cosine distance matrix is built once, and `fit_predict` runs
    once. Labels are renamed by descending cluster size, ties broken by the raw
    scikit-learn label, so `cluster_0` is always the dominant cluster.

    `dispersion_weight` is the weight of the dispersion component in the
    score (entropy weight `1 - dispersion_weight`). It defaults to the
    reported setting and exists so an offline sensitivity analysis can vary
    the mixture without touching the module constant.

    Args:
        embeddings: One embedding per sample of the item.
        semantic_embedding_backend: Backend label recorded in the diagnostics.
        distance_threshold: Cosine-distance threshold of the average-linkage
            clustering; it also scales the dispersion component.
        dispersion_weight: Weight of the dispersion component, in [0, 1].

    Returns:
        The diagnostics and the per-sample cluster labels; no samples give NaN
        metrics and one sample gives a single cluster with score 0.

    Raises:
        ValueError: If `dispersion_weight` is outside [0, 1].
    """
    if not 0.0 <= float(dispersion_weight) <= 1.0:
        raise ValueError(
            f"dispersion_weight must be in [0, 1], got {dispersion_weight!r}."
        )
    matrix = normalize_embedding_rows(embeddings)
    sample_count = int(matrix.shape[0])
    if sample_count == 0:
        return AcseClusterAnalysis(
            _acse_diagnostics_without_samples(
                semantic_embedding_backend, distance_threshold
            ),
            [],
        )
    if sample_count == 1:
        return AcseClusterAnalysis(
            _acse_diagnostics_for_single_sample(
                semantic_embedding_backend, distance_threshold
            ),
            ["cluster_0"],
        )

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
    raw_counts = Counter(raw_labels)
    remap = {
        label: f"cluster_{rank}"
        for rank, label in enumerate(
            sorted(raw_counts, key=lambda label: (-raw_counts[label], label))
        )
    }
    cluster_labels = [remap[label] for label in raw_labels]

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
    entropy_weight = 1.0 - float(dispersion_weight)
    uncertainty_score = min(
        1.0,
        entropy_weight * float(cluster_entropy)
        + float(dispersion_weight) * dispersion_component,
    )
    if abs(uncertainty_score) < 1e-12:
        uncertainty_score = 0.0
    diagnostics = {
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
    return AcseClusterAnalysis(diagnostics, cluster_labels)


def acse_cluster_labels_for_embeddings(
    embeddings: np.ndarray,
    distance_threshold: float = ACSE_PROXY_DISTANCE_THRESHOLD,
) -> list[str]:
    """Return the size-ranked cluster labels only; see `acse_semantic_cluster_analysis`.

    Args:
        embeddings: One embedding per sample of the item.
        distance_threshold: Cosine-distance threshold of the clustering.

    Returns:
        The cluster label of each sample, `cluster_0` being the largest cluster.
    """
    return acse_semantic_cluster_analysis(
        embeddings, "", distance_threshold
    ).cluster_labels


def acse_semantic_diagnostics_from_embeddings(
    embeddings: np.ndarray,
    semantic_embedding_backend: str,
    distance_threshold: float = ACSE_PROXY_DISTANCE_THRESHOLD,
    dispersion_weight: float = ACSE_PROXY_INTERNAL_DISPERSION_WEIGHT,
) -> dict[str, Any]:
    """Return the diagnostics only; see `acse_semantic_cluster_analysis`.

    Args:
        embeddings: One embedding per sample of the item.
        semantic_embedding_backend: Backend label recorded in the diagnostics.
        distance_threshold: Cosine-distance threshold of the clustering.
        dispersion_weight: Weight of the dispersion component, in [0, 1].

    Returns:
        The `semantic_*` diagnostic fields of the item.
    """
    return acse_semantic_cluster_analysis(
        embeddings,
        semantic_embedding_backend,
        distance_threshold,
        dispersion_weight=dispersion_weight,
    ).diagnostics


def acse_semantic_proxy_diagnostics(
    texts: list[str],
    distance_threshold: float = ACSE_PROXY_DISTANCE_THRESHOLD,
    embedding_backend: str | None = None,
    mlx_model_name: str | None = None,
) -> dict[str, Any]:
    """Embed one item's sample texts and return their ACSE diagnostics.

    Whitespace is collapsed before embedding. With zero or one text the
    diagnostics are returned without loading the embedding backend.

    Args:
        texts: Semantic response text of each sample.
        distance_threshold: Cosine-distance threshold of the clustering.
        embedding_backend: Backend override; see `semantic_embedding_backend_label`.
        mlx_model_name: MLX model override; see `semantic_embedding_backend_label`.

    Returns:
        The `semantic_*` diagnostic fields of the item, computed with the default
        dispersion weight.
    """
    cleaned = [re.sub(r"\s+", " ", str(text or "").strip()) for text in texts]
    backend_label, _ = semantic_embedding_backend_label(
        embedding_backend, mlx_model_name
    )
    sample_count = len(cleaned)
    # Short-circuit before paying for the embedding backend: neither degenerate
    # case can produce a different answer once the texts are embedded.
    if sample_count == 0:
        return _acse_diagnostics_without_samples(backend_label, distance_threshold)
    if sample_count == 1:
        return _acse_diagnostics_for_single_sample(backend_label, distance_threshold)

    embeddings, backend_label = semantic_embedding_matrix(
        cleaned,
        embedding_backend=embedding_backend,
        mlx_model_name=mlx_model_name,
    )
    return acse_semantic_diagnostics_from_embeddings(
        embeddings, backend_label, distance_threshold
    )


def majority_label(distribution: dict[str, float], label_order: list[str]) -> str:
    """Return the label with the highest share, breaking ties toward the weakest label.

    Ties go to the label that comes last in `label_order`, so a tie can never
    inflate over-commitment. Labels outside `label_order` lose every tie.

    Args:
        distribution: Share per label.
        label_order: Labels from strongest to weakest.

    Returns:
        The majority label.

    Raises:
        ValueError: If the distribution is empty.
    """
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
    """Return the distribution that puts all mass on one label.

    Args:
        label: Label that receives share 1.0.
        label_order: Labels to report; a label outside it gives all zeros.

    Returns:
        The share of each label in `label_order`.
    """
    return {candidate: 1.0 if candidate == label else 0.0 for candidate in label_order}


def task3_score_fields(
    item: dict[str, Any], pred_relation: str, evidence_phrase: Any = ""
) -> dict[str, Any]:
    """Return the Task 3 columns of a score row.

    Copies the gold relation, the Task 2 output under audit and the audit mode from
    the verification item, and checks whether the evidence phrase occurs in the
    source statement.

    Args:
        item: Task 3 verification item.
        pred_relation: Predicted relation.
        evidence_phrase: Evidence phrase the model quoted, if any.

    Returns:
        The Task 3 fields, keyed by score-row column.
    """
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
        "task3_reference_status": item.get("task3_reference_status", "available"),
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
    """Measure how often a score rises as the source modality weakens.

    Each seed's variants are ordered from strongest (`mandatory`) to weakest
    modality, and the largest step increase of the score along that order is
    taken; seeds without all four variants are skipped. A seed violates
    monotonicity strictly when that increase is positive and tolerantly when it
    exceeds `tolerance`.

    Args:
        rows: Score rows with `seed_id`, `source_modality` and `score_field`.
        score_field: Score that should not increase as the modality weakens;
            blank cells are dropped.
        tolerance: Increase allowed before a seed counts as a tolerant violation.

    Returns:
        The share of checked seeds with a tolerant and with a strict violation,
        the tolerance, and the mean and maximum of the per-seed largest increase;
        all NaN except the tolerance when no seed can be checked.
    """
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


def raw_batch_id(raw: Any) -> str:
    """Return the provider request a raw completion record was sent in.

    Blank for legacy rows written before requests were batched, for
    single-item requests, and for synthesised (fake-completion) rows.

    Args:
        raw: Raw completion record; anything but a mapping gives a blank id.

    Returns:
        The `batch_id`, or an empty string.
    """
    return str(raw.get("batch_id", "") or "") if isinstance(raw, Mapping) else ""


def sample_batch_ids_field(sample_rows: Iterable[Any]) -> str:
    """Return the semicolon-joined request ids behind one collapsed score row.

    A stochastic group collapses its repeated samples into a single score row,
    and each sample index was its own request, so the row is decided by several
    requests. First-seen order is kept so the value is stable across runs;
    blank ids are dropped, and the result is blank when nothing was batched.

    Args:
        sample_rows: Raw rows of the group's samples.

    Returns:
        The distinct request ids joined by `;`.
    """
    seen: dict[str, None] = {}
    for row in sample_rows:
        batch_id = raw_batch_id(row)
        if batch_id:
            seen.setdefault(batch_id, None)
    return ";".join(seen)


def answer_length_fields(
    raw: Mapping[str, Any], item: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the answer-length / bloat fields for one raw completion record.

    `requirement_word_count` is read from the raw record when available;
    legacy rows fall back to counting words in `parsed_json.requirement`.
    `completion_tokens` is populated only for single-item requests because a
    provider's batched usage is not attributable to an individual answer.
    `source_word_count` always comes from the benchmark `source_statement`.

    Args:
        raw: Raw completion record.
        item: Benchmark item the record answers.

    Returns:
        `requirement_word_count`, `source_word_count`, `length_ratio` (requirement
        words over source words) and `completion_tokens`; a value that cannot be
        determined is an empty string.
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
    batch_id = raw_batch_id(raw)
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
    """Return the answer-length fields with every value blank.

    Returns:
        The keys of `answer_length_fields`, each mapped to an empty string.
    """
    return {
        "requirement_word_count": "",
        "source_word_count": "",
        "length_ratio": "",
        "completion_tokens": "",
    }


def resolve_bootstrap_cluster_field(
    rows: list[dict[str, Any]],
    cluster_field: str = DEFAULT_BOOTSTRAP_CLUSTER_FIELD,
    fallback_field: str = BOOTSTRAP_CLUSTER_FALLBACK_FIELD,
    item_field: str | None = BOOTSTRAP_ITEM_CLUSTER_FIELD,
) -> str:
    """Return the field these rows can actually be clustered on.

    `cluster_field` is used only when *every* row carries a non-blank value
    for it. Requiring the column to be complete matters because a partially
    populated `batch_id` (legacy runs, synthesised completions) would collapse
    every unbatched row into one meaningless cluster keyed on the empty string.
    Rows without a request then cluster on `item_field` when every row
    carries one (the single-item protocol: one request per item), and on
    `fallback_field` otherwise. Pass `item_field=None` where the cluster
    must nest a coarser pairing unit, as the paired ablation deltas do. The
    resolved field is reported back to the caller — the CI dictionaries expose
    it as `bootstrap_ci_cluster_field` — so a slice that reverted to a finer
    or coarser unit is visible in the output rather than mislabelled.

    Args:
        rows: Rows to cluster.
        cluster_field: Preferred cluster field, the provider request by default;
            returned as is when it equals `fallback_field`.
        fallback_field: Field used when neither of the others is complete, the
            seed by default.
        item_field: Intermediate field tried before the fallback, the item by
            default; `None` skips it.

    Returns:
        The name of the field to cluster on.
    """
    if cluster_field == fallback_field:
        return cluster_field
    if rows and all(str(row.get(cluster_field, "") or "") for row in rows):
        return cluster_field
    if (
        item_field
        and item_field != fallback_field
        and rows
        and all(str(row.get(item_field, "") or "") for row in rows)
    ):
        logger.debug(
            "bootstrap cluster field %r is incomplete over %d rows; clustering on %r",
            cluster_field,
            len(rows),
            item_field,
        )
        return item_field
    logger.debug(
        "bootstrap cluster field %r is incomplete over %d rows; clustering on %r",
        cluster_field,
        len(rows),
        fallback_field,
    )
    return fallback_field


def bootstrap_seed_metric(
    rows: list[dict[str, Any]],
    metric: Callable[[list[dict[str, Any]]], float],
    cluster_field: str = DEFAULT_BOOTSTRAP_CLUSTER_FIELD,
    iterations: int = 1000,
    seed: int = 20260518,
    seed_field: str | None = None,
) -> tuple[float, float, float]:
    """Run a cluster bootstrap of `metric` over `rows`.

    Clusters default to the provider request (`batch_id`): batched runs sent
    16 items per request — four whole seeds x four source conditions — and
    model behaviour is decided per request, so the request nests the seed and
    is the conservative unit. Rows without a complete request column cluster on
    the item (single-item requests) or, failing that, the seed; see
    `resolve_bootstrap_cluster_field`.

    `seed_field` is the deprecated spelling of `cluster_field`. It names the
    field outright with no fallback, preserving the pre-request behaviour for
    callers that still pass it.

    Args:
        rows: Rows the metric is computed over.
        metric: Function from a list of rows to a value; NaN replicates are
            dropped before taking the percentiles.
        cluster_field: Preferred cluster field, resolved as described above.
        iterations: Number of bootstrap replicates.
        seed: Seed of the resampling generator.
        seed_field: Deprecated; overrides `cluster_field` without fallback.

    Returns:
        The point estimate on all rows and the 2.5 and 97.5 percentiles of the
        replicates. All three are NaN when there are no rows or the cluster field
        is absent, and the interval is NaN when every replicate is NaN.
    """
    if seed_field is not None:
        resolved_field = seed_field
    elif rows:
        resolved_field = resolve_bootstrap_cluster_field(rows, cluster_field)
    else:
        resolved_field = cluster_field
    if not rows:
        return math.nan, math.nan, math.nan
    frame = pd.DataFrame.from_records(rows)
    if resolved_field not in frame.columns:
        return math.nan, math.nan, math.nan
    by_cluster = {
        str(cluster_value): group.to_dict(orient="records")
        for cluster_value, group in frame.groupby(resolved_field, sort=False)
    }
    cluster_keys = np.asarray(list(by_cluster.keys()), dtype=object)
    if cluster_keys.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    point = metric(rows)
    samples = []
    for _ in range(iterations):
        sample_rows: list[dict[str, Any]] = []
        for cluster_key in rng.choice(
            cluster_keys, size=cluster_keys.size, replace=True
        ):
            sample_rows.extend(by_cluster[str(cluster_key)])
        samples.append(metric(sample_rows))
    sample_array = np.asarray([x for x in samples if not math.isnan(x)], dtype=float)
    if sample_array.size == 0:
        return point, math.nan, math.nan
    low, high = np.quantile(sample_array, [0.025, 0.975])
    return float(point), float(low), float(high)


class PairedDeltaCI(NamedTuple):
    """Complete-pair paired bootstrap result for `metric(b) - metric(a)`.

    `n_complete_pairs` is the number of pairing keys (seeds) observed in both
    arms — the population the point estimate and the interval describe — and
    `n_excluded_single_arm` the number dropped because only one arm ever
    answered them. `cluster_field` is the field the replicates were drawn on
    and `n_clusters` how many such clusters the cohort formed; the two differ
    from the pairing count whenever a cluster holds several seeds.

    Attributes:
        delta: Point estimate of the difference on the complete-pair cohort.
        ci_low: 2.5 percentile of the replicate differences.
        ci_high: 97.5 percentile of the replicate differences.
        n_complete_pairs: Pairing keys observed in both arms.
        n_excluded_single_arm: Pairing keys observed in only one arm.
        cluster_field: Field the replicates were drawn on.
        n_clusters: Number of clusters the cohort formed.
    """

    delta: float
    ci_low: float
    ci_high: float
    n_complete_pairs: int
    n_excluded_single_arm: int
    cluster_field: str = BOOTSTRAP_CLUSTER_FALLBACK_FIELD
    n_clusters: int = 0


def bootstrap_seed_metric_delta(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    metric: Callable[[list[dict[str, Any]]], float],
    cluster_field: str = DEFAULT_BOOTSTRAP_CLUSTER_FIELD,
    iterations: int = 1000,
    seed: int = 20260518,
    pair_field: str = BOOTSTRAP_CLUSTER_FALLBACK_FIELD,
    seed_field: str | None = None,
) -> PairedDeltaCI:
    """Bootstrap `metric(rows_b) - metric(rows_a)` over complete pairs, by cluster.

    The two arms score the same benchmark items under different conditions (for
    example `item_context` bare vs document), so their rows are **paired by
    seed** (`pair_field`). Each iteration draws one resample and evaluates the
    metric on *both* arms restricted to it before differencing; resampling the
    arms independently would treat paired observations as unrelated and
    overstate the interval.

    Pairing and resampling are separate units. The **resampling** unit defaults
    to the provider request, which holds four whole seeds, so a drawn request
    contributes all of its paired seeds. The two arms are different runs and
    their request ids therefore differ, but the partition of seeds into requests
    is a property of the batching and is the same in both; each paired seed is
    assigned to the first request observed for it in `rows_a`. Rows without a
    complete request column fall back to clustering on `pair_field`, which
    reproduces the earlier seed-clustered interval exactly.

    Only a seed observed in both arms can form a pair, so the complete-pair
    cohort is built once, before resampling. Single-arm seeds are counted and
    excluded up front instead of entering the draw and contributing an
    undefined difference to some replicates but not others, which would let the
    effective number of pairs vary by replicate. Both the point estimate and
    the 2.5/97.5 percentile interval describe that cohort; the delta is NaN
    when no complete pair exists.

    `seed_field` is the deprecated spelling that pairs *and* resamples on one
    field, preserving the pre-request behaviour for callers that still pass it.

    Args:
        rows_a: Rows of the reference arm.
        rows_b: Rows of the compared arm.
        metric: Function from a list of rows to a value.
        cluster_field: Preferred resampling field, the provider request by
            default.
        iterations: Number of bootstrap replicates.
        seed: Seed of the resampling generator.
        pair_field: Field that pairs rows across the arms, the seed by default.
        seed_field: Deprecated; overrides both `cluster_field` and `pair_field`.

    Returns:
        The delta, its interval and the cohort counts; the interval is NaN when
        every replicate is NaN.
    """
    if seed_field is not None:
        cluster_field = pair_field = seed_field

    def by_pair(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(str(row.get(pair_field, "")), []).append(row)
        return groups

    groups_a, groups_b = by_pair(rows_a), by_pair(rows_b)
    paired = sorted(set(groups_a) & set(groups_b))
    excluded = len(set(groups_a) ^ set(groups_b))
    if not paired:
        return PairedDeltaCI(math.nan, math.nan, math.nan, 0, excluded, pair_field, 0)

    cohort_a = [row for key in paired for row in groups_a[key]]
    cohort_b = [row for key in paired for row in groups_b[key]]
    point = metric(cohort_b) - metric(cohort_a)
    # The cluster must nest the pairing unit, so the item is never a fallback
    # here: rows without a request cluster on the pair itself.
    resolved_field = resolve_bootstrap_cluster_field(
        cohort_a + cohort_b, cluster_field, pair_field, item_field=None
    )
    clusters: dict[str, list[str]] = {}
    for pair_key in paired:
        cluster_key = (
            pair_key
            if resolved_field == pair_field
            else next(
                (
                    value
                    for value in (
                        str(row.get(resolved_field, "") or "")
                        for row in groups_a[pair_key]
                    )
                    if value
                ),
                pair_key,
            )
        )
        clusters.setdefault(cluster_key, []).append(pair_key)
    cluster_keys = np.asarray(list(clusters), dtype=object)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(iterations):
        sample_a: list[dict[str, Any]] = []
        sample_b: list[dict[str, Any]] = []
        for cluster_key in rng.choice(
            cluster_keys, size=cluster_keys.size, replace=True
        ):
            for pair_key in clusters[str(cluster_key)]:
                sample_a.extend(groups_a[pair_key])
                sample_b.extend(groups_b[pair_key])
        samples.append(metric(sample_b) - metric(sample_a))
    # Every replicate now draws both arms of a complete pair, so a non-finite
    # replicate can only mean the metric itself is undefined on that resample,
    # never that an arm was missing.
    sample_array = np.asarray([x for x in samples if not math.isnan(x)], dtype=float)
    counts = (len(paired), excluded, resolved_field, len(clusters))
    if sample_array.size == 0:
        return PairedDeltaCI(float(point), math.nan, math.nan, *counts)
    low, high = np.quantile(sample_array, [0.025, 0.975])
    return PairedDeltaCI(float(point), float(low), float(high), *counts)


def exact_item_metric_pairs(rows_a, rows_b, eligible):
    """Match source identities, never arm-local item IDs; reject ambiguous keys.

    The matched, unmatched, duplicate and ineligible counts are match keys (item
    identities), and `n_matched_capabilities` counts distinct keys without the
    modality. `n_missing_identity_rows` and every per-arm count are rows: the
    per-arm failed, unclassified and eligible counts run over all rows of the arm
    and equal item counts only once retries are deduplicated, and the per-arm
    duplicate count is the rows behind duplicated keys. A key that occurs more
    than once in either arm is excluded from both, not arbitrarily resolved.
    Retries should already have been deduplicated by completion key.

    The match key is model, dataset, benchmark variant, `source_identity` and gold
    modality (falling back to `source_modality`); rows missing the identity or
    the modality are counted and left out.

    Args:
        rows_a: Score rows of the first arm, reported as `bare` in the counts.
        rows_b: Score rows of the second arm, reported as `document` in the counts.
        eligible: Predicate on a row; a key is paired only when its row is
            eligible in both arms.

    Returns:
        The matched rows of each arm, in the same key order and tagged with a
        shared `exact_pair_id`, and a dict of matching counts (matched,
        unmatched, duplicate, ineligible and missing-identity counts, plus
        per-arm duplicate, failed, unclassified and eligible counts).
    """

    def index_by_identity(rows):
        result = {}
        missing = 0
        for row in rows:
            identity = str(row.get("source_identity", ""))
            modality = str(row.get("gold_modality", row.get("source_modality", "")))
            if not identity or not modality:
                missing += 1
                continue
            key = (
                str(row.get("model", "")),
                str(row.get("dataset_id", "")),
                str(row.get("benchmark_variant", "")),
                identity,
                modality,
            )
            result.setdefault(key, []).append(row)
        return result, missing

    index_a, missing_a = index_by_identity(rows_a)
    index_b, missing_b = index_by_identity(rows_b)
    duplicates = {
        key
        for key in index_a.keys() | index_b.keys()
        if len(index_a.get(key, [])) > 1 or len(index_b.get(key, [])) > 1
    }
    common = (index_a.keys() & index_b.keys()) - duplicates
    paired = sorted(
        key for key in common if eligible(index_a[key][0]) and eligible(index_b[key][0])
    )
    cohort_a, cohort_b = [], []
    for key in paired:
        pair_id = json.dumps(key)
        cohort_a.append({**index_a[key][0], "exact_pair_id": pair_id})
        cohort_b.append({**index_b[key][0], "exact_pair_id": pair_id})
    counts = {
        "n_matched_items": len(paired),
        "n_matched_capabilities": len({key[:4] for key in paired}),
        "n_unmatched_items": len((index_a.keys() ^ index_b.keys()) - duplicates),
        "n_duplicate_identities": len(duplicates),
        "n_excluded_ineligible_items": len(common) - len(paired),
        "n_missing_identity_rows": missing_a + missing_b,
    }
    for arm, index, rows in (("bare", index_a, rows_a), ("document", index_b, rows_b)):
        counts[f"n_{arm}_duplicate_rows"] = sum(
            len(v) for v in index.values() if len(v) > 1
        )
        counts[f"n_{arm}_failed_items"] = sum(
            r.get("response_status", "ok") != "ok" for r in rows
        )
        counts[f"n_{arm}_unclassified_items"] = sum(
            r.get("response_status", "ok") == "ok"
            and r.get("text_modality_parse_status") != "ok"
            for r in rows
        )
        counts[f"n_{arm}_eligible_items"] = sum(eligible(r) for r in rows)
    return cohort_a, cohort_b, counts


def paired_request_clusters(rows_a, rows_b):
    """Union request partitions from both arms, retaining whole exact pairs.

    Crossing request partitions require connected components, not choosing one
    arm's partition. Missing request metadata falls back to capability clusters.

    Args:
        rows_a: Matched rows of the first arm, as returned by
            `exact_item_metric_pairs`.
        rows_b: Matched rows of the second arm; `rows_b[i]` pairs with
            `rows_a[i]`.

    Returns:
        The two row lists (tagged with `paired_request_cluster` when request
        clustering applies), the cluster field to bootstrap on and a short
        description of the clustering used.
    """
    if not rows_a:
        return rows_a, rows_b, BOOTSTRAP_CLUSTER_FALLBACK_FIELD, "empty matched cohort"
    # Same completeness rule as every other cluster bootstrap: the request is
    # the unit only when both arms carry it on every row.
    if any(
        resolve_bootstrap_cluster_field(rows) != DEFAULT_BOOTSTRAP_CLUSTER_FIELD
        for rows in (rows_a, rows_b)
    ):
        return (
            rows_a,
            rows_b,
            BOOTSTRAP_CLUSTER_FALLBACK_FIELD,
            "missing request IDs; capability fallback",
        )
    parent = list(range(len(rows_a)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for rows in (rows_a, rows_b):
        seen = {}
        for i, row in enumerate(rows):
            key = (row.get("model", ""), row.get("run_id", ""), row["batch_id"])
            if key in seen:
                parent[root(i)] = root(seen[key])
            else:
                seen[key] = i
    clustered_a = [
        {**r, "paired_request_cluster": str(root(i))} for i, r in enumerate(rows_a)
    ]
    clustered_b = [
        {**r, "paired_request_cluster": str(root(i))} for i, r in enumerate(rows_b)
    ]
    return (
        clustered_a,
        clustered_b,
        "paired_request_cluster",
        "connected components of requests in both arms",
    )


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
    """Build one score row from a label distribution.

    The prediction is the majority label of the distribution (ties toward the
    weakest label) and the confidence is its largest share. Task 1 rows score the
    yes/no decision against `task1_gold_yes` and carry `p_yes`; Task 2 and Task 3
    rows score whether the predicted modality or relation matches the gold label.
    Text-modality fields are left blank.

    Args:
        raw: Raw row that supplies the identity and provenance fields.
        item: Benchmark item (or Task 3 verification item) of the row.
        uq_method: Name of the UQ method recorded in `uq_method`.
        distribution: Share per label of the task.
        valid_n: Samples that parsed.
        total_n: Samples the plan asked for.
        uncertainty_measure: Name of the uncertainty measure recorded with the score.
        uncertainty_score: Value of that measure.
        model_name: Replaces the raw row's `model` when given (ensemble rows).

    Returns:
        The score row.

    Raises:
        ValueError: If the raw row's task is unknown.
    """
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
    """Build the sampling-based score rows of one stochastic group.

    Every group yields a consistency row and a variation-ratio row (both scored by
    the variation ratio) and a predictive-entropy row. With `sample_rows`, an ACSE
    semantic-dispersion row is added, embedded with the backend recorded on `raw`,
    and every row records all requests the group's samples were sent in.

    Args:
        raw: Representative raw row of the group.
        item: Benchmark item of the group.
        distribution: Label distribution over the parsed samples.
        valid_n: Samples that parsed.
        total_n: Samples the plan asked for.
        consistency_method: Name of the consistency row's method
            (`label_self_consistency` or `modality_consistency`).
        model_name: Replaces the raw row's `model` when given.
        sample_rows: Parsed raw rows of the group; `None` skips the ACSE row.

    Returns:
        Three score rows, or four with the ACSE row.
    """
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
        # ``raw`` is the group's representative sample, so its ``batch_id``
        # names one request only. Record every request the group was decided
        # in, so a collapsed row can still be traced back to all of them.
        contributing = sample_batch_ids_field(sample_rows)
        for row in rows:
            row["sample_batch_ids"] = contributing
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

    Only the first parsed deterministic answer of each member counts, and a group
    is scored only when at least two members answered.

    Args:
        benchmark_rows: Current benchmark items; raw rows of other items or
            earlier item texts are dropped.
        raw_rows: Raw rows of every member.
        group_columns: Three columns that identify an ensemble group, task and
            item, in that order.
        required_columns: Columns the raw rows must carry; otherwise nothing is
            scored.
        member_key: Function that names the ensemble member of a raw row.
        uq_method: Method name recorded on the score rows.
        model_name_builder: Function from the group id and the number of
            answering members to the `model` recorded on the score row.
        run_group_id: Restrict to this run group when given.

    Returns:
        One variation-ratio score row per scored group and item.
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
    """Score label disagreement across the models of one run, item by item.

    Args:
        benchmark_rows: Current benchmark items.
        raw_rows: Raw rows of the run's models.

    Returns:
        Score rows with method `model_ensemble_disagreement`, one per run, task
        and item answered by at least two models.
    """
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
    """Score label disagreement across the runs of a run group, item by item.

    Members are keyed by provider, model and run id, so the same model run twice
    counts as two members.

    Args:
        benchmark_rows: Current benchmark items.
        raw_rows: Raw rows of the run group's runs.
        run_group_id: Restrict to this run group when given.

    Returns:
        Score rows with method `model_ensemble_disagreement_run_group`, one per
        run group, task and item answered by at least two members.
    """
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
    *,
    sampling_plan: SamplingPlan | None = None,
    infer_plan_from_observations: bool = False,
) -> list[dict[str, Any]]:
    """Build per-item UQ score rows under a declared `SamplingPlan`.

    `min_valid_samples` drops stochastic groups with fewer than that many
    successfully parsed samples. Every stochastic row carries
    `stochastic_complete` (`valid_n == total_n`); agreement and unanimity
    summaries must be restricted to complete rows, see
    `repeated_sample_agreement_metrics`.

    `sampling_plan` is required: `total_n` is
    `plan.total_samples(len(group))`, so a sample the run never wrote (crash,
    budget stop) counts as a parse failure and the group reads as incomplete.
    Inferring the denominator from the rows that happen to exist would report
    three observed samples of a planned five as a complete 3/3; that behaviour
    survives only behind `infer_plan_from_observations=True`, which stamps
    every row it produces with `sampling_plan_source="inferred"`.

    Duplicate rows for one `run_id` + `completion_record_key` (append-only
    raw files, resumed runs) are collapsed by `dedupe_raw_rows` first, so a
    re-requested item cannot be double-counted or scored from its stale attempt.

    Args:
        benchmark_rows: Current benchmark items.
        raw_rows: Raw Task 1 and Task 2 rows of one or more runs.
        min_valid_samples: Minimum parsed samples a stochastic group needs; values
            below 1 act as 1.
        sampling_plan: Plan the runs were executed under.
        infer_plan_from_observations: Explicit opt-in to the exploratory inferred
            plan when no `sampling_plan` is given.

    Returns:
        Verbalized-confidence rows for the deterministic answers, the
        sampling-based rows of every stochastic group and the cross-model ensemble
        rows, each stamped with `sampling_plan_source`.
    """
    plan = resolve_sampling_plan(
        sampling_plan, infer_plan_from_observations, caller="build_uq_scores"
    )
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
        return _stamp_sampling_plan_source(scores, plan)
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
        total_n = plan.total_samples(len(group))
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
    return _stamp_sampling_plan_source(scores, plan)


def _stamp_sampling_plan_source(
    scores: list[dict[str, Any]], plan: SamplingPlan
) -> list[dict[str, Any]]:
    """Record on every score row which plan its denominator came from.

    Args:
        scores: Score rows, updated in place.
        plan: Plan whose `source` is recorded.

    Returns:
        The same list.
    """
    for row in scores:
        row["sampling_plan_source"] = plan.source
    return scores


def build_task3_scores(
    task3_items: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    min_valid_samples: int = 1,
    *,
    sampling_plan: SamplingPlan | None = None,
) -> list[dict[str, Any]]:
    """Score only audits with a reference; retain all others in audit review rows.

    Production callers supply the recorded sampling plan. The optional plan
    retains compatibility with older notebook calls that used observed counts.

    Deterministic audits become verbalized-confidence rows, and stochastic groups
    become the sampling-based rows of `distribution_score_rows` with a
    `relation_consistency` row. Audits without a parsable relation, and items
    whose gold relation is not one of `TASK3_RELATIONS`, are skipped.

    Args:
        task3_items: Task 3 verification items.
        raw_rows: Raw rows; only Task 3 rows are used.
        min_valid_samples: Minimum parsed samples a stochastic group needs; values
            below 1 act as 1.
        sampling_plan: Plan the audit run was executed under; without it the
            denominator is the number of rows observed for the group.

    Returns:
        The Task 3 score rows.
    """
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
                sampling_plan.total_samples(len(group))
                if sampling_plan
                else len(group),
                "relation_consistency",
                sample_rows=valid,
            )
        )

    return scores


def task3_audit_review_rows(
    task3_items: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    *,
    expected_stochastic_samples: int = 5,
) -> list[dict[str, Any]]:
    """Build one row per auditable source, including missing and unscored judgments.

    No correctness label is inferred from the auditor's answer. Reference-free
    outputs remain reviewable, while the scored subset stays explicit.

    Args:
        task3_items: Task 3 verification items, one per audited Task 2 output.
        raw_rows: Raw Task 3 rows; duplicates are collapsed first.
        expected_stochastic_samples: Planned stochastic samples per item; only
            parsed samples with an index below it count towards completeness.

    Returns:
        One review row per item, with the deterministic audit's answer (blank when
        missing), the reference status and the stochastic sample count.

    Raises:
        ValueError: If the rows of one item and model come from more than one run.
    """
    by_item: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for raw in dedupe_raw_rows(raw_rows):
        if raw.get("task") == "task3":
            by_item[(str(raw.get("model", "")), str(raw.get("item_id", "")))].append(
                raw
            )
    result: list[dict[str, Any]] = []
    for item in task3_items:
        rows = by_item.get((str(item["task2_model"]), str(item["item_id"])), [])
        run_ids = {str(row.get("run_id", "")) for row in rows}
        if len(run_ids) > 1:
            raise ValueError(
                "Audit review requires one selected run per source item/model."
            )
        deterministic = next(
            (row for row in rows if row.get("sample_kind") == "deterministic"), {}
        )
        parsed = deterministic.get("parsed_json") or {}
        available = item.get("task3_gold_relation") in TASK3_RELATIONS
        valid_stochastic = {
            int(row["sample_index"])
            for row in rows
            if row.get("sample_kind") == "stochastic"
            and row.get("parse_status") == "ok"
            and 0 <= int(row["sample_index"]) < expected_stochastic_samples
        }
        result.append(
            {
                "model": item["task2_model"],
                "run_id": next(iter(run_ids), ""),
                "source_run_id": item["task2_run_id"],
                "item_id": item["item_id"],
                "source_item_id": item["source_item_id"],
                "seed_id": item["seed_id"],
                "source_modality": item["source_modality"],
                "source_statement": item["source_statement"],
                "task2_requirement": item["task2_requirement"],
                "task2_text_modality": item.get("task2_text_modality", ""),
                "reference_available": available,
                "reference_status": item.get("task3_reference_status")
                or ("available" if available else "unavailable"),
                "gold_relation": item.get("task3_gold_relation", ""),
                "audit_observed": bool(deterministic),
                "audit_parse_status": deterministic.get("parse_status", "missing"),
                "pred_relation": parsed.get("relation", ""),
                "confidence": parsed.get("confidence", ""),
                "evidence_phrase": parsed.get("evidence_phrase", ""),
                "brief_reason": parsed.get("brief_reason", ""),
                "valid_stochastic_samples": len(valid_stochastic),
                "expected_stochastic_samples": expected_stochastic_samples,
                "stochastic_complete": len(valid_stochastic)
                == expected_stochastic_samples,
            }
        )
    return result


def task3_audit_coverage_rows(
    review_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Count, per model, how many audits were observed, parsed and scorable.

    Args:
        review_rows: Rows from `task3_audit_review_rows`.

    Returns:
        One row per model, sorted by model, with counts of auditable items,
        items with and without a reference, observed, parsed, parsed-but-unscored
        and missing deterministic audits, and complete stochastic groups.
    """
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in review_rows:
        by_model[str(row["model"])].append(row)
    return [
        {
            "model": model,
            "auditable_source_items": len(rows),
            "reference_available_items": sum(
                row["reference_available"] for row in rows
            ),
            "reference_unavailable_items": sum(
                not row["reference_available"] for row in rows
            ),
            "observed_deterministic_audits": sum(row["audit_observed"] for row in rows),
            "parsed_deterministic_audits": sum(
                row["audit_parse_status"] == "ok" for row in rows
            ),
            "parsed_unscored_audits": sum(
                not row["reference_available"] and row["audit_parse_status"] == "ok"
                for row in rows
            ),
            "missing_deterministic_audits": sum(
                not row["audit_observed"] for row in rows
            ),
            "complete_stochastic_groups": sum(
                row["stochastic_complete"] for row in rows
            ),
        }
        for model, rows in sorted(by_model.items())
    ]


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
    "model_output_error",
    "missing_batch_result",
)

#: Value an :class:`ObservationIdentity` field takes when the row it was built
#: from does not carry that field at all.
#:
#: Raw JSONL rows written before the provider matrix existed have no
#: ``provider_id``/``profile_id``, and no raw row has ever carried
#: ``dataset_id``/``benchmark_variant`` -- those two travel with the FILE (see
#: :func:`artifact_path`), not with the row. Substituting an explicit sentinel
#: rather than an empty string keeps three properties:
#:
#: 1. a legacy row can never compare equal to a row that genuinely belongs to
#:    another provider or profile, because "unknown" is its own value;
#: 2. a plan built the same way (a legacy replay whose jobs also carry no
#:    provider) still matches its own cached rows, so resume does not
#:    re-request an entire old raw file;
#: 3. the fallback is visible in every key, log line, and test, instead of
#:    silently collapsing distinct cells onto one blank identity.
LEGACY_IDENTITY = "legacy"

#: The two views an :class:`AttemptLedger` exposes. Every metric derived from an
#: append-only raw file must name the one it consumes.
ATTEMPT_VIEW_ALL = "all_attempts"
ATTEMPT_VIEW_LATEST = "latest_logical_observations"
ATTEMPT_VIEWS = (ATTEMPT_VIEW_ALL, ATTEMPT_VIEW_LATEST)


def _identity_field(row: Mapping[str, Any], *names: str, default: str = "") -> str:
    """Return the first non-empty value among `names`, else `default`.

    Args:
        row: Row to read.
        *names: Field names to try, in order.
        default: Value returned when every field is blank or absent.

    Returns:
        The stripped field value or `default`.
    """
    for name in names:
        value = str(row.get(name, "") or "").strip()
        if value:
            return value
    return default


@dataclass(frozen=True, slots=True)
class ObservationIdentity:
    """What makes one planned observation distinct from every other one.

    The pipeline used to reconstruct this from a different subset of fields at
    every boundary: the resume cache keyed on
    `(model, task, item, sample_kind, sample_index)`, the Task 2 -> Task 3
    handoff on `model` alone, and the paper joins on `(model, item)`. All
    three keys are ambiguous in the current four-cell x multi-provider matrix:
    benchmark item ids are reused between the `must` and `shall` variants of
    a family, and the same model is served by more than one provider profile.
    This value type is the single answer, and every one of those boundaries
    derives its key from it.

    `provider_id`/`profile_id` come off the row; `dataset_id`/`variant`
    come off the row when it records them and otherwise from the caller's file
    context (a raw JSONL file holds exactly one dataset/variant). Anything still
    missing becomes `LEGACY_IDENTITY` -- explicitly, never an empty
    string. Recorded values always win over inherited context, so a row that
    names its own cell is never re-labelled by the file it was read from.

    Attributes:
        provider_id: Provider that served the observation.
        profile_id: Provider profile the observation was requested under.
        model: Model identifier.
        dataset_id: Benchmark dataset.
        variant: Benchmark variant (cell) of the dataset.
        task: `task1`, `task2` or `task3`.
        item_id: Benchmark item id.
        sample_kind: `deterministic` or `stochastic`.
        sample_index: Index of the sample within the item's samples.
    """

    provider_id: str
    profile_id: str
    model: str
    dataset_id: str
    variant: str
    task: str
    item_id: str
    sample_kind: str
    sample_index: int

    @classmethod
    def from_raw_row(
        cls,
        row: Mapping[str, Any],
        *,
        dataset_id: str | None = None,
        variant: str | None = None,
    ) -> ObservationIdentity:
        """Build the identity of a raw row, a planned job, or a score row.

        Planned jobs nest the benchmark item under `item` instead of carrying
        `item_id`, and the registry/score rows spell the cell
        `dataset_id`/`benchmark_variant` while cell dicts spell it
        `dataset`/`variant`; all of those spellings resolve to one identity
        so a job and the record it produced share a key.

        Args:
            row: Raw row, planned job or score row.
            dataset_id: Dataset of the file the row was read from; used only when
                the row records none.
            variant: Benchmark variant of that file; used only when the row records
                none.

        Returns:
            The identity; a missing or unparsable `sample_index` becomes 0.
        """
        item_id = _identity_field(row, "item_id")
        nested_item = row.get("item")
        if not item_id and isinstance(nested_item, Mapping):
            item_id = _identity_field(nested_item, "item_id")
        try:
            sample_index = int(row.get("sample_index", 0) or 0)
        except (TypeError, ValueError):
            sample_index = 0
        return cls(
            provider_id=_identity_field(row, "provider_id", default=LEGACY_IDENTITY),
            profile_id=_identity_field(row, "profile_id", default=LEGACY_IDENTITY),
            model=_identity_field(row, "model"),
            dataset_id=_identity_field(
                row,
                "dataset_id",
                "dataset",
                default=str(dataset_id or "").strip() or LEGACY_IDENTITY,
            ),
            variant=_identity_field(
                row,
                "benchmark_variant",
                "variant",
                default=str(variant or "").strip() or LEGACY_IDENTITY,
            ),
            task=_identity_field(row, "task"),
            item_id=item_id,
            sample_kind=_identity_field(row, "sample_kind"),
            sample_index=sample_index,
        )

    @property
    def record_key(self) -> tuple[str, str, str, str, str, str, str, str, int]:
        """Return the hashable form used as a cache, dedup, and join key.

        Returns:
            The identity fields as a tuple, ending with the sample index.
        """
        return (
            self.provider_id,
            self.profile_id,
            self.model,
            self.dataset_id,
            self.variant,
            self.task,
            self.item_id,
            self.sample_kind,
            self.sample_index,
        )


@dataclass(slots=True)
class AttemptLedger:
    """The two readings of an append-only raw file, side by side.

    A raw JSONL file records physical ATTEMPTS: a request that failed and was
    re-requested by a later resume leaves two rows for one planned observation.
    Consumers disagreed about which of those two numbers they wanted, so the
    same recovered run could read as `2/1` records, a degraded registry parse
    rate, and `1/1` task progress at the same time.

    This ledger holds both views and forces callers to name theirs:

    - `all_attempts` -- every row, in file order. The view for anything
      that describes provider work: API calls, retries, latency, tokens, and
      the parse-status histogram of what the provider actually returned.
    - `latest_logical_observations` -- one row per
      `run_id` + `ObservationIdentity`, keeping the last row whose
      `parse_status` is `ok` (a successful retry supersedes the earlier
      failure) or the last row of the group when none parsed. The view for
      anything counted against a PLAN: coverage, completion rate, parse success
      rate, and scoring.

    Attributes:
        all_attempts: Every raw row, in file order.
    """

    all_attempts: list[dict[str, Any]]
    #: Memoized `latest_logical_observations`. A live runner refreshes progress
    #: every ~100 records over a growing raw file, and each refresh reads this
    #: view several times; deduplicating once per ledger keeps that linear.
    _latest: list[dict[str, Any]] | None = dataclass_field(
        default=None, init=False, repr=False, compare=False
    )

    @classmethod
    def from_raw_rows(
        cls, raw_rows: Iterable[Mapping[str, Any]] | AttemptLedger
    ) -> AttemptLedger:
        """Accept rows or an existing ledger, so callers can pass either.

        Args:
            raw_rows: Raw rows, copied into the new ledger, or a ledger.

        Returns:
            The given ledger unchanged, or a new ledger over the rows.
        """
        if isinstance(raw_rows, AttemptLedger):
            return raw_rows
        return cls([dict(row) for row in raw_rows])

    @property
    def latest_logical_observations(self) -> list[dict[str, Any]]:
        """Return the latest-logical-observations view, deduplicated once and memoized.

        Returns:
            `all_attempts` reduced to one row per logical observation.
        """
        if self._latest is None:
            self._latest = dedupe_raw_rows(self.all_attempts)
        return self._latest

    def view(self, view: str) -> list[dict[str, Any]]:
        """Return the named view; raises rather than guessing which one was meant.

        Args:
            view: `all_attempts` or `latest_logical_observations`.

        Returns:
            The rows of that view.

        Raises:
            ValueError: If the view name is unknown.
        """
        if view == ATTEMPT_VIEW_ALL:
            return self.all_attempts
        if view == ATTEMPT_VIEW_LATEST:
            return self.latest_logical_observations
        raise ValueError(
            f"Unknown attempt view {view!r}; expected one of {ATTEMPT_VIEWS}."
        )


def configure_run_logging(
    level: str | int = "INFO",
    *,
    log_path: str | Path | None = None,
) -> logging.Logger:
    """Configure the shared `re_uq` logger for a CLI run.

    Always logs to stderr; when `log_path` is given the same records are also
    appended to that file so a long provider run leaves a durable trace.

    A file handler left by an earlier run in the same process is removed and
    closed, and the stderr handler is reused rather than duplicated.

    Args:
        level: Logging level as a number or a name such as `"DEBUG"`; unknown
            names fall back to `INFO`.
        log_path: File to append the records to; its parent directories are
            created.

    Returns:
        The configured `re_uq` logger.
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
    """Return whether a parse status counts as a failure.

    Any status other than `ok` (including `truncated`) is a parse failure.

    Args:
        status: Value of a row's `parse_status`.

    Returns:
        True unless the stripped status is `ok`.
    """
    return str(status or "").strip() != PARSE_STATUS_OK


def parse_status_histogram(raw_rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Return categorized parse-status counts with every known category always present.

    `ok` rows are not counted, and statuses outside `PARSE_STATUS_CATEGORIES` are
    counted under `other`.

    Args:
        raw_rows: Raw rows to count.

    Returns:
        The count of each failure category and of `other`.
    """
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


def run_quality_counters(
    attempt_rows: Iterable[Mapping[str, Any]] | AttemptLedger,
) -> dict[str, Any]:
    """Return the diagnostics shared by the registry row, finish event and summary line.

    Covers the categorized parse-status histogram, retry volume, truncation
    counts, latency percentiles, provider-reported completion tokens, and the
    tolerant-parse repair histogram. Request-level fields are deduplicated by
    `batch_id`; parse, repair, and truncation statuses remain per output
    record.

    View: `ATTEMPT_VIEW_ALL`. Everything here describes physical provider
    work -- what was sent, what came back, how long it took -- so a failed
    attempt that a later resume superseded still belongs in these counts. Pass
    `AttemptLedger.all_attempts` explicitly at the call site; the
    plan-facing counters live in `live_run_counters` and
    `run_progress_summary`.

    Args:
        attempt_rows: Every raw row of the run, or its ledger.

    Returns:
        `parse_status_histogram`, `parse_repairs`, `retry_total`,
        `truncated_records`, `latency_p50_s`, `latency_p95_s` and
        `usage_completion_tokens`; the latency and token fields are blank when no
        request reported them.
    """
    rows = AttemptLedger.from_raw_rows(attempt_rows).all_attempts
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
        # A response the tolerant parser had to repair is `ok`, so it is
        # invisible in the histogram above; report the repairs next to it.
        "parse_repairs": parse_repair_counts(rows),
        "retry_total": retry_total,
        "truncated_records": truncated_records,
        "latency_p50_s": _percentile(latencies, 0.50),
        "latency_p95_s": _percentile(latencies, 0.95),
        "usage_completion_tokens": usage_completion_tokens if usage_seen else "",
    }


def format_run_quality_line(run_id: str, quality: Mapping[str, Any]) -> str:
    """Format run quality counters as one human-readable log line.

    Args:
        run_id: Run the counters describe.
        quality: Counters from `run_quality_counters`.

    Returns:
        The line, listing only non-zero parse statuses and repairs.
    """
    histogram = dict(quality.get("parse_status_histogram", {}) or {})
    nonzero = {key: value for key, value in histogram.items() if value}
    latency_p50 = quality.get("latency_p50_s", "")
    latency_p95 = quality.get("latency_p95_s", "")
    latency_label = (
        f"latency_p50 {float(latency_p50):.2f}s, latency_p95 {float(latency_p95):.2f}s"
        if latency_p50 != "" and latency_p95 != ""
        else "latency_p50 unknown, latency_p95 unknown"
    )
    repairs = {
        key: value
        for key, value in dict(quality.get("parse_repairs", {}) or {}).items()
        if value
    }
    return (
        f"{run_id}: parse_status {nonzero or 'all ok'}, "
        f"repairs {repairs or 'none'}, "
        f"retries {int(quality.get('retry_total', 0) or 0)}, "
        f"truncated {int(quality.get('truncated_records', 0) or 0)}, "
        f"{latency_label}, "
        f"completion_tokens {quality.get('usage_completion_tokens', '') or 'unknown'}"
    )


def run_progress_summary(
    benchmark_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]] | AttemptLedger,
    expected_stochastic_samples: int = 5,
    *,
    view: str = ATTEMPT_VIEW_LATEST,
) -> list[dict[str, Any]]:
    """Summarize coverage and parse rate per (run, model, task).

    View: `ATTEMPT_VIEW_LATEST` by default. Progress is measured against a
    PLAN, so a failed request that a later resume re-requested successfully is
    one observation, not two: the run reads `1/1` rather than `2/1` with a
    depressed parse rate. Pass `view=ATTEMPT_VIEW_ALL` to summarize physical
    attempts instead (useful when auditing how much work a recovery cost).

    Args:
        benchmark_rows: Current benchmark items; rows of other items are ignored.
        raw_rows: Raw rows of the runs, or their ledger.
        expected_stochastic_samples: Planned stochastic samples per item; the
            expected record count is one deterministic record plus this many per
            item.
        view: Attempt view to summarize.

    Returns:
        One row per run, model and task with record counts, completion and parse
        success rates, and deterministic and stochastic item coverage; empty when
        the rows lack the identity or status columns.

    Raises:
        ValueError: If `view` is unknown.
    """
    ledger = AttemptLedger.from_raw_rows(raw_rows)
    rows = ledger.view(view)
    if not rows:
        return []
    frame = pd.DataFrame.from_records(rows)
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
    """Return the runs whose progress rows show every expected task complete.

    A run is complete when it has a progress row for each expected task and every
    row has full record completion, deterministic item coverage and complete
    stochastic items.

    Args:
        progress_rows: Rows from `run_progress_summary`.
        expected_tasks: Tasks a complete run must cover.
        prefix: Run prefix to restrict to; `None` keeps every run.

    Returns:
        The complete run ids, in first-seen order.
    """
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


def completion_record_key(
    row: Mapping[str, Any],
    *,
    dataset_id: str | None = None,
    variant: str | None = None,
) -> tuple[str, str, str, str, str, str, str, str, int]:
    """Return the cache/dedup key of one planned observation, from its full identity.

    Thin wrapper over `ObservationIdentity.from_raw_row` so the resume
    cache, `dedupe_raw_rows`, and batch reconstruction cannot drift apart.
    Provider and profile are part of the key, so a cell run under a different
    provider is never mistaken for one that is already complete; fields a legacy
    row does not carry fall back to `LEGACY_IDENTITY` on BOTH sides of the
    comparison, which is what keeps an old raw file resumable.

    Args:
        row: Raw row or planned job.
        dataset_id: Dataset of the file the row was read from, used when the row
            records none.
        variant: Benchmark variant of that file, used when the row records none.

    Returns:
        The `ObservationIdentity.record_key` of the row.
    """
    return ObservationIdentity.from_raw_row(
        row, dataset_id=dataset_id, variant=variant
    ).record_key


def pending_completion_jobs(
    jobs: Iterable[Mapping[str, Any]],
    raw_rows: Iterable[Mapping[str, Any]],
    run_id: str,
) -> list[dict[str, Any]]:
    """Return the planned jobs `run_id` has no reusable observation for.

    A record is reusable only when it shares the full
    `ObservationIdentity` of the planned job -- provider and profile
    included, so a cell re-run under a second provider is planned in full rather
    than served from the first provider's rows.

    Native output rejections are terminal failed observations, not missing
    calls. Ordinary resume must not draw again until such a sample succeeds.

    A reusable record must also have been produced under the job's current
    `job_config_sha`; a legacy record without one is reused with a warning.

    Args:
        jobs: Planned completion jobs.
        raw_rows: Raw rows already written; only rows of `run_id` with status
            `ok` or `model_output_error` count as done.
        run_id: Run being resumed.

    Returns:
        Copies of the jobs that still have to be requested, in plan order.
    """
    # Map each completed record to the job_config_sha it was produced under so
    # resume can re-run rows whose planned config has since changed. Legacy rows
    # predate the field and fall back to the old key-only match.
    completed: dict[tuple[str, str, str, str, str, str, str, str, int], str] = {}
    for row in raw_rows:
        if str(row.get("run_id", "")) == str(run_id) and str(
            row.get("parse_status", "")
        ) in {"ok", "model_output_error"}:
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
    """Build the run registry row of one run, model and task set.

    The row reports both readings of the raw file: `observed_records`,
    `parse_success_rate` and the coverage fields count logical observations
    against the plan, while `observed_attempts`, `observed_api_calls` and the
    run-quality counters count physical attempts. The provider provenance is
    checked for credential-shaped values before it is flattened to JSON.

    Args:
        benchmark_rows: Current benchmark items; they fix the planned item count.
        raw_rows: Raw rows; only rows of this run, model and tasks are used.
        run_id: Run the row describes.
        run_group_id: Run group recorded on the row.
        provider_id: Provider recorded on the row.
        profile_id: Provider profile recorded on the row.
        model: Model the row describes.
        dataset_id: Benchmark dataset, normalized before recording.
        variant: Benchmark variant, normalized before recording.
        tasks: Tasks the run covers.
        expected_stochastic_samples: Planned stochastic samples per item.
        started_at_utc: Start timestamp recorded on the row.
        finished_at_utc: Finish timestamp recorded on the row.
        status: Explicit status; `None` derives `complete` or `partial` from the
            record count and coverage.
        base_url: Provider endpoint recorded on the row.
        api_key_env: Name of the environment variable holding the API key.
        concurrency: Request concurrency recorded on the row.
        batch_size: Items per request; must be a positive integer and fixes the
            expected number of API calls.
        timeout_s: Request timeout recorded on the row.
        json_mode: Whether JSON mode was requested; recorded as `yes` or `no`.
        structured_output: Structured-output mode, normalized together with
            `json_mode`.
        request_extra_body: Extra request body, recorded as compact JSON.
        server_model_probe: What the server reported about the model, recorded as
            compact JSON.
        batch_order: Item order within requests, normalized before recording.
        item_context: Item context arm, normalized before recording.
        notes: Free-text notes.

    Returns:
        The registry row, keyed by `RUN_REGISTRY_FIELDS` columns.
    """
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
    # This row reports both readings of the raw file and says which is which:
    # `observed_records` / `parse_success_rate` / coverage count LOGICAL
    # observations against the plan, while `observed_attempts`,
    # `observed_api_calls`, and the run-quality block count physical attempts.
    ledger = AttemptLedger.from_raw_rows(run_rows)
    observation_rows = ledger.latest_logical_observations
    attempt_rows = ledger.all_attempts
    progress = run_progress_summary(
        benchmark_rows,
        ledger,
        expected_stochastic_samples=expected_stochastic_samples,
        view=ATTEMPT_VIEW_LATEST,
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
    observed_records = len(observation_rows)
    observed_attempts = len(attempt_rows)
    observed_api_calls = len(
        {
            str(row.get("batch_id") or f"single:{row.get('request_index', index)}")
            for index, row in enumerate(attempt_rows)
        }
    )
    ok_records = sum(
        1 for row in observation_rows if str(row.get("parse_status", "")) == "ok"
    )
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
        (
            row
            for row in observation_rows
            if str(row.get("sample_kind", "")) == "deterministic"
        ),
        observation_rows[0] if observation_rows else {},
    )
    run_config_shas = sorted(
        {
            str(row.get("job_config_sha", ""))
            for row in observation_rows
            if row.get("job_config_sha")
        }
    )
    config_sha = sha256_text("\n".join(run_config_shas)) if run_config_shas else ""
    quality = run_quality_counters(attempt_rows)
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
        "expected_stochastic_samples": int(expected_stochastic_samples),
        "expected_records": expected_records,
        # Logical observations (AttemptLedger.latest_logical_observations).
        "observed_records": observed_records,
        # Physical rows behind them (AttemptLedger.all_attempts); larger than
        # observed_records exactly when a resume re-requested a failed cell.
        "observed_attempts": observed_attempts,
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
        "parse_repairs": compact_json(
            {key: value for key, value in quality["parse_repairs"].items() if value}
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
    read-modify-write cycle is serialized with `file_lock` and published with
    a temp file + `os.replace` so readers never observe a half-written CSV.

    Args:
        path: Registry CSV; created when missing.
        row: Row to write; it replaces the row with the same run, profile, model,
            dataset and variant, or is appended.
        fieldnames: Column order of the file; `RUN_REGISTRY_FIELDS` by default.
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

    `expected_records` encodes both the number of planned benchmark items and
    stochastic samples. Validating it prevents deterministic-only or partial
    ablations from being selected merely because they are newer and marked
    complete.

    Args:
        row: Registry row to check.
        run_group_id: Run group the row must belong to.
        benchmark_item_count: Number of items in the benchmark.
        expected_stochastic_samples: Planned stochastic samples per item.
        required_tasks: Tasks the row must cover.
        exact_tasks: When given, the row's tasks must be exactly these.
        expected_batch_order: When given, the recorded batch order must match.
        expected_batch_size: When given, the recorded batch size must match.
        allow_partial_benchmark: Accept a plan over fewer items than the
            benchmark holds.
        allow_missing_batch_order: Accept a row that records no batch order.

    Returns:
        One description per mismatch; empty when the row is compatible.
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
    """Return the run-event JSONL for a dataset and variant.

    Args:
        root: Repository root.
        dataset_id: Dataset name or alias; selects the file-name suffix.
        variant: Benchmark variant; selects the file-name suffix.
        run_id: Run id; smoke runs are routed into the smoke tree.
        smoke: Force the smoke tree when true; `None` infers it from `run_id`.

    Returns:
        The event-log path, in the smoke tree for smoke runs.
    """
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
    """Return the live progress CSV for a dataset and variant.

    Args:
        root: Repository root.
        dataset_id: Dataset name or alias; selects the file-name suffix.
        variant: Benchmark variant; selects the file-name suffix.
        run_id: Run id; smoke runs are routed into the smoke tree.
        smoke: Force the smoke tree when true; `None` infers it from `run_id`.

    Returns:
        The live progress path, in the smoke tree for smoke runs.
    """
    return resolve_run_artifact_path(
        artifact_path(
            Path(root) / "data/processed/run_progress_live.csv", dataset_id, variant
        ),
        run_id=run_id,
        smoke=smoke,
    )


def live_run_counters(
    raw_rows: list[dict[str, Any]] | AttemptLedger,
    *,
    expected_records: int,
    expected_api_calls: int,
    started_monotonic: float | None = None,
    now_monotonic: float | None = None,
) -> dict[str, Any]:
    """Return live progress counters for one cell, from both views of its raw rows.

    `expected_records` is a count of PLANNED jobs, so every counter compared
    against it -- `observed_records`, the completion rate, parse/error rates,
    throughput, and the ETA -- reads
    `AttemptLedger.latest_logical_observations`. A failed request that a
    resume re-requested successfully is therefore one record at 100%, not two
    records at 200% with half the parse rate (which is what used to trip the
    parse-failure warning on a recovered run).

    `observed_attempts`, `observed_api_calls`, and the
    `run_quality_counters` block read `AttemptLedger.all_attempts`:
    those describe what the provider was actually asked to do.

    Args:
        raw_rows: Raw rows of the cell, or their ledger.
        expected_records: Planned records of the cell.
        expected_api_calls: Planned provider requests of the cell.
        started_monotonic: `time.monotonic()` reading at the start of the cell;
            `None` reports zero elapsed time.
        now_monotonic: Current monotonic reading; `None` reads the clock.

    Returns:
        The run-quality counters plus record, parse, request-error and API-call
        counts and rates, elapsed seconds, throughput and the ETA in seconds
        (blank when the throughput is still zero, 0 when nothing remains).
    """
    ledger = AttemptLedger.from_raw_rows(raw_rows)
    observation_rows = ledger.latest_logical_observations
    attempt_rows = ledger.all_attempts
    observed_records = len(observation_rows)
    ok_records = sum(
        1 for row in observation_rows if str(row.get("parse_status", "")) == "ok"
    )
    request_error_records = sum(
        1
        for row in observation_rows
        if str(row.get("parse_status", "")) == "request_error"
    )
    observed_api_calls = len(
        {
            str(row.get("batch_id") or f"single:{row.get('request_index', index)}")
            for index, row in enumerate(attempt_rows)
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
        **run_quality_counters(attempt_rows),
        "expected_records": int(expected_records),
        "observed_records": observed_records,
        "observed_attempts": len(attempt_rows),
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
    """Format live progress counters as one human-readable log line.

    Args:
        run_id: Run the counters describe.
        counters: Counters from `live_run_counters`.

    Returns:
        The line with record and API-call progress, parse success, errors,
        elapsed time and ETA.
    """
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
    """Append one event to the run-event JSONL, stamped with the current UTC time.

    Args:
        path: Run-event file.
        event: Event fields; a `created_at_utc` field in it overrides the stamp.
    """
    append_jsonl(path, {"created_at_utc": utc_now_iso(), **dict(event)})


def write_live_progress_csv(
    path: str | Path,
    benchmark_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]] | AttemptLedger,
    expected_stochastic_samples: int,
) -> None:
    """Publish per-task progress for one cell (logical-observation view).

    Args:
        path: Live progress CSV to overwrite.
        benchmark_rows: Current benchmark items.
        raw_rows: Raw rows of the cell, or their ledger.
        expected_stochastic_samples: Planned stochastic samples per item.
    """
    progress = run_progress_summary(
        benchmark_rows,
        raw_rows,
        expected_stochastic_samples=expected_stochastic_samples,
        view=ATTEMPT_VIEW_LATEST,
    )
    write_csv_rows(path, progress, fieldnames=RUN_PROGRESS_FIELDS)


def warning_events_for_counters(
    counters: Mapping[str, Any],
    logging_config: Mapping[str, Any],
    emitted_warning_types: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return warning events for parse-failure and request-error rates above threshold.

    Nothing is reported before `warn_after_records` records have been observed,
    and a warning type already emitted is not repeated.

    Args:
        counters: Counters from `live_run_counters`.
        logging_config: Run logging settings; missing thresholds fall back to
            `DEFAULT_RUN_LOGGING`.
        emitted_warning_types: Warning types already emitted for this run.

    Returns:
        Zero, one or two warning events, each carrying the threshold, the observed
        rate and all counters.
    """
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
    max_tokens: int = 48,
    completion_fn: Callable[..., dict[str, Any]] = chat_completion,
) -> dict[str, Any]:
    """Send one fixed-answer request and check that the reply parses.

    `max_tokens` defaults to a budget that fits the probe answer alone; a
    profile whose models always emit a reasoning channel (Muse Glimmer,
    gpt-oss) must pass its own budget, or the probe is truncated before the
    answer starts and the preflight fails with an empty text.

    The request is a single-item Task 1 request at temperature 0, with its
    response format resolved by `resolve_response_format_args`.

    Args:
        host: Provider base URL.
        model: Model to probe.
        api_key_env: Name of the environment variable holding the API key.
        timeout_s: Request timeout in seconds.
        prompt_version: Prompt version of the run; selects the confidence scale
            the reply is parsed with.
        json_mode: Legacy JSON-mode switch; requests `json_object` when no other
            response format applies.
        structured_output: Structured-output mode or alias.
        response_format: Profile-level response format, used only in `none` mode.
        extra_body: Provider-specific request body.
        instructor_mode: Instructor library mode name.
        validation_retries: Instructor validation retries.
        max_tokens: Completion token budget of the probe.
        completion_fn: Completion function; Instructor profiles are routed to
            `instructor_completion` when this is `chat_completion`.

    Returns:
        `ok` (the request succeeded and the reply parsed), `parse_status`,
        `error`, `latency_s` and the first 240 characters of `raw_text`.
    """
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
            max_tokens=max_tokens,
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
    """Return the rows of one run and model that belong to the given tasks.

    Args:
        rows: Raw rows to filter.
        run_id: Run to keep.
        model: Model to keep.
        tasks: Tasks to keep.

    Returns:
        The matching rows, in input order.
    """
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
    """Run the provider preflight with a profile's settings; fail if it does not parse.

    The probe gets the profile's per-item `max_tokens`, so a reasoning model has
    the same room before its answer as a real item.

    Args:
        profile: Provider profile; `base_url`, `api_key_env`, `timeout_s` and
            `json_mode` are required, the request options are optional.
        model: Model to probe.
        prompt_version: Prompt version of the run.
        completion_fn: Completion function passed to `provider_preflight`.

    Returns:
        The preflight result of `provider_preflight`.

    Raises:
        RuntimeError: If the preflight request failed or its reply did not parse.
    """
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
        # The profile's per-item budget: reasoning models spend most of it
        # before the answer, so the probe needs the same room a real item gets.
        max_tokens=int(profile.get("max_tokens", 48)),
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
    """Log new rate warnings for a run and append them to the run-event file.

    Args:
        counters: Counters from `live_run_counters`.
        logging_config: Run logging settings; `write_event_jsonl` must be present
            and decides whether events are appended to `events_path`.
        emitted_warning_types: Warning types already emitted; updated in place so
            each type is reported once per run.
        context: Run identity fields merged into every event; `run_id` also
            prefixes the log line.
        events_path: Run-event JSONL file.
    """
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
    """Return the output paths of the preliminary result snapshot.

    Args:
        root: Repository root.
        variant: Benchmark variant; selects the file-name suffix.
        dataset_id: Dataset name or alias; selects the file-name suffix.

    Returns:
        The `scores`, `summary` and `progress` CSVs and the markdown `table`.
    """
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
    include_baseline: bool = True,
    *,
    sampling_plan: SamplingPlan | None = None,
    infer_plan_from_observations: bool = False,
) -> dict[str, Any]:
    """Write the mid-run snapshot for the plan the run is executing under.

    The snapshot is read while a run is still in flight, which is exactly when
    unwritten samples exist, so it takes the same required
    `SamplingPlan` as `build_uq_scores` and reports progress
    against the same denominator.

    Only benchmark items with a current raw output are scored. The snapshot
    writes the score rows, the metric summary, the run progress and a markdown
    table of the last two.

    Args:
        benchmark_rows: Current benchmark items.
        raw_rows: Raw rows written so far.
        root: Repository root.
        variant: Benchmark variant; selects the file-name suffix.
        dataset_id: Dataset name or alias; selects the file-name suffix.
        include_baseline: Add the rule-based baseline scores.
        sampling_plan: Plan the run is executing under.
        infer_plan_from_observations: Explicit opt-in to the exploratory inferred
            plan when no `sampling_plan` is given.

    Returns:
        The written paths, the number of score, summary and progress rows, and the
        plan source and planned stochastic samples.
    """
    plan = resolve_sampling_plan(
        sampling_plan,
        infer_plan_from_observations,
        caller="write_preliminary_result_snapshot",
    )
    paths = preliminary_result_paths(root, variant, dataset_id=dataset_id)
    scored_benchmark_rows = benchmark_rows_with_current_raw_outputs(
        benchmark_rows, raw_rows
    )
    scores = build_uq_scores(scored_benchmark_rows, raw_rows, sampling_plan=plan)
    if include_baseline:
        scores.extend(build_rule_baseline_scores(scored_benchmark_rows))
    summary = metric_summary_by_model_task_method(scores)
    progress = run_progress_summary(
        benchmark_rows,
        raw_rows,
        expected_stochastic_samples=plan.stochastic_samples,
    )
    score_fields = [
        "run_id",
        "model",
        "task",
        "uq_method",
        "item_id",
        "seed_id",
        "batch_id",
        "sample_batch_ids",
        "source_modality",
        "ordinal_strength",
        "numeric_strength",
        "valid_n",
        "total_n",
        "parse_failures",
        "stochastic_complete",
        "sampling_plan_source",
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
        "strict_text_overcommit_kind",
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
        "sampling_plan_source": plan.source,
        "expected_stochastic_samples": plan.stochastic_samples,
    }


def score_base(
    raw: dict[str, Any],
    item: dict[str, Any],
    uq_method: str,
    valid_n: int,
    total_n: int,
) -> dict[str, Any]:
    """Return the identity and provenance fields shared by every score row.

    `batch_id` is the provider request this row's answer was decided in and
    is the default bootstrap cluster (`DEFAULT_BOOTSTRAP_CLUSTER_FIELD`);
    `sample_batch_ids` lists every contributing request, which only differs
    from `batch_id` for a collapsed stochastic group. Both are blank when the
    raw record carries no `batch_id` (legacy, single-item, or fake rows).

    Args:
        raw: Raw row that supplies the run identity and answer length.
        item: Benchmark item of the row.
        uq_method: UQ method recorded on the row.
        valid_n: Samples that parsed.
        total_n: Samples the plan asked for.

    Returns:
        The shared fields, including the parse-failure count, the
        `stochastic_complete` flag and the answer-length fields.
    """
    batch_id = raw_batch_id(raw)
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
        "batch_id": batch_id,
        "sample_batch_ids": batch_id,
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
    """Return the shared score-row fields for the rule-based baseline.

    The baseline sent no request, so it has no request cluster and no answer
    length, and it always counts as one complete, parsed sample.

    Args:
        item: Benchmark item of the row.
        task: Task the row scores.

    Returns:
        The shared fields, in the same shape as `score_base`.
    """
    return {
        "run_id": "rule-based-baseline",
        "model": RULE_BASELINE_MODEL,
        "task": task,
        "uq_method": RULE_BASELINE_METHOD,
        "item_id": item["item_id"],
        "seed_id": item["seed_id"],
        # The rule-based baseline never sent a request, so it has no cluster.
        "batch_id": "",
        "sample_batch_ids": "",
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
    """Score the deterministic rule-based baseline on Task 1 and Task 2.

    The baseline reads the modality off the source statement with
    `rule_based_source_modality`, answers Task 1 `yes` exactly for `mandatory`, and
    reports that modality for Task 2, always with confidence 1.

    Args:
        benchmark_rows: Benchmark items to score.

    Returns:
        One Task 1 and one Task 2 score row per item.

    Raises:
        ValueError: If the rules find no modality in a source statement.
    """
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
    """Summarize per-prompt Task 1 drift in the deterministic prompt-sensitivity probe.

    Args:
        benchmark_rows: Benchmark items of the probe.
        raw_rows: Raw rows of the probe; each prompt variant is its own run.

    Returns:
        One row per model and prompt run with the Task 1 item count and accuracy,
        the share of nonmandatory sources answered `yes` with confidence at least
        0.80 and 0.90, and the mean `p_yes` on those sources.
    """
    scores = build_uq_scores(
        benchmark_rows, raw_rows, sampling_plan=DETERMINISTIC_ONLY_SAMPLING_PLAN
    )
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
    """Summarize per-prompt Task 2 accuracy and over-commitment.

    Every parsed answer with a recognized modality is scored against the item's
    gold modality; the nice-to-have columns isolate the weakest gold modality.

    Args:
        benchmark_rows: Benchmark items of the probe.
        raw_rows: Raw rows of the probe; each prompt variant is its own run and
            only Task 2 rows are used.

    Returns:
        One row per model and prompt run with raw and valid counts, parse success,
        accuracy, nice-to-have accuracy and the share of nice-to-have items
        predicted `recommended`, over-commitment, and the high-confidence
        over-commitment rates at 0.80 and 0.90.
    """
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
    """Summarize the weak-modality probe per model, run, template and sample kind.

    Every probe item is a nice-to-have statement, so `accuracy` is the share of
    valid answers predicted `nice_to_have`, and any stronger prediction is an
    over-commitment.

    Args:
        probe_items: Probe items, keyed by `item_id`.
        raw_rows: Raw rows of the probe; only Task 2 rows are used.

    Returns:
        One row per model, run, template and sample kind, with raw and valid
        counts, parse success, accuracy, the share predicted `recommended`,
        over-commitment overall and at confidence 0.80 and 0.90, the share of each
        predicted modality and the mean confidence; sorted by model, run, sample
        kind and template.
    """
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
    """Write the weak-modality probe summary as a CSV and a markdown table.

    Args:
        summary_rows: Rows from `weak_modality_probe_summary`.
        output_dir: Output directory; created when missing.
        suffix: Appended to both file names before the extension.

    Returns:
        The `csv` and `markdown` paths.
    """
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
    """Group rows by the values of some columns.

    Key columns missing from every row are filled with an empty string. As with
    pandas grouping, a row whose key value is `None` or absent while other rows
    carry it is dropped.

    Args:
        rows: Rows to group.
        keys: Columns to group by.

    Returns:
        The rows of each group, keyed by the tuple of key values, in first-seen
        order.
    """
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
    """Return the over- and under-commitment rates of Task 2 rows.

    Over-commitment means the predicted modality is stronger than the gold one on
    the ordinal scale of `ORDINAL_STRENGTH`; its severity is the number of steps.

    Args:
        rows: Rows with `gold_modality` and `pred_modality`.

    Returns:
        `over_commitment`, `under_commitment`, and the mean severity over all rows
        and over the over-committed rows; all NaN without rows, and the
        conditional severity is NaN when nothing over-commits.
    """
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


def unsupported_mandatory_acceptance_counts(
    rows: list[dict[str, Any]], threshold: float
) -> tuple[int, int]:
    """Count high-confidence yes decisions / valid nonmandatory-source Task 1 answers.

    Args:
        rows: Score rows; only Task 1 rows with `y_true == 0` are eligible.
        threshold: Minimum confidence of an accepted `yes`.

    Returns:
        The number of accepted rows and the number of eligible rows.
    """
    weak_rows = [
        row
        for row in rows
        if str(row.get("task", "")) == "task1" and int(row["y_true"]) == 0
    ]
    accepted = sum(
        1
        for row in weak_rows
        if str(row.get("y_pred", "")) == "1"
        and row.get("confidence") not in {None, ""}
        and float(row["confidence"]) >= threshold
    )
    return accepted, len(weak_rows)


def unsupported_mandatory_acceptance_rate(
    rows: list[dict[str, Any]], threshold: float
) -> float:
    """Return the share of nonmandatory-source Task 1 answers accepted as mandatory.

    Args:
        rows: Score rows.
        threshold: Minimum confidence of an accepted `yes`.

    Returns:
        The rate from `unsupported_mandatory_acceptance_counts`, or NaN without
        eligible rows.
    """
    accepted, eligible = unsupported_mandatory_acceptance_counts(rows, threshold)
    return accepted / eligible if eligible else math.nan


def task2_high_confidence_overcommitment_rate(
    rows: list[dict[str, Any]],
    threshold: float,
    denominator: str = "all",
) -> float:
    """Return the share of Task 2 rows that over-commit with high confidence.

    A row counts when its predicted modality is stronger than the gold one and its
    confidence is at least `threshold`.

    Args:
        rows: Score rows; only Task 2 rows are used.
        threshold: Minimum confidence of a counted over-commitment.
        denominator: `all` counts every Task 2 row; `overcommittable` counts only
            rows whose gold modality is weaker than `mandatory`.

    Returns:
        The rate, or NaN without rows in the denominator.

    Raises:
        ValueError: If `denominator` is unknown.
    """
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
    """Return the share of nice-to-have Task 2 rows predicted with a stronger modality.

    Args:
        rows: Score rows; only Task 2 rows with gold `nice_to_have` are used.
        threshold: Minimum confidence of a counted strengthening; `None` counts
            every one.

    Returns:
        The rate, or NaN without nice-to-have rows.
    """
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
    """Return the share of correctly labelled Task 2 rows whose text over-commits.

    The label can be right while the rewritten requirement text still states a
    stronger modality (`text_overcommit`).

    Args:
        rows: Score rows; only Task 2 rows with a truthy `y_true` are used.
        threshold: Minimum confidence of a counted row; `None` counts every one.

    Returns:
        The rate, or NaN without correctly labelled rows.
    """
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
    """Return the share of nice-to-have Task 2 rows predicted `nice_to_have`.

    Args:
        rows: Score rows; only Task 2 rows with gold `nice_to_have` are used.

    Returns:
        The recall, or NaN without nice-to-have rows.
    """
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
    """Return the high-confidence over-commitment rate of one task.

    Task 1 uses `unsupported_mandatory_acceptance_rate` and Task 2 uses
    `task2_high_confidence_overcommitment_rate` over all rows; other tasks have
    no such rate.

    Args:
        rows: Score rows.
        task: Task whose rows are used.
        threshold: Minimum confidence of a counted over-commitment.

    Returns:
        The rate, or NaN for other tasks or without rows of the task.
    """
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
    """Return whether a CSV-style cell is true.

    Args:
        value: A bool, used as is, or a value whose stripped, lower-cased text is
            compared with `1`, `true` and `yes`.

    Returns:
        True only for a true bool or one of those strings.
    """
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
    """Return the text-modality summary metrics with every value blank.

    Returns:
        The keys of `text_modality_summary_metrics`, each mapped to an empty
        string.
    """
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
    """Return worst/best-case bounds for a rate whose denominator drops unknown text.

    The published rate uses `denominator` (rows whose text modality parsed).
    `*_lower_bound` charges every unknown row as non-strengthening and
    `*_upper_bound` charges every unknown row as strengthening, both over the
    coverage-complete denominator `denominator + unknown_excluded`.

    Args:
        metric: Name of the rate; prefixes every returned key.
        numerator: Rows counted by the rate.
        denominator: Rows whose text modality parsed.
        unknown_excluded: Rows left out because their text modality is unknown.

    Returns:
        The three counts and the two bounds; the bounds are blank when there are
        no rows at all.
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
    """Return text-modality drift metrics with coverage-adjusted bounds.

    The headline `text_over_commitment` / `strict_text_over_commitment`
    rates are computed over rows whose generated text yielded a modality
    (`text_modality_parse_status == "ok"`). Rows whose text modality is
    `unknown` or `negated` are excluded from that denominator, so each
    metric is also reported with its numerator, denominator, the number of
    excluded unknown rows, and the worst/best-case bounds that put every
    excluded row on one side.

    Args:
        rows: Score rows; only rows whose `text_modality_parse_status` is `ok` or
            `unknown` are used.

    Returns:
        The keys of `empty_text_modality_summary_metrics`. Everything is blank
        without such rows and the rates over parsed text are blank when no text
        parsed; the all-row accuracy, coverage, negated and multi-modal rates
        use every used row as denominator.
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
    """Return the variation ratio (1 - p_majority) for one repeated-sample score row.

    Args:
        row: Score row; its `uncertainty_score` is used when it is a variation
            ratio, otherwise the ratio is computed from `label_distribution`.

    Returns:
        The variation ratio, or `None` when neither field yields one.
    """
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
    """Return repeated-sample agreement restricted to coverage-complete rows.

    A published agreement / unanimity number may only be computed over items
    where every requested stochastic sample parsed (`stochastic_complete`).
    Items with `valid_n < total_n` are excluded and counted in
    `agreement_n_incomplete_excluded` so the reported share is auditable.

    Only rows that planned more than one sample count. When consistency-method
    rows are among them, only those are used, so the other sampling-based rows
    of the same item are not counted again.

    Args:
        rows: Score rows.

    Returns:
        `repeated_sample_unanimity` (share of complete rows whose samples all
        agree), `mean_repeated_sample_agreement` (mean share of the majority
        label), `agreement_n_complete` and `agreement_n_incomplete_excluded`; the
        two rates are blank when no complete row has a usable distribution.
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
    """Return the answer-length metrics with every value blank.

    Returns:
        The keys of `length_bloat_metrics`, including the per-tercile and
        per-modality keys, each mapped to an empty string.
    """
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
    """Return answer-length / bloat metrics and the length-vs-strengthening breakdown.

    Rows are bucketed into `LENGTH_TERCILE_COUNT` equal-count terciles of
    `length_ratio` (generated requirement words / source statement words) and
    the broad and strict text-strengthening rates are reported per bucket, so a
    bloat/strengthening interaction is visible rather than pooled away.

    Args:
        rows: Task 2 score rows with the answer-length fields; blank or NaN values
            are skipped per metric.

    Returns:
        The keys of `empty_length_bloat_metrics`: mean word counts, length ratio
        and completion tokens, the mean requirement length per source modality,
        and per tercile the parsed-text row count, mean length ratio and both
        strengthening rates (over rows whose text modality parsed), plus
        `|`-joined summaries of the tercile bounds and rates.
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
    """Pick the most severe high-confidence over-commitment examples.

    Task 1 examples are nonmandatory sources with `p_yes` at least `threshold`,
    ranked by `p_yes`. Task 2 examples predict a stronger modality than the gold
    one with confidence at least `threshold`, ranked by the number of steps plus
    the confidence. Rule-based baseline rows are skipped.

    Args:
        scores: Score rows.
        benchmark_rows: Benchmark items; supply the source statement and the
            candidate requirement.
        limit: Maximum number of examples.
        threshold: Minimum `p_yes` or confidence of an example.

    Returns:
        The top examples by descending risk score, then model and seed.
    """
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
    """Write the qualitative over-commitment examples as a CSV and a markdown table.

    Args:
        scores: Score rows.
        benchmark_rows: Benchmark items.
        output_dir: Output directory; created when missing.
        suffix: Appended to both file names before the extension.
        limit: Maximum number of examples.
        threshold: Minimum `p_yes` or confidence of an example.

    Returns:
        The `csv` and `markdown` paths; the markdown file holds a placeholder
        sentence when there are no examples.
    """
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
    """Return the probabilities a task's calibration metrics are computed on.

    Args:
        rows: Score rows of the task.
        task: `task1` uses `p_yes`; every other task uses `confidence`.

    Returns:
        One probability per row.
    """
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
    """Return whether each score row is a prediction error.

    Args:
        rows: Score rows of the task.
        task: `task1` compares `y_true` with `y_pred`; `task2` and `task3` rows
            carry correctness in `y_true`, so an error is `y_true == 0`.

    Returns:
        1 for an error and 0 otherwise, one per row.

    Raises:
        ValueError: If `rows` is not empty and the task is unknown.
    """
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
    """Return how well `uncertainty_score` separates errors from correct predictions.

    Rows with a blank, unparsable or NaN score are skipped.

    Args:
        rows: Score rows of the task.
        task: Task of the rows; decides what counts as an error.

    Returns:
        The AUROC of the uncertainty score against the error labels, or NaN when
        only one class remains.
    """
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
    """Min-max normalize ACSE uncertainty scores and split them for calibration.

    Scores are normalized within each run, model, task and embedding backend, and
    the group's seeds are split at random (fixed seed `ACSE_CALIBRATION_SEED`)
    into a calibration and an evaluation half, so threshold selection and
    evaluation never share a seed. A group with a single seed, or without seed
    ids, is calibration only.

    Args:
        scores: Score rows; only ACSE rows with a finite score are used.
        calibration_fraction: Share of seeds in the calibration split; at least
            one seed is always left for evaluation when there are two or more.

    Returns:
        One normalized row per ACSE score row, with the raw and normalized
        scores, the group's raw range, the split, the prediction error and the
        cluster diagnostics; sorted by model, task, backend and item.
    """
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
    """Check whether an ACSE deferral threshold tuned on calibration seeds transfers.

    For each target accepted error rate, the largest normalized threshold whose
    accepted rows (score at or below it) meet the target on the calibration split
    is selected and then applied to the evaluation split. Groups without
    evaluation rows are evaluated on the calibration rows themselves, which
    `evaluation_mode` records as `resubstitution_no_heldout`.

    Args:
        normalized_rows: Rows from `acse_normalized_score_rows`.
        target_error_rates: Accepted error rates to tune thresholds for.

    Returns:
        One row per run, model, task, backend and target with the selected
        threshold (blank when no threshold meets the target), the accepted count,
        coverage and accepted error rate on both splits, the evaluation deferred
        error rate, and the overall error rate and error-detection AUROC; sorted
        by model, task, backend and target.
    """
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
    """Return coverage and error rate after deferring the most uncertain predictions.

    For each fraction, the rounded-up share of rows with the highest
    `uncertainty_score` is deferred and the rest retained; rows with a blank or
    NaN score are left out entirely.

    Args:
        rows: Score rows of the task.
        task: Task of the rows; decides what counts as an error.
        defer_fractions: Shares of rows to defer.

    Returns:
        `selective_coverage_defer_<pct>` and `selective_error_defer_<pct>` per
        fraction, where `<pct>` is the fraction in percent; both are NaN without
        scored rows, and the error rate is NaN when nothing is retained.
    """
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


def cluster_ci_fields(
    rows: list[dict[str, Any]],
    metrics: Mapping[str, Callable[[list[dict[str, Any]]], float]],
    iterations: int = 1000,
    seed: int = 20260518,
    cluster_field: str = DEFAULT_BOOTSTRAP_CLUSTER_FIELD,
) -> dict[str, float | str]:
    """Run both cluster bootstraps for every named metric over one row slice.

    `{name}_ci_low` / `{name}_ci_high` are the primary interval, clustered
    on `cluster_field` (the provider request by default).
    `{name}_seed_ci_low` / `{name}_seed_ci_high` are the seed-clustered
    pair, reported alongside so a reader can see how much of the width comes
    from treating the request rather than the seed as independent.
    `bootstrap_ci_cluster_field` records the field the primary interval
    actually used; when it is already the seed the two intervals coincide and
    only one bootstrap runs. It is emitted even when there are no metrics, so a
    task with no headline risk still says which unit its slice resolved to
    rather than carrying a blank cell that reads like a missing column.

    Args:
        rows: Rows of the slice.
        metrics: Metric functions over a list of rows, keyed by the name that
            prefixes their fields.
        iterations: Number of bootstrap replicates per interval.
        seed: Seed of the resampling generator.
        cluster_field: Preferred primary cluster field, resolved with
            `resolve_bootstrap_cluster_field`.

    Returns:
        `bootstrap_ci_cluster_field` and the four bounds of every metric.
    """
    resolved_field = resolve_bootstrap_cluster_field(rows, cluster_field)
    if not metrics:
        return {"bootstrap_ci_cluster_field": resolved_field}
    fields: dict[str, float | str] = {"bootstrap_ci_cluster_field": resolved_field}
    for name, metric in metrics.items():
        _, low, high = bootstrap_seed_metric(
            rows,
            metric,
            cluster_field=resolved_field,
            iterations=iterations,
            seed=seed,
        )
        fields[f"{name}_ci_low"] = low
        fields[f"{name}_ci_high"] = high
        seed_low, seed_high = low, high
        if resolved_field != BOOTSTRAP_CLUSTER_FALLBACK_FIELD:
            _, seed_low, seed_high = bootstrap_seed_metric(
                rows,
                metric,
                cluster_field=BOOTSTRAP_CLUSTER_FALLBACK_FIELD,
                iterations=iterations,
                seed=seed,
            )
        fields[f"{name}_seed_ci_low"] = seed_low
        fields[f"{name}_seed_ci_high"] = seed_high
    return fields


def headline_risk_ci_fields(
    rows: list[dict[str, Any]],
    task: str,
    iterations: int = 1000,
    seed: int = 20260518,
) -> dict[str, float | str]:
    """Return request- and seed-clustered CI bounds for the Task 1/2 headline risks.

    The headline risks, each at the 0.80 and 0.90 confidence thresholds, are
    unsupported mandatory acceptance for Task 1, and over-commitment among
    over-committable items, weak strengthening and label-correct text
    over-commitment for Task 2.

    Args:
        rows: Score rows of one slice.
        task: `task1` or `task2`; any other task has no headline risk and only
            reports the cluster field.
        iterations: Number of bootstrap replicates per interval.
        seed: Seed of the resampling generator.

    Returns:
        The fields of `cluster_ci_fields` for the task's headline risks.
    """
    metrics: dict[str, Callable[[list[dict[str, Any]]], float]] = {}
    for threshold in HIGH_CONFIDENCE_THRESHOLDS:
        suffix = f"{int(threshold * 100):02d}"
        if task == "task1":
            metrics[f"unsupported_mandatory_acceptance_{suffix}"] = (
                lambda sample_rows, threshold=threshold: (
                    unsupported_mandatory_acceptance_rate(sample_rows, threshold)
                )
            )
        elif task == "task2":
            metrics[f"high_conf_overcommit_overcommittable_{suffix}"] = (
                lambda sample_rows, threshold=threshold: (
                    task2_high_confidence_overcommitment_rate(
                        sample_rows,
                        threshold,
                        denominator="overcommittable",
                    )
                )
            )
            metrics[f"weak_strengthening_{suffix}"] = (
                lambda sample_rows, threshold=threshold: weak_strengthening_rate(
                    sample_rows, threshold
                )
            )
            metrics[f"label_correct_text_overcommit_{suffix}"] = (
                lambda sample_rows, threshold=threshold: (
                    label_correct_text_overcommit_rate(sample_rows, threshold)
                )
            )
    return cluster_ci_fields(rows, metrics, iterations=iterations, seed=seed)


def text_strengthening_rate(rows: list[dict[str, Any]], strict: bool = False) -> float:
    """Return the broad (or strict) generated-text strengthening rate over parsed rows.

    Args:
        rows: Score rows; only rows whose text modality parsed are counted.
        strict: Count `strict_text_overcommit` instead of `text_overcommit`.

    Returns:
        The rate, or NaN when no text modality parsed.
    """
    text_rows = [
        row for row in rows if str(row.get("text_modality_parse_status", "")) == "ok"
    ]
    if not text_rows:
        return math.nan
    key = "strict_text_overcommit" if strict else "text_overcommit"
    return sum(1 for row in text_rows if _truthy(row.get(key))) / len(text_rows)


def weak_intent_strict_strengthening_rate(
    rows: list[dict[str, Any]],
    threshold: float | None = None,
    kind: str | None = None,
) -> float:
    """Return strict text strengthening over readable weak stakeholder-intent rows.

    The denominator is the Task 2 rows whose *source* modality is `nice_to_have`
    and whose generated text yielded a modality at all
    (`text_modality_parse_status` is `ok`), which is the denominator the
    per-model RQ1 table prints. There is no sample-kind filter, but only
    deterministic verbalized-confidence rows get a parsed text modality
    (stochastic and rule-baseline score rows leave it blank), so the denominator
    holds deterministic rows only.

    `threshold` gates the numerator on verbalized confidence:
    `threshold=0.90` reproduces the high-confidence diagnostic
    `strict_text_high_conf_overcommit_90`. `None` counts all strict
    strengthening and supplies the primary weak-intent headline. Keep the
    two quantities distinct even if they coincide in a particular run.
    `kind` restricts the numerator to one `strict_text_overcommit_kind`
    (`STRICT_OVERCOMMIT_ESCALATION` or `STRICT_OVERCOMMIT_FRAME_ONLY`).

    Args:
        rows: Score rows.
        threshold: Minimum confidence of a counted row, or `None`.
        kind: Strict over-commitment kind to count, or `None` for both.

    Returns:
        The rate, or NaN without readable weak-intent rows.
    """
    weak_rows = [
        row
        for row in rows
        if str(row.get("task", "")) == "task2"
        and str(row.get("source_modality", "")) == "nice_to_have"
    ]
    readable = [
        row
        for row in weak_rows
        if str(row.get("text_modality_parse_status", "")) == "ok"
    ]
    if not readable:
        return math.nan
    numerator = sum(
        1
        for row in readable
        if _truthy(row.get("strict_text_overcommit"))
        and (threshold is None or float(row.get("confidence", 0.0) or 0.0) >= threshold)
        and (kind is None or str(row.get("strict_text_overcommit_kind", "")) == kind)
    )
    return numerator / len(readable)


def text_over_commitment_ci_fields(
    rows: list[dict[str, Any]],
    iterations: int = 1000,
    seed: int = 20260518,
) -> dict[str, float | str]:
    """Return the point estimate, counts, and both cluster bootstrap CIs for text drift.

    The primary interval resamples provider *requests* with replacement (via
    `cluster_ci_fields`): a request holds four whole seeds, each seed's
    four source-modality variants share a capability, and strict strengthening
    is all-or-none within a request, so neither the item nor the seed is the
    independent unit. The seed-clustered pair is reported alongside as
    `*_seed_ci_low` / `*_seed_ci_high`.

    Args:
        rows: Score rows; only Task 2 rows are used.
        iterations: Number of bootstrap replicates; 0 or less skips the intervals.
        seed: Seed of the resampling generator.

    Returns:
        For `text_over_commitment` and `strict_text_over_commitment`: the rate
        over parsed text, its numerator and denominator, and the four bounds,
        plus `bootstrap_ci_cluster_field`. The rate and bounds are blank without
        Task 2 rows, and the bounds are blank when the intervals are skipped.
    """
    fields: dict[str, float | str] = {}
    task2_rows = [row for row in rows if str(row.get("task", "")) == "task2"]
    text_rows = [
        row
        for row in task2_rows
        if str(row.get("text_modality_parse_status", "")) == "ok"
    ]
    metrics: dict[str, Callable[[list[dict[str, Any]]], float]] = {}
    for metric_name, strict in (
        ("text_over_commitment", False),
        ("strict_text_over_commitment", True),
    ):
        key = "strict_text_overcommit" if strict else "text_overcommit"
        numerator = sum(1 for row in text_rows if _truthy(row.get(key)))
        fields[f"{metric_name}_n_numerator"] = numerator
        fields[f"{metric_name}_n_denominator"] = len(text_rows)
        fields[metric_name] = (
            text_strengthening_rate(task2_rows, strict=strict) if task2_rows else ""
        )
        if not task2_rows or iterations <= 0:
            for bound in ("ci_low", "ci_high", "seed_ci_low", "seed_ci_high"):
                fields[f"{metric_name}_{bound}"] = ""
            continue
        metrics[metric_name] = lambda sample_rows, strict=strict: (
            text_strengthening_rate(sample_rows, strict=strict)
        )
    fields.update(
        cluster_ci_fields(task2_rows, metrics, iterations=iterations, seed=seed)
    )
    return fields


def task3_strengthening_recall(rows: list[dict[str, Any]]) -> float:
    """Return the share of gold-strengthening Task 3 audits predicted `strengthens`.

    Args:
        rows: Task 3 score rows.

    Returns:
        The recall, or NaN without gold-strengthening rows.
    """
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
    """Return the share of gold-strengthening Task 3 audits predicted `preserves`.

    Args:
        rows: Task 3 score rows.

    Returns:
        The rate, or NaN without gold-strengthening rows.
    """
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
    """Return the share of quoted evidence phrases that occur in the source statement.

    Args:
        rows: Task 3 score rows; rows without an evidence phrase are left out.

    Returns:
        The rate, or an empty string when no row quotes evidence.
    """
    evidence_rows = [row for row in rows if str(row.get("evidence_phrase", "")).strip()]
    if not evidence_rows:
        return ""
    return sum(
        1 for row in evidence_rows if _truthy(row.get("evidence_phrase_in_source"))
    ) / len(evidence_rows)


def task_accuracy(rows: list[dict[str, Any]], task: str) -> float:
    """Return the accuracy of a task's score rows.

    Args:
        rows: Score rows of the task.
        task: `task2` compares modalities, `task3` relations, and any other task
            `y_true` with `y_pred`.

    Returns:
        The accuracy, or NaN without rows.
    """
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
    """Return the labels of `label_order` that occur in a column.

    Args:
        rows: Rows to scan.
        field: Column holding the labels.
        label_order: Labels in canonical order.

    Returns:
        The present labels in canonical order, or the whole `label_order` when
        none occurs.
    """
    present = {str(row.get(field, "")) for row in rows}
    labels = [label for label in label_order if label in present]
    return labels or list(label_order)


def metric_summary_by_model_task_method(
    scores: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize score rows per model, task and UQ method.

    Every group gets accuracy, Brier score and ECE (on `p_yes` for Task 1, on the
    confidence otherwise), error-detection AUROC, selective deferral, the parse
    failure rate, repeated-sample agreement and the high-confidence
    over-commitment rates. Task 1 adds F1, AUROC, the correlations of `p_yes` with
    modality strength and the monotonicity diagnostics; Task 2 adds macro F1,
    over-commitment, weak recall and the text-modality and answer-length metrics;
    Task 3 adds macro F1 over the gold relations present, strengthening recall,
    the false-preserve rate and the evidence-phrase source rate. Metrics that do
    not apply to a task are blank.

    Args:
        scores: Score rows of any tasks and methods.

    Returns:
        One summary row per group, sorted by model, task and method.
    """
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
    """Format a metric value for a markdown table.

    Args:
        value: Metric value; blanks stay blank and non-numeric values are shown
            as text.
        digits: Number of decimal places.

    Returns:
        The formatted value, `NA` for NaN.
    """
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
    """Render rows as a markdown table with every value passed through `format_metric`.

    Args:
        rows: Rows to render.
        fields: Columns to show, in order; columns missing from the rows are blank.

    Returns:
        The markdown table, or `_No rows._` when there are no rows.
    """
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
    """Return the inventory of UQ methods the study uses, conditions or excludes.

    Returns:
        One row per method with its survey family, access requirement, extra
        inference cost, headline status and notes.
    """
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
            "notes": "Explicitly excluded from this study because it requires internal activations or model modifications.",
        },
    ]


def write_uq_method_inventory(
    output_dir: str | Path, suffix: str = ""
) -> dict[str, Path]:
    """Write the UQ method inventory as a markdown table and a CSV.

    Args:
        output_dir: Output directory; created when missing.
        suffix: Appended to both file names before the extension.

    Returns:
        The `markdown` and `csv` paths.
    """
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
    """Plot mean Task 1 `p_yes` by source modality, one line per model and UQ method.

    Without Task 1 scores the figure holds a placeholder message instead.

    Args:
        scores: Score rows; Task 1 rows with a `p_yes` are plotted.
        path: Output file; its suffix selects the image format (SVG when it has
            none) and its parent directories are created.
    """
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
