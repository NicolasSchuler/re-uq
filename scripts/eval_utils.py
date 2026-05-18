from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

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

DEFAULT_CONFIG: dict[str, Any] = {
    "project": {
        "seed": 20260518,
        "target_seed_count": 120,
        "pilot_seed_count": 20,
        "prompt_version": "v1",
    },
    "llm": {
        "host": "http://localhost:8000/v1",
        "models": ["local-model"],
        "api_key_env": "LOCAL_OPENAI_API_KEY",
        "timeout_s": 120,
        "max_tokens": 256,
        "deterministic": {"temperature": 0.0, "top_p": 1.0, "samples": 1},
        "stochastic": {"temperature": 0.7, "top_p": 1.0, "samples": 5},
    },
    "datasets": {
        "nice_url": "https://zenodo.org/records/14590935/files/PROMISE-relabeled-NICE.csv?download=1",
        "nice_local_path": "data/raw/PROMISE-relabeled-NICE.csv",
    },
}


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


def write_csv_rows(
    path: str | Path,
    rows: list[dict[str, Any]] | pd.DataFrame,
    fieldnames: list[str] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame.from_records(rows)
    if fieldnames is not None:
        for field in fieldnames:
            if field not in frame.columns:
                frame[field] = ""
        frame = frame.loc[:, fieldnames]
    frame.to_csv(path, index=False)


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
    normalized_prefix = None
    if prefix:
        normalized_prefix = prefix if prefix.endswith("-") else f"{prefix}-"
    selected: list[str] = []
    for row in raw_rows:
        run_id = str(row.get("run_id", "")).strip()
        if not run_id:
            continue
        if normalized_prefix and not run_id.startswith(normalized_prefix):
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
    return selected_run_id, [row for row in raw_rows if row.get("run_id") == selected_run_id]


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


def normalize_space(text: str) -> str:
    return SPACE_RE.sub(" ", (text or "").strip())


def strip_final_punctuation(text: str) -> str:
    return normalize_space(text).rstrip(" .;:")


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
        r"^(?:the\s+)?(?:system|software|application|app|product|platform|service|tool|interface)\s+"
        r"(?:shall|must|should|may|will|can|could)\s+(?:be\s+able\s+to\s+)?",
        re.I,
    ),
    re.compile(
        r"^(?:the\s+)?(?:system|software|application|app|product|platform|service|tool|interface)\s+"
        r"(?:is|are)\s+(?:required|expected|recommended)\s+to\s+",
        re.I,
    ),
    re.compile(
        r"^(?:users?|administrators?|admins?|customers?|stakeholders?)\s+"
        r"(?:shall|must|should|may|will|can|could)\s+(?:be\s+able\s+to\s+)?",
        re.I,
    ),
]


def auto_capability_text(requirement: str) -> str:
    text = strip_final_punctuation(normalize_space(requirement))
    text = re.sub(r"^[\-\*\d.)\s]+", "", text)
    for pattern in LEADING_REQUIREMENT_PATTERNS:
        text = pattern.sub("", text).strip()
    text = re.sub(r"^(?:be\s+able\s+to|able\s+to)\s+", "", text, flags=re.I)
    return lower_initial(text)


def automatic_filter(requirement: str, capability: str) -> tuple[bool, str]:
    reasons: list[str] = []
    wc = word_count(requirement)
    if wc < 5:
        reasons.append("too_short")
    if wc > 35:
        reasons.append("too_long")
    if len(SENTENCE_END_RE.findall(requirement)) > 1:
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
    return not reasons, ";".join(reasons)


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


def make_seed_candidates(dataset_rows: list[dict[str, str]], target_count: int = 120) -> list[dict[str, Any]]:
    text_column = find_requirement_text_column(dataset_rows)
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
        candidates.append(
            {
                "seed_id": f"S{len(candidates) + 1:04d}",
                "source_dataset": "NICE",
                "original_requirement": original,
                "capability_text_auto": capability,
                "auto_include": "yes" if auto_include else "no",
                "auto_exclusion_reason": reason,
                "include": "yes" if include else "no",
                "exclusion_reason": "" if include else reason,
                "capability_text_final": capability if include else "",
            }
        )
    return candidates


def is_truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "include"}


def load_reviewed_seeds(path: str | Path, target_count: int = 120, strict: bool = True) -> list[dict[str, Any]]:
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


def capability_clause(capability: str) -> str:
    return strip_final_punctuation(capability)


def source_statement(capability: str, modality: str) -> str:
    cap = capability_clause(capability)
    if modality == "mandatory":
        return f"The system MUST {cap}."
    if modality == "recommended":
        return f"The system SHOULD {cap}."
    if modality == "optional":
        return f"The system MAY {cap}."
    if modality == "nice_to_have":
        return f"It would be useful if the system could {cap}."
    raise ValueError(f"Unknown modality: {modality}")


def candidate_requirement(capability: str) -> str:
    return f"The system MUST {capability_clause(capability)}."


def build_benchmark_items(
    seed_rows: list[dict[str, Any]],
    numeric_strength: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    numeric_strength = numeric_strength or NUMERIC_STRENGTH_DEFAULT
    items: list[dict[str, Any]] = []
    for seed in seed_rows:
        capability = seed["capability_text_final"]
        for modality in MODALITIES:
            items.append(
                {
                    "item_id": f"{seed['seed_id']}_{modality}",
                    "seed_id": seed["seed_id"],
                    "source_dataset": seed.get("source_dataset", "NICE"),
                    "original_requirement": seed.get("original_requirement", ""),
                    "capability_text": capability,
                    "source_modality": modality,
                    "source_statement": source_statement(capability, modality),
                    "candidate_requirement": candidate_requirement(capability),
                    "task1_gold_decision": "yes" if modality == "mandatory" else "no",
                    "task1_gold_yes": 1 if modality == "mandatory" else 0,
                    "task2_gold_modality": modality,
                    "ordinal_strength": ORDINAL_STRENGTH[modality],
                    "numeric_strength": numeric_strength[modality],
                }
            )
    return items


def load_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def render_prompt(template: str, **values: Any) -> str:
    return template.format(**values)


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

    raise ValueError(f"Unknown task: {task}")


def chat_completion(
    host: str,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int = 256,
    timeout_s: int = 120,
    api_key_env: str = "LOCAL_OPENAI_API_KEY",
) -> dict[str, Any]:
    api_key = os.getenv(api_key_env, "EMPTY")
    client = OpenAI(base_url=host.rstrip("/") + "/", api_key=api_key, timeout=timeout_s)
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
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
) -> dict[str, Any]:
    parsed_json, parse_status = parse_task_response(task, completion.get("raw_text", ""))
    if not completion.get("ok"):
        parse_status = "request_error"
    return {
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


def auroc_score(y_true: list[int], probabilities: list[float]) -> float:
    if not y_true or len(set(y_true)) < 2:
        return math.nan
    try:
        return float(roc_auc_score(y_true, probabilities))
    except ValueError:
        return math.nan


def monotonicity_violation_rate(rows: list[dict[str, Any]], score_field: str = "p_yes") -> float:
    frame = pd.DataFrame.from_records(rows)
    if frame.empty or score_field not in frame.columns:
        return math.nan
    frame = frame[frame[score_field] != ""].copy()
    if frame.empty:
        return math.nan
    frame[score_field] = pd.to_numeric(frame[score_field])
    frame["_modality_sort"] = frame["source_modality"].map(lambda value: -ORDINAL_STRENGTH[str(value)])
    checked = 0
    violated = 0
    for _, seed_frame in frame.groupby("seed_id", sort=False):
        if len(seed_frame) < len(MODALITIES):
            continue
        ordered = seed_frame.sort_values("_modality_sort")
        scores = ordered[score_field].to_numpy(dtype=float)
        checked += 1
        if bool(np.any(np.diff(scores) > 1e-12)):
            violated += 1
    return math.nan if checked == 0 else violated / checked


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


def build_uq_scores(benchmark_rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    benchmark_by_item = {row["item_id"]: row for row in benchmark_rows}
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
            scores.append(
                {
                    **base,
                    "y_true": int(item["task1_gold_yes"]),
                    "y_pred": pred_yes,
                    "p_yes": p_yes,
                    "confidence": confidence,
                    "gold_modality": "",
                    "pred_modality": "",
                }
            )
        elif raw["task"] == "task2":
            pred_modality = parsed["modality"]
            correct = 1 if pred_modality == item["task2_gold_modality"] else 0
            scores.append(
                {
                    **base,
                    "y_true": correct,
                    "y_pred": correct,
                    "p_yes": "",
                    "confidence": confidence,
                    "gold_modality": item["task2_gold_modality"],
                    "pred_modality": pred_modality,
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
        base = score_base(valid[0], item, "label_self_consistency" if task == "task1" else "modality_consistency", len(valid), len(group))
        if task == "task1":
            yes_count = sum(1 for row in valid if row["parsed_json"]["decision"] == "yes")
            p_yes = yes_count / len(valid)
            pred_yes = 1 if p_yes >= 0.5 else 0
            scores.append(
                {
                    **base,
                    "y_true": int(item["task1_gold_yes"]),
                    "y_pred": pred_yes,
                    "p_yes": p_yes,
                    "confidence": max(p_yes, 1 - p_yes),
                    "gold_modality": "",
                    "pred_modality": "",
                }
            )
        elif task == "task2":
            counts = Counter(row["parsed_json"]["modality"] for row in valid)
            pred_modality = sorted(counts.items(), key=lambda pair: (-pair[1], MODALITIES.index(pair[0])))[0][0]
            confidence = counts[pred_modality] / len(valid)
            correct = 1 if pred_modality == item["task2_gold_modality"] else 0
            scores.append(
                {
                    **base,
                    "y_true": correct,
                    "y_pred": correct,
                    "p_yes": "",
                    "confidence": confidence,
                    "gold_modality": item["task2_gold_modality"],
                    "pred_modality": pred_modality,
                }
            )
    return scores


def score_base(raw: dict[str, Any], item: dict[str, Any], uq_method: str, valid_n: int, total_n: int) -> dict[str, Any]:
    return {
        "run_id": raw.get("run_id", ""),
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


def overcommitment_metrics(rows: list[dict[str, Any]]) -> tuple[float, float, float]:
    if not rows:
        return math.nan, math.nan, math.nan
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
    return over / total, under / total, severity_sum / total


def calibration_probabilities(rows: list[dict[str, Any]], task: str) -> list[float]:
    if task == "task1":
        return [float(row["p_yes"]) for row in rows]
    return [float(row["confidence"]) for row in rows]


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
            "accuracy": accuracy_score(y_true, y_pred),
            "brier": brier_score(y_true, calibration_scores),
            "ece": ece_score(y_true, calibration_scores),
            "parse_failure_rate": group_frame["parse_failures"].astype(int).sum()
            / max(1, int(group_frame["total_n"].astype(int).sum())),
        }
        if task == "task1":
            p_yes = group_frame["p_yes"].astype(float).tolist()
            summary.update(
                {
                    "f1_or_macro_f1": binary_f1_score(y_true, y_pred),
                    "auroc": auroc_score(y_true, p_yes),
                    "spearman_modality_p_yes": spearman_corr(group_frame["numeric_strength"].astype(float).tolist(), p_yes),
                    "monotonicity_violations": monotonicity_violation_rate(rows, "p_yes"),
                    "over_commitment": "",
                    "under_commitment": "",
                    "over_commitment_severity": "",
                }
            )
        elif task == "task2":
            gold = group_frame["gold_modality"].astype(str).tolist()
            pred = group_frame["pred_modality"].astype(str).tolist()
            over, under, severity = overcommitment_metrics(rows)
            summary.update(
                {
                    "f1_or_macro_f1": macro_f1_score(gold, pred, MODALITIES),
                    "auroc": "",
                    "spearman_modality_p_yes": "",
                    "monotonicity_violations": "",
                    "over_commitment": over,
                    "under_commitment": under,
                    "over_commitment_severity": severity,
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
