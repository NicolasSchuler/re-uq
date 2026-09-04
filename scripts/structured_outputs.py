"""Pydantic response models for strict provider structured-output paths.

The models below are the single definition of every task response contract.
:func:`provider_json_schema` derives the strict provider JSON Schema from them,
so the schema a request carries and the schema Instructor validates against
cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

CONFIDENCE_SCALE_0_1 = "0_1"
INSTRUCTOR_OUTPUT_CONTRACT_VERSION = "instructor_v2_confidence_0_1"
PROMPT_OUTPUT_CONTRACT_VERSION = "prompt_v2_confidence_0_1"
CONFIDENCE_0_1_OUTPUT_CONTRACT_VERSIONS = {
    INSTRUCTOR_OUTPUT_CONTRACT_VERSION,
    PROMPT_OUTPUT_CONTRACT_VERSION,
}
INSTRUCTOR_CONFIDENCE_SCALE = CONFIDENCE_SCALE_0_1

ModalityLabel = Literal["mandatory", "recommended", "optional", "nice_to_have"]
DecisionLabel = Literal["yes", "no"]
RelationLabel = Literal["preserves", "strengthens", "weakens", "content_changed"]
Confidence01 = Annotated[float, Field(ge=0.0, le=1.0)]


class StrictResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("confidence", check_fields=False, mode="before")
    @classmethod
    def reject_non_numeric_confidence(cls, value: Any) -> Any:
        if isinstance(value, (bool, str)):
            raise ValueError("confidence must be a numeric decimal from 0.0 to 1.0")
        return value


class Task1Response(StrictResponseModel):
    decision: DecisionLabel
    confidence: Confidence01
    brief_reason: str = Field(max_length=200)


class Task1BatchItem(Task1Response):
    request_index: int


class Task1BatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    results: list[Task1BatchItem]


class Task2Response(StrictResponseModel):
    requirement: str
    modality: ModalityLabel
    confidence: Confidence01

    @field_validator("requirement")
    @classmethod
    def reject_blank_requirement(cls, value: str) -> str:
        """A blank extraction is a failed extraction, not an empty one.

        ``Field(min_length=1)`` would say the same thing declaratively, but it
        would also add ``"minLength"`` to the derived provider schema and
        change bytes that every cached request fingerprint depends on. Enforce
        it after validation instead, and keep the schema as it has always been.
        """
        if not value.strip():
            raise ValueError("requirement must not be blank or whitespace only")
        return value


class Task2BatchItem(Task2Response):
    request_index: int


class Task2BatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    results: list[Task2BatchItem]


class Task3Response(StrictResponseModel):
    relation: RelationLabel
    confidence: Confidence01
    evidence_phrase: str = Field(max_length=240)
    # Required, not defaulted: the provider schema has always listed
    # brief_reason in `required`, so a response that omits it is one the strict
    # path rejects. The tolerant parser still defaults it, but records that as
    # a repair (see eval_utils.parse_task_response).
    brief_reason: str = Field(max_length=240)


class Task3BatchItem(Task3Response):
    request_index: int


class Task3BatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    results: list[Task3BatchItem]


class ExternalTask2Response(Task2Response):
    external_item_id: str


TASK_RESPONSE_MODELS: dict[str, type[BaseModel]] = {
    "task1": Task1Response,
    "task2": Task2Response,
    "task3": Task3Response,
}

TASK_BATCH_RESPONSE_MODELS: dict[str, type[BaseModel]] = {
    "task1": Task1BatchResponse,
    "task2": Task2BatchResponse,
    "task3": Task3BatchResponse,
}


def response_model_for_task(task: str, *, batched: bool = False) -> type[BaseModel]:
    models = TASK_BATCH_RESPONSE_MODELS if batched else TASK_RESPONSE_MODELS
    try:
        return models[task]
    except KeyError as exc:
        raise ValueError(f"Unsupported task for structured output: {task}") from exc


def validated_payload_for_task(
    task: str, payload: Any, *, batched: bool = False
) -> dict[str, Any]:
    model = response_model_for_task(task, batched=batched)
    return model.model_validate(payload).model_dump(mode="json")


def validated_json_for_task(
    task: str, text: str, *, batched: bool = False
) -> dict[str, Any]:
    model = response_model_for_task(task, batched=batched)
    return model.model_validate_json(text).model_dump(mode="json")


# =============================================================================
# Provider JSON Schema derivation
# =============================================================================
# The `json_schema` response_format sent on the strict path used to be written
# out by hand next to the request builder, which let it drift from the models
# above (it required Task 3's brief_reason while the model defaulted it).
# Deriving it from the same models makes that drift impossible.
#
# Every archived run is resumed by request fingerprint, so the derived schema
# must serialize to exactly the bytes the handwritten one did. Those bytes are
# pinned for all six single/batched contracts in tests/test_structured_outputs.py;
# the constants below exist to reproduce them, not to express a preference.

#: Key order the handwritten provider schemas used, outermost concept first.
PROVIDER_SCHEMA_KEY_ORDER = (
    "type",
    "enum",
    "minimum",
    "maximum",
    "maxLength",
    "items",
    "properties",
    "required",
    "additionalProperties",
)
#: Pydantic bookkeeping a provider schema does not need. `default` only ever
#: appears on an optional field, and no field of these contracts is optional.
PROVIDER_SCHEMA_DROPPED_KEYS = frozenset({"$defs", "default", "description", "title"})
#: Batched contracts lead each item with the batch key; the models declare it
#: last because the batch item inherits the task fields from the single model.
PROVIDER_SCHEMA_LEADING_PROPERTIES = ("request_index",)


def _provider_schema_bound(value: Any) -> Any:
    """`ge=0.0` renders as `0.0`; the sent contract has always carried `0`."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _provider_schema_field_order(names: Iterable[str]) -> list[str]:
    ordered = list(names)
    leading = [name for name in PROVIDER_SCHEMA_LEADING_PROPERTIES if name in ordered]
    return leading + [name for name in ordered if name not in leading]


def _provider_subschema(
    node: Mapping[str, Any], defs: Mapping[str, Any]
) -> dict[str, Any]:
    ref = node.get("$ref")
    if ref is not None:
        siblings = {key: value for key, value in node.items() if key != "$ref"}
        node = {**defs[ref.rsplit("/", 1)[-1]], **siblings}

    resolved: dict[str, Any] = {}
    for key, value in node.items():
        if key in PROVIDER_SCHEMA_DROPPED_KEYS:
            continue
        if key == "properties":
            resolved[key] = {
                name: _provider_subschema(value[name], defs)
                for name in _provider_schema_field_order(value)
            }
        elif key == "items":
            resolved[key] = _provider_subschema(value, defs)
        elif key == "required":
            resolved[key] = _provider_schema_field_order(value)
        elif key in {"minimum", "maximum"}:
            resolved[key] = _provider_schema_bound(value)
        else:
            resolved[key] = value

    unplaced = sorted(set(resolved) - set(PROVIDER_SCHEMA_KEY_ORDER))
    if unplaced:
        # Silently appending an unknown key would move the serialized bytes of
        # a live contract, so refuse instead: decide where it belongs, then
        # re-pin the schema bytes in the tests.
        raise ValueError(
            "No provider-contract position for JSON Schema keys "
            f"{unplaced}; extend PROVIDER_SCHEMA_KEY_ORDER or "
            "PROVIDER_SCHEMA_DROPPED_KEYS and re-pin the schema bytes."
        )
    return {key: resolved[key] for key in PROVIDER_SCHEMA_KEY_ORDER if key in resolved}


def provider_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Strict provider JSON Schema for `model`, with `$defs`/`$ref` inlined."""
    schema = model.model_json_schema()
    return _provider_subschema(schema, schema.get("$defs", {}))


def provider_schema_for_task(task: str, *, batched: bool = False) -> dict[str, Any]:
    """Strict provider JSON Schema for a task's single or batched contract."""
    return provider_json_schema(response_model_for_task(task, batched=batched))
