"""Pydantic response models for strict provider structured-output paths."""

from __future__ import annotations

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
        if isinstance(value, bool) or isinstance(value, str):
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


class Task2BatchItem(Task2Response):
    request_index: int


class Task2BatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    results: list[Task2BatchItem]


class Task3Response(StrictResponseModel):
    relation: RelationLabel
    confidence: Confidence01
    evidence_phrase: str = Field(max_length=240)
    brief_reason: str = Field(default="", max_length=240)


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


def validated_payload_for_task(task: str, payload: Any, *, batched: bool = False) -> dict[str, Any]:
    model = response_model_for_task(task, batched=batched)
    return model.model_validate(payload).model_dump(mode="json")


def validated_json_for_task(task: str, text: str, *, batched: bool = False) -> dict[str, Any]:
    model = response_model_for_task(task, batched=batched)
    return model.model_validate_json(text).model_dump(mode="json")
