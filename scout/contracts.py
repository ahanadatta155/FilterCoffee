"""Shared, JSON-serializable extraction and telemetry contracts.

These enforce shape and bounds. Source/evidence verification belongs to the
pipeline validator, not these transport models.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

FieldType = Literal["string", "number", "integer", "boolean", "date", "url", "list[string]"]
Confidence = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
NonnegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
NonnegativeInt = Annotated[int, Field(ge=0)]
SourceState = Literal["ok", "empty", "unavailable", "error"]


class FieldSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
    type: FieldType
    description: str
    required: bool = False


class ExtractionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["single", "list"]
    fields: list[FieldSpec] = Field(min_length=1, max_length=12)
    item_hint: str | None = None
    notes: str = ""

    @field_validator("fields")
    @classmethod
    def unique_names(cls, fields: list[FieldSpec]) -> list[FieldSpec]:
        if len({field.name for field in fields}) != len(fields):
            raise ValueError("Field names must be unique.")
        return fields


class Evidence(BaseModel):
    field: str
    quote: str | None = None
    selector: str | None = None


class ExtractedItem(BaseModel):
    data: dict[str, Any]
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: Confidence = 0.0
    model_confidence: Confidence | None = None


class ModelCall(BaseModel):
    stage: str
    model: str
    prompt_tokens: NonnegativeInt
    completion_tokens: NonnegativeInt
    latency_ms: NonnegativeInt
    cost_usd: NonnegativeFloat
    ok: bool


class SourceReport(BaseModel):
    """Per-source outcome for one research run."""

    name: str
    status: SourceState
    hits: NonnegativeInt = 0
    latency_ms: NonnegativeInt = 0
    warnings: list[str] = Field(default_factory=list)


class RunMeta(BaseModel):
    run_id: str
    url: str
    fetch_tier: str
    status_code: int | None = None
    page_title: str | None = None
    model_calls: list[ModelCall] = Field(default_factory=list)
    prompt_tokens: NonnegativeInt = 0
    completion_tokens: NonnegativeInt = 0
    estimated_cost_usd: NonnegativeFloat = 0.0
    latency_ms: NonnegativeInt = 0
    cache_hit: bool = False
    warnings: list[str] = Field(default_factory=list)
    sources: list[SourceReport] = Field(default_factory=list)


class ExtractResult(BaseModel):
    url: str
    instruction: str
    plan: ExtractionPlan
    items: list[ExtractedItem]
    meta: RunMeta
    raw_text_preview: str = ""
