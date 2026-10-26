from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Language = Annotated[str, StringConstraints(pattern=r"^(auto|[a-z]{2,3}(-[A-Za-z]{2,8})?)$")]
Category = Literal[
    "meal", "hydration", "mobility", "sleep", "hygiene", "medication", "condition", "other"
]
Text = Annotated[str, StringConstraints(min_length=1, max_length=4000)]
CHECK_QUESTION = "이 관찰의 의미와 빠진 정보를 원문 작성자에게 확인해 주세요."


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class DraftRequest(Contract):
    memo: Text
    source_language: Language = "auto"
    # Required attestation, not a claim that heuristic PII detection is complete.
    data_use: Literal["synthetic_deidentified"]


class SourceSegment(Contract):
    segment_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: Text


class ModelSegment(Contract):
    segment_id: str
    translation: Text
    category: Category
    needs_confirmation: bool


class ModelDraft(Contract):
    segments: list[ModelSegment] = Field(min_length=1, max_length=40)


class Source(Contract):
    kind: Literal["text", "audio"]
    language: Language
    text: Text
    sha256: str


class TranslationSegment(Contract):
    text: Text
    evidence: SourceSegment


class Translation(Contract):
    language: Literal["ko"] = "ko"
    text: str = Field(max_length=160000)
    segments: list[TranslationSegment] = Field(max_length=40)


class Observation(Contract):
    id: str
    category: Category
    text: Text
    evidence: SourceSegment


class AdditionalCheck(Contract):
    id: str
    question: Literal[CHECK_QUESTION] = CHECK_QUESTION
    evidence: SourceSegment


class QAIssue(Contract):
    code: str
    severity: Literal["hold", "review"]
    message: str


class QACheck(Contract):
    code: str
    passed: bool


class QAResult(Contract):
    checks: list[QACheck]
    issues: list[QAIssue]
    requires_human_review: Literal[True] = True
    external_share_allowed: Literal[False] = False


class Draft(Contract):
    schema_version: Literal["carebridge.v1"] = "carebridge.v1"
    id: str
    status: Literal["review_required", "hold"]
    label: Literal["검토 전 초안 · 공식기록 아님"] = "검토 전 초안 · 공식기록 아님"
    mode: Literal["source_preserving", "model_translation"]
    source: Source
    translation: Translation
    observations: list[Observation] = Field(max_length=40)
    additional_checks: list[AdditionalCheck] = Field(max_length=40)
    qa: QAResult
