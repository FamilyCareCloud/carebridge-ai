"""Stateless source bindings and human edits; persistence/authorization belong to the backend."""

import json
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import Field, StringConstraints

from app.carebridge.provider import translation_prompt_material
from app.carebridge.safety import assess, check_input, digest
from app.carebridge.schemas import (
    Category,
    Contract,
    Draft,
    DraftRequest,
    QAResult,
    SourceSegment,
    Text,
)
from app.carebridge.semantic import AUDIT_PROMPT, SemanticReview
from app.config import Settings

WORKFLOW_VERSION = "carebridge.workflow.v1"
QA_REVISION = "carebridge.qa.v2"
SHA256 = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
RequestId = Annotated[
    str,
    StringConstraints(
        pattern=r"^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$"
    ),
]
Revision = Annotated[int, Field(ge=1, le=9007199254740991)]


class WorkflowConflict(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class Pipeline(Contract):
    qa_revision: Literal["carebridge.qa.v2"] = QA_REVISION
    translation_backend: Literal["local", "http"]
    translation_model: str
    translation_prompt_sha256: SHA256
    semantic_prompt_sha256: SHA256


def pipeline_info(config: Settings) -> Pipeline:
    return Pipeline(
        translation_backend=config.carebridge_backend,
        translation_model=(
            config.carebridge_local_model
            if config.carebridge_backend == "local"
            else config.carebridge_model
        ),
        translation_prompt_sha256=digest(translation_prompt_material()),
        semantic_prompt_sha256=digest(AUDIT_PROMPT),
    )


class GenerationRequest(DraftRequest):
    request_id: RequestId
    source_revision: Revision
    # This value attests that the author already checked the transcription.
    input_kind: Literal["text", "reviewed_transcript"] = "text"


class Generation(Contract):
    contract_version: Literal["carebridge.workflow.v1"] = WORKFLOW_VERSION
    origin: Literal["ai_generated"] = "ai_generated"
    request_id: RequestId
    source_revision: Revision
    input_kind: Literal["text", "reviewed_transcript"]
    draft_sha256: SHA256
    draft: Draft
    pipeline: Pipeline


def draft_digest(draft: Draft) -> str:
    # Canonical JSON of the validated Draft, not a signature or an authorization token.
    return digest(
        json.dumps(
            draft.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    )


def generation_result(request: GenerationRequest, draft: Draft, config: Settings) -> Generation:
    return Generation(
        request_id=request.request_id,
        source_revision=request.source_revision,
        input_kind=request.input_kind,
        draft_sha256=draft_digest(draft),
        draft=draft,
        pipeline=pipeline_info(config),
    )


class EditedSegment(Contract):
    segment_id: Annotated[str, StringConstraints(pattern=r"^s\d{3}$")]
    text: Text
    category: Category


class EditRequest(Contract):
    data_use: Literal["synthetic_deidentified"]
    request_id: RequestId
    generation: Generation
    # The backend must supply these from its current stored source, not the browser.
    expected_source_revision: Revision
    expected_source_sha256: SHA256
    editor: Literal["author", "reviewer"]
    segments: list[EditedSegment] = Field(min_length=1, max_length=40)


class SegmentChange(Contract):
    evidence: SourceSegment
    before: Text
    after: Text
    category_before: Category
    category_after: Category


class EditReview(Contract):
    contract_version: Literal["carebridge.workflow.v1"] = WORKFLOW_VERSION
    origin: Literal["author", "reviewer"]
    request_id: RequestId
    generation_id: str
    original_draft_sha256: SHA256
    source_revision: Revision
    source_sha256: SHA256
    draft_sha256: SHA256
    draft: Draft
    changes: list[SegmentChange]
    review: SemanticReview
    pipeline: Pipeline


def prepare_edit(request: EditRequest) -> tuple[Draft, list[SegmentChange]]:
    original = request.generation.draft
    if (
        request.expected_source_revision != request.generation.source_revision
        or request.expected_source_sha256 != original.source.sha256
    ):
        raise WorkflowConflict(
            "SOURCE_REVISION_CONFLICT",
            "원문이 변경되었습니다. 현재 원문으로 초안을 다시 생성해 주세요.",
        )
    if draft_digest(original) != request.generation.draft_sha256:
        raise WorkflowConflict(
            "GENERATION_SNAPSHOT_CONFLICT",
            "생성 원본이 변경되었습니다. 보관한 생성 원본을 사용해 주세요.",
        )
    check_input(original.source.text)
    structural = {
        "SOURCE_INTEGRITY",
        "SOURCE_COVERAGE",
        "EVIDENCE_PRESERVED",
        "TRANSLATION_CONSISTENCY",
        "OBSERVATION_GROUNDING",
        "FOLLOW_UP_GROUNDING",
    }
    if any(not c.passed for c in assess(original).checks if c.code in structural):
        raise WorkflowConflict(
            "INVALID_GENERATION_EVIDENCE", "생성 원본의 원문·근거 연결을 확인할 수 없습니다."
        )
    if [s.segment_id for s in request.segments] != [
        s.evidence.segment_id for s in original.translation.segments
    ]:
        raise WorkflowConflict(
            "EDIT_SEGMENTS_MISMATCH",
            "원문 구간의 추가·삭제·재정렬은 지원하지 않습니다. 원문을 수정하고 다시 생성해 주세요.",
        )
    # A human edit is a new draft snapshot. Never mutate the preserved generation.
    edited = original.model_copy(deep=True)
    edited.id = str(uuid4())
    edited.qa = QAResult(checks=[], issues=[])
    edited.status = "review_required"
    changes = []
    for change, translated, observation in zip(
        request.segments, edited.translation.segments, edited.observations, strict=True
    ):
        if change.text != translated.text or change.category != observation.category:
            changes.append(
                SegmentChange(
                    evidence=translated.evidence.model_copy(deep=True),
                    before=translated.text,
                    after=change.text,
                    category_before=observation.category,
                    category_after=change.category,
                )
            )
        translated.text = observation.text = change.text
        observation.category = change.category
    edited.translation.text = "\n".join(s.text for s in edited.translation.segments)
    return edited, changes


def edit_result(
    request: EditRequest,
    draft: Draft,
    changes: list[SegmentChange],
    review: SemanticReview,
    config: Settings,
) -> EditReview:
    draft.qa, draft.status = review.qa, review.status
    return EditReview(
        origin=request.editor,
        request_id=request.request_id,
        generation_id=request.generation.draft.id,
        original_draft_sha256=request.generation.draft_sha256,
        source_revision=request.generation.source_revision,
        source_sha256=draft.source.sha256,
        draft_sha256=draft_digest(draft),
        draft=draft,
        changes=changes,
        review=review,
        pipeline=pipeline_info(config),
    )
