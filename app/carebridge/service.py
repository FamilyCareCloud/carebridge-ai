import re
from uuid import uuid4

from app.carebridge.language import korean_output
from app.carebridge.provider import InvalidModelOutput, TranslationProvider
from app.carebridge.safety import assess, check_input, digest, lexical_signals
from app.carebridge.schemas import (
    AdditionalCheck,
    Draft,
    DraftRequest,
    ModelDraft,
    ModelSegment,
    Observation,
    QAResult,
    Source,
    Translation,
    TranslationSegment,
)


def is_korean(text: str) -> bool:
    # A Korean note with foreign words still needs translation. Do not silently
    # copy mixed-language notes into a supposedly Korean translation.
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and all("가" <= c <= "힣" or "ㄱ" <= c <= "ㅣ" for c in letters)


def korean_category(text: str):
    # Labels only: preserve every source sentence verbatim, with no inference.
    for category, pattern in [
        ("meal", r"식사|아침|점심|저녁|밥|간식"),
        ("hydration", r"물[을은이도 ]|수분|음료"),
        ("mobility", r"걷|걸었|보행|산책|이동"),
        ("sleep", r"수면|잠|주무"),
        ("hygiene", r"목욕|세면|양치|씻"),
        ("medication", r"복약|복용|약[을은이 ]"),
    ]:
        if re.search(pattern, text):
            return category
    return "other"


class DraftService:
    def __init__(self, provider: TranslationProvider):
        self.provider = provider

    async def create(self, request: DraftRequest, *, audio: bool = False) -> Draft:
        segments = check_input(request.memo)
        local = is_korean(request.memo) and request.source_language.split("-")[0] in {"auto", "ko"}
        if local:
            candidate = ModelDraft(
                segments=[
                    ModelSegment(
                        segment_id=s.segment_id,
                        translation=s.quote,
                        category=korean_category(s.quote),
                        needs_confirmation=bool(
                            re.search(r"불명|모르|같[다아습은]|아마|확인|추정|\?", s.quote)
                        ),
                    )
                    for s in segments
                ]
            )
        else:
            candidate = await self.provider.generate(segments, request.source_language)
        if [s.segment_id for s in candidate.segments] != [s.segment_id for s in segments]:
            raise InvalidModelOutput("초안의 원문 구간이 누락·중복·변경되었습니다.")
        if any(not korean_output(s.translation) for s in candidate.segments):
            raise InvalidModelOutput(
                "모든 원문 구간에 외국어 문장이 남지 않은 한국어 번역이 필요합니다."
            )
        translations = []
        observations = []
        additional_checks = []
        for model_segment, evidence in zip(candidate.segments, segments, strict=True):
            translations.append(
                TranslationSegment(text=model_segment.translation, evidence=evidence)
            )
            observations.append(
                Observation(
                    id=f"o{len(observations) + 1:03d}",
                    category=model_segment.category,
                    text=model_segment.translation,
                    evidence=evidence,
                )
            )
            if model_segment.needs_confirmation or lexical_signals(evidence.quote)[1]:
                additional_checks.append(
                    AdditionalCheck(id=f"c{len(additional_checks) + 1:03d}", evidence=evidence)
                )
        draft = Draft(
            id=str(uuid4()),
            status="review_required",
            mode="source_preserving" if local else "model_translation",
            source=Source(
                kind="audio" if audio else "text",
                language="ko" if local else request.source_language,
                text=request.memo,
                sha256=digest(request.memo),
            ),
            translation=Translation(
                text="\n".join(t.text for t in translations), segments=translations
            ),
            observations=observations,
            additional_checks=additional_checks,
            qa=QAResult(checks=[], issues=[]),
        )
        draft.qa = assess(draft)
        draft.status = (
            "hold" if any(i.severity == "hold" for i in draft.qa.issues) else "review_required"
        )
        return draft
