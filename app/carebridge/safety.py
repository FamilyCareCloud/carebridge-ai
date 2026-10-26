"""Conservative structural checks, never a clinical or semantic accuracy score."""

import hashlib
import re
from collections import Counter

from app.carebridge.schemas import (
    CHECK_QUESTION,
    Draft,
    QACheck,
    QAIssue,
    QAResult,
    SourceSegment,
)

# Reject likely direct identifiers before any external model call. These are
# deliberately bounded heuristics; passing does not prove deidentification.
IDENTIFIERS = [
    re.compile(r"(?<!\d)\d{6}[- ]?[1-8]\d{6}(?!\d)"),
    re.compile(r"(?<!\d)(?:\+82[- .]?)?0?1[016789][- .]?\d{3,4}[- .]?\d{4}(?!\d)"),
    re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    re.compile(r"(?:이름|성명|수급자명|환자명|주소|생년월일)\s*[:：]\s*\S+"),
    re.compile(r"\b(?:patient\s*name|full\s*name|name|address|date\s*of\s*birth)\s*:\s*\S+", re.I),
]
MEDICAL = re.compile(
    r"진단|처방|응급|119|치매(?:입니다|이다)|복용량.{0,10}(?:늘|줄|변경)|"
    r"(?:약|복용).{0,8}(?:중단하|끊으|늘리|줄이)|"
    r"\b(?:diagnos\w*|prescrib\w*|emergency|increase\s+(?:the\s+)?dose|stop\s+(?:the\s+)?medication)\b",
    re.I,
)
INSTRUCTIONS = re.compile(
    r"ignore.{0,50}(?:instruction|previous)|system\s*prompt|"
    r"이전.{0,15}지시.{0,15}무시|자동.{0,5}(?:승인|공유)|검토.{0,5}(?:생략|우회)",
    re.I,
)


class InputRejected(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def has_identifiers(text: str) -> bool:
    return any(pattern.search(text) for pattern in IDENTIFIERS)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_segments(text: str) -> list[SourceSegment]:
    # Keep original Unicode code-point offsets; never normalize or rewrite input.
    segments = []
    for line in re.finditer(r"[^\r\n]+", text):
        value = line.group()
        if not value.strip():
            continue
        start = line.start() + len(value) - len(value.lstrip())
        end = line.end() - (len(value) - len(value.rstrip()))
        segments.append(
            SourceSegment(
                segment_id=f"s{len(segments) + 1:03d}",
                start=start,
                end=end,
                quote=text[start:end],
            )
        )
    return segments


def check_input(text: str) -> list[SourceSegment]:
    segments = source_segments(text)
    if not segments:
        raise InputRejected("EMPTY_MEMO", "관찰 메모를 입력해 주세요.")
    if len(segments) > 40:
        raise InputRejected("TOO_MANY_SEGMENTS", "짧은 관찰 메모를 40줄 이내로 입력해 주세요.")
    if has_identifiers(text):
        raise InputRejected(
            "IDENTIFYING_INFORMATION",
            "식별 가능 정보가 감지되었습니다. 교육용 가상·비식별 메모만 입력해 주세요.",
        )
    if INSTRUCTIONS.search(text):
        raise InputRejected(
            "INSTRUCTION_IN_MEMO", "관찰 메모에 처리 지시나 검토 우회 요청을 포함할 수 없습니다."
        )
    return segments


# Deliberately narrow lexical tripwires in the initial evaluation languages.
# These catch obvious dropped/introduced markers; they are not entailment or
# a complete grammar. Paraphrases can require human review (false positives).
NEGATION = re.compile(
    r"\b(?:not|no|never|without|cannot|can['’]t|didn['’]t|doesn['’]t|wasn['’]t|"
    r"weren['’]t|hasn['’]t|haven['’]t|isn['’]t|aren['’]t|"
    r"không|chưa|chẳng|tidak|belum|bukan|tanpa)\b",
    re.I,
)
KOREAN_NEGATION = re.compile(r"않|없|아니|못(?:했|하|해|먹|마|걸|자)|(?:^|\s)안\s")
UNCERTAINTY = re.compile(
    r"\b(?:may|might|perhaps|possibly|maybe|seems?|uncertain|unsure|"
    r"có\s+thể|có\s+lẽ|hình\s+như|mungkin|sepertinya|tampaknya)\b",
    re.I,
)
KOREAN_UNCERTAINTY = re.compile(
    r"같(?:습|아|다|은)|아마|추정|듯|수(?:도)?\s*있|불명|확인.{0,8}필요"
)


def lexical_signals(text: str) -> tuple[bool, bool]:
    return (
        bool(NEGATION.search(text) or KOREAN_NEGATION.search(text)),
        bool(UNCERTAINTY.search(text) or KOREAN_UNCERTAINTY.search(text)),
    )


def assess(draft: Draft) -> QAResult:
    """Recompute QA from content, ignoring caller-supplied QA and status."""
    checks: list[QACheck] = []
    issues = [
        QAIssue(
            code="HUMAN_REVIEW_REQUIRED",
            severity="review",
            message="근거 연결은 번역의 의미 정확성을 보증하지 않습니다. 원문·번역·초안을 담당자가 대조해야 합니다.",
        )
    ]

    def check(code: str, passed: bool, message: str) -> None:
        checks.append(QACheck(code=code, passed=passed))
        if not passed:
            issues.append(QAIssue(code=code, severity="hold", message=message))

    expected = source_segments(draft.source.text)
    refs = {s.segment_id: s for s in expected}
    translated = draft.translation.segments
    check(
        "SOURCE_INTEGRITY",
        digest(draft.source.text) == draft.source.sha256,
        "원문 무결성 값이 일치하지 않습니다.",
    )
    check(
        "SOURCE_COVERAGE",
        bool(expected) and len(expected) <= 40 and [s.evidence for s in translated] == expected,
        "원문 구간이 누락·중복·변경되었습니다.",
    )
    all_items = [*translated, *draft.observations, *draft.additional_checks]
    check(
        "EVIDENCE_PRESERVED",
        all(refs.get(item.evidence.segment_id) == item.evidence for item in all_items),
        "초안 항목의 원문 인용 또는 위치가 원문과 일치하지 않습니다.",
    )
    check(
        "TRANSLATION_CONSISTENCY",
        draft.translation.text == "\n".join(s.text for s in translated)
        and all(s.text.strip() and re.search(r"[가-힣]", s.text) for s in translated),
        "한국어 번역이 비어 있거나 구간별 번역과 전체 번역이 다릅니다.",
    )
    check(
        "OBSERVATION_GROUNDING",
        len(draft.observations) == len(translated)
        and all(
            o.text == t.text and o.evidence == t.evidence
            for o, t in zip(draft.observations, translated, strict=False)
        ),
        "관찰 초안이 번역 구간과 다르거나 근거 없는 항목이 추가되었습니다.",
    )
    check(
        "NUMERIC_FACTS_PRESERVED",
        all(
            Counter(re.findall(r"\d+(?:\.\d+)?", t.text))
            == Counter(re.findall(r"\d+(?:\.\d+)?", t.evidence.quote))
            for t in translated
        ),
        "번역의 숫자가 원문과 다릅니다. 시각·수량을 담당자가 대조해 주세요.",
    )
    check(
        "NEGATION_SIGNAL_PRESERVED",
        all(lexical_signals(t.evidence.quote)[0] == lexical_signals(t.text)[0] for t in translated),
        "원문과 번역의 부정 표현 신호가 다릅니다. 의미가 반대로 바뀌었는지 대조해 주세요.",
    )
    check(
        "UNCERTAINTY_SIGNAL_PRESERVED",
        all(lexical_signals(t.evidence.quote)[1] == lexical_signals(t.text)[1] for t in translated),
        "원문과 번역의 추측·불확실성 신호가 다릅니다. 단정 여부를 대조해 주세요.",
    )
    check(
        "FOLLOW_UP_GROUNDING",
        len({c.evidence.segment_id for c in draft.additional_checks})
        == len(draft.additional_checks)
        and all(c.question == CHECK_QUESTION for c in draft.additional_checks),
        "추가확인 항목에 중복 또는 허용되지 않은 생성 문장이 있습니다.",
    )
    check(
        "NO_IDENTIFIERS_DETECTED",
        not has_identifiers(draft.source.text) and not has_identifiers(draft.translation.text),
        "식별 가능 정보가 감지되었습니다. 비식별 여부를 다시 검토해 주세요.",
    )
    check(
        "NO_MEDICAL_JUDGMENT",
        not MEDICAL.search(draft.source.text + "\n" + draft.translation.text),
        "진단·처방·응급판단 관련 표현이 있어 보류합니다. 이 서비스는 의료 판단을 수행하지 않습니다.",
    )
    check(
        "NO_INSTRUCTIONS",
        not INSTRUCTIONS.search(draft.source.text + "\n" + draft.translation.text),
        "처리 지시 또는 검토 우회 표현이 감지되었습니다.",
    )
    check(
        "SHARING_BLOCKED",
        draft.qa.external_share_allowed is False,
        "검토 전 초안은 가족·외부 공유를 허용할 수 없습니다.",
    )
    if draft.source.kind == "audio":
        issues.append(
            QAIssue(
                code="TRANSCRIPT_REVIEW_REQUIRED",
                severity="review",
                message="원문은 자동 전사 결과입니다. 전사 내용도 작성자가 확인해야 합니다.",
            )
        )
    return QAResult(checks=checks, issues=issues)
