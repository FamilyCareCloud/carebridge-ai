"""Small source-matched translation hints; never replacements for source or model output."""

import json
import re

from app.carebridge.facts import (
    AFTER_WALKING,
    DELAY_SOURCE,
    DURATION_UNITS,
    LIFETIME_NEGATION,
    REPORTED_SPEECH,
    SIMPLE_NEGATION,
    SOURCE_CONDITION,
)
from app.carebridge.schemas import SourceSegment

TERM_HINTS = [
    (r"\bdinner\b|bữa\s+tối|ăn\s+tối|\bmakan\s+malam\b", "저녁 식사"),
    (r"\bwheelchairs?\b|xe\s+lăn|\bkursi\s+roda\b", "이동 도구의 한국어 명칭: 휠체어"),
]
AFTER_HINT = "걷기 이후에 관찰한 순서입니다. 한국어 표현: '걸은 후' 또는 '걷고 나서'."
DELAY_HINT = "잠들기 전 간격: {value}{unit} 후. 잠을 잔 지속시간이 아닙니다."
NEGATION_HINT = "단순 부정을 담은 발언입니다. 간접 인용 표현: '...하지 않았다고 말했습니다' 또는 '...없다고 말했습니다'. 원문의 부정 범위와 시제를 유지하세요."
LIFETIME_HINT = (
    "해당 행동을 한 경험이 없다는 발언입니다. 한국어 표현: '...한 적이 없다고 말했습니다'."
)
EXPLICIT_LIFETIME = re.compile(r"\bnever\b|chưa\s+bao\s+giờ|\b(?:tidak|belum)\s+pernah\b", re.I)
HABITUAL_WALKING = re.compile(
    r"\bwhenever\s+walking\b|mỗi\s+khi\s+đi\s+bộ|\bsetiap\s+kali\s+berjalan\b", re.I
)
HABITUAL_HINT = "걷는 때마다 반복되는 관찰입니다. 한국어 표현: '걸을 때마다'."
NEGATED_WALKING = re.compile(
    r"\b(?:did|does|do)\s+not\s+walk\b|không\s+đi\s+bộ|\btidak\s+berjalan\b", re.I
)
NEGATED_WALKING_HINT = "걷지 않았다는 관찰입니다. 한국어 표현: '걷지 않았으며'. 걷는 능력에 관한 판단은 추가하지 마세요."
MOVEMENT_HELP = re.compile(
    r"\bwith\s+(?:help|assistance)\b|với\s+(?:sự\s+)?hỗ\s+trợ|\b(?:dengan|dan)\s+bantuan\b", re.I
)
MOVEMENT_HELP_HINT = (
    "이동할 때 도움을 받았습니다. 한국어 표현: '도움을 받아'. 도움 제공자는 원문대로 유지하세요."
)
UNITS = {"min": "분", "hour": "시간"}
MAX_HINTS = 8


def guidance_definition() -> str:
    """Include the matched-hint templates/rules in the pipeline prompt fingerprint."""
    return json.dumps(
        {
            "version": "source-hints.v2",
            "terms": TERM_HINTS,
            "after_walking": [AFTER_WALKING.pattern, SOURCE_CONDITION.pattern, AFTER_HINT],
            "delay": [[p.pattern for p in DELAY_SOURCE], DURATION_UNITS, UNITS, DELAY_HINT],
            "reported_negation": [
                REPORTED_SPEECH.pattern,
                SIMPLE_NEGATION.pattern,
                LIFETIME_NEGATION.pattern,
                NEGATION_HINT,
            ],
            "reported_lifetime": [EXPLICIT_LIFETIME.pattern, LIFETIME_HINT],
            "habitual_walking": [HABITUAL_WALKING.pattern, HABITUAL_HINT],
            "negated_walking": [NEGATED_WALKING.pattern, NEGATED_WALKING_HINT],
            "wheelchair_assistance": [MOVEMENT_HELP.pattern, MOVEMENT_HELP_HINT],
            "matching": "Unicode regex, case insensitive; first term/after match; all delays",
            "max_hints_per_segment": MAX_HINTS,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def source_hints(source: str) -> list[dict[str, str]]:
    hints = []
    for pattern, meaning in TERM_HINTS:
        if match := re.search(pattern, source, re.I):
            hints.append({"source_phrase": match.group(), "meaning": meaning})
    if (match := AFTER_WALKING.search(source)) and not SOURCE_CONDITION.search(source):
        hints.append({"source_phrase": match.group(), "meaning": AFTER_HINT})
    for pattern, meaning in [
        (HABITUAL_WALKING, HABITUAL_HINT),
        (NEGATED_WALKING, NEGATED_WALKING_HINT),
    ]:
        if match := pattern.search(source):
            hints.append({"source_phrase": match.group(), "meaning": meaning})
    if re.search(TERM_HINTS[1][0], source, re.I) and (match := MOVEMENT_HELP.search(source)):
        hints.append({"source_phrase": match.group(), "meaning": MOVEMENT_HELP_HINT})
    for pattern in DELAY_SOURCE:
        for match in pattern.finditer(source):
            hints.append(
                {
                    "source_phrase": match.group(),
                    "meaning": DELAY_HINT.format(
                        value=match[1], unit=UNITS[DURATION_UNITS[match[2].lower()]]
                    ),
                }
            )
    if (
        REPORTED_SPEECH.search(source)
        and (match := SIMPLE_NEGATION.search(source))
        and not LIFETIME_NEGATION.search(source)
    ):
        hints.append({"source_phrase": match.group(), "meaning": NEGATION_HINT})
    if REPORTED_SPEECH.search(source) and (match := EXPLICIT_LIFETIME.search(source)):
        hints.append({"source_phrase": match.group(), "meaning": LIFETIME_HINT})
    return hints[:MAX_HINTS]


def translation_payload(segments: list[SourceSegment], language: str) -> dict:
    data = []
    for segment in segments:
        item = {"segment_id": segment.segment_id, "text": segment.quote}
        if hints := source_hints(segment.quote):
            item["source_hints"] = hints
        data.append(item)
    return {"source_language": language, "segments": data}
