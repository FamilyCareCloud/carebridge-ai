"""Small source-matched translation hints; never replacements for source or model output."""

import json
import re

from app.carebridge.facts import (
    AFTER_WALKING,
    DELAY_SOURCE,
    DURATION_UNITS,
    EXPERIENTIAL_NEGATION,
    LIFETIME_NEGATION,
    REPORTED_SPEECH,
    SIMPLE_NEGATION,
    SOURCE_CONDITION,
)
from app.carebridge.schemas import SourceSegment

TERM_HINTS = [
    (r"\bdinner\b|bữa\s+tối|ăn\s+tối|\bmakan\s+malam\b", "저녁 식사"),
    (r"\bwheelchairs?\b|xe\s+lăn|\bkursi\s+roda\b", "휠체어"),
    (r"\bpergi\b", "갔습니다"),
    (r"\bvườn\b", "정원"),
    (r"\bseem(?:s|ed)?\s+thirsty\b|có\s+vẻ\s+khát(?:\s+nước)?|\btampak\s+haus\b", "목말라 보였음"),
]
AFTER_HINT = "걸은 후"
DELAY_HINT = "{value}{unit} 후에 잠들었음"
NEGATION_HINT = "해당 시점의 행동 부정"
LIFETIME_HINT = (
    "해당 행동을 한 경험이 없다는 발언입니다. 한국어 표현: '...한 적이 없다고 말했습니다'."
)
EXPLICIT_LIFETIME = EXPERIENTIAL_NEGATION
REPORTED_ID_TERMS = [
    (r"\btidak\s+keluar\s+rumah\b", "집 밖으로 나가지 않았다"),
    (r"\btidak\s+minum\b", "마시지 않았다"),
    (r"\btidak\s+makan\b", "먹지 않았다"),
    (r"\bsup\b", "수프"),
    (r"\bsusu\b", "우유"),
    (r"\bmakan\b", "먹다"),
    (r"\bminum\b", "마시다"),
    (r"\bkeluar\s+rumah\b", "집 밖으로 나가다"),
    (r"\bhari\s+ini\b", "오늘"),
]
REPORTED_OUTSIDE_VI = (r"ra\s+ngoài", "밖으로 나가다")
HABITUAL_WALKING = re.compile(
    r"\bwhenever\s+walking\b|mỗi\s+khi\s+đi\s+bộ|\bsetiap\s+kali\s+berjalan\b", re.I
)
HABITUAL_HINT = "걸을 때마다"
NEGATED_WALKING = re.compile(
    r"\b(?:did|does|do)\s+not\s+walk\b|không\s+đi\s+bộ|\btidak\s+berjalan\b", re.I
)
NEGATED_WALKING_HINT = "걷지 않았음"
MOVEMENT_HELP = re.compile(
    r"\bwith\s+(?:help|assistance)\b|với\s+(?:sự\s+)?hỗ\s+trợ|\b(?:dengan|dan)\s+bantuan\b", re.I
)
MOVEMENT_HELP_HINT = "도움을 받아"
UNITS = {"min": "분", "hour": "시간"}
MAX_HINTS = 8


def guidance_definition() -> str:
    """Include the matched-hint templates/rules in the pipeline prompt fingerprint."""
    return json.dumps(
        {
            "version": "source-hints.v4",
            "system_rendering": "id reported simple negation: Korean instructions; lifetime: JSON guides; other hints: short per-segment lines; no hints: original system",
            "terms": TERM_HINTS,
            "after_walking": [AFTER_WALKING.pattern, SOURCE_CONDITION.pattern, AFTER_HINT],
            "delay": [[p.pattern for p in DELAY_SOURCE], DURATION_UNITS, UNITS, DELAY_HINT],
            "reported_negation": [
                "id",
                REPORTED_SPEECH.pattern,
                SIMPLE_NEGATION.pattern,
                LIFETIME_NEGATION.pattern,
                NEGATION_HINT,
            ],
            "reported_lifetime": [EXPLICIT_LIFETIME.pattern, LIFETIME_HINT],
            "reported_id_terms": REPORTED_ID_TERMS,
            "reported_outside_vi": REPORTED_OUTSIDE_VI,
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


def source_hints(source: str, language: str = "") -> list[dict[str, str]]:
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
        language == "id"
        and REPORTED_SPEECH.search(source)
        and (match := SIMPLE_NEGATION.search(source))
        and not LIFETIME_NEGATION.search(source)
    ):
        hints.append({"source_phrase": match.group(), "meaning": NEGATION_HINT})
    if REPORTED_SPEECH.search(source) and (match := EXPLICIT_LIFETIME.search(source)):
        hints.append({"source_phrase": match.group(), "meaning": LIFETIME_HINT})
    if REPORTED_SPEECH.search(source):
        terms = (
            REPORTED_ID_TERMS
            if language == "id" and any(h["meaning"] == NEGATION_HINT for h in hints)
            else [REPORTED_OUTSIDE_VI]
            if language == "vi"
            else []
        )
        for pattern, meaning in terms:
            if match := re.search(pattern, source, re.I):
                hints.append({"source_phrase": match.group(), "meaning": meaning})
    return hints[:MAX_HINTS]


def translation_payload(segments: list[SourceSegment], language: str) -> dict:
    data = []
    for segment in segments:
        item = {"segment_id": segment.segment_id, "text": segment.quote}
        if hints := source_hints(segment.quote, language):
            item["source_hints"] = hints
        data.append(item)
    return {"source_language": language, "segments": data}
