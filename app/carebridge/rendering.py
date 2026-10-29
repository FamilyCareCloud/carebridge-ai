"""Bounded source-backed presentation fixes, applied before all existing QA.

No corpus IDs, reference translations, user edits or Korean source text are used.
Unknown clauses retain model metadata. This is not general semantic inference.
"""

import json
import re

from app.carebridge.facts import DELAY_SOURCE, DURATION_UNITS
from app.carebridge.safety import (
    KOREAN_NEGATION,
    KOREAN_UNCERTAINTY,
    NEGATION,
    UNCERTAINTY,
    lexical_signals,
)
from app.carebridge.schemas import ModelSegment

REPORTS = {
    "en": r'(?:said|reported|stated)\s*[:,]?\s*["“](?P<body>[^"“”\n]+)["”][.!]?',
    "vi": r'nói(?:\s+rằng)?\s*[:,]?\s*["“](?P<body>[^"“”\n]+)["”][.!]?',
    "id": r'(?:mengatakan|berkata)(?:\s+bahwa)?\s*[:,]?\s*["“](?P<body>[^"“”\n]+)["”][.!]?',
}
MAX_RENDERED_CHARACTERS = 4000
# Full-clause matches, not keyword classification. Extra modifiers, additional
# propositions, pronoun ambiguity and uncertain/conditional claims abstain.
CLEAR_QUOTED = {
    "en": [
        ("hydration", r"(?:I\s+)?(?:did not drink|have never drunk) (?:water|milk|juice)[.]?"),
        ("meal", r"(?:I\s+)?(?:did not eat|have never eaten) (?:soup|porridge|rice)[.]?"),
        ("mobility", r"(?:I\s+)?(?:did not go|have never gone) outside(?: today)?[.]?"),
    ],
    "vi": [
        ("hydration", r"(?:tôi\s+)?(?:không|chưa bao giờ) uống (?:nước|sữa|nước ép)[.]?"),
        ("meal", r"(?:tôi\s+)?(?:không|chưa bao giờ) ăn (?:súp|cháo|cơm)[.]?"),
        ("mobility", r"(?:hôm nay\s+)?(?:tôi\s+)?(?:không|chưa bao giờ) ra ngoài[.]?"),
    ],
    "id": [
        ("hydration", r"(?:saya\s+)?(?:tidak(?: pernah)?|belum pernah) minum (?:air|susu|jus)[.]?"),
        ("meal", r"(?:saya\s+)?(?:tidak(?: pernah)?|belum pernah) makan (?:sup|bubur|nasi)[.]?"),
        (
            "mobility",
            r"(?:saya\s+)?(?:tidak(?: pernah)?|belum pernah) keluar rumah(?: hari ini)?[.]?",
        ),
    ],
}
CLOCK = r"(?:[01]?\d|2[0-3]):[0-5]\d"
CLEAR_OBSERVATIONS = {
    "en": [("medication", rf"at {CLOCK}, had not taken the medicine yet[.]?")],
    "vi": [("medication", rf"lúc {CLOCK} chưa uống thuốc[.]?")],
    "id": [
        ("medication", rf"pada pukul {CLOCK} belum minum obat[.]?"),
        ("hydration", r"tidak minum (?:air|susu|jus)(?: pagi ini|hari ini)[.]?"),
        ("condition", r"mengatakan tidak ada (?:rasa sakit|nyeri)(?: hari ini)?[.]?"),
    ],
}
EXACT_HALF_MEAL = {
    "en": r"ate half of (?P<meal>breakfast|lunch|dinner)[.]?",
    "vi": r"(?:đã\s+)?ăn một nửa bữa (?P<meal>sáng|trưa|tối)[.]?",
    "id": r"(?:sudah\s+)?makan setengah (?P<meal>sarapan|makan siang|makan malam)[.]?",
}
MEALS = {
    "breakfast": "아침",
    "lunch": "점심",
    "dinner": "저녁",
    "sáng": "아침",
    "trưa": "점심",
    "tối": "저녁",
    "sarapan": "아침",
    "makan siang": "점심",
    "makan malam": "저녁",
}
HALF_KO = (
    r"(?P<meal>아침|점심|저녁)(?:\s*식사)?(?:를|의)?\s*(?:반 정도|반량)(?:을|를)?\s*먹었습니다[.]?"
)
# A past marker on the main verb belongs on the negative auxiliary. Limit to
# known malformed forms with explicit auxiliary endings; never normalize quotes
# in Korean source notes or user edits.
PAST_STEMS = {"마셨": "마시", "먹었": "먹", "걸었": "걷", "갔": "가", "잤": "자"}
PAST_NEGATION = r"(?<![가-힣])(?P<stem>마셨|먹었|걸었|갔|잤)지 않(?P<end>았다고|았습니다|았다|다고|습니다|다)(?![가-힣])"
NEGATIVE_ENDINGS = {
    "았다고": "았다고",
    "았습니다": "았습니다",
    "았다": "았다",
    "다고": "았다고",
    "습니다": "았습니다",
    "다": "았다",
}
DELAY_KO = r"(?P<meal>아침|점심|저녁) 식사 후 (?P<value>\d+(?:\.\d+)?)(?P<unit>분|시간) 후(?:에)? 잠들었(?P<end>습니다[.]?|음[.]?)"
REPORTED_LAYOUT = r'말(?:했다|했습니다)\.\s*["“](?P<quote>[^"“”\n]+)["”]\s*'
HELP_SOURCE = r"pergi ke (?:ruang makan|kamar|taman) dengan kursi roda dan bantuan[.]?"
HELP_KO = r"휠체어로 도움을 받아 (?P<rest>(?:식사실|식당|방|정원|공원)(?:으)?로 갔습니다[.]?)"
ALREADY_SOURCE = (
    r"mungkin sudah (?P<meal>sarapan|makan siang|makan malam), tetapi belum dikonfirmasi[.]?"
)
ALREADY_KO = r"(?P<meal>아침|점심|저녁) 식사했을 수도 있지만, 확인되지 않았습니다[.]?"


def full(pattern: str, text: str):
    return re.fullmatch(pattern, text.strip(), re.I)


def quoted_body(source: str, language: str) -> str | None:
    pattern = REPORTS.get(language)
    match = full(pattern, source) if pattern else None
    return match["body"] if match else None


def clear_category(source: str, language: str) -> str | None:
    if lexical_signals(source)[1]:
        return None
    body = quoted_body(source, language)
    patterns = CLEAR_QUOTED.get(language, []) if body else CLEAR_OBSERVATIONS.get(language, [])
    text = body if body else source
    for category, pattern in patterns:
        if full(pattern, text):
            return category
    return None


def render_segment(source: str, language: str, segment: ModelSegment) -> ModelSegment:
    # Only the model-translation path calls this; source-preserving notes and
    # human edits must retain exact wording and author-selected metadata.
    if language not in REPORTS:
        return segment
    text = segment.translation
    if lexical_signals(source)[0]:
        text = re.sub(
            PAST_NEGATION,
            lambda m: PAST_STEMS[m["stem"]] + "지 않" + NEGATIVE_ENDINGS[m["end"]],
            text,
        )
    pattern = EXACT_HALF_MEAL.get(language)
    source_half = full(pattern, source) if pattern else None
    if source_half and (match := full(HALF_KO, text)):
        if MEALS[source_half["meal"].lower()] == match["meal"]:
            text = match["meal"] + " 식사를 절반 먹었습니다."
    if match := full(DELAY_KO, text):
        expected_unit = "min" if match["unit"] == "분" else "hour"
        delays = [m for p in DELAY_SOURCE for m in p.finditer(source)]
        if (
            len(delays) == 1
            and delays[0][1] == match["value"]
            and DURATION_UNITS[delays[0][2].lower()] == expected_unit
        ):
            text = f"{match['meal']} 식사 {match['value']}{match['unit']} 후에 잠들었습니다."
    if quoted_body(source, language) and (match := full(REPORTED_LAYOUT, text)):
        text = '"' + match["quote"].strip() + '"라고 말했습니다.'
    if language == "id":
        if full(HELP_SOURCE, source) and (match := full(HELP_KO, text)):
            text = "도움을 받아 휠체어를 이용해 " + match["rest"]
        already = full(ALREADY_SOURCE, source)
        if already and (match := full(ALREADY_KO, text)):
            if MEALS[already["meal"].lower()] == match["meal"]:
                text = f"이미 {match['meal']} 식사를 했을 수도 있지만, 확인되지 않았습니다."
    category = clear_category(source, language)
    if len(text) > MAX_RENDERED_CHARACTERS:
        text = segment.translation
    return segment.model_copy(
        update={
            "translation": text,
            "category": category or segment.category,
            "needs_confirmation": False if category else segment.needs_confirmation,
        }
    )


def rendering_definition() -> str:
    return json.dumps(
        {
            "version": "source-bound-rendering.v1",
            "max_output_characters": MAX_RENDERED_CHARACTERS,
            "lexical_signals": [
                NEGATION.pattern,
                KOREAN_NEGATION.pattern,
                UNCERTAINTY.pattern,
                KOREAN_UNCERTAINTY.pattern,
            ],
            "reports": REPORTS,
            "quoted": CLEAR_QUOTED,
            "observations": CLEAR_OBSERVATIONS,
            "half": [EXACT_HALF_MEAL, HALF_KO, MEALS],
            "negation": [PAST_STEMS, PAST_NEGATION, NEGATIVE_ENDINGS],
            "delay": [DELAY_KO, [p.pattern for p in DELAY_SOURCE], DURATION_UNITS],
            "layout": REPORTED_LAYOUT,
            "help": [HELP_SOURCE, HELP_KO],
            "already": [ALREADY_SOURCE, ALREADY_KO],
            "scope": "en/vi/id model outputs only; whole-clause metadata; lexical uncertainty always wins; all existing QA runs afterwards",
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
