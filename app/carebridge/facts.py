"""Bounded source-fact tripwires, not a general semantic accuracy score."""

import re
from collections import Counter

CLOCK = re.compile(
    r"(?<![\w:])(?P<hour>[01]?\d|2[0-3]):(?P<minute>[0-5]\d)"
    r"(?:\s*(?P<period>am|pm)\b)?(?![:\d])",
    re.I,
)
KOREAN_CLOCK = re.compile(
    r"(?<!\d)(?:(?P<period>오전|오후)\s*)?(?P<hour>\d{1,2})시(?!간)"
    r"(?:\s*(?P<minute>\d{1,2})분)?"
)
ASSISTANCE = re.compile(
    r"\b(?:with\s+(?:help|assistance)|assisted|dengan\s+bantuan)\b|"
    r"với\s+(?:sự\s+)?hỗ\s+trợ|도움|부축|(?:보조|지원)(?:을|를)?\s*받",
    re.I,
)
ZONE = re.compile(r"(?<![A-Za-z])(?:UTC|GMT|KST)(?:[+-]\d{1,2}(?::[0-5]\d)?)?(?![A-Za-z])", re.I)


def numeric_facts(text: str) -> tuple[Counter, Counter, Counter]:
    clocks: Counter = Counter()

    def clock(match: re.Match) -> str:
        hour = int(match["hour"])
        minute = int(match["minute"] or 0)
        period = (match["period"] or "").lower()
        if minute > 59:
            return match.group()
        if period:
            if not 1 <= hour <= 12:
                return match.group()
            hour = hour % 12 + (12 if period in {"pm", "오후"} else 0)
        elif match.re is KOREAN_CLOCK and 1 <= hour <= 12:
            # Bare Korean "3시" does not establish AM/PM. Do not guess.
            return match.group()
        elif hour > 23:
            return match.group()
        clocks[(hour, minute)] += 1
        return " "

    zones = Counter(m.group().upper() for m in ZONE.finditer(text))
    remainder = CLOCK.sub(clock, ZONE.sub(" ", text))
    remainder = KOREAN_CLOCK.sub(clock, remainder)
    return clocks, Counter(re.findall(r"\d+(?:\.\d+)?", remainder)), zones


def assistance_preserved(source: str, translation: str) -> bool:
    # Check explicit supported source phrases only. No inference for other languages.
    return not ASSISTANCE.search(source) or bool(ASSISTANCE.search(translation))


# Small, explicit lexical tripwires discovered in synthetic regression runs.
# These can only downgrade model QA, never establish semantic accuracy.
FACT_MARKERS = [
    (
        re.compile(r"\b(?:walk(?:ed|ing)?|berjalan)\b|đi\s+bộ", re.I),
        re.compile(r"걷|걸[어었으음]|보행|도보"),
    ),
    (re.compile(r"\b(?:soup|súp|sup)\b", re.I), re.compile(r"수프|스프|국(?:을|은|이|물|\s|$)")),
    (
        re.compile(r"\b(?:said|reported|stated|mengatakan|berkata)\b|nói\s+rằng", re.I),
        re.compile(r"말했|말하|말합|말씀|진술|호소|언급|전했|(?:다고|라고|다며)\s*(?:했|하|함|합)"),
    ),
]
DURATION_SOURCE = [
    re.compile(r"\bfor\s+(\d+(?:\.\d+)?)\s+(minutes?|hours?)\b", re.I),
    re.compile(r"\b(?:ngủ|nghỉ(?:\s+ngơi)?)\s+(?:trong\s+)?(\d+(?:\.\d+)?)\s*(phút|giờ)\b", re.I),
    re.compile(r"\bselama\s+(\d+(?:\.\d+)?)\s*(menit|jam)\b", re.I),
]
DURATION_KO = re.compile(
    r"(\d+(?:\.\d+)?)\s*(분|시간)(?:\s*(?:동안|간)|\s*(?:을|를)?\s*(?:잤|자고|잠을|주무|쉬었|휴식|걸었|걷))"
)
DURATION_UNITS = {
    "minute": "min",
    "minutes": "min",
    "phút": "min",
    "menit": "min",
    "분": "min",
    "hour": "hour",
    "hours": "hour",
    "giờ": "hour",
    "jam": "hour",
    "시간": "hour",
}


def explicit_facts_preserved(source: str, translation: str) -> bool:
    return all(not src.search(source) or target.search(translation) for src, target in FACT_MARKERS)


def durations_preserved(source: str, translation: str) -> bool:
    def extract(text: str, patterns: list) -> Counter:
        return Counter(
            (m[1], DURATION_UNITS[m[2].lower()])
            for pattern in patterns
            for m in pattern.finditer(text)
        )

    expected = extract(source, [*DURATION_SOURCE, DURATION_KO])
    actual = extract(translation, [DURATION_KO])
    return all(actual[key] >= count for key, count in expected.items())
