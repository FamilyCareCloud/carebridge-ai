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
    r"với\s+(?:sự\s+)?hỗ\s+trợ|도움|부축|보조를?\s*받",
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
