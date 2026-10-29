"""Conservative Korean-output guard, not a fluency or semantic score."""

import re

# Permit common measurement/time notation and uppercase abbreviations. Never
# delete foreign text to make a response look complete; reject the whole output.
UNITS = {
    "mL",
    "ml",
    "ML",
    "L",
    "l",
    "mg",
    "g",
    "kg",
    "cm",
    "mm",
    "km",
    "mmHg",
    "kcal",
    "bpm",
    "SpO",
}


def korean_output(text: str) -> bool:
    if not re.search(r"[가-힣]", text):
        return False
    for token in re.findall(r"[^\W\d_]+", text, flags=re.UNICODE):
        # Split adjacent Korean/unit text (e.g. 150mL를) without accepting prose.
        for word in re.findall(r"[^가-힣ㄱ-ㅣ]+", token):
            if word not in UNITS and not re.fullmatch(r"[A-Z]{2,8}", word):
                return False
    return True
