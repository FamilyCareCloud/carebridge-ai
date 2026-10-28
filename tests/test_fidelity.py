"""Regressions from the frozen 2026-10-27 corpus, without model self-grading."""

import pytest
from fastapi.testclient import TestClient

from app.carebridge.facts import numeric_facts
from app.main import create_app
from tests.conftest import FakeProvider


def draft(config, headers, source, translation, language="en"):
    with TestClient(create_app(config, provider=FakeProvider([translation]))) as client:
        response = client.post(
            "/carebridge/drafts",
            headers=headers,
            json={
                "memo": source,
                "source_language": language,
                "data_use": "synthetic_deidentified",
            },
        )
    assert response.status_code == 200
    return response.json()


@pytest.mark.parametrize(
    "source,language",
    [
        ("Walked to the bathroom with help.", "en"),
        ("Đã đi bộ đến nhà vệ sinh với sự hỗ trợ.", "vi"),
        ("Berjalan ke kamar mandi dengan bantuan.", "id"),
    ],
)
def test_missing_assistance_is_held_but_faithful_translation_is_not(
    config, headers, source, language
):
    lost = draft(config, headers, source, "화장실까지 걸어갔습니다.", language)
    faithful = draft(config, headers, source, "도움을 받아 화장실까지 걸어갔습니다.", language)
    assert lost["status"] == "hold"
    assert any(
        c["code"] == "ASSISTANCE_FACT_PRESERVED" and not c["passed"] for c in lost["qa"]["checks"]
    )
    assert faithful["status"] == "review_required"
    assert lost["source"]["text"] == source
    assert lost["qa"]["external_share_allowed"] is False


@pytest.mark.parametrize(
    "source,target",
    [
        (
            "Slept for 30 minutes after lunch and woke up at 15:00.",
            "점심 후 30분 동안 잤고 오후 3시에 깨어났습니다.",
        ),
        ("Started resting at 14:30.", "14시 30분에 휴식을 시작했습니다."),
        ("Woke up at 00:05.", "오전 12시 5분에 깨어났습니다."),
        ("Woke up at 12:00.", "오후 12시에 깨어났습니다."),
        ("Woke up at 3:00 PM.", "15:00에 깨어났습니다."),
        ("Woke up at 15:00 KST.", "오후 3시 KST에 깨어났습니다."),
    ],
)
def test_equivalent_explicit_clocks_do_not_trigger_numeric_hold(config, headers, source, target):
    assert draft(config, headers, source, target)["status"] == "review_required"


@pytest.mark.parametrize(
    "source,target",
    [
        ("Woke up at 15:00.", "오전 3시에 깨어났습니다."),
        ("Woke up at 15:00.", "3시에 깨어났습니다."),
        ("Rested for 30 minutes at 15:00.", "오후 3시에 35분 동안 쉬었습니다."),
        ("Started resting at 14:30.", "04:30에 휴식을 시작했습니다."),
        ("Woke up at 12:00.", "오전 12시에 깨어났습니다."),
        ("Woke up at 15:00.", "오후 3시와 오후 3시에 깨어났습니다."),
        ("Woke up at 15:00.", "깨어났습니다."),
        ("Woke up at 15:00 UTC.", "오후 3시 KST에 깨어났습니다."),
        ("Woke up at 15:00 UTC+09:00.", "오후 3시에 깨어났습니다."),
    ],
)
def test_changed_missing_ambiguous_or_duplicated_time_is_still_held(
    config, headers, source, target
):
    result = draft(config, headers, source, target)
    assert any(
        c["code"] == "NUMERIC_FACTS_PRESERVED" and not c["passed"] for c in result["qa"]["checks"]
    )
    assert result["status"] == "hold"


@pytest.mark.parametrize(
    "source,target,language",
    [
        (
            "Seemed tired. The reason was unknown.",
            "피곤해 보였지만 이유는 알려지지 않았습니다.",
            "en",
        ),
        ("Có vẻ mệt. Không rõ lý do.", "피곤해 보입니다. 이유는 알 수 없습니다.", "vi"),
        (
            "Tampak lelah. Alasannya tidak diketahui.",
            "피곤해 보입니다. 이유는 알려져 있지 않습니다.",
            "id",
        ),
    ],
)
def test_known_appearance_and_absence_paraphrases_are_preserved(
    config, headers, source, target, language
):
    assert draft(config, headers, source, target, language)["status"] == "review_required"


def test_duration_does_not_become_a_clock():
    assert numeric_facts("14시간 30분") == numeric_facts("14 hours 30 minutes")
    assert numeric_facts("15:00:30") != numeric_facts("15:00")
