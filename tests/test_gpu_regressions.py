"""Regressions from the recorded real-weight spec evaluation, with CPU-only doubles."""

import json

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import FakeProvider
from tests.test_local_gpu import GOOD, local_client


def make_draft(config, headers, source, translation):
    with TestClient(create_app(config, provider=FakeProvider([translation]))) as client:
        response = client.post(
            "/carebridge/drafts",
            headers=headers,
            json={
                "memo": source,
                "source_language": "auto",
                "data_use": "synthetic_deidentified",
            },
        )
    assert response.status_code == 200
    return response.json()


@pytest.mark.parametrize(
    "source,target,passed",
    [
        ("Walked with help.", "지원을 받으며 걸었습니다.", True),
        ("Walked with help.", "보조를 받아 걸었습니다.", True),
        ("Walked with help.", "걸었습니다.", False),
        ("Walked with help.", "지원 여부는 모릅니다.", False),
    ],
)
def test_assistance_synonyms_do_not_erase_missing_help(config, headers, source, target, passed):
    draft = make_draft(config, headers, source, target)
    assert (
        next(c["passed"] for c in draft["qa"]["checks"] if c["code"] == "ASSISTANCE_FACT_PRESERVED")
        is passed
    )


@pytest.mark.parametrize(
    "source,target,passed",
    [
        ("Slept for 30 minutes after lunch.", "점심 후 30분 동안 잤습니다.", True),
        ("Slept for 30 minutes after lunch.", "점심 후 30분간 잤습니다.", True),
        ("Slept for 30 minutes after lunch.", "점심 후 30분 잤습니다.", True),
        ("Slept for 30 minutes after lunch.", "점심 후 30분 후에 잤습니다.", False),
        ("Đã ngủ 30 phút sau bữa trưa.", "점심 후 30분 후에 잤습니다.", False),
        ("Đã ngủ 30 phút sau bữa trưa.", "점심 후 30분 동안 잤습니다.", True),
        ("Tidur selama 30 menit setelah makan siang.", "점심 후 30분 잤습니다.", True),
        ("Slept for 2 hours.", "2시간 동안 잤습니다.", True),
        ("Slept for 2 hours.", "2분 동안 잤습니다.", False),
        ("Fell asleep 20 minutes after breakfast.", "아침 식사 20분 후에 잠들었습니다.", True),
        ("Fell asleep 20 minutes after breakfast.", "아침 식사 후 20분 동안 잤습니다.", False),
        ("Đã ngủ thiếp đi 20 phút sau bữa sáng.", "아침 식사 20분 뒤에 잠들었습니다.", True),
        ("Đã ngủ thiếp đi 20 phút sau bữa sáng.", "아침 식사 후 20분간 잤습니다.", False),
        ("Tertidur 20 menit setelah sarapan.", "아침 식사 20분이 지난 뒤 잠들었습니다.", True),
        ("Tertidur 20 menit setelah sarapan.", "아침 식사 후 20분 동안 잤습니다.", False),
        (
            "Fell asleep 10 minutes after lunch and slept for 20 minutes.",
            "점심 식사 10분 후에 잠들어 20분 동안 잤습니다.",
            True,
        ),
        (
            "Fell asleep 10 minutes after lunch and slept for 20 minutes.",
            "점심 식사 20분 후에 잠들어 10분 동안 잤습니다.",
            False,
        ),
    ],
)
def test_duration_is_not_a_delay(config, headers, source, target, passed):
    draft = make_draft(config, headers, source, target)
    assert (
        next(c["passed"] for c in draft["qa"]["checks"] if c["code"] == "NUMERIC_FACTS_PRESERVED")
        is passed
    )


@pytest.mark.parametrize(
    "source,target,passed",
    [
        ("Walked to the bathroom with help.", "화장실로 가는 데 도움을 받았습니다.", False),
        ("Berjalan ke kamar mandi dengan bantuan.", "화장실로 가는 데 도움을 받았습니다.", False),
        ("Đã đi bộ đến nhà vệ sinh với sự hỗ trợ.", "도움을 받아 화장실로 걸어갔습니다.", True),
        ("Walked with help.", "지원을 받아 보행했습니다.", True),
        ("Walked with help.", "도움을 받아 걸으셨습니다.", True),
        ("Seemed tired after walking.", "걸은 후 피곤해 보였습니다.", True),
        ("Seemed tired after walking.", "피곤해 보였습니다.", False),
        ("Tidak keluar rumah hari ini.", "오늘은 화장실을 이용하지 않았습니다.", False),
        ("Tidak keluar rumah hari ini.", "오늘은 외출하지 않았습니다.", True),
        ("Did not go outside today.", "오늘은 밖에 나가지 않았습니다.", True),
        ("Không ra ngoài hôm nay.", "오늘은 외출하지 않았습니다.", True),
        ("Ate half of the soup.", "식사를 절반 먹었습니다.", False),
        ("Ate half of the soup.", "수프를 절반 먹었습니다.", True),
        ("Said there was no pain today.", "오늘은 통증이 없었습니다.", False),
        ("Said there was no pain today.", "오늘은 통증이 없다고 합니다.", True),
        ("Nói rằng hôm nay không bị đau.", "오늘 통증이 없다고 말했습니다.", True),
        ("Mengatakan tidak ada rasa sakit hari ini.", "오늘 통증이 없었습니다.", False),
    ],
)
def test_explicit_lost_facts_hold_before_model_and_complete_facts_still_use_model(
    config, headers, monkeypatch, source, target, passed
):
    draft = make_draft(config, headers, source, target)
    assert draft["status"] == "review_required"
    client, calls = local_client(config, json.dumps({"segments": [GOOD]}), monkeypatch)
    with client:
        result = client.post("/carebridge/qa/semantic", headers=headers, json=draft).json()
    assert result["model_checked"] is passed
    assert len(calls) == int(passed)
    assert (
        next(c["passed"] for c in result["qa"]["checks"] if c["code"] == "SEMANTIC_OMITTED_FACTS")
        is passed
    )
    if passed:
        assert next(
            c["passed"]
            for c in result["segments"][0]["checks"]
            if c["code"] == "SEMANTIC_OMITTED_FACTS"
        )
    else:
        assert result["segments"] == []
    assert result["status"] == ("review_required" if passed else "hold")
