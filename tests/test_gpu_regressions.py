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
        ("Ate half of the soup.", "식사를 절반 먹었습니다.", False),
        ("Ate half of the soup.", "수프를 절반 먹었습니다.", True),
        ("Said there was no pain today.", "오늘은 통증이 없었습니다.", False),
        ("Said there was no pain today.", "오늘은 통증이 없다고 합니다.", True),
        ("Nói rằng hôm nay không bị đau.", "오늘 통증이 없다고 말했습니다.", True),
        ("Mengatakan tidak ada rasa sakit hari ini.", "오늘 통증이 없었습니다.", False),
    ],
)
def test_model_pass_does_not_override_explicit_lost_fact(
    config, headers, monkeypatch, source, target, passed
):
    draft = make_draft(config, headers, source, target)
    assert draft["status"] == "review_required"
    client, calls = local_client(config, json.dumps({"segments": [GOOD]}), monkeypatch)
    with client:
        result = client.post("/carebridge/qa/semantic", headers=headers, json=draft).json()
    assert result["model_checked"] is True
    assert len(calls) == 1
    assert (
        next(c["passed"] for c in result["qa"]["checks"] if c["code"] == "SEMANTIC_OMITTED_FACTS")
        is passed
    )
    assert (
        next(
            c["passed"]
            for c in result["segments"][0]["checks"]
            if c["code"] == "SEMANTIC_OMITTED_FACTS"
        )
        is passed
    )
    assert result["status"] == ("review_required" if passed else "hold")
