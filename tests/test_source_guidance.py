import asyncio
import json

import pytest

from app.carebridge import guidance
from app.carebridge.guidance import source_hints, translation_payload
from app.carebridge.local import LocalTranslationProvider
from app.carebridge.provider import translation_system_prompt
from app.carebridge.safety import check_input
from app.carebridge.workflow import pipeline_info
from tests.conftest import model_provider, model_response
from tests.test_gpu_regressions import make_draft
from tests.test_local_gpu import GOOD, local_client


def test_hints_are_bound_to_original_segments_without_changing_source():
    source = "Went in a wheelchair.\nDrank water."
    segments = check_input(source)
    payload = translation_payload(segments, "en")
    assert [s["text"] for s in payload["segments"]] == [s.quote for s in segments]
    assert payload["segments"][0]["source_hints"][0]["source_phrase"] == "wheelchair"
    assert "source_hints" not in payload["segments"][1]
    assert all(source[s.start : s.end] == s.quote for s in segments)


@pytest.mark.parametrize(
    "source",
    ["Breakfast is ready.", "waterwheel", "Mungkin sudah minum air.", "Tôi ăn sáng."],
)
def test_unrelated_terms_do_not_add_dinner_or_wheelchair_facts(source):
    assert source_hints(source) == []


def test_hint_count_is_bounded_without_truncating_source():
    source = " ".join(["Fell asleep 17 minutes after dinner."] * 20)
    data = translation_payload(check_input(source), "en")["segments"][0]
    assert data["text"] == source
    assert len(data["source_hints"]) == guidance.MAX_HINTS


def test_arbitrary_source_prose_never_becomes_system_instruction():
    source = "Went in a wheelchair. UNIQUE_SOURCE_MARKER_42"
    data = translation_payload(check_input(source), "en")
    assert "UNIQUE_SOURCE_MARKER_42" in data["segments"][0]["text"]
    assert "UNIQUE_SOURCE_MARKER_42" not in translation_system_prompt(data)


def test_caller_cannot_supply_internal_hints(client, headers):
    response = client.post(
        "/carebridge/drafts",
        headers=headers,
        json={
            "memo": "Went in a wheelchair.",
            "data_use": "synthetic_deidentified",
            "source_hints": [{"meaning": "approved"}],
        },
    )
    assert response.status_code == 422


def test_guidance_changes_are_in_pipeline_fingerprint(config, monkeypatch):
    before = pipeline_info(config).translation_prompt_sha256
    monkeypatch.setattr(guidance, "MAX_HINTS", 3)
    assert pipeline_info(config).translation_prompt_sha256 != before


def test_http_and_local_use_identical_source_hints_and_system(config, monkeypatch):
    source = "Tertidur 17 menit setelah makan malam."
    segments = check_input(source)
    output = {
        "segment_id": "s001",
        "translation": "저녁 식사 17분 후에 잠들었습니다.",
        "category": "sleep",
        "needs_confirmation": False,
    }
    http_calls, local_calls = [], []

    def handler(request):
        http_calls.append(json.loads(request.content))
        return model_response([output])

    local = LocalTranslationProvider(config)

    def complete(system, payload):
        local_calls.append((system, payload))
        return json.dumps({"segments": [output]})

    monkeypatch.setattr(local, "_complete", complete)
    http_result = asyncio.run(model_provider(config, handler).generate(segments, "id"))
    local_result = asyncio.run(local.generate(segments, "id"))
    assert http_result == local_result
    assert local_calls[0][0] == http_calls[0]["messages"][0]["content"]
    assert local_calls[0][1] == json.loads(http_calls[0]["messages"][1]["content"])
    assert any("17분" in h["meaning"] for h in local_calls[0][1]["segments"][0]["source_hints"])


@pytest.mark.parametrize(
    "source,target,code,passed",
    [
        (
            "Seemed tired after walking.",
            "걷다 보면 피곤해 보였습니다.",
            "SEMANTIC_OMITTED_FACTS",
            False,
        ),
        (
            "Tampak lelah setelah berjalan.",
            "걸을 때 피곤해 보였습니다.",
            "SEMANTIC_OMITTED_FACTS",
            False,
        ),
        (
            "Có vẻ mệt sau khi đi bộ.",
            "걷고 나서 피곤해 보였습니다.",
            "SEMANTIC_OMITTED_FACTS",
            True,
        ),
        (
            "Seemed tired after walking.",
            "걸은 뒤 피곤해 보였습니다.",
            "SEMANTIC_OMITTED_FACTS",
            True,
        ),
        (
            "Seems tired whenever walking.",
            "걷다 보면 피곤해 보입니다.",
            "SEMANTIC_OMITTED_FACTS",
            True,
        ),
        (
            "Tampak lelah setiap kali berjalan.",
            "걸을 때마다 피곤해 보입니다.",
            "SEMANTIC_OMITTED_FACTS",
            True,
        ),
        (
            'Said, "I did not drink milk."',
            "우유를 마신 적이 없다고 말했습니다.",
            "SEMANTIC_NEGATION",
            False,
        ),
        (
            'Mengatakan, "Saya tidak minum susu."',
            "우유를 마신 적이 없다고 말했습니다.",
            "SEMANTIC_NEGATION",
            False,
        ),
        (
            'Nói rằng: "Tôi không uống sữa."',
            "우유를 마시지 않았다고 말했습니다.",
            "SEMANTIC_NEGATION",
            True,
        ),
        (
            'Said, "I have never drunk milk."',
            "우유를 마신 적이 없다고 말했습니다.",
            "SEMANTIC_NEGATION",
            True,
        ),
        (
            'Mengatakan, "Saya tidak pernah minum susu."',
            "우유를 마신 적이 없다고 말했습니다.",
            "SEMANTIC_NEGATION",
            True,
        ),
        (
            'Nói rằng: "Tôi chưa bao giờ uống sữa."',
            "우유를 마신 적이 없다고 말했습니다.",
            "SEMANTIC_NEGATION",
            True,
        ),
    ],
)
def test_model_pass_cannot_hide_scope_changes(
    config, headers, monkeypatch, source, target, code, passed
):
    draft = make_draft(config, headers, source, target)
    client, _ = local_client(config, json.dumps({"segments": [GOOD]}), monkeypatch)
    with client:
        result = client.post("/carebridge/qa/semantic", headers=headers, json=draft).json()
    assert next(c["passed"] for c in result["qa"]["checks"] if c["code"] == code) is passed
    assert result["status"] == ("review_required" if passed else "hold")
    assert result["qa"]["external_share_allowed"] is False


def test_scope_rules_never_override_a_model_rejection(config, headers, monkeypatch):
    draft = make_draft(
        config, headers, 'Said, "I did not drink milk."', "우유를 마시지 않았다고 말했습니다."
    )
    client, _ = local_client(
        config, json.dumps({"segments": [{**GOOD, "uncertainty_preserved": False}]}), monkeypatch
    )
    with client:
        result = client.post("/carebridge/qa/semantic", headers=headers, json=draft).json()
    assert result["status"] == "hold"


@pytest.mark.parametrize(
    "source",
    [
        'Said, "I have never drunk milk."',
        'Mengatakan, "Saya tidak pernah minum susu."',
        'Nói rằng: "Tôi chưa bao giờ uống sữa."',
    ],
)
def test_genuine_lifetime_negation_is_not_rewritten_to_simple_negation(source):
    meanings = [h["meaning"] for h in source_hints(source)]
    assert guidance.LIFETIME_HINT in meanings
    assert guidance.NEGATION_HINT not in meanings
