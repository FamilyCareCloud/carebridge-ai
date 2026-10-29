import asyncio
import json

import pytest

from app.carebridge import guidance
from app.carebridge.guidance import source_hints, translation_payload
from app.carebridge.local import LocalTranslationProvider
from app.carebridge.provider import KOREAN_CONTEXT_PROMPT, SYSTEM_PROMPT, translation_system_prompt
from app.carebridge.safety import check_input, lexical_signals
from app.carebridge.schemas import Draft, DraftRequest
from app.carebridge.semantic import AUDIT_PROMPT, REPORTED_AUDIT_PROMPT, audit_prompt
from app.carebridge.service import DraftService
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


def test_unhinted_request_keeps_original_model_input():
    data = translation_payload(check_input("Walked with help."), "en")
    assert data == {
        "source_language": "en",
        "segments": [{"segment_id": "s001", "text": "Walked with help."}],
    }
    assert translation_system_prompt(data) == SYSTEM_PROMPT


def test_indonesian_reported_negation_uses_source_context_not_object_lookup():
    source = 'Mengatakan, "Saya tidak makan sup."'
    data = translation_payload(check_input(source), "id")
    assert data["segments"][0]["text"] == source
    assert translation_system_prompt(data) == KOREAN_CONTEXT_PROMPT
    assert "sup" not in translation_system_prompt(data)


def test_actual_thirst_is_not_given_uncertain_hint():
    assert source_hints("Was thirsty after walking.") == [
        {"source_phrase": "after walking", "meaning": guidance.AFTER_HINT}
    ]


def test_habitual_never_is_not_glossed_as_experiential_negation():
    assert source_hints('Said, "I never walk alone."', "en") == []


@pytest.mark.parametrize(
    "source,target,reported",
    [
        ('Mengatakan, "Saya tidak makan sup."', "수프를 먹지 않았다고 말했습니다.", True),
        ("Said there was no pain today.", "오늘 통증이 없었다고 말했습니다.", False),
        ("Drank water but did not eat soup.", "물은 마셨지만 수프는 먹지 않았습니다.", False),
        (
            'Said, "I drank water but did not eat soup."',
            "물은 마셨지만 수프는 먹지 않았다고 말했습니다.",
            False,
        ),
    ],
)
def test_reported_audit_does_not_replace_general_or_contrast_checks(
    config, headers, source, target, reported
):
    draft = Draft.model_validate(make_draft(config, headers, source, target))
    assert audit_prompt(draft) == (REPORTED_AUDIT_PROMPT if reported else AUDIT_PROMPT)


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
            "Tampak lelah setiap kali berjalan.",
            "걸은 후 피곤해 보였습니다.",
            "SEMANTIC_OMITTED_FACTS",
            False,
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
        (
            'Said, "I have never drunk milk."',
            "우유를 마시지 않았다고 말했습니다.",
            "SEMANTIC_NEGATION",
            False,
        ),
        (
            'Mengatakan, "Saya tidak pernah minum susu."',
            "우유를 마시지 않았다고 말했습니다.",
            "SEMANTIC_NEGATION",
            False,
        ),
        (
            'Nói rằng: "Tôi chưa bao giờ uống sữa."',
            "우유를 마시지 않았다고 말했습니다.",
            "SEMANTIC_NEGATION",
            False,
        ),
        (
            'Said, "I have never drunk milk."',
            "우유를 한 번도 마시지 않았다고 말했습니다.",
            "SEMANTIC_NEGATION",
            True,
        ),
        (
            'Mengatakan, "Saya tidak keluar rumah hari ini."',
            "오늘 집 밖으로 나가지 않겠다고 말했습니다.",
            "SEMANTIC_ADDED_FACTS",
            False,
        ),
        (
            'Said, "I will not go outside today."',
            "오늘 밖에 나가지 않겠다고 말했습니다.",
            "SEMANTIC_ADDED_FACTS",
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


@pytest.mark.parametrize(
    "text,uncertain",
    [
        ("걸은 후 목말라 보였습니다.", True),
        ("걸은 후 목이 말라 보였습니다.", True),
        ("목마른 듯합니다.", True),
        ("걸은 후 목이 말랐습니다.", False),
        ("물이 보였습니다.", False),
        ("탈수 증상이 보입니다.", False),
    ],
)
def test_thirst_appearance_preserves_uncertainty_without_accepting_visibility(text, uncertain):
    assert lexical_signals(text)[1] is uncertain


@pytest.mark.parametrize(
    "source,language,target,needs_check",
    [
        ("Seemed thirsty after walking.", "en", "걸은 후 목말라 보였습니다.", True),
        ("Có vẻ khát sau khi đi bộ.", "vi", "걸은 후 목말라 보였습니다.", True),
        ("Tampak haus setelah berjalan.", "id", "걸은 후 목말라 보였습니다.", True),
        ("Drank water.", "en", "물을 마셨습니다.", False),
    ],
)
def test_model_cannot_omit_confirmation_for_explicit_source_uncertainty(
    config, source, language, target, needs_check
):
    provider = model_provider(
        config,
        lambda request: model_response(
            [
                {
                    "segment_id": "s001",
                    "translation": target,
                    "category": "hydration",
                    "needs_confirmation": False,
                }
            ]
        ),
    )
    draft = asyncio.run(
        DraftService(provider).create(
            DraftRequest(memo=source, source_language=language, data_use="synthetic_deidentified")
        )
    )
    assert bool(draft.additional_checks) is needs_check
    assert draft.translation.text == target
    if needs_check:
        assert draft.additional_checks[0].evidence.quote == source
