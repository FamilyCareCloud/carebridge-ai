"""Regressions from new-object controls in real GPU evaluation v10."""

import json

import pytest

from app.carebridge import semantic
from app.carebridge.facts import explicit_facts_preserved
from app.carebridge.guidance import source_hints
from app.carebridge.workflow import pipeline_info
from tests.test_gpu_regressions import make_draft
from tests.test_local_gpu import GOOD, local_client


@pytest.mark.parametrize(
    "language,source,phrase",
    [
        ("en", 'Said, "I did not drink juice."', "juice"),
        ("vi", 'Nói rằng: "Tôi không uống nước ép."', "nước ép"),
        ("id", 'Mengatakan, "Saya tidak minum jus."', "jus"),
    ],
)
def test_juice_hint_is_bound_to_an_actual_source_phrase(language, source, phrase):
    assert {"source_phrase": phrase, "meaning": "주스"} in source_hints(source, language)


def test_reported_drinking_negation_is_not_experiential():
    hints = source_hints('Said, "I did not drink juice."', "en")
    assert {"source_phrase": "did not drink", "meaning": "마시지 않았다"} in hints
    assert not any(
        h["meaning"] == "마시지 않았다"
        for h in source_hints('Said, "I have never drunk juice."', "en")
    )


@pytest.mark.parametrize("source", ["just resting", "juicebox", "nước", "ajus"])
def test_juice_hint_does_not_match_substrings_or_water(source):
    assert not any(h["meaning"] == "주스" for h in source_hints(source))


@pytest.mark.parametrize(
    "source,target,passed",
    [
        ('Said, "I did not drink juice."', "주스를 마시지 않았다고 말했습니다.", True),
        ('Said, "I did not drink juice."', "우유를 마시지 않았다고 말했습니다.", False),
        ('Nói rằng: "Tôi không uống nước ép."', "주스를 마시지 않았다고 말했습니다.", True),
        ('Nói rằng: "Tôi không uống nước ép."', "수프를 마시지 않았다고 말했습니다.", False),
        ('Mengatakan, "Saya tidak minum jus."', "쥬스를 마시지 않았다고 말했습니다.", True),
        ('Mengatakan, "Saya tidak minum jus."', "물을 마시지 않았다고 말했습니다.", False),
        (
            'Nói rằng: "Tôi có thể đã uống nước ép."',
            "수프를 마셨을 수도 있다고 말했습니다.",
            False,
        ),
    ],
)
def test_model_pass_cannot_hide_a_changed_juice_object(
    config, headers, monkeypatch, source, target, passed
):
    draft = make_draft(config, headers, source, target)
    client, _ = local_client(config, json.dumps({"segments": [GOOD]}), monkeypatch)
    with client:
        result = client.post("/carebridge/qa/semantic", headers=headers, json=draft).json()
    check = next(c for c in result["qa"]["checks"] if c["code"] == "SEMANTIC_OMITTED_FACTS")
    assert check["passed"] is passed
    assert result["status"] == ("review_required" if passed else "hold")
    assert not result["qa"]["external_share_allowed"]


def test_object_tripwire_does_not_change_unrelated_or_korean_notes():
    assert explicit_facts_preserved("Drank water.", "물을 마셨습니다.")
    assert explicit_facts_preserved("주스를 마셨습니다.", "주스를 마셨습니다.")


def test_object_tripwires_are_included_in_the_semantic_fingerprint(config, monkeypatch):
    before = pipeline_info(config).semantic_prompt_sha256
    monkeypatch.setattr(semantic, "FACT_MARKERS", [])
    assert pipeline_info(config).semantic_prompt_sha256 != before
