"""Source revisions, immutable generation snapshots, and fail-closed human-edit QA."""

import copy
import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.carebridge.provider import InferenceTimeout, ModelUnavailable
from app.carebridge.schemas import Draft
from app.carebridge.workflow import draft_digest
from app.main import create_app
from tests.conftest import FakeProvider
from tests.test_local_gpu import GOOD, local_client


def generate(client, headers, **extra):
    return client.post(
        "/carebridge/generations",
        headers=headers,
        json={
            "request_id": str(uuid4()),
            "source_revision": 1,
            "memo": "🙂 물을 마셨습니다.\r\n  도움을 받아 걸었습니다.  ",
            "source_language": "ko",
            "data_use": "synthetic_deidentified",
            **extra,
        },
    )


def edit_request(generation, **extra):
    return {
        "data_use": "synthetic_deidentified",
        "request_id": str(uuid4()),
        "generation": copy.deepcopy(generation),
        "expected_source_revision": generation["source_revision"],
        "expected_source_sha256": generation["draft"]["source"]["sha256"],
        "editor": "author",
        "segments": [
            {
                "segment_id": s["evidence"]["segment_id"],
                "text": s["text"],
                "category": o["category"],
            }
            for s, o in zip(
                generation["draft"]["translation"]["segments"],
                generation["draft"]["observations"],
                strict=True,
            )
        ],
        **extra,
    }


def test_generation_binds_source_revision_request_and_exact_unicode(client, headers):
    request_id = str(uuid4())
    value = generate(client, headers, request_id=request_id).json()
    assert value["origin"] == "ai_generated"
    assert value["request_id"] == request_id
    assert value["source_revision"] == 1
    assert value["draft_sha256"] == draft_digest(Draft.model_validate(value["draft"]))
    for segment in value["draft"]["translation"]["segments"]:
        ref = segment["evidence"]
        assert value["draft"]["source"]["text"][ref["start"] : ref["end"]] == ref["quote"]
    assert len(value["pipeline"]["translation_prompt_sha256"]) == 64
    assert client.get("/health").json()["carebridge_workflow_version"] == "carebridge.workflow.v1"


def test_regeneration_has_new_hash_and_does_not_change_old_snapshot(client, headers):
    original = generate(client, headers).json()
    preserved = copy.deepcopy(original)
    new = generate(client, headers, source_revision=2, memo="물을 마시지 않았습니다.").json()
    assert original == preserved
    assert original["draft"]["source"]["sha256"] != new["draft"]["source"]["sha256"]
    assert original["draft"]["id"] != new["draft"]["id"]
    payload = edit_request(
        original,
        expected_source_revision=2,
        expected_source_sha256=new["draft"]["source"]["sha256"],
    )
    response = client.post("/carebridge/edits/qa", headers=headers, json=payload)
    assert response.status_code == 409
    assert response.json()["code"] == "SOURCE_REVISION_CONFLICT"


@pytest.mark.parametrize("editor", ["author", "reviewer"])
def test_human_edit_keeps_generation_and_rebuilds_consistent_evidence(
    client, config, headers, monkeypatch, editor
):
    original = generate(client, headers).json()
    before = copy.deepcopy(original)
    payload = edit_request(original, editor=editor)
    payload["segments"][0]["text"] = "🙂 물을 마셨다."
    payload["segments"][0]["category"] = "hydration"
    local, calls = local_client(
        config, json.dumps({"segments": [GOOD, {**GOOD, "segment_id": "s002"}]}), monkeypatch
    )
    with local:
        response = local.post("/carebridge/edits/qa", headers=headers, json=payload)
    assert response.status_code == 200
    result = response.json()
    assert original == before == payload["generation"]
    assert result["origin"] == editor
    assert result["original_draft_sha256"] == original["draft_sha256"]
    assert result["generation_id"] == original["draft"]["id"]
    assert result["draft"]["id"] != result["generation_id"]
    assert result["draft"]["source"] == original["draft"]["source"]
    assert (
        result["draft"]["translation"]["segments"][0]["text"]
        == result["draft"]["observations"][0]["text"]
    )
    assert result["changes"][0]["before"] == "🙂 물을 마셨습니다."
    assert result["changes"][0]["evidence"] == original["draft"]["observations"][0]["evidence"]
    assert result["draft_sha256"] == draft_digest(Draft.model_validate(result["draft"]))
    assert result["draft"]["status"] == "review_required"
    assert result["review"]["model_checked"] is True
    assert result["draft"]["qa"]["external_share_allowed"] is False
    assert len(calls) == 1
    assert calls[0][1]["pairs"][0]["translation"] == "🙂 물을 마셨다."


@pytest.mark.parametrize(
    "mutation,code",
    [
        (lambda p: p.update(expected_source_revision=2), "SOURCE_REVISION_CONFLICT"),
        (lambda p: p.update(expected_source_sha256="0" * 64), "SOURCE_REVISION_CONFLICT"),
        (
            lambda p: p["generation"]["draft"]["source"].update(text="변경된 원문"),
            "GENERATION_SNAPSHOT_CONFLICT",
        ),
        (
            lambda p: p["generation"]["draft"]["translation"].update(text="변경된 번역"),
            "GENERATION_SNAPSHOT_CONFLICT",
        ),
        (
            lambda p: p["generation"]["draft"]["qa"]["checks"].clear(),
            "GENERATION_SNAPSHOT_CONFLICT",
        ),
        (lambda p: p["segments"].reverse(), "EDIT_SEGMENTS_MISMATCH"),
        (lambda p: p["segments"].pop(), "EDIT_SEGMENTS_MISMATCH"),
        (lambda p: p["segments"].append(copy.deepcopy(p["segments"][0])), "EDIT_SEGMENTS_MISMATCH"),
        (lambda p: p["segments"][0].update(segment_id="s003"), "EDIT_SEGMENTS_MISMATCH"),
    ],
)
def test_stale_or_malformed_edits_rejected_before_inference(
    client, config, headers, monkeypatch, mutation, code
):
    payload = edit_request(generate(client, headers).json())
    mutation(payload)
    local, calls = local_client(config, "must not run", monkeypatch)
    with local:
        response = local.post("/carebridge/edits/qa", headers=headers, json=payload)
    assert response.status_code == 409
    assert response.json()["code"] == code
    assert "물을" not in response.text
    assert calls == []


def test_snapshot_hash_does_not_replace_evidence_validation(client, config, headers, monkeypatch):
    payload = edit_request(generate(client, headers).json())
    payload["generation"]["draft"]["translation"]["segments"][0]["evidence"]["start"] += 1
    payload["generation"]["draft_sha256"] = draft_digest(
        Draft.model_validate(payload["generation"]["draft"])
    )
    local, calls = local_client(config, "must not run", monkeypatch)
    with local:
        response = local.post("/carebridge/edits/qa", headers=headers, json=payload)
    assert response.status_code == 409
    assert response.json()["code"] == "INVALID_GENERATION_EVIDENCE"
    assert calls == []


@pytest.mark.parametrize(
    "text", ["걸었습니다.", "2명의 도움을 받아 걸었습니다.", "도움을 받아 걸었 perhaps."]
)
def test_bad_human_edit_is_hold_without_mutating_original(
    client, config, headers, monkeypatch, text
):
    payload = edit_request(generate(client, headers).json())
    payload["segments"][1]["text"] = text
    local, calls = local_client(config, "must not run", monkeypatch)
    with local:
        response = local.post("/carebridge/edits/qa", headers=headers, json=payload)
    result = response.json()
    assert response.status_code == 200
    assert result["draft"]["status"] == "hold"
    assert result["review"]["model_checked"] is False
    assert result["draft"]["qa"]["requires_human_review"] is True
    assert calls == []


@pytest.mark.parametrize(
    "error,status", [(InferenceTimeout("timed out"), 504), (ModelUnavailable("busy"), 503)]
)
def test_edit_model_failure_is_not_a_success(client, config, headers, monkeypatch, error, status):
    payload = edit_request(generate(client, headers).json())
    local, _ = local_client(config, error, monkeypatch)
    with local:
        response = local.post("/carebridge/edits/qa", headers=headers, json=payload)
    assert response.status_code == status
    assert "draft" not in response.json()


def test_retry_is_stateless_and_never_creates_submission(client, headers):
    request_id = str(uuid4())
    a = generate(client, headers, request_id=request_id).json()
    b = generate(client, headers, request_id=request_id).json()
    assert a["request_id"] == b["request_id"]
    assert a["draft"]["source"] == b["draft"]["source"]
    assert a["draft"]["id"] != b["draft"]["id"]
    assert "submission_id" not in a and "submission_id" not in b


@pytest.mark.parametrize(
    "extra",
    [
        {"source_revision": True},
        {"source_revision": 0},
        {"source_revision": "1"},
        {"request_id": "person-name"},
        {"data_use": "real_patient"},
        {"subject_name": "이름"},
        {"memo": "가" * 4001},
    ],
)
def test_generation_rejects_unsupported_contract(client, headers, extra):
    assert generate(client, headers, **extra).status_code == 422


def test_confirmed_transcript_is_text_source_for_backend(client, headers):
    value = generate(client, headers, input_kind="reviewed_transcript").json()
    assert value["input_kind"] == "reviewed_transcript"
    assert value["draft"]["source"]["kind"] == "text"


@pytest.mark.parametrize("path", ["/carebridge/generations", "/carebridge/edits/qa"])
def test_new_workflow_endpoints_require_service_auth(client, path):
    assert client.post(path, json={}).status_code == 401


@pytest.mark.parametrize(
    "translation",
    [
        "점심을 먹은 후 아마도 잠을 자 perhaps.",
        "물을 drank water.",
        "물을 마셨습니다. Đã ăn.",
        "물을 마셨습니다。喝了水",
        "Just drank water.",
    ],
)
def test_partial_foreign_output_is_rejected(config, headers, translation):
    with TestClient(create_app(config, provider=FakeProvider([translation]))) as client:
        response = generate(client, headers, memo="Drank water.", source_language="en")
    assert response.status_code == 502
    assert "draft" not in response.json()


@pytest.mark.parametrize("translation", ["물 150mL를 마셨습니다.", "15:00 KST에 물을 마셨습니다."])
def test_units_and_timezones_are_allowed(config, headers, translation):
    with TestClient(create_app(config, provider=FakeProvider([translation]))) as client:
        assert (
            generate(client, headers, memo="Drank water.", source_language="en").status_code == 200
        )
