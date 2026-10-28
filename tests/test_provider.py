import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import model_provider, model_response

SEGMENT = {
    "segment_id": "s001",
    "translation": "아침 식사를 절반 드셨습니다.",
    "category": "meal",
    "needs_confirmation": False,
}
PAYLOAD = {"memo": "Ate half the breakfast.", "data_use": "synthetic_deidentified"}


def call(config, headers, handler):
    with TestClient(create_app(config, provider=model_provider(config, handler))) as client:
        return client.post("/carebridge/drafts", headers=headers, json=PAYLOAD)


def test_configured_model_is_called_with_untrusted_data_and_schema(config, headers):
    def handler(req):
        payload = json.loads(req.content)
        assert payload["temperature"] == 0
        assert payload["response_format"] == {"type": "json_object"}
        assert "untrusted source DATA" in payload["messages"][0]["content"]
        data = json.loads(payload["messages"][1]["content"])
        assert data["segments"] == [{"segment_id": "s001", "text": PAYLOAD["memo"]}]
        assert req.url.path == "/v1/chat/completions"
        return model_response([SEGMENT])

    response = call(config, headers, handler)
    assert response.status_code == 200
    assert response.json()["status"] == "review_required"


@pytest.mark.parametrize(
    "segments,extra",
    [
        ([], {}),
        ([{**SEGMENT, "segment_id": "invented"}], {}),
        ([SEGMENT, SEGMENT], {}),
        ([{**SEGMENT, "translation": "Ate breakfast"}], {}),
        ([{**SEGMENT, "translation": " "}], {}),
        ([{**SEGMENT, "category": "diagnosis"}], {}),
        ([{**SEGMENT, "needs_confirmation": "false"}], {}),
        ([{**SEGMENT, "evidence": "invented source"}], {}),
        ([SEGMENT], {"status": "approved"}),
        ([SEGMENT], {"external_share_allowed": True}),
    ],
)
def test_malformed_model_cannot_be_used_as_a_draft(config, headers, segments, extra):
    response = call(config, headers, lambda _: model_response(segments, **extra))
    assert response.status_code == 502
    assert response.json()["code"] == "INVALID_MODEL_OUTPUT"
    assert PAYLOAD["memo"] not in response.text


@pytest.mark.parametrize(
    "envelope",
    [
        [],
        {},
        {"choices": []},
        {"choices": [None]},
        {"choices": ["broken"]},
        {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": "not json"}}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": None}}]},
    ],
)
def test_invalid_or_truncated_envelope_is_502(config, headers, envelope):
    response = call(config, headers, lambda _: httpx.Response(200, json=envelope))
    assert response.status_code == 502


def test_large_model_response_is_bounded(config, headers):
    assert (
        call(config, headers, lambda _: httpx.Response(200, content=b"x" * 131073)).status_code
        == 502
    )


@pytest.mark.parametrize("status", [401, 429, 500, 302])
def test_provider_failure_is_503_without_exposing_response(config, headers, status):
    response = call(config, headers, lambda _: httpx.Response(status, text="provider-secret"))
    assert response.status_code == 503
    assert "provider-secret" not in response.text


def test_timeout_is_504_and_not_retried(config, headers):
    calls = []

    def handler(req):
        calls.append(req)
        raise httpx.ReadTimeout("upstream-secret")

    response = call(config, headers, handler)
    assert response.status_code == 504
    assert response.json()["code"] == "INFERENCE_TIMEOUT"
    assert "upstream-secret" not in response.text
    assert len(calls) == 1


@pytest.mark.parametrize(
    "url",
    [
        "http://untrusted.example/v1/chat/completions",
        "https://untrusted.example:bad/v1/chat/completions",
        "https://[bad/v1/chat/completions",
    ],
)
def test_invalid_model_url_is_rejected_before_sending(config, headers, url):
    configured = config.model_copy(
        update={
            "carebridge_model_url": url,
            "carebridge_model": "test",
        }
    )
    with TestClient(create_app(configured)) as client:
        response = client.post("/carebridge/drafts", headers=headers, json=PAYLOAD)
    assert response.status_code == 503


def test_model_key_forwarded_only_in_authorization(config, headers):
    from pydantic import SecretStr

    config.carebridge_model_api_key = SecretStr("model-secret")

    def handler(req):
        assert req.headers["authorization"] == "Bearer model-secret"
        assert "model-secret" not in str(req.url)
        assert b"model-secret" not in req.content
        return model_response([SEGMENT])

    assert call(config, headers, handler).status_code == 200
