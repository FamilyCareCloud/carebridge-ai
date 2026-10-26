import threading

import pytest
from fastapi.testclient import TestClient

from app.audio import AudioRejected
from app.main import create_app
from tests.conftest import FakeProvider


class FakeTranscriber:
    device = "cpu"

    def __init__(self, text="아침 식사를 절반 드셨습니다.", error=None):
        self.text = text
        self.error = error
        self.calls = []
        self.loaded = False

    def load(self):
        self.loaded = True

    def transcribe(self, data, *, language="korean"):
        self.calls.append((data, language, threading.current_thread().name))
        if self.error:
            raise self.error
        return self.text


def make(config, engine):
    return TestClient(
        create_app(config.model_copy(update={"stt_enabled": True}), transcriber=engine)
    )


def test_existing_transcribe_contract_and_worker_thread(config, headers):
    engine = FakeTranscriber()
    with make(config, engine) as client:
        response = client.post(
            "/transcribe", headers=headers, files={"file": ("sample.wav", b"audio", "audio/wav")}
        )
        assert client.get("/health").json()["device"] == "cpu"
    assert engine.loaded
    assert response.status_code == 200
    assert response.json() == {"text": engine.text}
    assert engine.calls[0][:2] == (b"audio", "korean")
    assert "worker" in engine.calls[0][2].lower()


def test_audio_generates_draft_with_transcript_warning(config, headers):
    engine = FakeTranscriber()
    with make(config, engine) as client:
        response = client.post(
            "/carebridge/drafts/audio",
            headers=headers,
            files={"file": ("sample.webm", b"audio", "audio/webm")},
            data={"data_use": "synthetic_deidentified", "source_language": "ko"},
        )
    assert response.status_code == 200
    draft = response.json()
    assert draft["source"]["kind"] == "audio"
    assert draft["source"]["text"] == engine.text
    assert "TRANSCRIPT_REVIEW_REQUIRED" in [i["code"] for i in draft["qa"]["issues"]]
    assert engine.calls[0][1] == "ko"
    assert draft["qa"]["external_share_allowed"] is False


def test_auto_multilingual_audio_passes_language_detection_to_stt(config, headers):
    engine = FakeTranscriber("Ate half the breakfast.")
    with TestClient(
        create_app(
            config.model_copy(update={"stt_enabled": True}),
            transcriber=engine,
            provider=FakeProvider(),
        )
    ) as client:
        response = client.post(
            "/carebridge/drafts/audio",
            headers=headers,
            files={"file": ("sample.wav", b"audio")},
            data={"data_use": "synthetic_deidentified"},
        )
    assert response.status_code == 200
    assert engine.calls[0][1] is None


@pytest.mark.parametrize(
    "data,status",
    [
        ({}, 422),
        ({"data_use": "real_patient"}, 422),
        ({"data_use": "synthetic_deidentified", "source_language": "xx"}, 422),
    ],
)
def test_audio_attestation_and_language_checked_before_inference(config, headers, data, status):
    engine = FakeTranscriber()
    with make(config, engine) as client:
        response = client.post(
            "/carebridge/drafts/audio",
            headers=headers,
            files={"file": ("a.wav", b"audio")},
            data=data,
        )
    assert response.status_code == status
    assert engine.calls == []


def test_audio_auth_and_size_limit_precede_inference(config, headers):
    engine = FakeTranscriber()
    config.max_audio_bytes = 4
    with make(config, engine) as client:
        assert client.post("/transcribe", files={"file": ("a.wav", b"audio")}).status_code == 401
        assert (
            client.post(
                "/carebridge/drafts/audio",
                files={"file": ("a.wav", b"audio")},
                data={"data_use": "synthetic_deidentified"},
            ).status_code
            == 401
        )
        assert (
            client.post(
                "/transcribe", headers=headers, files={"file": ("a.wav", b"audio")}
            ).status_code
            == 413
        )
        assert (
            client.post("/transcribe", headers=headers, files={"file": ("a.wav", b"")}).status_code
            == 400
        )
    assert engine.calls == []


@pytest.mark.parametrize(
    "error,status,hidden",
    [
        (RuntimeError("C:/secret/path"), 500, "secret"),
        (AudioRejected("INVALID_AUDIO", "오디오를 읽을 수 없습니다."), 422, None),
        (AudioRejected("AUDIO_TOO_LONG", "너무 깁니다.", 413), 413, None),
    ],
)
def test_audio_failures_have_bounded_safe_errors(config, headers, error, status, hidden):
    with make(config, FakeTranscriber(error=error)) as client:
        response = client.post("/transcribe", headers=headers, files={"file": ("a.wav", b"audio")})
    assert response.status_code == status
    if hidden:
        assert hidden not in response.text


def test_silence_and_transcript_identifiers_are_not_a_draft(config, headers):
    for text, code in [(" ", "NO_SPEECH"), ("이름: 홍길동", "IDENTIFYING_INFORMATION")]:
        with make(config, FakeTranscriber(text)) as client:
            response = client.post(
                "/carebridge/drafts/audio",
                headers=headers,
                files={"file": ("a.wav", b"audio")},
                data={"data_use": "synthetic_deidentified"},
            )
        assert response.status_code == 422
        assert response.json()["code"] == code


def test_text_only_health_and_stt_unavailable(client, headers):
    assert client.get("/health").json()["stt_enabled"] is False
    assert (
        client.post("/transcribe", headers=headers, files={"file": ("a.wav", b"audio")}).status_code
        == 503
    )
