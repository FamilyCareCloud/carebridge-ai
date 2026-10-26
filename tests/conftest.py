import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.carebridge.provider import TranslationProvider
from app.carebridge.schemas import ModelDraft, ModelSegment
from app.config import Settings
from app.main import create_app


@pytest.fixture
def config():
    return Settings(
        stt_enabled=False,
        api_key="test-secret",
        _env_file=None,
        carebridge_model_url="",
        carebridge_model="",
    )


@pytest.fixture
def headers():
    return {"X-API-Key": "test-secret"}


@pytest.fixture
def client(config):
    with TestClient(create_app(config)) as client:
        yield client


class FakeProvider:
    def __init__(self, translations=None, *, needs_confirmation=False):
        self.translations = translations or ["아침 식사를 절반 드셨습니다."]
        self.needs_confirmation = needs_confirmation
        self.calls = []

    async def generate(self, segments, language):
        self.calls.append((segments, language))
        return ModelDraft(
            segments=[
                ModelSegment(
                    segment_id=s.segment_id,
                    translation=t,
                    category="meal",
                    needs_confirmation=self.needs_confirmation,
                )
                for s, t in zip(segments, self.translations, strict=True)
            ]
        )


def model_provider(config, handler):
    configured = config.model_copy(
        update={
            "carebridge_model_url": "https://translation.example/v1/chat/completions",
            "carebridge_model": "test-model",
        }
    )
    return TranslationProvider(configured, transport=httpx.MockTransport(handler))


def model_response(segments, **extra):
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps({"segments": segments, **extra}, ensure_ascii=False),
                    },
                }
            ]
        },
    )
