import sys
import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from app.audio import AudioRejected
from app.carebridge.local import LocalTranslationProvider
from app.carebridge.provider import InferenceTimeout
from app.inference import GPU_INFERENCE_LOCK
from app.stt import Transcriber


@pytest.mark.parametrize("lock_name", ["gpu", "transcriber"])
def test_stt_queue_is_bounded_and_does_not_load_model(config, monkeypatch, lock_name):
    engine = Transcriber(config.model_copy(update={"carebridge_queue_timeout_seconds": 0.01}))
    monkeypatch.setattr("app.stt.decode_audio", lambda *_: [0.1])
    lock = GPU_INFERENCE_LOCK if lock_name == "gpu" else engine._lock
    entered, release = threading.Event(), threading.Event()

    def hold():
        with lock:
            entered.set()
            release.wait(5)

    thread = threading.Thread(target=hold)
    thread.start()
    assert entered.wait(2)
    try:
        with pytest.raises(AudioRejected) as error:
            engine.transcribe(b"audio", language="en")
        assert error.value.code == "STT_BUSY"
        assert error.value.status == 503
        assert engine._model is None
    finally:
        release.set()
        thread.join()
    assert engine._lock.acquire(timeout=0.01)
    engine._lock.release()


@pytest.mark.parametrize("timeout", [False, True])
def test_incomplete_stt_never_returns_partial_transcript(config, monkeypatch, timeout):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setattr("app.stt.decode_audio", lambda *_: [0.1])
    engine = Transcriber(config)
    ticks = iter([0, config.stt_timeout_seconds + 1 if timeout else 1])
    monkeypatch.setattr("app.stt.time.monotonic", lambda: next(ticks))
    engine._model = SimpleNamespace(
        generate=lambda *a, **k: [[1, 2]],
        generation_config=SimpleNamespace(eos_token_id=99),
    )
    engine._processor = SimpleNamespace(
        feature_extractor=lambda *a, **k: SimpleNamespace(
            input_features=SimpleNamespace(to=lambda _: None)
        ),
        tokenizer=SimpleNamespace(batch_decode=lambda *a, **k: pytest.fail("must not decode")),
    )
    with pytest.raises(AudioRejected) as error:
        engine.transcribe(b"audio", language="ko")
    assert error.value.code == ("STT_TIMEOUT" if timeout else "STT_INCOMPLETE")
    assert engine._lock.acquire(timeout=0.01)
    engine._lock.release()


def test_local_timeout_never_returns_even_completed_text(config, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    ticks = iter([0, config.carebridge_timeout_seconds + 1])
    monkeypatch.setattr("app.carebridge.local.time.monotonic", lambda: next(ticks))
    provider = LocalTranslationProvider(config)
    tensor = SimpleNamespace(shape=[1, 20], to=lambda _: None)

    class Output:
        def __getitem__(self, _):
            return [1, 99]

    provider._model = SimpleNamespace(
        device="cuda",
        generation_config=SimpleNamespace(eos_token_id=99),
        generate=lambda **k: Output(),
    )
    provider._tokenizer = SimpleNamespace(
        apply_chat_template=lambda *a, **k: {"input_ids": tensor},
        eos_token_id=99,
        decode=lambda *a, **k: pytest.fail("must not decode"),
    )
    with pytest.raises(InferenceTimeout):
        provider._complete("system", {})
