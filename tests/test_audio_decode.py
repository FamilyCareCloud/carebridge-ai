import io
import math
import struct
import wave

import pytest

from app.audio import SAMPLE_RATE, AudioRejected, decode_audio


def wav(seconds=1, rate=16000, channels=1, amplitude=5000):
    output = io.BytesIO()
    with wave.open(output, "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(2)
        f.setframerate(rate)
        samples = [
            int(amplitude * math.sin(2 * math.pi * 440 * n / rate))
            for n in range(round(seconds * rate))
            for _ in range(channels)
        ]
        f.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return output.getvalue()


def test_decodes_and_resamples_real_stereo_wav():
    audio = decode_audio(wav(rate=8000, channels=2), 30)
    assert audio.ndim == 1
    assert len(audio) == SAMPLE_RATE
    assert audio.dtype.name == "float32"


@pytest.mark.parametrize(
    "data,code,status",
    [
        (b"not audio", "INVALID_AUDIO", 422),
        (wav(amplitude=0), "NO_SPEECH", 422),
        (wav(seconds=0.05), "INVALID_AUDIO", 422),
        (wav(seconds=1.001), "AUDIO_TOO_LONG", 413),
    ],
    ids=["corrupt", "silence", "too_short", "too_long"],
)
def test_rejects_invalid_silent_short_and_long_audio(data, code, status):
    with pytest.raises(AudioRejected) as result:
        decode_audio(data, 1)
    assert result.value.code == code
    assert result.value.status == status


def test_exact_limit_is_not_truncated_or_rejected():
    assert len(decode_audio(wav(), 1)) == SAMPLE_RATE
