import io

SAMPLE_RATE = 16000


class AudioRejected(ValueError):
    def __init__(self, code: str, message: str, status: int = 422):
        self.code = code
        self.status = status
        super().__init__(message)


def decode_audio(audio_bytes: bytes, max_seconds: float):
    import librosa
    import numpy as np

    try:
        # Read one extra sample to detect overflow without decoding an entire
        # compressed upload or silently truncating at Whisper's context window.
        audio, _ = librosa.load(
            io.BytesIO(audio_bytes),
            sr=SAMPLE_RATE,
            mono=True,
            duration=max_seconds + 1 / SAMPLE_RATE,
        )
    except Exception as exc:
        raise AudioRejected("INVALID_AUDIO", "오디오를 읽을 수 없습니다.") from exc
    if len(audio) > int(max_seconds * SAMPLE_RATE):
        raise AudioRejected("AUDIO_TOO_LONG", "음성은 설정된 최대 길이 이내로 녹음해 주세요.", 413)
    if len(audio) < SAMPLE_RATE // 10 or not np.isfinite(audio).all():
        raise AudioRejected("INVALID_AUDIO", "유효한 음성을 녹음해 주세요.")
    if float(np.sqrt(np.mean(audio**2))) < 0.0005:
        raise AudioRejected("NO_SPEECH", "음성에서 말을 인식하지 못했습니다.")
    return audio
