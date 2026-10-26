import logging
from contextlib import nullcontext
from threading import Lock

from app.audio import SAMPLE_RATE, AudioRejected, decode_audio
from app.config import Settings, settings
from app.inference import GPU_INFERENCE_LOCK

logger = logging.getLogger(__name__)


class Transcriber:
    """Whisper + LoRA. Heavy imports are deferred for text-only installations."""

    def __init__(self, config: Settings = settings) -> None:
        self.settings = config
        self._processor = None
        self._model = None
        self._device = config.device
        self._lock = Lock()

    def load(self) -> None:
        with self._lock, GPU_INFERENCE_LOCK:
            self._load()

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self._device = self.settings.device if torch.cuda.is_available() else "cpu"
        if self.settings.device == "cuda" and self._device == "cpu":
            logger.warning("CUDA를 사용할 수 없어 CPU로 대체합니다.")
        processor = WhisperProcessor.from_pretrained(self.settings.adapter_path)
        base_model = WhisperForConditionalGeneration.from_pretrained(self.settings.base_model)
        model = PeftModel.from_pretrained(base_model, self.settings.adapter_path)
        self._model = model.to(self._device).eval()
        self._processor = processor
        logger.info("STT 모델 로드 완료: device=%s", self._device)

    @property
    def device(self) -> str:
        return self._device

    def transcribe(self, audio_bytes: bytes, *, language: str | None = "korean") -> str:
        audio = decode_audio(audio_bytes, self.settings.max_audio_seconds)

        # GPU inference is serialized, and callers run it outside the event loop.
        with self._lock, GPU_INFERENCE_LOCK:
            self._load()
            import torch

            inputs = self._processor.feature_extractor(
                audio,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
            ).input_features.to(self._device)
            # The senior LoRA was trained on Korean commands, not multilingual
            # free-form observation notes. Auto/non-Korean input uses base Whisper.
            senior = language is not None and language.lower().split("-")[0] in {"ko", "korean"}
            adapter = nullcontext() if senior else self._model.disable_adapter()
            with adapter, torch.inference_mode():
                generated_ids = self._model.generate(
                    inputs, language=language, task=self.settings.task
                )
            text = self._processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
                0
            ].strip()
        if not text:
            raise AudioRejected("NO_SPEECH", "음성에서 말을 인식하지 못했습니다.")
        return text


transcriber = Transcriber()
