"""Optional local CUDA backend. Importing this module does not import torch."""

import json
import logging
import time

from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from app.carebridge.guidance import translation_payload
from app.carebridge.provider import (
    InferenceTimeout,
    InvalidModelOutput,
    ModelUnavailable,
    translation_system_prompt,
)
from app.carebridge.schemas import ModelDraft, SourceSegment
from app.config import Settings
from app.inference import GPU_INFERENCE_LOCK

logger = logging.getLogger(__name__)


class LocalTranslationProvider:
    """4-bit causal LM, bounded context/output, no HTTP or hidden fallback."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._tokenizer = None
        self._model = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if not GPU_INFERENCE_LOCK.acquire(timeout=self.settings.carebridge_queue_timeout_seconds):
            raise ModelUnavailable("GPU가 다른 요청을 처리 중입니다. 잠시 후 다시 시도해 주세요.")
        try:
            self._load()
        finally:
            GPU_INFERENCE_LOCK.release()

    def _load(self) -> None:
        if self.loaded:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            if self.settings.device != "cuda" or not torch.cuda.is_available():
                raise ModelUnavailable("로컬 번역에는 CUDA GPU가 필요합니다.")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            options = {
                "local_files_only": self.settings.carebridge_local_files_only,
                "trust_remote_code": False,
            }
            tokenizer = AutoTokenizer.from_pretrained(
                self.settings.carebridge_local_model, **options
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.settings.carebridge_local_model,
                **options,
                use_safetensors=True,
                device_map={"": "cuda:0"},
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=dtype,
                ),
            ).eval()
            self._tokenizer = tokenizer
            self._model = model
        except ModelUnavailable:
            raise
        except Exception as exc:
            logger.warning("Local model loading failed: %s", type(exc).__name__)
            raise ModelUnavailable(
                "로컬 모델을 불러오지 못했습니다. GPU 패키지와 모델 캐시를 확인해 주세요."
            ) from exc

    def _complete(self, system: str, payload: dict) -> str:
        config = self.settings
        if not GPU_INFERENCE_LOCK.acquire(timeout=config.carebridge_queue_timeout_seconds):
            raise ModelUnavailable("GPU가 다른 요청을 처리 중입니다. 잠시 후 다시 시도해 주세요.")
        try:
            self._load()
            import torch

            encoded = self._tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
                return_dict=True,
            )
            size = encoded["input_ids"].shape[-1]
            if size > config.carebridge_local_max_input_tokens:
                raise InvalidModelOutput("메모가 모델 문맥 제한을 초과했습니다. 짧게 나눠 주세요.")
            encoded = {key: value.to(self._model.device) for key, value in encoded.items()}
            started = time.monotonic()
            with torch.inference_mode():
                output = self._model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=config.carebridge_local_max_new_tokens,
                    max_time=config.carebridge_timeout_seconds,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            # max_time stops decoding; it cannot cancel a worker thread. Keep the
            # lock until generation returns, even when a client has disconnected.
            generated = output[0, size:]
            eos = self._model.generation_config.eos_token_id
            eos_ids = eos if isinstance(eos, list) else [eos]
            if time.monotonic() - started > config.carebridge_timeout_seconds:
                raise InferenceTimeout(
                    "로컬 모델 응답 시간이 초과되었습니다. 원문을 유지하고 다시 시도해 주세요."
                )
            if not len(generated) or int(generated[-1]) not in eos_ids:
                raise InvalidModelOutput(
                    "로컬 모델 응답이 제한 시간·출력 길이 안에 끝나지 않았습니다."
                )
            content = self._tokenizer.decode(generated, skip_special_tokens=True)
            if len(content.encode("utf-8")) > 128 * 1024:
                raise InvalidModelOutput("로컬 모델 응답이 너무 큽니다.")
            return content
        except (InvalidModelOutput, ModelUnavailable):
            raise
        except Exception as exc:
            logger.warning("Local inference failed: %s", type(exc).__name__)
            raise ModelUnavailable("로컬 GPU 추론을 완료하지 못했습니다.") from exc
        finally:
            GPU_INFERENCE_LOCK.release()

    async def generate(self, segments: list[SourceSegment], source_language: str) -> ModelDraft:
        payload = translation_payload(segments, source_language)
        raw = await run_in_threadpool(
            self._complete,
            translation_system_prompt(payload)
            + "\nSchema: "
            + json.dumps(ModelDraft.model_json_schema()),
            payload,
        )
        try:
            result = ModelDraft.model_validate_json(raw)
        except ValidationError as exc:
            raise InvalidModelOutput("로컬 모델의 번역 형식이 올바르지 않습니다.") from exc
        if [s.segment_id for s in result.segments] != [s.segment_id for s in segments]:
            raise InvalidModelOutput("번역 결과의 원문 구간이 누락·중복·변경되었습니다.")
        return result
