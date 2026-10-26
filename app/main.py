import hmac
import logging
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.audio import AudioRejected
from app.carebridge.local import LocalTranslationProvider
from app.carebridge.provider import InvalidModelOutput, ModelUnavailable, TranslationProvider
from app.carebridge.safety import InputRejected, assess
from app.carebridge.schemas import Draft, DraftRequest, Language, QAResult
from app.carebridge.semantic import SemanticReview, review_semantics
from app.carebridge.service import DraftService
from app.config import Settings, settings
from app.stt import Transcriber

logger = logging.getLogger(__name__)
AUDIO_LANGUAGES = {"ko", "en", "vi", "id", "th", "tl", "zh", "ja", "ne", "uz", "ru"}


def create_app(config: Settings = settings, *, transcriber=None, provider=None) -> FastAPI:
    engine = (transcriber or Transcriber(config)) if config.stt_enabled else None
    translator = provider or (
        LocalTranslationProvider(config)
        if config.carebridge_backend == "local"
        else TranslationProvider(config)
    )
    service = DraftService(translator)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if engine is not None:
            await run_in_threadpool(engine.load)
        if config.carebridge_preload and isinstance(translator, LocalTranslationProvider):
            await run_in_threadpool(translator.load)
        yield

    app = FastAPI(
        title="fcc-ai / CareBridge",
        version="0.2.0",
        lifespan=lifespan,
        description="음성 인식과 교육용 비식별 관찰메모 검토 전 초안 API",
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request, exc):
        # FastAPI's default includes rejected input. Do not echo potentially
        # identifying notes, audio or provider output back in validation errors.
        return JSONResponse(
            status_code=422,
            content={
                "code": "INVALID_REQUEST",
                "detail": [{"loc": list(e["loc"]), "type": e["type"]} for e in exc.errors()],
            },
        )

    @app.exception_handler(InputRejected)
    async def input_error(_request, exc):
        return JSONResponse(status_code=422, content={"code": exc.code, "detail": str(exc)})

    @app.exception_handler(AudioRejected)
    async def audio_error(_request, exc):
        return JSONResponse(status_code=exc.status, content={"code": exc.code, "detail": str(exc)})

    @app.exception_handler(ModelUnavailable)
    async def model_error(_request, exc):
        return JSONResponse(
            status_code=503, content={"code": "TRANSLATION_UNAVAILABLE", "detail": str(exc)}
        )

    @app.exception_handler(InvalidModelOutput)
    async def output_error(_request, exc):
        return JSONResponse(
            status_code=502, content={"code": "INVALID_MODEL_OUTPUT", "detail": str(exc)}
        )

    def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        if config.api_key and not hmac.compare_digest(
            (x_api_key or "").encode(), config.api_key.encode()
        ):
            raise HTTPException(status_code=401, detail="유효하지 않은 API 키입니다.")

    async def read_audio(file: UploadFile) -> bytes:
        data = bytearray()
        try:
            while chunk := await file.read(65536):
                data.extend(chunk)
                if len(data) > config.max_audio_bytes:
                    raise AudioRejected(
                        "AUDIO_TOO_LARGE", "오디오 파일 크기 제한을 초과했습니다.", 413
                    )
        finally:
            await file.close()
        if not data:
            raise AudioRejected("EMPTY_AUDIO", "빈 파일입니다.", 400)
        return bytes(data)

    async def transcribe_bytes(data: bytes, language: str | None) -> str:
        if engine is None:
            raise HTTPException(status_code=503, detail="음성 인식이 비활성화되어 있습니다.")
        try:
            result = await run_in_threadpool(engine.transcribe, data, language=language)
        except AudioRejected:
            raise
        except Exception as exc:
            # Do not expose paths, model details, source audio or transcript.
            logger.error("STT 추론 실패: %s", type(exc).__name__)
            raise HTTPException(status_code=500, detail="음성 인식에 실패했습니다.") from exc
        if not isinstance(result, str) or not result.strip():
            raise AudioRejected("NO_SPEECH", "음성에서 말을 인식하지 못했습니다.")
        return result.strip()

    protected = [Depends(require_api_key)]

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "device": engine.device if engine else "disabled",
            "stt_enabled": config.stt_enabled,
            "carebridge_schema": "carebridge.v1",
            "translation_backend": config.carebridge_backend,
            "translation_loaded": getattr(translator, "loaded", False),
            "semantic_check_enabled": config.carebridge_semantic_check_enabled,
        }

    @app.post("/transcribe", dependencies=protected)
    async def transcribe(file: Annotated[UploadFile, File()]):
        # Preserve the backend's multipart input and {text: ...} output contract.
        return {"text": await transcribe_bytes(await read_audio(file), config.language)}

    @app.post("/carebridge/drafts", response_model=Draft, dependencies=protected)
    async def draft(request: DraftRequest):
        return await checked_draft(await service.create(request))

    async def checked_draft(value: Draft) -> Draft:
        if config.carebridge_semantic_check_enabled:
            review = await semantic_review(value)
            value.qa, value.status = review.qa, review.status
        return value

    async def semantic_review(value: Draft) -> SemanticReview:
        if not isinstance(translator, LocalTranslationProvider):
            raise ModelUnavailable("의미 검사는 로컬 GPU 모델 설정이 필요합니다.")
        return await review_semantics(value, translator)

    @app.post("/carebridge/drafts/audio", response_model=Draft, dependencies=protected)
    async def audio_draft(
        file: Annotated[UploadFile, File()],
        data_use: Annotated[Literal["synthetic_deidentified"], Form()],
        source_language: Annotated[Language, Form()] = "auto",
    ):
        code = source_language.split("-")[0]
        if source_language != "auto" and code not in AUDIO_LANGUAGES:
            await file.close()
            raise HTTPException(status_code=422, detail="이 음성 입력 언어는 지원하지 않습니다.")
        text = await transcribe_bytes(await read_audio(file), None if code == "auto" else code)
        if len(text) > 4000:
            raise InputRejected("MEMO_TOO_LONG", "전사 결과는 4,000자 이내여야 합니다.")
        return await checked_draft(
            await service.create(
                DraftRequest(memo=text, source_language=source_language, data_use=data_use),
                audio=True,
            )
        )

    @app.post("/carebridge/qa", response_model=QAResult, dependencies=protected)
    async def qa(draft: Draft):
        if config.carebridge_semantic_check_enabled:
            return (await semantic_review(draft)).qa
        return assess(draft)

    @app.post("/carebridge/qa/semantic", response_model=SemanticReview, dependencies=protected)
    async def semantic_qa(draft: Draft):
        return await semantic_review(draft)

    return app


app = create_app()
