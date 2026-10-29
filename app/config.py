from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FCC_AI_", env_file=".env", extra="ignore")

    base_model: str = "openai/whisper-small"
    adapter_path: str = "models/whisper-senior-lora"
    device: Literal["cuda", "cpu"] = "cuda"
    language: str = "korean"
    task: str = "transcribe"
    host: str = "0.0.0.0"
    port: int = 8000
    # 공인 터널(Cloudflare Tunnel/ngrok 등)로 노출할 때 설정. 비워두면 인증 없이 허용(로컬 전용 실행 가정).
    api_key: str = ""
    stt_enabled: bool = True
    max_audio_bytes: int = Field(default=4 * 1024 * 1024, ge=1, le=32 * 1024 * 1024)
    max_audio_seconds: float = Field(default=30, gt=0, le=30)
    stt_timeout_seconds: float = Field(default=10, gt=0, le=120)
    stt_max_new_tokens: int = Field(default=384, ge=32, le=400)
    # OpenAI-compatible chat completions endpoint; used only for non-Korean text.
    carebridge_model_url: str = ""
    carebridge_model: str = ""
    carebridge_model_api_key: SecretStr = SecretStr("")
    carebridge_timeout_seconds: float = Field(default=30, gt=0, le=120)
    carebridge_backend: Literal["http", "local"] = "http"
    carebridge_local_model: str = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
    carebridge_local_files_only: bool = True
    carebridge_local_max_input_tokens: int = Field(default=8192, ge=512, le=16384)
    carebridge_local_max_new_tokens: int = Field(default=2048, ge=128, le=8192)
    carebridge_queue_timeout_seconds: float = Field(default=2, gt=0, le=30)
    carebridge_semantic_check_enabled: bool = False
    carebridge_preload: bool = False


settings = Settings()
