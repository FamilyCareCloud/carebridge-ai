# CareBridge AI

다국어 음성·텍스트 기반 돌봄 기록 생성 및 검토 지원 AI **CareBridge**의 음성 전사·한국어 초안 생성·원문 대조 QA 서버입니다. 원문, 번역, 관찰 내용과 추가 확인사항을 구분해 담당자의 검토를 돕습니다.

- **팀:** 영남대학교 YU42 · 임현승(프론트엔드), 전채영(백엔드), 김우주(AI)
- **서비스:** https://develop.db2l4u9y804jp.amplifyapp.com/
- **구현·평가 기준:** `main`

## 현재 기능

- 음성 전사 후 작성자가 원문을 확인·수정하는 흐름을 지원합니다.
- 한국어 원문 보존과 영어·베트남어·인도네시아어 메모의 한국어 초안 생성을 지원합니다.
- 원문 인용·위치와 생성 내용을 연결하고 부정·불확실성·사실 누락·추가를 검사합니다.
- 원문 개정, 사람이 편집한 최종 내용의 QA와 의미 검사를 제공합니다.
- 모호하거나 확인되지 않은 정보는 확인 항목으로 남깁니다.

AI 응답은 검토 전 초안이며 공식 기록이 아닙니다. AI 서버는 승인·공개 권한을 부여하지 않습니다. 담당자 확정과 지정 수신자 공개는 백엔드·프론트엔드에서 처리합니다.

## 모델과 데이터

| 항목 | 구성 |
|---|---|
| 서버 | Python · FastAPI · uvicorn |
| 한국어 음성 인식 | `openai/whisper-small` + LoRA (`r=16`, `alpha=32`, `q_proj`, `v_proj`) |
| 비한국어·자동 감지 음성 | 기본 Whisper 다국어 경로, 한국어 LoRA 비활성화 |
| 로컬 번역·초안·의미 검사 | `unsloth/Qwen3-8B-unsloth-bnb-4bit` |
| 음성 학습 데이터 | AI Hub 명령어 음성 노인남녀 데이터 |
| 최종 음성 학습·검증 규모 | 학습 기록 기준 11,996건: 학습 약 10,796건·검증 약 1,200건 |
| 개발·평가 자료 | 합성 메모 94건, 정상·변조 QA 131건 |

번역 모델에는 프로젝트 추가 학습을 수행하지 않았습니다. 입력 안내, 제한된 원문 기반 표현 보정과 구조·의미 검사를 적용합니다. 합성 메모와 QA 자료는 학습 데이터가 아닌 개발·평가 사례입니다.

음성 학습은 정형 명령어 중심으로 자유 발화에 한계가 있습니다. 모델 가중치와 원본 학습 데이터는 Git에 포함하지 않습니다. STT 실행에는 `models/whisper-senior-lora/`에 별도 어댑터를 배치해야 합니다. 학습 스크립트는 저장소에 포함돼 있지 않습니다.

## 로컬 GPU 실행

Python 가상환경과 GPU에 맞는 PyTorch를 준비합니다. 다음은 Windows·CUDA 12.8 환경의 예입니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
Copy-Item .env.example .env
.\scripts\start-local-gpu.ps1
```

기본 주소는 `http://127.0.0.1:8002`입니다. 캐시된 Qwen 모델을 사용하며, 최초 다운로드를 허용하려면 `FCC_AI_CAREBRIDGE_LOCAL_FILES_ONLY=false`를 설정합니다. 어댑터를 준비하고 음성 인식을 함께 실행하려면 다음을 사용합니다.

```powershell
.\scripts\start-local-gpu.ps1 -EnableStt
```

설치·모델 캐시·메모리·실행 설정은 [로컬 GPU 안내](docs/LOCAL_GPU.md)를 참고하세요. STT와 LLM은 GPU 추론 잠금을 공유하며, 문맥·대기 한도 초과나 잘린 응답을 정상 결과로 처리하지 않습니다.

## 텍스트 API 개발 실행

GPU 모델을 로드하지 않고 한국어 원문 보존·API 계약을 확인하는 모드입니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-core.txt
$env:FCC_AI_STT_ENABLED="false"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8002
```

비한국어 번역에는 로컬 GPU 모델 또는 `.env.example`의 OpenAI-compatible 모델 연결이 별도로 필요합니다. 설정이 없거나 요청이 실패하면 정상 번역으로 대신하지 않습니다.

## API와 연결 설정

| 경로 | 용도 |
|---|---|
| `GET /health` | 서버·번역 백엔드·모델 로드 상태 |
| `POST /carebridge/transcriptions` | 작성자가 확인할 음성 전사문 |
| `POST /carebridge/generations` | 원문 개정과 초안 생성 |
| `POST /carebridge/edits/qa` | 사람이 편집한 최종 내용 검사 |
| `POST /carebridge/qa/semantic` | 원문·번역 의미 대조 |
| `POST /carebridge/drafts`, `/carebridge/drafts/audio`, `/carebridge/qa` | Draft 생성·검사 계약 |

`FCC_AI_API_KEY`를 설정하면 POST 요청에 `X-API-Key`가 필요합니다. 외부 연결은 키를 설정하고 백엔드에서만 전달합니다. 백엔드의 `CAREBRIDGE_SERVICE_URL`은 이 서버 주소, `CAREBRIDGE_SERVICE_API_KEY`는 같은 키입니다.

로컬 GPU 모드는 입력을 외부 번역 API로 보내지 않습니다. HTTP 백엔드 모드는 설정한 모델 서버에 요청합니다. AI 서버는 별도 GPU PC·서비스 프로세스에서 실행하며 GitHub Actions는 테스트만 수행합니다. PC·서버·터널이 중단되면 연결한 AI 기능을 사용할 수 없습니다.

## 검증 결과와 한계

현재 전체 평가 자료는 생성 94건·QA 131건입니다. 수정 후 최종 GPU 재검증 v16은 영향받는 원문 6개와 대조군 16개, 관련 QA 41개를 대상으로 수행했습니다.

| 최종 재검증 항목 | 확인 결과 |
|---|---|
| 초안 생성 | 22건 응답·원문 연결·분류·확인 항목·실행 지문 검사 통과 |
| 정상 QA | 22건 모두 담당자 검토 대상으로 유지 |
| 의도적 변조 QA | 19건 모두 보류 |

원본 v16의 `all_passed:false`는 참조문과 표현이 다른 이동 문장 2건을 직접 검토 대상으로 남긴 결과입니다. 별도 원문 대조에서 의미의 동등성을 확인했으며 원본 실행 결과와 이전 실패 기록을 보존합니다. 개발 검증과 코딩 도우미의 대조 검토는 원어민·현장 검증이나 전체 번역 정확도 100%를 뜻하지 않습니다.

- [최종 평가 해석과 적용 범위](docs/REMAINING_QUALITY_2026-10-30.md)
- [v16 실행 원본](evaluation/translation-live-v16.json)
- [v16 원문 대조 기록](evaluation/translation-reference-review-v16.json)
- [전체 평가 입력](evaluation/translation-corpus-v9.json)

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check app tests scripts
python -m ruff format --check app tests scripts
```

CI는 Python 3.10·3.12에서 계약·회귀 테스트와 정적 검사를 수행합니다. 실제 GPU 모델 실행 평가는 별도 기록으로 확인합니다.

## 구성과 문서

| 경로 | 역할 |
|---|---|
| `app/main.py`, `app/config.py` | API 진입점·환경 설정 |
| `app/stt.py`, `app/audio.py` | 음성 인식·오디오 검증 |
| `app/carebridge/` | 원문 근거·초안 생성·편집 QA·의미 검사 |
| `evaluation/`, `scripts/` | 평가 입력·실행 결과·검증 도구 |
| `tests/` | 모델 없이 실행하는 API·안전·회귀 검사 |

- [생성·편집 QA 계약](docs/SPEC_WORKFLOW_V1.md) · [API 안내](docs/CAREBRIDGE.md)
- [로컬 GPU](docs/LOCAL_GPU.md) · [표현·문맥 검증](docs/TRANSLATION_CONTEXT_V1.md)
- [프론트엔드](https://github.com/FamilyCareCloud/carebridge-frontend) · [백엔드](https://github.com/FamilyCareCloud/carebridge-backend)

세부 문서에는 작성 당시의 구현 기준과 검증 시점이 함께 기재돼 있습니다.
