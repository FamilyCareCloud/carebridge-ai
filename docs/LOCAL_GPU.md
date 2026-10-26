# 로컬 GPU CareBridge

기준 코드: `main`의 788d85c(CareBridge 초안·QA API). 오래된 `develop`에는 이 기능이 없으므로
그 위에 중복 API를 만들지 않고 최신 구현을 확장한다.

## 추가된 기능

- `FCC_AI_CAREBRIDGE_BACKEND=local`: HTTP 번역 서버 대신 로컬 Qwen3 8B 4bit 실행.
  추론 입력은 이 프로세스 안에서 처리하며 외부 번역 API로 보내지 않는다.
- `POST /carebridge/qa/semantic`: 원문/한국어 번역을 별도 호출로 비교하고
  부정·불확실성·사실 추가·의료 판단 여부를 검토 보조 신호로 반환한다.
- 기존 구조 검사에도 영어/한국어/베트남어/인도네시아어의 제한된
  부정·추측 표현 신호 비교를 추가했다. 명백한 의미 반전은 모델의 통과 판단보다 우선한다.
- STT는 한국어 명령에 기존 senior LoRA를 사용하고, 자동 감지·비한국어에는
  `disable_adapter()`로 base Whisper를 사용한다. 다국어용 별도 가중치를 추가로 올리지 않는다.
- STT·LLM은 하나의 GPU 추론 잠금을 공유한다. LLM은 대기 한도 초과 시 503을 반환하고,
  문맥을 자르거나 출력이 잘린 JSON을 정상 초안으로 사용하지 않는다.

기존 `/transcribe`, `/carebridge/drafts`, `/carebridge/drafts/audio`의 요청·응답 계약은 유지한다.
서비스 키는 계속 `X-API-Key`로 검사한다.

## 설치

기존 GPU용 Python과 분리된 환경에 설치한다. RTX 50 시리즈는 CUDA 12.8 이상을 지원하는
PyTorch 빌드가 필요하다. PyTorch는 PC에 맞게 설치한 뒤 나머지를 설치한다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

이미 CUDA PyTorch가 있는 환경을 공유하려면 그 Python으로
`-m venv --system-site-packages .venv`를 실행할 수 있다. 부모 환경에 의존하는 개발용 선택지이므로,
다른 PC로 환경 폴더를 복사하지 말고 위 설치 절차를 사용한다.

기본 모델은 `unsloth/Qwen3-8B-unsloth-bnb-4bit`.
기본값은 **캐시된 모델만 사용**한다. 캐시가 없으면 조용히 클라우드로 전환하지 않고 실패한다.
최초 다운로드를 허용하려면 `FCC_AI_CAREBRIDGE_LOCAL_FILES_ONLY=false`를 설정한다.
추가 모델 코드는 실행하지 않으며 safetensors 가중치만 사용한다.
실모델 검증은 위 Qwen 체크포인트로 했으며 임의의 다른 모델 호환성을 보증하지 않는다.

## 실행

저장소 루트에서:

```powershell
.\scripts\start-local-gpu.ps1
```

기본 주소는 `http://127.0.0.1:8002`, 모델을 시작 시 로딩하고 초안 생성 시 의미 검사를 활성화한다.
기존 8000 포트 서버에 영향을 주지 않는다. 서비스는 worker 1개로 실행한다.
STT까지 사용하려면 `models/whisper-senior-lora`를 배치하거나
`FCC_AI_ADAPTER_PATH`에 기존 어댑터 경로를 지정한 뒤:

```powershell
.\scripts\start-local-gpu.ps1 -EnableStt
```

외부 공개와 AWS 배포는 이 실행 스크립트가 수행하지 않는다. 백엔드가 CareBridge API를
연결할 때 사용자/그룹/검토자 권한을 확인하고 서비스 키를 서버에서만 전달해야 한다.
AI 결과를 가족에게 자동 공유하거나 공식기록으로 저장해서는 안 된다.

## 설정

| 환경변수 | 기본값 | 의미 |
| --- | --- | --- |
| `FCC_AI_CAREBRIDGE_BACKEND` | `http` | 기존 HTTP 또는 새 `local` |
| `FCC_AI_CAREBRIDGE_LOCAL_MODEL` | 위 Qwen 모델 ID | 캐시 모델 ID 또는 로컬 디렉터리 |
| `FCC_AI_CAREBRIDGE_LOCAL_FILES_ONLY` | `true` | 모델 다운로드 차단 |
| `FCC_AI_CAREBRIDGE_PRELOAD` | `false` | 시작할 때 모델 로드(실패하면 준비되지 않음) |
| `FCC_AI_CAREBRIDGE_LOCAL_MAX_INPUT_TOKENS` | `8192` | 초과 시 실패, 무단 절단 없음 |
| `FCC_AI_CAREBRIDGE_LOCAL_MAX_NEW_TOKENS` | `2048` | 출력 토큰 상한 |
| `FCC_AI_CAREBRIDGE_QUEUE_TIMEOUT_SECONDS` | `2` | GPU 잠금 대기 시간 |
| `FCC_AI_CAREBRIDGE_SEMANTIC_CHECK_ENABLED` | `false` | 초안 생성·기존 QA에서도 의미 검사 적용(로컬 필수) |
| `FCC_AI_CAREBRIDGE_TIMEOUT_SECONDS` | `30` | 호출별 생성 한도 |

시간 제한은 모델 생성의 중단 조건이다. GPU 커널/모델 로딩을 강제로 취소하는 타이머가 아니며,
실제 생성이 끝나기 전에 잠금을 풀지 않는다. 생성+의미 검사는 두 번의 모델 호출이므로,
백엔드 총 타임아웃은 그 합과 STT 시간을 고려해야 한다.

`/health`는 기존 필드 외에 `translation_backend`, `translation_loaded`,
`semantic_check_enabled`를 제공한다. `device`는 기존 STT 디바이스 필드다.
STT가 꺼져 있어 `device=disabled`여도 로컬 번역 모델은 CUDA에서 동작할 수 있다.

## 의미 검사 API

기존 `/carebridge/qa`와 같이 **전체 Draft**를 JSON으로 전송한다.

```text
POST /carebridge/qa/semantic
X-API-Key: <서비스 키>
Content-Type: application/json
```

응답 필드:

- `status`: `hold` 또는 `review_required`, 승인 상태는 없음.
- `model_checked`: 구조 검사를 먼저 통과해 실제 의미 모델까지 호출했는지.
- `segments[]`: 원문의 정확한 인용·코드 포인트 위치와 항목별 검사 결과.
- `qa`: 재계산한 구조+의미 검사 결과. 사람 검토 필수, 외부 공유 불허.

원문/근거/숫자/부정·추측 검사에서 이미 보류되면 모델을 부르지 않고
`model_checked=false`로 반환한다. 모델 오류나 응답 형식 위반은 503/502이며
검사 성공으로 대체하지 않는다. 기존 `/qa`에도 옵션을 켜면 같은 재검사가 적용된다.

## 실제 GPU 확인

`evaluation/local-gpu-smoke.json`은 이 PC에서 **실제 가중치**로 실행한 결과다.
RTX 5070 Ti 16GB에서 Whisper+Qwen 최대 할당 VRAM 약 8.1GiB.
3개 언어의 합성 메모 번역, 충실한 초안의 재검토 상태, 의도적으로 바꾼
부정/추측/추가 사실의 보류, 합성 영어 음성 경로를 점검했다.
번역 예제의 단일 호출 시간은 약 1.4~2.9초였으며 일반 처리량 측정이 아니다.

최초 실험에서 LLM만으로는 부정 반전을 놓쳤다. 최종 경로에는 독립된 어휘 신호 검사를
추가해 이 사례를 보류하도록 했다. **같은 모델의 재검사는 독립 검증이나 정확도 증명이 아니다.**
예제 번역에도 어색한 한국어가 있으므로 사람이 원문과 비교해야 한다.
어휘 검사는 제한된 패턴으로, 누락·오탐이 있고 다른 언어와 우회 표현을 보장하지 않는다.

재현:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_gpu.py
# 별도 생성한 30초 이하의 교육용 합성 영어 WAV까지 검증:
.\.venv\Scripts\python.exe scripts/evaluate_gpu.py --audio evaluation/synthetic-english.wav --adapter-path models/whisper-senior-lora
```

합성 음성 파일과 모델 가중치는 git에 넣지 않는다. 평가 스크립트의 `passed`는
HTTP 계약·인용·공유 금지·사례별 상태를 확인하는 값이고 번역 정확도 점수가 아니다.
기획서의 60/180/90건 수치는 이 실행 결과로 재사용하지 않는다.

## 전체 서비스 연동

fcc-backend의 CareBridge 서비스와 프론트 케어브릿지 메뉴가 작성자·검토자·가족 권한,
불변 원문/버전, 승인·수정·보류와 승인된 가족 미리보기를 구현한다.
[백엔드의 전체 실행·배포 계약](https://github.com/FamilyCareCloud/fcc-backend/blob/feature/carebridge-workflow/docs/CAREBRIDGE.md)을 참고한다.
세 저장소를 같은 부모 폴더에 둔 뒤 백엔드 scripts/start-carebridge-local.ps1로 함께 실행할 수 있다.
공개 터널·AWS 배포 연결은 별도로 승인·설정해야 하며, 로컬 실행이 클라우드 배포를 의미하지 않는다.
