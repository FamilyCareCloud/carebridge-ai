# CareBridge 검토 전 초안 API

로컬 GPU 번역과 의미 검사 확장은 [LOCAL_GPU.md](LOCAL_GPU.md)를 참조하세요.

기준: 사용자 제공 「케어브릿지: 다국어 돌봄 관찰메모 안전검토 AI」 3쪽 명세서.
기존 FCC 음성 질의 서버에 교육용 가상·비식별 메모 처리 기능을 추가한다.
이는 의료기록 작성, 진단, 처방, 응급판단, 검토자 승인 시스템이 아니다.

## 요구사항과 구현 경계

| 명세서 요구사항 | 이 레포의 구현 | 연동 시스템에 필요한 작업 |
| --- | --- | --- |
| 모국어·한국어의 짧은 메모 | 텍스트 입력, 한국어 원문 보존, 다국어 모델 번역 | 입력 화면 |
| 전사·번역·관찰/추가확인 구조화 | 음성 → 전사 → 같은 초안 파이프라인; 근거별 관찰·추가확인 항목 | 전사 결과를 작성자가 수정할 화면 |
| 원문·번역문·초안 동시 표시 | 세 부분을 별도 필드로 반환, 모든 항목에 원문 인용·위치 | 나란히 비교하는 검토 화면 |
| 담당자 승인·수정·보류 | `review_required`/`hold` 초안만 생성 | 역할 확인, 리뷰 버전·결정·수정 사유·시간 저장 |
| 승인 전 가족·외부 공유 차단 | AI 응답은 항상 `external_share_allowed=false`; 공유 엔드포인트 없음 | 실제 가족 조회·내보내기 경로에서 승인된 버전만 반환 |
| 근거·금지표현·자동 공유 QA | 생성 시 QA + 수정 초안을 재검사하는 `/carebridge/qa` | 승인 트랜잭션에서 재검사하고 보류 사유 해결 |
| 합성 시나리오 검증 | 이 변경의 테스트와 `examples/memos.json` | 교육기관·참여자 검증 및 검토 부담 측정 |

명세서의 기존 데모 60/180/90건 결과를 이 레포의 측정 결과로 재사용하지 않는다.
현재 테스트는 계약·차단 흐름 검증이며 번역 정확도, 임상 품질, 시간절감 지표가 아니다.

## 실행

Python 3.10 이상. 텍스트 기능에는 GPU나 Whisper 파일이 필요 없다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-core.txt
Copy-Item .env.example .env
$env:FCC_AI_STT_ENABLED = 'false'
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

`.env`의 `FCC_AI_API_KEY`를 설정하고 백엔드에서만 호출한다. 브라우저에 서비스 키를
노출하지 않는다. `/docs`와 `/openapi.json`에서 요청·응답 필드 전체를 확인할 수 있다.
기존 STT를 사용하려면 `requirements.txt` 설치, LoRA 배치, `FCC_AI_STT_ENABLED=true`가 필요하다.

다국어 번역은 `FCC_AI_CAREBRIDGE_MODEL_URL`에 **전체 chat completions URL**,
`FCC_AI_CAREBRIDGE_MODEL`에 모델명을 지정한다. 모델 서버는 `response_format=json_object`,
system/user 메시지와 비스트리밍 응답을 지원해야 한다. 키는 선택 설정이다.
OpenAI, 호환 클라우드 또는 로컬 모델 서버를 연결할 수 있지만 언어별 품질은 별도로 검증해야 한다.
한국어 문장에 영문 등이 섞이면 원문 복사로 번역을 대체하지 않고 모델을 사용한다.

모델에 전달되는 내용은 원문 구간, 요청 언어, 고정 system 지시와 스키마다.
모델 출력은 엄격한 JSON 스키마로 검증한다. 구간 누락·중복·순서 변경·추가 필드·
비한국어 출력·잘린 응답은 502이며 정상 초안으로 사용하지 않는다. 연결 장애·미설정은 503이다.
재시도·리다이렉트는 하지 않고 응답은 128KiB로 제한한다.

## POST /carebridge/drafts

공통 인증: `X-API-Key: <FCC_AI_API_KEY>` (설정 시 필수).

```json
{
  "memo": "아침 식사를 절반 드셨습니다.\n물을 조금 마신 것 같습니다.",
  "source_language": "ko",
  "data_use": "synthetic_deidentified"
}
```

- `data_use`는 필수이며 교육용 가상·비식별 입력이라는 호출자 확인이다.
- `memo`: 원문 공백·개행 보존, 1~4,000자, 비어 있지 않은 최대 40줄.
- `source_language`: `auto` 기본값, 또는 `ko`, `en`, `vi`, `id`, `th` 등의 언어 코드.
- 한 줄을 원문 구간 하나로 취급한다. 간단한 메모를 관찰별로 줄바꿈하면 검토하기 쉽다.

주요 응답 필드:

| 필드 | 의미 |
| --- | --- |
| `schema_version` | `carebridge.v1` |
| `status` | `review_required` 또는 QA 보류가 있는 `hold`; `approved`는 없음 |
| `label` | `검토 전 초안 · 공식기록 아님` |
| `mode` | `source_preserving`(한국어 원문 보존) / `model_translation` |
| `source` | 원문 종류(`text`/`audio`), 요청 언어 또는 로컬에서 확인된 `ko`, 텍스트, SHA-256 |
| `translation` | 한국어 전체 번역과 구간별 번역·원문 근거 |
| `observations[]` | 번역 구간을 그대로 사용한 관찰 초안과 분류·근거 |
| `additional_checks[]` | 확인 필요 구간에 연결된 고정 확인 질문; 새로운 사실·조언을 생성하지 않음 |
| `qa` | 검사 결과, 검토/보류 사유, 항상 사람이 검토하고 외부 공유를 금지하는 플래그 |

분류: `meal`, `hydration`, `mobility`, `sleep`, `hygiene`, `medication`, `condition`, `other`.
한국어는 보수적 키워드 분류이고, 다국어는 모델의 미검증 분류다. 분류와 확인 필요 여부도 검토 대상이다.

각 근거는 `segment_id`, `quote`, `start`, `end`를 가진다. 위치는 **Unicode 코드 포인트** 기준,
`start` 포함·`end` 제외이며 JavaScript의 UTF-16 인덱스와 다를 수 있다.
프론트에서 위치로 자르려면 `Array.from(source.text).slice(start, end).join('')`을 사용한다.
SHA-256은 정확한 원문 문자열의 UTF-8 바이트로 계산한다.

## POST /carebridge/drafts/audio

`multipart/form-data`:

- `file`: 오디오. 기본 최대 4MiB, 최대 30초. 길거나 손상된 음성을 조용히 잘라 사용하지 않는다.
- `data_use=synthetic_deidentified`: 필수.
- `source_language=auto`: 기본 언어 감지; 명시 언어는 `ko/en/vi/id/th/tl/zh/ja/ne/uz/ru` 및 지역 접미사.

```bash
curl -X POST http://127.0.0.1:8000/carebridge/drafts/audio \
  -H "X-API-Key: $FCC_AI_API_KEY" \
  -F "file=@synthetic-note.wav" \
  -F "data_use=synthetic_deidentified" \
  -F "source_language=ko"
```

응답 구조는 텍스트와 같고 `source.kind=audio`이며 전사 내용 확인 경고가 추가된다.
이때 근거는 오디오 타임스탬프가 아니라 **자동 전사 텍스트의 위치**다.
업로드는 임시 파일로 처리될 수 있으나 이 앱은 원문·음성·초안을 DB나 파일로 저장하지 않는다.
한국어 LoRA의 다국어 전사 정확도는 검증되지 않았다. 운영 전 언어별 평가가 필요하다.

기존 `POST /transcribe`의 `file` 입력과 `{"text":"..."}` 응답은 유지한다.
기존 경로에는 설정된 한국어 디코딩을 유지하고, 신규 CareBridge 경로만 `auto`와 명시 언어를 사용한다.
STT 추론은 작업 스레드에서 직렬 처리하여 이벤트 루프를 막지 않는다.

## POST /carebridge/qa

수정한 **전체 Draft 객체**를 전송한다. 이전 `qa` 결과를 신뢰하지 않고 재계산한다.
응답은 QAResult이며 승인 상태를 반환하지 않는다. `approved`, 공유 허용, 검토 면제 등
계약 밖 상태를 요청에 넣으면 422이다.

검사 대상은 원문 해시·구간 전체 보존·인용 위치 일치·관찰의 번역 일치·추가확인 근거,
숫자 보존, 식별정보 신호, 의료 판단·처리 지시 표현, 공유 차단이다.
숫자는 아라비아 숫자 토큰을 대조하므로 `2`를 `두`로 번역한 경우도 보수적으로 보류할 수 있다.

**검사의 한계:** 원문 인용 연결은 의미적 충실성의 증명이 아니다. 부정·불확실성·문맥의
번역 오류나 모든 언어의 의료 표현·모든 식별정보를 자동 검출하지 못한다. 항상 사람의 검토가 필요하다.
식별정보는 전화번호·주민번호·이메일·명시 이름/주소/생년월일 등 일부 패턴만 감지한다.
의료 표현은 한국어·영어의 제한된 패턴을 사용하며 인용된 의료 용어도 보류할 수 있다.
현재 범위에서는 실제 개인정보를 넣지 않으며, 탐지를 통과했다고 비식별이 증명되는 것은 아니다.
부정·불확실성 어휘 신호 검사도 추가되어 신호가 달라지면 보류한다(영/한/베트남/인도네시아어 제한 패턴).

`/qa`는 클라이언트가 전송한 원문의 내부 일관성만 검증한다. 해시는 서명·위변조 방지 토큰이 아니며,
원문과 해시를 함께 바꾼 공격을 막는 저장소가 아니다. 백엔드는 최초 원문·버전을 저장하고 대조해야 한다.

## 오류

| 상태 | 예 |
| --- | --- |
| 400 | 빈 오디오 파일 |
| 401 | 서비스 API 키 불일치 |
| 413 | 오디오 바이트 수·길이 초과 |
| 422 | 계약 위반, 식별정보/처리 지시 감지, 공백 메모, 무음·손상 오디오 |
| 500 | STT 추론 내부 오류 (상세 경로·모델 오류는 노출하지 않음) |
| 502 | 모델의 초안 형식·원문 구간 계약 위반 |
| 503 | STT 비활성 또는 번역 모델 미설정/연결 장애 |

진단 등 QA 보류는 원문을 지우지 않고 `200`, `status=hold`로 반환한다.
유효성 검사 오류에는 원문이나 외부 모델 응답을 넣지 않는다.

## 백엔드 연동에서 반드시 완성할 흐름

1. 서비스 키는 백엔드에만 저장하고 사용자 인증·그룹 소속·검토자 역할을 확인한다.
2. AI 응답을 원문·초안 버전·작성자와 함께 비공개 초안으로 저장한다. 자동 가족 공유나 공식기록 생성은 하지 않는다.
3. 수정하면 최초 원문과 근거를 대조하고 `/carebridge/qa`를 다시 실행한다. 보류 사유는 사람이 검토한다.
4. 검토자 결정(`approve/edit/hold`), 수정 사유, 사용자 ID, 시간, 대상 버전을 서버에서 기록한다.
5. 승인과 버전 일치를 같은 트랜잭션에서 확인한다. 승인 후 수정된 버전에는 재승인이 필요하다.
6. 가족 조회·내보내기 경로는 서버에 저장된 승인 버전만 수동으로 미리보기 한다.
   클라이언트가 보낸 `approved`나 AI의 QA 결과만으로 접근을 허용하지 않는다.

현재 FCC 일정 승인(`/approvals`)은 일정 취소·변경 기능이다. 관찰메모 승인과는 별도 도메인으로 연결해야 한다.

## 검증

```powershell
pip install -r requirements-dev.txt
python -m pytest -q
ruff check app tests
ruff format --check app tests
```

테스트는 한국어 로컬 모드, 합성 다국어 응답, HTTP 모델 계약, 근거 변조, 숫자 변경,
의료·식별정보·처리 지시, 승인/공유 우회, STT HTTP 계약을 확인한다.
실제 WAV 디코딩·리샘플링·길이 제한·무음 처리를 검증한다.
외부 모델 호출은 테스트용 HTTP 전송 계층으로, STT 모델 추론은 대역으로 검증한다.
실제 모델 가중치·GPU·모델 서비스가 제공되지 않은 환경에서 실모델 정확도를 주장하지 않는다.

## 전사문 확인과 전체 서비스

POST /carebridge/transcriptions는 drafts/audio와 같은 multipart 입력(file, data_use, source_language)을 받지만
초안을 생성하지 않고 text와 requires_transcript_review=true만 반환한다. 비식별·언어·음성 길이·인증 검사는 같다.
프론트는 작성자가 이 전사문을 확인·수정한 뒤 백엔드 notes 생성 API에 제출한다.
백엔드는 이때 확정한 텍스트를 불변 원문으로 보존한다.
[전체 검토·가족 열람 계약](https://github.com/FamilyCareCloud/fcc-backend/blob/feature/carebridge-workflow/docs/CAREBRIDGE.md).
