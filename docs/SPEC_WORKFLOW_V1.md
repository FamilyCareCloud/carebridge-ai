# 기능명세 v1.0의 AI 구현 계약

기준: 2026-10-28 작성된 `CareBridge-UseCase-v1.0.md` 및 `CareBridge-기능명세서-v1.0.md`의 §15–18. AI 저장소 구현이며, 백엔드·프론트의 새로운 기관/제출/공개 흐름까지 연결됐다는 뜻은 아니다.

## 구현 범위

| 요구사항 | AI 구현 | 다른 계층에서 필요한 작업 |
|---|---|---|
| AI-I01 / UC-AI05 | 원문 개정·요청 ID·원문/생성본 해시 연결, 편집 시 대조 | 저장된 최신 개정·해시 사용, 늦게 온 응답 폐기 |
| AI-I02 / UC-AI05 | 생성 원본 유지, 편집본 새 ID, author/reviewer 구분, 원본 대비 변경 표시 | 실제 사용자·권한 확인, 생성/편집 이력 보관 |
| AI-I03 / UC-AI04–05 | 같은 근거 구간의 문장·분류 편집과 GPU 의미 재검사 | 최종본과 QA를 같은 버전으로 저장 |
| AI-I04 | 제출·공개 부작용 없음 | 명시적 제출과 중복 제출 방지 |
| AI-I05 / UC-AI07 | 합성·비식별 입력 제한 유지 | 실제 정보 허용 조건 D01 합의 |
| AI-I06 / UC-AI06 | qa.v2, hold/review_required만 반환 | 사람 보류·확정·공개 분리 및 D10 결정 |
| AI-I07 | 모델 장애/시간초과/절단/대기 초과 실패 처리 | 입력 보존, 재시도, 제출 idempotency |
| AI-C02 / UC-AI02 | 남은 외국어 산문을 제한된 규칙으로 거절 | 언어 코드 허용과 품질 검증을 구분 |
| AI-C08 / UC-AI03 | 전사→사람 확인→텍스트 생성, STT 대기·추론 제한 | 확인 화면 및 확인 행위 저장 |

AI-F01–F08은 향후 후보이며 전부 구현했다고 표시하지 않는다. 이번 변경에는 AI-F04 관련 qa.v2 보강과 AI-F07의 원본 대비 변경/근거 표시가 포함된다. 의미 자동 분할·맞춤 질문·한국어 자연어 재작성·긴 입력 확대·공개문 생성은 도입하지 않았다.

## 호환성과 버전

기존 Draft/QA/음성 API와 내부 `carebridge.v1` 형식을 유지한다. 신규 연결은 `carebridge.workflow.v1` 외부 객체다. `/health`의 `carebridge_workflow_version`, `carebridge_qa_revision`으로 확인한다.

이 브랜치는 평가 PR #5의 동결 사례 및 qa.v2 PR #6을 포함한다. 백엔드 #8의 검증기에는 **내부 draft 및 review**를 전달한다. 외부 객체 전체를 기존 `validateDraft`에 전달하지 않는다. 새 API 호출/저장은 백엔드 후속 연동 사항이다.

## POST /carebridge/generations

서비스 `X-API-Key`가 필요하다.

```json
{
  "request_id": "3ca83515-8d36-43b8-83ce-36258f797d05",
  "source_revision": 1,
  "memo": "Walked with help.",
  "source_language": "en",
  "data_use": "synthetic_deidentified",
  "input_kind": "text"
}
```

- request_id: UUID 모양 문자열, 응답 대조용. **AI의 중복 실행 방지 키가 아니다.** 재시도마다 새 생성 ID가 생길 수 있다. 제출 기록은 만들지 않는다.
- source_revision: 백엔드가 관리하는 1 이상 JS 안전 정수. 원문 변경 시 증가시킨다.
- input_kind: text 또는 reviewed_transcript. 후자는 작성자가 전사문을 확인했다는 호출자의 선언이며 AI가 확인 행위를 증명하지 않는다. 확인한 텍스트의 source.kind는 기존 백엔드에 맞춰 text다.
- 4,000자/40개 비어 있지 않은 줄 제한 유지. 기관·대상자 이름/번호·수신자 필드는 받지 않는다.

응답: origin=ai_generated, 요청 ID·개정·입력 종류, draft, draft_sha256, pipeline. 원문 SHA-256은 UTF-8 원문 기준이며 공백/개행/Unicode 정규화 없이 보존한다. 근거 위치는 Unicode 코드 포인트다.

draft_sha256은 검증된 Draft를 `json.dumps(..., ensure_ascii=False, sort_keys=True, separators=(",", ":"))`로 직렬화한 UTF-8 해시다. 반환 해시와 원본 객체를 함께 보관해 다시 보내기를 권장한다. **해시는 서명·인증 수단이 아니다.** AI에는 DB가 없으므로 최신 개정이나 생성 원본의 진위를 독립적으로 확인할 수 없다. 백엔드의 신뢰된 저장소 값으로 비교해야 한다.

pipeline은 QA 계약, 설정 backend/model, 번역/의미 프롬프트 SHA-256이다. 한국어 보존 경로에서도 설정 모델명이 포함될 수 있으며 실제 생성 경로는 draft.mode를 본다. 실행 코드·가중치 스냅샷·GPU 패키지 버전은 평가 보고서에 기록한다.

번역 지문에는 원문 단서 규칙·용어·템플릿·개수 한도, 언어/의미에 따른 지침 선택 및 원문 기반 출력 보정 정의도 포함한다. 의미 검사 지문에는 명시적 대상·행동·발언 출처 보존 규칙도 포함한다. 요청별 최종 프롬프트 자체의 해시는 아니며 실제 단서와 보정은 보존된 원문·모델 출력·고정 코드로 재구성한다. 원문/생성본 해시와 응답 스키마는 유지한다. 한국어 원문과 사람 편집본에는 생성문 보정을 적용하지 않는다. [확장 평가 오류 개선](TRANSLATION_CONTEXT_V1.md)에 방법과 검증 근거를 기록한다.

## POST /carebridge/edits/qa

필수 필드:

| 필드 | 값 |
|---|---|
| data_use | synthetic_deidentified |
| request_id | 새 UUID 문자열 |
| generation | 위 생성 API가 반환한 전체 원본 객체 |
| expected_source_revision | 백엔드에 저장된 현재 원문 개정 |
| expected_source_sha256 | 백엔드에 저장된 현재 원문 해시 |
| editor | author 또는 reviewer |
| segments | 아래 형태의 전체 구간 배열 |

```json
[{"segment_id":"s001","text":"도움을 받아 걸었습니다.","category":"mobility"}]
```

원래 ID·개수·순서를 유지한다. 최종 문장/category만 보내면 원문 근거를 복사하고 번역·관찰 항목을 일치시킨다. 근거 없는 새 문장이 기존 구간 안에 들어간 경우도 의미 QA 대상이며 모델이 항상 찾아낸다는 보증은 없다. 새 사실은 원문을 수정하고 재생성한다. 구간 추가/삭제/병합/분할/재정렬은 이번 계약에서 지원하지 않는다.

응답은 origin=author|reviewer, request_id, generation_id, original_draft_sha256, 원문 개정/해시, 새 ID의 draft와 해시, changes(생성 원본 대비 문장·분류·근거), review(구간별 검사/집계), 현재 pipeline이다.

- draft.mode는 최초 생성 경로를 유지한다. **편집본 출처는 외부 origin을 사용**하며 편집본을 AI 원출력으로 표시하지 않는다.
- changes는 최초 생성본 대비 차이다. 작성자 편집본→검토자 편집본 이력은 백엔드가 별도로 보관한다.
- editor 필드는 실제 권한 증명이 아니다. 백엔드가 사용자 권한을 검증해야 한다.
- 구조 보류면 모델 호출 없이 model_checked=false/hold. 정상 구조는 로컬 의미 모델로 검사하며, 모델 비활성/장애를 통과로 바꾸지 않는다.
- AI hold와 사람 held/confirmed/published를 분리한다. D10의 확정 차단 정책을 이 API로 결정하지 않는다.

## 오류 및 제한

| HTTP / code | 의미 |
|---|---|
| 409 SOURCE_REVISION_CONFLICT | 원문 개정/해시와 생성본 불일치 |
| 409 GENERATION_SNAPSHOT_CONFLICT | 생성본과 보관한 해시 불일치 |
| 409 INVALID_GENERATION_EVIDENCE | 생성본의 원문·근거·구조 오류 |
| 409 EDIT_SEGMENTS_MISMATCH | 추가·삭제·재정렬·누락·중복 구간 |
| 422 INVALID_REQUEST / 입력 검사 코드 | 스키마·한도·데이터 사용·식별 정보 등 |
| 502 INVALID_MODEL_OUTPUT / STT_INCOMPLETE | 형식 오류·출력 절단·미완성 전사 |
| 503 TRANSLATION_UNAVAILABLE / STT_BUSY | 모델 비활성/장애 또는 GPU 대기 초과 |
| 504 INFERENCE_TIMEOUT / STT_TIMEOUT | 생성/전사 시간 초과 |

기존 HTTP 번역 provider의 시간초과도 504로 구분한다. 입력을 유지하며 재시도는 자동 수행하지 않는다. 생성과 의미 검사는 각각 제한 시간을 갖는다. max_time은 OS 강제 중단 타이머가 아니므로 한 번의 추론 연산이 한도를 초과할 수 있다. 실제 반환까지 GPU 잠금을 유지한다. 최초 모델 로딩은 별도이며 운영은 preload를 권장한다.

STT 기본 제한: 30초/4MiB 입력, 10초 추론, 384 새 토큰. FCC_AI_STT_TIMEOUT_SECONDS, FCC_AI_STT_MAX_NEW_TOKENS로 조정한다. 끝 토큰 없이 종료하면 성공 반환하지 않는다. 대기 제한은 FCC_AI_CAREBRIDGE_QUEUE_TIMEOUT_SECONDS를 공유한다.

외국어 산문 규칙은 문자/단위/대문자 약어 기반이다. 정상 고유명사 등을 거절하거나 대문자 산문을 놓칠 수 있다. 한국어 문장의 문법·의미 정확도는 증명하지 않는다.

GPU 평가로 추가한 보조 규칙은 지원을 받는다는 표현, 명시적인 걷기/수프/발언 출처, 숫자로 적은 일부 수면/휴식 지속시간이다. 의미 모델이 통과시켜도 알려진 사실 표지가 없으면 SEMANTIC_OMITTED_FACTS를 실패로 바꾼다. 지속시간이 지연 시점으로 바뀌면 NUMERIC_FACTS_PRESERVED로 보류한다. 코드 목록과 qa.v2 계약은 유지한다. 영어·베트남어·인도네시아어 일부 표현만 다루며 포괄적인 의미 검증이 아니다. 이 규칙에도 오탐/미탐이 가능하다.

## GPU 재현

기존 GPU 서비스를 확인해 VRAM 중복 로드를 피하고 평가 후 복구한다. 평가기는 네트워크 서버를 열지 않고 TestClient로 실제 CUDA 가중치를 호출한다.

```powershell
python scripts/evaluate_spec_gpu.py --output evaluation/spec-gpu-새실행.json --audio evaluation/spec-synthetic-english.wav --adapter-path C:/별도준비/whisper-senior-lora
```

고정 36건 생성/36건 QA, 원문 재생성·사람 편집·충돌·사실 누락/추가/발언 출처·동등 시각·실제 시간초과/절단·선택적 합성 영어 STT를 기록한다. 출력 파일 덮어쓰기를 거절한다. 프롬프트/코드/사례/음성/가중치 버전과 실제 실패를 보관한다. 모델 자기 검사·HTTP 성공은 번역 정확도 점수가 아니다.

`--corpus evaluation/translation-corpus-v2.json`으로 실행하면 기존 사례를 포함한 54건 생성/60건 QA를 검사한다. 후속 개선에서는 수면 전 간격이 지속시간으로 바뀌는 오류와 명시된 외출의 누락도 검사한다. 직접/간접 인용의 형식 차이와 한국어 '걸은' 활용형의 오탐을 보완했다. 최신 결과와 남은 한계는 [TRANSLATION_IMPROVEMENT_V3.md](TRANSLATION_IMPROVEMENT_V3.md)를 참고한다.
