param(
    [string]$Python = "",
    [int]$Port = 8002,
    [switch]$EnableStt
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $repo
try {
    if (-not $Python) { $Python = Join-Path $repo ".venv/Scripts/python.exe" }
    if (-not (Test-Path -LiteralPath $Python)) {
        throw "Python을 찾지 못했습니다. docs/LOCAL_GPU.md의 전용 환경 설정을 먼저 실행하세요."
    }
    $env:FCC_AI_CAREBRIDGE_BACKEND = "local"
    $env:FCC_AI_CAREBRIDGE_PRELOAD = "true"
    $env:FCC_AI_CAREBRIDGE_SEMANTIC_CHECK_ENABLED = "true"
    $env:FCC_AI_STT_ENABLED = if ($EnableStt) { "true" } else { "false" }
    if ($EnableStt -and -not $env:FCC_AI_ADAPTER_PATH) {
        $adapter = Join-Path $repo "models/whisper-senior-lora"
        if (-not (Test-Path -LiteralPath $adapter)) {
            throw "Whisper 어댑터가 없습니다. FCC_AI_ADAPTER_PATH에 모델 경로를 지정하세요."
        }
    }
    # Loopback only; a separate authenticated backend must mediate public access.
    & $Python -m uvicorn app.main:app --host 127.0.0.1 --port $Port --workers 1
    if ($LASTEXITCODE) { throw "GPU 서버가 오류로 종료됐습니다." }
}
finally { Pop-Location }
