"""Run tiny synthetic real-weight checks, not a translation/clinical benchmark."""

import argparse
import copy
import json
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("evaluation/local-gpu-smoke.json"))
    parser.add_argument("--model", default="unsloth/Qwen3-8B-unsloth-bnb-4bit")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--audio", type=Path, help="Optional short synthetic English PCM WAV")
    parser.add_argument("--adapter-path", default="models/whisper-senior-lora")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.main import create_app

    config = Settings(
        _env_file=None,
        stt_enabled=args.audio is not None,
        api_key="local-smoke-only",
        adapter_path=args.adapter_path,
        carebridge_backend="local",
        carebridge_preload=True,
        carebridge_timeout_seconds=60,
        carebridge_local_model=args.model,
        carebridge_local_files_only=not args.allow_download,
    )
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(0),
        "model": config.carebridge_local_model,
        "runtime_versions": {
            name: version(name) for name in ["torch", "transformers", "bitsandbytes"]
        },
        "tests": [],
        "scope": "synthetic smoke, not accuracy or clinical validation",
    }
    started = time.perf_counter()
    print("Loading local model on CUDA...", flush=True)
    headers = {"X-API-Key": "local-smoke-only"}
    with TestClient(create_app(config)) as client:
        report["load_seconds"] = round(time.perf_counter() - started, 3)
        original = None
        for language, memo in [
            ("en", "Did not drink water this morning.\nMay have slept after lunch."),
            ("vi", "Đã ăn nửa bữa sáng."),
            ("id", "Makan setengah sarapan."),
        ]:
            started = time.perf_counter()
            response = client.post(
                "/carebridge/drafts",
                headers=headers,
                json={
                    "memo": memo,
                    "source_language": language,
                    "data_use": "synthetic_deidentified",
                },
            )
            result = response.json()
            passed = response.status_code == 200 and result["qa"]["external_share_allowed"] is False
            if passed:
                passed = all(
                    memo[s["evidence"]["start"] : s["evidence"]["end"]] == s["evidence"]["quote"]
                    for s in result["translation"]["segments"]
                )
            entry = {
                "language": language,
                "http_status": response.status_code,
                "passed": passed,
                "seconds": round(time.perf_counter() - started, 3),
                "response": result,
            }
            report["tests"].append(entry)
            print(
                f"translate {language}: HTTP {response.status_code}, {entry['seconds']}s",
                flush=True,
            )
            if response.status_code == 200 and language == "en":
                original = result
        if original:
            for name, translation in [
                ("faithful", None),
                ("negation_reversed", "아침에 물을 마셨습니다."),
                ("uncertainty_removed", "점심 식사 후 잠을 잤습니다."),
                ("unsupported_fact", "아침에 물을 마시지 않았고 보호자에게 전화를 걸었습니다."),
            ]:
                value = copy.deepcopy(original)
                if translation is not None:
                    idx = 1 if name == "uncertainty_removed" else 0
                    value["translation"]["segments"][idx]["text"] = translation
                    value["observations"][idx]["text"] = translation
                    value["translation"]["text"] = "\n".join(
                        s["text"] for s in value["translation"]["segments"]
                    )
                started = time.perf_counter()
                response = client.post("/carebridge/qa/semantic", headers=headers, json=value)
                result = response.json()
                expected = "review_required" if name == "faithful" else "hold"
                passed = (
                    response.status_code == 200
                    and result["status"] == expected
                    and result["qa"]["external_share_allowed"] is False
                )
                entry = {
                    "audit": name,
                    "http_status": response.status_code,
                    "passed": passed,
                    "seconds": round(time.perf_counter() - started, 3),
                    "response": result,
                }
                report["tests"].append(entry)
                print(f"audit {name}: {result.get('status')}, expected {expected}", flush=True)
        else:
            report["tests"].append({"audit": "skipped_no_translation", "passed": False})
        if args.audio:
            started = time.perf_counter()
            response = client.post(
                "/carebridge/drafts/audio",
                headers=headers,
                files={"file": ("synthetic.wav", args.audio.read_bytes(), "audio/wav")},
                data={"data_use": "synthetic_deidentified", "source_language": "en"},
            )
            result = response.json()
            passed = (
                response.status_code == 200
                and result["source"]["kind"] == "audio"
                and result["qa"]["external_share_allowed"] is False
            )
            report["tests"].append(
                {
                    "audio": "synthetic_english",
                    "passed": passed,
                    "http_status": response.status_code,
                    "seconds": round(time.perf_counter() - started, 3),
                    "response": result,
                }
            )
            print(f"audio: HTTP {response.status_code}", flush=True)
        report["health"] = client.get("/health").json()
        report["peak_cuda_mib"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    report["all_passed"] = all(t["passed"] for t in report["tests"])
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {args.output}, all_passed={report['all_passed']}", flush=True)
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
