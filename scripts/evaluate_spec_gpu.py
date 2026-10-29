"""Offline, real-CUDA specification evaluation; never submits or publishes service records.

Run with the serving process stopped to avoid loading a second model into its VRAM.
The output contains ONLY synthetic fixtures, model responses and reproducibility metadata.
HTTP success and model self-review are not translation-accuracy measurements.
"""

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_translation import reference_draft, save, summarize  # noqa: E402


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--corpus", type=Path, default=ROOT / "evaluation/translation-corpus-v1.json"
    )
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--audio-reference", default="Drank water this morning. Walked with help.")
    parser.add_argument("--adapter-path", default="models/whisper-senior-lora")
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an evaluation. Choose a new output path.")

    import torch
    from fastapi.testclient import TestClient

    from app.carebridge.local import LocalTranslationProvider
    from app.config import Settings
    from app.main import create_app

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback evaluation.")
    config = Settings(
        _env_file=None,
        device="cuda",
        api_key="offline-evaluation-only",
        stt_enabled=bool(args.audio),
        adapter_path=args.adapter_path,
        carebridge_backend="local",
        carebridge_local_files_only=True,
        carebridge_preload=True,
        carebridge_semantic_check_enabled=True,
        carebridge_timeout_seconds=args.timeout,
    )

    class RecordingProvider(LocalTranslationProvider):
        last_model_output = None

        async def generate(self, segments, source_language):
            candidate = await super().generate(segments, source_language)
            self.last_model_output = candidate.model_dump(mode="json")
            return candidate

    provider = RecordingProvider(config)
    corpus_raw = args.corpus.read_bytes()
    fixture = json.loads(corpus_raw.decode("utf-8-sig"))
    code_files = {
        p.relative_to(ROOT).as_posix(): sha(p.read_bytes())
        for p in sorted((ROOT / "app").rglob("*.py"))
    }
    report = {
        "schema": "carebridge.spec-evaluation.v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "worktree_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ),
        "code_sha256": code_files,
        "runner_sha256": sha(Path(__file__).read_bytes()),
        "corpus_sha256": sha(corpus_raw),
        "fixture_counts": {
            "translations": len(fixture["translation_cases"]),
            "qa": len(fixture["qa_cases"]),
        },
        "scope": "Synthetic offline TestClient + real CUDA weights. No backend storage, native-speaker or field validation.",
        "gpu": torch.cuda.get_device_name(0),
        "runtime_versions": {
            name: version(name) for name in ["torch", "transformers", "bitsandbytes", "peft"]
        },
        "config": {
            k: getattr(config, k)
            for k in [
                "carebridge_local_model",
                "carebridge_timeout_seconds",
                "carebridge_local_max_input_tokens",
                "carebridge_local_max_new_tokens",
                "carebridge_queue_timeout_seconds",
                "stt_timeout_seconds",
                "stt_max_new_tokens",
                "max_audio_seconds",
                "max_audio_bytes",
            ]
        },
        "translations": [],
        "qa_cases": [],
        "workflow_cases": [],
    }
    headers = {"X-API-Key": config.api_key}
    generations = {}

    def persist():
        save(args.output, report)

    def post(client, path, payload=None, **kwargs):
        started = time.perf_counter()
        response = client.post(path, headers=headers, json=payload, **kwargs)
        return {
            "http_status": response.status_code,
            "seconds": round(time.perf_counter() - started, 3),
            "response": response.json(),
        }

    def generation(client, memo, language="ko", revision=1, **extra):
        provider.last_model_output = None
        result = post(
            client,
            "/carebridge/generations",
            {
                "request_id": str(uuid4()),
                "source_revision": revision,
                "memo": memo,
                "source_language": language,
                "data_use": "synthetic_deidentified",
                **extra,
            },
        )
        if provider.last_model_output is not None:
            result["model_output"] = provider.last_model_output
        return result

    def edit_payload(original, editor="author"):
        draft = original["draft"]
        return {
            "request_id": str(uuid4()),
            "data_use": "synthetic_deidentified",
            "generation": copy.deepcopy(original),
            "editor": editor,
            "expected_source_revision": original["source_revision"],
            "expected_source_sha256": draft["source"]["sha256"],
            "segments": [
                {
                    "segment_id": t["evidence"]["segment_id"],
                    "text": t["text"],
                    "category": o["category"],
                }
                for t, o in zip(
                    draft["translation"]["segments"], draft["observations"], strict=True
                )
            ],
        }

    def record(case_id, result, expected_http=200, expected_status=None, **extra):
        status = result.get("response", {}).get("draft", result.get("response", {})).get("status")
        passed = result["http_status"] == expected_http and (
            expected_status is None or status == expected_status
        )
        report["workflow_cases"].append(
            {
                "id": case_id,
                "expected_http": expected_http,
                "expected_status": expected_status,
                "passed": passed,
                **extra,
                **result,
            }
        )
        persist()
        print(
            f"workflow {case_id}: HTTP {result['http_status']}, {status}, passed={passed}",
            flush=True,
        )

    persist()
    print("Loading Qwen and optional Whisper weights on CUDA...", flush=True)
    started = time.perf_counter()
    with TestClient(create_app(config, provider=provider)) as client:
        report["load_seconds"] = round(time.perf_counter() - started, 3)
        report["health"] = client.get("/health").json()
        report["model_snapshot"] = getattr(provider._model.config, "_commit_hash", None)
        if args.audio:
            adapter = Path(args.adapter_path)
            report["adapter_sha256"] = {
                p.name: sha(p.read_bytes()) for p in sorted(adapter.glob("*")) if p.is_file()
            }
        persist()
        for case in fixture["translation_cases"]:
            result = generation(client, case["source"], case["language"])
            entry = {"id": case["id"], **result}
            if result["http_status"] == 200:
                envelope = result["response"]
                generations[case["id"]] = envelope
                entry["generation_binding"] = {k: v for k, v in envelope.items() if k != "draft"}
                entry["response"] = draft = envelope["draft"]
                refs = [t["evidence"] for t in draft["translation"]["segments"]]
                entry["source_evidence_valid"] = (
                    draft["source"]["text"] == case["source"]
                    and draft["source"]["sha256"] == sha(case["source"].encode())
                    and len(refs) == len(case["reference"])
                    and all(case["source"][r["start"] : r["end"]] == r["quote"] for r in refs)
                )
                entry["sharing_blocked"] = (
                    draft["qa"]["external_share_allowed"] is False
                    and draft["qa"]["requires_human_review"] is True
                )
            report["translations"].append(entry)
            persist()
            print(
                f"translation {case['id']}: HTTP {result['http_status']}, {entry['response'].get('status')}, {result['seconds']}s",
                flush=True,
            )

        by_id = {c["id"]: c for c in fixture["translation_cases"]}
        for case in fixture["qa_cases"]:
            draft = reference_draft(by_id[case["base_id"]], case["translation"])
            result = post(client, "/carebridge/qa/semantic", draft)
            report["qa_cases"].append(
                {"id": case["id"], "expected": case["expected"], "reason": case["reason"], **result}
            )
            persist()
            print(
                f"QA {case['id']}: HTTP {result['http_status']}, {result['response'].get('status')}, expected={case['expected']}",
                flush=True,
            )

        # Human edits use actual generated snapshots, never a fabricated model response.
        for language in ["en", "vi", "id"]:
            original = generations.get(language + "-assistance")
            if original is None:
                record(
                    language + "-edit-missing-generation",
                    {"http_status": None, "response": {}},
                    expected_status="review_required",
                )
                continue
            for editor in ["author", "reviewer"]:
                payload = edit_payload(original, editor)
                payload["segments"][0].update(
                    text=by_id[language + "-assistance"]["reference"][0], category="mobility"
                )
                record(
                    language + "-" + editor + "-edit",
                    post(client, "/carebridge/edits/qa", payload),
                    expected_status="review_required",
                    original_generation=original,
                )
            payload = edit_payload(original)
            payload["segments"][0]["text"] = "화장실까지 걸었습니다."
            record(
                language + "-missing-assistance",
                post(client, "/carebridge/edits/qa", payload),
                expected_status="hold",
            )

        korean = generation(client, "🙂 물을 마셨습니다.\r\n  도움을 받아 걸었습니다.  ")
        record("korean-source-preserving", korean, expected_status="review_required")
        if korean["http_status"] == 200:
            original = korean["response"]
            payload = edit_payload(original)
            payload["segments"][0]["text"] = "🙂 물을 마셨다."
            record(
                "korean-author-edit",
                post(client, "/carebridge/edits/qa", payload),
                expected_status="review_required",
                original_generation=original,
            )
            changed = generation(client, "물을 마시지 않았습니다.", revision=2)
            record("source-regeneration", changed, expected_status="review_required")
            payload = edit_payload(original)
            payload["expected_source_revision"] = 2
            record(
                "stale-revision-rejected",
                post(client, "/carebridge/edits/qa", payload),
                expected_http=409,
            )
            payload = edit_payload(original)
            payload["generation"]["draft"]["translation"]["text"] = "원본 덮어쓰기"
            record(
                "mutated-generation-rejected",
                post(client, "/carebridge/edits/qa", payload),
                expected_http=409,
            )
            payload = edit_payload(original)
            payload["segments"].append(
                {"segment_id": "s003", "text": "산책했습니다.", "category": "mobility"}
            )
            record(
                "unsupported-new-segment",
                post(client, "/carebridge/edits/qa", payload),
                expected_http=409,
            )

        for case_id, source, target, expected in [
            (
                "equivalent-clock",
                "Started resting at 15:00.",
                "오후 3시에 휴식을 시작했습니다.",
                "review_required",
            ),
            (
                "changed-clock",
                "Started resting at 15:00.",
                "오후 4시에 휴식을 시작했습니다.",
                "hold",
            ),
            ("reported-speech-lost", "Said there was no pain.", "통증이 없었습니다.", "hold"),
            ("object-lost", "Ate half of the soup.", "식사를 절반 먹었습니다.", "hold"),
            ("action-lost", "Walked with help.", "도움을 받았습니다.", "hold"),
            ("added-action", "Walked with help.", "도움을 받아 걸었고 물을 마셨습니다.", "hold"),
        ]:
            draft = reference_draft(
                {
                    "id": case_id,
                    "language": "en",
                    "source": source,
                    "expected_categories": ["other"],
                    "confirmation_segments": [],
                },
                [target],
            )
            record(
                case_id, post(client, "/carebridge/qa/semantic", draft), expected_status=expected
            )

        if args.audio:
            audio = args.audio.read_bytes()
            result = post(
                client,
                "/carebridge/transcriptions",
                files={"file": ("synthetic.wav", audio, "audio/wav")},
                data={"source_language": "en", "data_use": "synthetic_deidentified"},
            )
            record(
                "english-transcription",
                result,
                audio_sha256=sha(audio),
                reference=args.audio_reference,
            )
            if result["http_status"] == 200:
                # Simulate the caller's confirmed text using the prewritten reference,
                # preserving the raw transcription above for independent comparison.
                confirmed = generation(
                    client, args.audio_reference, "en", input_kind="reviewed_transcript"
                )
                record(
                    "confirmed-transcript-generation", confirmed, expected_status="review_required"
                )
            bad = post(
                client,
                "/carebridge/transcriptions",
                files={"file": ("bad.wav", b"invalid", "audio/wav")},
                data={"source_language": "en", "data_use": "synthetic_deidentified"},
            )
            record("invalid-audio-rejected", bad, expected_http=422)

        config.carebridge_timeout_seconds = 0.001
        record(
            "real-generation-timeout", generation(client, "Drank water.", "en"), expected_http=504
        )
        config.carebridge_timeout_seconds = args.timeout
        config.carebridge_local_max_new_tokens = 128
        record(
            "real-truncated-output",
            generation(client, "\n".join(["Drank water."] * 8), "en"),
            expected_http=502,
        )
        config.carebridge_local_max_new_tokens = 2048
        report["peak_cuda_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
        report["peak_cuda_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 1024**2, 1)

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["summary"] = summarize(report)
    report["workflow_passed"] = sum(c["passed"] for c in report["workflow_cases"])
    report["workflow_total"] = len(report["workflow_cases"])
    persist()
    print(
        json.dumps(
            {
                "summary": report["summary"],
                "workflow": [report["workflow_passed"], report["workflow_total"]],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return (
        0
        if (
            report["workflow_passed"] == report["workflow_total"]
            and report["summary"]["translation_http_200"] == len(fixture["translation_cases"])
            and report["summary"]["qa_request_failures"] == 0
            and report["summary"]["injected_errors_missed"] == 0
            and report["summary"]["faithful_controls_false_hold"] == 0
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
