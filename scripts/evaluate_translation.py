"""Evaluate a running GPU service using fixed, synthetic translation references.

This runner records evidence and gate behavior. It does NOT use the serving
model's self-review, reference string equality, or HTTP success as accuracy.
Review translation outputs against the prewritten meaning checklist separately.
Only loopback connections are supported so a local API key cannot be sent away.
"""

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
CHECK_QUESTION = "이 관찰의 의미와 빠진 정보를 원문 작성자에게 확인해 주세요."


class RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def read_key() -> str:
    key = os.environ.get("FCC_AI_API_KEY", "")
    if not key:
        for line in (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() == "FCC_AI_API_KEY":
                key = value.strip().strip("\"'")
                break
    if not key:
        raise RuntimeError("Configure FCC_AI_API_KEY in the environment or .env.")
    return key


def request_json(base: str, path: str, key: str, payload=None):
    started = time.perf_counter()
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
    headers = {"Content-Type": "application/json", "X-API-Key": key}
    request = Request(base + path, data=data, headers=headers)
    try:
        opener = build_opener(ProxyHandler({}), RejectRedirects())
        with opener.open(request, timeout=60) as response:
            status, body = response.status, response.read()
    except HTTPError as exc:
        status, body = exc.code, exc.read()
    except (URLError, TimeoutError):
        return {
            "http_status": None,
            "seconds": round(time.perf_counter() - started, 3),
            "transport_error": True,
        }
    try:
        result = json.loads(body)
    except (ValueError, UnicodeError):
        result = {"error": "non_json_response"}
    return {
        "http_status": status,
        "seconds": round(time.perf_counter() - started, 3),
        "response": result,
    }


def reference_draft(case: dict, translations: list[str]) -> dict:
    source = case["source"]
    evidence = [
        {
            "segment_id": f"s{i:03d}",
            "start": match.start(),
            "end": match.end(),
            "quote": match.group(),
        }
        for i, match in enumerate(re.finditer(r"[^\r\n]+", source), start=1)
    ]
    if len(evidence) != len(translations):
        raise ValueError("Reference segment count mismatch.")
    return {
        "schema_version": "carebridge.v1",
        "id": "evaluation-" + case["id"],
        "status": "review_required",
        "label": "검토 전 초안 · 공식기록 아님",
        "mode": "model_translation",
        "source": {
            "kind": "text",
            "language": case["language"],
            "text": source,
            "sha256": hashlib.sha256(source.encode()).hexdigest(),
        },
        "translation": {
            "language": "ko",
            "text": "\n".join(translations),
            "segments": [
                {"text": text, "evidence": ref}
                for text, ref in zip(translations, evidence, strict=True)
            ],
        },
        "observations": [
            {
                "id": f"o{i:03d}",
                "category": case["expected_categories"][0],
                "text": text,
                "evidence": ref,
            }
            for i, (text, ref) in enumerate(zip(translations, evidence, strict=True), start=1)
        ],
        "additional_checks": [
            {"id": f"c{i + 1:03d}", "question": CHECK_QUESTION, "evidence": evidence[i]}
            for i in case["confirmation_segments"]
        ],
        "qa": {
            "checks": [],
            "issues": [],
            "requires_human_review": True,
            "external_share_allowed": False,
        },
    }


def save(path: Path, report: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def summarize(report: dict) -> dict:
    translations = report["translations"]
    qa = report["qa_cases"]
    good = [r for r in qa if r["expected"] == "review_required"]
    bad = [r for r in qa if r["expected"] == "hold"]
    successful = [r for r in translations if r["http_status"] == 200]
    latencies = sorted(r["seconds"] for r in successful)
    return {
        "translation_requests": len(translations),
        "translation_http_200": len(successful),
        "translation_hold": sum(r.get("response", {}).get("status") == "hold" for r in successful),
        "source_evidence_valid": sum(r.get("source_evidence_valid", False) for r in successful),
        "sharing_blocked": sum(r.get("sharing_blocked", False) for r in successful),
        "faithful_controls": len(good),
        "faithful_controls_review_required": sum(
            r.get("response", {}).get("status") == "review_required" for r in good
        ),
        "faithful_controls_false_hold": sum(
            r.get("response", {}).get("status") == "hold" for r in good
        ),
        "injected_errors": len(bad),
        "injected_errors_caught": sum(r.get("response", {}).get("status") == "hold" for r in bad),
        "injected_errors_missed": sum(
            r.get("response", {}).get("status") == "review_required" for r in bad
        ),
        "qa_request_failures": sum(r["http_status"] != 200 for r in qa),
        "translation_latency_p50_seconds": round(median(latencies), 3) if latencies else None,
        "translation_latency_max_seconds": max(latencies) if latencies else None,
        "accuracy": "Not automatically scored; see separate reference-based AI review.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument(
        "--corpus", type=Path, default=ROOT / "evaluation/translation-corpus-v1.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "evaluation/translation-baseline-v1.json"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    parsed = urlsplit(args.base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        parser.error("Only a plain loopback HTTP service URL is allowed.")
    base = args.base_url.rstrip("/")
    raw = args.corpus.read_bytes()
    fingerprint = hashlib.sha256(raw).hexdigest()
    fixture = json.loads(raw.decode("utf-8-sig"))
    key = read_key()
    health = request_json(base, "/health", key)
    if health["http_status"] != 200 or not health["response"].get("translation_loaded"):
        raise RuntimeError("Running translation service is unavailable.")
    if args.resume and args.output.exists():
        report = json.loads(args.output.read_text(encoding="utf-8"))
        if report["corpus_sha256"] != fingerprint:
            raise ValueError("Cannot resume with a changed reference corpus.")
    else:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        report = {
            "schema": "carebridge.translation-results.v1",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "git_revision": revision,
            "base_url": base,
            "corpus_sha256": fingerprint,
            "scope": fixture["scope"],
            "health": health["response"],
            "translations": [],
            "qa_cases": [],
        }
    save(args.output, report)
    done = {r["id"] for r in report["translations"]}
    for case in fixture["translation_cases"]:
        if case["id"] in done:
            continue
        result = request_json(
            base,
            "/carebridge/drafts",
            key,
            {
                "memo": case["source"],
                "source_language": case["language"],
                "data_use": "synthetic_deidentified",
            },
        )
        entry = {"id": case["id"], **result}
        if result["http_status"] == 200:
            draft = result["response"]
            refs = [s["evidence"] for s in draft["translation"]["segments"]]
            entry["source_evidence_valid"] = (
                draft["source"]["text"] == case["source"]
                and draft["source"]["sha256"] == hashlib.sha256(case["source"].encode()).hexdigest()
                and [r["segment_id"] for r in refs]
                == [f"s{i:03d}" for i in range(1, len(case["reference"]) + 1)]
                and all(case["source"][r["start"] : r["end"]] == r["quote"] for r in refs)
            )
            entry["sharing_blocked"] = (
                draft["qa"]["external_share_allowed"] is False
                and draft["qa"]["requires_human_review"] is True
            )
        report["translations"].append(entry)
        save(args.output, report)
        print(
            f"translation {case['id']}: HTTP {result['http_status']}, "
            f"{result.get('response', {}).get('status')}, {result['seconds']}s",
            flush=True,
        )
    by_id = {c["id"]: c for c in fixture["translation_cases"]}
    done = {r["id"] for r in report["qa_cases"]}
    for case in fixture["qa_cases"]:
        if case["id"] in done:
            continue
        draft = reference_draft(by_id[case["base_id"]], copy.deepcopy(case["translation"]))
        result = request_json(base, "/carebridge/qa/semantic", key, draft)
        report["qa_cases"].append(
            {
                "id": case["id"],
                "expected": case["expected"],
                "reason": case["reason"],
                **result,
            }
        )
        save(args.output, report)
        print(
            f"QA {case['id']}: HTTP {result['http_status']}, "
            f"{result.get('response', {}).get('status')}, "
            f"model={result.get('response', {}).get('model_checked')}",
            flush=True,
        )
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["summary"] = summarize(report)
    save(args.output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)
    return (
        0
        if (
            report["summary"]["translation_http_200"] == len(fixture["translation_cases"])
            and report["summary"]["qa_request_failures"] == 0
            and report["summary"]["injected_errors_caught"]
            == sum(c["expected"] == "hold" for c in fixture["qa_cases"])
            and report["summary"]["faithful_controls_review_required"]
            == sum(c["expected"] == "review_required" for c in fixture["qa_cases"])
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
