"""Recheck the bounded habitual-context audit change against a frozen full GPU run.

Keeps the parent immutable. Replays every affected generated draft and QA fixture,
three lost-repetition mutations, and the three contrast regressions on real CUDA.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.carebridge.facts import HABITUAL_WALKING  # noqa: E402
from app.carebridge.local import LocalTranslationProvider  # noqa: E402
from app.carebridge.provider import translation_prompt_material  # noqa: E402
from app.carebridge.safety import digest  # noqa: E402
from app.carebridge.semantic import semantic_prompt_material  # noqa: E402
from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from scripts.evaluate_translation import reference_draft, save  # noqa: E402


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite evidence")
    parent_raw, corpus_raw = args.parent.read_bytes(), args.corpus.read_bytes()
    parent, corpus = json.loads(parent_raw), json.loads(corpus_raw)
    assert parent.get("finished_at") and parent["corpus_sha256"] == sha(corpus_raw)
    codes = {
        p.relative_to(ROOT).as_posix(): sha(p.read_bytes())
        for p in sorted((ROOT / "app").rglob("*.py"))
    }
    changed = [p for p, value in codes.items() if value != parent["code_sha256"].get(p)]
    assert set(changed) == {
        "app/carebridge/facts.py",
        "app/carebridge/guidance.py",
        "app/carebridge/semantic.py",
    }, changed
    old_pipeline = parent["translations"][0]["generation_binding"]["pipeline"]
    assert digest(translation_prompt_material()) == old_pipeline["translation_prompt_sha256"]
    cases = {c["id"]: c for c in corpus["translation_cases"]}
    affected = {cid for cid, c in cases.items() if HABITUAL_WALKING.search(c["source"])}
    assert len(affected) == 3
    selected_qa = [
        c
        for c in corpus["qa_cases"]
        if c["base_id"] in affected or c["id"].endswith("-negation_scope")
    ]
    assert len(selected_qa) == 6
    requests = (
        [
            {
                "kind": "generated",
                "id": r["id"],
                "draft": r["response"],
                "expected": "review_required",
            }
            for r in parent["translations"]
            if r["id"] in affected
        ]
        + [
            {
                "kind": "qa",
                "id": c["id"],
                "draft": reference_draft(cases[c["base_id"]], c["translation"]),
                "expected": c["expected"],
            }
            for c in selected_qa
        ]
        + [
            {
                "kind": "lost_repetition",
                "id": cid + "-lost-repetition",
                "draft": reference_draft(cases[cid], ["걸은 후 피곤해 보였습니다."]),
                "expected": "hold",
            }
            for cid in sorted(affected)
        ]
    )
    report = {
        "schema": "carebridge.targeted-audit-evaluation.v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Targeted semantic rerun, not another full generation/workflow benchmark. Parent stays immutable. No native/held-out validation.",
        "parent_sha256": sha(parent_raw),
        "corpus_sha256": sha(corpus_raw),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "runner_sha256": sha(Path(__file__).read_bytes()),
        "code_sha256": codes,
        "changed_app_files": changed,
        "affected_translation_ids": sorted(affected),
        "translation_prompt_unchanged": True,
        "semantic_prompt_sha256": digest(semantic_prompt_material()),
        "parent_semantic_prompt_sha256": old_pipeline["semantic_prompt_sha256"],
        "results": [],
    }
    import torch
    from fastapi.testclient import TestClient

    assert torch.cuda.is_available()
    config = Settings(
        _env_file=None,
        device="cuda",
        api_key="offline-evaluation-only",
        stt_enabled=False,
        carebridge_backend="local",
        carebridge_preload=True,
        carebridge_local_files_only=True,
        carebridge_semantic_check_enabled=True,
        carebridge_timeout_seconds=10,
    )
    provider = LocalTranslationProvider(config)
    with TestClient(create_app(config, provider=provider)) as client:
        report["gpu"] = torch.cuda.get_device_name(0)
        report["model_snapshot"] = provider._model.config._commit_hash
        for item in requests:
            started = time.perf_counter()
            response = client.post(
                "/carebridge/qa/semantic", headers={"X-API-Key": config.api_key}, json=item["draft"]
            )
            result = {
                **item,
                "http_status": response.status_code,
                "response": response.json(),
                "seconds": round(time.perf_counter() - started, 3),
            }
            result["passed"] = (
                response.status_code == 200 and response.json().get("status") == item["expected"]
            )
            report["results"].append(result)
            save(args.output, report)
            print(item["kind"], item["id"], result["passed"], flush=True)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["passed"] = sum(r["passed"] for r in report["results"])
    report["total"] = len(report["results"])
    save(args.output, report)
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
