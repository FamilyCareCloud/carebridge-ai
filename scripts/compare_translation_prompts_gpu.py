"""Compare a pinned previous prompt and the working prompt on synthetic data, on CUDA.

Records every response, including invalid ones. This is a development comparison,
not a held-out accuracy score. It never starts a server or changes service records.
"""

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_translation import save  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-only", action="store_true")
    parser.add_argument(
        "--corpus", type=Path, default=ROOT / "evaluation/translation-corpus-v1.json"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path; previous evidence must not be overwritten.")

    from app.carebridge.guidance import guidance_definition, translation_payload
    from app.carebridge.language import korean_output
    from app.carebridge.local import LocalTranslationProvider
    from app.carebridge.provider import (
        SYSTEM_PROMPT,
        translation_prompt_material,
        translation_system_prompt,
    )
    from app.carebridge.safety import check_input
    from app.carebridge.schemas import ModelDraft
    from app.config import Settings

    baseline_commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", args.baseline_ref + "^{commit}"], cwd=ROOT, text=True
    ).strip()
    old_source = subprocess.check_output(
        ["git", "show", baseline_commit + ":app/carebridge/provider.py"], cwd=ROOT
    ).decode("utf-8")
    baseline = next(
        ast.literal_eval(node.value)
        for node in ast.parse(old_source).body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "SYSTEM_PROMPT" for t in node.targets)
    )
    provider = LocalTranslationProvider(
        Settings(
            _env_file=None,
            device="cuda",
            stt_enabled=False,
            carebridge_local_files_only=True,
            carebridge_timeout_seconds=10,
        )
    )
    provider.load()
    corpus_raw = args.corpus.read_bytes()
    fixture = json.loads(corpus_raw.decode("utf-8-sig"))
    report = {
        "schema": "carebridge.prompt-comparison.v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "baseline_commit": baseline_commit,
        "corpus_sha256": hashlib.sha256(corpus_raw).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_snapshot": provider._model.config._commit_hash,
        "scope": "Synthetic development comparison, not independent/native-speaker validation.",
        "prompts": {"baseline": baseline, "candidate": SYSTEM_PROMPT},
        "candidate_only": args.candidate_only,
        "candidate_guidance": json.loads(guidance_definition()),
        "candidate_prompt_material": translation_prompt_material(),
        "results": [],
    }
    for name, prompt in report["prompts"].items():
        if args.candidate_only and name == "baseline":
            continue
        for case in fixture["translation_cases"]:
            started = time.perf_counter()
            entry = {"prompt": name, "id": case["id"]}
            try:
                segments = check_input(case["source"])
                payload = (
                    translation_payload(segments, case["language"])
                    if name == "candidate"
                    else {
                        "source_language": case["language"],
                        "segments": [
                            {"segment_id": s.segment_id, "text": s.quote} for s in segments
                        ],
                    }
                )
                entry["input"] = payload
                system = translation_system_prompt(payload) if name == "candidate" else prompt
                raw = provider._complete(
                    system + "\nSchema: " + json.dumps(ModelDraft.model_json_schema()),
                    payload,
                )
                entry["raw"] = raw
                parsed = ModelDraft.model_validate_json(raw)
                entry["parsed"] = parsed.model_dump()
                entry["valid_ids"] = [s.segment_id for s in parsed.segments] == [
                    s.segment_id for s in segments
                ]
                entry["korean_output"] = all(korean_output(s.translation) for s in parsed.segments)
            except Exception as exc:
                entry["error"] = type(exc).__name__
            entry["seconds"] = round(time.perf_counter() - started, 3)
            report["results"].append(entry)
            save(args.output, report)
            print(f"{name} {case['id']}: {entry.get('error', 'completed')}", flush=True)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    save(args.output, report)


if __name__ == "__main__":
    main()
