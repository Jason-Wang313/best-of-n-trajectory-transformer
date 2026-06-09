from __future__ import annotations

import json
from pathlib import Path

from .config import DOCS_DIR, PAPER_DIR, REPO_ROOT, RESULTS_DIR


FORBIDDEN_OVERCLAIMS = [
    "state-of-the-art",
    "solves offline rl",
    "guarantees safe",
    "guaranteed safe",
    "benchmark-proven",
    "real robot validated",
    "human validated",
]


def scan_claims() -> dict:
    files = [
        REPO_ROOT / "README.md",
        DOCS_DIR / "claims.md",
        DOCS_DIR / "final_audit.md",
        PAPER_DIR / "main.tex",
    ]
    hits: list[dict] = []
    for path in files:
        if not path.exists():
            hits.append({"file": str(path), "phrase": "missing file"})
            continue
        text = path.read_text(encoding="utf-8").lower()
        for phrase in FORBIDDEN_OVERCLAIMS:
            if phrase in text:
                hits.append({"file": str(path), "phrase": phrase})
    required = {
        "summary_json": (RESULTS_DIR / "summary.json").exists(),
        "final_pdf": (PAPER_DIR / "final" / "iclr_submission.pdf").exists(),
        "novelty_map": (DOCS_DIR / "novelty_map.md").exists(),
        "proof_attack": (DOCS_DIR / "proof_attack.md").exists(),
    }
    verdict = "pass" if not hits and all(required.values()) else "fail"
    return {
        "verdict": verdict,
        "forbidden_hits": hits,
        "required_outputs": required,
        "files_scanned": [str(path) for path in files],
    }


def write_claim_audit() -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = scan_claims()
    (RESULTS_DIR / "claims_status.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Claim Audit", "", f"Verdict: **{result['verdict']}**", ""]
    if result["forbidden_hits"]:
        lines.append("Forbidden or missing-file hits:")
        for hit in result["forbidden_hits"]:
            lines.append(f"- {hit['file']}: {hit['phrase']}")
    else:
        lines.append("No forbidden overclaim phrases found in scanned claim surfaces.")
    lines.append("")
    lines.append("Required outputs:")
    for key, present in result["required_outputs"].items():
        lines.append(f"- {key}: {'present' if present else 'missing'}")
    (RESULTS_DIR / "claims_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
