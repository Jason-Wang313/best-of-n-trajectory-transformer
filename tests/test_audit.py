from pathlib import Path

from trajectory_transformer_best_of_n.audit import FORBIDDEN_OVERCLAIMS, write_claim_audit
from trajectory_transformer_best_of_n.config import RESULTS_DIR


def test_claim_audit_writes_json_and_checks_forbidden_phrases():
    assert "state-of-the-art" in FORBIDDEN_OVERCLAIMS
    result = write_claim_audit()
    assert result["verdict"] in {"pass", "fail"}
    assert Path(RESULTS_DIR / "claims_status.json").exists()
