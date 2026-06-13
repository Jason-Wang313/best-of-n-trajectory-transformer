from pathlib import Path

from trajectory_token_sieve.experiments.run_expansion_suite import run_expansion_suite


def test_expansion_suite_quick_writes_claims(tmp_path):
    manifest = run_expansion_suite(quick=True, write_figures=False, output_dir=tmp_path / "expansion")

    assert Path(manifest["aggregate_metrics"]).exists()
    assert Path(manifest["finite_law_stress"]).exists()
    assert Path(manifest["claims"]).exists()
    assert Path(tmp_path / "expansion" / "manifest.json").exists()
