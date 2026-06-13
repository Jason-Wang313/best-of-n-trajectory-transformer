from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
DOCS_DIR = REPO_ROOT / "docs"
PAPER_DIR = REPO_ROOT / "paper"


@dataclass(frozen=True)
class ExperimentConfig:
    horizon: int = 10
    train_trajectories: int = 640
    eval_episodes: int = 24
    seed: int = 13
    n_values: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    context: int = 5
    alpha: float = 0.04
    sample_temperature: float = 1.12
    high_support: float = 0.08
    risky_support: float = 0.0
    output_prefix: str = "all"
    variants: tuple[str, ...] = field(
        default_factory=lambda: (
            "in_support",
            "out_of_support",
            "anti_aligned_scorer",
            "horizon_stress",
        )
    )


def ensure_output_dirs() -> None:
    for path in (RESULTS_DIR, FIGURES_DIR, DOCS_DIR, PAPER_DIR, PAPER_DIR / "final"):
        path.mkdir(parents=True, exist_ok=True)
