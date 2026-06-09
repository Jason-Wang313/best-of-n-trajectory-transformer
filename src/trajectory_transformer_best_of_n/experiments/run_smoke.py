from __future__ import annotations

from ..config import ExperimentConfig
from .core import run_suite


def main() -> None:
    cfg = ExperimentConfig(
        horizon=8,
        train_trajectories=260,
        eval_episodes=10,
        n_values=(1, 4, 16, 64),
        seed=31,
        output_prefix="smoke",
        variants=("out_of_support",),
    )
    payload = run_suite(cfg, variants=cfg.variants)
    print("smoke complete")
    print(payload["summary_by_variant"])


if __name__ == "__main__":
    main()
