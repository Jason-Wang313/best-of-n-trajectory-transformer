from __future__ import annotations

from ..config import ExperimentConfig
from .core import run_suite


def main() -> None:
    cfg = ExperimentConfig()
    payload = run_suite(cfg)
    print("full experiment complete")
    print(payload["summary_by_variant"])


if __name__ == "__main__":
    main()
