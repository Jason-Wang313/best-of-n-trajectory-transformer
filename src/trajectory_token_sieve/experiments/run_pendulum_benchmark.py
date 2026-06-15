from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..pendulum_benchmark import run_pendulum_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    result = run_pendulum_benchmark(
        quick=args.quick,
        output_dir=args.output,
        write_figures=not args.no_figures,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
