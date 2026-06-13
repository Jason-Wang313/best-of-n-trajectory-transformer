from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .theory import expected_selected_utility, monte_carlo_selected_utility, tail_misalignment_example


def _group(rows: list[dict], variant: str, method: str) -> list[dict]:
    selected = [r for r in rows if r["variant"] == variant and r["method"] == method]
    return sorted(selected, key=lambda r: int(r["N"]))


def plot_reward_extremization(rows: list[dict], outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    available = {r["variant"] for r in rows}
    variant = "horizon_stress" if "horizon_stress" in available else "out_of_support"
    raw = _group(rows, variant, "raw")
    sieve = _group(rows, variant, "sieve")
    xs = [int(r["N"]) for r in raw]

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6))
    ax = axes[0, 0]
    ax.plot(xs, [r["predicted_return"] for r in raw], marker="o", label="raw predicted")
    ax.plot(xs, [r["realized_return"] for r in raw], marker="s", label="raw realized")
    ax.set_xscale("log", base=2)
    ax.set_title("Reward-token extremization")
    ax.set_xlabel("Candidate count")
    ax.set_ylabel("Return")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(xs, [r["avg_logprob"] for r in raw], marker="o", label="raw")
    ax.plot(xs, [r["avg_logprob"] for r in sieve], marker="s", label="sieve")
    ax.set_xscale("log", base=2)
    ax.set_title("Selected support")
    ax.set_xlabel("Candidate count")
    ax.set_ylabel("Average log probability")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.plot(xs, [r["dynamics_error"] for r in raw], marker="o", label="raw")
    ax.plot(xs, [r["dynamics_error"] for r in sieve], marker="s", label="sieve")
    ax.set_xscale("log", base=2)
    ax.set_title("Dynamics inconsistency")
    ax.set_xlabel("Candidate count")
    ax.set_ylabel("Mean token/simulator error")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(xs, [r["candidate_risk_rate"] for r in raw], marker="o", label="candidate risk")
    ax.plot(xs, [r["mode_collapse"] for r in raw], marker="s", label="selected collapse")
    ax.set_xscale("log", base=2)
    ax.set_title("Search risk and collapse")
    ax.set_xlabel("Candidate count")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(outdir / "reward_extremization.png", dpi=180)
    fig.savefig(outdir / "reward_extremization.pdf")
    plt.close(fig)


def plot_repair_comparison(rows: list[dict], outdir: Path) -> None:
    variants = sorted({r["variant"] for r in rows})
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    width = 0.24
    x = np.arange(len(variants))
    methods = ["raw", "sieve", "oracle"]
    colors = ["#5f6c7b", "#2f8f71", "#b7791f"]
    for offset, method, color in zip([-width, 0.0, width], methods, colors):
        vals = []
        for variant in variants:
            candidates = [r for r in rows if r["variant"] == variant and r["method"] == method and int(r["N"]) == 64]
            vals.append(candidates[0]["realized_return"] if candidates else np.nan)
        ax.bar(x + offset, vals, width=width, label=method, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace("_", "\n") for v in variants])
    ax.set_ylabel("Realized return at candidate count 64")
    ax.set_title("Support-Calibrated Plan Sieve versus score-only selection")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "repair_comparison.png", dpi=180)
    fig.savefig(outdir / "repair_comparison.pdf")
    plt.close(fig)


def plot_control_panels(rows: list[dict], outdir: Path) -> None:
    variants = sorted({r["variant"] for r in rows})
    fig, axes = plt.subplots(1, len(variants), figsize=(12.0, 3.3), sharey=True)
    if len(variants) == 1:
        axes = [axes]
    for ax, variant in zip(axes, variants):
        for method, marker in [("raw", "o"), ("sieve", "s")]:
            group = _group(rows, variant, method)
            ax.plot(
                [int(r["N"]) for r in group],
                [r["realized_return"] for r in group],
                marker=marker,
                label=method,
            )
        ax.set_xscale("log", base=2)
        ax.set_title(variant.replace("_", " "))
        ax.set_xlabel("Candidate count")
    axes[0].set_ylabel("Realized return")
    axes[-1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "control_panels.png", dpi=180)
    fig.savefig(outdir / "control_panels.pdf")
    plt.close(fig)


def plot_exact_law(outdir: Path) -> dict[str, list[float]]:
    dist = tail_misalignment_example()
    ns = [1, 2, 4, 8, 16, 32, 64]
    exact = [expected_selected_utility(dist, n) for n in ns]
    mc = [monte_carlo_selected_utility(dist, n, trials=30000, seed=100 + n) for n in ns]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(ns, exact, marker="o", label="exact")
    ax.scatter(ns, mc, label="Monte Carlo", color="#b7791f")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Candidate count")
    ax.set_ylabel("Expected selected utility")
    ax.set_title("Finite-candidate identity under tail misalignment")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "exact_law_validation.png", dpi=180)
    fig.savefig(outdir / "exact_law_validation.pdf")
    plt.close(fig)
    return {"N": ns, "exact": exact, "monte_carlo": mc}


def summarize_by_variant(rows: list[dict]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = defaultdict(dict)
    for variant in sorted({r["variant"] for r in rows}):
        raw1 = next(r for r in rows if r["variant"] == variant and r["method"] == "raw" and int(r["N"]) == 1)
        raw64 = next(r for r in rows if r["variant"] == variant and r["method"] == "raw" and int(r["N"]) == 64)
        sieve64 = next(r for r in rows if r["variant"] == variant and r["method"] == "sieve" and int(r["N"]) == 64)
        summary[variant] = {
            "raw_predicted_gain_1_to_64": float(raw64["predicted_return"] - raw1["predicted_return"]),
            "raw_realized_change_1_to_64": float(raw64["realized_return"] - raw1["realized_return"]),
            "sieve_minus_raw_realized_at_64": float(sieve64["realized_return"] - raw64["realized_return"]),
            "raw_risk_rate_at_64": float(raw64["candidate_risk_rate"]),
        }
    return dict(summary)
