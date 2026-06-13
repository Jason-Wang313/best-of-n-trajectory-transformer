from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..config import FIGURES_DIR, RESULTS_DIR, ExperimentConfig, ensure_output_dirs
from ..diagnostics import SupportThresholds, calibrate_support
from ..environment import generate_offline_dataset
from ..model import SmoothedAutoregressiveTT
from ..planning import (
    SupportCalibratedPlanSieve,
    candidate_diversity,
    sample_candidates,
    select_oracle,
    select_raw,
)
from ..theory import OutcomeDistribution, expected_selected_utility, monte_carlo_selected_utility, tail_misalignment_example
from ..tokenizer import TrajectoryTokenizer
from .core import aggregate_rows, fit_variant, stable_variant_offset, variant_config


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _record(variant: str, n: int, method: str, selection, risk_rate: float, diversity: float) -> dict:
    d = selection.candidate.diagnostics
    return {
        "variant": variant,
        "N": int(n),
        "method": method,
        "predicted_return": d.predicted_return,
        "realized_return": d.realized_return,
        "avg_logprob": d.avg_logprob,
        "prefix_surprise": d.prefix_surprise,
        "dynamics_error": d.dynamics_error,
        "action_energy": d.action_energy,
        "candidate_risk_rate": risk_rate,
        "candidate_diversity": diversity,
        "blocked": float(selection.blocked),
        "accepted_count": float(selection.accepted_count),
        "mode_signature": repr(selection.candidate.diagnostics.mode_signature),
    }


def _tag_rows(rows: list[dict], *, family: str, setting: str, value: str | float | int) -> list[dict]:
    tagged = []
    for row in rows:
        item = dict(row)
        item["family"] = family
        item["setting"] = setting
        item["setting_value"] = value
        tagged.append(item)
    return tagged


def _run_aggregate(
    *,
    family: str,
    setting: str,
    value: str | float | int,
    cfg: ExperimentConfig,
    variant: str,
    tokenizer: TrajectoryTokenizer,
) -> list[dict]:
    episode_rows, thresholds = _run_variant_with_tokenizer(cfg, variant, tokenizer)
    aggregate = aggregate_rows(episode_rows, {variant: thresholds})
    return _tag_rows(aggregate, family=family, setting=setting, value=value)


def _run_variant_with_tokenizer(
    cfg: ExperimentConfig,
    variant: str,
    tokenizer: TrajectoryTokenizer,
) -> tuple[list[dict], SupportThresholds]:
    model, thresholds, _dataset = fit_variant(cfg, variant, tokenizer)
    sieve = SupportCalibratedPlanSieve(thresholds)
    variant_cfg = variant_config(cfg, variant)
    rng = np.random.default_rng(cfg.seed * 1009 + stable_variant_offset(variant))
    rows: list[dict] = []
    horizon = int(variant_cfg["horizon"])
    for _episode in range(cfg.eval_episodes):
        initial_state = float(rng.normal(0.0, 0.10))
        for n in cfg.n_values:
            candidates = sample_candidates(
                model=model,
                tokenizer=tokenizer,
                initial_state=initial_state,
                horizon=horizon,
                n=int(n),
                rng=rng,
                temperature=float(variant_cfg["temperature"]),
            )
            diversity = candidate_diversity(candidates)
            sieve_selection = sieve.select(candidates)
            risk_rate = sieve_selection.candidate_risk_rate
            raw_selection = select_raw(candidates, score_mode=str(variant_cfg["score_mode"]))
            oracle_selection = select_oracle(candidates)
            rows.append(_record(variant, int(n), "raw", raw_selection, risk_rate, diversity))
            rows.append(_record(variant, int(n), "sieve", sieve_selection, risk_rate, diversity))
            rows.append(_record(variant, int(n), "oracle", oracle_selection, risk_rate, diversity))
    return rows, thresholds


def _run_custom_bias(
    *,
    family: str,
    setting: str,
    value: str,
    cfg: ExperimentConfig,
    variant: str,
    tokenizer: TrajectoryTokenizer,
    reward_tail_bias: float,
    action_tail_bias: float,
) -> list[dict]:
    variant_cfg = variant_config(cfg, variant)
    dataset = generate_offline_dataset(
        n_trajectories=cfg.train_trajectories,
        horizon=int(variant_cfg["horizon"]),
        seed=cfg.seed + stable_variant_offset(variant),
        high_support=float(variant_cfg["high_support"]),
        risky_support=float(variant_cfg["risky_support"]),
    )
    tokens = [tokenizer.encode(t.states, t.actions, t.rewards) for t in dataset]
    model = SmoothedAutoregressiveTT(
        tokenizer=tokenizer,
        context=cfg.context,
        alpha=float(variant_cfg["alpha"]),
        reward_tail_bias=reward_tail_bias,
        action_tail_bias=action_tail_bias,
    ).fit(tokens)
    thresholds = calibrate_support(tokens, [float(t.states[0]) for t in dataset], model, tokenizer)
    sieve = SupportCalibratedPlanSieve(thresholds)
    rng = np.random.default_rng(cfg.seed * 733 + stable_variant_offset(value))
    rows: list[dict] = []
    for _episode in range(cfg.eval_episodes):
        initial_state = float(rng.normal(0.0, 0.10))
        for n in cfg.n_values:
            candidates = sample_candidates(
                model=model,
                tokenizer=tokenizer,
                initial_state=initial_state,
                horizon=int(variant_cfg["horizon"]),
                n=int(n),
                rng=rng,
                temperature=float(variant_cfg["temperature"]),
            )
            diversity = candidate_diversity(candidates)
            sieve_selection = sieve.select(candidates)
            risk_rate = sieve_selection.candidate_risk_rate
            rows.append(_record(variant, int(n), "raw", select_raw(candidates), risk_rate, diversity))
            rows.append(_record(variant, int(n), "sieve", sieve_selection, risk_rate, diversity))
            rows.append(_record(variant, int(n), "oracle", select_oracle(candidates), risk_rate, diversity))
    aggregate = aggregate_rows(rows, {variant: thresholds})
    return _tag_rows(aggregate, family=family, setting=setting, value=value)


def _run_sieve_ablation(*, cfg: ExperimentConfig, variant: str, tokenizer: TrajectoryTokenizer) -> list[dict]:
    model, thresholds, _dataset = fit_variant(cfg, variant, tokenizer)
    variant_cfg = variant_config(cfg, variant)
    rng = np.random.default_rng(cfg.seed * 443 + stable_variant_offset("sieve_ablation"))
    methods = {
        "full_sieve": SupportCalibratedPlanSieve(thresholds),
        "likelihood_only": SupportCalibratedPlanSieve(
            SupportThresholds(thresholds.min_avg_logprob, 999.0, 999.0, thresholds.max_risk_rate)
        ),
        "prefix_only": SupportCalibratedPlanSieve(
            SupportThresholds(-999.0, thresholds.max_prefix_surprise, 999.0, thresholds.max_risk_rate)
        ),
        "dynamics_only": SupportCalibratedPlanSieve(
            SupportThresholds(-999.0, 999.0, thresholds.max_dynamics_error, thresholds.max_risk_rate)
        ),
        "lenient_full": SupportCalibratedPlanSieve(
            SupportThresholds(
                thresholds.min_avg_logprob - 0.20,
                thresholds.max_prefix_surprise + 0.60,
                thresholds.max_dynamics_error + 0.12,
                thresholds.max_risk_rate,
            )
        ),
        "strict_full": SupportCalibratedPlanSieve(
            SupportThresholds(
                thresholds.min_avg_logprob + 0.05,
                thresholds.max_prefix_surprise - 0.20,
                max(0.0, thresholds.max_dynamics_error - 0.04),
                thresholds.max_risk_rate,
            )
        ),
    }
    rows: list[dict] = []
    for _episode in range(cfg.eval_episodes):
        initial_state = float(rng.normal(0.0, 0.10))
        candidates = sample_candidates(
            model=model,
            tokenizer=tokenizer,
            initial_state=initial_state,
            horizon=int(variant_cfg["horizon"]),
            n=64,
            rng=rng,
            temperature=float(variant_cfg["temperature"]),
        )
        diversity = candidate_diversity(candidates)
        full_selection = methods["full_sieve"].select(candidates)
        risk_rate = full_selection.candidate_risk_rate
        rows.append(_record(variant, 64, "raw", select_raw(candidates), risk_rate, diversity))
        rows.append(_record(variant, 64, "oracle", select_oracle(candidates), risk_rate, diversity))
        for name, sieve in methods.items():
            rows.append(_record(variant, 64, name, sieve.select(candidates), risk_rate, diversity))
    aggregate = aggregate_rows(rows, {variant: thresholds})
    return _tag_rows(aggregate, family="sieve_ablation", setting="ablation", value="N64")


def _law_rows(*, trials: int = 20000) -> list[dict]:
    examples = {
        "tail_misalignment": tail_misalignment_example(),
        "rare_bad_tail": OutcomeDistribution(
            probabilities=np.asarray([0.82, 0.14, 0.04]),
            scores=np.asarray([0.20, 0.95, 2.30]),
            utilities=np.asarray([0.35, 0.80, -1.10]),
        ).normalized(),
        "benign_tail": OutcomeDistribution(
            probabilities=np.asarray([0.70, 0.24, 0.06]),
            scores=np.asarray([0.30, 0.80, 1.80]),
            utilities=np.asarray([0.40, 0.90, 1.20]),
        ).normalized(),
    }
    ns = [1, 2, 4, 8, 16, 32, 64, 128]
    rows: list[dict] = []
    for name, dist in examples.items():
        for n in ns:
            rows.append(
                {
                    "family": "exact_law_stress",
                    "setting": name,
                    "N": n,
                    "exact_utility": expected_selected_utility(dist, n),
                    "monte_carlo_utility": monte_carlo_selected_utility(dist, n, trials=trials, seed=900 + n),
                }
            )
    return rows


def _row(rows: list[dict], *, family: str, setting: str, method: str, n: int) -> dict:
    matches = [
        row
        for row in rows
        if row["family"] == family and row["setting"] == setting and row["method"] == method and int(row["N"]) == n
    ]
    if not matches:
        raise ValueError(f"missing row for {family=} {setting=} {method=} {n=}")
    return matches[0]


def _delta(rows: list[dict], *, family: str, setting: str, method: str, high_n: int, metric: str) -> float:
    low = _row(rows, family=family, setting=setting, method=method, n=1)
    high = _row(rows, family=family, setting=setting, method=method, n=high_n)
    return float(high[metric] - low[metric])


def audit_expansion(rows: list[dict], law_rows: list[dict]) -> dict:
    candidate_pred_gain = _delta(
        rows, family="candidate_count_tail", setting="horizon_stress", method="raw", high_n=256, metric="predicted_return"
    )
    candidate_realized_change = _delta(
        rows, family="candidate_count_tail", setting="horizon_stress", method="raw", high_n=256, metric="realized_return"
    )
    tail_sieve_256 = _row(rows, family="candidate_count_tail", setting="horizon_stress", method="sieve", n=256)
    tail_raw_256 = _row(rows, family="candidate_count_tail", setting="horizon_stress", method="raw", n=256)
    ablation_raw = _row(rows, family="sieve_ablation", setting="ablation", method="raw", n=64)
    ablation_full = _row(rows, family="sieve_ablation", setting="ablation", method="full_sieve", n=64)

    horizon_rows = [row for row in rows if row["family"] == "horizon_sweep" and row["method"] == "raw" and int(row["N"]) == 64]
    context_rows = [row for row in rows if row["family"] == "context_sweep" and row["method"] == "raw" and int(row["N"]) == 64]
    tokenizer_rows = [row for row in rows if row["family"] == "tokenizer_sweep" and row["method"] == "raw" and int(row["N"]) == 64]
    temp_rows = [row for row in rows if row["family"] == "temperature_sweep" and row["method"] == "raw" and int(row["N"]) == 64]
    bias_rows = [row for row in rows if row["family"] == "tail_bias_stress" and row["method"] == "raw" and int(row["N"]) == 64]
    ablation_prefix_reduction = float(ablation_raw["prefix_surprise"] - ablation_full["prefix_surprise"])
    ablation_dynamics_reduction = float(ablation_raw["dynamics_error"] - ablation_full["dynamics_error"])
    temp_realized_span = max(float(r["realized_return"]) for r in temp_rows) - min(float(r["realized_return"]) for r in temp_rows)
    bias_realized_span = max(float(r["realized_return"]) for r in bias_rows) - min(float(r["realized_return"]) for r in bias_rows)

    law_tail = [row for row in law_rows if row["setting"] == "tail_misalignment"]
    law_benign = [row for row in law_rows if row["setting"] == "benign_tail"]
    law_tail_drop = float(law_tail[-1]["exact_utility"] - law_tail[0]["exact_utility"])
    law_benign_gain = float(law_benign[-1]["exact_utility"] - law_benign[0]["exact_utility"])

    def claim(status: bool, value: float, threshold: float, description: str) -> dict:
        return {
            "status": "pass" if status else "fail",
            "value": float(value),
            "threshold": float(threshold),
            "description": description,
        }

    claims = {
        "candidate_tail_predicted_reward_extremizes": claim(
            candidate_pred_gain > 4.0,
            candidate_pred_gain,
            4.0,
            "Increasing candidate count to 256 raises raw decoded reward-token score in horizon stress.",
        ),
        "candidate_tail_realized_return_drops": claim(
            candidate_realized_change < -2.0,
            candidate_realized_change,
            -2.0,
            "The same high-candidate raw selection lowers realized simulator return by more than two return units.",
        ),
        "sieve_repairs_high_candidate_tail": claim(
            float(tail_sieve_256["realized_return"] - tail_raw_256["realized_return"]) > 3.0,
            float(tail_sieve_256["realized_return"] - tail_raw_256["realized_return"]),
            3.0,
            "The support-calibrated sieve improves realized return at N=256 in horizon stress.",
        ),
        "sieve_ablation_reduces_token_pathology": claim(
            min(ablation_prefix_reduction, ablation_dynamics_reduction) > 0.05,
            min(ablation_prefix_reduction, ablation_dynamics_reduction),
            0.05,
            "The full sieve lowers both prefix surprise and token/simulator dynamics error relative to raw selection.",
        ),
        "horizon_changes_realized_tail": claim(
            max(float(r["realized_return"]) for r in horizon_rows) - min(float(r["realized_return"]) for r in horizon_rows) > 1.0,
            max(float(r["realized_return"]) for r in horizon_rows) - min(float(r["realized_return"]) for r in horizon_rows),
            1.0,
            "Horizon changes the raw realized tail at N=64.",
        ),
        "context_changes_support_risk": claim(
            max(float(r["candidate_risk_rate"]) for r in context_rows) - min(float(r["candidate_risk_rate"]) for r in context_rows) > 0.05,
            max(float(r["candidate_risk_rate"]) for r in context_rows) - min(float(r["candidate_risk_rate"]) for r in context_rows),
            0.05,
            "Autoregressive context length changes high-N support risk.",
        ),
        "tokenizer_resolution_changes_dynamics_error": claim(
            max(float(r["dynamics_error"]) for r in tokenizer_rows) - min(float(r["dynamics_error"]) for r in tokenizer_rows) > 0.05,
            max(float(r["dynamics_error"]) for r in tokenizer_rows) - min(float(r["dynamics_error"]) for r in tokenizer_rows),
            0.05,
            "Tokenizer resolution changes selected token/simulator dynamics inconsistency.",
        ),
        "temperature_changes_realized_tail": claim(
            temp_realized_span > 3.0,
            temp_realized_span,
            3.0,
            "Sampling temperature changes high-candidate realized return even when support-risk rates remain saturated.",
        ),
        "tail_bias_changes_realized_tail": claim(
            bias_realized_span > 2.0,
            bias_realized_span,
            2.0,
            "Reward/action tail bias changes high-candidate realized return.",
        ),
        "finite_law_detects_bad_tail": claim(
            law_tail_drop < -0.7,
            law_tail_drop,
            -0.7,
            "The exact finite-candidate law predicts utility collapse under a bad high-score tail.",
        ),
        "finite_law_keeps_benign_tail_positive": claim(
            law_benign_gain > 0.3,
            law_benign_gain,
            0.3,
            "The exact law distinguishes a benign high-score tail from a harmful one.",
        ),
    }
    return {
        "all_passed": all(item["status"] == "pass" for item in claims.values()),
        "claims": claims,
        "summary": (
            f"N=256 predicted gain {candidate_pred_gain:.3f}, realized change {candidate_realized_change:.3f}, "
            f"sieve repair {float(tail_sieve_256['realized_return'] - tail_raw_256['realized_return']):.3f}; "
            f"finite-law bad-tail change {law_tail_drop:.3f}."
        ),
    }


def plot_expansion(rows: list[dict], law_rows: list[dict], outdir: Path) -> list[str]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    tail = [r for r in rows if r["family"] == "candidate_count_tail"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for method, marker in [("raw", "o"), ("sieve", "s"), ("oracle", "^")]:
        sub = sorted([r for r in tail if r["method"] == method], key=lambda r: int(r["N"]))
        axes[0].plot([int(r["N"]) for r in sub], [r["realized_return"] for r in sub], marker=marker, label=method)
    raw = sorted([r for r in tail if r["method"] == "raw"], key=lambda r: int(r["N"]))
    axes[1].plot([int(r["N"]) for r in raw], [r["predicted_return"] for r in raw], marker="o", label="predicted")
    axes[1].plot([int(r["N"]) for r in raw], [r["candidate_risk_rate"] for r in raw], marker="s", label="risk")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("candidate count")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    axes[0].set_title("Candidate-count tail")
    axes[0].set_ylabel("realized return")
    axes[1].set_title("Raw reward/risk")
    fig.tight_layout()
    path = outdir / "candidate_count_tail.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    for family, name, metric in [
        ("horizon_sweep", "horizon_context_sweep.png", "realized_return"),
        ("tokenizer_sweep", "tokenizer_temperature_sweep.png", "dynamics_error"),
    ]:
        fig, ax = plt.subplots(figsize=(7.0, 3.8))
        for method in ["raw", "sieve"]:
            sub = [r for r in rows if r["family"] == family and r["method"] == method and int(r["N"]) == 64]
            ax.plot([str(r["setting_value"]) for r in sub], [r[metric] for r in sub], marker="o", label=method)
        ax.set_title(family.replace("_", " "))
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = outdir / name
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for family, marker in [("context_sweep", "o"), ("temperature_sweep", "s")]:
        sub = [r for r in rows if r["family"] == family and r["method"] == "raw" and int(r["N"]) == 64]
        ax.plot([str(r["setting_value"]) for r in sub], [r["candidate_risk_rate"] for r in sub], marker=marker, label=family)
    ax.set_title("Context and temperature risk stress")
    ax.set_ylabel("candidate risk rate")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = outdir / "context_temperature_risk.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    sub = [r for r in rows if r["family"] == "sieve_ablation" and int(r["N"]) == 64]
    ax.bar([r["method"] for r in sub], [r["realized_return"] for r in sub])
    ax.set_title("Sieve ablation at N=64")
    ax.set_ylabel("realized return")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path = outdir / "sieve_ablation.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    sub = [r for r in rows if r["family"] == "tail_bias_stress" and r["method"] == "raw" and int(r["N"]) == 64]
    ax.plot([r["setting_value"] for r in sub], [r["predicted_return"] for r in sub], marker="o", label="predicted")
    ax.plot([r["setting_value"] for r in sub], [r["realized_return"] for r in sub], marker="s", label="realized")
    ax.set_title("Reward/action tail-bias stress")
    ax.set_ylabel("return")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = outdir / "tail_bias_stress.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for setting in sorted({r["setting"] for r in law_rows}):
        sub = sorted([r for r in law_rows if r["setting"] == setting], key=lambda r: int(r["N"]))
        ax.plot([int(r["N"]) for r in sub], [r["exact_utility"] for r in sub], marker="o", label=setting)
    ax.set_xscale("log", base=2)
    ax.set_title("Finite-candidate law stress")
    ax.set_xlabel("candidate count")
    ax.set_ylabel("expected selected utility")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = outdir / "exact_law_stress.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))
    return paths


def run_expansion_suite(*, quick: bool = False, write_figures: bool = True, output_dir: Path | None = None) -> dict:
    ensure_output_dirs()
    outdir = Path(output_dir) if output_dir is not None else RESULTS_DIR / "expansion"
    outdir.mkdir(parents=True, exist_ok=True)
    episodes = 1 if quick else 8
    train = 80 if quick else 560
    rows: list[dict] = []
    default_tokenizer = TrajectoryTokenizer()

    rows.extend(
        _run_aggregate(
            family="candidate_count_tail",
            setting="horizon_stress",
            value="Nmax256",
            cfg=ExperimentConfig(
                train_trajectories=train,
                eval_episodes=episodes,
                n_values=(1, 16, 64, 128, 256),
                seed=101,
                variants=("horizon_stress",),
            ),
            variant="horizon_stress",
            tokenizer=default_tokenizer,
        )
    )

    for horizon in [6, 10, 16, 22]:
        rows.extend(
            _run_aggregate(
                family="horizon_sweep",
                setting="base_horizon",
                value=horizon,
                cfg=ExperimentConfig(
                    horizon=horizon,
                    train_trajectories=train,
                    eval_episodes=episodes,
                    n_values=(1, 64),
                    seed=110 + horizon,
                    variants=("horizon_stress",),
                ),
                variant="horizon_stress",
                tokenizer=default_tokenizer,
            )
        )

    for context in [2, 5, 8]:
        rows.extend(
            _run_aggregate(
                family="context_sweep",
                setting="context",
                value=context,
                cfg=ExperimentConfig(
                    context=context,
                    train_trajectories=train,
                    eval_episodes=episodes,
                    n_values=(1, 64),
                    seed=130 + context,
                    variants=("horizon_stress",),
                ),
                variant="horizon_stress",
                tokenizer=default_tokenizer,
            )
        )

    tokenizers = {
        "coarse": TrajectoryTokenizer(state_bins=15, action_bins=7, reward_bins=15),
        "default": TrajectoryTokenizer(),
        "fine": TrajectoryTokenizer(state_bins=31, action_bins=15, reward_bins=31),
    }
    for name, tokenizer in tokenizers.items():
        rows.extend(
            _run_aggregate(
                family="tokenizer_sweep",
                setting="resolution",
                value=name,
                cfg=ExperimentConfig(
                    train_trajectories=train,
                    eval_episodes=episodes,
                    n_values=(1, 64),
                    seed=150 + len(name),
                    variants=("horizon_stress",),
                ),
                variant="horizon_stress",
                tokenizer=tokenizer,
            )
        )

    for temp in [0.85, 1.12, 1.45, 1.80]:
        rows.extend(
            _run_aggregate(
                family="temperature_sweep",
                setting="temperature",
                value=temp,
                cfg=ExperimentConfig(
                    sample_temperature=temp,
                    train_trajectories=train,
                    eval_episodes=episodes,
                    n_values=(1, 64),
                    seed=int(170 + temp * 100),
                    variants=("horizon_stress",),
                ),
                variant="horizon_stress",
                tokenizer=default_tokenizer,
            )
        )

    rows.extend(
        _run_sieve_ablation(
            cfg=ExperimentConfig(
                train_trajectories=train,
                eval_episodes=episodes,
                n_values=(64,),
                seed=220,
                variants=("horizon_stress",),
            ),
            variant="horizon_stress",
            tokenizer=default_tokenizer,
        )
    )

    for label, reward_bias, action_bias in [
        ("low_tail", 0.18, 0.08),
        ("default_tail", 0.48, 0.24),
        ("high_reward_tail", 0.74, 0.24),
        ("high_reward_action_tail", 0.74, 0.42),
    ]:
        rows.extend(
            _run_custom_bias(
                family="tail_bias_stress",
                setting="bias",
                value=label,
                cfg=ExperimentConfig(
                    train_trajectories=train,
                    eval_episodes=episodes,
                    n_values=(1, 64),
                    seed=260,
                    variants=("horizon_stress",),
                ),
                variant="horizon_stress",
                tokenizer=default_tokenizer,
                reward_tail_bias=reward_bias,
                action_tail_bias=action_bias,
            )
        )

    law_rows = _law_rows(trials=2000 if quick else 20000)
    claims = audit_expansion(rows, law_rows)
    figure_paths = plot_expansion(rows, law_rows, FIGURES_DIR) if write_figures else []

    write_csv(outdir / "aggregate_metrics.csv", rows)
    write_csv(outdir / "finite_law_stress.csv", law_rows)
    (outdir / "claims.json").write_text(json.dumps(claims, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "aggregate_metrics": str(outdir / "aggregate_metrics.csv"),
        "finite_law_stress": str(outdir / "finite_law_stress.csv"),
        "claims": str(outdir / "claims.json"),
        "figures": figure_paths,
        "summary": claims["summary"],
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    manifest = run_expansion_suite()
    print(manifest["summary"])
    print(f"Manifest: {manifest['claims']}")


if __name__ == "__main__":
    main()
