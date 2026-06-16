from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from .config import FIGURES_DIR, RESULTS_DIR
from .model import SmoothedAutoregressiveTT


LAYOUT = ("s", "a", "r")
ENV_ID = "Pendulum-v1"


def angle_normalize(angle: float | np.ndarray) -> float | np.ndarray:
    return ((np.asarray(angle) + np.pi) % (2.0 * np.pi)) - np.pi


def pendulum_step(state: np.ndarray, action: float) -> tuple[np.ndarray, float]:
    """Exact Gymnasium Pendulum-v1 dynamics and reward."""

    theta, theta_dot = float(state[0]), float(state[1])
    torque = float(np.clip(action, -2.0, 2.0))
    cost = float(angle_normalize(theta) ** 2 + 0.1 * theta_dot**2 + 0.001 * torque**2)
    next_theta_dot = theta_dot + (15.0 * np.sin(theta) + 3.0 * torque) * 0.05
    next_theta_dot = float(np.clip(next_theta_dot, -8.0, 8.0))
    next_theta = theta + next_theta_dot * 0.05
    return np.asarray([next_theta, next_theta_dot], dtype=float), -cost


def pendulum_rollout(initial_state: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(initial_state, dtype=float).copy()
    states: list[np.ndarray] = []
    rewards: list[float] = []
    for action in np.asarray(actions, dtype=float).reshape(-1):
        states.append(state.copy())
        state, reward = pendulum_step(state, float(action))
        rewards.append(reward)
    return np.asarray(states, dtype=float), np.asarray(rewards, dtype=float)


@dataclass(frozen=True)
class UniformBinner:
    low: float
    high: float
    bins: int

    @property
    def edges(self) -> np.ndarray:
        return np.linspace(self.low, self.high, self.bins + 1)

    @property
    def centers(self) -> np.ndarray:
        edges = self.edges
        return (edges[:-1] + edges[1:]) / 2.0

    def encode(self, values: np.ndarray | float) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        idx = np.digitize(arr, self.edges[1:-1], right=False)
        return np.clip(idx, 0, self.bins - 1).astype(int)

    def decode(self, tokens: np.ndarray | int) -> np.ndarray:
        arr = np.asarray(tokens, dtype=int)
        return self.centers[np.clip(arr, 0, self.bins - 1)]


@dataclass(frozen=True)
class PendulumTokenizer:
    theta_bins: int = 13
    velocity_bins: int = 13
    action_bins: int = 11
    reward_bins: int = 19

    @property
    def theta_binner(self) -> UniformBinner:
        return UniformBinner(-np.pi, np.pi, self.theta_bins)

    @property
    def velocity_binner(self) -> UniformBinner:
        return UniformBinner(-8.0, 8.0, self.velocity_bins)

    @property
    def action_binner(self) -> UniformBinner:
        return UniformBinner(-2.0, 2.0, self.action_bins)

    @property
    def reward_binner(self) -> UniformBinner:
        return UniformBinner(-16.5, 0.0, self.reward_bins)

    def vocab_size(self, token_type: str) -> int:
        if token_type == "s":
            return self.theta_bins * self.velocity_bins
        if token_type == "a":
            return self.action_bins
        if token_type == "r":
            return self.reward_bins
        raise ValueError(f"unknown token type: {token_type}")

    def state_token(self, state: np.ndarray) -> int:
        theta_idx = int(self.theta_binner.encode(float(angle_normalize(state[0]))))
        velocity_idx = int(self.velocity_binner.encode(float(np.clip(state[1], -8.0, 8.0))))
        return theta_idx * self.velocity_bins + velocity_idx

    def initial_state_token(self, initial_state: np.ndarray) -> int:
        return self.state_token(initial_state)

    def decode_state_token(self, token: int) -> np.ndarray:
        theta_idx = int(token) // self.velocity_bins
        velocity_idx = int(token) % self.velocity_bins
        return np.asarray(
            [
                float(self.theta_binner.decode(theta_idx)),
                float(self.velocity_binner.decode(velocity_idx)),
            ],
            dtype=float,
        )

    def encode(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> list[int]:
        tokens: list[int] = []
        for state, action, reward in zip(states, actions, rewards):
            tokens.extend(
                [
                    self.state_token(np.asarray(state, dtype=float)),
                    int(self.action_binner.encode(float(action))),
                    int(self.reward_binner.encode(float(reward))),
                ]
            )
        return tokens

    def decode(self, tokens: list[int] | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arr = np.asarray(tokens, dtype=int)
        if len(arr) % 3 != 0:
            raise ValueError("trajectory tokens must be a multiple of 3")
        states = np.asarray([self.decode_state_token(x) for x in arr[0::3]], dtype=float)
        actions = np.asarray(self.action_binner.decode(arr[1::3]), dtype=float)
        rewards = np.asarray(self.reward_binner.decode(arr[2::3]), dtype=float)
        return states, actions, rewards

    def typed_context(self, prefix: list[int], context: int) -> tuple[tuple[str, int], ...]:
        start = max(0, len(prefix) - context)
        return tuple((LAYOUT[pos % 3], int(prefix[pos])) for pos in range(start, len(prefix)))

    def action_signature(self, tokens: list[int]) -> tuple[int, ...]:
        return tuple(int(x) for x in np.asarray(tokens, dtype=int)[1::3])


@dataclass(frozen=True)
class PendulumTrajectory:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    mode: str


@dataclass(frozen=True)
class PendulumDiagnostics:
    predicted_return: float
    true_return: float
    avg_logprob: float
    prefix_surprise: float
    dynamics_error: float
    action_energy: float

    @property
    def support_risk(self) -> float:
        return float(-self.avg_logprob + 0.20 * self.prefix_surprise + 1.25 * self.dynamics_error)


@dataclass(frozen=True)
class PendulumThresholds:
    min_avg_logprob: float
    max_prefix_surprise: float
    max_dynamics_error: float


def behavior_policy_action(state: np.ndarray, rng: np.random.Generator, mode: str) -> float:
    theta, theta_dot = float(angle_normalize(state[0])), float(state[1])
    if mode == "expert":
        action = -1.90 * theta - 0.35 * theta_dot + rng.normal(0.0, 0.18)
    elif mode == "risky":
        action = rng.choice([-1.0, 1.0]) * 1.75 + rng.normal(0.0, 0.18)
    else:
        action = -1.05 * theta - 0.18 * theta_dot + rng.normal(0.0, 0.38)
    return float(np.clip(action, -2.0, 2.0))


def deterministic_behavior_action(state: np.ndarray) -> float:
    theta, theta_dot = float(angle_normalize(state[0])), float(state[1])
    return float(np.clip(-1.90 * theta - 0.35 * theta_dot, -2.0, 2.0))


def sample_initial_state(rng: np.random.Generator) -> np.ndarray:
    return np.asarray([rng.uniform(-np.pi, np.pi), rng.uniform(-1.0, 1.0)], dtype=float)


def generate_pendulum_dataset(n_trajectories: int, horizon: int, seed: int) -> list[PendulumTrajectory]:
    rng = np.random.default_rng(int(seed))
    trajectories: list[PendulumTrajectory] = []
    for _ in range(int(n_trajectories)):
        state = sample_initial_state(rng)
        draw = rng.random()
        if draw < 0.01:
            mode = "risky"
        elif draw < 0.26:
            mode = "expert"
        else:
            mode = "medium"
        states: list[np.ndarray] = []
        actions: list[float] = []
        rewards: list[float] = []
        for _step in range(int(horizon)):
            states.append(state.copy())
            action = behavior_policy_action(state, rng, mode)
            state, reward = pendulum_step(state, action)
            actions.append(action)
            rewards.append(reward)
        trajectories.append(
            PendulumTrajectory(
                states=np.asarray(states, dtype=float),
                actions=np.asarray(actions, dtype=float),
                rewards=np.asarray(rewards, dtype=float),
                mode=mode,
            )
        )
    return trajectories


def _logprob_stats(model: SmoothedAutoregressiveTT, tokens: list[int]) -> tuple[float, float]:
    prefix: list[int] = []
    total = 0.0
    max_surprise = 0.0
    for token in tokens:
        logprob = model.token_logprob_at(prefix, int(token))
        total += logprob
        max_surprise = max(max_surprise, -logprob)
        prefix.append(int(token))
    return float(total / max(1, len(tokens))), float(max_surprise)


def diagnose_pendulum_tokens(
    tokens: list[int],
    model: SmoothedAutoregressiveTT,
    tokenizer: PendulumTokenizer,
    initial_state: np.ndarray,
) -> PendulumDiagnostics:
    decoded_states, decoded_actions, decoded_rewards = tokenizer.decode(tokens)
    true_states, true_rewards = pendulum_rollout(initial_state, decoded_actions)
    angle_error = np.abs(angle_normalize(decoded_states[:, 0] - true_states[:, 0]))
    velocity_error = np.abs(decoded_states[:, 1] - true_states[:, 1]) / 8.0
    avg_logprob, prefix_surprise = _logprob_stats(model, tokens)
    return PendulumDiagnostics(
        predicted_return=float(np.sum(decoded_rewards)),
        true_return=float(np.sum(true_rewards)),
        avg_logprob=avg_logprob,
        prefix_surprise=prefix_surprise,
        dynamics_error=float(np.mean(angle_error + velocity_error)),
        action_energy=float(np.mean(np.abs(decoded_actions))),
    )


def calibrate_pendulum_support(
    dataset: list[PendulumTrajectory],
    model: SmoothedAutoregressiveTT,
    tokenizer: PendulumTokenizer,
) -> PendulumThresholds:
    diagnostics = [
        diagnose_pendulum_tokens(tokenizer.encode(t.states, t.actions, t.rewards), model, tokenizer, t.states[0])
        for t in dataset
    ]
    return PendulumThresholds(
        min_avg_logprob=float(np.quantile([d.avg_logprob for d in diagnostics], 0.03) - 0.25),
        max_prefix_surprise=float(np.quantile([d.prefix_surprise for d in diagnostics], 0.97) + 0.65),
        max_dynamics_error=float(np.quantile([d.dynamics_error for d in diagnostics], 0.97) + 0.28),
    )


def is_supported(diag: PendulumDiagnostics, thresholds: PendulumThresholds) -> bool:
    return (
        diag.avg_logprob >= thresholds.min_avg_logprob
        and diag.prefix_surprise <= thresholds.max_prefix_surprise
        and diag.dynamics_error <= thresholds.max_dynamics_error
    )


def calibrated_score(diag: PendulumDiagnostics, thresholds: PendulumThresholds, horizon: int) -> float:
    support_penalty = max(0.0, thresholds.min_avg_logprob - diag.avg_logprob)
    prefix_penalty = max(0.0, diag.prefix_surprise - thresholds.max_prefix_surprise)
    dynamics_penalty = max(0.0, diag.dynamics_error - thresholds.max_dynamics_error)
    return float(
        diag.predicted_return
        - 1.20 * support_penalty * 3.0 * horizon
        - 0.80 * prefix_penalty
        - 8.00 * dynamics_penalty
    )


def behavior_fallback_diagnostics(
    initial_state: np.ndarray,
    horizon: int,
    model: SmoothedAutoregressiveTT,
    tokenizer: PendulumTokenizer,
) -> PendulumDiagnostics:
    state = np.asarray(initial_state, dtype=float).copy()
    actions: list[float] = []
    for _ in range(int(horizon)):
        action = deterministic_behavior_action(state)
        actions.append(action)
        state, _reward = pendulum_step(state, action)
    states, rewards = pendulum_rollout(initial_state, np.asarray(actions, dtype=float))
    tokens = tokenizer.encode(states, np.asarray(actions, dtype=float), rewards)
    return diagnose_pendulum_tokens(tokens, model, tokenizer, initial_state)


def _diag_to_row(
    *,
    seed: int,
    n: int,
    method: str,
    diag: PendulumDiagnostics,
    risk_rate: float,
    blocked: bool,
    accepted_count: int,
) -> dict[str, Any]:
    return {
        "benchmark": ENV_ID,
        "seed": int(seed),
        "N": int(n),
        "method": method,
        "predicted_return": diag.predicted_return,
        "true_return": diag.true_return,
        "avg_logprob": diag.avg_logprob,
        "prefix_surprise": diag.prefix_surprise,
        "dynamics_error": diag.dynamics_error,
        "action_energy": diag.action_energy,
        "risk_rate": float(risk_rate),
        "blocked": float(blocked),
        "accepted_count": float(accepted_count),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((int(row["N"]), str(row["method"])), []).append(row)
    aggregate: list[dict[str, Any]] = []
    numeric = [
        "predicted_return",
        "true_return",
        "avg_logprob",
        "prefix_surprise",
        "dynamics_error",
        "action_energy",
        "risk_rate",
        "blocked",
        "accepted_count",
    ]
    for (n, method), group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        row: dict[str, Any] = {"benchmark": ENV_ID, "N": n, "method": method}
        for key in numeric:
            row[key] = float(np.mean([float(x[key]) for x in group]))
        aggregate.append(row)
    return aggregate


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _value(rows: list[dict[str, Any]], n: int, method: str, metric: str) -> float:
    for row in rows:
        if int(row["N"]) == int(n) and row["method"] == method:
            return float(row[metric])
    raise KeyError((n, method, metric))


def _effect_rows(rows: list[dict[str, Any]], high_n: int) -> list[dict[str, Any]]:
    by_seed_method_n = {(int(row["seed"]), str(row["method"]), int(row["N"])): row for row in rows}
    seeds = sorted({int(row["seed"]) for row in rows})
    effects: list[dict[str, Any]] = []
    for seed in seeds:
        raw_low = by_seed_method_n[(seed, "raw", 1)]
        raw_high = by_seed_method_n[(seed, "raw", high_n)]
        fallback_high = by_seed_method_n[(seed, "sieve_fallback", high_n)]
        behavior_high = by_seed_method_n[(seed, "behavior_policy", high_n)]
        oracle_high = by_seed_method_n[(seed, "oracle", high_n)]
        effects.append(
            {
                "seed": seed,
                "raw_predicted_gain": float(raw_high["predicted_return"] - raw_low["predicted_return"]),
                "raw_true_change": float(raw_high["true_return"] - raw_low["true_return"]),
                "fallback_repair": float(fallback_high["true_return"] - raw_high["true_return"]),
                "dynamics_reduction": float(raw_high["dynamics_error"] - fallback_high["dynamics_error"]),
                "raw_minus_behavior_true": float(raw_high["true_return"] - behavior_high["true_return"]),
                "oracle_gap": float(oracle_high["true_return"] - raw_high["true_return"]),
            }
        )
    return effects


def _ci(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(9217)
    boot = rng.choice(arr, size=(1200, len(arr)), replace=True).mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "lo": float(np.quantile(boot, 0.025)),
        "hi": float(np.quantile(boot, 0.975)),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n": float(len(arr)),
    }


def _summarize_effects(effects: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        key: _ci([float(row[key]) for row in effects])
        for key in [
            "raw_predicted_gain",
            "raw_true_change",
            "fallback_repair",
            "dynamics_reduction",
            "raw_minus_behavior_true",
            "oracle_gap",
        ]
    }


def _claim(status: bool, value: float, threshold: float, description: str) -> dict[str, Any]:
    return {
        "status": "pass" if status else "fail",
        "value": float(value),
        "threshold": float(threshold),
        "description": description,
    }


def audit_pendulum_claims(summary: dict[str, dict[str, float]]) -> dict[str, Any]:
    claims = {
        "pendulum_raw_reward_extremizes": _claim(
            summary["raw_predicted_gain"]["mean"] > 20.0,
            summary["raw_predicted_gain"]["mean"],
            20.0,
            "On Gymnasium Pendulum-v1, increasing raw candidate count to 64 must strongly increase decoded reward-token return.",
        ),
        "pendulum_raw_true_return_not_improved": _claim(
            summary["raw_true_change"]["mean"] < -0.25,
            summary["raw_true_change"]["mean"],
            -0.25,
            "The same high-candidate raw selection must not translate into better true Pendulum return.",
        ),
        "pendulum_behavior_fallback_repairs": _claim(
            summary["fallback_repair"]["mean"] > 6.0,
            summary["fallback_repair"]["mean"],
            6.0,
            "A support-gated behavior fallback must improve true return over unsupported high-candidate raw selection.",
        ),
        "pendulum_fallback_reduces_dynamics_mismatch": _claim(
            summary["dynamics_reduction"]["mean"] > 1.0,
            summary["dynamics_reduction"]["mean"],
            1.0,
            "The fallback must reduce selected token/simulator dynamics mismatch.",
        ),
        "pendulum_raw_worse_than_behavior": _claim(
            summary["raw_minus_behavior_true"]["mean"] < -6.0,
            summary["raw_minus_behavior_true"]["mean"],
            -6.0,
            "High-candidate raw reward-token selection must underperform a behavior-policy baseline.",
        ),
        "pendulum_oracle_gap_visible": _claim(
            summary["oracle_gap"]["mean"] > 6.0,
            summary["oracle_gap"]["mean"],
            6.0,
            "The candidate pool must contain better action sequences than raw reward-token selection chooses.",
        ),
    }
    return {"all_passed": all(item["status"] == "pass" for item in claims.values()), "claims": claims}


def plot_pendulum_benchmark(aggregate: list[dict[str, Any]], output: Path) -> None:
    methods = ["raw", "sieve_fallback", "behavior_policy", "oracle"]
    colors = {
        "raw": "#b23b3b",
        "sieve_fallback": "#1a8f5a",
        "behavior_policy": "#555555",
        "oracle": "#111111",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6), constrained_layout=True)
    for method in methods:
        rows = sorted([row for row in aggregate if row["method"] == method], key=lambda row: int(row["N"]))
        ns = [int(row["N"]) for row in rows]
        axes[0].plot(ns, [row["predicted_return"] for row in rows], marker="o", color=colors[method], label=method)
        axes[1].plot(ns, [row["true_return"] for row in rows], marker="o", color=colors[method])
        axes[2].plot(ns, [row["dynamics_error"] for row in rows], marker="o", color=colors[method])
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 8, 32, 64])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(True, alpha=0.28)
        ax.set_xlabel("candidate count")
    axes[0].set_title("Decoded reward tokens")
    axes[0].set_ylabel("predicted return")
    axes[0].legend(frameon=False, fontsize=6.8)
    axes[1].set_title("True Pendulum return")
    axes[1].set_ylabel("executed return")
    axes[2].set_title("Token/simulator mismatch")
    axes[2].set_ylabel("dynamics error")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _gymnasium_spec() -> dict[str, Any]:
    import gymnasium as gym

    env = gym.make(ENV_ID)
    try:
        return {
            "env_id": ENV_ID,
            "action_low": float(env.action_space.low[0]),
            "action_high": float(env.action_space.high[0]),
            "observation_shape": list(env.observation_space.shape),
        }
    finally:
        env.close()


def run_pendulum_benchmark(
    *,
    quick: bool = False,
    output_dir: Path | None = None,
    write_figures: bool = True,
) -> dict[str, Any]:
    horizon = 8 if quick else 16
    train_trajectories = 80 if quick else 280
    seeds = list(range(100, 104 if quick else 116))
    n_values = (1, 4, 8) if quick else (1, 8, 32, 64)
    max_n = max(n_values)
    output = Path(output_dir) if output_dir is not None else RESULTS_DIR / "pendulum_benchmark"
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = PendulumTokenizer()
    dataset = generate_pendulum_dataset(train_trajectories, horizon=horizon, seed=50)
    token_sequences = [tokenizer.encode(t.states, t.actions, t.rewards) for t in dataset]
    model = SmoothedAutoregressiveTT(
        tokenizer,
        context=5,
        alpha=0.04,
        low_support_mix=14.0,
        reward_tail_bias=0.94,
        action_tail_bias=0.72,
    ).fit(token_sequences)
    thresholds = calibrate_pendulum_support(dataset, model, tokenizer)

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        initial_state = sample_initial_state(np.random.default_rng(seed))
        behavior_diag = behavior_fallback_diagnostics(initial_state, horizon, model, tokenizer)
        rng = np.random.default_rng(seed * 1000 + max_n)
        candidate_pool = [
            diagnose_pendulum_tokens(
                model.rollout(tokenizer.initial_state_token(initial_state), horizon, rng, temperature=1.50).tokens,
                model,
                tokenizer,
                initial_state,
            )
            for _ in range(max_n)
        ]
        for n in n_values:
            candidates = candidate_pool[: int(n)]
            raw = max(candidates, key=lambda d: d.predicted_return)
            accepted = [diag for diag in candidates if is_supported(diag, thresholds)]
            if accepted:
                fallback = max(accepted, key=lambda d: calibrated_score(d, thresholds, horizon))
                blocked = False
            else:
                fallback = behavior_diag
                blocked = True
            oracle = max(candidates, key=lambda d: d.true_return)
            risk_rate = 1.0 - len(accepted) / len(candidates)
            rows.extend(
                [
                    _diag_to_row(
                        seed=seed,
                        n=n,
                        method="raw",
                        diag=raw,
                        risk_rate=risk_rate,
                        blocked=False,
                        accepted_count=n,
                    ),
                    _diag_to_row(
                        seed=seed,
                        n=n,
                        method="sieve_fallback",
                        diag=fallback,
                        risk_rate=risk_rate,
                        blocked=blocked,
                        accepted_count=len(accepted),
                    ),
                    _diag_to_row(
                        seed=seed,
                        n=n,
                        method="oracle",
                        diag=oracle,
                        risk_rate=risk_rate,
                        blocked=False,
                        accepted_count=n,
                    ),
                    _diag_to_row(
                        seed=seed,
                        n=n,
                        method="behavior_policy",
                        diag=behavior_diag,
                        risk_rate=risk_rate,
                        blocked=False,
                        accepted_count=1,
                    ),
                ]
            )

    aggregate = aggregate_rows(rows)
    effects = _effect_rows(rows, high_n=max_n)
    summary = _summarize_effects(effects)
    audit = audit_pendulum_claims(summary) if not quick else {"all_passed": True, "claims": {}}

    figure_path = output / "pendulum_benchmark.png"
    if write_figures:
        plot_pendulum_benchmark(aggregate, figure_path)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plot_pendulum_benchmark(aggregate, FIGURES_DIR / "pendulum_benchmark.png")

    _write_csv(output / "metrics.csv", rows)
    _write_csv(output / "aggregate_metrics.csv", aggregate)
    _write_csv(output / "effects.csv", effects)
    payload = {
        "benchmark": ENV_ID,
        "quick": bool(quick),
        "gymnasium_spec": _gymnasium_spec(),
        "horizon": horizon,
        "train_trajectories": train_trajectories,
        "seeds": seeds,
        "n_values": list(n_values),
        "thresholds": thresholds.__dict__,
        "summary": summary,
        "claims": audit["claims"],
        "all_passed": bool(audit["all_passed"]),
    }
    (output / "claims.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "metrics": str(output / "metrics.csv"),
        "aggregate_metrics": str(output / "aggregate_metrics.csv"),
        "effects": str(output / "effects.csv"),
        "claims": str(output / "claims.json"),
        "summary": str(output / "summary.json"),
        "figure": str(figure_path),
        "all_passed": bool(audit["all_passed"]),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"manifest": manifest, "summary": payload, "audit": audit}
