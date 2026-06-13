from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .environment import dynamics, simulate_actions
from .model import SmoothedAutoregressiveTT
from .tokenizer import TrajectoryTokenizer


@dataclass(frozen=True)
class PlanDiagnostics:
    predicted_return: float
    realized_return: float
    avg_logprob: float
    prefix_surprise: float
    dynamics_error: float
    action_energy: float
    mode_signature: tuple[int, ...]

    @property
    def support_risk(self) -> float:
        return float(-self.avg_logprob + 0.20 * self.prefix_surprise + 1.25 * self.dynamics_error)


@dataclass(frozen=True)
class SupportThresholds:
    min_avg_logprob: float
    max_prefix_surprise: float
    max_dynamics_error: float
    max_risk_rate: float = 0.38


def dynamics_consistency(tokens: list[int], tokenizer: TrajectoryTokenizer) -> float:
    decoded = tokenizer.decode(tokens)
    if len(decoded.actions) <= 1:
        return 0.0
    errors = []
    for t in range(len(decoded.actions) - 1):
        predicted_next = dynamics(float(decoded.states[t]), float(decoded.actions[t]))
        errors.append(abs(predicted_next - float(decoded.states[t + 1])))
    return float(np.mean(errors)) if errors else 0.0


def diagnose_plan(
    tokens: list[int],
    model: SmoothedAutoregressiveTT,
    tokenizer: TrajectoryTokenizer,
    initial_state: float,
) -> PlanDiagnostics:
    decoded = tokenizer.decode(tokens)
    real = simulate_actions(initial_state, decoded.actions)
    nlls = model.prefix_nlls(tokens)
    return PlanDiagnostics(
        predicted_return=float(np.sum(decoded.rewards)),
        realized_return=real.return_,
        avg_logprob=float(model.sequence_logprob(tokens) / max(1, len(tokens))),
        prefix_surprise=float(np.max(nlls)),
        dynamics_error=dynamics_consistency(tokens, tokenizer),
        action_energy=float(np.mean(np.abs(decoded.actions))),
        mode_signature=tokenizer.action_signature(tokens),
    )


def calibrate_support(
    token_sequences: list[list[int]],
    trajectories_initial_state: list[float],
    model: SmoothedAutoregressiveTT,
    tokenizer: TrajectoryTokenizer,
) -> SupportThresholds:
    diagnostics = [
        diagnose_plan(seq, model, tokenizer, initial)
        for seq, initial in zip(token_sequences, trajectories_initial_state)
    ]
    avg_logprob = np.asarray([d.avg_logprob for d in diagnostics], dtype=float)
    prefix = np.asarray([d.prefix_surprise for d in diagnostics], dtype=float)
    dyn = np.asarray([d.dynamics_error for d in diagnostics], dtype=float)
    return SupportThresholds(
        min_avg_logprob=float(np.quantile(avg_logprob, 0.03) - 0.04),
        max_prefix_surprise=float(np.quantile(prefix, 0.97) + 0.20),
        max_dynamics_error=float(np.quantile(dyn, 0.97) + 0.06),
    )


def is_supported(diag: PlanDiagnostics, thresholds: SupportThresholds) -> bool:
    return (
        diag.avg_logprob >= thresholds.min_avg_logprob
        and diag.prefix_surprise <= thresholds.max_prefix_surprise
        and diag.dynamics_error <= thresholds.max_dynamics_error
    )
