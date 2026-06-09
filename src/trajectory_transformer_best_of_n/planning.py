from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .diagnostics import PlanDiagnostics, SupportThresholds, diagnose_plan, is_supported
from .model import SmoothedAutoregressiveTT
from .tokenizer import TrajectoryTokenizer


@dataclass(frozen=True)
class PlanCandidate:
    tokens: list[int]
    diagnostics: PlanDiagnostics


@dataclass(frozen=True)
class Selection:
    candidate: PlanCandidate
    blocked: bool = False
    accepted_count: int = 0
    candidate_risk_rate: float = 0.0


def sample_candidates(
    model: SmoothedAutoregressiveTT,
    tokenizer: TrajectoryTokenizer,
    initial_state: float,
    horizon: int,
    n: int,
    rng: np.random.Generator,
    temperature: float = 1.0,
) -> list[PlanCandidate]:
    initial_token = tokenizer.initial_state_token(initial_state)
    candidates: list[PlanCandidate] = []
    for _ in range(n):
        rollout = model.rollout(initial_token, horizon, rng, temperature=temperature)
        diagnostics = diagnose_plan(rollout.tokens, model, tokenizer, initial_state)
        candidates.append(PlanCandidate(tokens=rollout.tokens, diagnostics=diagnostics))
    return candidates


def select_raw(candidates: list[PlanCandidate], score_mode: str = "predicted_reward") -> Selection:
    if not candidates:
        raise ValueError("cannot select from an empty candidate set")
    if score_mode == "anti_aligned":
        key = lambda c: (
            c.diagnostics.predicted_return
            + 0.75 * c.diagnostics.prefix_surprise
            + 2.25 * c.diagnostics.dynamics_error
            + 0.18 * c.diagnostics.action_energy
        )
    else:
        key = lambda c: c.diagnostics.predicted_return
    selected = max(candidates, key=key)
    return Selection(candidate=selected, accepted_count=len(candidates), candidate_risk_rate=0.0)


def select_oracle(candidates: list[PlanCandidate]) -> Selection:
    if not candidates:
        raise ValueError("cannot select from an empty candidate set")
    return Selection(candidate=max(candidates, key=lambda c: c.diagnostics.realized_return))


class SupportCalibratedPlanSieve:
    def __init__(
        self,
        thresholds: SupportThresholds,
        support_weight: float = 1.15,
        prefix_weight: float = 0.85,
        dynamics_weight: float = 2.8,
    ) -> None:
        self.thresholds = thresholds
        self.support_weight = support_weight
        self.prefix_weight = prefix_weight
        self.dynamics_weight = dynamics_weight

    def accepts(self, candidate: PlanCandidate) -> bool:
        return is_supported(candidate.diagnostics, self.thresholds)

    def calibrated_score(self, candidate: PlanCandidate) -> float:
        d = candidate.diagnostics
        support_penalty = max(0.0, self.thresholds.min_avg_logprob - d.avg_logprob)
        prefix_penalty = max(0.0, d.prefix_surprise - self.thresholds.max_prefix_surprise)
        dynamics_penalty = max(0.0, d.dynamics_error - self.thresholds.max_dynamics_error)
        return float(
            d.predicted_return
            - self.support_weight * support_penalty * len(candidate.tokens)
            - self.prefix_weight * prefix_penalty
            - self.dynamics_weight * dynamics_penalty * len(d.mode_signature)
        )

    def select(self, candidates: list[PlanCandidate]) -> Selection:
        if not candidates:
            raise ValueError("cannot select from an empty candidate set")
        accepted = [candidate for candidate in candidates if self.accepts(candidate)]
        risk_rate = 1.0 - len(accepted) / len(candidates)
        if accepted:
            selected = max(accepted, key=self.calibrated_score)
            return Selection(
                candidate=selected,
                blocked=False,
                accepted_count=len(accepted),
                candidate_risk_rate=float(risk_rate),
            )
        selected = min(candidates, key=lambda c: c.diagnostics.support_risk)
        return Selection(
            candidate=selected,
            blocked=True,
            accepted_count=0,
            candidate_risk_rate=float(risk_rate),
        )


def candidate_diversity(candidates: list[PlanCandidate]) -> float:
    if not candidates:
        return 0.0
    signatures = {c.diagnostics.mode_signature for c in candidates}
    return float(len(signatures) / len(candidates))


def adaptive_n_from_rows(rows: list[dict], thresholds: SupportThresholds) -> int | None:
    safe = [
        int(row["N"])
        for row in rows
        if row["method"] == "raw" and float(row["candidate_risk_rate"]) <= thresholds.max_risk_rate
    ]
    return max(safe) if safe else None
