from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np


@dataclass(frozen=True)
class OutcomeDistribution:
    probabilities: np.ndarray
    scores: np.ndarray
    utilities: np.ndarray

    def normalized(self) -> "OutcomeDistribution":
        probs = np.asarray(self.probabilities, dtype=float)
        probs = probs / probs.sum()
        return OutcomeDistribution(
            probabilities=probs,
            scores=np.asarray(self.scores, dtype=float),
            utilities=np.asarray(self.utilities, dtype=float),
        )


def selected_outcome_probabilities(dist: OutcomeDistribution, n: int) -> np.ndarray:
    """Exact finite-N law for score-selected candidates with uniform tie breaking.

    For outcome i, distinguish one sampled copy of i. It is selected when no
    other sample has strictly larger score; if m other samples tie its score,
    the distinguished copy wins the tie with probability 1/(m+1).
    """

    d = dist.normalized()
    out = np.zeros_like(d.probabilities, dtype=float)
    for i, (p_i, score_i) in enumerate(zip(d.probabilities, d.scores)):
        p_equal = float(np.sum(d.probabilities[d.scores == score_i]))
        p_lower = float(np.sum(d.probabilities[d.scores < score_i]))
        total = 0.0
        for m in range(n):
            total += comb(n - 1, m) * (p_equal**m) * (p_lower ** (n - 1 - m)) / (m + 1)
        out[i] = n * p_i * total
    return out / out.sum()


def expected_selected_utility(dist: OutcomeDistribution, n: int) -> float:
    probs = selected_outcome_probabilities(dist, n)
    return float(np.sum(probs * dist.utilities))


def selected_utility_law(dist: OutcomeDistribution, n: int) -> dict[float, float]:
    probs = selected_outcome_probabilities(dist, n)
    law: dict[float, float] = {}
    for utility, prob in zip(dist.utilities, probs):
        law[float(utility)] = law.get(float(utility), 0.0) + float(prob)
    return law


def monte_carlo_selected_utility(
    dist: OutcomeDistribution,
    n: int,
    trials: int,
    seed: int = 0,
) -> float:
    d = dist.normalized()
    rng = np.random.default_rng(seed)
    indices = np.arange(len(d.probabilities))
    draws = rng.choice(indices, size=(trials, n), p=d.probabilities)
    scores = d.scores[draws]
    max_scores = np.max(scores, axis=1, keepdims=True)
    tie_noise = rng.random(size=scores.shape)
    tie_noise = np.where(scores == max_scores, tie_noise, -1.0)
    winner_cols = np.argmax(tie_noise, axis=1)
    winners = draws[np.arange(trials), winner_cols]
    return float(np.mean(d.utilities[winners]))


def tail_misalignment_example() -> OutcomeDistribution:
    return OutcomeDistribution(
        probabilities=np.asarray([0.70, 0.24, 0.06]),
        scores=np.asarray([0.3, 0.8, 1.8]),
        utilities=np.asarray([0.4, 0.9, -0.6]),
    ).normalized()
