from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log

import numpy as np

from .tokenizer import LAYOUT, TrajectoryTokenizer


@dataclass(frozen=True)
class Rollout:
    tokens: list[int]
    token_logprob: float


class SmoothedAutoregressiveTT:
    """A compact autoregressive surrogate with Trajectory Transformer token layout.

    The model is deliberately small: it is an n-gram style token model over
    (state, action, reward) triples. This keeps the experiments reproducible
    while preserving the planning interface that matters for the paper.
    """

    def __init__(
        self,
        tokenizer: TrajectoryTokenizer,
        context: int = 5,
        alpha: float = 0.04,
        low_support_mix: float = 10.0,
        reward_tail_bias: float = 0.42,
        action_tail_bias: float = 0.18,
    ) -> None:
        self.tokenizer = tokenizer
        self.context = context
        self.alpha = alpha
        self.low_support_mix = low_support_mix
        self.reward_tail_bias = reward_tail_bias
        self.action_tail_bias = action_tail_bias
        self.context_counts: dict[tuple[str, tuple[tuple[str, int], ...]], Counter[int]] = defaultdict(Counter)
        self.global_counts: dict[str, Counter[int]] = {typ: Counter() for typ in LAYOUT}
        self._prob_cache: dict[tuple[str, tuple[tuple[str, int], ...]], np.ndarray] = {}
        self.fitted = False

    def fit(self, token_sequences: list[list[int]]) -> "SmoothedAutoregressiveTT":
        for seq in token_sequences:
            prefix: list[int] = []
            for pos, token in enumerate(seq):
                typ = LAYOUT[pos % 3]
                self.global_counts[typ][int(token)] += 1
                for k in range(self.context + 1):
                    ctx = self.tokenizer.typed_context(prefix, k)
                    self.context_counts[(typ, ctx)][int(token)] += 1
                prefix.append(int(token))
        self._prob_cache.clear()
        self.fitted = True
        return self

    def _base_probs(self, typ: str) -> np.ndarray:
        vocab = self.tokenizer.vocab_size(typ)
        counts = self.global_counts[typ]
        values = np.asarray([counts.get(i, 0.0) for i in range(vocab)], dtype=float)
        probs = values + self.alpha
        return probs / probs.sum()

    def next_probs(self, prefix: list[int]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("fit must be called before next_probs")
        typ = LAYOUT[len(prefix) % 3]
        ctx = self.tokenizer.typed_context(prefix, self.context)
        cache_key = (typ, ctx)
        cached = self._prob_cache.get(cache_key)
        if cached is not None:
            return cached
        vocab = self.tokenizer.vocab_size(typ)
        base = self._base_probs(typ)
        counts = self.context_counts.get((typ, ctx), Counter())
        count_sum = float(sum(counts.values()))
        values = np.asarray([counts.get(i, 0.0) for i in range(vocab)], dtype=float)
        if count_sum <= 0.0:
            self._prob_cache[cache_key] = base
            return base
        contextual = (values + self.alpha) / (count_sum + self.alpha * vocab)
        # Rare contexts back off strongly to the global reward/action prior.
        lam = count_sum / (count_sum + self.low_support_mix)
        probs = lam * contextual + (1.0 - lam) * base
        if typ == "r":
            centers = self.tokenizer.reward_binner.centers
            tail = np.exp((centers - np.max(centers)) / 0.34)
            tail /= tail.sum()
            weight = self.reward_tail_bias * (1.0 - lam)
            probs = (1.0 - weight) * probs + weight * tail
        elif typ == "a":
            centers = self.tokenizer.action_binner.centers
            tail = np.abs(centers) ** 2.4 + 0.02
            tail /= tail.sum()
            weight = self.action_tail_bias * (1.0 - lam)
            probs = (1.0 - weight) * probs + weight * tail
        probs = probs / probs.sum()
        self._prob_cache[cache_key] = probs
        return probs

    def sample_next(
        self,
        prefix: list[int],
        rng: np.random.Generator,
        temperature: float = 1.0,
    ) -> tuple[int, float]:
        probs = self.next_probs(prefix)
        if temperature != 1.0:
            logits = np.log(np.maximum(probs, 1e-15)) / temperature
            logits -= np.max(logits)
            probs = np.exp(logits)
            probs /= probs.sum()
        token = int(rng.choice(np.arange(len(probs)), p=probs))
        return token, float(log(max(probs[token], 1e-15)))

    def rollout(
        self,
        initial_state_token: int,
        horizon: int,
        rng: np.random.Generator,
        temperature: float = 1.0,
    ) -> Rollout:
        target_len = 3 * horizon
        tokens = [int(initial_state_token)]
        logprob = self.token_logprob_at([], int(initial_state_token))
        while len(tokens) < target_len:
            token, lp = self.sample_next(tokens, rng, temperature=temperature)
            tokens.append(token)
            logprob += lp
        return Rollout(tokens=tokens, token_logprob=logprob)

    def token_logprob_at(self, prefix: list[int], token: int) -> float:
        probs = self.next_probs(prefix)
        return float(log(max(probs[int(token)], 1e-15)))

    def sequence_logprob(self, tokens: list[int]) -> float:
        prefix: list[int] = []
        total = 0.0
        for token in tokens:
            total += self.token_logprob_at(prefix, int(token))
            prefix.append(int(token))
        return float(total)

    def prefix_nlls(self, tokens: list[int]) -> np.ndarray:
        prefix: list[int] = []
        nlls: list[float] = []
        for token in tokens:
            nlls.append(-self.token_logprob_at(prefix, int(token)))
            prefix.append(int(token))
        return np.asarray(nlls, dtype=float)
