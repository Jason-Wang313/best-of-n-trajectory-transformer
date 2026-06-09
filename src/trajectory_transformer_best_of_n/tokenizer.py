from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LAYOUT = ("s", "a", "r")


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
class DecodedTokens:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray


@dataclass(frozen=True)
class TrajectoryTokenizer:
    state_bins: int = 21
    action_bins: int = 11
    reward_bins: int = 23
    state_low: float = -2.4
    state_high: float = 2.4
    action_low: float = -2.2
    action_high: float = 2.2
    reward_low: float = -2.0
    reward_high: float = 2.4

    @property
    def state_binner(self) -> UniformBinner:
        return UniformBinner(self.state_low, self.state_high, self.state_bins)

    @property
    def action_binner(self) -> UniformBinner:
        return UniformBinner(self.action_low, self.action_high, self.action_bins)

    @property
    def reward_binner(self) -> UniformBinner:
        return UniformBinner(self.reward_low, self.reward_high, self.reward_bins)

    def vocab_size(self, token_type: str) -> int:
        if token_type == "s":
            return self.state_bins
        if token_type == "a":
            return self.action_bins
        if token_type == "r":
            return self.reward_bins
        raise ValueError(f"unknown token type: {token_type}")

    def encode(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> list[int]:
        horizon = len(actions)
        if len(rewards) != horizon:
            raise ValueError("actions and rewards must have the same length")
        if len(states) < horizon:
            raise ValueError("states must contain at least one state per action")

        s_tokens = self.state_binner.encode(states[:horizon])
        a_tokens = self.action_binner.encode(actions)
        r_tokens = self.reward_binner.encode(rewards)
        tokens: list[int] = []
        for t in range(horizon):
            tokens.extend([int(s_tokens[t]), int(a_tokens[t]), int(r_tokens[t])])
        return tokens

    def decode(self, tokens: list[int] | np.ndarray) -> DecodedTokens:
        arr = np.asarray(tokens, dtype=int)
        if len(arr) % 3 != 0:
            raise ValueError("trajectory tokens must be a multiple of 3")
        states = self.state_binner.decode(arr[0::3])
        actions = self.action_binner.decode(arr[1::3])
        rewards = self.reward_binner.decode(arr[2::3])
        return DecodedTokens(states=states, actions=actions, rewards=rewards)

    def typed_context(self, prefix: list[int], context: int) -> tuple[tuple[str, int], ...]:
        start = max(0, len(prefix) - context)
        typed: list[tuple[str, int]] = []
        for pos in range(start, len(prefix)):
            typed.append((LAYOUT[pos % 3], int(prefix[pos])))
        return tuple(typed)

    def initial_state_token(self, initial_state: float) -> int:
        return int(self.state_binner.encode(np.asarray([initial_state]))[0])

    def action_signature(self, tokens: list[int]) -> tuple[int, ...]:
        return tuple(int(x) for x in np.asarray(tokens, dtype=int)[1::3])
