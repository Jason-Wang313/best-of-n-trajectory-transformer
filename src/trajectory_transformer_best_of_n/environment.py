from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContinuousTrajectory:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    mode: str = "unknown"

    @property
    def return_(self) -> float:
        return float(np.sum(self.rewards))


def dynamics(state: float, action: float) -> float:
    next_state = state + 0.55 * action - 0.12 * action**3 + 0.04 * np.sin(2.2 * state)
    return float(np.clip(next_state, -2.4, 2.4))


def reward(state: float, action: float, next_state: float) -> float:
    target_bonus = 0.52 * np.exp(-((next_state - 1.05) / 0.36) ** 2)
    shaped = 1.48 - 1.20 * (next_state - 1.05) ** 2 - 0.14 * action**2
    action_cliff = 2.40 * max(0.0, abs(action) - 1.05) ** 2
    state_cliff = 2.10 * max(0.0, abs(next_state) - 1.48) ** 2
    return float(np.clip(shaped + target_bonus - action_cliff - state_cliff, -2.0, 2.4))


def behavior_action(rng: np.random.Generator, state: float, mode: str) -> float:
    if mode == "high":
        action = 0.68 * (1.05 - state) + rng.normal(0.0, 0.08)
        return float(np.clip(action, -0.15, 1.05))
    if mode == "risky":
        action = 1.48 + rng.normal(0.0, 0.12)
        return float(np.clip(action, 1.10, 1.95))
    action = 0.31 * (1.00 - state) + rng.normal(0.0, 0.18)
    return float(np.clip(action, -0.70, 0.85))


def rollout_behavior(
    rng: np.random.Generator,
    horizon: int,
    high_support: float,
    risky_support: float = 0.0,
) -> ContinuousTrajectory:
    draw = rng.random()
    if draw < risky_support:
        mode = "risky"
    elif draw < risky_support + high_support:
        mode = "high"
    else:
        mode = "safe"

    states = [float(rng.normal(0.0, 0.10))]
    actions: list[float] = []
    rewards: list[float] = []
    for _ in range(horizon):
        a_t = behavior_action(rng, states[-1], mode)
        s_next = dynamics(states[-1], a_t)
        r_t = reward(states[-1], a_t, s_next)
        states.append(s_next)
        actions.append(a_t)
        rewards.append(r_t)
    return ContinuousTrajectory(
        states=np.asarray(states, dtype=float),
        actions=np.asarray(actions, dtype=float),
        rewards=np.asarray(rewards, dtype=float),
        mode=mode,
    )


def generate_offline_dataset(
    n_trajectories: int,
    horizon: int,
    seed: int,
    high_support: float = 0.08,
    risky_support: float = 0.0,
) -> list[ContinuousTrajectory]:
    rng = np.random.default_rng(seed)
    return [
        rollout_behavior(rng, horizon, high_support=high_support, risky_support=risky_support)
        for _ in range(n_trajectories)
    ]


def simulate_actions(initial_state: float, actions: np.ndarray) -> ContinuousTrajectory:
    states = [float(initial_state)]
    rewards: list[float] = []
    for action in np.asarray(actions, dtype=float):
        s_next = dynamics(states[-1], float(action))
        rewards.append(reward(states[-1], float(action), s_next))
        states.append(s_next)
    return ContinuousTrajectory(
        states=np.asarray(states, dtype=float),
        actions=np.asarray(actions, dtype=float),
        rewards=np.asarray(rewards, dtype=float),
        mode="simulated",
    )
