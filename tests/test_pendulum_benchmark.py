from pathlib import Path

import numpy as np

from trajectory_token_sieve.pendulum_benchmark import (
    PendulumTokenizer,
    generate_pendulum_dataset,
    pendulum_rollout,
    pendulum_step,
    run_pendulum_benchmark,
)


def test_pendulum_step_rewards_upright_more_than_fallen():
    upright_next, upright_reward = pendulum_step(np.asarray([0.0, 0.0]), 0.0)
    fallen_next, fallen_reward = pendulum_step(np.asarray([np.pi, 0.0]), 0.0)

    assert upright_reward > fallen_reward
    assert upright_next.shape == (2,)
    assert fallen_next.shape == (2,)


def test_pendulum_tokenizer_round_trip_shapes():
    tokenizer = PendulumTokenizer()
    traj = generate_pendulum_dataset(1, horizon=5, seed=7)[0]
    tokens = tokenizer.encode(traj.states, traj.actions, traj.rewards)
    states, actions, rewards = tokenizer.decode(tokens)

    assert len(tokens) == 15
    assert states.shape == (5, 2)
    assert actions.shape == (5,)
    assert rewards.shape == (5,)


def test_pendulum_rollout_and_quick_benchmark_write_outputs(tmp_path):
    states, rewards = pendulum_rollout(np.asarray([0.1, 0.0]), np.zeros(4))
    assert states.shape == (4, 2)
    assert rewards.shape == (4,)

    result = run_pendulum_benchmark(quick=True, output_dir=tmp_path / "pendulum", write_figures=False)
    manifest = result["manifest"]
    assert result["summary"]["benchmark"] == "Pendulum-v1"
    assert manifest["all_passed"]
    assert Path(manifest["metrics"]).exists()
    assert Path(manifest["aggregate_metrics"]).exists()
    assert Path(manifest["summary"]).exists()
