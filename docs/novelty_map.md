# Novelty Map

## Closest Prior Ideas

- Trajectory Transformers model offline RL trajectories as token sequences and use beam-style planning over state, action, and reward tokens.
- Reward-model overoptimization studies show that selecting from larger candidate pools can overoptimize proxy reward.
- Offline RL support-constrained methods penalize unsupported actions or model rollouts.

## What Would Be Incremental

It would be incremental to say only that larger candidate pools need regularization, that low-likelihood trajectories should be penalized, or that proxy reward can be overoptimized. Those ideas are already present in reward overoptimization and offline RL support-mismatch work.

## What Is Distinct Here

The paper audits reward-token extremization inside a Trajectory Transformer-style token planner. The selected object is not just an action sequence. It is an interleaved state/action/reward string whose prefixes have probabilities and whose next-state tokens can be checked against decoded actions.

The distinctive diagnostics are:

- decoded reward-token return versus realized simulator return,
- average token log-likelihood,
- maximum prefix surprise,
- token/simulator dynamics error,
- candidate-set risk rate,
- blocked and accepted-count behavior under a support gate,
- tokenizer, context, temperature, horizon, and tail-bias sensitivity.

## Strongest Contribution

The strongest contribution is a bounded mechanism paper with a full stress suite and a standard Pendulum-v1 benchmark stress tier. It defines a trajectory-token audit, proves and validates an exact finite-candidate selected-utility identity, implements a support-calibrated plan sieve, and reports both repairs and conservative losses.

## Remaining Gap

The main remaining gap is neural Trajectory Transformer validation on richer offline RL benchmarks. The current repo is submission-ready as a mechanism study with a standard-environment stress test, not as a benchmark leaderboard paper.
