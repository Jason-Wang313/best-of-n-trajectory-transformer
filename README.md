# Reward-Token Tail Audits for Trajectory Transformer Planning

This repository contains the source, experiments, figures, audits, and final build for a submission-ready mechanism paper about Trajectory Transformer planning.

The paper asks:

> When a planner searches over full state/action/reward trajectory-token strings, does a larger candidate set find better plans, or can it select better-looking reward-token tails whose realized simulator utility is worse?

The implemented answer is a controlled yes. In the expanded horizon-stress suite, raw score-only selection from 256 candidates raises decoded reward-token return by 5.093 relative to one candidate while lowering realized simulator return by 2.532. The Support-Calibrated Plan Sieve repairs that high-candidate tail by 8.822 realized-return units. The larger four-control run contains a sharper horizon-stress case at 64 candidates: predicted reward rises by 5.572 while realized return falls by 7.553. The v4 benchmark tier adds Gymnasium Pendulum-v1: raw selection at 64 candidates raises decoded reward by 28.875 while true return changes by -0.660 and remains 9.663 below the behavior baseline; the support-gated fallback repairs that slice and reduces token/simulator mismatch by 1.491.

## What This Is

- A reproducible Trajectory Transformer-style audit over interleaved `(state, action, reward)` tokens.
- A full paper with candidate-count stress, horizon sweeps, context-length ablations, tokenizer-resolution stress, sampling-temperature stress, tail-bias stress, sieve ablations, finite-candidate theory, a standard Pendulum-v1 stress tier, negative controls, and a reviewer-facing claim boundary.
- An inspectable autoregressive surrogate, chosen so token likelihoods, prefix surprises, and token/simulator dynamics errors are measurable.
- A bounded mechanism study, not a benchmark leaderboard or neural Transformer deployment claim.

## Quick Start

```bash
python -m trajectory_token_sieve.experiments.run_smoke
python -m trajectory_token_sieve.experiments.run_all
python -m trajectory_token_sieve.experiments.run_expansion_suite
python -m trajectory_token_sieve.experiments.run_pendulum_benchmark
python -m trajectory_token_sieve.experiments.run_claim_audit
pytest
.\scripts\build_paper.ps1
```

## Main Outputs

- `results/summary.json`: primary four-control summary.
- `results/all_results.csv`: primary per-method aggregate results.
- `results/expansion/aggregate_metrics.csv`: expanded stress-suite metrics.
- `results/expansion/claims.json`: numeric audit of headline claims.
- `results/pendulum_benchmark/aggregate_metrics.csv`: standard Pendulum-v1 benchmark metrics.
- `results/pendulum_benchmark/claims.json`: benchmark claim gates.
- `results/claims_status.json`: repository-facing claim audit.
- `figures/`: generated figures used by the paper.
- `paper/main.tex`: paper source.
- `paper/final/`: repository-local final PDF artifact.

## Claim Boundary

The paper claims a controlled mechanism: score-only candidate selection in a Trajectory Transformer-style token planner can extremize reward tokens while selecting low-support or dynamically inconsistent strings. It includes a standard Pendulum-v1 stress test, but it does not claim leaderboard superiority, neural Trajectory Transformer validation, deployment readiness, or a safety guarantee.
