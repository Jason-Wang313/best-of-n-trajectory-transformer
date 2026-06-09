# Best-of-N Trajectory Transformer Diagnostics

This repository is a paper-quality first pass at the controlled research question:

> Can Best-of-N or beam-style planning with a Trajectory Transformer push reward-token scores upward while selecting low-support, dynamically inconsistent trajectory-token strings whose realized simulator utility gets worse?

The implemented answer is a synthetic but runnable **yes**. The repo does not claim benchmark-scale offline RL performance. It isolates the mechanism in a discretized autoregressive Trajectory Transformer surrogate with the token layout

```text
(state_token, action_token, reward_token) repeated over the horizon
```

and separates internal reward-token score from realized return under a ground-truth simulator.

## Quick Start

```bash
python -m trajectory_transformer_best_of_n.experiments.run_smoke
python -m trajectory_transformer_best_of_n.experiments.run_all
python -m trajectory_transformer_best_of_n.experiments.run_claim_audit
pytest
```

## What Is Included

- A reproducible offline dataset with support-limited high-return corridors.
- A smoothed autoregressive trajectory-token model with log-likelihood, prefix surprise, rollout sampling, and reward-token scoring.
- Best-of-N planning for `N = {1, 2, 4, 8, 16, 32, 64}`.
- Metrics for predicted reward, realized return, support log-likelihood, dynamics inconsistency, prefix surprise, candidate diversity, and selected-mode collapse.
- Controls for in-support planning, out-of-support stress, an anti-aligned scorer, and horizon stress.
- A repair method, **Support-Calibrated Plan Sieve**, that filters or downweights candidates with low support, surprising prefixes, or inconsistent dynamics and reports when high-N search should be blocked.
- Exact finite-N Best-of-N selected-utility law with Monte Carlo validation.
- ICLR-style paper source and a compiled PDF under `paper/final/iclr_submission.pdf`.

## Main Outputs

After `run_all`, inspect:

- `results/all_results.csv`
- `results/summary.json`
- `results/claims_status.json`
- `figures/reward_extremization.png`
- `figures/repair_comparison.png`
- `figures/control_panels.png`
- `figures/exact_law_validation.png`
- `docs/final_audit.md`
- `paper/final/iclr_submission.pdf`

## Claim Boundary

This is a controlled synthetic v1. It is designed to make a Trajectory Transformer-specific failure mode measurable and repairable in a small setting. It does not establish top benchmark performance, safety guarantees, or benchmark validity.
