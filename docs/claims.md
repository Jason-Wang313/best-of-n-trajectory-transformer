# Claims

## Supported By This Repo

- In a synthetic Trajectory Transformer-style token model, increasing candidate count can increase selected reward-token return while lowering support likelihood and increasing dynamics inconsistency.
- The exact finite-candidate selection identity for score-selected i.i.d. candidates matches Monte Carlo in the included test.
- The Support-Calibrated Plan Sieve can improve realized utility or block unsafe large candidate-count sets in the controlled experiments.
- Prefix surprise and token-level dynamics consistency expose failures that are hidden if one only reads the predicted reward-token sum.

## Not Claimed

- No claim of top benchmark performance.
- No safety guarantee.
- No validation on real robots, human preference labels, or production control systems.
- No claim that every Trajectory Transformer or every beam search implementation has this failure.
- No claim that the n-gram surrogate is a drop-in replacement for a neural Transformer.

## Claim Audit Rule

The audit script scans the README, paper, final audit, and this file for forbidden overclaim phrases and checks required output presence. It writes `results/claims_status.json`.
