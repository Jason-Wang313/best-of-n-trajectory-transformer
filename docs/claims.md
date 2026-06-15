# Claims

## Supported By This Repository

- In a controlled Trajectory Transformer-style token planner, raw score-only candidate selection can increase decoded reward-token return while lowering realized simulator return.
- In the expanded horizon-stress suite, increasing raw candidate count from 1 to 256 raises decoded reward-token return by 5.093 and lowers realized simulator return by 2.532.
- At candidate count 256 in that suite, the Support-Calibrated Plan Sieve improves realized return over raw selection by 8.822.
- In the larger four-control run, the horizon-stress control raises predicted reward by 5.572 from candidate count 1 to 64 while realized return changes by -7.553.
- Prefix surprise and token/simulator dynamics error expose failures that are hidden if one only reads decoded reward-token return.
- Context length, tokenizer resolution, sampling temperature, and reward/action tail bias materially change high-candidate diagnostics.
- On Gymnasium Pendulum-v1, raw score-only selection at 64 candidates extremizes decoded reward tokens without improving true return, underperforms a behavior-policy baseline, and is repaired in that slice by a support-gated behavior fallback.
- The exact finite-candidate selected-utility identity matches Monte Carlo and distinguishes harmful high-score tails from benign high-score tails.

## Not Claimed

- No benchmark leaderboard claim.
- No neural Trajectory Transformer validation claim.
- No safety guarantee.
- No deployment claim for real robots, humans, or production control systems.
- No claim that every Trajectory Transformer or every decoding procedure has this failure.
- No claim that the n-gram surrogate is a drop-in replacement for a neural Transformer.
- No claim that the Support-Calibrated Plan Sieve is always realized-return optimal.

## Audit Rule

The claim audit scans the README, paper, final audit, and this file for forbidden overclaims and stale artifact references. It also checks for the expanded results, Pendulum benchmark outputs, claim JSON files, novelty map, proof audit, and final repository PDF.
