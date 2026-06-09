# Reviewer Attacks

## Attack 1: This Is Just Reward Hacking

Response: The repo explicitly connects to Best-of-N reward overoptimization, but the diagnostic is not generic. It tests whether the selected object is a coherent Trajectory Transformer token string: support log-likelihood, maximum prefix surprise, and state/action/reward dynamics consistency are measured before the action sequence is executed.

## Attack 2: The Model Is Too Small

Response: Correct. The current model is a discretized autoregressive surrogate. It is useful for a first mechanism test because every probability and token diagnostic is inspectable. The next required experiment is a neural TT on D4RL-style environments.

## Attack 3: The Repair Is Likelihood Regularization

Response: Plain likelihood regularization is part of the repair, but not the whole repair. The Plan Sieve also checks prefix-level surprise and a trajectory-token consistency condition: decoded next-state tokens must agree with a simulator rollout from decoded state/action tokens.

## Attack 4: Beam Search Is Not I.I.D. Best-of-N

Response: The exact theorem is stated for i.i.d. Best-of-N. The experiments use sampling as a beam-style stress test because it gives clean finite-N control. The paper does not claim the law exactly describes all deterministic beam implementations.

## Attack 5: Synthetic Results Are Not Enough

Response: Agreed. The repo's final audit labels this as paper-worthy v1 for mechanism and diagnostic only. Benchmark validation is a necessary next step.

## Attack 6: Oracle Selection Is Unfair

Response: The oracle selector is included only as an upper-bound diagnostic. It is never presented as deployable.

## Attack 7: Thresholds Might Be Tuned To The Toy Environment

Response: Thresholds are calibrated from offline training trajectories by quantiles. The method still needs cross-domain validation; the claim is that the signals are principled enough to test, not that the exact numbers transfer.
