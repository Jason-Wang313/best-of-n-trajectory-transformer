# Novelty Map

## Search Grounding

The closest primary sources found in the novelty pass are:

- Janner, Li, and Levine, **Offline Reinforcement Learning as One Big Sequence Modeling Problem**. This introduces Trajectory Transformer planning and explicitly repurposes beam search over trajectory tokens. Source: <https://arxiv.org/abs/2106.02039>
- The Trajectory Transformer project page describes reward-biased decoding and beam planning with the learned trajectory sequence model. Source: <https://trajectory-transformer.github.io/>
- Gao et al., **Scaling Laws for Reward Model Overoptimization**, studies proxy reward overoptimization under rejection-style candidate selection. Source: <https://arxiv.org/abs/2210.10760>
- The ICLR 2026 author guide points to the official 2026 LaTeX template used in this repo. Source: <https://iclr.cc/Conferences/2026/AuthorGuide>

## What Is Already Known

Trajectory Transformers model offline RL trajectories as token sequences and use beam-style decoding to search for high-reward continuations. Score-only candidate selection is also known to overoptimize imperfect proxy reward models: as candidate count grows, selected candidates can move into proxy-score tails where real utility drops.

Offline RL already has a broad support-mismatch literature. Behavior constraints, pessimism, model uncertainty, and rollout pruning all address versions of the idea that planning outside the data distribution can be harmful.

## What Would Be Incremental

It would be incremental to say only that "large candidate pools need regularization" or that "low-likelihood trajectories should be penalized." That story is already familiar from reward overoptimization and offline RL support constraints.

It would also be incremental to run generic reranking over complete candidate trajectories without using the Trajectory Transformer tokenization. Such a repair would not explain why interleaving state, action, and reward tokens creates a distinctive failure surface.

## What Looks Genuinely New Here

The sharp angle is **reward-token extremization inside an autoregressive trajectory-token model**. In a Trajectory Transformer, the score being optimized is not just an external reward model. Reward tokens are part of the same sequence as state and action tokens. Increasing candidate count can therefore select strings where:

- reward tokens become extreme,
- state/action prefixes have low autoregressive support,
- sampled next-state tokens disagree with the simulator implied by previous state/action tokens,
- and realized utility under the actual dynamics drops.

The proposed diagnostic is architecture-specific: measure the selected candidate as a full token string, not just as an action sequence. The repair, **Support-Calibrated Plan Sieve**, rejects or downweights candidates by trajectory log-likelihood, prefix surprise, and token-level dynamics inconsistency before reward-token score is allowed to decide.

## Reviewer Attack Surface

Reviewers will likely argue:

- The experiments are synthetic and use an n-gram surrogate, not a neural Transformer.
- The repair resembles likelihood regularization unless the paper clearly shows why prefix surprise and state/action/reward consistency matter.
- The proposition is a selection identity, not a full control result.
- The method needs D4RL-style benchmark validation and a trained TT reproduction.
- Beam search has implementation details, and i.i.d. sampled candidates are only one decoding abstraction.

## Most Worth Pursuing

The strongest first-pass contribution is a mechanism paper:

1. Validate the finite-candidate selection identity and the conditional tail-misalignment consequence.
2. Demonstrate reward-token extremization in a controlled Trajectory Transformer-style surrogate.
3. Show that token-string support diagnostics catch failures that raw predicted reward misses.
4. Present the Support-Calibrated Plan Sieve as a repair and adaptive candidate-count gate.
5. State clearly that neural TT benchmark validation is the next step.
