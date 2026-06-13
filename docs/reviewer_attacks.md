# Reviewer Attacks

## Attack 1: This Is Just Generic Reward Hacking

Response: The paper connects to reward overoptimization, but the diagnostic is specific to Trajectory Transformer planning. It audits the full interleaved state/action/reward token string and measures prefix surprise plus token/simulator dynamics consistency.

## Attack 2: The Model Is Too Small

Response: Correct. The paper is a controlled mechanism study using an inspectable autoregressive surrogate. It does not claim neural benchmark validation. The manuscript includes a neural validation protocol as future work.

## Attack 3: The Repair Is Likelihood Regularization

Response: Likelihood is only one component. The Plan Sieve separately checks prefix surprise and dynamics consistency. The ablation table reports that likelihood-only wins one realized-return slice while the full sieve better reduces token pathology.

## Attack 4: More Candidates Are Not Always Harmful

Response: Agreed. The paper reports benign and mixed cases, including the out-of-support control and benign finite-law tail. The claim is conditional tail risk, not universal anti-scaling.

## Attack 5: Beam Search Candidates Are Not I.I.D.

Response: The exact identity is explicitly stated for i.i.d. finite candidate sets. It is used as a selection-law sanity check, not as a theorem for deterministic beam search.

## Attack 6: The Sieve Loses In One Ablation

Response: The manuscript reports that loss. The repair claim is made where the audit passes: high-candidate horizon stress at 256 candidates. The loss is framed as a conservative-gate cost.

## Attack 7: Synthetic Results Are Not Enough

Response: Agreed. The paper claims a reproducible mechanism and audit protocol. Benchmark-scale neural TT validation remains the next required evidence layer.

## Attack 8: The Claim Thresholds Were Chosen After Seeing Results

Response: The claim audit records numeric thresholds and forces unsupported claims to be revised. One proposed temperature-risk claim failed and was replaced by a truthful realized-return sensitivity claim.

## Attack 9: Dynamics Error Uses Simulator Knowledge

Response: In this controlled setting, simulator access is used to evaluate and audit token/simulator mismatch. Deployment would need an environment checker or learned dynamics consistency model.

## Attack 10: The Paper Is Too Similar To A Generic Candidate-Selection Story

Response: The final title, diagnostics, experiments, and claims are all trajectory-token-specific: reward tokens, prefix surprise, state/action/reward consistency, tokenizer resolution, and TT-style decoding are central throughout.
