# Final Audit

1. **What is the discovered main thesis?**

   Score-only Trajectory Transformer-style planning can extremize decoded reward tokens while selecting trajectory-token strings whose prefixes are low-support or whose state tokens are dynamically inconsistent with their action tokens. The failure is visible only when the selected object is audited as a full state/action/reward token string.

2. **What is genuinely new?**

   The contribution is a Trajectory Transformer-specific audit. It does not merely say that proxy rewards can be overoptimized. It measures prefix surprise, average token support, candidate-set risk, and token/simulator dynamics consistency on interleaved trajectory-token strings.

3. **What proof survived adversarial checking?**

   The exact finite-candidate selected-utility identity for i.i.d. score-selected candidates with uniform tie-breaking survived proof attacks about duplicate outcomes, score ties, and finite support. The corollary remains conditional: more candidates can lower realized utility when the selected high-score tail has lower true utility.

4. **What is the strongest empirical result?**

   The primary four-control run shows the sharpest horizon-stress case: candidate count 64 raises predicted reward by 5.572 while realized return changes by -7.553. The expanded run extends candidate-count stress to 256 and shows raw predicted reward gain of 5.093 with realized return change of -2.532. The Pendulum-v1 tier shows raw decoded reward gain of 28.875 at 64 candidates while true return changes by -0.660 and remains 9.663 below the behavior baseline.

5. **What is the strongest repair result?**

   In the expanded horizon-stress suite at candidate count 256, the Support-Calibrated Plan Sieve improves realized return over raw score-only selection by 8.822. In Pendulum-v1, the support-gated behavior fallback repairs the high-candidate raw slice by 9.663 and reduces token/simulator mismatch by 1.491.

6. **What are the biggest weaknesses?**

   The main model is an inspectable autoregressive surrogate rather than a trained neural Transformer, and the repair thresholds are calibrated on offline behavior data. The paper now includes a standard Pendulum-v1 stress tier, but neural TT validation on richer benchmark datasets remains the main missing evidence.

7. **Is this submission-ready under the project standard?**

   Yes for a bounded mechanism paper: the manuscript is at least 25 pages, includes substantial new experiments, reports ablations, negative controls, and a standard benchmark stress tier, avoids generic duplicate framing, states limitations clearly, builds from source, and is covered by a claim audit.

8. **Where is the final PDF saved?**

   Repository copy: `paper/final/` as produced by `scripts/build_paper.ps1`.
