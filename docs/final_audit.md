# Final Audit

1. **What is the discovered main thesis?**

   Score-only candidate selection with a Trajectory Transformer-style autoregressive model can extremize reward tokens faster than it improves actual plans. In controlled stress settings, the selected sequence's predicted reward rises with candidate count while support likelihood and dynamics consistency deteriorate, so simulator return can fall.

2. **What is genuinely new?**

   The contribution is not generic reranking. It is a Trajectory Transformer-specific diagnostic that treats the full interleaved state/action/reward token string as the object under selection, measures prefix surprise and token-level dynamics inconsistency, and uses those signals to gate large candidate-count search.

3. **What proposition/proof survived adversarial checking?**

   The exact finite-candidate selected-utility law for score-selected i.i.d. candidates with uniform tie-breaking survived the proof attack. The corollary is deliberately conditional: increasing the candidate count worsens realized utility when higher proxy-score tails carry lower conditional real utility.

4. **What is the strongest empirical result?**

   In the `horizon_stress` control, score-only selection changed predicted return by 5.572 from candidate count 1 to 64 while realized return changed by -7.553. This is synthetic evidence for reward-token extremization, not a benchmark claim.

5. **What is the strongest repair result?**

   The strongest repair result is in `horizon_stress`: at candidate count 64, the Support-Calibrated Plan Sieve improved realized return over score-only selection by 5.455 or blocked unsafe candidate sets when support diagnostics crossed calibration thresholds.

6. **What are the biggest weaknesses?**

   The experiments are synthetic, the model is an n-gram surrogate rather than a trained neural Transformer, and the repair thresholds are calibrated on generated offline data. Benchmark validation on D4RL-style environments and a neural TT implementation is the main missing evidence.

7. **Is this paper-worthy v1, needs stronger experiments, needs benchmark validation, or requires redesign?**

   Paper-worthy v1 for the mechanism and diagnostic, but it needs stronger experiments and benchmark validation before the empirical claim should be treated as field-level evidence.

8. **Where exactly is the final PDF saved?**

   Repository copy: `paper\final\iclr_submission.pdf`
   Downloads copy: `C:\Users\wangz\Downloads\iclr_submission_trajectory_token_sieve.pdf`
