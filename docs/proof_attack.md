# Proof Attack

## Proposition Under Test

Let candidates \(X_1,\dots,X_K\) be i.i.d. from a finite distribution over outcomes \(i\) with probability \(p_i\), proxy score \(s_i\), and realized utility \(u_i\). The selected candidate is the one with maximum score; ties are broken uniformly among tied samples.

For outcome \(i\), define

\[
p_{=i}=\Pr(s(X)=s_i), \quad p_{<i}=\Pr(s(X)<s_i).
\]

Then the exact probability that outcome \(i\) is selected is

\[
\Pr(\hat X_K=i)
=
K p_i \sum_{m=0}^{K-1}
\binom{K-1}{m}
\frac{p_{=i}^{m} p_{<i}^{K-1-m}}{m+1}.
\]

The selected expected utility is \(\sum_i u_i \Pr(\hat X_K=i)\).

## Proof Sketch

Distinguish one sampled copy of outcome \(i\). It is selected if no other sample has strictly larger score. If exactly \(m\) of the other \(K-1\) samples have equal score, then the distinguished copy wins uniform tie-breaking with probability \(1/(m+1)\). The other samples must have either equal or lower score, giving the binomial term. Multiply by \(Kp_i\) because any of the \(K\) positions could be the distinguished copy.

## Attacks And Results

1. **Ties between different outcomes.** The formula uses score-level probability \(p_{=i}\), so ties across outcomes are included.
2. **Duplicate copies of the same outcome.** Duplicate copies are part of the equal-score group; the distinguished-copy argument still holds.
3. **Continuous scores.** The proposition is finite-distribution only. Continuous analogues are outside this repo's claim.
4. **Non-i.i.d. beam candidates.** The proposition is not a deterministic-beam theorem. The manuscript states this boundary.
5. **Control optimality.** The identity is about score-selected utility under a candidate distribution, not optimal control.
6. **Numerical edge cases.** The exact law is validated against Monte Carlo and includes harmful and benign high-score tails.

## Surviving Corollary

If the highest proxy-score tail has lower realized utility than the lower-score mass and its selection probability increases with candidate count, then expected selected utility can decrease. This is conditional, not universal.

## Trajectory Transformer Mechanism Statement

For an autoregressive model over interleaved state/action/reward tokens, reward-token selection pressure can select low-support prefixes or dynamics-inconsistent token strings when high reward-token probability is available through smoothing, rare contexts, or model error. This is a mechanism statement, not a guarantee for all Trajectory Transformers.
