# Proof Attack

## Theorem Under Test

Let candidates \(X_1,\dots,X_N\) be i.i.d. from a finite distribution over outcomes \(i\) with probability \(p_i\), proxy score \(s_i\), and realized utility \(u_i\). The selected candidate is the one with maximum score; ties are broken uniformly among tied samples.

For outcome \(i\), define

\[
p_{=i}=\Pr(s(X)=s_i), \quad p_{<i}=\Pr(s(X)<s_i).
\]

Then the exact probability that outcome \(i\) is selected is

\[
\Pr(\hat X_N=i)
=
N p_i \sum_{m=0}^{N-1}
\binom{N-1}{m}
\frac{p_{=i}^{m} p_{<i}^{N-1-m}}{m+1}.
\]

The selected expected utility is \(\sum_i u_i \Pr(\hat X_N=i)\).

## Proof Sketch

Distinguish one sampled copy of outcome \(i\). It is selected if no other sample has strictly larger score. If exactly \(m\) of the other \(N-1\) samples have equal score, then the distinguished copy wins uniform tie-breaking with probability \(1/(m+1)\). The other samples must have either equal or lower score, giving the binomial term. Multiply by \(N p_i\) because any of the \(N\) positions could be the distinguished copy.

## Attacks And Results

1. **Ties between different outcomes.**
   The formula uses score-level probability \(p_{=i}\), not outcome identity, so ties across outcomes are included.

2. **Duplicate copies of the same outcome.**
   Duplicate copies are just part of the equal-score group. The distinguished-copy argument still gives uniform tie-breaking over samples.

3. **Continuous scores.**
   The theorem is stated for finite distributions. Continuous-score analogues follow by replacing score masses with CDF terms, but this repo does not rely on that stronger claim.

4. **Non-i.i.d. beam candidates.**
   Beam search candidates are dependent. The theorem is used as a finite-N Best-of-N law, not as a proof for every beam implementation. The paper marks this boundary.

5. **Control optimality.**
   The theorem says nothing about optimal control. It only characterizes score-selected utility under a candidate distribution.

6. **Numerical edge cases.**
   Unit tests compare the exact law to Monte Carlo and include a tail-misalignment distribution where increasing N lowers expected selected utility.

## Surviving Corollary

If the highest proxy-score tail has lower realized utility than the lower-score mass and its selection probability increases with N, then expected selected utility can decrease with N. This is a conditional corollary, not a universal monotonicity result.

## Trajectory Transformer Proposition

For an autoregressive model over interleaved \((s,a,r)\) tokens, reward-token selection pressure can increase the probability of selecting low-support prefixes whenever high reward-token probability is available through smoothing, backoff, or model error in rare contexts. This proposition depends on a support-mismatch condition and is documented as a mechanism statement, not a guarantee for all TTs.
