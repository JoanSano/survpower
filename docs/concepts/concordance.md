# Concordance questions

"We expect the biomarker to reach C ≈ 0.70" can mean three different things,
with sample sizes an order of magnitude apart. With 20% censoring and Harrell's C:

| Question | Command | N for C = 0.70 |
|---|---|---|
| **Association**: is the biomarker related to survival at all? | `--c-index 0.70` | ≈ 21 |
| **Precision**: estimate C to ±0.05 | `--c-index 0.70 --precision 0.05` | ≈ 150 |
| **Benchmark**: show C exceeds C₀ = 0.65 | `--c-index 0.70 --c0 0.65` | ≈ 230 |

State the question explicitly in the protocol. The generated summaries do.

## Association

The target C is converted to a per-SD HR ([Effect sizes](effect-sizes.md)) and
powered with the **Cox Wald test**. Under PH this is more efficient than a
test based on C: the concordance discards how far apart the biomarker values
are. At C = 0.60 the Cox test needs N ≈ 84; a test of C = 0.5 needs ≈ 107.

## Benchmark: H₀: C ≤ C₀

One-sided test on the estimated concordance, \(Z = (\hat C - C_0)/\widehat{\text{se}}\), with

\[
\text{Power}(N) \approx \Phi\!\left(\frac{(C_\infty - C_0)\sqrt N}{\sigma_C} - z_{1-\alpha}\right),
\qquad
N = \left\lceil \left(\frac{(z_{1-\alpha} + z_{1-\gamma})\,\sigma_C}{C_\infty - C_0}\right)^2 \right\rceil,
\]

where \(C_\infty\) is **what the estimator converges to under your design**, and
\(\sigma_C = \lim \sqrt{N\,\text{Var}(\hat C)}\). `survpower` computes both by
simulation, uses the closed form as an anchor, and then calibrates N against
simulation of the actual test.

The margin \(C - C_0\) drives everything: halving it roughly quadruples N.

## Harrell's or Uno's C?

| | Harrell's C (`--c-estimator harrell`) | Uno's C (`--c-estimator uno`) |
|---|---|---|
| Comparable pairs | shorter observed time is an event | same, re-weighted by \(1/\hat G(T_i)^2\) and truncated at τ |
| Converges to | \(C_H\), which depends on the censoring pattern and exceeds the population C | \(K_\tau\), the censoring-free concordance up to τ |
| Example, C = 0.70, 20% censoring | 0.7032 | 0.7009 (τ = 52 months) |
| Best when | light censoring, comparison with published Harrell values | heavier or covariate-related censoring, a fixed clinical horizon |

\(\hat G\) is the Kaplan–Meier estimate of the censoring distribution. With
Uno's C the horizon τ matters. By default it is the 90th percentile of observed
follow-up, which keeps the weights bounded. Set it with `--c-tau` to a
clinically meaningful horizon (e.g. 24 months) and report it.

!!! note "The assumed C matters more than the estimator's offset"
    With 20% censoring Harrell's C converges to 0.7032, not 0.70. `survpower`
    plans on that limit (column `c_limit`). Because N ∝ (C − C₀)⁻², even this
    0.003 offset moves N by 13% at a 0.05 margin. Uncertainty in the assumed C
    itself is far larger, though, so run several plausible values
    (`--c-index 0.68 0.70 0.72`) rather than trusting one.

!!! warning "A fixed benchmark is a simplification"
    The benchmark test treats C₀ as a known constant. If C₀ is the concordance
    of a competing model (e.g. a clinical score) estimated **on the same
    patients**, the right analysis is a paired comparison of two correlated C
    estimates (DeLong et al., 1988; Kang et al., 2015). The correlation usually
    reduces the variance of the difference, so the fixed-benchmark N can err in
    either direction. Simulate the paired test under a joint model for both
    predictors instead. `survpower` does not implement this yet.

## Standard errors

Both estimators use the Hájek projection of the (weighted) ratio of
U-statistics (Hoeffding, 1948; see also Pencina & D'Agostino, 2004), giving a delta-method variance
\(\widehat{\text{Var}}(\hat C) = B^{-2} \sum_i (a_i - \hat C\, b_i)^2\). Here
\(a_i\) and \(b_i\) are the concordant and comparable pair weights involving
subject \(i\), and \(B\) is the total comparable weight. For Uno's C the IPCW
weights are treated as known. The test suite checks both SEs against the
simulated SD of \(\hat C\).

!!! danger "Small samples"
    Below ~60 patients the U-statistic SE is anti-conservative (type I error up
    to 0.10–0.12 at N = 10–15). `survpower` warns when a benchmark design falls
    there. For association at large C, where N is a few dozen, rely on the Cox
    test.

## Computing C on your data

```python
from survpower import concordance_index
c, se = concordance_index(time, event, risk_score, method="uno", tau=24)
```

`risk_score` is oriented so that higher means shorter survival (e.g. the
biomarker itself, or a Cox linear predictor). Pair matrices are \(n \times n\):
fine up to a few thousand patients.
