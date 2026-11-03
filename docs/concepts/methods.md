# How the simulations work

## One replicate

1. Draw \(X_i \sim \mathcal N(0,1)\) and \(T_i \sim \text{Exp}(\lambda_0 e^{\beta X_i})\)
   (plus adjustment covariates, or a non-PH generator, where applicable).
2. Draw \(C_i \sim \text{Exp}(\lambda_c)\), with \(\lambda_c\) calibrated once
   by bisection on a 500,000-subject probe so that \(P(C < T)\) equals the target.
3. Fit the planned Cox model and record the Wald test on \(X\), two-sided
   (or one-sided in the direction of the effect).

Power is the rejection rate over replicates. Its Monte Carlo SE is at most
\(\sqrt{0.25/n_{\text{sims}}}\) (0.016 at 1000 replicates). Tables show a Wilson 95% interval.

## Engines

| Engine | Used for | Notes |
|---|---|---|
| `fast` (default) | univariate fits | vectorized Newton–Raphson on the partial likelihood; agrees with lifelines to ~1e-6, 10–50× faster |
| `lifelines` | adjusted models, PH tests, report reproduction | `CoxPHFitter`, with a small ridge fallback if a fit is near-singular |

Both engines give identical decisions on the same seeds (tested).

## Solving for N

Power is approximately probit-linear in \(\sqrt N\):
\(\Phi^{-1}(\text{Power}) \approx a\sqrt N + b\).

1. Anchor on the closed-form \(N_0\); simulate power at \(N_0 \times \{0.6, 0.8, 1, 1.2, 1.4\}\).
2. Fit \(a, b\) by weighted least squares, with a continuity-corrected probit
   and binomial delta-method weights.
3. Invert for the target power, giving \(N^*\).
4. Confirm \(N^*\) with an independent, larger simulation (`--n-sims`).

**Small-N fallback.** When \(N^* < 100\) the probit form is unreliable (the
report's Appendix B: fitted N* = 19 confirmed at only 75.5% for C = 0.70). In
that case `survpower` scans nine integers in \([0.7N^*, 1.5N^*]\) at full
precision and keeps the smallest N that reaches the target. The `calibration`
column says which method was used.

## Solving for the detectable effect

Same idea at fixed N: grid over \(\log \mathrm{HR}\) around the closed-form
value, fit \(\Phi^{-1}(\text{Power}) = a \log \mathrm{HR} + b\), invert, confirm.

## Reproducibility

Every function takes a seed and draws in a fixed order. `survpower report`
uses the lifelines engine and the original seeds, and reproduces the report's
tables exactly. The regression tests pin this against values produced by the
original script.
