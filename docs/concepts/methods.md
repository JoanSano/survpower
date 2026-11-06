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

## Performance measures and Monte Carlo SEs

Every simulation output carries `n_sims` and, next to each Monte Carlo
estimate `x`, a column `x_mcse` (helpers in `survpower.results`). The
formulas follow Morris, White & Crowther (2019, Table 6), over
\(n_{\text{sims}}\) replicates with estimates \(\hat\beta_i\):

| Measure | Estimate | MCSE |
|---|---|---|
| power / type-I error / coverage | \(\hat p\) | \(\sqrt{\hat p(1-\hat p)/n_{\text{sims}}}\) |
| bias (`bias_beta`) | \(\overline{\hat\beta} - \beta\) | \(\mathrm{SD}(\hat\beta)/\sqrt{n_{\text{sims}}}\) |
| empirical SE (`emp_se`) | \(\mathrm{SD}(\hat\beta)\) | \(\mathrm{SD}(\hat\beta)/\sqrt{2(n_{\text{sims}}-1)}\) |
| mean HR (`mean_HR_hat`) | \(e^{\overline{\hat\beta}}\) | \(e^{\overline{\hat\beta}}\,\mathrm{SD}(\hat\beta)/\sqrt{n_{\text{sims}}}\) (delta method) |

Coverage is that of the Wald interval \(\hat\beta \pm z_{1-\alpha/2}\,\widehat{\mathrm{SE}}\),
and `mean_model_se` / `rel_err_model_se` compare the average model SE with the
empirical SE.

- **Bias is on the log-HR scale.** `mean_HR_hat` is kept for continuity but is
  inflated by about \(\exp(\mathrm{Var}(\hat\beta)/2)\) even when \(\hat\beta\)
  is unbiased.
- **Failed fits.** A replicate whose fit fails (fewer than two events, or a
  non-positive information) is counted in `n_failed`. It counts as "not
  rejected" in power and is left out of the other moments.
- **Power at 0 or 1.** The binomial MCSE is then exactly 0, so the planning
  tables also report a Wilson 95% interval.

## Engines

| Engine | Used for | Notes |
|---|---|---|
| `fast` (default) | univariate fits | vectorized Newton–Raphson on the partial likelihood; agrees with lifelines to ~1e-6, 10–50× faster |
| `lifelines` | adjusted models, PH tests | `CoxPHFitter`, with a small ridge fallback if a fit is near-singular |

Both engines give identical decisions on the same seeds (tested).

## Solving for N

Power is approximately probit-linear in \(\sqrt N\):
\(\Phi^{-1}(\text{Power}) \approx a\sqrt N + b\).

1. Anchor on the closed-form \(N_0\); simulate power at \(N_0 \times \{0.6, 0.8, 1, 1.2, 1.4\}\).
2. Fit \(a, b\) by weighted least squares, with a continuity-corrected probit
   and binomial delta-method weights.
3. Invert for the target power, giving \(N^*\), and quantify its Monte Carlo
   uncertainty: resample the binomial counts at each grid point (parametric
   bootstrap, 4000 resamples), refit and invert. The 95% percentile interval is
   reported as `n_calibrated_ci_low` / `n_calibrated_ci_high`.
4. Confirm \(N^*\) with an independent, larger simulation (`--n-sims`).

!!! warning "Report the interval, and check it excludes the closed form"
    A calibrated N whose interval contains the closed-form N has not been
    calibrated. The correction it is meant to detect (~2%) is smaller than the
    noise. For HR 1.10 (closed-form N = 1081), the interval is about 130 patients
    wide at 300 replicates per point, 65 at 1000 and 47 at 2000, against a
    correction of about 20 patients. The default here is 1000 per point; the summary
    sentence says when the interval contains the closed form. Raise
    `--n-sims-grid` for the final numbers.

**Small-N fallback.** When \(N^* < 100\) the probit fit is not precise enough.
Each patient moves power by a visible amount there, and for C = 0.70 and a target
of 80% the fitted N* confirmed as low as 77% in 15 runs. In
that case `survpower` scans nine integers in \([0.7N^*, 1.5N^*]\) at full
precision and keeps the smallest N that reaches the target. The `calibration`
column says which method was used.

## Solving for the detectable effect

Same idea at fixed N: grid over \(\log \mathrm{HR}\) around the closed-form
value, fit \(\Phi^{-1}(\text{Power}) = a \log \mathrm{HR} + b\), invert (with a
bootstrap interval, `hr_ci_low` / `hr_ci_high`), confirm.

## Reproducibility

Every function takes a seed and draws in a fixed order, so the same seed gives
the same result within one installed version. Following Burton et al. (2006) and Morris et al. (2019),
report the seed, the number of replicates and the Monte Carlo SE with any
simulated quantity.
