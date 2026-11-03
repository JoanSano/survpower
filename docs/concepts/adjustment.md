# Adjustment covariates

Biomarkers are usually analysed adjusted for clinical covariates (age,
performance status, extent of resection, other markers). Those covariates are
rarely independent of the biomarker.

## The (1 − R²) penalty

The closed form inflates the required events by \(1/(1 - R^2)\), where \(R^2\)
is the multiple correlation of the biomarker with the adjustment set:

```bash
survpower --hr 1.2 --n-covariates 5 --r2 0.3     # ~43% more events than unadjusted
```

`--r2` can be estimated from pilot or registry data **without knowing the
covariates' own effects**. It is the number to discuss in the sample-size
section.

In the simulation, \(p\) covariates are built from one shared factor,
\(Z_j = \rho X + \sqrt{1-\rho^2}\,W_j\), with \(\rho\) solved so that the
multiple \(R^2\) equals the target exactly:
\(R^2 = \rho^2 p / \{1 + \rho^2 (p-1)\}\).

## What the simulations showed (and you can re-check)

| Check | Finding | Re-run with |
|---|---|---|
| Does the number of covariates matter at fixed R²? | No detectable extra penalty down to ~10 events per parameter | `--sensitivity collinearity` |
| Very few events per variable (EPV < 10) | Power looks fine, but the HR is biased away from the null and its variance grows; Firth's penalized likelihood is the usual fix | `--sensitivity epv` |
| Covariates that are themselves prognostic | No robust extra cost beyond (1 − R²) for typical covariate HRs up to ~1.5 | `--sensitivity heterogeneous` |

!!! note "Power is not a diagnostic of model reliability"
    At low events-per-variable, power can stay flat or even rise while the
    point estimate becomes biased and unstable. Check `mean_HR_hat` and
    `sd_beta_hat` in the EPV table, not only `power`.

## Concordance and added value

The concordance modes are univariate. To show that a biomarker **adds** to a
clinical model, do not power a ΔC comparison: tests of ΔC between nested models
are badly calibrated. Power the biomarker's **adjusted hazard ratio** (this
page), and report ΔC with a confidence interval as a secondary result.
