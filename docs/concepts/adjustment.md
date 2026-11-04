# Adjustment covariates

Biomarkers are usually analysed adjusted for clinical covariates: age,
performance status, extent of resection, MGMT and IDH status, other markers.
Two separate questions matter for the sample size: what adjustment **costs**
when the covariates are correlated with the biomarker, and what it **gains**
when they are prognostic.

## The (1 − R²) cost of collinearity

The closed form inflates the required events by \(1/(1 - R^2)\), where \(R^2\)
is the multiple correlation of the biomarker with the adjustment set (Hsieh,
Bloch & Larsen, 1998; extended to the Cox model by Hsieh & Lavori, 2000):

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

## The gain from adjusting for prognostic covariates

In linear regression, leaving out a prognostic covariate that is independent
of \(X\) only adds residual variance. In the Cox model it also **attenuates**
\(\hat\beta_X\) towards zero, because the hazard ratio is non-collapsible
(Gail et al., 1984; Struthers & Kalbfleisch, 1986; Schmoor & Schumacher, 1997).
So pre-specified adjustment for strongly prognostic covariates, when they are
only weakly correlated with the biomarker, is a power **gain**, not a cost
(Hernández et al., 2006). In the revised report's design (N = 1200, five
independent prognostic covariates), unadjusted power fell from 0.82 to 0.28 as
the covariates' effects grew, while adjusted power stayed at about 0.83.

```bash
survpower --hr 1.10 --n 1200 --sensitivity omitted
```

The closed form refers to the **conditional** HR of the adjusted model. Plan
and analyse with the same model.

## What the simulations showed (and you can re-check)

| Check | Finding | Re-run with |
|---|---|---|
| Does the number of covariates matter at fixed R²? | No detectable extra penalty down to ~10 events per parameter (resolution ±0.05 in power at 300 replicates) | `--sensitivity collinearity` |
| Very few events per variable (EPV < 5) | β̂ biased away from the null, model SE too small, Wald type-I error inflated (≈ 0.09–0.14 at EPV 5–2.5) | `--sensitivity epv` |
| Covariates that are themselves prognostic (adjusted analysis) | No detectable extra cost beyond (1 − R²) | `--sensitivity heterogeneous` |
| Prognostic covariates left out | Attenuated HR and a large loss of power | `--sensitivity omitted` |

!!! warning "At low EPV, nominal power is mostly size distortion"
    Below EPV ≈ 5 the inverse-information SE understates the true sampling SD
    (by ~26% at EPV = 2.5), so the Wald test rejects a true null 9–14% of the
    time. Nominal power then *looks* stable while the size-adjusted power falls.
    The EPV table reports `type1_error`, `size_adj_power`, `bias_beta` (log-HR
    scale, with MCSE) and `sd_beta_hat` vs. `mean_se`. Judge the design by
    those, not by `power`. Remedies: Firth's penalized partial likelihood
    (Firth, 1993; Heinze & Schemper, 2001), or reduce the covariate set using
    prediction-model sample-size criteria (Riley et al., 2019). EPV alone is a
    crude criterion (van Smeden et al., 2016).

## Concordance and added value

The concordance modes are univariate. To show that a biomarker **adds** to a
clinical model, do not power a ΔC comparison: tests of ΔC between nested models
are poorly behaved (Pepe et al., 2013). Power the biomarker's **adjusted hazard
ratio** (this page), and report ΔC with a confidence interval as a secondary
result. Gönen–Heller's C handles the multivariable models directly:
`concordance_index(time, event, X, method="gonen-heller")` with the covariate
matrix `X` ([Concordance questions](concordance.md#gonenhellers-concordance-probability)).
