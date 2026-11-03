# Command-line reference

```text
survpower [options]                       plan a study
survpower report [--appendix-b] [...]     reproduce the report's tables and figures
```

## What gets solved for

`survpower` infers the question from the options you give:

| Options | Question | Output file |
|---|---|---|
| `--hr` or `--c-index`, **and** `--n` | power at that N | `power_analysis.csv` |
| `--hr` or `--c-index` only | required N for each `--power` | `sample_size.csv`, calibration plot |
| `--n` only | minimum detectable HR and C | `detectable_effect.csv` |
| `--c-index` + `--c0` (± `--n`) | show C > C₀: N, or power at N | `cindex_benchmark.csv` |
| `--c-index` + `--precision` (or `--n`) | CI half-width of the estimated C | `cindex_precision.csv` |

## Effect size and sample size

| Option | Default | Meaning |
|---|---|---|
| `--hr HR [HR ...]` | | hazard ratio(s) per SD of the biomarker (per raw unit with `--biomarker-sd`); < 1 for protective |
| `--c-index C [C ...]` | | target concordance(s) in (0.5, 1); exclusive with `--hr` |
| `--n N [N ...]` | | sample size(s), patients **enrolled** |
| `--power P [P ...]` | 0.80 | target power(s) when solving for N or the effect |
| `--alpha` | 0.05 | type I error |
| `--sides {1,2}` | 2 | one- or two-sided Wald test (benchmark tests are always one-sided) |
| `--biomarker-sd SD` | | SD of the raw biomarker; `--hr` is then per raw unit |

## Concordance questions

| Option | Default | Meaning |
|---|---|---|
| `--c0 C0 [...]` | | benchmark concordance(s): one-sided test of H₀: C ≤ C₀ |
| `--precision H [...]` | | target 95% CI half-width(s) for C |
| `--c-estimator {harrell,uno}` | harrell | estimator used for benchmark / precision |
| `--c-tau MONTHS` | 90th pct. of follow-up | truncation horizon for Uno's C |

## Survival, follow-up and attrition

| Option | Default | Meaning |
|---|---|---|
| `--median-os` | 15 | baseline median overall survival, months |
| `--censor-rate` | 0.20 | marginal censoring probability (alternative to the accrual design) |
| `--accrual-months`, `--followup-months` | | uniform accrual period and minimum follow-up after the last enrolment |
| `--dropout` | 0 | annual loss-to-follow-up fraction (with the accrual design) |
| `--non-evaluable` | 0 | fraction of enrolled patients with no usable biomarker value |

## Analysis model

| Option | Default | Meaning |
|---|---|---|
| `--n-covariates` | 0 | adjustment covariates in the Cox model |
| `--r2` | 0 | R² of the biomarker on the adjustment covariates |

Concordance questions (`--c0`, `--precision`) are univariate. To show that a
biomarker adds value to a clinical model, power its *adjusted* hazard ratio
instead (`--hr` with `--n-covariates` and `--r2`).

## Sensitivity analyses

Run at the given or solved (N, HR) for hazard-ratio questions.

| Option | Default | Meaning |
|---|---|---|
| `--sensitivity A [A ...]` | | any of `collinearity`, `epv`, `heterogeneous`, `omitted`, `waning`, `crossing`, `all` |
| `--tau` | 6 | change-point (months) for `waning` / `crossing` |
| `--waning-hr` | planning HR | HR before τ in the waning scenario |
| `--crossing-hr EARLY LATE` | 1/HR, HR | HRs before / after τ in the crossing scenario |

## Simulation and output

| Option | Default | Meaning |
|---|---|---|
| `--engine {fast,lifelines}` | fast | Cox fitter for univariate simulations |
| `--n-sims` | 1000 | replicates per power estimate / confirmation |
| `--n-sims-grid` | 1000 | replicates per calibration grid point |
| `--n-sims-sensitivity` | 300 | replicates per sensitivity scenario |
| `--quick` | | 200 / 100 / 100 replicates (for exploring) |
| `--seed` | 42 | random seed |
| `--outdir` | survpower_output | output directory |
| `--no-plots` | | skip PDF figures |

## `survpower report`

| Option | Default | Meaning |
|---|---|---|
| `--appendix-b` | | reproduce Appendix B (concordance) instead of the main report |
| `--n-sims` | 1000 | replicates per cell (Appendix B) |
| `--outdir` | survpower_report | output directory |
