# Command-line reference

```text
survpower COMMAND [options]
survpower --version
```

The command is the question:

| Command | Required | Optional | Output file |
|---|---|---|---|
| `power` | `--hr` or `--c-index`; `--n` | adjustment, sensitivity | `power_analysis.csv` |
| `sample-size` | `--hr` or `--c-index` | `--power`, adjustment, sensitivity | `sample_size.csv`, calibration plot |
| `detectable-effect` | `--n` | `--power`, adjustment, sensitivity | `detectable_effect.csv` |
| `benchmark` | `--c-index`; `--c0` | `--n` or `--power` (exclusive), estimator | `cindex_benchmark.csv` |
| `precision` | `--c-index`; `--half-width` or `--n` (exclusive) | estimator | `cindex_precision.csv` |

`survpower COMMAND --help` lists a command's flags, with examples. A command offers a
flag only if its path reads that flag, so anything else exits with code 2 rather than
being ignored. Flags cannot be abbreviated.

## How the CLI is organized

Four independent axes, each with one home. New options follow the same rule.

| Axis | Examples | Where it goes |
|---|---|---|
| **The question** | power, sample size, detectable effect | A command |
| **The family of comparison** | continuous biomarker (default), categorical groups | A namespace: `survpower groups <question>` (planned) |
| **The assumptions** | event-time model, censoring, accrual, covariates | Shared flag groups on the commands that support them, never a command |
| **The effect scale** | `--hr`, `--c-index` | One mutually exclusive group |

## Effect and sample size

| Option | Commands | Default | Meaning |
|---|---|---|---|
| `--hr HR [HR ...]` | `power`, `sample-size` | | hazard ratio(s) per SD of the biomarker (per raw unit with `--biomarker-sd`); < 1 for protective |
| `--c-index C [C ...]` | `power`, `sample-size`, `benchmark`, `precision` | | target concordance(s) in (0.5, 1); exclusive with `--hr` |
| `--n N [N ...]` | `power`, `detectable-effect`, `benchmark`, `precision` | | sample size(s), patients **enrolled** |
| `--power P [P ...]` | `sample-size`, `detectable-effect`, `benchmark` | 0.80 | target power(s) |
| `--c0 C0 [...]` | `benchmark` | | benchmark concordance(s): one-sided test of H₀: C ≤ C₀ |
| `--half-width H [...]` | `precision` | | target 95% CI half-width(s) for C |

## Test

| Option | Commands | Default | Meaning |
|---|---|---|---|
| `--alpha` | all but `precision` | 0.05 | type I error |
| `--sides {1,2}` | `power`, `sample-size`, `detectable-effect` | 2 | one- or two-sided Wald test. `benchmark` is always one-sided |

## Population and censoring

On every command.

| Option | Default | Meaning |
|---|---|---|
| `--median-os` | 15 | baseline median overall survival, months |
| `--biomarker-sd SD` | | SD of the raw biomarker; `--hr` is then per raw unit |
| `--non-evaluable` | 0 | fraction of enrolled patients with no usable biomarker value |
| `--censor-rate` | 0.20 | marginal censoring probability (alternative to the accrual design) |
| `--accrual-months`, `--followup-months` | | uniform accrual period and minimum follow-up after the last enrolment |
| `--dropout` | 0 | annual loss-to-follow-up fraction (with the accrual design) |

## Event model

On every command. Not with `--sensitivity`, whose scenarios have a constant baseline.

| Option | Default | Meaning |
|---|---|---|
| `--baseline {exponential,weibull,gompertz,loglogistic}` | exponential | shape of the baseline hazard, calibrated so that the baseline median is `--median-os` |
| `--baseline-shape K` | | required with a non-exponential `--baseline`: `k` for `weibull` and `loglogistic`; for `gompertz`, `a` per month (the hazard is multiplied by e^(12a) each year; `a < 0` gives a cure fraction) |

A piecewise-exponential or user-supplied hazard is available from Python only; see
[the shape of the baseline hazard](concepts/censoring-and-accrual.md#shape-of-the-baseline-hazard).

## Adjustment

On `power`, `sample-size` and `detectable-effect`.

| Option | Default | Meaning |
|---|---|---|
| `--n-covariates` | 0 | adjustment covariates in the Cox model |
| `--r2` | 0 | R² of the biomarker on the adjustment covariates |

`benchmark` and `precision` are univariate. To show that a biomarker adds value to a
clinical model, power its *adjusted* hazard ratio instead (`--hr` with `--n-covariates`
and `--r2`).

## Sensitivity analyses

On `power`, `sample-size` and `detectable-effect`, run at the given or solved (N, HR).

| Option | Default | Meaning |
|---|---|---|
| `--sensitivity A [A ...]` | | any of `collinearity`, `epv`, `heterogeneous`, `omitted`, `waning`, `crossing`, `all` |
| `--tau` | 6 | change-point (months) for `waning` / `crossing` |
| `--waning-hr` | planning HR | HR before τ in the waning scenario |
| `--crossing-hr EARLY LATE` | 1/HR, HR | HRs before / after τ in the crossing scenario |
| `--n-sims-sensitivity` | 300 | replicates per scenario |

## Concordance estimator

On `benchmark` and `precision`.

| Option | Default | Meaning |
|---|---|---|
| `--c-estimator {harrell,uno,gonen-heller}` | harrell | estimator of C |
| `--c-tau MONTHS` | 90th pct. of follow-up | truncation horizon, with `--c-estimator uno` only |

## Simulation and output

| Option | Commands | Default | Meaning |
|---|---|---|---|
| `--n-sims` | all | 1000 | replicates per power estimate / confirmation |
| `--seed` | all | 42 | random seed |
| `--quick` | all | | 200 replicates, 100 per grid point or scenario (for exploring) |
| `--engine {fast,lifelines}` | `power`, `sample-size`, `detectable-effect` | fast | Cox fitter for univariate simulations |
| `--n-sims-grid` | `sample-size`, `detectable-effect`, `benchmark` without `--n` | 1000 | replicates per calibration grid point |
| `--outdir` | all | survpower_output | output directory |
| `--no-plots` | all | | skip PDF figures |

## Migrating from the flag-only syntax

The flag-only syntax was removed in 0.4.1. A first argument that starts with `--` (other
than `--help` and `--version`) now exits with code 2 and lists the commands. Rewrite old
calls as below. Flags the new command rejects were being ignored before, so drop them;
results do not change.

| Before | Now |
|---|---|
| `survpower --hr 1.15 --n 250` | `survpower power --hr 1.15 --n 250` |
| `survpower --hr 1.15` | `survpower sample-size --hr 1.15` |
| `survpower --n 250` | `survpower detectable-effect --n 250` |
| `survpower --c-index 0.7 --c0 0.65` | `survpower benchmark --c-index 0.7 --c0 0.65` |
| `survpower --c-index 0.7 --precision 0.05` | `survpower precision --c-index 0.7 --half-width 0.05` |
| `survpower --c-index 0.7 --precision 0.05 --n 200` | `survpower precision --c-index 0.7 --n 200` |
