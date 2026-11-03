# Python API

Everything the CLI does is available from Python. The most-used names are
re-exported from the top-level package (`from survpower import ...`).

| Module | Contents |
|---|---|
| `survpower.design` | `StudyDesign`, the shared design assumptions |
| `survpower.planning` | the solve-for-X planning modes behind the CLI |
| `survpower.concordance` | C ↔ HR mapping, Harrell / Uno estimators with SEs, concordance power |
| `survpower.closed_form` | Schoenfeld / Hsieh–Lavori, Freedman and median-split formulas |
| `survpower.calibration` | simulation-calibrated N and detectable HR, with bootstrap MC intervals |
| `survpower.simulate` | Monte Carlo power engines (PH, dichotomized, adjusted, omitted covariates, non-PH) |
| `survpower.censoring` | censoring calibration and the accrual calendar |
| `survpower.sensitivity` | collinearity, EPV, prognostic / omitted covariate and non-PH studies |
| `survpower.fitting` | fast univariate Cox solver and Wald decisions |
| `survpower.dgp` | event-time generators |
| `survpower.baseline` | baseline hazard calibration |
| `survpower.report` | reproduction of the report and Appendix B |
| `survpower.plotting` | figures |

## Design

::: survpower.design.StudyDesign

## Planning modes

::: survpower.planning
    options:
      members: [run_power_mode, run_sample_size_mode, run_detectable_effect_mode,
                run_cindex_benchmark_mode, run_cindex_precision_mode]

## Concordance

::: survpower.concordance
    options:
      members: [c_of_beta, beta_of_c, hr_of_c, beta_of_c_approx, harrell_limit, uno_limit,
                default_uno_tau, concordance_limit, harrell_c_with_se, uno_c_with_se,
                concordance_index, censoring_survival_left, simulate_cindex_power, sigma_c,
                benchmark_power, benchmark_sample_size, precision_half_width,
                precision_sample_size]

## Closed-form formulas

::: survpower.closed_form

## Calibration

::: survpower.calibration

## Simulation engines

::: survpower.simulate
    options:
      members: [simulate_power, simulate_power_design, simulate_continuous_vs_dichotomized,
                simulate_power_with_adjustment, simulate_epv_diagnostic,
                simulate_power_with_heterogeneous_adjustment, simulate_adjusted_vs_unadjusted,
                simulate_waning_effect, simulate_crossing_hazards]

## Censoring

::: survpower.censoring

## Sensitivity analyses

::: survpower.sensitivity

## Fitting

::: survpower.fitting

## Data-generating processes

::: survpower.dgp

## Baseline hazard

::: survpower.baseline

## Report reproduction

::: survpower.report

## Plotting

::: survpower.plotting
