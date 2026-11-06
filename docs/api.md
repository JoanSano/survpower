# Python API

Everything the CLI does is available from Python. The most-used names are
re-exported from the top-level package (`from survpower import ...`).

| Module | Contents |
|---|---|
| `survpower.design` | `StudyDesign`, the shared design assumptions |
| `survpower.planning` | the solve-for-X planning modes behind the CLI |
| `survpower.concordance` | C ↔ HR mapping, Harrell / Uno / Gönen–Heller estimators with SEs, concordance power |
| `survpower.closed_form` | Schoenfeld / Hsieh–Lavori, Freedman and median-split formulas |
| `survpower.calibration` | simulation-calibrated N and detectable HR, with bootstrap MC intervals |
| `survpower.scenario` | composable simulation scenarios: covariate, event-time and censoring models plus analyses (provisional) |
| `survpower.simulate` | Monte Carlo power engines (PH, dichotomized, adjusted, omitted covariates, non-PH) |
| `survpower.results` | performance measures with Monte Carlo SEs (Morris et al. 2019) |
| `survpower.censoring` | censoring calibration and the accrual calendar |
| `survpower.sensitivity` | collinearity, EPV, prognostic / omitted covariate and non-PH studies |
| `survpower.fitting` | fast univariate Cox solver and Wald decisions |
| `survpower.dgp` | event-time generators |
| `survpower.baseline` | baseline hazard calibration |
| `survpower.plotting` | figures |

## Design

::: survpower.design.StudyDesign

## Planning modes

::: survpower.planning
    options:
      members: [sample_size, SampleSizeResult, run_power_mode, run_sample_size_mode,
                run_detectable_effect_mode,
                run_cindex_benchmark_mode, run_cindex_precision_mode]

## Concordance

::: survpower.concordance
    options:
      members: [c_of_beta, beta_of_c, hr_of_c, beta_of_c_approx, harrell_limit, uno_limit,
                default_uno_tau, concordance_limit, mapping_table, harrell_c_with_se, uno_c_with_se,
                gonen_heller_c_with_se, fit_cox_coefficients,
                concordance_index, censoring_survival_left, simulate_cindex_power, sigma_c,
                benchmark_power, benchmark_sample_size, precision_half_width,
                precision_sample_size]

## Closed-form formulas

::: survpower.closed_form

## Calibration

::: survpower.calibration

## Scenarios (provisional)

!!! warning "Provisional API"
    `survpower.scenario` may change before 1.0. `Scenario` and
    `simulate_scenario` (= `scenario.simulate`) are exported from `survpower`.

A `Scenario` composes one model per step of the Monte Carlo loop: covariates →
event times → censoring → analyses. Models are plain objects satisfying the
protocols below, so a new data-generating option is a new class, not a new
loop. `simulate_power`, `simulate_continuous_vs_dichotomized` and the unadjusted
planning modes run on it, through `scenario_for`.

```python
import numpy as np
from survpower.scenario import (Scenario, ExponentialPH, ExponentialCensoring,
                                CoxWald, LogRank, simulate)

sc = Scenario(event_times=ExponentialPH(log_hr=np.log(1.15)),
              censoring=ExponentialCensoring(target_rate=0.20),
              analyses=(CoxWald(), CoxWald(dichotomize=True), LogRank()))
simulate(sc, n=500, n_sims=1000, seed=1)  # one row per analysis, with MCSEs
```

The same seed gives the same result within one installed version.

::: survpower.scenario
    options:
      members: [Scenario, simulate, run_replicates, Replicates, Effect, scenario_for, Covariates,
                CovariateModel, EventTimeModel, CensoringModel, Analysis,
                StandardNormal, ExponentialPH, ExponentialCensoring, CoxWald, LogRank]

## Simulation engines

::: survpower.simulate
    options:
      members: [simulate_power, simulate_power_design, simulate_continuous_vs_dichotomized,
                simulate_power_with_adjustment, simulate_epv_diagnostic,
                simulate_power_with_heterogeneous_adjustment, simulate_adjusted_vs_unadjusted,
                simulate_waning_effect, simulate_crossing_hazards]

## Monte Carlo performance measures

::: survpower.results
    options:
      members: [Estimate, proportion_mcse, mean_mcse, sd_mcse, exp_mean_mcse,
                summarize_replicates]

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

## Plotting

::: survpower.plotting
