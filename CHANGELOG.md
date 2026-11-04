# Changelog

## Unreleased

### Added
- `survpower.scenario` (provisional): a `Scenario` composes a covariate model, an
  event-time model, a censoring model and a tuple of analyses; `simulate(scenario,
  n, ...)` returns one row per analysis with the MCSE columns of
  `results.summarize_replicates`. Shipped models: `StandardNormal`, `ExponentialPH`,
  `ExponentialCensoring`, `CoxWald` (fast or lifelines, optionally on the median
  split) and `LogRank`. A model is any object with the protocol's methods, so new
  data-generating options need no new simulation loop. Exported as
  `survpower.Scenario` and `survpower.simulate_scenario` (#9).
- `censoring.calibrate_censoring_for(sampler, target, seed)`: censoring
  calibration for any event-time generator. The four existing calibrators delegate
  to it (results unchanged) (#9).

### Changed
- `simulate_power` and `simulate_continuous_vs_dichotomized` run on the scenario
  engine. Their signatures and results are unchanged (bit for bit, checked against
  0.2.1). The guarantee is now "the same seed gives the same result within one
  installed version"; later releases may change the draw order (#9).

### Removed
- `tests/test_regression.py`, the exact regression against the original report
  script. Install the `v0.2.0` tag to reproduce the report (#9).

## 0.2.1 — 2026-11-04

Milestone M0 (quick wins and Monte Carlo standard errors): #1, #2, #4, #6, #8.

### Added
- Gönen & Heller's (2005) concordance probability estimate (#6):
  `concordance.gonen_heller_c_with_se(X, beta, cov)` for a Cox model with any
  number of covariates, with the paper's SE (smoothed U-statistic variance plus
  the delta-method term for beta_hat). `concordance_index(time, event, X,
  method="gonen-heller")` accepts a covariate vector or matrix and fits the Cox
  model (`concordance.fit_cox_coefficients`). The benchmark and precision modes
  accept `--c-estimator gonen-heller` (biomarker-only model). It converges to the
  population C at any censoring and needs about 25% fewer patients than Harrell's
  C to show C > C0, but assumes the model is correct.
- Log-rank test for the median split (#34). `survpower.fitting.logrank_1d` is the Cox
  score test of beta = 0 (equal to lifelines' `logrank_test` without ties).
  `simulate_continuous_vs_dichotomized` reports `power_dich_logrank` with its MCSE on
  the same cohorts as the Wald test, and the power table gains
  `power_dichotomized_logrank_sim`: the simulated counterpart of the Schoenfeld closed
  form, which is the asymptotic log-rank power.
- Monte Carlo standard errors in every simulation output (#1). The new module
  `survpower.results` has `summarize_replicates` and MCSE helpers (Morris, White &
  Crowther 2019). Every `simulate_*` dict, `simulate_cindex_power` and every
  `sensitivity.run_*` table now carries `n_sims`, `n_failed` and a `*_mcse` column per
  estimate, adding bias, empirical SE and coverage where a true log-HR exists. The
  planning tables add `power_sim_mcse`, `confirmed_power_mcse` and
  `half_width_simulated_mcse`. The CLI CSVs gain the same columns.
- Detectable-effect mode reports a 95% Monte Carlo interval for the protective HR
  (`hr_protective_ci_low` / `hr_protective_ci_high`, the reciprocal of the harmful
  interval), and per-unit intervals with `--biomarker-sd`. The summary quotes both
  intervals.
- `survpower.concordance.mapping_table(levels, censor_rate, design=None)`: the
  C ↔ per-SD HR table (report Table 8) with calibrated censoring, the Harrell limit
  and closed-form events / N, moved out of `survpower.report` (#8).
- `survpower.censoring.event_probability_exponential`: closed-form P(event) for the
  accrual calendar (uniform entry, exponential survival and dropout), marginalized
  over the biomarker by Gauss-Hermite quadrature. Tested against the Monte Carlo
  `censor_rate_from_design` and against Lachin & Foulkes (1986), Section 2 and
  Table 2 (#4).
- Docs: the censoring page derives the event probability, shows how accrual and
  follow-up trade off (contour plot and table) and cross-checks Lachin & Foulkes.
  The worked example now reads 31.5%. Doc figures are drawn at build time by
  `scripts/docs_figures.py` (an MkDocs hook; MkDocs >= 1.6) (#4).
- `simulate_power(..., details=True)` and `simulate_power_design(..., details=True)`
  return the full dict. The default tuple return is unchanged.

### Changed
- Docs: the sidebar sections (e.g. "Concepts") are collapsible on desktop and
  start collapsed, except the section holding the current page. "Concepts" has a
  landing page. On mobile, Material's drill-down menu is unchanged (#2).
- Docs are published for every branch: `main` at `/survpower/main/` and each
  branch at `/survpower/<branch>/`, with a branch selector in the header.
  Links from before this change redirect to `main/`. `docs/dev/` is never
  published (#2).
- Detectable-effect mode with an accrual design iterates the closed-form detectable
  HR and the design's censoring rate to a fixed point, instead of one refinement
  from HR = 1 (`planning.detectable_censor_rate`). The difference is negligible in
  typical designs and about 1 percentage point of censoring in extreme ones.
  The simulation search is then re-run at the censoring of the simulated HR until
  the two agree (at most three searches, `planning.MAX_DETECTABLE_RUNS`), so the
  reported `censor_rate` is the one at the reported HR.
- Every summary sentence states the censored fraction it used (e.g. "~97 expected
  events, 51% censored"). With an accrual design the CLI design summary no longer
  prints the fraction at HR = 1, which no result uses. The censoring page lists
  which HR each planning mode uses, and describes how censoring is generated
  (always exponential; the calendar only sets the fraction).
- Summaries: the concordance questions (`--c0`, `--precision`) state the HR the
  target C implies, per SD and, with `--biomarker-sd`, per unit.
- Median-split comparison in every HR planning mode (unadjusted analyses only):
  power, the fitted high-vs-low HR with the 95% CI a typical trial reports, the
  plug-in HR (HR per SD^1.596), expected events and model-based median OS per
  group. Power mode reports it at the given N, sample-size mode at the N a
  median split needs, detectable-HR mode at the detectable HR. It is simulated with
  the fast engine and closed form otherwise, in the summary sentence and in new
  `split_*` columns (`planning.median_split_summary`). New closed forms:
  `closed_form.median_split_median_survival`, `median_split_expected_ci`;
  `simulate_continuous_vs_dichotomized` also returns `group_hr_ci_low/high` and
  `events_high/low`.

### Deprecated
- `survpower.report`, the `survpower report` subcommand and
  `scripts/reproduce_report.py` / `scripts/reproduce_appendix_b.py` (#8). Importing
  the module emits a `DeprecationWarning`; the subcommand prints a notice on stderr.
  They will be removed in 0.4.0 (#27). To reproduce the report exactly, install the
  `v0.2.0` tag; tutorials replace them (#12, #26).

### Removed
- `survpower._utils.mc_se` (unused; replaced by `survpower.results.proportion_mcse`).

## 0.2.0 — 2026-11-03

Aligned with the revised report (November 3, 2026).

### Fixed
- Attribution of the (1 − R²) Cox sample-size formula: Hsieh, Bloch & Larsen
  (1998) and Hsieh & Lavori (2000), not "Hsieh & Lloyd".
- Docs no longer state that power is a usable diagnostic at low events-per-variable,
  or that competing risks apply to overall survival.

### Added
- Bootstrap 95% Monte Carlo interval for the calibrated N* and detectable HR, with
  a warning in the summary when the interval contains the closed-form N.
- Schoenfeld two-group formula alongside Freedman's (lifelines) for the
  dichotomized analysis; Monte Carlo dichotomized power on the same cohorts as
  the continuous analysis, with the fitted group HR.
- Low-EPV diagnostics: bias on the log-HR scale with MCSE, mean model SE,
  type-I error and size-adjusted power (null run at the same p).
- `--sensitivity omitted`: adjusted vs. unadjusted analysis with prognostic
  covariates independent of the biomarker (non-collapsibility).
- References page; caveats on fixed concordance benchmarks.

### Changed
- `--n-sims-grid` default 300 → 1000 (300 cannot resolve the ~2% calibration).
- `survpower report` follows the revised report's procedures and numbering
  (Tables 1–10, Figures 1–7).

## 0.1.0 — 2026-11-03

First packaged release, from the single-file `cox_ph_power_simulations.py` and the
Appendix B prototype (`cindex_power.py`).

### Added
- Installable package (`pip install -e .`) with a `survpower` command.
- Planning modes: power at N, required N, minimum detectable HR.
- Concordance-index effect sizes: exact C ↔ HR mapping, Harrell's and Uno's C with
  analytic standard errors, benchmark (C > C₀) and precision questions.
- Fast univariate Cox solver (`--engine fast`, default), 10–50× faster than lifelines.
- Small-N scan fallback for the simulation calibration.
- Accrual / follow-up / dropout calendar, non-evaluable fraction, per-unit HRs,
  one-sided tests.
- `survpower report [--appendix-b]` to regenerate the report's tables and figures.
- Test suite, including exact regression against the original script.
- MkDocs documentation site.

### Changed
- The four near-identical censoring calibrators share one bisection routine
  (results unchanged).
