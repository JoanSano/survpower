# Changelog

## Unreleased

### Added
- **Baseline-hazard families** (#14), calibrated in closed form so that the baseline median
  is `median_os`: `Exponential`, `Weibull`, `Gompertz` (with a cure fraction when its shape
  is negative), `PiecewiseExponential`, `LogLogistic` and `NumericBaseline` (any hazard
  shape, by quadrature; API only), in `survpower.baseline`. `StudyDesign(baseline=...)`
  selects one; the CLI has `--baseline {exponential,weibull,gompertz,loglogistic}` and
  `--baseline-shape K` on every command. Every planning path supports them except
  `--sensitivity`, which raises. `scenario.PHModel` samples PH event times by inverting the
  cumulative hazard; `marginal_median` gives the median over the biomarker. With no
  `baseline`, results at fixed seeds are unchanged.
- **A baseline estimated from a cohort** (#84): `EmpiricalBaseline.from_data(time, event,
  biomarker=None)` gives Breslow's baseline cumulative hazard at the mean biomarker
  (Nelson-Aalen without one), interpolated between event times, with a constant-hazard
  tail. From the CLI: `--baseline-data cohort.csv --baseline-columns TIME EVENT [BIOMARKER]`;
  the median OS then comes from the data. `--median-os` now defaults to unset (15 months).
- `censor_rate_from_design`, `calibrate_censoring_rate`, the concordance limits and
  simulations, `simulate_power`, `simulate_power_design`, `simulate_power_with_adjustment`
  and `simulate_continuous_vs_dichotomized` take a `baseline=` keyword. `harrell_limit`
  falls back to Monte Carlo over censored pairs for a non-exponential baseline.
- With a PH baseline, the median-split summaries report each group's model-based median
  OS for any family, not only the exponential one.
- `survpower.simulate_cohort` (provisional): one simulated cohort from a `Scenario`, with
  `time`, `event`, `X`, `Z0..Zp-1` and the latent `T` and `C`, drawn with the same models
  and censoring calibration as `simulate` (#78). `scenario.CorrelatedCovariates` gives
  scenarios adjustment covariates with a target R² on X.

### Changed
- `StudyDesign.lambda0` raises for a non-exponential baseline, pointing to
  `StudyDesign.baseline`. `require_default_models` takes `supports=`, the model fields a
  path honours.

### Removed
- **The flag-only CLI syntax** (`survpower --hr 1.15 ...`), deprecated in 0.4.0. A first
  argument that starts with `--`, other than `--help` and `--version`, now exits with
  code 2 and lists the commands; see "Migrating from the flag-only syntax" in
  `docs/cli.md` (#66).
- The `survpower report` stub. `report` is now an invalid command; the report is still
  reproducible from the `v0.2.0` tag (#66).
- The docs and README no longer refer to the report, and the "Reproducing the report"
  page is gone. Numbers that cited the
  report are recomputed with the package, and tutorial 01 now checks its simulated power
  against the closed forms (#70).
- Docstrings, comments and two messages in `survpower` (the concordance-question
  `ValueError` and the small-N warning) no longer cite the report's sections, tables,
  figures or equation numbers (#70).

### Fixed
- `docs/concepts/methods.md` no longer describes regression tests that were removed in
  0.3.0.

## 0.4.0 — 2026-11-06

Milestone M1.5 (CLI and model seams): question subcommands (#61), one route from
`StudyDesign` to a `Scenario` (#62) and removal of `survpower.report` (#27). No numeric
results change: at fixed seeds, the CSVs are byte-identical to those from before these
changes. `StudyDesign` also now validates its values at construction and is frozen (#50).
**Scripts should move to the subcommands**: the old syntax is removed in 0.4.1 (#66).

### Added
- `survpower.Effect` (a scale and a per-SD value) and `survpower.scenario_for`
  (provisional), plus a contributor guide to adding models,
  `docs/dev/adding-a-model.md` (#62).
- `survpower --version` (#61; #54 builds on it).

### Changed
- **The CLI's question is now a command**, not inferred from the flags (#61). A missing
  flag used to change the question with no error: `--hr 1.15 --n 250` without `--n`
  became a sample-size calibration. CSV names, columns and every number are unchanged.

  | Before | Now |
  |---|---|
  | `survpower --hr 1.15 --n 250` | `survpower power --hr 1.15 --n 250` |
  | `survpower --hr 1.15` | `survpower sample-size --hr 1.15` |
  | `survpower --n 250` | `survpower detectable-effect --n 250` |
  | `survpower --c-index 0.7 --c0 0.65` | `survpower benchmark --c-index 0.7 --c0 0.65` |
  | `survpower --c-index 0.7 --precision 0.05` | `survpower precision --c-index 0.7 --half-width 0.05` |
  | `survpower --c-index 0.7 --precision 0.05 --n 200` | `survpower precision --c-index 0.7 --n 200` |

  `--c-index` also works with `power` and `sample-size`. The old syntax still works,
  with a warning on stderr that shows the new command, and is removed in 0.4.1 (#66).
- Each command offers only the flags its path reads, so a flag that used to be accepted
  and ignored now exits with code 2: `--sides` on `benchmark`, `--c-estimator` on `power`,
  `--power` on `power`, `--sensitivity` on `precision`, `--c-tau` without
  `--c-estimator uno`, and `--tau`, `--waning-hr`, `--crossing-hr` or `--n-sims-sensitivity`
  without an analysis that uses them. The old syntax drops these instead, and the warning
  lists them, so old scripts keep their results (#61).
- `--precision` is now `--half-width` (#61). Flags can no longer be abbreviated on the
  commands, so that adding a flag cannot break a script.
- `survpower` with no command prints the help and exits with code 2 (#61).
- `CensoringModel.draw` now also receives the cohort's event times: `draw(cov,
  event_times, rng)`. Custom censoring models must add the argument (the scenario API is
  provisional). Existing models ignore it; a model can use it to cut an analysis at a
  target number of events (#62).
- The unadjusted power, sample-size and detectable-effect paths now build their
  simulations with `scenario_for(design, effect, analyses)`, so a new design option has
  one place to reach them. Results are identical at a fixed seed. Adjusted, sensitivity
  and concordance paths raise `ValueError` for any model they cannot honour, rather than
  fall back to the exponential model, and `method="closed_form"` raises for a model with no
  closed form (#62).
- Passing `censor_rate` together with `accrual_months` / `followup_months` to
  `StudyDesign` now raises, as `dropout_annual` without an accrual design does. Both were
  silently ignored in Python, and the CLI already rejected them (#50).
- `StudyDesign` is now frozen, so a design stays valid after construction and is hashable.
  Code that assigns to a field, such as `design.n_sims = 5000`, now raises
  `FrozenInstanceError`; use `dataclasses.replace(design, n_sims=5000)` instead.

### Fixed
- `precision --n` returns the half-width at that N without a dummy target (#61). The
  design summary no longer prints a two-sided Wald test for `benchmark`, which is always
  one-sided (#61).
- `StudyDesign` now raises `ValueError` at construction for invalid values, naming the
  field. Before, the Python API accepted them, and some returned impossible answers
  with no error: `sample_size(hr=1.3, design=StudyDesign(censor_rate=1.2),
  method="closed_form")` gave N = -570. The rules moved here from the CLI, which reports
  them with the same messages and exit code (#50).
- New checks that the CLI also gains: `median_os`, `c_tau` and `biomarker_sd` must be
  positive and finite, `n_sims` and `n_sims_grid` positive integers, and
  `--accrual-months 0 --followup-months 0`, which used to crash with an `OverflowError`,
  is rejected (#50).

### Removed
- `survpower.report`, the `report` subcommand and `scripts/reproduce_report.py` /
  `scripts/reproduce_appendix_b.py`, deprecated in 0.3.0. `survpower report` now exits
  with a message pointing to the `v0.2.0` tag, which reproduces every report table
  exactly, and to the tutorials. `concordance.mapping_table` still gives Table 8. The
  plotting helpers the module used are kept for the tutorials (#27).

## 0.3.0 — 2026-11-04

Milestone M1 (foundations): scenario engine (#9), `sample_size()` (#11) and
executable tutorials (#12).

### Added
- `survpower.scenario` (provisional): a `Scenario` composes a covariate model, an
  event-time model, a censoring model and a tuple of analyses; `simulate(scenario,
  n, ...)` returns one row per analysis with the MCSE columns of
  `results.summarize_replicates`. Shipped models: `StandardNormal`, `ExponentialPH`,
  `ExponentialCensoring`, `CoxWald` (fast or lifelines, optionally on the median
  split) and `LogRank`. A model is any object with the protocol's methods, so new
  data-generating options need no new simulation loop. Exported as
  `survpower.Scenario` and `survpower.simulate_scenario` (#9).
- `sample_size(hr=..., c_index=..., power, design, method)`: one entry point for the
  required N, given an HR or a population concordance (converted with `hr_of_c`).
  `method` is `"closed_form"` (Hsieh & Lavori 2000), `"calibrated"` (probit
  simulation calibration) or `"auto"` (default: calibrated, with a direct integer
  scan below N = 100 or when the probit slope is not positive). It returns a
  `SampleSizeResult` with N evaluable / enrolled, events, the closed-form N, the
  Monte Carlo interval, the confirmed power as an `Estimate` (value and MCSE) and a
  summary sentence. Tested against Hsieh & Lavori's (2000) multiple-myeloma example
  as printed in the Stata manual (E/N = 78/106) (#11).
- `closed_form.sample_size_continuous_cph(..., var_x=)`: the covariate variance, as
  an alternative to `sigma_x` (#11).
- `results.Estimate`: a Monte Carlo value with its MCSE (#11).
- Executable tutorials: Jupytext `.py` files in `docs/tutorials/`, rendered and
  executed by `mkdocs-jupyter`. First two: continuous vs. median split, and sample
  size for a target power. A `tutorials` CI job runs each at
  `SURVPOWER_TUTORIAL_SIMS=50`; `docs/tutorials/README.md` explains how to add one.
  The `docs` extra gains `mkdocs-jupyter` and `jupytext` (#12).
- `censoring.calibrate_censoring_for(sampler, target, seed)`: censoring
  calibration for any event-time generator. The four existing calibrators delegate
  to it (results unchanged) (#9).

### Changed
- `run_sample_size_mode` is built on `sample_size(method="auto")`. Its columns and
  values are unchanged, except when the probit slope over the N grid is not
  positive: it then scans integers around the closed-form N, instead of reporting
  the closed-form N with a warning (#11).
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
