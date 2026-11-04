# Changelog

## Unreleased

### Added
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
