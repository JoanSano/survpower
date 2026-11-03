# Changelog

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
