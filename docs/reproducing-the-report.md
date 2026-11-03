# Reproducing the report

The package grew out of the report *Power Analysis for a Continuous Biomarker
Under the Cox Proportional-Hazards Model* and its Appendix B (*Power for a
target concordance index*). Both can be regenerated:

```bash
survpower report --outdir report_out                      # main report, ~10-15 min
survpower report --appendix-b --outdir appB_out           # Appendix B, a few min
survpower report --appendix-b --n-sims 200 --outdir tmp   # quick look
```

or `python scripts/reproduce_report.py` / `python scripts/reproduce_appendix_b.py`.

The main report runs with the lifelines engine and the original seeds, so the
tables match the PDF exactly. The regression tests in `tests/test_regression.py`
pin the package against values produced by the original single-file script.

## Where each result comes from

| Report item | Function | Output |
|---|---|---|
| Table 1 (continuous vs. dichotomized power) | `report.run_part1` | `part1_continuous_vs_dichotomized.csv` |
| Table 2, Figure 1 (calibrated N) | `report.run_part1b`, `plotting.plot_calibration_curves` | `part1b_continuous_sample_size.csv`, `calibration_curve.pdf` |
| Figure 2 (collinearity) | `sensitivity.run_collinearity_study_both_cohorts` | `part1c_collinearity_study.csv`, `collinearity_power.pdf` |
| Table 3, Figure 3 (low EPV) | `sensitivity.run_epv_stress_test` | `part1d_epv_stress_test.csv`, `epv_bias.pdf` |
| Table 4, Figure 4 (prognostic covariates) | `sensitivity.run_heterogeneous_adjustment_study` | `part1e_heterogeneous_adjustment.csv`, `informative_adjustment.pdf` |
| Table 5 (waning effect) | `report.run_part2` | `part2_waning_effect.csv` |
| Table 6 (crossing hazards) | `report.run_part3` | `part3_crossing_hazards.csv` |
| Table B1, Figure B1 (C ↔ HR) | `report.run_cindex_mapping` | `tableB1_mapping.csv`, `figB1_concordance_mapping.pdf` |
| Figure B2 (association power) | `report.run_cindex_association_grid` | `figB2_association_grid.csv`, `figB2_association_power.pdf` |
| Table B2 (type I error) | `report.run_cindex_null` | `tableB2_type1_error.csv` |
| Table B3, Figure B3 (benchmark) | `report.run_cindex_benchmark` | `tableB3_benchmark_n.csv`, `figB3_benchmark_power.pdf` |

Appendix B in the package uses the fast Cox solver and the analytic σ_C. The
original prototype used an empirical SD at N = 3000 (σ_C = 0.323 there vs.
≈0.313 here). Its numbers therefore agree with the PDF up to Monte Carlo error
rather than digit for digit. For example, the calibrated N for C₀ = 0.65 is
≈231 here vs. 239 in the PDF; for C₀ = 0.60 it is 67 in both.
