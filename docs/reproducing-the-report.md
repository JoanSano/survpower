# Reproducing the report

The package grew out of the report *Power Analysis for a Continuous Biomarker
Under the Cox Proportional-Hazards Model: Continuous vs. Dichotomized Analysis,
Adjustment, Concordance, and Robustness to Non-Proportional Hazards* (revised
November 3, 2026). Both parts can be regenerated:

```bash
survpower report --outdir report_out                      # Sections 1-10, Appendix A (~35 min)
survpower report --appendix-b --outdir appB_out           # Appendix B (a few minutes)
survpower report --appendix-b --n-sims 200 --outdir tmp   # quick look
```

or `python scripts/reproduce_report.py` / `python scripts/reproduce_appendix_b.py`.

## Where each result comes from

| Report item | Function | Output |
|---|---|---|
| Table 1 (continuous vs. dichotomized, same data) | `report.run_part1` | `table1_continuous_vs_dichotomized.csv` |
| Table 2, Figure 1 (calibrated N with MC interval) | `report.run_part1b`, `plotting.plot_calibration_curves` | `table2_calibrated_sample_size.csv`, `fig1_calibration_curve.pdf` |
| Figure 2 (collinearity) | `sensitivity.run_collinearity_study_both_cohorts` | `fig2_collinearity_study.csv`, `fig2_collinearity_power.pdf` |
| Table 3, Figure 3 (low EPV: bias, SE, type-I error) | `sensitivity.run_epv_stress_test` | `table3_epv_stress_test.csv`, `fig3_epv.pdf` |
| Table 4, Figure 4 (prognostic covariates, adjusted) | `sensitivity.run_heterogeneous_adjustment_study` | `table4_heterogeneous_adjustment.csv`, `fig4_informative_adjustment.pdf` |
| Table 5 (omitted prognostic covariates) | `sensitivity.run_omitted_covariates_study` | `table5_omitted_covariates.csv` |
| Table 6 (waning effect) | `report.run_part2` | `table6_waning_effect.csv` |
| Table 7 (crossing hazards) | `report.run_part3` | `table7_crossing_hazards.csv` |
| Table 8, Figure 5 (C ↔ HR) | `report.run_cindex_mapping` | `table8_concordance_mapping.csv`, `fig5_concordance_mapping.pdf` |
| Figure 6 (association power vs. N) | `report.run_cindex_association_grid` | `fig6_association_grid.csv`, `fig6_association_power.pdf` |
| Table 9 (type-I error of both tests) | `report.run_cindex_null` | `table9_type1_error.csv` |
| Table 10, Figure 7 (benchmark) | `report.run_cindex_benchmark` | `table10_benchmark_n.csv`, `fig7_benchmark_power.pdf` |

## How closely it matches

The simulation engines are pinned to the original single-file script by
`tests/test_regression.py` (identical results on identical seeds). The revised
report re-ran several analyses with new seeds and more replicates, so its
Monte Carlo numbers are matched up to Monte Carlo error rather than digit for
digit:

| Item | Revised report | `survpower report` | |
|---|---|---|---|
| Table 1 closed forms (Freedman / Schoenfeld) | 0.652/0.654, 0.931/0.933, 0.188/0.188, 0.348/0.351 | same | exact |
| Table 1 Monte Carlo power, continuous | 0.849, 0.990, 0.258, 0.491 | 0.832, 0.986, 0.250, 0.479 | within ~1.5 MCSE |
| Table 1 fitted group HR | 1.167, 1.246, 1.162, 1.249 | 1.162, 1.246, 1.157, 1.242 | MC error |
| Table 2 calibrated N* [95% interval] | 1107 [1084, 1130]; 514 [503, 524] | 1109 [1086, 1132]; 515 [504, 525] | MC error |
| Table 3 type-I error at EPV 4.9 / 2.5 | 0.087 / 0.143 | 0.089 / 0.135 | MC error |
| Table 5 unadjusted power / HR at σ_Z = 1 | 0.281 / 1.042 | 0.249 / 1.041 | MC error |
| Tables 6–7 (non-PH) | | | exact |
| Table 8 (C ↔ HR, Harrell limit) | | | exact |

Two implementation differences are worth knowing:

* **Calibration (Table 2)** uses the fast univariate Cox solver. Decisions
  agree with lifelines on the same data (tested); the confirmation run uses
  10,000 replicates as in the report.
* **σ_C in Appendix B** is estimated from the analytic SE at N = 2000 (≈ 0.313
  at C = 0.70), where the report used the empirical SD at N = 3000 (0.323). The
  calibrated benchmark N is the same either way within MC error (67 for
  C₀ = 0.60; ≈ 231 vs. 239 for C₀ = 0.65). As the revised report notes, the
  300-replicate calibration in Table 10 is itself uncertain by several percent.
