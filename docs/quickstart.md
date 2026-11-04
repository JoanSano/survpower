# Quickstart: grant and trial recipes

Each recipe is one command. Results print as a table followed by "Summary (for
protocol / grant text)" sentences, and are saved as CSV (plus PDF figures) in
`--outdir` (default `survpower_output/`). Add `--quick` while exploring; drop it
for the numbers you report.

!!! tip "N always means patients enrolled"
    Sample sizes you pass with `--n`, and the `n_enrolled` column, are
    **enrolled** patients. Simulations use the evaluable subset after
    `--non-evaluable`, so missing tissue or failed assays are accounted for
    automatically.

## 1. "How many patients do I need?"

```bash
survpower --hr 1.20 --power 0.80 0.90
```

Required events and N at each target power: the closed form and the
simulation-calibrated value with its 95% Monte Carlo interval, plus the N a
median-split (dichotomized) analysis would need, for the "why we keep the
biomarker continuous" paragraph. A median split keeps only 2/π ≈ 64% of the
information, equivalent to discarding ~36% of the events.

## 2. "Is my existing cohort big enough?"

```bash
survpower --hr 1.10 1.15 --n 250 1200
```

Power for every (N, HR) pair, with a 95% Monte Carlo interval, the closed-form
prediction and the dichotomized power.

## 3. "What is the smallest effect my cohort can detect?"

```bash
survpower --n 250
```

The minimum detectable hazard ratio per SD and its protective reciprocal, each
with a 95% Monte Carlo interval, expressed as a concordance index too. With 15-month median OS and 20%
censoring, N = 250 (~200 events) detects HR ≈ 1.23 per SD (C ≈ 0.56) with 80%
power.

## 4. A realistic trial calendar

```bash
survpower --hr 1.25 \
    --accrual-months 36 --followup-months 12 --dropout 0.05 \
    --non-evaluable 0.10 --median-os 15
```

The censoring rate follows from uniform accrual over 36 months, 12 further
months of follow-up and 5% annual loss to follow-up. N is inflated for the 10%
of patients without an evaluable biomarker. See
[Censoring, accrual and attrition](concepts/censoring-and-accrual.md).

## 5. The biomarker will be adjusted for clinical covariates

```bash
survpower --hr 1.20 --n-covariates 5 --r2 0.3
```

`--r2` is the R² of the biomarker regressed on the adjustment covariates (from
pilot or registry data). Power is lost through this correlation, not through the
number of covariates. Add `--sensitivity collinearity epv` to check that
directly. If the covariates are prognostic but nearly independent of the
biomarker, adjusting for them *increases* power (`--sensitivity omitted`). See [Adjustment covariates](concepts/adjustment.md).

## 6. The effect is pre-registered as a concordance index

```bash
survpower --c-index 0.70                       # association: N to detect any effect
survpower --c-index 0.70 --c0 0.65             # discrimination beats a benchmark
survpower --c-index 0.70 --precision 0.05      # precision of the estimated C
```

These three questions need very different sample sizes for the same C = 0.70:
about **21**, **230** and **150** patients respectively (20% censoring).
State which one your study answers. See
[Concordance questions](concepts/concordance.md). Use
`--c-estimator uno` for Uno's censoring-robust C.

## 7. Per-unit instead of per-SD effects

```bash
survpower --hr 1.05 --biomarker-sd 4.2
```

The HR is per raw unit of the biomarker (e.g. per mm³ or per index point);
results are reported on both scales.

## 8. What if hazards are not proportional?

```bash
survpower --hr 1.30 --n 400 --sensitivity waning crossing --tau 6
```

Re-simulates the planning point under an effect that disappears after 6 months,
and one that reverses direction, and reports the naive Cox power, the bias and
how often the Schoenfeld-residual test would flag the problem. See
[Non-proportional hazards](concepts/ph-violations.md).

## From Python

```python
from survpower import StudyDesign, run_sample_size_mode, run_cindex_benchmark_mode

design = StudyDesign(accrual_months=36, followup_months=12, dropout_annual=0.05,
                     non_evaluable=0.1)
table, _ = run_sample_size_mode(design, hrs=[1.2, 1.3], powers=[0.8, 0.9])
print(table[["hr", "target_power", "n_enrolled", "expected_events"]])

bench = run_cindex_benchmark_mode(StudyDesign(c_estimator="uno"), cs=[0.70], c0s=[0.65])
print(bench["summary"][0])
```

## Writing it up

A defensible sample-size paragraph states:

1. the effect size and **why** (prior study, pilot data, minimal clinically
   important effect), on the scale you will analyse;
2. the test (two-sided Wald test from a Cox model, α), and the adjustment set
   with its assumed R²;
3. the survival assumptions (median OS, accrual, follow-up, dropout) and the
   resulting **expected number of events**;
4. that N was obtained by simulation of the planned analysis (`survpower`,
   seed, number of replicates), with its Monte Carlo interval, and confirmed
   at the chosen N;
5. the inflation for non-evaluable patients;
6. a sensitivity analysis, e.g. smaller effect, waning effect.
