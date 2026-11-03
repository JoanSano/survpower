# survpower

Power and sample-size planning for a **continuous biomarker** analysed with a
**Cox proportional-hazards model**, built for grant applications, protocols and
clinical-trial design. Effect sizes can be a hazard ratio (per SD or per raw
unit) or a **concordance index** (Harrell's or Uno's C).

Give two of *effect size*, *sample size* and *power*, and `survpower` solves for
the third by simulating the planned analysis (closed-form formulas are used as
an anchor and reported alongside). Every result comes with a sentence ready for
the protocol.

```console
$ survpower --c-index 0.55 0.60 0.70
...
  - To detect a hazard ratio of 1.2 per SD of the biomarker (concordance 0.550) with 80%
    power (two-sided Wald test at alpha = 0.05), about 250 events are required:
    N = 312 evaluable / 312 enrolled patients (simulation-confirmed power 81%;
    closed-form N = 306 enrolled). A median-split analysis would need ~488 enrolled.
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,docs]"
```

## Use

| You give | Command | You get |
|---|---|---|
| effect + N | `survpower --hr 1.15 --n 250 1200` | power (with Monte Carlo CI) |
| effect | `survpower --hr 1.15 --power 0.8 0.9` | events and N needed |
| N | `survpower --n 250` | minimum detectable HR and C |
| C vs. benchmark | `survpower --c-index 0.70 --c0 0.65` | N to show C > C₀ |
| precision of C | `survpower --c-index 0.70 --precision 0.05` | N for a ±0.05 CI |

Design options: `--median-os`, `--censor-rate` or `--accrual-months/--followup-months/--dropout`,
`--non-evaluable`, `--n-covariates/--r2`, `--sides`, `--biomarker-sd`,
`--c-estimator {harrell,uno}`, `--c-tau`. Sensitivity analyses:
`--sensitivity collinearity epv heterogeneous omitted waning crossing`. See
`survpower --help` and [`examples/grant_recipes.sh`](examples/grant_recipes.sh).

From Python:

```python
from survpower import StudyDesign, run_sample_size_mode
table, _ = run_sample_size_mode(StudyDesign(non_evaluable=0.1), hrs=[1.2], powers=[0.8])
print(table.loc[0, "summary"])
```

## Documentation

```bash
mkdocs serve      # live site at http://127.0.0.1:8000
mkdocs build      # static HTML in site/
```

The docs cover the recipes, every CLI option, and the methods: effect sizes and
the C ↔ HR mapping, censoring and accrual, adjustment covariates, the three
concordance questions, non-proportional hazards, and how the simulation
calibration works.

## Layout

```
src/survpower/
  design.py        StudyDesign: the shared design assumptions
  planning.py      the solve-for-X modes behind the CLI
  concordance.py   C <-> HR, Harrell / Uno estimators with SEs, concordance power
  closed_form.py   Schoenfeld / Hsieh-Lavori, Freedman and median-split formulas
  calibration.py   simulation-calibrated N / HR with bootstrap MC intervals (+ small-N scan)
  simulate.py      Monte Carlo engines (PH, dichotomized, adjusted, omitted covariates, non-PH)
  censoring.py     censoring calibration and the accrual calendar
  sensitivity.py   collinearity / EPV / heterogeneous / omitted / waning / crossing studies
  fitting.py       fast univariate Cox solver, Wald decisions
  dgp.py, baseline.py, plotting.py, report.py, cli.py
tests/             pytest suite, incl. exact regression vs. the original script
docs/              MkDocs sources
```

## Reproducing the report

```bash
survpower report --outdir report_out             # main report (revised Nov 2026)
survpower report --appendix-b --outdir appB_out  # Appendix B (concordance)
```

## Tests

```bash
pytest -q
```

## License

MIT
