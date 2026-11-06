# survpower

[![tests](https://github.com/JoanSano/survpower/actions/workflows/tests.yml/badge.svg)](https://github.com/JoanSano/survpower/actions/workflows/tests.yml)
[![docs](https://github.com/JoanSano/survpower/actions/workflows/docs.yml/badge.svg)](https://joansano.github.io/survpower/)

**Documentation: https://joansano.github.io/survpower/**

Power and sample-size planning for a **continuous biomarker** analysed with a
**Cox proportional-hazards model**, built for grant applications, protocols and
clinical-trial design. Effect sizes can be a hazard ratio (per SD or per raw
unit) or a **concordance index** (Harrell's or Uno's C).

Ask for one of *power*, *sample size* or *detectable effect*, give the other two,
and `survpower` solves for it by simulating the planned analysis (closed-form formulas are used as
an anchor and reported alongside). Every result comes with a sentence ready for
the protocol.

```console
$ survpower sample-size --c-index 0.55 0.60 0.70
...
  - To detect a hazard ratio of 1.2 per SD of the biomarker (concordance 0.550) with 80%
    power (two-sided Wald test at alpha = 0.05), about 258 events are required (20%
    censored): N = 322 evaluable / 322 enrolled patients (simulation-calibrated; 95% Monte
    Carlo interval 313-332 evaluable; confirmed power 81%; closed-form N = 306 enrolled).
    A median-split analysis would need ~482 enrolled (Schoenfeld closed form); at that N
    it would have 80% power and would estimate a high-vs-low HR of 1.33 (typical 95% CI
    1.09 to 1.63; plug-in HR per SD^1.596 = 1.33), with ~198 vs ~187 events and median OS
    13.0 vs 17.3 months in the high vs low group.
```

## Install

Requires Python ≥ 3.9 (tested on 3.9 to 3.14) on macOS or Linux.

```bash
git clone https://github.com/JoanSano/survpower.git
cd survpower
```

**With conda** (or mamba / micromamba), using the included `environment.yml`:

```bash
conda env create -f environment.yml    # all dependencies + survpower (editable)
conda activate survpower
```

**With a virtual environment** (venv + pip):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs]"           # drop [dev,docs] for the package only
```

Or install straight from GitHub without cloning:
`pip install git+https://github.com/JoanSano/survpower.git`.

Check the install with `survpower --help` and `pytest -q`.

## Use

| You give | Command | You get |
|---|---|---|
| effect + N | `survpower power --hr 1.15 --n 250 1200` | power (with Monte Carlo CI) |
| effect | `survpower sample-size --hr 1.15 --power 0.8 0.9` | events and N needed |
| N | `survpower detectable-effect --n 250` | minimum detectable HR and C |
| C vs. benchmark | `survpower benchmark --c-index 0.70 --c0 0.65` | N to show C > C₀ |
| precision of C | `survpower precision --c-index 0.70 --half-width 0.05` | N for a ±0.05 CI |
| precision at N | `survpower precision --c-index 0.70 --n 200` | CI half-width at that N |

Design options: `--median-os`, `--censor-rate` or `--accrual-months/--followup-months/--dropout`,
`--non-evaluable`, `--n-covariates/--r2`, `--sides`, `--biomarker-sd`,
`--c-estimator {harrell,uno,gonen-heller}`, `--c-tau`. Each command only offers the flags
it reads. Sensitivity analyses:
`--sensitivity collinearity epv heterogeneous omitted waning crossing`. See
`survpower COMMAND --help`, the [CLI reference](docs/cli.md) and [`examples/grant_recipes.sh`](examples/grant_recipes.sh).

From Python:

```python
from survpower import StudyDesign, run_sample_size_mode
table, _ = run_sample_size_mode(StudyDesign(non_evaluable=0.1), hrs=[1.2], powers=[0.8])
print(table.loc[0, "summary"])
```

## Documentation

Online at **https://joansano.github.io/survpower/**, rebuilt automatically on
every push to `main`. To preview locally:

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
  concordance.py   C <-> HR, Harrell / Uno / Gonen-Heller estimators with SEs, concordance power
  closed_form.py   Schoenfeld / Hsieh-Lavori, Freedman and median-split formulas
  calibration.py   simulation-calibrated N / HR with bootstrap MC intervals (+ small-N scan)
  scenario.py      composable scenarios: covariate, event-time, censoring models + analyses
  simulate.py      Monte Carlo engines (PH, dichotomized, adjusted, omitted covariates, non-PH)
  censoring.py     censoring calibration and the accrual calendar
  sensitivity.py   collinearity / EPV / heterogeneous / omitted / waning / crossing studies
  fitting.py       fast univariate Cox solver, Wald decisions
  dgp.py, baseline.py, plotting.py, cli.py
tests/             pytest suite
docs/              MkDocs sources
```

## Reproducing the report

The code that generated the report's tables was removed in 0.4.0, and tutorials
covering the same analyses are being written. To regenerate the report's numbers
exactly, install the `v0.2.0` tag:

```bash
pip install "git+https://github.com/JoanSano/survpower.git@v0.2.0"
```

See [Reproducing the report](https://joansano.github.io/survpower/main/reproducing-the-report/).

## Tests

```bash
pytest -q
```

## License

MIT
