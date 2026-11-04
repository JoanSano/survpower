# survpower

**Power and sample-size planning for a continuous biomarker analysed with a Cox
proportional-hazards model**, built for grant applications, protocols and
clinical-trial design.

You describe the study (survival, follow-up, attrition, adjustment model) and
give *two* of the three linked quantities. `survpower` solves for the third:

| You give | You get |
|---|---|
| effect size **and** sample size | power, with a Monte Carlo confidence interval |
| effect size | the number of events and patients needed for a target power |
| sample size | the smallest effect detectable with a target power |

The effect size can be a **hazard ratio per SD** of the biomarker (or per raw
unit), or a **concordance index** (Harrell's or Uno's C). For concordance targets
`survpower` also answers two questions that a hazard ratio cannot: *does the
biomarker beat a benchmark C₀?* and *how precisely will C be estimated?*

Every result row includes a ready-to-paste sentence for the protocol or grant:

> To detect a hazard ratio of 1.2 per SD of the biomarker (concordance 0.550)
> with 80% power (two-sided Wald test at alpha = 0.05), about 250 events are
> required: N = 312 evaluable / 312 enrolled patients (simulation-confirmed
> power 81%; closed-form N = 306 enrolled). A median-split analysis would need
> ~488 enrolled.

## Why simulation, not only formulas

The textbook Schoenfeld / Hsieh–Lavori formula is a fast starting point, but it
is optimistic: by under one percentage point of power (about 2% in N) at small
hazard ratios, and by about **50%** in N at large effects (C ≈ 0.70: the formula
says N = 14, the truth is N ≈ 21). `survpower`
anchors on the closed form, then **calibrates against simulation of the exact
analysis**, reports the Monte Carlo uncertainty of the calibrated N, and
confirms the answer with an independent run. See
[How the simulations work](concepts/methods.md).

## Installation

Requires Python ≥ 3.9 (tested on 3.9 to 3.14) on macOS or Linux, with NumPy,
SciPy, pandas, lifelines and matplotlib.

```bash
git clone https://github.com/JoanSano/survpower.git
cd survpower
```

=== "conda / mamba"

    ```bash
    conda env create -f environment.yml   # dependencies from conda-forge + survpower
    conda activate survpower
    ```

=== "venv + pip"

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e .            # the package and the `survpower` command
    pip install -e ".[dev]"     # + pytest
    pip install -e ".[docs]"    # + MkDocs, to build this site (mkdocs serve)
    ```

=== "pip, no clone"

    ```bash
    pip install git+https://github.com/JoanSano/survpower.git
    ```

Check the install with `survpower --help`.

## Thirty-second tour

```bash
survpower --hr 1.15                                   # N for 80% power
survpower --hr 1.15 --n 250 1200                      # power at two cohort sizes
survpower --n 250                                     # minimum detectable HR / C
survpower --c-index 0.70 --c0 0.65                    # N to show C > 0.65
survpower --c-index 0.70 --precision 0.05             # N for a ±0.05 CI on C
```

Next: [Quickstart recipes](quickstart.md) · [CLI reference](cli.md) ·
[Python API](api.md).

These pages document the `main` branch. To read the docs of a branch under
development, use the selector next to the site title.

## Scope and assumptions

* One continuous biomarker, approximately normal after any transform; effects
  are proportional hazards unless you run the [non-PH sensitivity
  analyses](concepts/ph-violations.md).
* A constant baseline hazard calibrated to a median overall survival
  (default 15 months, typical of glioblastoma). For a proportional-hazards
  analysis, power and N do not depend on this choice. Only the non-PH
  scenarios, which use fixed clinical time windows, do.
* Independent censoring, given directly or derived from accrual, follow-up and
  dropout.
* Time is in months throughout.
