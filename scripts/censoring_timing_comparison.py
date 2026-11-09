"""Does the timing of censoring matter, at a matched censored fraction? (#16)

At the docs example (median OS 15 months, accrual A = 36, follow-up F = 12, 5%/yr
dropout) this compares two ways of simulating the same calendar:

* "exponential": exponential censoring calibrated to the calendar's censored fraction
  (the default `censoring_model`);
* "calendar": the calendar simulated directly (`CalendarCensoring`).

For each per-SD HR, N is the closed-form N for 80% power at that HR's censored
fraction. Each cell runs `N_SIMS` replicates and records Cox Wald power, and the power
of the test of C = 0.5 and the mean C for Harrell's, Uno's (fixed horizon `TAU`)
and Gonen-Heller's C. The two models' runs are independent, so the MCSE of a
difference is the root sum of squares of the two MCSEs.

    python scripts/censoring_timing_comparison.py [out.csv]
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from survpower.closed_form import sample_size_continuous_cph
from survpower.concordance import simulate_cindex_power
from survpower.design import StudyDesign
from survpower.scenario import CalendarCensoring

HRS = (1.10, 1.15, 1.5)
N_SIMS = 2000
TAU = 24.0  # months: one Uno horizon for both models, inside every patient's follow-up window
METHODS = ("harrell", "uno", "gonen-heller")
DESIGN = StudyDesign(median_os=15, accrual_months=36, followup_months=12, dropout_annual=0.05)
CALENDAR = CalendarCensoring(36, 12, dropout_annual=0.05)


def run(n_sims: int = N_SIMS) -> pd.DataFrame:
    rows = []
    for i, hr in enumerate(HRS):
        cr = DESIGN.censor_rate_for(hr)
        _, n = sample_size_continuous_cph(hr, 0.80, alpha=0.05, event_prob=1 - cr)
        beta, lam_c = np.log(hr), DESIGN.lambda_c_for(hr)
        for model, censoring in (("exponential", None), ("calendar", CALENDAR)):
            row = dict(hr=hr, n=n, censor_rate_design=cr, model=model)
            for j, method in enumerate(METHODS):
                r = simulate_cindex_power(n, beta, lam_c, n_sims=n_sims, method=method, tau=TAU,
                                          lambda0=DESIGN.lambda0, censoring=censoring,
                                          seed=1000 * i + 100 * j + (model == "calendar"))
                if j == 0:
                    row.update(power_cox=r["power_cox"], power_cox_mcse=r["power_cox_mcse"],
                               mean_events=r["mean_events"],
                               censor_rate_realized=r["censor_rate_realized"],
                               censor_rate_realized_mcse=r["censor_rate_realized_mcse"])
                key = method.replace("-", "_")
                row.update({f"power_c05_{key}": r["power_c05"],
                            f"power_c05_{key}_mcse": r["power_c05_mcse"],
                            f"mean_c_{key}": r["mean_c"], f"mean_c_{key}_mcse": r["mean_c_mcse"]})
            rows.append(row)
    return pd.DataFrame(rows)


def differences(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar minus exponential for each measure, with its MCSE and the ratio."""
    measures = [c for c in df.columns if f"{c}_mcse" in df.columns]
    out = []
    for hr, g in df.groupby("hr"):
        e, c = g[g.model == "exponential"].iloc[0], g[g.model == "calendar"].iloc[0]
        for m in measures:
            diff = c[m] - e[m]
            mcse = float(np.hypot(c[f"{m}_mcse"], e[f"{m}_mcse"]))
            out.append(dict(hr=hr, n=int(e["n"]), measure=m, exponential=e[m], calendar=c[m],
                            diff=diff, diff_mcse=mcse, z=diff / mcse if mcse > 0 else np.nan))
    return pd.DataFrame(out)


if __name__ == "__main__":
    df = run()
    path = sys.argv[1] if len(sys.argv) > 1 else "censoring_timing_comparison.csv"
    df.to_csv(path, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(differences(df).round(4).to_string(index=False))
