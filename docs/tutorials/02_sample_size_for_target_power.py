# %% [markdown]
# # Sample size for a target power
#
# **Question.** How many patients does a study need for 80% power, and how far can
# the closed-form answer be trusted?
#
# **Design assumptions.** X ~ N(0, 1), exponential survival with a median of
# 15 months, 20% independent censoring, two-sided α = 0.05, no adjustment
# covariates. The closed form is Hsieh & Lavori (2000). The calibration simulates
# power on a grid of N around it, fits a probit line in √N and inverts that line
# for the target.

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from survpower import StudyDesign, sample_size

# Replicates per grid point and for confirmation. The docs build uses the default;
# CI smoke-tests with 50.
N_SIMS = int(os.environ.get("SURVPOWER_TUTORIAL_SIMS", 1000))
TARGET = 0.80
design = StudyDesign(censor_rate=0.20, n_sims=N_SIMS, n_sims_grid=N_SIMS, seed=42)

# %% [markdown]
# ## Solve for N
#
# `method="calibrated"` always uses the probit fit, so each row has a bootstrap
# Monte Carlo interval for N\*. `confirmed` is the simulated power at N\* from an
# independent run, with its MCSE.

# %%
results = {hr: sample_size(hr=hr, power=TARGET, design=design, method="calibrated")
           for hr in (1.10, 1.15, 1.20)}
table = pd.DataFrame([dict(
    hr_per_sd=hr, c_index=round(r.c_index, 3), events_closed_form=r.events_closed_form,
    n_closed_form=r.n_closed_form, n_star=r.n_evaluable,
    n_ci_low=r.n_ci[0] if r.n_ci else np.nan, n_ci_high=r.n_ci[1] if r.n_ci else np.nan,
    confirmed=r.confirmed_power.value, confirmed_mcse=r.confirmed_power.mcse,
) for hr, r in results.items()])
table

# %% [markdown]
# ## The calibration, for HR = 1.15

# %%
r = results[1.15]
grid = pd.DataFrame(r.details["grid"])
a, b = r.details["a_hat"], r.details["b_hat"]
mcse = np.sqrt(grid["power_hat"] * (1 - grid["power_hat"]) / N_SIMS)
ns = np.linspace(grid["n"].min(), grid["n"].max(), 200)

BLUE, ORANGE = "#2a78d6", "#eb6834"
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.errorbar(grid["n"], grid["power_hat"], yerr=1.96 * mcse, fmt="o", color=BLUE, markersize=7,
            capsize=0, linewidth=2, label="Simulated power (95% MC interval)")
ax.plot(ns, norm.cdf(a * np.sqrt(ns) + b), color=BLUE, linewidth=2, label="Probit fit")
ax.axhline(TARGET, color="#8a8983", linewidth=1)
ax.axvline(r.n_closed_form, color=ORANGE, linewidth=2, linestyle="--",
           label=f"Closed form N = {r.n_closed_form}")
ax.axvline(r.n_evaluable, color=BLUE, linewidth=2, linestyle=":",
           label=f"Calibrated N* = {r.n_evaluable}")
ax.set_xlabel("Evaluable patients N")
ax.set_ylabel("Power")
ax.grid(axis="y", color="#e5e4e0", linewidth=1)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left")
ax.set_title("Calibrating N for 80% power at HR = 1.15", loc="left")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Interpretation
#
# The closed form is close when there are hundreds of events, but slightly
# optimistic. If the interval for N\* contains the closed-form N, this budget has
# not resolved a difference between the two; more replicates per grid point would
# narrow the interval. Quote N\*, its interval and the confirmed power together.

# %%
for hr, res in results.items():
    verdict = ("does not resolve a difference from" if res.n_ci and
               res.n_ci[0] <= res.n_closed_form <= res.n_ci[1] else "differs from")
    print(f"HR {hr:.2f}: N* = {res.n_evaluable} {verdict} closed-form N = {res.n_closed_form}")

# %% [markdown]
# ## Agreement with the closed form
#
# The closed-form values are exact: 864 events and N = 1081 for HR = 1.10, and
# 402 events and N = 503 for HR = 1.15 (Hsieh & Lavori, 2000). This script always
# checks them. At the default budget it also checks that power at N\* is within
# 3 MCSE of the target.

# %%
assert (results[1.10].events_closed_form, results[1.10].n_closed_form) == (864.0, 1081)
assert (results[1.15].events_closed_form, results[1.15].n_closed_form) == (401.8, 503)
if N_SIMS >= 1000:
    for res in results.values():
        gap = abs(res.confirmed_power.value - TARGET) / res.confirmed_power.mcse
        assert gap <= 3, "confirmed power is more than 3 MCSE from the target"

# %% [markdown]
# ## What would change this
#
# - **Large effects.** With `c_index=0.70` the closed form gives N = 14, far too few.
#   Use `method="auto"`, which scans N directly below 100 patients. See
#   [Effect sizes](../concepts/effect-sizes.md).
# - **Censoring** enters through P(event): N scales with 1 / (1 − censoring).
# - **Adjustment.** Correlated covariates inflate N by 1 / (1 − R²); set
#   `n_covariates` and `r2` on the design.
