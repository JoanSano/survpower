# %% [markdown]
# # Continuous vs. median-split analysis
#
# **Question.** A biomarker is measured on a continuous scale. How much power is
# lost if the analysis compares patients above and below the median instead?
#
# **Design assumptions.** X ~ N(0, 1), exponential survival with a median of
# 15 months, 20% independent censoring, two-sided α = 0.05. Both analyses run on
# the *same* simulated cohorts, so the difference is not Monte Carlo noise between
# separate runs.

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from survpower.closed_form import dichotomized_power, theoretical_power_with_r2
from survpower.simulate import simulate_continuous_vs_dichotomized

# Replicates per cell. The docs build uses the default; CI smoke-tests with 50.
N_SIMS = int(os.environ.get("SURVPOWER_TUTORIAL_SIMS", 1000))
CENSOR_RATE = 0.20

# %% [markdown]
# ## Simulate

# %%
rows = []
for n in (1200, 250):
    for hr in (1.10, 1.15):
        r = simulate_continuous_vs_dichotomized(n=n, hr=hr, censor_rate=CENSOR_RATE,
                                                n_sims=N_SIMS, seed=42)
        rows.append(dict(
            n=n, hr_per_sd=hr, events=r["mean_events"],
            power_cont=r["power_cont"], power_cont_mcse=r["power_cont_mcse"],
            power_dich=r["power_dich"], power_dich_mcse=r["power_dich_mcse"],
            schoenfeld=round(dichotomized_power(n, hr, CENSOR_RATE, 0.05, 2,
                                                method="schoenfeld"), 3),
            group_hr=r["group_hr_fitted"], group_hr_mcse=r["group_hr_fitted_mcse"],
        ))
table = pd.DataFrame(rows)
table

# %% [markdown]
# Every power has its Monte Carlo standard error (`*_mcse`). `schoenfeld` is the
# closed-form power of the median-split comparison, and `group_hr` is the fitted
# high-vs-low hazard ratio.

# %%
BLUE, ORANGE = "#2a78d6", "#eb6834"
labels = [f"N = {n}, HR {hr:.2f}" for n, hr in zip(table["n"], table["hr_per_sd"])]
y = np.arange(len(table))[::-1]

fig, ax = plt.subplots(figsize=(7, 3.2))
for col, color, marker, name, shift in (("cont", BLUE, "o", "Continuous", 0.12),
                                        ("dich", ORANGE, "s", "Median split", -0.12)):
    ax.errorbar(table[f"power_{col}"], y + shift, xerr=1.96 * table[f"power_{col}_mcse"],
                fmt=marker, color=color, markersize=7, capsize=0, linewidth=2, label=name)
ax.set_yticks(y, labels)
ax.set_xlim(0, 1)
ax.set_xlabel("Simulated power (95% Monte Carlo interval)")
ax.grid(axis="x", color="#e5e4e0", linewidth=1)
ax.set_axisbelow(True)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.legend(frameon=False, loc="lower right")
ax.set_title("A median split loses power in every cell", loc="left")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Interpretation
#
# A median split keeps only a fraction of the information in the biomarker. For a
# normal covariate and a small effect, its asymptotic relative efficiency is
# 2/π ≈ 0.64: a median-split analysis needs about 1/0.64 ≈ 1.57 times the events
# for the same power. The fitted group HR looks larger than the per-SD HR because
# it compares two groups whose means are about 1.6 SD apart, not because the
# analysis gained anything.

# %%
print(f"Asymptotic relative efficiency 2/pi = {2 / np.pi:.3f}")
print(f"Events multiplier pi/2 = {np.pi / 2:.3f}")

# %% [markdown]
# ## Agreement with the closed forms
#
# At N = 1200, HR = 1.10, the simulated continuous power should match the closed form
# for the Wald test, and the median-split power should match Schoenfeld's closed form.
# At the default budget this script fails if either differs by more than 3 MCSE.

# %%
CLOSED_CONT = theoretical_power_with_r2(n=1200, hr=1.10, censor_rate=0.20, r2_target=0.0)
CLOSED_DICH = dichotomized_power(1200, 1.10, 0.20, method="schoenfeld")
cell = table[(table["n"] == 1200) & (table["hr_per_sd"] == 1.10)].iloc[0]
gap_cont = abs(cell["power_cont"] - CLOSED_CONT) / cell["power_cont_mcse"]
gap_dich = abs(cell["power_dich"] - CLOSED_DICH) / cell["power_dich_mcse"]
print(f"Continuous:   {cell['power_cont']:.3f} (MCSE {cell['power_cont_mcse']:.3f}), "
      f"closed form {CLOSED_CONT:.3f}: {gap_cont:.1f} MCSE apart")
print(f"Median split: {cell['power_dich']:.3f} (MCSE {cell['power_dich_mcse']:.3f}), "
      f"Schoenfeld {CLOSED_DICH:.3f}: {gap_dich:.1f} MCSE apart")
if N_SIMS >= 1000:
    assert gap_cont <= 3 and gap_dich <= 3, "simulation contradicts the closed forms"

# %% [markdown]
# ## What would change this
#
# - **More censoring** reduces events, and power falls in both analyses.
# - **A skewed biomarker** changes the 2/π figure, which assumes normality.
# - **A cut-point chosen from the data** is worse than a pre-specified median
#   split: it inflates the type-I error unless the search is corrected for.
