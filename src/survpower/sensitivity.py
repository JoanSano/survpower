"""Sensitivity analyses at a planning point (report Sections 9-10, Appendix A).

These need a concrete (N, HR): the planning CLI runs them at the supplied
pair, or at the solved one.

* collinearity  -- power vs. number of adjustment covariates p at fixed R^2
* epv           -- bias / variance of the HR at single-digit events-per-variable
* heterogeneous -- adjustment covariates that are themselves prognostic
* omitted       -- what leaving prognostic covariates out of the model costs
* waning        -- effect present only before tau
* crossing      -- effect reverses direction at tau
"""

from __future__ import annotations

import os

import pandas as pd

from .baseline import DEFAULT_LAMBDA0, DEFAULT_TAU_MONTHS
from .closed_form import theoretical_power_with_r2
from .design import StudyDesign
from .simulate import (
    simulate_adjusted_vs_unadjusted,
    simulate_crossing_hazards,
    simulate_epv_diagnostic,
    simulate_power_with_adjustment,
    simulate_power_with_heterogeneous_adjustment,
    simulate_waning_effect,
)

SENSITIVITY_CHOICES = ("collinearity", "epv", "heterogeneous", "omitted", "waning", "crossing")


def run_collinearity_study(
    n: int = 1200,
    hr: float = 1.10,
    censor_rate: float = 0.20,
    r2_grid: tuple = (0.0, 0.3, 0.6),
    p_grid: tuple = (0, 1, 2, 5, 10, 20),
    lambda0: float = DEFAULT_LAMBDA0,
    n_sims: int = 300,
    seed: int = 0,
    alpha: float = 0.05,
    sides: int = 2,
) -> pd.DataFrame:
    """Simulated power and mean recovered HR across (R^2, p), with the
    p-independent closed-form prediction (report Section 9, Figure 2)."""
    rows = []
    combo_id = 0
    for r2 in r2_grid:
        for p in p_grid:
            out = simulate_power_with_adjustment(
                n=n, hr=hr, censor_rate=censor_rate, p=p, r2_target=r2,
                lambda0=lambda0, n_sims=n_sims, alpha=alpha, sides=sides,
                seed=seed + 137 * combo_id,
            )
            out["n"] = n
            # at p = 0 nothing is adjusted for, so the reference is R^2 = 0
            r2_for_theory = r2 if p > 0 else 0.0
            out["theoretical_power"] = round(theoretical_power_with_r2(
                n=n, hr=hr, censor_rate=censor_rate, r2_target=r2_for_theory,
                alpha=alpha, sides=sides), 3)
            rows.append(out)
            combo_id += 1
    cols = ["n", "r2_target", "p", "mean_events", "power", "power_mcse", "theoretical_power",
            "mean_HR_hat", "mean_HR_hat_mcse", "bias_beta", "bias_beta_mcse", "emp_se",
            "emp_se_mcse", "mean_model_se", "coverage", "coverage_mcse", "n_sims", "n_failed"]
    return pd.DataFrame(rows)[cols]


def run_collinearity_study_both_cohorts(n_grid: tuple = (1200, 250), seed: int = 0,
                                        **kwargs) -> pd.DataFrame:
    """`run_collinearity_study` at each cohort size, concatenated."""
    return pd.concat(
        [run_collinearity_study(n=n, seed=seed + 10_000 * i, **kwargs)
         for i, n in enumerate(n_grid)],
        ignore_index=True,
    )


def run_epv_stress_test(
    n: int = 250,
    hr: float = 1.10,
    censor_rate: float = 0.20,
    r2_target: float = 0.3,
    p_grid: tuple = (0, 5, 10, 20, 40, 60, 80),
    lambda0: float = DEFAULT_LAMBDA0,
    n_sims: int = 1000,
    seed: int = 10_000,
    alpha: float = 0.05,
    sides: int = 2,
) -> pd.DataFrame:
    """Bias (log-HR scale), SD vs. model SE, nominal power, type-I error and
    size-adjusted power for X as p grows, pushing events-per-variable into
    single digits (report Section 9.5, Table 3, Figure 3)."""
    rows = [simulate_epv_diagnostic(
        n=n, hr=hr, censor_rate=censor_rate, p=p, r2_target=r2_target, lambda0=lambda0,
        n_sims=n_sims, alpha=alpha, sides=sides, seed=seed + 1000 * i,
    ) for i, p in enumerate(p_grid)]
    cols = ["p", "epv", "mean_events", "power", "power_mcse", "type1_error", "type1_error_mcse",
            "size_adj_power", "size_adj_power_mcse", "bias_beta", "bias_beta_mcse",
            "median_HR_hat", "mean_HR_hat", "mean_HR_hat_mcse", "sd_beta_hat", "sd_beta_hat_mcse",
            "mean_se", "mean_se_mcse", "frac_unstable", "n_sims"]
    return pd.DataFrame(rows)[cols]


def run_omitted_covariates_study(
    n: int = 1200,
    hr: float = 1.10,
    censor_rate: float = 0.20,
    p: int = 5,
    sigma_z_grid: tuple = (0.0, 0.3, 0.6, 1.0),
    lambda0: float = DEFAULT_LAMBDA0,
    n_sims: int = 1000,
    seed: int = 20_000,
    alpha: float = 0.05,
    sides: int = 2,
) -> pd.DataFrame:
    """Adjusted vs. unadjusted analysis when p prognostic covariates are
    independent of X (report Section 10.4, Table 5): omitting them
    attenuates the HR (non-collapsibility) and costs power."""
    return pd.DataFrame([simulate_adjusted_vs_unadjusted(
        n=n, hr=hr, censor_rate=censor_rate, p=p, sigma_z=s, lambda0=lambda0, n_sims=n_sims,
        alpha=alpha, sides=sides, seed=seed + 137 * i,
    ) for i, s in enumerate(sigma_z_grid)])


def run_heterogeneous_adjustment_study(
    n: int = 1200,
    hr: float = 1.10,
    censor_rate: float = 0.20,
    r2_target: float = 0.3,
    p_grid: tuple = (5, 20),
    sigma_z_grid: tuple = (0.0, 0.3, 0.6),
    lambda0: float = DEFAULT_LAMBDA0,
    n_sims: int = 300,
    seed: int = 0,
    alpha: float = 0.05,
    sides: int = 2,
) -> pd.DataFrame:
    """Power for X as the adjustment covariates' own effects become more
    heterogeneous (report Section 10, Figure 4)."""
    rows = []
    combo = 0
    for p in p_grid:
        for sigma_z in sigma_z_grid:
            rows.append(simulate_power_with_heterogeneous_adjustment(
                n=n, hr=hr, censor_rate=censor_rate, p=p, r2_target=r2_target,
                sigma_z=sigma_z, lambda0=lambda0, n_sims=n_sims, alpha=alpha,
                sides=sides, seed=seed + 137 * combo,
            ))
            combo += 1
    df = pd.DataFrame(rows)
    df["theoretical_power"] = round(theoretical_power_with_r2(
        n=n, hr=hr, censor_rate=censor_rate, r2_target=r2_target, alpha=alpha, sides=sides), 3)
    cols = ["p", "r2_target", "sigma_z", "n", "mean_events", "power", "power_mcse",
            "theoretical_power", "mean_HR_hat", "mean_HR_hat_mcse", "bias_beta", "bias_beta_mcse",
            "emp_se", "emp_se_mcse", "coverage", "coverage_mcse", "median_abs_HRz_realized",
            "n_sims", "n_failed"]
    return df[cols]


def run_sensitivity(
    design: StudyDesign,
    n: int,
    hr: float,
    which: list,
    tau: float = DEFAULT_TAU_MONTHS,
    waning_hr: float | None = None,
    crossing_hr: tuple | None = None,
    n_sims: int = 300,
    outdir: str = ".",
    tag: str = "",
    plots: bool = True,
) -> dict:
    """Run the requested analyses at one planning point (evaluable n,
    per-SD hr). Waning defaults to an early-only effect of the planning HR;
    crossing to 1/HR before tau and HR after. Returns {name: DataFrame} and
    writes figures to `outdir` if `plots`."""
    from . import plotting  # matplotlib only when needed

    design.require_default_models("sensitivity analyses")
    cr = design.censor_rate_for(hr)
    r2 = design.r2 if design.r2 > 0 else 0.3
    common = dict(n=n, hr=hr, censor_rate=cr, lambda0=design.lambda0, n_sims=n_sims,
                  alpha=design.alpha, sides=design.sides)
    results = {}
    if "collinearity" in which:
        df = run_collinearity_study(
            r2_grid=tuple(sorted({0.0, 0.3, 0.6, design.r2_eff})),
            p_grid=tuple(sorted({0, 1, 2, 5, 10, 20, design.n_covariates})),
            seed=design.seed, **common,
        )
        results["collinearity"] = df
        if plots:
            plotting.plot_collinearity_power(df, outpath=os.path.join(outdir, f"collinearity_power_{tag}.pdf"))
    if "epv" in which:
        df = run_epv_stress_test(r2_target=r2, seed=design.seed + 10_000, **common)
        results["epv"] = df
        if plots:
            plotting.plot_epv_bias(df, true_hr=round(hr, 3), outpath=os.path.join(outdir, f"epv_bias_{tag}.pdf"))
    if "heterogeneous" in which:
        p_grid = {5, 20} | ({design.n_covariates} if design.n_covariates > 0 else set())
        df = run_heterogeneous_adjustment_study(r2_target=r2, p_grid=tuple(sorted(p_grid)),
                                                seed=design.seed, **common)
        results["heterogeneous"] = df
        if plots:
            plotting.plot_heterogeneous_adjustment(
                df, outpath=os.path.join(outdir, f"informative_adjustment_{tag}.pdf"))
    if "omitted" in which:
        results["omitted"] = run_omitted_covariates_study(
            p=max(design.n_covariates, 5), seed=design.seed + 20_000, **common)
    if "waning" in which:
        out = simulate_waning_effect(
            n=n, hr_early=waning_hr or hr, tau=tau, censor_rate=cr, lambda0=design.lambda0,
            n_sims=n_sims, alpha=design.alpha, sides=design.sides, seed=design.seed + 7,
        )
        results["waning"] = pd.DataFrame([out])
    if "crossing" in which:
        hr_early, hr_late = crossing_hr or (1 / hr, hr)
        out = simulate_crossing_hazards(
            n=n, hr_early=hr_early, hr_late=hr_late, tau=tau, censor_rate=cr,
            lambda0=design.lambda0, n_sims=n_sims, alpha=design.alpha, seed=design.seed + 11,
        )
        results["crossing"] = pd.DataFrame([out])
    return results
