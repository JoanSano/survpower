"""Monte Carlo power engines (report Sections 4, 9, 10 and Appendix A).

Every engine: draws a cohort from a known data-generating process, applies
exponential censoring calibrated to a target marginal rate, fits a Cox model
and records the Wald test on the biomarker X. Seeds make every result exactly
reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines.statistics import proportional_hazard_test

from ._utils import tqdm
from .baseline import DEFAULT_LAMBDA0
from .censoring import (
    calibrate_censoring_crossing,
    calibrate_censoring_rate,
    calibrate_censoring_rate_heterogeneous,
    calibrate_censoring_waning,
)
from .dgp import gen_adjustment_covariates, gen_crossing, gen_waning
from .fitting import ENGINES, cox_fit_1d, fit_cox, rejects, wald_rejects


def _censor(T: np.ndarray, lam_c: float, rng: np.random.Generator):
    C = rng.exponential(scale=1.0 / lam_c, size=len(T))
    return np.minimum(T, C), (T <= C).astype(int)


def _design_frame(time_, event, X, Z=None) -> pd.DataFrame:
    cols = {"time": time_, "event": event, "X": X}
    if Z is not None:
        for j in range(Z.shape[1]):
            cols[f"Z{j}"] = Z[:, j]
    return pd.DataFrame(cols)


def simulate_power(
    n: int,
    hr: float,
    censor_rate: float,
    lambda0: float = DEFAULT_LAMBDA0,
    n_sims: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    sides: int = 2,
    engine: str = "fast",
) -> tuple[float, float, float]:
    """Monte Carlo power for a continuous covariate in a univariate Cox model.

    Returns (power, mean_events, lambda_c).
    """
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}")
    beta = np.log(hr)
    lam_c = calibrate_censoring_rate(beta, censor_rate, lambda0=lambda0, seed=seed + 999)
    rng = np.random.default_rng(seed)
    rejections, events = 0, []
    for _ in tqdm(range(n_sims), desc=f"continuous n={n} HR={hr:.3g}", leave=False):
        X = rng.standard_normal(n)
        T = rng.exponential(scale=1.0 / (lambda0 * np.exp(beta * X)))
        time_, event = _censor(T, lam_c, rng)
        events.append(event.sum())
        if engine == "fast":
            b, se = cox_fit_1d(time_, event, X)
            rejections += wald_rejects(b / se, beta, alpha, sides)
        else:
            cph, _ = fit_cox(_design_frame(time_, event, X))
            rejections += rejects(cph, beta, alpha, sides)
    return rejections / n_sims, float(np.mean(events)), lam_c


def simulate_power_with_adjustment(
    n: int,
    hr: float,
    censor_rate: float,
    p: int,
    r2_target: float,
    lambda0: float = DEFAULT_LAMBDA0,
    n_sims: int = 300,
    alpha: float = 0.05,
    seed: int = 0,
    sides: int = 2,
) -> dict:
    """Power for X in a Cox model adjusted for p null covariates whose
    multiple correlation with X is r2_target (Section 9). The true hazard
    depends only on X."""
    beta = np.log(hr)
    lam_c = calibrate_censoring_rate(beta, censor_rate, lambda0=lambda0, seed=seed + 999)
    rng = np.random.default_rng(seed)
    rejections, events, beta_hats = 0, [], []
    for _ in tqdm(range(n_sims), desc=f"p={p} R2={r2_target}", leave=False):
        X, Z = gen_adjustment_covariates(n, r2_target, p, rng)
        T = rng.exponential(scale=1.0 / (lambda0 * np.exp(beta * X)))
        time_, event = _censor(T, lam_c, rng)
        events.append(event.sum())
        cph, _ = fit_cox(_design_frame(time_, event, X, Z))
        rejections += rejects(cph, beta, alpha, sides)
        beta_hats.append(cph.summary.loc["X", "coef"])
    return dict(
        p=p, r2_target=r2_target,
        power=round(rejections / n_sims, 3),
        mean_events=round(float(np.mean(events)), 1),
        mean_HR_hat=round(float(np.exp(np.mean(beta_hats))), 3),
    )


def simulate_power_design(
    n: int,
    hr: float,
    censor_rate: float,
    p: int = 0,
    r2_target: float = 0.0,
    lambda0: float = DEFAULT_LAMBDA0,
    n_sims: int = 1000,
    alpha: float = 0.05,
    sides: int = 2,
    seed: int = 0,
    engine: str = "fast",
) -> tuple[float, float]:
    """Power for X, unadjusted (p = 0, `simulate_power` with `engine`) or
    adjusted (`simulate_power_with_adjustment`, lifelines). Returns
    (power, mean_events)."""
    if p == 0:
        power, mean_ev, _ = simulate_power(
            n=n, hr=hr, censor_rate=censor_rate, lambda0=lambda0, n_sims=n_sims,
            alpha=alpha, sides=sides, seed=seed, engine=engine,
        )
        return power, mean_ev
    out = simulate_power_with_adjustment(
        n=n, hr=hr, censor_rate=censor_rate, p=p, r2_target=r2_target,
        lambda0=lambda0, n_sims=n_sims, alpha=alpha, sides=sides, seed=seed,
    )
    return out["power"], out["mean_events"]


def simulate_epv_diagnostic(
    n: int,
    hr: float,
    censor_rate: float,
    p: int,
    r2_target: float,
    lambda0: float = DEFAULT_LAMBDA0,
    n_sims: int = 300,
    alpha: float = 0.05,
    seed: int = 0,
    sides: int = 2,
) -> dict:
    """Like `simulate_power_with_adjustment`, but also tracks bias (mean /
    median recovered HR), estimation variance (SD of log-HR) and numerical
    instability -- the diagnostics that degrade at low events-per-variable
    even when power does not (Section 9.5)."""
    beta = np.log(hr)
    lam_c = calibrate_censoring_rate(beta, censor_rate, lambda0=lambda0, seed=seed + 999)
    rng = np.random.default_rng(seed)
    rejections, events, beta_hats, n_unstable = 0, [], [], 0
    for _ in tqdm(range(n_sims), desc=f"EPV probe p={p}", leave=False):
        X, Z = gen_adjustment_covariates(n, r2_target, p, rng)
        T = rng.exponential(scale=1.0 / (lambda0 * np.exp(beta * X)))
        time_, event = _censor(T, lam_c, rng)
        events.append(event.sum())
        cph, unstable = fit_cox(_design_frame(time_, event, X, Z))
        coef, se = cph.summary.loc["X", "coef"], cph.summary.loc["X", "se(coef)"]
        if se > 5 or abs(coef) > 5:
            unstable = True
        n_unstable += int(unstable)
        rejections += rejects(cph, beta, alpha, sides)
        beta_hats.append(coef)
    beta_hats = np.array(beta_hats)
    return dict(
        p=p, r2_target=r2_target, n=n,
        epv=round(float(np.mean(events)) / (p + 1), 2),
        mean_events=round(float(np.mean(events)), 1),
        power=round(rejections / n_sims, 3),
        mean_HR_hat=round(float(np.exp(np.mean(beta_hats))), 3),
        median_HR_hat=round(float(np.exp(np.median(beta_hats))), 3),
        sd_beta_hat=round(float(np.std(beta_hats)), 3),
        frac_unstable=round(n_unstable / n_sims, 3),
    )


def simulate_power_with_heterogeneous_adjustment(
    n: int,
    hr: float,
    censor_rate: float,
    p: int,
    r2_target: float,
    sigma_z: float = 0.0,
    lambda0: float = DEFAULT_LAMBDA0,
    n_sims: int = 300,
    alpha: float = 0.05,
    seed: int = 0,
    sides: int = 2,
) -> dict:
    """Power for X adjusted for p covariates that are themselves prognostic,
    each with log-HR ~ N(0, sigma_z^2) redrawn every replicate (Section 10).
    sigma_z = 0 recovers the null-adjustment case."""
    beta = np.log(hr)
    lam_c = calibrate_censoring_rate_heterogeneous(
        beta, p, r2_target, sigma_z, censor_rate, lambda0=lambda0, seed=seed + 999
    )
    rng = np.random.default_rng(seed)
    rejections, events, beta_hats, realized_abs_betaz = 0, [], [], []
    for _ in tqdm(range(n_sims), desc=f"p={p} sigma_z={sigma_z}", leave=False):
        X, Z = gen_adjustment_covariates(n, r2_target, p, rng)
        if p == 0:
            lin_pred = beta * X
        else:
            beta_z_vec = rng.normal(0.0, sigma_z, size=p)
            lin_pred = beta * X + Z @ beta_z_vec
            realized_abs_betaz.extend(np.abs(beta_z_vec).tolist())
        T = rng.exponential(scale=1.0 / (lambda0 * np.exp(lin_pred)))
        time_, event = _censor(T, lam_c, rng)
        events.append(event.sum())
        cph, _ = fit_cox(_design_frame(time_, event, X, Z))
        rejections += rejects(cph, beta, alpha, sides)
        beta_hats.append(cph.summary.loc["X", "coef"])
    return dict(
        p=p, r2_target=r2_target, sigma_z=sigma_z, n=n, n_sims=n_sims,
        power=round(rejections / n_sims, 3),
        mean_events=round(float(np.mean(events)), 1),
        mean_HR_hat=round(float(np.exp(np.mean(beta_hats))), 3),
        median_abs_HRz_realized=(
            round(float(np.exp(np.median(realized_abs_betaz))), 3) if realized_abs_betaz else 1.0
        ),
    )


def simulate_waning_effect(
    n: int,
    hr_early: float,
    tau: float,
    censor_rate: float,
    lambda0: float = DEFAULT_LAMBDA0,
    n_sims: int = 400,
    alpha: float = 0.05,
    seed: int = 0,
    sides: int = 2,
) -> dict:
    """Waning effect (HR = hr_early before tau, 1 after) analysed with a
    constant-HR Cox model: naive power and bias, plus the Grambsch-Therneau
    test's power to detect the violation (Appendix A.3)."""
    beta0 = np.log(hr_early)
    lam_c = calibrate_censoring_waning(beta0, tau, censor_rate, lambda0=lambda0, seed=seed + 999)
    rng = np.random.default_rng(seed)
    rejections_naive, rejections_zph, beta_hats, events = 0, 0, [], []
    for _ in tqdm(range(n_sims), desc=f"waning n={n}", leave=False):
        T, X = gen_waning(n, beta0, tau, lambda0, rng)
        time_, event = _censor(T, lam_c, rng)
        events.append(event.sum())
        df = _design_frame(time_, event, X)
        cph, _ = fit_cox(df)
        rejections_naive += rejects(cph, beta0, alpha, sides)
        beta_hats.append(cph.summary.loc["X", "coef"])
        zph = proportional_hazard_test(cph, df, time_transform="rank")
        rejections_zph += zph.summary.loc["X", "p"] < alpha
    return dict(
        n=n, hr_early=hr_early, tau=tau,
        power_naive=round(rejections_naive / n_sims, 3),
        power_zph_detect=round(rejections_zph / n_sims, 3),
        mean_HR_hat=round(float(np.exp(np.mean(beta_hats))), 3),
        mean_events=round(float(np.mean(events)), 1),
    )


def simulate_crossing_hazards(
    n: int,
    hr_early: float,
    hr_late: float,
    tau: float,
    censor_rate: float,
    lambda0: float = DEFAULT_LAMBDA0,
    n_sims: int = 400,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Crossing hazards (HR = hr_early before tau, hr_late after), analysed
    with a constant-HR Cox model (Appendix A.4). Always two-sided: the effect
    has no single direction."""
    beta_early, beta_late = np.log(hr_early), np.log(hr_late)
    lam_c = calibrate_censoring_crossing(beta_early, beta_late, tau, censor_rate,
                                         lambda0=lambda0, seed=seed + 999)
    rng = np.random.default_rng(seed)
    rejections_naive, rejections_zph, beta_hats, events = 0, 0, [], []
    for _ in tqdm(range(n_sims), desc=f"crossing n={n}", leave=False):
        T, X = gen_crossing(n, beta_early, beta_late, tau, lambda0, rng)
        time_, event = _censor(T, lam_c, rng)
        events.append(event.sum())
        df = _design_frame(time_, event, X)
        cph, _ = fit_cox(df)
        rejections_naive += cph.summary.loc["X", "p"] < alpha
        beta_hats.append(cph.summary.loc["X", "coef"])
        zph = proportional_hazard_test(cph, df, time_transform="rank")
        rejections_zph += zph.summary.loc["X", "p"] < alpha
    return dict(
        n=n, hr_early=hr_early, hr_late=hr_late, tau=tau,
        power_naive=round(rejections_naive / n_sims, 3),
        power_zph_detect=round(rejections_zph / n_sims, 3),
        mean_HR_hat=round(float(np.exp(np.mean(beta_hats))), 3),
        mean_events=round(float(np.mean(events)), 1),
    )
