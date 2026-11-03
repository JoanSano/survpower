"""Independent censoring and its calibration (report Section 3).

Censoring is C ~ Exponential(lambda_c), independent of T. lambda_c is chosen
so that the marginal censoring probability P(C < T) hits a target. Because
P(C < T) is strictly increasing in lambda_c, bisection on a large Monte Carlo
probe sample converges to the unique root.
"""

from __future__ import annotations

import numpy as np

from .baseline import DEFAULT_LAMBDA0
from .dgp import gen_adjustment_covariates, gen_crossing, gen_exponential_ph, gen_waning


def calibrate_censoring(
    T: np.ndarray,
    target_censor_rate: float,
    rng: np.random.Generator,
    lo: float = 1e-6,
    hi: float = 50.0,
    n_iter: int = 50,
) -> float:
    """Exponential censoring rate lambda_c giving P(C < T) = target over the
    probe sample of event times `T`, by bisection (fresh C each iteration)."""
    n = len(T)
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        C = rng.exponential(scale=1.0 / mid, size=n)
        lo, hi = (mid, hi) if np.mean(C < T) < target_censor_rate else (lo, mid)
    return (lo + hi) / 2


def calibrate_censoring_rate(
    beta: float,
    target_censor_rate: float,
    lambda0: float = DEFAULT_LAMBDA0,
    n_probe: int = 500_000,
    seed: int = 1,
) -> float:
    """lambda_c for the exponential PH model with per-SD log-HR `beta`."""
    rng = np.random.default_rng(seed)
    T, _ = gen_exponential_ph(n_probe, beta, lambda0, rng)
    return calibrate_censoring(T, target_censor_rate, rng)


def calibrate_censoring_rate_heterogeneous(
    beta: float, p: int, r2_target: float, sigma_z: float,
    target_censor_rate: float, lambda0: float = DEFAULT_LAMBDA0,
    n_probe: int = 300_000, n_chunks: int = 50, seed: int = 1,
) -> float:
    """lambda_c for the full linear predictor beta*X + Z @ beta_z, with a fresh
    beta_z ~ N(0, sigma_z^2)^p per chunk so the calibration marginalizes over
    the distribution of adjustment-covariate effect sizes (Section 10)."""
    rng = np.random.default_rng(seed)
    chunk = max(n_probe // n_chunks, 1000)
    T_parts = []
    for _ in range(n_chunks):
        X, Z = gen_adjustment_covariates(chunk, r2_target, p, rng)
        if p == 0:
            lin_pred = beta * X
        else:
            beta_z_vec = rng.normal(0.0, sigma_z, size=p)
            lin_pred = beta * X + Z @ beta_z_vec
        T_parts.append(rng.exponential(scale=1.0 / (lambda0 * np.exp(lin_pred))))
    return calibrate_censoring(np.concatenate(T_parts), target_censor_rate, rng)


def calibrate_censoring_waning(
    beta0: float, tau: float, target_censor_rate: float,
    lambda0: float = DEFAULT_LAMBDA0, n_probe: int = 300_000, seed: int = 1,
) -> float:
    """lambda_c for the waning-effect generator (Appendix A.3)."""
    rng = np.random.default_rng(seed)
    T, _ = gen_waning(n_probe, beta0, tau, lambda0, rng)
    return calibrate_censoring(T, target_censor_rate, rng)


def calibrate_censoring_crossing(
    beta_early: float, beta_late: float, tau: float, target_censor_rate: float,
    lambda0: float = DEFAULT_LAMBDA0, n_probe: int = 300_000, seed: int = 1,
) -> float:
    """lambda_c for the crossing-hazards generator (Appendix A.4)."""
    rng = np.random.default_rng(seed)
    T, _ = gen_crossing(n_probe, beta_early, beta_late, tau, lambda0, rng)
    return calibrate_censoring(T, target_censor_rate, rng)


def censor_rate_from_design(
    hr: float,
    accrual_months: float,
    followup_months: float,
    dropout_annual: float = 0.0,
    lambda0: float = DEFAULT_LAMBDA0,
    n_probe: int = 500_000,
    seed: int = 1,
) -> float:
    """Marginal censoring probability implied by a trial calendar.

    Entry is uniform over [0, accrual_months] and the analysis happens
    `followup_months` after the last patient is enrolled, so each patient's
    administrative censoring time is C_adm ~ U(followup, accrual + followup).
    Optional exponential dropout has annual loss fraction `dropout_annual`.

    The simulation engines then use exponential censoring calibrated to this
    rate. Cox power is driven almost entirely by the number of events, so
    matching the event probability is what matters; the exact shape of the
    censoring distribution is second order.
    """
    rng = np.random.default_rng(seed)
    T, _ = gen_exponential_ph(n_probe, np.log(hr), lambda0, rng)
    C = rng.uniform(followup_months, accrual_months + followup_months, size=n_probe)
    if dropout_annual > 0:
        lam_d = -np.log(1 - dropout_annual) / 12.0
        C = np.minimum(C, rng.exponential(scale=1.0 / lam_d, size=n_probe))
    return float(np.mean(C < T))
