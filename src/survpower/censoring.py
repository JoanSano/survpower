"""Independent censoring and its calibration.

Censoring is C ~ Exponential(lambda_c), independent of T. lambda_c is chosen
so that the marginal censoring probability P(C < T) hits a target. Because
P(C < T) is strictly increasing in lambda_c, bisection on a large Monte Carlo
probe sample converges to the unique root.
"""

from __future__ import annotations

import numpy as np

from .baseline import DEFAULT_LAMBDA0, DEFAULT_MEDIAN_OS_MONTHS, median_os_to_lambda0
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


def calibrate_censoring_for(
    sampler,
    target_censor_rate: float,
    seed: int = 1,
    n_probe: int = 500_000,
) -> float:
    """lambda_c for any event-time generator: draws a probe sample
    ``T = sampler(n_probe, rng)`` from ``default_rng(seed)`` and bisects on it
    with the same generator (`calibrate_censoring`)."""
    rng = np.random.default_rng(seed)
    return calibrate_censoring(np.asarray(sampler(n_probe, rng)), target_censor_rate, rng)


def calibrate_censoring_rate(
    beta: float,
    target_censor_rate: float,
    lambda0: float = DEFAULT_LAMBDA0,
    n_probe: int = 500_000,
    seed: int = 1,
) -> float:
    """lambda_c for the exponential PH model with per-SD log-HR `beta`."""
    return calibrate_censoring_for(lambda n, rng: gen_exponential_ph(n, beta, lambda0, rng)[0],
                                   target_censor_rate, seed=seed, n_probe=n_probe)


def calibrate_censoring_rate_heterogeneous(
    beta: float, p: int, r2_target: float, sigma_z: float,
    target_censor_rate: float, lambda0: float = DEFAULT_LAMBDA0,
    n_probe: int = 300_000, n_chunks: int = 50, seed: int = 1,
) -> float:
    """lambda_c for the full linear predictor beta*X + Z @ beta_z, with a fresh
    beta_z ~ N(0, sigma_z^2)^p per chunk so the calibration marginalizes over
    the distribution of adjustment-covariate effect sizes."""
    def sampler(n, rng):
        chunk = max(n // n_chunks, 1000)
        T_parts = []
        for _ in range(n_chunks):
            X, Z = gen_adjustment_covariates(chunk, r2_target, p, rng)
            if p == 0:
                lin_pred = beta * X
            else:
                beta_z_vec = rng.normal(0.0, sigma_z, size=p)
                lin_pred = beta * X + Z @ beta_z_vec
            T_parts.append(rng.exponential(scale=1.0 / (lambda0 * np.exp(lin_pred))))
        return np.concatenate(T_parts)
    return calibrate_censoring_for(sampler, target_censor_rate, seed=seed, n_probe=n_probe)


def calibrate_censoring_waning(
    beta0: float, tau: float, target_censor_rate: float,
    lambda0: float = DEFAULT_LAMBDA0, n_probe: int = 300_000, seed: int = 1,
) -> float:
    """lambda_c for the waning-effect generator."""
    return calibrate_censoring_for(lambda n, rng: gen_waning(n, beta0, tau, lambda0, rng)[0],
                                   target_censor_rate, seed=seed, n_probe=n_probe)


def calibrate_censoring_crossing(
    beta_early: float, beta_late: float, tau: float, target_censor_rate: float,
    lambda0: float = DEFAULT_LAMBDA0, n_probe: int = 300_000, seed: int = 1,
) -> float:
    """lambda_c for the crossing-hazards generator."""
    return calibrate_censoring_for(
        lambda n, rng: gen_crossing(n, beta_early, beta_late, tau, lambda0, rng)[0],
        target_censor_rate, seed=seed, n_probe=n_probe)


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
    censoring distribution is second order. In closed form this
    rate is 1 - `event_probability_exponential`.
    """
    rng = np.random.default_rng(seed)
    T, _ = gen_exponential_ph(n_probe, np.log(hr), lambda0, rng)
    C = rng.uniform(followup_months, accrual_months + followup_months, size=n_probe)
    if dropout_annual > 0:
        lam_d = -np.log(1 - dropout_annual) / 12.0
        C = np.minimum(C, rng.exponential(scale=1.0 / lam_d, size=n_probe))
    return float(np.mean(C < T))


def event_probability_exponential(
    accrual_months: float,
    followup_months: float,
    dropout_annual: float = 0.0,
    median_os: float = DEFAULT_MEDIAN_OS_MONTHS,
    hr_per_sd: float = 1.0,
    n_nodes: int = 64,
) -> float:
    """P(event) under exponential survival, uniform accrual, exponential dropout;
    marginalized over X ~ N(0, 1) by Gauss-Hermite quadrature.

    Closed-form counterpart of ``1 - censor_rate_from_design(...)`` (same
    calendar and dropout parametrization). For a single hazard lambda with
    dropout hazard delta = -ln(1 - dropout_annual) / 12 and r = lambda + delta,

        P(event | lambda) = (lambda / r) [1 - (e^{-rF} - e^{-r(A+F)}) / (rA)],

    with the A -> 0 limit (lambda / r)(1 - e^{-rF}); this is Lachin & Foulkes
    (1986, eq. 4.1) with R = A and T = A + F. The subject hazard is
    lambda0 * hr_per_sd ** X.
    """
    A, F = float(accrual_months), float(followup_months)
    if A < 0 or F < 0 or A + F <= 0:
        raise ValueError("need accrual_months >= 0, followup_months >= 0 and A + F > 0")
    if not 0 <= dropout_annual < 1:
        raise ValueError("dropout_annual must be in [0, 1)")
    delta = -np.log1p(-dropout_annual) / 12.0
    x, w = np.polynomial.hermite_e.hermegauss(n_nodes)  # weight exp(-x^2 / 2)
    lam = median_os_to_lambda0(median_os) * np.exp(np.log(hr_per_sd) * x)
    r = lam + delta
    # (e^{-rF} - e^{-r(A+F)}) / (rA) = e^{-rF} (1 - e^{-rA}) / (rA) -> e^{-rF} as A -> 0
    rA = r * A
    window = np.ones_like(rA) if A == 0 else -np.expm1(-rA) / rA
    p = lam / r * (1.0 - np.exp(-r * F) * window)
    return float(w @ p / np.sqrt(2 * np.pi))
