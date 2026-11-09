"""Independent censoring and its calibration.

By default censoring is C ~ Exponential(lambda_c), independent of T. lambda_c is
chosen so that the marginal censoring probability P(C < T) hits a target. Because
P(C < T) is strictly increasing in lambda_c, bisection on a large Monte Carlo
probe sample converges to the unique root.

The other calibrated censoring shapes (#16) share one form: a cumulative hazard
H_C(t) = lambda_c R(t) with a fixed, increasing R (t^k for Weibull, a piecewise-
linear R for piecewise-constant hazards). Then

    P(C < T) = E[1 - exp(-lambda_c R(T))],

which is strictly increasing in lambda_c, pointwise in T, so the calibration again
has a unique root (`calibrate_cumhaz_scale`).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .baseline import (DEFAULT_LAMBDA0, DEFAULT_MEDIAN_OS_MONTHS, marginal_cure_fraction,
                       median_os_to_lambda0)
from .dgp import gen_adjustment_covariates, gen_crossing, gen_ph, gen_waning


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


def calibrate_cumhaz_scale(T: np.ndarray, rel_cumhaz, target_censor_rate: float) -> float:
    """Scale lambda_c of a censoring cumulative hazard H_C(t) = lambda_c R(t) giving
    P(C < T) = target over the probe sample of event times `T`.

    P(C < T | T) = 1 - exp(-lambda_c R(T)) is computed exactly, so no censoring
    times are drawn, and its mean is strictly increasing in lambda_c (pointwise in
    T, for R(T) > 0): the root is unique, found by Brent's method on log lambda_c.
    Event times T = inf (the cured of a defective baseline) are always censored,
    so the target must be above their fraction."""
    r = rel_cumhaz(np.asarray(T, dtype=float))
    def excess(log_lam):
        return float(np.mean(-np.expm1(-np.exp(log_lam) * r))) - target_censor_rate
    lo, hi = -60.0, 60.0
    if not excess(lo) < 0 < excess(hi):
        raise ValueError(f"no censoring scale gives P(C < T) = {target_censor_rate:.3g}: "
                         f"it must be in ({excess(lo) + target_censor_rate:.3g}, "
                         f"{excess(hi) + target_censor_rate:.3g})")
    return float(np.exp(brentq(excess, lo, hi, xtol=1e-12, rtol=1e-12)))


def calibrate_censoring_for(
    sampler,
    target_censor_rate: float,
    seed: int = 1,
    n_probe: int = 500_000,
    rel_cumhaz=None,
) -> float:
    """lambda_c for any event-time generator: draws a probe sample
    ``T = sampler(n_probe, rng)`` from ``default_rng(seed)``.

    Without `rel_cumhaz`, censoring is exponential and lambda_c is bisected on the
    probe with the same generator (`calibrate_censoring`), as it always has been.
    With it, lambda_c is the scale of H_C(t) = lambda_c * rel_cumhaz(t)
    (`calibrate_cumhaz_scale`)."""
    rng = np.random.default_rng(seed)
    T = np.asarray(sampler(n_probe, rng))
    if rel_cumhaz is None:
        return calibrate_censoring(T, target_censor_rate, rng)
    return calibrate_cumhaz_scale(T, rel_cumhaz, target_censor_rate)


def piecewise_cumhaz(t, breaks, rates) -> np.ndarray:
    """Cumulative hazard of a piecewise-constant hazard: `rates[k]` on
    [breaks[k-1], breaks[k]), with breaks[-1] = 0 and the last rate to infinity."""
    starts = np.concatenate([[0.0], np.asarray(breaks, dtype=float)])
    rates = np.asarray(rates, dtype=float)
    h_starts = np.concatenate([[0.0], np.cumsum(rates[:-1] * np.diff(starts))])
    t = np.asarray(t, dtype=float)
    k = np.searchsorted(starts, t, side="right") - 1
    return h_starts[k] + rates[k] * (t - starts[k])


def piecewise_inv_cumhaz(h, breaks, rates) -> np.ndarray:
    """Inverse of `piecewise_cumhaz` (all rates positive)."""
    starts = np.concatenate([[0.0], np.asarray(breaks, dtype=float)])
    rates = np.asarray(rates, dtype=float)
    h_starts = np.concatenate([[0.0], np.cumsum(rates[:-1] * np.diff(starts))])
    h = np.asarray(h, dtype=float)
    k = np.searchsorted(h_starts, h, side="right") - 1
    return starts[k] + (h - h_starts[k]) / rates[k]


def dropout_scale(dropout_annual: float, shape: float = 1.0) -> float:
    """Scale lambda_d of a Weibull dropout cumulative hazard H_D(t) = lambda_d t^shape
    (t in months) whose 12-month loss probability is `dropout_annual`:
    lambda_d = -ln(1 - d) / 12^shape. With shape 1, the exponential dropout rate
    of `censor_rate_from_design`."""
    return float(-np.log1p(-dropout_annual) / 12.0 ** shape)


def require_censoring_above_cure(baseline, log_hr: float, censor_rate: float) -> None:
    """Raise if the censoring target is not above the marginal cure fraction of a
    defective baseline: the cured are never observed to fail, so they are always
    censored, and the censoring calibration has that floor."""
    cure = marginal_cure_fraction(baseline, log_hr)
    if cure > 0 and censor_rate <= cure + 1e-3:
        raise ValueError(f"censor_rate {censor_rate:.3g} must be above the cure fraction "
                         f"{cure:.3g} of the {baseline.label()} baseline: the cured are "
                         f"always censored")


def calibrate_censoring_rate(
    beta: float,
    target_censor_rate: float,
    lambda0: float = DEFAULT_LAMBDA0,
    n_probe: int = 500_000,
    seed: int = 1,
    baseline=None,
) -> float:
    """lambda_c for the PH model with per-SD log-HR `beta`: exponential with rate
    `lambda0`, or with the given `baseline` hazard (`survpower.baseline`)."""
    return calibrate_censoring_for(lambda n, rng: gen_ph(n, beta, rng, lambda0, baseline)[0],
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
    baseline=None,
) -> float:
    """Marginal censoring probability implied by a trial calendar.

    Entry is uniform over [0, accrual_months] and the analysis happens
    `followup_months` after the last patient is enrolled, so each patient's
    administrative censoring time is C_adm ~ U(followup, accrual + followup).
    Optional exponential dropout has annual loss fraction `dropout_annual`.

    By default the simulation engines then use exponential censoring calibrated
    to this rate (`StudyDesign(censoring_model="calendar")` simulates the
    calendar itself). At the docs example this leaves Cox power unchanged within
    Monte Carlo error, but understates the power of tests on Harrell's and Uno's
    C by 4-6 points (#16; docs/concepts/censoring-and-accrual.md). In closed
    form this rate is 1 - `event_probability_exponential`.

    Event times are exponential with rate `lambda0`, or follow the given
    `baseline` hazard (`survpower.baseline`) under PH. A defective baseline's
    cured patients are always censored.
    """
    rng = np.random.default_rng(seed)
    T, _ = gen_ph(n_probe, np.log(hr), rng, lambda0, baseline)
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
