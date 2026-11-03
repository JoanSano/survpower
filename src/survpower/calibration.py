"""Simulation-calibrated sample size and minimum detectable effect
(report Section 8.3, Appendix B.3).

Closed forms anchor a small grid; simulated power at each grid point is fitted
on the probit scale (Phi^-1(power) = a*x + b, with x = sqrt(N) or log-HR) and
inverted for the target power, then confirmed by an independent, larger run.

At very small N (a few dozen subjects, e.g. a large concordance index) the
probit-linear form breaks down (Appendix B.3), so below `scan_below` the
calibrated N is replaced by a direct scan over integers: the smallest N whose
simulated power reaches the target.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
from scipy.stats import norm

from .baseline import DEFAULT_LAMBDA0
from .closed_form import detectable_hr_continuous_cph, sample_size_continuous_cph
from .simulate import simulate_power_design

PowerFn = Callable[[int, int, int], "tuple[float, float]"]  # (n, n_sims, seed) -> (power, events)


def fit_probit_line(x, powers, n_sims_per_point: int) -> tuple[float, float]:
    """Weighted least-squares fit of Phi^-1(power) = a*x + b, with a
    continuity-corrected probit transform and delta-method binomial weights."""
    p_tilde = (np.asarray(powers) * n_sims_per_point + 0.5) / (n_sims_per_point + 1)
    y = norm.ppf(p_tilde)
    w = n_sims_per_point * norm.pdf(y) ** 2 / (p_tilde * (1 - p_tilde))
    x = np.asarray(x, dtype=float)
    A = np.column_stack([x, np.ones_like(x)])
    W = np.diag(w)
    a_hat, b_hat = np.linalg.solve(A.T @ W @ A, A.T @ W @ y)
    return float(a_hat), float(b_hat)


def calibrate_sample_size(
    power_fn: PowerFn,
    n0: int,
    target_power: float,
    n_grid: list | None = None,
    n_sims_per_point: int = 300,
    confirm_n_sims: int = 2000,
    seed: int = 0,
    min_n: int = 10,
    scan_below: int = 100,
    scan_points: int = 9,
) -> dict:
    """Generic simulation calibration of N for any power function.

    1. Grid N0 x {0.6, 0.8, 1.0, 1.2, 1.4} (or `n_grid`).
    2. Simulated power at each point; WLS fit of Phi^-1(power) = a sqrt(N) + b.
    3. Invert for the target; confirm with `confirm_n_sims` replicates.
    4. If the calibrated N < `scan_below`, replace steps 3-4 by a direct scan
       of `scan_points` integers in [0.7 N*, 1.5 N*] at `confirm_n_sims`
       replicates each, keeping the smallest N reaching the target.

    Returns a dict with n_calibrated, a_hat, b_hat, grid, confirmed_power,
    confirmed_mean_events and method ("probit" or "scan").
    """
    if n_grid is None:
        n_grid = sorted({max(int(round(n0 * f)), min_n) for f in (0.6, 0.8, 1.0, 1.2, 1.4)})
    powers, grid = [], []
    for i, n in enumerate(n_grid):
        power_hat, mean_ev = power_fn(n, n_sims_per_point, seed + 1000 * (i + 1))
        powers.append(power_hat)
        grid.append(dict(n=n, power_hat=round(power_hat, 3), mean_events=round(mean_ev, 1)))

    a_hat = b_hat = float("nan")
    if len(n_grid) < 2:  # grid collapsed onto min_n: nothing to fit, the scan takes over
        n_star = int(n0)
    else:
        a_hat, b_hat = fit_probit_line(np.sqrt(n_grid), powers, n_sims_per_point)
        if a_hat > 0:
            n_star = int(np.ceil(((norm.ppf(target_power) - b_hat) / a_hat) ** 2))
        else:
            warnings.warn(f"simulated power does not increase over the N grid {n_grid}; "
                          "falling back to the closed-form N. Increase n_sims.")
            n_star = int(n0)
    n_star = max(n_star, min_n)

    out = dict(n0=int(n0), n_calibrated=n_star, a_hat=round(a_hat, 5), b_hat=round(b_hat, 5),
               grid=grid, method="probit")
    if n_star < scan_below:
        lo, hi = max(min_n, int(np.floor(0.7 * n_star))), int(np.ceil(1.5 * n_star))
        candidates = sorted({int(round(v)) for v in np.linspace(lo, hi, scan_points)})
        scan = []
        for k, n in enumerate(candidates):
            pw, ev = power_fn(n, confirm_n_sims, seed + 500_000 + k)
            scan.append(dict(n=n, power=round(pw, 3), mean_events=round(ev, 1)))
        hits = [s for s in scan if s["power"] >= target_power]
        if not hits:
            warnings.warn(f"no scanned N in [{lo}, {hi}] reached power {target_power}; "
                          "reporting the largest scanned N.")
        best = hits[0] if hits else scan[-1]
        out.update(n_calibrated=best["n"], confirmed_power=best["power"],
                   confirmed_mean_events=best["mean_events"], method="scan", scan=scan)
        return out

    confirm_power, confirm_events = power_fn(n_star, confirm_n_sims, seed + 999_999)
    out.update(confirmed_power=round(confirm_power, 3),
               confirmed_mean_events=round(confirm_events, 1))
    return out


def estimate_required_n_by_simulation(
    hr: float,
    target_power: float,
    censor_rate: float = 0.20,
    lambda0: float = DEFAULT_LAMBDA0,
    alpha: float = 0.05,
    n_grid: list | None = None,
    n_sims_per_point: int = 300,
    confirm_n_sims: int = 2000,
    seed: int = 0,
    p: int = 0,
    r2_target: float = 0.0,
    sides: int = 2,
    engine: str = "fast",
    scan_below: int = 100,
) -> dict:
    """Required N for a continuous covariate (optionally adjusted for p
    covariates at multiple correlation r2_target): closed form, then
    simulation-calibrated (`calibrate_sample_size`)."""
    D0, N0 = sample_size_continuous_cph(
        hr, target_power, alpha=alpha, event_prob=1 - censor_rate,
        rho_sq=r2_target if p > 0 else 0.0, sides=sides,
    )

    def power_fn(n, n_sims, s):
        return simulate_power_design(
            n=n, hr=hr, censor_rate=censor_rate, p=p, r2_target=r2_target,
            lambda0=lambda0, n_sims=n_sims, alpha=alpha, sides=sides, seed=s, engine=engine,
        )

    out = calibrate_sample_size(
        power_fn, N0, target_power, n_grid=n_grid, n_sims_per_point=n_sims_per_point,
        confirm_n_sims=confirm_n_sims, seed=seed, min_n=10, scan_below=scan_below,
    )
    return dict(
        hr=hr, target_power=target_power,
        events_closed_form=round(D0, 1),
        n_closed_form=N0, n_calibrated=out["n_calibrated"],
        a_hat=out["a_hat"], b_hat=out["b_hat"],
        grid=out["grid"],
        confirmed_power=out["confirmed_power"],
        confirmed_mean_events=out["confirmed_mean_events"],
        method=out["method"],
    )


def estimate_detectable_hr_by_simulation(
    n: int,
    target_power: float,
    censor_rate: float = 0.20,
    lambda0: float = DEFAULT_LAMBDA0,
    alpha: float = 0.05,
    p: int = 0,
    r2_target: float = 0.0,
    sides: int = 2,
    beta_factors: tuple = (0.7, 0.85, 1.0, 1.15, 1.3),
    n_sims_per_point: int = 300,
    confirm_n_sims: int = 2000,
    seed: int = 0,
    engine: str = "fast",
) -> dict:
    """Minimum detectable per-SD HR at sample size n: closed form, then a
    grid in log-HR, WLS fit of Phi^-1(power) = a log(HR) + b (near-linear at
    fixed N), inversion and confirmation."""
    hr0 = detectable_hr_continuous_cph(
        n, target_power, alpha=alpha, event_prob=1 - censor_rate,
        rho_sq=r2_target if p > 0 else 0.0, sides=sides,
    )
    betas = np.log(hr0) * np.array(beta_factors)
    powers, grid = [], []
    for i, b in enumerate(betas):
        power_hat, mean_ev = simulate_power_design(
            n=n, hr=float(np.exp(b)), censor_rate=censor_rate, p=p, r2_target=r2_target,
            lambda0=lambda0, n_sims=n_sims_per_point, alpha=alpha, sides=sides,
            seed=seed + 1000 * (i + 1), engine=engine,
        )
        powers.append(power_hat)
        grid.append(dict(hr=round(float(np.exp(b)), 4), power_hat=round(power_hat, 3),
                         mean_events=round(mean_ev, 1)))

    a_hat, b_hat = fit_probit_line(betas, powers, n_sims_per_point)
    if a_hat <= 0:
        warnings.warn(f"N={n}: simulated power does not increase over the HR grid; "
                      "falling back to the closed-form HR. Increase n_sims.")
        hr_calibrated = hr0
    else:
        hr_calibrated = float(np.exp((norm.ppf(target_power) - b_hat) / a_hat))

    confirm_power, confirm_events = simulate_power_design(
        n=n, hr=hr_calibrated, censor_rate=censor_rate, p=p, r2_target=r2_target,
        lambda0=lambda0, n_sims=confirm_n_sims, alpha=alpha, sides=sides,
        seed=seed + 999_999, engine=engine,
    )
    return dict(
        n=n, target_power=target_power,
        hr_closed_form=round(hr0, 4), hr_calibrated=round(hr_calibrated, 4),
        a_hat=round(a_hat, 5), b_hat=round(b_hat, 5),
        grid=grid,
        confirmed_power=round(confirm_power, 3),
        confirmed_mean_events=round(confirm_events, 1),
    )
