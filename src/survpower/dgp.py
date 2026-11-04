"""Event-time generators (report Sections 2, 9, 10 and Appendix A).

All generators take a `numpy.random.Generator` and draw in a fixed order so
that seeded results are reproducible within one installed version.
"""

from __future__ import annotations

import numpy as np


def gen_exponential_ph(n: int, beta: float, lambda0: float, rng: np.random.Generator):
    """X ~ N(0, 1) and T | X ~ Exponential(lambda0 * exp(beta * X)) -- an exact
    draw from a Cox PH model with constant baseline hazard. Returns (T, X)."""
    X = rng.standard_normal(n)
    T = rng.exponential(scale=1.0 / (lambda0 * np.exp(beta * X)))
    return T, X


# --- Correlated adjustment covariates (Section 9) ---------------------------
#
# Z_j = rho * X + sqrt(1 - rho^2) * W_j, W_j iid N(0, 1): corr(X, Z_j) = rho and
# corr(Z_i, Z_j) = rho^2. The multiple correlation of X on (Z_1..Z_p) is
#     R^2 = rho^2 p / (1 + rho^2 (p - 1)),
# inverted by `rho_from_target_r2`.

def rho_from_target_r2(r2_target: float, p: int) -> float:
    """Per-covariate correlation rho giving multiple correlation R^2 of X on
    p shared-factor covariates. Returns 0 for p = 0 or r2_target = 0."""
    if p == 0 or r2_target == 0:
        return 0.0
    denom = p - r2_target * (p - 1)
    return float(np.sqrt(r2_target / denom))


def gen_adjustment_covariates(n: int, r2_target: float, p: int, rng: np.random.Generator):
    """Draw X ~ N(0, 1) and p adjustment covariates Z with multiple correlation
    R^2 = r2_target on X. Returns (X, Z); Z is None when p = 0."""
    X = rng.standard_normal(n)
    if p == 0:
        return X, None
    rho = rho_from_target_r2(r2_target, p)
    W = rng.standard_normal((n, p))
    Z = rho * X[:, None] + np.sqrt(1 - rho ** 2) * W
    return X, Z


# --- Non-proportional hazards (Appendix A) -----------------------------------

def gen_waning(n: int, beta0: float, tau: float, lambda0: float, rng: np.random.Generator):
    """Hazard ratio exp(beta0) for t < tau and 1 afterwards, by inverse
    cumulative-hazard sampling. Returns (T, X)."""
    X = rng.standard_normal(n)
    U = rng.uniform(size=n)
    target = -np.log(U)
    haz_early = lambda0 * np.exp(beta0 * X)
    thresh = haz_early * tau
    T = np.where(target <= thresh, target / haz_early, tau + (target - thresh) / lambda0)
    return T, X


def gen_crossing(n: int, beta_early: float, beta_late: float, tau: float,
                 lambda0: float, rng: np.random.Generator):
    """Hazard ratio exp(beta_early) for t < tau and exp(beta_late) afterwards.
    Returns (T, X)."""
    X = rng.standard_normal(n)
    U = rng.uniform(size=n)
    target = -np.log(U)
    haz_early = lambda0 * np.exp(beta_early * X)
    thresh = haz_early * tau
    t_early_branch = target / haz_early
    haz_late = lambda0 * np.exp(beta_late * X)
    t_late_branch = tau + (target - thresh) / haz_late
    T = np.where(target <= thresh, t_early_branch, t_late_branch)
    return T, X
