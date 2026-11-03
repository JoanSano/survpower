"""Closed-form (asymptotic) power and sample-size formulas.

* Continuous covariate: Schoenfeld (1983) / Hsieh & Lloyd (1998), report
  Section 8.1. Information for beta is ~ D * sigma_x^2 * (1 - R^2), where D is
  the number of events and R^2 the multiple correlation of X with the
  adjustment covariates. Mildly optimistic at small N or large effects --
  use the simulation-calibrated versions in `survpower.calibration`.
* Dichotomized (median-split) covariate: Schoenfeld/Freedman two-group
  formula via lifelines (report Sections 5-6).

All functions are scale-invariant to the baseline hazard lambda0.
"""

from __future__ import annotations

import numpy as np
from lifelines.statistics import power_under_cph, sample_size_necessary_under_cph
from scipy.optimize import brentq
from scipy.stats import norm

# E[X | X > 0] - E[X | X < 0] for X ~ N(0, 1): 2 * sqrt(2 / pi)
MEDIAN_SPLIT_SEPARATION = 1.5958


def sample_size_continuous_cph(
    hr: float,
    power: float,
    sigma_x: float = 1.0,
    alpha: float = 0.05,
    event_prob: float = 0.80,
    rho_sq: float = 0.0,
    sides: int = 2,
) -> tuple[float, int]:
    """Required events D and sample size N for a continuous Cox covariate.

    hr         : per-unit (per-SD if sigma_x = 1) hazard ratio
    power      : target power
    event_prob : expected marginal probability of an observed event
    rho_sq     : R^2 of X on the other covariates (0 for a univariate model)
    sides      : 2 (two-sided) or 1 (one-sided) test

    Returns (D, N).
    """
    beta = np.log(hr)
    za, zb = norm.ppf(1 - alpha / sides), norm.ppf(power)
    D = (za + zb) ** 2 / (beta ** 2 * sigma_x ** 2 * (1 - rho_sq))
    N = int(np.ceil(D / event_prob))
    return float(D), N


def detectable_hr_continuous_cph(
    n: int,
    power: float,
    sigma_x: float = 1.0,
    alpha: float = 0.05,
    event_prob: float = 0.80,
    rho_sq: float = 0.0,
    sides: int = 2,
) -> float:
    """Smallest hazard ratio > 1 detectable with `power` at sample size n
    (inverse of `sample_size_continuous_cph`). By symmetry of X, 1/HR is
    detectable with the same power."""
    za, zb = norm.ppf(1 - alpha / sides), norm.ppf(power)
    beta = (za + zb) / np.sqrt(n * event_prob * sigma_x ** 2 * (1 - rho_sq))
    return float(np.exp(beta))


def theoretical_power_with_r2(
    n: int, hr: float, censor_rate: float, r2_target: float = 0.0,
    sigma_x: float = 1.0, alpha: float = 0.05, sides: int = 2,
) -> float:
    """Closed-form Wald power at sample size n, with information reduced by
    (1 - R^2) for adjustment covariates. Independent of how many covariates
    make up R^2 -- exactly the prediction the collinearity study checks."""
    beta = np.log(hr)
    info = n * (1 - censor_rate) * sigma_x ** 2 * (1 - r2_target)
    z_alpha = norm.ppf(1 - alpha / sides)
    return float(norm.cdf(abs(beta) * np.sqrt(info) - z_alpha))


# --- Dichotomized (median split) ---------------------------------------------

def group_hr_from_persd_hr(hr_per_sd: float) -> float:
    """Plug-in group-level HR for a median split of X ~ N(0, 1):
    HR_group ~ HR_per_SD ** 1.5958. Optimistic (non-collapsibility attenuates
    the HR a real dichotomized fit estimates), see report Section 5."""
    return float(np.exp(np.log(hr_per_sd) * MEDIAN_SPLIT_SEPARATION))


def solve_p_con(group_hr: float, marginal_p: float = 0.80) -> float:
    """Control-arm event probability such that the pooled 50/50 event
    probability equals `marginal_p`, using S_exp = S_con ** HR."""
    f = lambda p: 0.5 * (1 - (1 - p) ** group_hr) + 0.5 * p - marginal_p
    return float(brentq(f, 1e-6, 1 - 1e-6))


def dichotomized_design(hr_per_sd: float, censor_rate: float) -> tuple[float, float, float]:
    """Median-split group HR and arm-wise event probabilities (p_con, p_exp)."""
    ghr = group_hr_from_persd_hr(hr_per_sd)
    p_con = solve_p_con(ghr, marginal_p=1 - censor_rate)
    p_exp = 1 - (1 - p_con) ** ghr
    return ghr, p_con, p_exp


def _two_sided_alpha(alpha: float, sides: int) -> float:
    # lifelines' formulas are two-sided; one-sided at alpha has the same
    # normal-approximation power as two-sided at 2 * alpha
    return alpha * (2 if sides == 1 else 1)


def dichotomized_power(n: int, hr_per_sd: float, censor_rate: float,
                       alpha: float = 0.05, sides: int = 2) -> float:
    """Closed-form power of a median-split analysis with n subjects."""
    ghr, p_con, p_exp = dichotomized_design(hr_per_sd, censor_rate)
    return float(power_under_cph(n // 2, n - n // 2, p_exp, p_con, ghr,
                                 alpha=_two_sided_alpha(alpha, sides)))


def dichotomized_sample_size(power: float, hr_per_sd: float, censor_rate: float,
                             alpha: float = 0.05, sides: int = 2) -> int:
    """Total N for a median-split analysis to reach `power`."""
    ghr, p_con, p_exp = dichotomized_design(hr_per_sd, censor_rate)
    n_exp, n_con = sample_size_necessary_under_cph(
        power, 1.0, p_exp, p_con, ghr, alpha=_two_sided_alpha(alpha, sides))
    return int(n_exp + n_con)
