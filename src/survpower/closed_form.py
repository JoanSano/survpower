"""Closed-form (asymptotic) power and sample-size formulas.

* Continuous covariate: Schoenfeld (1983), with the (1 - R^2) variance
  inflation of Hsieh, Bloch & Larsen (1998) extended to the Cox model with
  non-binary covariates by Hsieh & Lavori (2000) (report Section 8.1).
  Information for beta is ~ D * sigma_x^2 * (1 - R^2), where D is the number
  of events and R^2 the multiple correlation of X with the adjustment
  covariates. Slightly optimistic, and badly so at large effects -- use the
  simulation-calibrated versions in `survpower.calibration`.
* Dichotomized (median-split) covariate: the Freedman (1982) and Schoenfeld
  (1983) two-group formulas (report Sections 5-6). lifelines'
  `power_under_cph` implements Freedman's. With the plug-in group HR,
  Schoenfeld's is algebraically the asymptotic-relative-efficiency
  prediction: a median split keeps 2/pi of the information.

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


def median_split_median_survival(hr_per_sd: float, lambda0: float) -> tuple[float, float]:
    """Model-based median survival (low, high) of the two median-split groups
    of X ~ N(0, 1): S_g(t) = E[exp(-lambda0 e^{beta X} t) | group g], solved
    for S_g(t) = 1/2. Not Kaplan-Meier: it exists however heavy the censoring."""
    from scipy.integrate import quad

    beta = np.log(hr_per_sd)

    def surv(t, lo, hi):
        f = lambda x: norm.pdf(x) * np.exp(-lambda0 * np.exp(beta * x) * t)
        return quad(f, lo, hi)[0] / 0.5

    m0 = np.log(2.0) / lambda0
    medians = []
    for lo, hi in ((-12.0, 0.0), (0.0, 12.0)):  # N(0, 1) mass beyond 12 SD is < 1e-32
        medians.append(float(brentq(lambda t: surv(t, lo, hi) - 0.5, m0 * 1e-4, m0 * 1e4)))
    return medians[0], medians[1]


def median_split_expected_ci(n: int, hr_per_sd: float, censor_rate: float,
                             level: float = 0.95) -> tuple[float, float, float, float]:
    """Closed-form fallback for a median-split trial of n subjects: plug-in
    group HR, the 95% Wald CI a typical trial reports (var log HR ~ 1/D_low +
    1/D_high) and expected events (low, high)."""
    ghr, p_con, p_exp = dichotomized_design(hr_per_sd, censor_rate)
    d_low, d_high = n / 2 * p_con, n / 2 * p_exp
    half = norm.ppf(0.5 + level / 2) * np.sqrt(1 / d_low + 1 / d_high)
    return float(ghr * np.exp(-half)), float(ghr * np.exp(half)), float(d_low), float(d_high)


def _two_sided_alpha(alpha: float, sides: int) -> float:
    # lifelines' formulas are two-sided; one-sided at alpha has the same
    # normal-approximation power as two-sided at 2 * alpha
    return alpha * (2 if sides == 1 else 1)


def schoenfeld_two_group_power(n_exp: int, n_con: int, p_exp: float, p_con: float,
                               hr: float, alpha: float = 0.05) -> float:
    """Schoenfeld (1983) two-group power: Phi(sqrt(k m)/(1 + k) |log HR| - z),
    with m expected events and allocation ratio k = n_exp / n_con (two-sided)."""
    k = n_exp / n_con
    m = n_exp * p_exp + n_con * p_con
    return float(norm.cdf(np.sqrt(k * m) / (1 + k) * abs(np.log(hr)) - norm.ppf(1 - alpha / 2)))


DICHOTOMIZED_METHODS = ("freedman", "schoenfeld")


def dichotomized_power(n: int, hr_per_sd: float, censor_rate: float,
                       alpha: float = 0.05, sides: int = 2, method: str = "freedman") -> float:
    """Closed-form power of a median-split analysis with n subjects
    (Freedman 1982, as in lifelines, or Schoenfeld 1983; they differ by at
    most ~0.003 at the HRs of the report, Freedman's always slightly lower)."""
    ghr, p_con, p_exp = dichotomized_design(hr_per_sd, censor_rate)
    a2 = _two_sided_alpha(alpha, sides)
    if method == "schoenfeld":
        return schoenfeld_two_group_power(n // 2, n - n // 2, p_exp, p_con, ghr, alpha=a2)
    return float(power_under_cph(n // 2, n - n // 2, p_exp, p_con, ghr, alpha=a2))


def dichotomized_sample_size(power: float, hr_per_sd: float, censor_rate: float,
                             alpha: float = 0.05, sides: int = 2, method: str = "freedman") -> int:
    """Total N for a median-split analysis to reach `power` (1:1 split)."""
    ghr, p_con, p_exp = dichotomized_design(hr_per_sd, censor_rate)
    a2 = _two_sided_alpha(alpha, sides)
    if method == "schoenfeld":
        events = 4 * (norm.ppf(1 - a2 / 2) + norm.ppf(power)) ** 2 / np.log(ghr) ** 2
        return int(2 * np.ceil(events / (p_con + p_exp)))  # events = (N/2)(p_con + p_exp)
    n_exp, n_con = sample_size_necessary_under_cph(power, 1.0, p_exp, p_con, ghr, alpha=a2)
    return int(n_exp + n_con)
