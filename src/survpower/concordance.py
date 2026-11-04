"""Concordance-index effect sizes and power (report Appendix B).

Three questions a target concordance C can mean, each with its own N:

1. **Association** -- is the biomarker related to survival at all? Convert C
   to a per-SD log-HR with `beta_of_c` and use the Cox Wald test (efficient
   under PH; see `survpower.planning`).
2. **Benchmark** -- does the biomarker's C exceed a benchmark C0 (a minimum
   useful level)? One-sided test of H0: C <= C0 on the estimated C with its
   asymptotic SE (eq. 29). C0 is treated as a known constant. When the
   benchmark is a competing model's C estimated on the same patients, the
   right analysis is a paired comparison of correlated C estimates
   (DeLong et al., 1988; Kang et al., 2015), which this module does not
   implement; the fixed-benchmark N can then be conservative or not.
3. **Precision** -- how precisely is C estimated? N for a target 95% CI
   half-width.

Mapping (eq. 25): for X ~ N(0, 1) under PH with *any* baseline hazard,
P(T_i < T_j | X_i, X_j) = expit(beta (X_i - X_j)), so the population
(Goenen-Heller) concordance K(beta) is a 1-D integral over D = X_i - X_j ~ N(0, 2).

Estimators:

* **Harrell's C** counts a pair as comparable when the shorter observed time
  is an event. Under censoring it converges to `harrell_limit` (> K(beta));
  benchmark sample sizes use that limit (Table 10). The offset is small at
  20% censoring (0.7032 vs. 0.70) and grows with censoring; because
  N ~ (C - C0)^-2, uncertainty in the assumed C itself matters far more, so
  run several plausible C values.
* **Uno's C** (Uno et al., 2011) re-weights comparable pairs by 1/G(T_i)^2,
  G = Kaplan-Meier survival of the censoring distribution, and truncates at
  a horizon tau. It converges to the censoring-free, tau-truncated
  concordance `uno_limit`, so it does not depend on the censoring pattern.

Standard errors use the Hajek projection of the (weighted) ratio of
U-statistics with the delta method (eq. 28); for Uno's C the IPCW weights
are treated as known, which slightly understates the SE (checked by
simulation in the test-suite). Both estimators build n x n pair matrices:
practical up to a few thousand subjects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import integrate, optimize, stats
from scipy.special import expit

from ._utils import tqdm
from .baseline import DEFAULT_LAMBDA0
from .censoring import calibrate_censoring_rate
from .fitting import cox_fit_1d, wald_rejects
from .results import mean_mcse, proportion_mcse, sd_mcse

if TYPE_CHECKING:  # design.py imports this module
    from .design import StudyDesign

ESTIMATORS = ("harrell", "uno")
RULE_OF_THUMB_KAPPA = 1.64  # fitted to the exact inverse of eq. 25 (Appendix B.1)


# --- C <-> beta ---------------------------------------------------------------

def c_of_beta(beta: float) -> float:
    """Population concordance K(beta) = P(T_i < T_j | eta_i > eta_j) for a
    standard-normal biomarker with per-SD log-HR beta (eq. 25). Symmetric in
    the sign of beta (the concordance of the correctly oriented score)."""
    b = abs(beta)
    s = np.sqrt(2.0)
    f = lambda d: expit(b * d) * stats.norm.pdf(d, scale=s)
    return float(2.0 * integrate.quad(f, 0, np.inf, epsabs=1e-12, epsrel=1e-12)[0])


def beta_of_c(c: float) -> float:
    """Per-SD log-HR beta > 0 with K(beta) = c, for 0.5 < c < 1 (Brent)."""
    if not 0.5 < c < 1:
        raise ValueError("concordance must be in (0.5, 1)")
    return float(optimize.brentq(lambda b: c_of_beta(b) - c, 1e-9, 20.0, xtol=1e-12))


def hr_of_c(c: float) -> float:
    """Per-SD hazard ratio implied by a target concordance."""
    return float(np.exp(beta_of_c(c)))


def beta_of_c_approx(c: float, kappa: float = RULE_OF_THUMB_KAPPA) -> float:
    """Closed-form rule of thumb (eq. 26): beta ~ (kappa / sqrt 2) tan(pi (C - 1/2)),
    within 3% of the exact inverse over C in [0.55, 0.75]."""
    return float(kappa / np.sqrt(2.0) * np.tan(np.pi * (c - 0.5)))


# --- What the estimators converge to ------------------------------------------

def harrell_limit(beta: float, lam_c: float, lambda0: float = DEFAULT_LAMBDA0,
                  n: int = 2_000_000, seed: int = 7) -> float:
    """Large-sample limit of Harrell's C under exponential censoring (eq. 27).
    A pair is comparable with probability (r_i + r_j) / (r_i + r_j + 2 lam_c)
    and then concordant with probability expit(|beta| |X_i - X_j|)."""
    rng = np.random.default_rng(seed)
    xi, xj = rng.standard_normal(n), rng.standard_normal(n)
    ri, rj = lambda0 * np.exp(beta * xi), lambda0 * np.exp(beta * xj)
    w = (ri + rj) / (ri + rj + 2 * lam_c)
    return float(np.sum(w * expit(abs(beta) * np.abs(xi - xj))) / np.sum(w))


def uno_limit(beta: float, tau: float, lambda0: float = DEFAULT_LAMBDA0,
              n: int = 2_000_000, seed: int = 7) -> float:
    """Large-sample limit of Uno's C truncated at tau: the censoring-free
    concordance among pairs whose first failure occurs before tau."""
    rng = np.random.default_rng(seed)
    xi, xj = rng.standard_normal(n), rng.standard_normal(n)
    ti = rng.exponential(scale=1.0 / (lambda0 * np.exp(beta * xi)))
    tj = rng.exponential(scale=1.0 / (lambda0 * np.exp(beta * xj)))
    first_i = ti < tj
    valid = np.minimum(ti, tj) < tau
    s = np.sign(beta) if beta != 0 else 1.0
    conc = np.where(first_i, s * xi > s * xj, s * xj > s * xi)
    return float(np.mean(conc[valid]))


def default_uno_tau(beta: float, lam_c: float, lambda0: float = DEFAULT_LAMBDA0,
                    quantile: float = 0.9, n_probe: int = 200_000, seed: int = 3) -> float:
    """Default Uno horizon: the `quantile` of observed follow-up min(T, C)
    under the design, which keeps the IPCW weights bounded."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(n_probe)
    T = rng.exponential(scale=1.0 / (lambda0 * np.exp(beta * X)))
    C = rng.exponential(scale=1.0 / lam_c, size=n_probe)
    return float(np.quantile(np.minimum(T, C), quantile))


def concordance_limit(beta: float, lam_c: float, method: str = "harrell",
                      tau: float | None = None, lambda0: float = DEFAULT_LAMBDA0) -> float:
    """Value the chosen estimator converges to under the design."""
    if method == "harrell":
        return harrell_limit(beta, lam_c, lambda0=lambda0)
    if method == "uno":
        tau = default_uno_tau(beta, lam_c, lambda0) if tau is None else tau
        return uno_limit(beta, tau, lambda0=lambda0)
    raise ValueError(f"method must be one of {ESTIMATORS}")


# --- C <-> HR table (Table 8) ---------------------------------------------------

def mapping_table(levels=(0.55, 0.60, 0.65, 0.70, 0.75), censor_rate: float = 0.20,
                  design: StudyDesign | None = None, alpha: float = 0.05,
                  power: float = 0.80) -> pd.DataFrame:
    """C <-> per-SD HR table (Table 8): for each target concordance, the exact
    beta and HR, the censoring rate lam_c calibrated to `censor_rate`, the
    large-sample limit of Harrell's C under that censoring, and the
    closed-form events D and enrolled N for `power` to detect beta != 0 with
    the two-sided Wald test (eq. 16).

    With a `design`, its baseline hazard and censoring (fixed rate or
    accrual / follow-up / dropout) replace `censor_rate`. Values are unrounded."""
    lambda0 = DEFAULT_LAMBDA0 if design is None else design.lambda0
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    rows = []
    for c in levels:
        b = beta_of_c(c)
        hr = float(np.exp(b))
        if design is None:
            rate = censor_rate
            lam_c = calibrate_censoring_rate(b, rate, lambda0=lambda0)
        else:
            rate, lam_c = design.censor_rate_for(hr), design.lambda_c_for(hr)
        d = z ** 2 / b ** 2
        rows.append(dict(C=float(c), beta=b, hr=hr, lam_c=lam_c,
                         harrell_limit=harrell_limit(b, lam_c, lambda0),
                         D80=d, N80_closed_form=int(np.ceil(d / (1 - rate)))))
    return pd.DataFrame(rows)


# --- Estimators with standard errors ------------------------------------------

def _weighted_c_with_se(comp_w: np.ndarray, conc: np.ndarray) -> tuple[float, float]:
    """Ratio-of-U-statistics C = A / B with Hajek-projection SE, where
    comp_w[i, j] is the weight of the comparable ordered pair (i fails first)
    and conc[i, j] whether i has the higher risk score."""
    W = comp_w
    Wc = W * conc
    B, A = W.sum(), Wc.sum()
    if B <= 0:
        return float("nan"), float("nan")
    c = A / B
    a = Wc.sum(1) + Wc.sum(0)
    b = W.sum(1) + W.sum(0)
    se = np.sqrt(np.sum((a - c * b) ** 2)) / B
    return float(c), float(se)


def harrell_c_with_se(time: np.ndarray, event: np.ndarray, eta: np.ndarray) -> tuple[float, float]:
    """Harrell's C (higher eta = higher risk; no ties) with its asymptotic SE
    (eq. 28)."""
    time, event, eta = np.asarray(time), np.asarray(event).astype(bool), np.asarray(eta)
    comp = event[:, None] & (time[:, None] < time[None, :])
    conc = eta[:, None] > eta[None, :]
    return _weighted_c_with_se(comp.astype(float), conc)


def censoring_survival_left(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    """Kaplan-Meier survival of the censoring distribution, G(t-), evaluated
    at each subject's own observed time (continuous times, no ties)."""
    time, event = np.asarray(time), np.asarray(event).astype(bool)
    order = np.argsort(time, kind="mergesort")
    cens = ~event[order]
    at_risk = len(time) - np.arange(len(time))
    g_after = np.cumprod(np.where(cens, 1.0 - 1.0 / at_risk, 1.0))
    g_before = np.concatenate([[1.0], g_after[:-1]])
    out = np.empty(len(time))
    out[order] = g_before
    return out


def uno_c_with_se(time: np.ndarray, event: np.ndarray, eta: np.ndarray,
                  tau: float | None = None) -> tuple[float, float]:
    """Uno's IPCW concordance truncated at tau (default: largest observed
    time) with a Hajek-projection SE treating the weights as known."""
    time, event, eta = np.asarray(time), np.asarray(event).astype(bool), np.asarray(eta)
    tau = float(np.max(time)) if tau is None else tau
    G = censoring_survival_left(time, event)
    w = np.where(event & (time < tau), 1.0 / np.maximum(G, 1e-12) ** 2, 0.0)
    comp = (time[:, None] < time[None, :]) * w[:, None]
    conc = eta[:, None] > eta[None, :]
    return _weighted_c_with_se(comp, conc)


def concordance_index(time, event, eta, method: str = "harrell",
                      tau: float | None = None) -> tuple[float, float]:
    """Concordance index and its asymptotic SE. `eta` is a risk score
    (higher = shorter survival), e.g. the biomarker or a Cox linear predictor."""
    if method == "harrell":
        return harrell_c_with_se(time, event, eta)
    if method == "uno":
        return uno_c_with_se(time, event, eta, tau=tau)
    raise ValueError(f"method must be one of {ESTIMATORS}")


# --- Simulation ---------------------------------------------------------------

def simulate_cindex_power(
    n: int,
    beta: float,
    lam_c: float,
    n_sims: int = 1000,
    alpha: float = 0.05,
    sides: int = 2,
    c0: float | None = None,
    method: str = "harrell",
    tau: float | None = None,
    lambda0: float = DEFAULT_LAMBDA0,
    seed: int = 0,
) -> dict:
    """One simulation cell of Appendix B: cohorts of size n with per-SD log-HR
    beta and exponential censoring rate lam_c. Records

    * power_cox  : Cox Wald test of beta = 0 (`sides`, level alpha)
    * power_c05  : test of C = 0.5 from the estimated C and its SE (`sides`)
    * power_cc0  : one-sided test of H0: C <= c0 at level alpha (if c0 given)

    plus the mean / SD of the estimated C, its mean SE, events and mean HR,
    each Monte Carlo estimate with a `*_mcse` column. Cohorts with fewer than
    two events are skipped (counted in `n_failed`, not rejected).
    The biomarker is oriented by the sign of beta (risk score = sign(beta) X).
    """
    if method == "uno" and tau is None:
        tau = default_uno_tau(beta, lam_c, lambda0)
    s = np.sign(beta) if beta != 0 else 1.0
    z_one = stats.norm.ppf(1 - alpha)
    rng = np.random.default_rng(seed)
    rej_cox = rej_c05 = rej_cc0 = 0
    cs, ses, bh, ev = [], [], [], []
    for _ in tqdm(range(n_sims), desc=f"c-index n={n}", leave=False):
        X = rng.standard_normal(n)
        T = rng.exponential(scale=1.0 / (lambda0 * np.exp(beta * X)))
        C = rng.exponential(scale=1.0 / lam_c, size=n)
        t, d = np.minimum(T, C), T <= C
        ev.append(d.sum())
        if d.sum() < 2:
            continue
        b, se_b = cox_fit_1d(t, d, X)
        c, se_c = concordance_index(t, d, s * X, method=method, tau=tau)
        rej_cox += wald_rejects(b / se_b, beta, alpha, sides)
        if np.isfinite(se_c) and se_c > 0:
            rej_c05 += wald_rejects((c - 0.5) / se_c, 1.0, alpha, sides)
            if c0 is not None:
                rej_cc0 += (c - c0) / se_c > z_one
        cs.append(c); ses.append(se_c); bh.append(b)
    cs, ses, bh = map(np.asarray, (cs, ses, bh))
    out = dict(
        n=n, n_sims=n_sims, method=method, mean_events=round(float(np.mean(ev)), 1),
        power_cox=rej_cox / n_sims, power_c05=rej_c05 / n_sims,
        mean_c=float(np.nanmean(cs)) if len(cs) else float("nan"),
        sd_c=float(np.nanstd(cs, ddof=1)) if len(cs) > 1 else float("nan"),
        mean_se_c=float(np.nanmean(ses)) if len(ses) else float("nan"),
        mean_hr=float(np.nanmean(np.exp(bh))) if len(bh) else float("nan"),
        power_cox_mcse=proportion_mcse(rej_cox / n_sims, n_sims),
        power_c05_mcse=proportion_mcse(rej_c05 / n_sims, n_sims),
        mean_c_mcse=mean_mcse(cs), sd_c_mcse=sd_mcse(cs),
        mean_se_c_mcse=mean_mcse(ses), mean_hr_mcse=mean_mcse(np.exp(bh)),
        n_failed=n_sims - len(cs),
    )
    if c0 is not None:
        out.update(c0=c0, power_cc0=float(rej_cc0) / n_sims,
                   power_cc0_mcse=proportion_mcse(float(rej_cc0) / n_sims, n_sims))
    if method == "uno":
        out["tau"] = tau
    return out


def sigma_c(beta: float, lam_c: float, method: str = "harrell", tau: float | None = None,
            lambda0: float = DEFAULT_LAMBDA0, n: int = 2000, n_sims: int = 40,
            seed: int = 99) -> float:
    """Asymptotic SD of sqrt(N) * C_hat, sigma_C = lim sqrt(N Var(C_hat)),
    estimated from the analytic SE in `n_sims` large cohorts of size n
    (Appendix B.4 used the empirical SD at N = 3000; the analytic SE is far
    less noisy and agrees within a few percent at this size)."""
    out = _se_draws(n, beta, lam_c, method, tau, lambda0, n_sims, seed)
    return float(np.sqrt(n) * np.sqrt(np.mean(out ** 2)))


def _se_draws(n, beta, lam_c, method, tau, lambda0, n_sims, seed):
    if method == "uno" and tau is None:
        tau = default_uno_tau(beta, lam_c, lambda0)
    s = np.sign(beta) if beta != 0 else 1.0
    rng = np.random.default_rng(seed)
    ses = []
    for _ in range(n_sims):
        X = rng.standard_normal(n)
        T = rng.exponential(scale=1.0 / (lambda0 * np.exp(beta * X)))
        C = rng.exponential(scale=1.0 / lam_c, size=n)
        ses.append(concordance_index(np.minimum(T, C), T <= C, s * X, method=method, tau=tau)[1])
    return np.asarray(ses)


# --- Closed forms (eq. 29) ---------------------------------------------------

def benchmark_power(n: int, c_limit: float, c0: float, sigma: float, alpha: float = 0.05) -> float:
    """Approximate power of the one-sided test of H0: C <= c0 at level alpha."""
    return float(stats.norm.cdf((c_limit - c0) * np.sqrt(n) / sigma - stats.norm.ppf(1 - alpha)))


def benchmark_sample_size(c_limit: float, c0: float, sigma: float, power: float = 0.80,
                          alpha: float = 0.05) -> int:
    """N for the one-sided benchmark test (eq. 29). Use the estimator's
    limit (`concordance_limit`), not the nominal C."""
    if c_limit <= c0:
        raise ValueError("the estimator's limit must exceed the benchmark c0")
    z = stats.norm.ppf(1 - alpha) + stats.norm.ppf(power)
    return int(np.ceil((z * sigma / (c_limit - c0)) ** 2))


def precision_half_width(n: int, sigma: float, level: float = 0.95) -> float:
    """Expected CI half-width for C at sample size n."""
    return float(stats.norm.ppf(0.5 + level / 2) * sigma / np.sqrt(n))


def precision_sample_size(sigma: float, half_width: float, level: float = 0.95) -> int:
    """N for a target CI half-width on C: N = (z sigma / h)^2."""
    return int(np.ceil((stats.norm.ppf(0.5 + level / 2) * sigma / half_width) ** 2))
