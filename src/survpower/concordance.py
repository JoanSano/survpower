"""Concordance-index effect sizes and power.

Three questions a target concordance C can mean, each with its own N:

1. **Association** -- is the biomarker related to survival at all? Convert C
   to a per-SD log-HR with `beta_of_c` and use the Cox Wald test (efficient
   under PH; see `survpower.planning`).
2. **Benchmark** -- does the biomarker's C exceed a benchmark C0 (a minimum
   useful level)? One-sided test of H0: C <= C0 on the estimated C with its
   asymptotic SE. C0 is treated as a known constant. When the
   benchmark is a competing model's C estimated on the same patients, the
   right analysis is a paired comparison of correlated C estimates
   (DeLong et al., 1988; Kang et al., 2015), which this module does not
   implement; the fixed-benchmark N can then be conservative or not.
3. **Precision** -- how precisely is C estimated? N for a target 95% CI
   half-width.

Mapping: for X ~ N(0, 1) under PH with *any* baseline hazard,
P(T_i < T_j | X_i, X_j) = expit(beta (X_i - X_j)), so the population
(Goenen-Heller) concordance K(beta) is a 1-D integral over D = X_i - X_j ~ N(0, 2).

Estimators:

* **Harrell's C** counts a pair as comparable when the shorter observed time
  is an event. Under censoring it converges to `harrell_limit` (> K(beta));
  benchmark sample sizes use that limit. The offset is small at
  20% censoring (0.7032 vs. 0.70) and grows with censoring; because
  N ~ (C - C0)^-2, uncertainty in the assumed C itself matters far more, so
  run several plausible C values.
* **Uno's C** (Uno et al., 2011) re-weights comparable pairs by 1/G(T_i)^2,
  G = Kaplan-Meier survival of the censoring distribution, and truncates at
  a horizon tau. It converges to the censoring-free, tau-truncated
  concordance `uno_limit`, so it does not depend on the censoring pattern.
* **Gonen-Heller's CPE** (Gonen & Heller, 2005) is model-based: it uses only
  the covariates and the fitted Cox coefficients, not the observed times, so
  it is unaffected by censoring (given a consistent beta_hat). Each pair
  contributes expit(|eta_i - eta_j|), eta = X beta_hat, which works for any
  number of covariates (the concordance of the whole linear predictor). It
  converges to K(beta) (`c_of_beta`) in the univariate model.

Standard errors use the Hajek projection of the (weighted) ratio of
U-statistics with the delta method; for Uno's C the IPCW weights
are treated as known, which slightly understates the SE (checked by
simulation in the test-suite). The Gonen-Heller SE adds the uncertainty of
beta_hat to the U-statistic variance by the delta method, on the smoothed
estimator of the paper. All estimators build n x n pair matrices: practical
up to a few thousand subjects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import integrate, optimize, stats
from scipy.special import expit, ndtr

from ._utils import tqdm
from .baseline import DEFAULT_LAMBDA0, is_exponential
from .censoring import calibrate_censoring_rate
from .dgp import ph_event_times
from .fitting import cox_fit_1d, wald_rejects
from .results import censoring_columns, mean_mcse, proportion_mcse, sd_mcse

if TYPE_CHECKING:  # design.py imports this module
    from .design import StudyDesign

ESTIMATORS = ("harrell", "uno", "gonen-heller")
RULE_OF_THUMB_KAPPA = 1.64  # fitted to the exact inverse of `c_of_beta`


# --- C <-> beta ---------------------------------------------------------------

def c_of_beta(beta: float) -> float:
    """Population concordance K(beta) = P(T_i < T_j | eta_i > eta_j) for a
    standard-normal biomarker with per-SD log-HR beta. Symmetric in
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
    """Closed-form rule of thumb: beta ~ (kappa / sqrt 2) tan(pi (C - 1/2)),
    within 3% of the exact inverse over C in [0.55, 0.75]."""
    return float(kappa / np.sqrt(2.0) * np.tan(np.pi * (c - 0.5)))


# --- What the estimators converge to ------------------------------------------

def harrell_limit(beta: float, lam_c: float, lambda0: float = DEFAULT_LAMBDA0,
                  n: int = 2_000_000, seed: int = 7, baseline=None) -> float:
    """Large-sample limit of Harrell's C under exponential censoring.

    Exponential baseline: a pair is comparable with probability
    (r_i + r_j) / (r_i + r_j + 2 lam_c) and then concordant with probability
    expit(|beta| |X_i - X_j|), averaged over `n` pairs of X. Any other
    `baseline` (`survpower.baseline`) has no such form: the limit is then the
    concordance among the comparable pairs of `n` simulated, censored pairs
    (Monte Carlo SE about 0.0004 at the default n)."""
    if not is_exponential(baseline):
        return _harrell_limit_mc(beta, lam_c, baseline, n, seed)
    lambda0 = lambda0 if baseline is None else baseline.rate
    rng = np.random.default_rng(seed)
    xi, xj = rng.standard_normal(n), rng.standard_normal(n)
    ri, rj = lambda0 * np.exp(beta * xi), lambda0 * np.exp(beta * xj)
    w = (ri + rj) / (ri + rj + 2 * lam_c)
    return float(np.sum(w * expit(abs(beta) * np.abs(xi - xj))) / np.sum(w))


def _harrell_limit_mc(beta, lam_c, baseline, n, seed) -> float:
    rng = np.random.default_rng(seed)
    xi, xj = rng.standard_normal(n), rng.standard_normal(n)
    Ti = ph_event_times(beta * xi, rng, baseline=baseline)
    Tj = ph_event_times(beta * xj, rng, baseline=baseline)
    Ci, Cj = rng.exponential(scale=1.0 / lam_c, size=(2, n))
    ti, tj, ei, ej = np.minimum(Ti, Ci), np.minimum(Tj, Cj), Ti <= Ci, Tj <= Cj
    first_i = ti < tj
    comparable = np.where(first_i, ei, ej)
    s = np.sign(beta) if beta != 0 else 1.0
    conc = np.where(first_i, s * xi > s * xj, s * xj > s * xi)
    return float(np.mean(conc[comparable]))


def uno_limit(beta: float, tau: float, lambda0: float = DEFAULT_LAMBDA0,
              n: int = 2_000_000, seed: int = 7, baseline=None) -> float:
    """Large-sample limit of Uno's C truncated at tau: the censoring-free
    concordance among pairs whose first failure occurs before tau, by Monte
    Carlo over `n` pairs, with exponential event times of rate `lambda0` or
    the given `baseline`."""
    rng = np.random.default_rng(seed)
    xi, xj = rng.standard_normal(n), rng.standard_normal(n)
    ti = ph_event_times(beta * xi, rng, lambda0, baseline)
    tj = ph_event_times(beta * xj, rng, lambda0, baseline)
    first_i = ti < tj
    valid = np.minimum(ti, tj) < tau
    s = np.sign(beta) if beta != 0 else 1.0
    conc = np.where(first_i, s * xi > s * xj, s * xj > s * xi)
    return float(np.mean(conc[valid]))


def default_uno_tau(beta: float, lam_c: float, lambda0: float = DEFAULT_LAMBDA0,
                    quantile: float = 0.9, n_probe: int = 200_000, seed: int = 3,
                    baseline=None) -> float:
    """Default Uno horizon: the `quantile` of observed follow-up min(T, C)
    under the design, which keeps the IPCW weights bounded."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(n_probe)
    T = ph_event_times(beta * X, rng, lambda0, baseline)
    C = rng.exponential(scale=1.0 / lam_c, size=n_probe)
    return float(np.quantile(np.minimum(T, C), quantile))


def concordance_limit(beta: float, lam_c: float, method: str = "harrell",
                      tau: float | None = None, lambda0: float = DEFAULT_LAMBDA0,
                      baseline=None) -> float:
    """Value the chosen estimator converges to under the design (exponential
    baseline of rate `lambda0`, or the given `baseline`)."""
    if method == "harrell":
        return harrell_limit(beta, lam_c, lambda0=lambda0, baseline=baseline)
    if method == "uno":
        tau = default_uno_tau(beta, lam_c, lambda0, baseline=baseline) if tau is None else tau
        return uno_limit(beta, tau, lambda0=lambda0, baseline=baseline)
    if method == "gonen-heller":
        return c_of_beta(beta)
    raise ValueError(f"method must be one of {ESTIMATORS}")


# --- C <-> HR table -----------------------------------------------------------

def mapping_table(levels=(0.55, 0.60, 0.65, 0.70, 0.75), censor_rate: float = 0.20,
                  design: StudyDesign | None = None, alpha: float = 0.05,
                  power: float = 0.80) -> pd.DataFrame:
    """C <-> per-SD HR table: for each target concordance, the exact
    beta and HR, the censoring rate lam_c calibrated to `censor_rate`, the
    large-sample limit of Harrell's C under that censoring, and the
    closed-form events D and enrolled N for `power` to detect beta != 0 with
    the two-sided Wald test.

    With a `design`, its baseline hazard and censoring (fixed rate or
    accrual / follow-up / dropout) replace `censor_rate`. Values are unrounded."""
    model = dict(lambda0=DEFAULT_LAMBDA0) if design is None else design.baseline_kwargs()
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    rows = []
    for c in levels:
        b = beta_of_c(c)
        hr = float(np.exp(b))
        if design is None:
            rate = censor_rate
            lam_c = calibrate_censoring_rate(b, rate, **model)
        else:
            rate, lam_c = design.censor_rate_for(hr), design.lambda_c_for(hr)
        d = z ** 2 / b ** 2
        rows.append(dict(C=float(c), beta=b, hr=hr, lam_c=lam_c,
                         harrell_limit=harrell_limit(b, lam_c, **model),
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
    """Harrell's C (higher eta = higher risk; no ties) with its asymptotic SE."""
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


def _as_matrix(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[:, None] if x.ndim == 1 else x


def gonen_heller_c_with_se(X, beta, cov=None, bandwidth: float | None = None
                           ) -> tuple[float, float]:
    """Gonen & Heller's (2005) concordance probability estimate (CPE) of a Cox
    model with covariate matrix X (n x p, or a 1-D vector) and coefficients
    beta (length p), with its asymptotic SE.

    CPE = (n choose 2)^-1 sum_{i<j} expit(|eta_i - eta_j|), eta = X beta: the
    model-implied probability that the member of a pair with the higher risk
    fails first. It uses no survival times, so censoring does not bias it.

    The SE follows the paper: the U-statistic (Hajek projection) variance of
    the smoothed CPE, where the indicator I(eta_i > eta_j) is replaced by
    Phi((eta_i - eta_j) / h) with h = 0.5 sd(eta) n^(-1/3), plus the
    delta-method term g' cov g for the uncertainty of beta_hat (g = gradient
    of the smoothed CPE in beta). Without `cov`, beta is treated as known.
    """
    X = _as_matrix(X)
    beta = np.atleast_1d(np.asarray(beta, dtype=float))
    if X.shape[1] != beta.size:
        raise ValueError(f"X has {X.shape[1]} column(s) but beta has {beta.size}")
    n = X.shape[0]
    if n < 3 or not np.all(np.isfinite(beta)):
        return float("nan"), float("nan")
    eta = X @ beta
    D = eta[None, :] - eta[:, None]          # D[i, j] = eta_j - eta_i
    npairs = n * (n - 1)                     # ordered pairs (each unordered pair twice)
    E = expit(np.abs(D))
    np.fill_diagonal(E, 0.0)
    c = float(E.sum() / npairs)

    if bandwidth is None:
        bandwidth = 0.5 * float(np.std(eta, ddof=1)) * n ** (-1 / 3)
    if not bandwidth > 0:                    # all eta equal (e.g. beta = 0)
        return c, float("nan")
    u = D / bandwidth
    # smoothed kernel f(d) = Phi(-d/h) expit(-d) + Phi(d/h) expit(d), symmetric in d
    pp, ep = ndtr(u), expit(D)
    pm, em = 1.0 - pp, 1.0 - ep
    F = pm * em + pp * ep
    np.fill_diagonal(F, 0.0)
    c_s = F.sum() / npairs
    # U-statistic variance 4/n Var(h1), h1(i) = mean_j f(eta_j - eta_i)
    h1 = F.sum(1) / (n - 1)
    var = 4.0 / n * np.sum((h1 - c_s) ** 2) / (n - 1)
    if cov is not None:
        cov = np.atleast_2d(np.asarray(cov, dtype=float))
        # f'(d), antisymmetric in (i, j)
        G = np.exp(-0.5 * u * u) / (np.sqrt(2 * np.pi) * bandwidth) * (ep - em) + (pp - pm) * ep * em
        np.fill_diagonal(G, 0.0)
        # dCPE/dbeta = sum_{i != j} G_ij (x_j - x_i) / npairs = -2 sum_i (sum_j G_ij) x_i / npairs
        grad = -2.0 * (G.sum(1) @ X) / npairs
        var += float(grad @ cov @ grad)
    return c, float(np.sqrt(var))


def fit_cox_coefficients(time, event, X) -> tuple[np.ndarray, np.ndarray]:
    """Cox coefficients and their covariance matrix for covariates X (n x p,
    or 1-D): the fast solver for one covariate, lifelines otherwise."""
    X = _as_matrix(X)
    if X.shape[1] == 1:
        b, se = cox_fit_1d(time, event, X[:, 0])
        return np.array([b]), np.array([[se ** 2]])
    from lifelines import CoxPHFitter
    cols = [f"x{k}" for k in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols).assign(time=np.asarray(time),
                                              event=np.asarray(event).astype(int))
    cph = CoxPHFitter().fit(df, duration_col="time", event_col="event")
    return cph.params_[cols].to_numpy(), cph.variance_matrix_.loc[cols, cols].to_numpy()


def concordance_index(time, event, eta, method: str = "harrell",
                      tau: float | None = None, beta=None, cov=None) -> tuple[float, float]:
    """Concordance index and its asymptotic SE.

    For Harrell's and Uno's C, `eta` is a risk score (higher = shorter
    survival), e.g. the biomarker or a Cox linear predictor.

    For ``method="gonen-heller"``, `eta` holds the model's covariates: a
    vector (one covariate) or an n x p matrix (multivariable model). The Cox
    model of (time, event) on them is fitted unless `beta` (and optionally
    its covariance `cov`) is given; the SE includes the uncertainty of
    beta_hat whenever a covariance is available."""
    if method == "gonen-heller":
        if beta is None:
            beta, cov = fit_cox_coefficients(time, event, eta)
        return gonen_heller_c_with_se(eta, beta, cov)
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
    baseline=None,
) -> dict:
    """One simulation cell: cohorts of size n with per-SD log-HR
    beta and exponential censoring rate lam_c, with an exponential baseline
    of rate `lambda0` or the given `baseline`. Records

    * power_cox  : Cox Wald test of beta = 0 (`sides`, level alpha)
    * power_c05  : test of C = 0.5 from the estimated C and its SE (`sides`)
    * power_cc0  : one-sided test of H0: C <= c0 at level alpha (if c0 given)

    plus the mean / SD of the estimated C, its mean SE, events and mean HR,
    each Monte Carlo estimate with a `*_mcse` column. Cohorts with fewer than
    two events are skipped (counted in `n_failed`, not rejected).
    The biomarker is oriented by the sign of beta (risk score = sign(beta) X).
    The Gonen-Heller C comes from the fitted Cox model and is >= 0.5 by
    construction (it orients itself by the sign of beta_hat).
    """
    if method == "uno" and tau is None:
        tau = default_uno_tau(beta, lam_c, lambda0, baseline=baseline)
    s = np.sign(beta) if beta != 0 else 1.0
    z_one = stats.norm.ppf(1 - alpha)
    rng = np.random.default_rng(seed)
    rej_cox = rej_c05 = rej_cc0 = 0
    cs, ses, bh, ev = [], [], [], []
    for _ in tqdm(range(n_sims), desc=f"c-index n={n}", leave=False):
        X = rng.standard_normal(n)
        T = ph_event_times(beta * X, rng, lambda0, baseline)
        C = rng.exponential(scale=1.0 / lam_c, size=n)
        t, d = np.minimum(T, C), T <= C
        ev.append(d.sum())
        if d.sum() < 2:
            continue
        b, se_b = cox_fit_1d(t, d, X)
        if method == "gonen-heller":
            c, se_c = gonen_heller_c_with_se(X, b, se_b ** 2)
        else:
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
        **censoring_columns(ev, n),
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
            lambda0: float = DEFAULT_LAMBDA0, n: int = 2000, n_sims: int | None = None,
            seed: int = 99, baseline=None) -> float:
    """Asymptotic SD of sqrt(N) * C_hat, sigma_C = lim sqrt(N Var(C_hat)),
    estimated from the analytic SE in `n_sims` large cohorts of size n
    (the analytic SE is far less noisy than the empirical SD at N = 3000, and
    agrees with it within a few percent). `n_sims`
    defaults to 40, or 10 for Gonen-Heller's C, whose SE varies little
    between cohorts (~1.5% per draw) but costs ~10x more per cohort."""
    if n_sims is None:
        n_sims = 10 if method == "gonen-heller" else 40
    out = _se_draws(n, beta, lam_c, method, tau, lambda0, n_sims, seed, baseline)
    return float(np.sqrt(n) * np.sqrt(np.mean(out ** 2)))


def _se_draws(n, beta, lam_c, method, tau, lambda0, n_sims, seed, baseline=None):
    if method == "uno" and tau is None:
        tau = default_uno_tau(beta, lam_c, lambda0, baseline=baseline)
    s = np.sign(beta) if beta != 0 else 1.0
    rng = np.random.default_rng(seed)
    ses = []
    for _ in range(n_sims):
        X = rng.standard_normal(n)
        T = ph_event_times(beta * X, rng, lambda0, baseline)
        C = rng.exponential(scale=1.0 / lam_c, size=n)
        if method == "gonen-heller":
            ses.append(concordance_index(np.minimum(T, C), T <= C, X, method=method)[1])
        else:
            ses.append(concordance_index(np.minimum(T, C), T <= C, s * X, method=method,
                                         tau=tau)[1])
    return np.asarray(ses)


# --- Closed forms -------------------------------------------------------------

def benchmark_power(n: int, c_limit: float, c0: float, sigma: float, alpha: float = 0.05) -> float:
    """Approximate power of the one-sided test of H0: C <= c0 at level alpha."""
    return float(stats.norm.cdf((c_limit - c0) * np.sqrt(n) / sigma - stats.norm.ppf(1 - alpha)))


def benchmark_sample_size(c_limit: float, c0: float, sigma: float, power: float = 0.80,
                          alpha: float = 0.05) -> int:
    """N for the one-sided benchmark test. Use the estimator's
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
