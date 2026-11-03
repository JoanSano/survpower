"""Cox model fitting and Wald-test decisions.

Two engines are available for univariate simulations:

* ``"fast"`` -- `cox_fit_1d`, a vectorized Newton-Raphson partial-likelihood
  solver for one continuous covariate without tied times (agrees with
  lifelines to ~1e-7; 10-50x faster). Default for univariate power runs.
* ``"lifelines"`` -- `lifelines.CoxPHFitter`. Used for multivariable models,
  for the Grambsch-Therneau PH test, and to reproduce the report exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import norm

ENGINES = ("fast", "lifelines")


def cox_fit_1d(time: np.ndarray, event: np.ndarray, x: np.ndarray,
               max_iter: int = 50, tol: float = 1e-9) -> tuple[float, float]:
    """Univariate Cox partial-likelihood MLE (Breslow = exact without ties).

    Returns (beta_hat, se(beta_hat)) from the observed information, or
    (nan, nan) if there are fewer than two events.
    """
    event = np.asarray(event).astype(bool)
    if event.sum() < 2:
        return float("nan"), float("nan")
    o = np.argsort(-np.asarray(time), kind="mergesort")  # descending: cumsums are risk sets
    d, x = event[o], np.asarray(x, dtype=float)[o]
    b = 0.0
    info = np.nan
    for _ in range(max_iter):
        bx = b * x
        w = np.exp(bx - bx.max())
        s0, s1, s2 = np.cumsum(w), np.cumsum(w * x), np.cumsum(w * x * x)
        xbar = s1[d] / s0[d]
        score = np.sum(x[d] - xbar)
        info = np.sum(s2[d] / s0[d] - xbar ** 2)
        if not info > 0:
            return float("nan"), float("nan")
        step = score / info
        b += float(np.clip(step, -1.0, 1.0))
        if abs(step) < tol:
            break
    return b, float(1.0 / np.sqrt(info))


def wald_rejects(z: float, beta: float, alpha: float, sides: int = 2) -> bool:
    """Wald decision from a z statistic. One-sided tests reject only in the
    direction of the true effect (sign of `beta`; beta = 0 counts as positive)."""
    if not np.isfinite(z):
        return False
    if sides == 2:
        return bool(abs(z) > norm.ppf(1 - alpha / 2))
    direction = 1.0 if beta >= 0 else -1.0
    return bool(direction * z > norm.ppf(1 - alpha))


def rejects(cph: CoxPHFitter, beta: float, alpha: float, sides: int = 2) -> bool:
    """Wald rejection of H0: beta_X = 0 from a fitted lifelines model. Two-sided
    uses lifelines' p-value directly (as the report does)."""
    if sides == 2:
        return bool(cph.summary.loc["X", "p"] < alpha)
    return wald_rejects(cph.summary.loc["X", "z"], beta, alpha, sides)


def fit_cox(df: pd.DataFrame, penalizer_fallback: float = 1e-3) -> tuple[CoxPHFitter, bool]:
    """Fit `time ~ all other columns` with lifelines; if the unpenalized fit
    fails (near-singular information at high p / high R^2), refit with a
    small ridge penalty. Returns (model, used_fallback)."""
    try:
        return CoxPHFitter().fit(df, duration_col="time", event_col="event"), False
    except Exception:
        cph = CoxPHFitter(penalizer=penalizer_fallback)
        return cph.fit(df, duration_col="time", event_col="event"), True
