"""Performance measures of a simulation study, with Monte Carlo SEs.

Formulas follow Morris, White & Crowther (2019, Table 6). Every engine
reports flat columns `x` and `x_mcse`, plus `n_sims`; replicates whose fit
failed (non-finite beta_hat) are counted in `n_failed`, still count as "not
rejected" in power, and are excluded from the other moments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class Estimate:
    """A Monte Carlo estimate with its Monte Carlo standard error."""

    value: float
    mcse: float

    def __float__(self) -> float:
        return float(self.value)


def proportion_mcse(p_hat: float, n: int) -> float:
    """MCSE of a simulated proportion (power, type-I error, coverage):
    sqrt(p(1 - p) / n). Exactly 0 at p in {0, 1}; report a Wilson interval
    (`_utils.wilson_ci`) there instead."""
    return float(np.sqrt(p_hat * (1 - p_hat) / n))


def _finite(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def mean_mcse(x) -> float:
    """MCSE of a mean over replicates, SD(x) / sqrt(n); non-finite values dropped."""
    x = _finite(x)
    return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def sd_mcse(x) -> float:
    """MCSE of the empirical SD over replicates, SD(x) / sqrt(2(n - 1))."""
    x = _finite(x)
    return float(np.std(x, ddof=1) / np.sqrt(2 * (len(x) - 1))) if len(x) > 1 else float("nan")


def exp_mean_mcse(x) -> float:
    """Delta-method MCSE of exp(mean(x)), e.g. the mean HR from log-HRs."""
    f = _finite(x)
    return float(np.exp(np.mean(f)) * mean_mcse(f)) if len(f) > 1 else float("nan")


def summarize_replicates(
    reject: np.ndarray,
    beta_hat: np.ndarray | None = None,
    se_hat: np.ndarray | None = None,
    beta_true: float | None = None,
    events: np.ndarray | None = None,
    alpha: float = 0.05,
    n: int | None = None,
) -> dict:
    """Power, bias, empirical SE, mean model SE, relative error of the model
    SE, coverage of the (1 - alpha) Wald interval and mean events, each with
    its MCSE where one applies; with the cohort size `n` too, the realized
    censored fraction (`censoring_columns`). Columns appear only when their
    inputs are given; values are unrounded."""
    reject = np.asarray(reject, dtype=bool)
    n_sims = len(reject)
    power = float(np.mean(reject))
    out = dict(power=power, power_mcse=proportion_mcse(power, n_sims), n_sims=n_sims, n_failed=0)
    if beta_hat is not None:
        beta_hat = np.asarray(beta_hat, dtype=float)
        ok = np.isfinite(beta_hat)
        out["n_failed"] = int(n_sims - ok.sum())
        b = beta_hat[ok]
        if beta_true is not None:
            out.update(bias_beta=float(np.mean(b) - beta_true), bias_beta_mcse=mean_mcse(b))
        out.update(emp_se=float(np.std(b, ddof=1)) if len(b) > 1 else float("nan"),
                   emp_se_mcse=sd_mcse(b))
        if se_hat is not None:
            se = np.asarray(se_hat, dtype=float)[ok]
            mean_se = float(np.mean(se))
            out.update(mean_model_se=mean_se, rel_err_model_se=mean_se / out["emp_se"] - 1)
            if beta_true is not None:
                z = norm.ppf(1 - alpha / 2)
                cov = float(np.mean(np.abs(b - beta_true) <= z * se))
                out.update(coverage=cov, coverage_mcse=proportion_mcse(cov, len(b)))
    if events is not None:
        out["mean_events"] = float(np.mean(events))
        if n is not None:
            out.update(censoring_columns(events, n))
    return out


def censoring_columns(events, n: int) -> dict:
    """The realized censored fraction, 1 - events / n averaged over replicates, and
    its MCSE (`censor_rate_realized`, `censor_rate_realized_mcse`): the check on a
    calibrated censoring model, and the result of an uncalibrated one (a trial
    calendar), whose fraction depends on the effect and the baseline."""
    censored = 1.0 - np.asarray(events, dtype=float) / n
    return dict(censor_rate_realized=float(np.mean(censored)),
                censor_rate_realized_mcse=mean_mcse(censored))
