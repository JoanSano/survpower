"""Composable simulation scenarios (provisional API; may change before 1.0).

A `Scenario` names one model per step of the Monte Carlo loop::

    covariates -> event times -> censoring -> analyses

and `simulate` runs it. A new data-generating option (baseline hazard,
censoring shape, covariate distribution, ...) is a new model class, not a new
hand-written loop. Models are plain objects that satisfy the protocols below;
nothing has to inherit from a library class.

Example::

    from survpower.scenario import Scenario, ExponentialPH, ExponentialCensoring, simulate
    sc = Scenario(event_times=ExponentialPH(log_hr=np.log(1.15)),
                  censoring=ExponentialCensoring(target_rate=0.2))
    simulate(sc, n=250, n_sims=1000, seed=1)   # one row per analysis

Reproducibility: the same seed gives the same result within one installed
version. Censoring is calibrated on a probe sample drawn with ``seed + 999``;
replicates use ``seed``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from ._utils import tqdm
from .baseline import (DEFAULT_LAMBDA0, BaselineHazard, is_exponential, marginal_median,
                       median_os_to_lambda0)
from .censoring import (calibrate_censoring_for, dropout_scale, piecewise_cumhaz,
                        piecewise_inv_cumhaz, require_censoring_above_cure)
from .dgp import gen_adjustment_covariates, ph_event_times
from .fitting import ENGINES, cox_fit_1d, fit_cox, logrank_1d, rejects, wald_rejects
from .results import summarize_replicates

#: Seed offset of the censoring-calibration probe sample.
CALIBRATION_SEED_OFFSET = 999


@dataclass(frozen=True)
class Covariates:
    """One cohort's covariates: the biomarker `X` (n,), optional adjustment
    covariates `Z` (n, p), and model-specific `meta` (e.g. realized R^2)."""

    X: np.ndarray
    Z: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.X)


Sampler = Callable[[int, np.random.Generator], "tuple[np.ndarray, Covariates]"]


@runtime_checkable
class CovariateModel(Protocol):
    def draw(self, n: int, rng: np.random.Generator) -> Covariates: ...


@runtime_checkable
class EventTimeModel(Protocol):
    """Latent event times given covariates. `is_ph` says whether the model is
    proportional hazards in X. A model with a single true log-HR for X exposes
    it as `log_hr`; `simulate` then reports bias and coverage against it and
    uses its sign as the direction of one-sided tests."""

    is_ph: bool

    def draw(self, cov: Covariates, rng: np.random.Generator) -> np.ndarray: ...

    def survival(self, t, x) -> np.ndarray: ...


@runtime_checkable
class CensoringModel(Protocol):
    """Censoring times for a cohort. `draw` also receives the cohort's latent
    event times, so that a model can depend on them (an analysis cut at a target
    number of events); independent censoring ignores them."""

    def calibrated(self, sampler: Sampler, seed: int) -> "CensoringModel":
        """A frozen copy whose parameters are fitted to the event times that
        `sampler(n, rng)` draws (the scenario's covariate and event-time models)."""
        ...

    def draw(self, cov: Covariates, event_times: np.ndarray,
             rng: np.random.Generator) -> np.ndarray: ...


@runtime_checkable
class Analysis(Protocol):
    """Fits one model to one replicate. Returns a dict with at least
    ``reject`` (bool); Wald-type analyses add ``beta_hat``, ``se`` and ``z``.
    Extra keys are kept per replicate (see `Replicates.values`)."""

    name: str

    def __call__(self, time: np.ndarray, event: np.ndarray, cov: Covariates, *,
                 alpha: float, sides: int, beta: float) -> dict: ...


# --- Covariate models ---------------------------------------------------------

@dataclass(frozen=True)
class StandardNormal:
    """X ~ N(0, 1), no adjustment covariates."""

    def draw(self, n: int, rng: np.random.Generator) -> Covariates:
        return Covariates(rng.standard_normal(n))


@dataclass(frozen=True)
class CorrelatedCovariates:
    """X ~ N(0, 1) plus `p` adjustment covariates Z whose multiple R^2 on X is
    `r2` (`dgp.gen_adjustment_covariates`). The Z carry no effect of their own:
    they enter the hazard only through an event-time model that uses them."""

    p: int = 1
    r2: float = 0.0

    def __post_init__(self):
        if not (isinstance(self.p, (int, np.integer)) and self.p >= 0):
            raise ValueError("p must be a non-negative integer")
        if not 0 <= self.r2 < 1:
            raise ValueError("r2 must be in [0, 1)")
        if self.r2 > 0 and self.p == 0:
            raise ValueError("r2 requires p >= 1")

    def draw(self, n: int, rng: np.random.Generator) -> Covariates:
        X, Z = gen_adjustment_covariates(n, self.r2, self.p, rng)
        return Covariates(X, Z)


# --- Event-time models ----------------------------------------------------------

@dataclass(frozen=True)
class ExponentialPH:
    """T | X ~ Exponential(lambda0 * exp(log_hr * X)): a Cox PH model with a
    constant baseline hazard."""

    log_hr: float = 0.0
    lambda0: float = DEFAULT_LAMBDA0
    is_ph: bool = field(default=True, init=False)

    @classmethod
    def from_median_os(cls, log_hr: float, median_os: float) -> "ExponentialPH":
        return cls(log_hr=log_hr, lambda0=median_os_to_lambda0(median_os))

    def _rate(self, x) -> np.ndarray:
        return self.lambda0 * np.exp(self.log_hr * np.asarray(x, dtype=float))

    def draw(self, cov: Covariates, rng: np.random.Generator) -> np.ndarray:
        return rng.exponential(scale=1.0 / self._rate(cov.X))

    def survival(self, t, x) -> np.ndarray:
        return np.exp(-self._rate(x) * np.asarray(t, dtype=float))


@dataclass(frozen=True)
class PHModel:
    """Cox PH with any baseline hazard (`survpower.baseline`): hazard
    h0(t) exp(log_hr * X), drawn as T = H0^{-1}(E exp(-log_hr * X)) with
    E ~ Exp(1). A defective baseline (Gompertz with shape < 0) gives T = inf
    for the cured fraction.

    With an `Exponential` baseline it makes the same draws as
    ``ExponentialPH(b, ln 2 / m)`` (`dgp.ph_event_times`), which `scenario_for`
    still builds for the default design."""

    baseline: BaselineHazard
    log_hr: float = 0.0
    is_ph: bool = field(default=True, init=False)

    def draw(self, cov: Covariates, rng: np.random.Generator) -> np.ndarray:
        return ph_event_times(self.log_hr * cov.X, rng, baseline=self.baseline)

    def survival(self, t, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.exp(-self.baseline.cumhaz(t) * np.exp(self.log_hr * x))

    def marginal_median(self) -> float:
        """Median of T over X ~ N(0, 1): differs from the baseline (X = 0)
        median `baseline.median` once log_hr != 0."""
        return marginal_median(self.baseline, self.log_hr)


# --- Censoring models -------------------------------------------------------------

@dataclass(frozen=True)
class ExponentialCensoring:
    """C ~ Exponential(rate), independent of T. `calibrated` sets `rate` so
    that the marginal P(C < T) equals `target_rate`; a
    given `rate` is used as is."""

    target_rate: float = 0.20
    rate: float | None = None
    n_probe: int = 500_000

    def calibrated(self, sampler: Sampler, seed: int) -> "ExponentialCensoring":
        if self.rate is not None:
            return self
        lam_c = calibrate_censoring_for(lambda n, rng: sampler(n, rng)[0],
                                        self.target_rate, seed=seed, n_probe=self.n_probe)
        return replace(self, rate=lam_c)

    def draw(self, cov: Covariates, event_times: np.ndarray,
             rng: np.random.Generator) -> np.ndarray:
        if self.rate is None:
            raise ValueError("ExponentialCensoring has no rate; call calibrated() first")
        return rng.exponential(scale=1.0 / self.rate, size=cov.n)


class _ScaledCumhazCensoring:
    """Censoring with cumulative hazard H_C(t) = rate * R(t), R fixed by the shape
    parameters. `calibrated` sets `rate` so that P(C < T) = `target_rate`
    (`censoring.calibrate_cumhaz_scale`: unique root, since P(C < T) is strictly
    increasing in the rate); a given `rate` is used as is. C = R^{-1}(E / rate),
    E ~ Exp(1)."""

    def rel_cumhaz(self, t) -> np.ndarray: ...

    def rel_inv_cumhaz(self, h) -> np.ndarray: ...

    def _check_target(self):
        if not 0 < self.target_rate < 1:
            raise ValueError("target_rate must be in (0, 1)")
        if self.rate is not None and not 0 < self.rate < np.inf:
            raise ValueError("rate must be positive and finite")

    def calibrated(self, sampler: Sampler, seed: int):
        if self.rate is not None:
            return self
        rate = calibrate_censoring_for(lambda n, rng: sampler(n, rng)[0], self.target_rate,
                                       seed=seed, n_probe=self.n_probe,
                                       rel_cumhaz=self.rel_cumhaz)
        return replace(self, rate=rate)

    def draw(self, cov: Covariates, event_times: np.ndarray,
             rng: np.random.Generator) -> np.ndarray:
        if self.rate is None:
            raise ValueError(f"{type(self).__name__} has no rate; call calibrated() first")
        return self.rel_inv_cumhaz(rng.standard_exponential(cov.n) / self.rate)


@dataclass(frozen=True)
class WeibullCensoring(_ScaledCumhazCensoring):
    """Weibull censoring, H_C(t) = rate * t^shape (t in months), independent of T,
    calibrated to `target_rate`. shape > 1 puts censoring late (a rising censoring
    hazard), shape < 1 early; shape 1 is exponential censoring (but calibrated
    without Monte Carlo noise, so not bit-identical to `ExponentialCensoring`)."""

    target_rate: float = 0.20
    shape: float = 1.0
    rate: float | None = None
    n_probe: int = 500_000

    def __post_init__(self):
        self._check_target()
        if not 0 < self.shape < np.inf:
            raise ValueError("shape must be positive and finite")

    def rel_cumhaz(self, t):
        return np.asarray(t, dtype=float) ** self.shape

    def rel_inv_cumhaz(self, h):
        return np.asarray(h, dtype=float) ** (1.0 / self.shape)


@dataclass(frozen=True)
class PiecewiseCensoring(_ScaledCumhazCensoring):
    """Piecewise-constant censoring hazard, independent of T: proportional to
    `relative_rates` (one more than `breaks`, in months), scaled by `rate` and
    calibrated to `target_rate`. E.g. ``breaks=(24,), relative_rates=(1, 4)``
    censors four times as fast after two years."""

    target_rate: float = 0.20
    breaks: tuple = ()
    relative_rates: tuple = (1.0,)
    rate: float | None = None
    n_probe: int = 500_000

    def __post_init__(self):
        self._check_target()
        breaks = tuple(float(b) for b in self.breaks)
        rates = tuple(float(r) for r in self.relative_rates)
        object.__setattr__(self, "breaks", breaks)
        object.__setattr__(self, "relative_rates", rates)
        if len(rates) != len(breaks) + 1:
            raise ValueError("relative_rates needs one more value than breaks")
        if any(not 0 < b < np.inf for b in breaks) or any(np.diff(breaks) <= 0):
            raise ValueError("breaks must be positive, finite and increasing")
        if any(not 0 < r < np.inf for r in rates):
            raise ValueError("relative_rates must be positive and finite")

    def rel_cumhaz(self, t):
        return piecewise_cumhaz(t, self.breaks, self.relative_rates)

    def rel_inv_cumhaz(self, h):
        return piecewise_inv_cumhaz(h, self.breaks, self.relative_rates)


@dataclass(frozen=True)
class CalendarCensoring:
    """The trial calendar, simulated directly: entry ~ U(0, A), analysis at A + F,
    so administrative censoring is C_adm = A + F - entry ~ U(F, A + F); with
    dropout, C = min(C_adm, D), D Weibull with H_D(t) = lambda_d t^dropout_shape
    and a 12-month loss probability of `dropout_annual` (`censoring.dropout_scale`).
    Months throughout.

    Not calibrated: the censored fraction depends on the effect and the baseline,
    and is an output (``censor_rate_realized``). With exponential dropout
    (shape 1) its expectation is `censoring.censor_rate_from_design`."""

    accrual_months: float
    followup_months: float
    dropout_annual: float = 0.0
    dropout_shape: float = 1.0

    def __post_init__(self):
        A, F = self.accrual_months, self.followup_months
        if not (0 <= A < np.inf and 0 <= F < np.inf and A + F > 0):
            raise ValueError("need accrual_months >= 0, followup_months >= 0 (finite) "
                             "and accrual_months + followup_months > 0")
        if not 0 <= self.dropout_annual < 1:
            raise ValueError("dropout_annual must be in [0, 1)")
        if not 0 < self.dropout_shape < np.inf:
            raise ValueError("dropout_shape must be positive and finite")

    def calibrated(self, sampler: Sampler, seed: int) -> "CalendarCensoring":
        return self

    def draw(self, cov: Covariates, event_times: np.ndarray,
             rng: np.random.Generator) -> np.ndarray:
        A, F = self.accrual_months, self.followup_months
        C = rng.uniform(F, A + F, size=cov.n)
        if self.dropout_annual > 0:
            lam_d = dropout_scale(self.dropout_annual, self.dropout_shape)
            if self.dropout_shape == 1.0:
                D = rng.exponential(scale=1.0 / lam_d, size=cov.n)
            else:
                D = (rng.standard_exponential(cov.n) / lam_d) ** (1.0 / self.dropout_shape)
            C = np.minimum(C, D)
        return C


@dataclass(frozen=True)
class EmpiricalCensoring:
    """The censoring distribution of a cohort, estimated by reverse Kaplan-Meier
    (censoring as the event): build it with `from_data`.

    C takes the cohort's distinct censoring times (`times`) with cumulative
    probabilities `cdf`. The probability that the reverse Kaplan-Meier leaves past
    the last censoring time is put at the largest follow-up in the data (`times[-1]`):
    nobody in the cohort was followed longer, so the simulated study ends there too.

    Not calibrated: the censored fraction is an output (``censor_rate_realized``),
    as for `CalendarCensoring`. `note` describes the source in summaries."""

    times: tuple = ()
    cdf: tuple = ()
    note: str = ""

    def __post_init__(self):
        times = tuple(float(t) for t in self.times)
        cdf = tuple(float(p) for p in self.cdf)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "cdf", cdf)
        if len(times) < 1 or len(times) != len(cdf):
            raise ValueError("times and cdf need the same length, at least 1")
        if times[0] <= 0 or any(np.diff(times) <= 0) or not np.isfinite(times[-1]):
            raise ValueError("times must be positive, finite and increasing")
        if cdf[0] <= 0 or any(np.diff(cdf) <= 0) or cdf[-1] != 1.0:
            raise ValueError("cdf must be positive, increasing and end at 1")

    @classmethod
    def from_data(cls, time, event, note: str = "") -> "EmpiricalCensoring":
        """Reverse Kaplan-Meier of a cohort's follow-up (months) and event indicator
        (1 = death, 0 = censored). At tied times deaths precede censoring, as usual,
        so a death at t is not at risk of censoring at t. Rows with a missing value
        are dropped."""
        t = np.asarray(time, dtype=float)
        d = np.asarray(event, dtype=float)
        keep = np.isfinite(t) & np.isfinite(d)
        t, d = t[keep], d[keep]
        if np.any(t <= 0):
            raise ValueError(f"follow-up times must be positive ({int(np.sum(t <= 0))} row(s) "
                             f"are not)")
        if not np.all(np.isin(d, (0, 1))):
            raise ValueError("the event indicator must be 0 or 1")
        cens = d == 0
        if not cens.any():
            raise ValueError("no censored follow-up: the censoring distribution cannot be "
                             "estimated")
        c_times, n_cens = np.unique(t[cens], return_counts=True)
        t_sorted = np.sort(t)
        # at risk of censoring at c: still followed after c, or censored at c
        at_risk = t.size - np.searchsorted(t_sorted, c_times, side="right") + n_cens
        surv = np.cumprod(1.0 - n_cens / at_risk)
        t_max = float(t_sorted[-1])
        times, cdf = list(c_times), list(1.0 - surv)
        if t_max > times[-1]:
            times.append(t_max)
            cdf.append(1.0)
        cdf[-1] = 1.0
        return cls(times=tuple(times), cdf=tuple(cdf), note=note)

    def probability_before(self, t) -> np.ndarray:
        """P(C < t), pointwise: the censoring probability of an event at t."""
        k = np.searchsorted(np.asarray(self.times), np.asarray(t, dtype=float), side="left")
        return np.concatenate([[0.0], self.cdf])[k]

    def calibrated(self, sampler: Sampler, seed: int) -> "EmpiricalCensoring":
        return self

    def draw(self, cov: Covariates, event_times: np.ndarray,
             rng: np.random.Generator) -> np.ndarray:
        k = np.searchsorted(np.asarray(self.cdf), rng.uniform(size=cov.n), side="left")
        return np.asarray(self.times)[np.minimum(k, len(self.times) - 1)]

    def label(self) -> str:
        src = f" from {self.note}" if self.note else ""
        return f"empirical (reverse Kaplan-Meier{src}; {len(self.times)} censoring times)"


# --- Analyses -----------------------------------------------------------------------

def median_split(x: np.ndarray) -> np.ndarray:
    """Group indicator 1{x > median(x)} as floats."""
    return (x > np.median(x)).astype(float)


def design_frame(time_, event, X, Z=None) -> pd.DataFrame:
    """lifelines input: ``time``, ``event``, ``X`` and ``Z0..Z{p-1}``."""
    cols = {"time": time_, "event": event, "X": X}
    if Z is not None:
        for j in range(Z.shape[1]):
            cols[f"Z{j}"] = Z[:, j]
    return pd.DataFrame(cols)


@dataclass(frozen=True)
class CoxWald:
    """Wald test of the biomarker's coefficient in a Cox model.

    ``engine="fast"`` uses `fitting.cox_fit_1d` (univariate only);
    ``"lifelines"`` fits X and every column of Z. With ``dichotomize`` the
    biomarker enters as its median split; the result then also has
    ``events_high``, the events in the high group.
    """

    engine: str = "fast"
    dichotomize: bool = False

    def __post_init__(self):
        if self.engine not in ENGINES:
            raise ValueError(f"engine must be one of {ENGINES}")

    @property
    def name(self) -> str:
        return "cox_wald" + ("_median_split" if self.dichotomize else "")

    def __call__(self, time, event, cov, *, alpha, sides, beta) -> dict:
        x = median_split(cov.X) if self.dichotomize else cov.X
        if self.engine == "fast":
            if cov.Z is not None:
                raise ValueError("the fast engine fits X alone; use engine='lifelines'")
            b, se = cox_fit_1d(time, event, x)
            z = b / se
            r = wald_rejects(z, beta, alpha, sides)
        else:
            cph, _ = fit_cox(design_frame(time, event, x, cov.Z))
            r = rejects(cph, beta, alpha, sides)
            row = cph.summary.loc["X"]
            b, se, z = row["coef"], row["se(coef)"], row["z"]
        out = dict(reject=r, beta_hat=b, se=se, z=z)
        if self.dichotomize:
            out["events_high"] = event[x == 1].sum()
        return out


@dataclass(frozen=True)
class LogRank:
    """Log-rank test of the median-split groups (default) or the Cox score
    test of the continuous biomarker (``dichotomize=False``), via
    `fitting.logrank_1d`."""

    dichotomize: bool = True

    @property
    def name(self) -> str:
        return "logrank" + ("_median_split" if self.dichotomize else "")

    def __call__(self, time, event, cov, *, alpha, sides, beta) -> dict:
        x = median_split(cov.X) if self.dichotomize else cov.X
        z = logrank_1d(time, event, x)
        return dict(reject=wald_rejects(z, beta, alpha, sides), z=z)


# --- Composition ------------------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    """One data-generating process plus the analyses applied to each replicate."""

    covariates: CovariateModel = field(default_factory=StandardNormal)
    event_times: EventTimeModel = field(default_factory=ExponentialPH)
    censoring: CensoringModel = field(default_factory=ExponentialCensoring)
    analyses: tuple = field(default_factory=lambda: (CoxWald(engine="fast"),))

    def __post_init__(self):
        names = [a.name for a in self.analyses]
        if len(set(names)) != len(names):
            raise ValueError(f"analysis names must be unique, got {names}")

    @property
    def log_hr(self) -> float | None:
        """The true log-HR of X, if the event-time model has one."""
        return getattr(self.event_times, "log_hr", None)

    @property
    def has_closed_form(self) -> bool:
        """Whether the closed-form power and sample-size formulas apply: PH only, for now."""
        return bool(self.event_times.is_ph)

    def sample_latent(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, Covariates]:
        """Covariates and latent event times (no censoring): the probe sampler."""
        cov = self.covariates.draw(n, rng)
        return self.event_times.draw(cov, rng), cov

    def calibrated(self, seed: int) -> "Scenario":
        """A copy with the censoring model calibrated on a probe sample from `seed`."""
        return replace(self, censoring=self.censoring.calibrated(self.sample_latent, seed))

    def draw_latent(self, n: int, rng: np.random.Generator):
        """One cohort before observation: (T, C, covariates), the latent event
        and censoring times. Needs a calibrated censoring model."""
        T, cov = self.sample_latent(n, rng)
        return T, self.censoring.draw(cov, T, rng), cov

    def draw(self, n: int, rng: np.random.Generator):
        """One observed cohort: (time, event, covariates)."""
        T, C, cov = self.draw_latent(n, rng)
        return np.minimum(T, C), (T <= C).astype(int), cov


#: The effect scales a model can be parametrized by. #63 adds "time_ratio".
EFFECT_SCALES = ("hr",)


@dataclass(frozen=True)
class Effect:
    """The size of the biomarker's effect: a scale and its value per SD."""

    scale: str
    per_sd: float

    def __post_init__(self):
        if self.scale not in EFFECT_SCALES:
            raise ValueError(f"scale must be one of {EFFECT_SCALES}")
        if not 0 < self.per_sd < np.inf:
            raise ValueError("per_sd must be positive and finite")

    @classmethod
    def hr(cls, per_sd: float) -> "Effect":
        """A hazard ratio per SD of the biomarker."""
        return cls("hr", per_sd)


#: The model fields of `StudyDesign` that `scenario_for` turns into models.
SCENARIO_FOR_SUPPORTS = ("baseline", "censoring")


def scenario_for(design, effect: Effect, analyses: tuple | None = None,
                 censor_rate: float | None = None) -> Scenario:
    """The scenario that a `StudyDesign` and an `Effect` describe: the one place
    that turns a design into models. Unadjusted analyses only; the other paths
    are guarded by `StudyDesign.require_default_models`.

    analyses    : default, the design's Wald test (`CoxWald(engine=design.engine)`).
    censor_rate : the target marginal censoring, for a path that has solved for
                  it; default, `design.censor_rate_for(effect.per_sd)`. Calendar and
                  empirical censoring are not calibrated and ignore it.

    A design option that this function does not turn into a model raises, so
    that it can never fall back to the exponential model without saying so.
    """
    if design.n_covariates > 0:
        raise ValueError("scenario_for covers unadjusted analyses; this design has "
                         f"n_covariates = {design.n_covariates}")
    design.require_default_models("scenario_for", supports=SCENARIO_FOR_SUPPORTS)
    if effect.scale != "hr":
        raise ValueError(f"no event-time model for an effect on the {effect.scale} scale")
    if censor_rate is None:
        censor_rate = design.censor_rate_for(effect.per_sd)
    log_hr = np.log(effect.per_sd)
    if is_exponential(design.baseline):
        event_times = ExponentialPH(log_hr=log_hr, lambda0=design.lambda0)
    else:
        if design.censoring_calibrated:
            require_censoring_above_cure(design.baseline, log_hr, censor_rate)
        event_times = PHModel(design.baseline, log_hr=log_hr)
    return Scenario(StandardNormal(), event_times, design.censoring_for(censor_rate),
                    analyses if analyses is not None else (CoxWald(engine=design.engine),))


@dataclass
class Replicates:
    """Per-replicate output of `run_replicates`: `values[name][key]` is an
    array over replicates of each key an analysis returned; `events` the
    events per replicate; `scenario` the calibrated scenario that was run."""

    scenario: Scenario
    n: int
    n_sims: int
    alpha: float
    sides: int
    events: np.ndarray
    values: dict

    def summary(self, name: str) -> dict:
        """`results.summarize_replicates` for one analysis (unrounded)."""
        v = self.values[name]
        return summarize_replicates(v["reject"], v.get("beta_hat"), v.get("se"),
                                    beta_true=self.scenario.log_hr, events=self.events,
                                    alpha=self.alpha, n=self.n)


def run_replicates(scenario: Scenario, n: int, n_sims: int = 1000, seed: int = 0,
                   alpha: float = 0.05, sides: int = 2, desc: str | None = None) -> Replicates:
    """Calibrate `scenario` (probe seed ``seed + 999``), then draw `n_sims`
    cohorts of size `n` from ``default_rng(seed)`` and apply every analysis
    to each. Analyses must not consume random numbers."""
    if sides not in (1, 2):
        raise ValueError("sides must be 1 or 2")
    sc = scenario.calibrated(seed + CALIBRATION_SEED_OFFSET)
    beta = sc.log_hr if sc.log_hr is not None else 0.0
    rng = np.random.default_rng(seed)
    events = []
    rows = {a.name: [] for a in sc.analyses}
    for _ in tqdm(range(n_sims), desc=desc or f"scenario n={n}", leave=False):
        time_, event, cov = sc.draw(n, rng)
        events.append(event.sum())
        for a in sc.analyses:
            rows[a.name].append(a(time_, event, cov, alpha=alpha, sides=sides, beta=beta))
    values = {name: {k: np.array([r[k] for r in rs]) for k in (rs[0] if rs else {})}
              for name, rs in rows.items()}
    return Replicates(sc, n, n_sims, alpha, sides, np.array(events), values)


def simulate_cohort(scenario: Scenario, n: int, seed: int = 0) -> pd.DataFrame:
    """One simulated cohort of size `n` from `scenario`'s data-generating process.

    The censoring model is calibrated first, exactly as in `simulate` (probe seed
    ``seed + 999``), and the cohort is drawn from ``default_rng(seed)``: it is the
    first replicate that ``simulate(scenario, n, seed=seed)`` analyses.

    Columns: ``time`` = min(T, C), ``event`` = 1{T <= C} (0/1), ``X``,
    ``Z0..Z{p-1}`` (only when the covariate model draws adjustment covariates),
    then ``T`` and ``C``.

    ``T`` (latent event time) and ``C`` (latent censoring time) are never both
    observed in a real study: only ``time`` and ``event`` are. They are returned
    so that the data-generating model itself can be checked; an analysis of the
    cohort should use ``time``, ``event`` and the covariates only.
    """
    if n < 1:
        raise ValueError("n must be a positive integer")
    sc = scenario.calibrated(seed + CALIBRATION_SEED_OFFSET)
    T, C, cov = sc.draw_latent(n, np.random.default_rng(seed))
    frame = design_frame(np.minimum(T, C), (T <= C).astype(int), cov.X, cov.Z)
    frame["T"] = T
    frame["C"] = C
    return frame


def simulate(scenario: Scenario, n: int, n_sims: int = 1000, seed: int = 0,
             alpha: float = 0.05, sides: int = 2) -> pd.DataFrame:
    """Monte Carlo performance of each analysis in `scenario`.

    Returns one row per analysis: ``analysis``, ``n`` and the
    `results.summarize_replicates` columns (power, bias, empirical / model SE
    and coverage, each with its MCSE where it applies; bias and coverage only
    when the event-time model has a `log_hr`), the realized censored fraction
    ``censor_rate_realized`` with its MCSE, plus the censoring scale the
    calibration found (``lambda_c``) when the censoring model has one.
    """
    reps = run_replicates(scenario, n, n_sims, seed, alpha, sides)
    lam_c = getattr(reps.scenario.censoring, "rate", None)
    rows = []
    for a in reps.scenario.analyses:
        row = dict(analysis=a.name, n=n, **reps.summary(a.name))
        if lam_c is not None:
            row["lambda_c"] = lam_c
        rows.append(row)
    return pd.DataFrame(rows)
