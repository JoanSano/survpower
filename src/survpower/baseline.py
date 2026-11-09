"""Baseline hazards, calibrated to a median overall survival.

The default data-generating process is a constant ("exponential") baseline
hazard h0(t) = lambda0, calibrated so the baseline (X = 0) median survival
equals a realistic median overall survival:

    lambda0 = ln(2) / median_os

lambda0 = 1 recovers an arbitrary-time-unit model and lambda0 -> 0 the
degenerate "no baseline risk" limit. Power and required-event results for a
PH model are exactly invariant to lambda0 (Cox regression only uses the
ordering of times); only time-denominated quantities (censoring rates, fixed
clinical windows such as tau) change.

Other shapes of the baseline hazard (#14) are classes with a common base,
each calibrated in closed form so that its baseline (X = 0) median is
`median_os`:

    Exponential(median_os)                          h0 constant
    Weibull(median_os, shape)                       H0 = ln2 (t / m)^k
    Gompertz(median_os, shape)                      h0 = lam e^{a t}, a = shape per month
    PiecewiseExponential(median_os, breaks, relative_rates)
    LogLogistic(median_os, shape)                   S0 = 1 / (1 + (t / m)^k)
    NumericBaseline(hazard, median_os)              any hazard shape, by quadrature

Each gives the cumulative hazard H0 and its inverse, which is all that
inverse-cumulative-hazard sampling needs (Bender, Augustin & Blettner 2005):
under PH, T = H0^{-1}(E e^{-beta X}) with E ~ Exp(1) (`scenario.PHModel`).

Under a non-exponential baseline, a fixed censoring *fraction* with
exponential censoring is not a time-rescaled copy of the exponential design,
so power differs slightly between baselines at the same median and censoring.
That difference is real, not an error to remove.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Callable, ClassVar, Protocol, runtime_checkable

import numpy as np
from scipy import optimize

DEFAULT_MEDIAN_OS_MONTHS = 15.0  # typical glioblastoma median overall survival
DEFAULT_TAU_MONTHS = 6.0         # early-risk window for the non-PH scenarios
DEFAULT_CENSOR_RATE = 0.20

LN2 = float(np.log(2.0))


def median_os_to_lambda0(median_os: float) -> float:
    """Rate of a constant baseline hazard with the given median survival
    (same time unit as `median_os`; months throughout this package)."""
    return float(np.log(2.0) / median_os)


DEFAULT_LAMBDA0 = median_os_to_lambda0(DEFAULT_MEDIAN_OS_MONTHS)  # ~0.04621 / month


def median_survival(hr: float, lambda0: float = DEFAULT_LAMBDA0) -> float:
    """Median survival for a subject whose hazard is `hr` times the baseline."""
    return float(np.log(2.0) / (lambda0 * hr))


# --- Baseline-hazard families ------------------------------------------------------

@runtime_checkable
class BaselineHazard(Protocol):
    """What an event-time model needs from a baseline hazard. `inv_cumhaz` may
    return inf, for a defective distribution (a cure fraction)."""

    @property
    def median(self) -> float: ...

    def cumhaz(self, t) -> np.ndarray: ...

    def inv_cumhaz(self, h) -> np.ndarray: ...


def _positive(name: str, value: float) -> None:
    if not 0 < value < np.inf:
        raise ValueError(f"{name} must be positive and finite")


class _Baseline:
    """Common base of the families: survival, density, cure fraction, JSON form.
    Subclasses are frozen dataclasses whose first field is `median_os`."""

    family: ClassVar[str]
    #: An API-only model cannot be written to JSON or the CLI (it holds a function).
    api_only: ClassVar[bool] = False

    def __post_init__(self):
        _positive("median_os", self.median_os)
        object.__setattr__(self, "median_os", float(self.median_os))
        if hasattr(self, "shape"):
            object.__setattr__(self, "shape", float(self.shape))

    @property
    def median(self) -> float:
        """Baseline (X = 0) median survival, months."""
        return float(self.median_os)

    @property
    def cumhaz_max(self) -> float:
        """H0(infinity): infinite unless the distribution is defective."""
        return np.inf

    @property
    def cure_fraction(self) -> float:
        """Baseline probability of never having the event, exp(-H0(infinity))."""
        return float(np.exp(-self.cumhaz_max))

    def survival(self, t) -> np.ndarray:
        return np.exp(-self.cumhaz(t))

    def density(self, t) -> np.ndarray:
        return self.hazard(t) * self.survival(t)

    def to_dict(self) -> dict:
        """JSON-ready form, rebuilt by `baseline_from_dict`."""
        if self.api_only:
            raise TypeError(f"{type(self).__name__} is API-only: it cannot be saved as JSON")
        out = {"family": self.family}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = list(value) if isinstance(value, tuple) else value
        return out

    def label(self) -> str:
        """Short description for summaries."""
        return self.family


@dataclass(frozen=True)
class Exponential(_Baseline):
    """Constant baseline hazard ln 2 / median_os: the package default."""

    median_os: float = DEFAULT_MEDIAN_OS_MONTHS
    family: ClassVar[str] = "exponential"

    @property
    def rate(self) -> float:
        """lambda0, per month."""
        return median_os_to_lambda0(self.median_os)

    def hazard(self, t):
        return np.full(np.shape(t), self.rate)

    def cumhaz(self, t):
        return self.rate * np.asarray(t, dtype=float)

    def inv_cumhaz(self, h):
        return np.asarray(h, dtype=float) / self.rate


@dataclass(frozen=True)
class Weibull(_Baseline):
    """H0(t) = lam t^k with lam = ln 2 / m^k: hazard rising (k > 1) or falling
    (k < 1) over time; k = 1 is exponential."""

    median_os: float = DEFAULT_MEDIAN_OS_MONTHS
    shape: float = 1.0
    family: ClassVar[str] = "weibull"

    def __post_init__(self):
        super().__post_init__()
        _positive("shape", self.shape)

    def hazard(self, t):
        t = np.asarray(t, dtype=float)
        k, m = self.shape, self.median_os
        with np.errstate(divide="ignore"):
            return LN2 * k / m * (t / m) ** (k - 1)

    def cumhaz(self, t):
        return LN2 * (np.asarray(t, dtype=float) / self.median_os) ** self.shape

    def inv_cumhaz(self, h):
        return self.median_os * (np.asarray(h, dtype=float) / LN2) ** (1.0 / self.shape)

    def label(self) -> str:
        return f"Weibull (shape {self.shape:g})"


@dataclass(frozen=True)
class Gompertz(_Baseline):
    """h0(t) = lam e^{a t}, a = `shape` per month: the hazard is multiplied by
    e^{12 a} each year (`from_annual_hr`). a = 0 is exponential.

    With a < 0 the hazard dies out: H0(infinity) = lam / |a| is finite and a
    fraction exp(-lam / |a|) is never observed to fail (T = inf, a cure
    fraction). Calibrated to a median, that fraction is always below 1/2, and
    the censoring fraction cannot be set below it."""

    median_os: float = DEFAULT_MEDIAN_OS_MONTHS
    shape: float = 0.0
    family: ClassVar[str] = "gompertz"

    def __post_init__(self):
        super().__post_init__()
        if not np.isfinite(self.shape):
            raise ValueError("shape must be finite")
        if self.shape * self.median_os > 700:
            raise ValueError("shape * median_os must be at most 700: the hazard would overflow")

    @classmethod
    def from_annual_hr(cls, median_os: float, hr_per_year: float) -> "Gompertz":
        """The Gompertz baseline whose hazard is multiplied by `hr_per_year` each year."""
        _positive("hr_per_year", hr_per_year)
        return cls(median_os, float(np.log(hr_per_year) / 12.0))

    @property
    def annual_hr(self) -> float:
        """e^{12 a}: the ratio of the hazard a year later to the hazard now."""
        return float(np.exp(12.0 * self.shape))

    @property
    def lam(self) -> float:
        """Hazard at t = 0, per month."""
        a, m = self.shape, self.median_os
        return LN2 / m if a == 0 else float(a * LN2 / np.expm1(a * m))

    @property
    def cumhaz_max(self) -> float:
        return np.inf if self.shape >= 0 else self.lam / -self.shape

    def hazard(self, t):
        return self.lam * np.exp(self.shape * np.asarray(t, dtype=float))

    def cumhaz(self, t):
        t, a = np.asarray(t, dtype=float), self.shape
        if a == 0:
            return self.lam * t
        with np.errstate(over="ignore"):
            return self.lam * np.expm1(a * t) / a

    def inv_cumhaz(self, h):
        h, a = np.asarray(h, dtype=float), self.shape
        if a == 0:
            return h / self.lam
        x = a * h / self.lam
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(x > -1, np.log1p(np.maximum(x, -1.0)) / a, np.inf)

    def label(self) -> str:
        return (f"Gompertz (shape {self.shape:g} per month; hazard x{self.annual_hr:.3g} "
                f"per year)")


@dataclass(frozen=True)
class PiecewiseExponential(_Baseline):
    """Constant hazard between `breaks` (months), proportional to
    `relative_rates` (one more rate than breaks), scaled so the median is
    `median_os`: the scale c solves c * H_shape(m) = ln 2 directly."""

    median_os: float = DEFAULT_MEDIAN_OS_MONTHS
    breaks: tuple = (6.0, 12.0)
    relative_rates: tuple = (1.0, 1.5, 2.0)
    family: ClassVar[str] = "piecewise_exponential"

    def __post_init__(self):
        super().__post_init__()
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

    def _pieces(self):
        """Start of each piece, H_shape at each start, the rates, and the scale c."""
        starts = np.concatenate([[0.0], self.breaks])
        rates = np.asarray(self.relative_rates)
        h_starts = np.concatenate([[0.0], np.cumsum(rates[:-1] * np.diff(starts))])
        k = np.searchsorted(starts, self.median_os, side="right") - 1
        h_median = h_starts[k] + rates[k] * (self.median_os - starts[k])
        return starts, h_starts, rates, LN2 / h_median

    def hazard(self, t):
        starts, _, rates, c = self._pieces()
        return c * rates[np.searchsorted(starts, np.asarray(t, dtype=float), side="right") - 1]

    def cumhaz(self, t):
        starts, h_starts, rates, c = self._pieces()
        t = np.asarray(t, dtype=float)
        k = np.searchsorted(starts, t, side="right") - 1
        return c * (h_starts[k] + rates[k] * (t - starts[k]))

    def inv_cumhaz(self, h):
        starts, h_starts, rates, c = self._pieces()
        x = np.asarray(h, dtype=float) / c
        k = np.searchsorted(h_starts, x, side="right") - 1
        return starts[k] + (x - h_starts[k]) / rates[k]

    def label(self) -> str:
        return (f"piecewise exponential (breaks {', '.join(f'{b:g}' for b in self.breaks)} "
                f"months; relative rates {', '.join(f'{r:g}' for r in self.relative_rates)})")


@dataclass(frozen=True)
class LogLogistic(_Baseline):
    """S0(t) = 1 / (1 + (t / m)^k): the hazard rises then falls when k > 1,
    and only falls when k <= 1. Used as a PH baseline (hazards multiplied by
    e^{beta X}); the accelerated-failure-time form is #63."""

    median_os: float = DEFAULT_MEDIAN_OS_MONTHS
    shape: float = 1.5
    family: ClassVar[str] = "loglogistic"

    def __post_init__(self):
        super().__post_init__()
        _positive("shape", self.shape)

    def hazard(self, t):
        u = np.asarray(t, dtype=float) / self.median_os
        k = self.shape
        with np.errstate(divide="ignore"):
            return k / self.median_os * u ** (k - 1) / (1.0 + u ** k)

    def cumhaz(self, t):
        return np.log1p((np.asarray(t, dtype=float) / self.median_os) ** self.shape)

    def inv_cumhaz(self, h):
        with np.errstate(over="ignore"):
            return self.median_os * np.expm1(np.asarray(h, dtype=float)) ** (1.0 / self.shape)

    def label(self) -> str:
        return f"log-logistic (shape {self.shape:g})"


@dataclass(frozen=True, eq=False)
class NumericBaseline(_Baseline):
    """Any baseline hazard shape: `hazard(t)` (vectorized, non-negative, any
    scale) is rescaled so that the median is `median_os`.

    H0 is computed by Gauss-Legendre quadrature in u with s = t u^2, which
    keeps hazards with a t^(-1/2) or t^(1/2) singularity at 0 exact or nearly
    so. H0^{-1} is found by safeguarded Newton steps inside a bisection bracket,
    vectorized over the sample. About 50x slower than a closed form, and the
    hazard is assumed to integrate to infinity (no cure fraction): times whose
    target cumulative hazard is not reached by 2^60 medians are returned as inf.
    API-only: it holds a function, so it is not available from the CLI or JSON.
    """

    hazard_fn: Callable = None
    median_os: float = DEFAULT_MEDIAN_OS_MONTHS
    n_nodes: int = 64
    family: ClassVar[str] = "numeric"
    api_only: ClassVar[bool] = True

    _CHUNK: ClassVar[int] = 20_000
    _MAX_DOUBLINGS: ClassVar[int] = 60

    def __post_init__(self):
        super().__post_init__()
        if not callable(self.hazard_fn):
            raise ValueError("hazard_fn must be a function of t")
        u, w = np.polynomial.legendre.leggauss(int(self.n_nodes))
        u, w = (u + 1) / 2, w / 2                      # nodes and weights on [0, 1]
        object.__setattr__(self, "_nodes", (u, w))
        h_m = self._integral(np.array([self.median_os]))[0]
        if not 0 < h_m < np.inf:
            raise ValueError("hazard_fn must integrate to a positive, finite value over "
                             "[0, median_os]")
        object.__setattr__(self, "_scale", LN2 / h_m)

    # identity semantics: two NumericBaselines are equal only if they are the same object
    __hash__ = object.__hash__

    def _integral(self, t: np.ndarray) -> np.ndarray:
        """int_0^t hazard_fn(s) ds = int_0^1 hazard_fn(t u^2) 2 t u du."""
        u, w = self._nodes
        out = np.empty(t.shape)
        flat, res = t.ravel(), out.ravel()
        for i in range(0, flat.size, self._CHUNK):
            tc = flat[i:i + self._CHUNK, None]
            vals = np.asarray(self.hazard_fn(tc * u ** 2), dtype=float)
            res[i:i + self._CHUNK] = (vals * 2 * tc * u) @ w
        return out

    def hazard(self, t):
        return self._scale * np.asarray(self.hazard_fn(np.asarray(t, dtype=float)), dtype=float)

    def cumhaz(self, t):
        t = np.asarray(t, dtype=float)
        out = np.full(t.shape, np.inf)
        finite = np.isfinite(t)
        out[finite] = self._scale * self._integral(t[finite])
        return out

    def inv_cumhaz(self, h, rtol: float = 1e-13, max_iter: int = 100):
        h = np.asarray(h, dtype=float)
        flat = h.ravel()
        t = np.zeros(flat.size)
        pos = flat > 0
        target = flat[pos]
        # bracket: double hi from the median until H(hi) >= target
        lo, hi = np.zeros(target.size), np.full(target.size, float(self.median_os))
        todo = np.ones(target.size, bool)
        for _ in range(self._MAX_DOUBLINGS):
            short = self.cumhaz(hi[todo]) < target[todo]
            idx = np.flatnonzero(todo)
            lo[idx[short]] = hi[idx[short]]
            hi[idx[short]] *= 2
            todo[idx[~short]] = False
            if not todo.any():
                break
        unreached = todo
        x = (lo + hi) / 2
        active = ~unreached
        for _ in range(max_iter):
            idx = np.flatnonzero(active)
            if idx.size == 0:
                break
            xi = x[idx]
            f = self.cumhaz(xi) - target[idx]
            below = f < 0
            lo[idx[below]] = xi[below]
            hi[idx[~below]] = xi[~below]
            with np.errstate(divide="ignore", invalid="ignore"):
                newton = xi - f / self.hazard(xi)
            ok = (newton > lo[idx]) & (newton < hi[idx]) & np.isfinite(newton)
            x_new = np.where(ok, newton, (lo[idx] + hi[idx]) / 2)
            done = (np.abs(x_new - xi) <= rtol * x_new) | (hi[idx] - lo[idx] <= rtol * hi[idx])
            x[idx] = x_new
            active[idx[done]] = False
        x[unreached] = np.inf
        t[pos] = x
        return t.reshape(h.shape)

    def label(self) -> str:
        return "numeric (user-supplied hazard)"


#: The families that `baseline_from_dict` and the CLI can build, by name.
FAMILIES = {cls.family: cls for cls in (Exponential, Weibull, Gompertz,
                                        PiecewiseExponential, LogLogistic)}
#: The families with one `shape` parameter: the CLI's `--baseline` choices other than exponential.
SHAPE_FAMILIES = ("weibull", "gompertz", "loglogistic")


def baseline_from_dict(d: dict) -> _Baseline:
    """Rebuild a baseline from `to_dict()` output."""
    d = dict(d)
    family = d.pop("family", None)
    if family not in FAMILIES:
        raise ValueError(f"unknown baseline family {family!r}; one of {tuple(FAMILIES)}")
    return FAMILIES[family](**d)


def is_exponential(baseline) -> bool:
    """Whether `baseline` is the default model: None or an `Exponential`."""
    return baseline is None or isinstance(baseline, Exponential)


def _normal_rule(lo: float = -12.0, hi: float = 12.0, n_nodes: int = 96):
    """Gauss-Legendre nodes and normalized weights for E[g(X)], X ~ N(0, 1) on (lo, hi)."""
    from scipy.stats import norm

    x, w = np.polynomial.legendre.leggauss(n_nodes)
    x = (hi - lo) / 2 * x + (hi + lo) / 2
    w = w * norm.pdf(x)
    return x, w / w.sum()


def marginal_cure_fraction(baseline, log_hr: float) -> float:
    """P(T = inf) over X ~ N(0, 1) under PH: E[exp(-H0(inf) e^{beta X})]. Zero
    unless the baseline is defective."""
    h_max = getattr(baseline, "cumhaz_max", np.inf)
    if not np.isfinite(h_max):
        return 0.0
    x, w = _normal_rule()
    return float(w @ np.exp(-h_max * np.exp(log_hr * x)))


def marginal_cumhaz_median(log_hr: float, lo: float = -12.0, hi: float = 12.0,
                           n_nodes: int = 96) -> float:
    """h* with E[exp(-h* e^{beta X}) | lo < X < hi] = 1/2 for X ~ N(0, 1). The
    marginal median under PH with any baseline is then H0^{-1}(h*); h* = ln 2 at
    beta = 0."""
    x, w = _normal_rule(lo, hi, n_nodes)
    r = np.exp(log_hr * x)
    return float(optimize.brentq(lambda h: w @ np.exp(-h * r) - 0.5, 1e-12, 1e6, xtol=1e-14,
                                 rtol=1e-14))


def marginal_median(baseline: BaselineHazard, log_hr: float) -> float:
    """Median of T over X ~ N(0, 1) under PH: the root of
    E_X[S0(t)^{exp(beta X)}] = 1/2. inf if more than half never fail."""
    return float(baseline.inv_cumhaz(np.array(marginal_cumhaz_median(log_hr))))


__all__ = ["BaselineHazard", "Exponential", "Weibull", "Gompertz", "PiecewiseExponential",
           "LogLogistic", "NumericBaseline", "baseline_from_dict", "is_exponential",
           "marginal_cure_fraction", "marginal_median", "median_os_to_lambda0", "median_survival",
           "DEFAULT_LAMBDA0", "DEFAULT_MEDIAN_OS_MONTHS", "DEFAULT_TAU_MONTHS",
           "DEFAULT_CENSOR_RATE"]
