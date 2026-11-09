"""`StudyDesign`: everything about a planned study other than N, the effect
size and the target power."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .baseline import (DEFAULT_CENSOR_RATE, DEFAULT_MEDIAN_OS_MONTHS, Exponential, _Baseline,
                       is_exponential, median_os_to_lambda0)
from .censoring import (calibrate_censoring_rate, censor_rate_from_design,
                        require_censoring_above_cure)
from .concordance import ESTIMATORS, c_of_beta
from .dgp import gen_ph
from .fitting import ENGINES
from .scenario import CalendarCensoring, EmpiricalCensoring, ExponentialCensoring, WeibullCensoring

#: The values of `StudyDesign.censoring_model`.
CENSORING_MODELS = ("exponential", "calendar", "empirical")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_positive(x: float) -> bool:
    """Positive and finite; false for NaN."""
    return 0 < x < np.inf


@dataclass(frozen=True)
class StudyDesign:
    """Design assumptions shared by all planning modes.

    Invalid values raise `ValueError` at construction, naming the field.
    A design is immutable; to change a field, make a new, validated design
    with `dataclasses.replace`.

    alpha, sides     : type I error, in (0, 0.5); one- (1) or two-sided (2)
                       Wald test.
    median_os        : baseline (biomarker at its mean) median OS, months.
    censor_rate      : marginal censoring probability (default 0.20). Give
                       this or accrual_months / followup_months, not both.
    accrual_months,
    followup_months  : uniform accrual period and minimum follow-up after the
                       last enrolment, given together and not both zero;
                       censoring is then derived with
                       `censor_rate_from_design`.
    dropout_annual   : annual loss-to-follow-up fraction (accrual design only).
    censoring_model  : how censoring times are simulated (#16):
                       "exponential" (default): exponential censoring
                         calibrated to the censored fraction, which an
                         accrual design only sets;
                       "calendar": the accrual design simulated directly
                         (`scenario.CalendarCensoring`); the censored fraction
                         is an output;
                       "empirical": drawn from `censoring_data`, the reverse
                         Kaplan-Meier of a cohort; the fraction is an output.
    censoring_shape  : Weibull censoring, H_C(t) = lambda_c t^k with this k,
                       calibrated to a fixed censor_rate
                       (`scenario.WeibullCensoring`); "exponential" model only.
    censoring_data   : `scenario.EmpiricalCensoring` for "empirical".
    non_evaluable    : fraction of enrolled patients without a usable
                       biomarker value (missing tissue, assay failure, ...).
    n_covariates, r2 : adjustment covariates and the multiple R^2 of the
                       biomarker on them.
    biomarker_sd     : if set, user-facing HRs are per raw unit and convert
                       to per-SD as HR ** biomarker_sd.
    c_estimator      : "harrell", "uno" or "gonen-heller" for
                       concordance-based questions.
    c_tau            : truncation horizon (months) for Uno's C; default is
                       the 90th percentile of observed follow-up.
    engine           : "fast" or "lifelines" for univariate Cox fits.
    n_sims           : replicates for power estimates / confirmation runs.
    n_sims_grid      : replicates per grid point when solving for N or HR.
    baseline         : the baseline hazard (`survpower.baseline`: Weibull,
                       Gompertz, ...), whose median must equal `median_os`.
                       Default None, the exponential baseline
                       ``Exponential(median_os)``.
    """

    alpha: float = 0.05
    sides: int = 2
    median_os: float = DEFAULT_MEDIAN_OS_MONTHS
    censor_rate: float | None = None
    accrual_months: float | None = None
    followup_months: float | None = None
    dropout_annual: float = 0.0
    censoring_model: str = "exponential"
    censoring_shape: float | None = None
    censoring_data: EmpiricalCensoring | None = None
    non_evaluable: float = 0.0
    n_covariates: int = 0
    r2: float = 0.0
    biomarker_sd: float | None = None
    c_estimator: str = "harrell"
    c_tau: float | None = None
    engine: str = "fast"
    n_sims: int = 1000
    n_sims_grid: int = 1000
    seed: int = 42
    baseline: _Baseline | None = None

    def __post_init__(self) -> None:
        _require(0 < self.alpha < 0.5, "alpha must be in (0, 0.5)")
        _require(self.sides in (1, 2), "sides must be 1 or 2")
        _require(_is_positive(self.median_os), "median_os must be positive and finite")

        has_accrual = self.accrual_months is not None
        _require(has_accrual == (self.followup_months is not None),
                 "accrual_months and followup_months must be given together")
        _require(not (has_accrual and self.censor_rate is not None),
                 "give either censor_rate or an accrual/follow-up design, not both")
        if has_accrual:
            _require(0 <= self.accrual_months < np.inf and 0 <= self.followup_months < np.inf,
                     "accrual_months and followup_months must be non-negative and finite")
            _require(self.accrual_months + self.followup_months > 0,
                     "accrual_months and followup_months cannot both be zero: "
                     "no patient would be followed up")
        _require(not self.dropout_annual or has_accrual,
                 "dropout_annual requires accrual_months/followup_months "
                 "(with censor_rate, include dropout in the censoring rate)")
        if self.censor_rate is not None:
            _require(0 <= self.censor_rate < 1, "censor_rate must be in [0, 1)")
        self._validate_censoring(has_accrual)
        for name in ("dropout_annual", "non_evaluable", "r2"):
            _require(0 <= getattr(self, name) < 1, f"{name} must be in [0, 1)")

        _require(isinstance(self.n_covariates, Integral) and self.n_covariates >= 0,
                 "n_covariates must be a non-negative integer")
        _require(self.r2 == 0 or self.n_covariates >= 1, "r2 requires n_covariates >= 1")
        if self.biomarker_sd is not None:
            _require(_is_positive(self.biomarker_sd), "biomarker_sd must be positive and finite")
        _require(self.c_estimator in ESTIMATORS, f"c_estimator must be one of {ESTIMATORS}")
        if self.c_tau is not None:
            _require(_is_positive(self.c_tau), "c_tau must be positive and finite")
        _require(self.engine in ENGINES, f"engine must be one of {ENGINES}")
        for name in ("n_sims", "n_sims_grid"):
            value = getattr(self, name)
            _require(isinstance(value, Integral) and value >= 1,
                     f"{name} must be a positive integer")
        if self.baseline is not None:
            _require(isinstance(self.baseline, _Baseline),
                     "baseline must be a survpower.baseline family, such as Weibull(...)")
            _require(np.isclose(self.baseline.median, self.median_os, rtol=1e-12, atol=0),
                     f"baseline median ({self.baseline.median:g}) must equal median_os "
                     f"({self.median_os:g}): give median_os={self.baseline.median:g} too")

    def _validate_censoring(self, has_accrual: bool) -> None:
        model = self.censoring_model
        _require(model in CENSORING_MODELS, f"censoring_model must be one of {CENSORING_MODELS}")
        _require(model != "calendar" or has_accrual,
                 "censoring_model 'calendar' requires accrual_months and followup_months")
        _require((model == "empirical") == (self.censoring_data is not None),
                 "censoring_model 'empirical' and censoring_data must be given together")
        if model == "empirical":
            _require(isinstance(self.censoring_data, EmpiricalCensoring),
                     "censoring_data must be a scenario.EmpiricalCensoring")
            _require(not has_accrual and self.censor_rate is None,
                     "censoring_model 'empirical' sets the censoring: give neither censor_rate "
                     "nor accrual_months/followup_months")
        if self.censoring_shape is not None:
            _require(_is_positive(self.censoring_shape),
                     "censoring_shape must be positive and finite")
            _require(model == "exponential" and not has_accrual,
                     "censoring_shape applies to a fixed censor_rate (not with an "
                     "accrual/follow-up design or another censoring_model)")
            _require(self.censor_rate != 0, "censoring_shape needs a censor_rate above 0")

    @property
    def censoring_calibrated(self) -> bool:
        """Whether censoring is calibrated to a target fraction (exponential or Weibull),
        rather than simulated with the fraction as an output (calendar, empirical)."""
        return self.censoring_model == "exponential"

    def non_default_models(self) -> tuple:
        """Names of the fields that select a model other than exponential PH with
        exponential censoring: ``baseline`` when it is not exponential, and
        ``censoring`` for another censoring model or shape."""
        out = () if is_exponential(self.baseline) else ("baseline",)
        if self.censoring_model != "exponential" or self.censoring_shape is not None:
            out += ("censoring",)
        return out

    def censoring_for(self, censor_rate: float):
        """The censoring model (`survpower.scenario`): calibrated to `censor_rate` when
        `censoring_calibrated`, else the calendar or the cohort's censoring, which
        ignore it."""
        if self.censoring_model == "calendar":
            return CalendarCensoring(self.accrual_months, self.followup_months,
                                     dropout_annual=self.dropout_annual)
        if self.censoring_model == "empirical":
            return self.censoring_data
        if self.censoring_shape is not None:
            return WeibullCensoring(target_rate=censor_rate, shape=self.censoring_shape)
        return ExponentialCensoring(target_rate=censor_rate)

    def censoring_label(self) -> str:
        """The censoring model, for summaries."""
        if self.censoring_model == "calendar":
            return "calendar simulated directly (uniform entry, fixed analysis date)"
        if self.censoring_model == "empirical":
            return self.censoring_data.label()
        if self.censoring_shape is not None:
            return f"Weibull, shape {self.censoring_shape:g}, calibrated to the fraction"
        return "exponential, calibrated to the fraction"

    def require_default_models(self, path: str, supports: tuple = ()) -> None:
        """Raise if `path` cannot honour this design's models. Every path that is
        not built on `scenario.scenario_for` calls this, with the model fields it
        does honour in `supports`: an unsupported combination must fail, never
        fall back to the exponential model."""
        unsupported = [name for name in self.non_default_models() if name not in supports]
        if unsupported:
            raise ValueError(f"{path} supports only exponential PH with exponential censoring; "
                             f"this design sets: {', '.join(unsupported)}")

    @property
    def baseline_hazard(self) -> _Baseline:
        """The baseline hazard: `baseline`, or ``Exponential(median_os)`` by default."""
        return Exponential(self.median_os) if self.baseline is None else self.baseline

    @property
    def lambda0(self) -> float:
        """The constant baseline hazard, per month. Only an exponential baseline has one."""
        if not is_exponential(self.baseline):
            raise ValueError(f"lambda0 is only defined for an exponential baseline; this "
                             f"design's baseline is {self.baseline.label()}: use "
                             f"design.baseline")
        return median_os_to_lambda0(self.median_os)

    def baseline_kwargs(self) -> dict:
        """``lambda0=`` for the exponential baseline, else ``baseline=``: the
        keyword that the simulation engines take for the event-time model. The
        exponential path keeps its random stream."""
        if is_exponential(self.baseline):
            return dict(lambda0=self.lambda0)
        return dict(baseline=self.baseline)

    @property
    def r2_eff(self) -> float:
        return self.r2 if self.n_covariates > 0 else 0.0

    def censor_rate_for(self, hr_per_sd: float, n_probe: int = 500_000, seed: int = 1) -> float:
        """Marginal censoring probability at a given per-SD HR. A fixed rate must
        be above the cure fraction of a defective baseline (Gompertz, shape < 0).
        For "empirical" censoring, the expected fraction over a probe sample of
        event times."""
        if self.censoring_model == "empirical":
            T, _ = gen_ph(n_probe, np.log(hr_per_sd), np.random.default_rng(seed),
                          **self.baseline_kwargs())
            return float(np.mean(self.censoring_data.probability_before(T)))
        if self.accrual_months is None:
            rate = DEFAULT_CENSOR_RATE if self.censor_rate is None else self.censor_rate
            if not is_exponential(self.baseline):
                require_censoring_above_cure(self.baseline, np.log(hr_per_sd), rate)
            return rate
        return censor_rate_from_design(
            hr_per_sd, self.accrual_months, self.followup_months,
            dropout_annual=self.dropout_annual, **self.baseline_kwargs(),
        )

    def lambda_c_for(self, hr_per_sd: float) -> float:
        """Calibrated exponential censoring rate (per month). Only exponential censoring
        has one."""
        if "censoring" in self.non_default_models():
            raise ValueError(f"lambda_c is only defined for exponential censoring; this "
                             f"design's censoring is {self.censoring_label()}")
        return calibrate_censoring_rate(np.log(hr_per_sd), self.censor_rate_for(hr_per_sd),
                                        **self.baseline_kwargs())

    def evaluable(self, n_enrolled: int) -> int:
        return int(np.floor(n_enrolled * (1 - self.non_evaluable)))

    def enrolled(self, n_evaluable: int) -> int:
        return int(np.ceil(n_evaluable / (1 - self.non_evaluable)))

    def to_per_sd(self, hr: float) -> float:
        return hr if self.biomarker_sd is None else hr ** self.biomarker_sd

    def to_per_unit(self, hr_per_sd: float) -> float:
        return hr_per_sd if self.biomarker_sd is None else hr_per_sd ** (1 / self.biomarker_sd)

    def hr_value_label(self, hr_per_sd: float) -> str:
        """The HR on the user's scale, without the concordance (for C questions)."""
        if self.biomarker_sd is None:
            return f"{hr_per_sd:.3g} per SD"
        return f"{self.to_per_unit(hr_per_sd):.3g} per biomarker unit or {hr_per_sd:.3g} per SD"

    def hr_label(self, hr_per_sd: float) -> str:
        c = c_of_beta(np.log(hr_per_sd))
        if self.biomarker_sd is None:
            return f"{hr_per_sd:.3g} per SD of the biomarker (concordance {c:.3f})"
        return (f"{self.to_per_unit(hr_per_sd):.3g} per biomarker unit "
                f"({hr_per_sd:.3g} per SD; concordance {c:.3f})")

    def test_label(self) -> str:
        return f"{'two' if self.sides == 2 else 'one'}-sided Wald test at alpha = {self.alpha:g}"

    def adjustment_label(self) -> str:
        if self.n_covariates == 0:
            return ""
        return (f", adjusted for {self.n_covariates} covariate(s) "
                f"(R^2 of the biomarker on them = {self.r2:g})")

    def estimator_label(self) -> str:
        return {"harrell": "Harrell's C", "uno": "Uno's C",
                "gonen-heller": "Gonen-Heller's C"}[self.c_estimator]
