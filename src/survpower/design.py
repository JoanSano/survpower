"""`StudyDesign`: everything about a planned study other than N, the effect
size and the target power."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .baseline import DEFAULT_CENSOR_RATE, DEFAULT_MEDIAN_OS_MONTHS, median_os_to_lambda0
from .censoring import calibrate_censoring_rate, censor_rate_from_design
from .concordance import c_of_beta


@dataclass
class StudyDesign:
    """Design assumptions shared by all planning modes.

    alpha, sides     : type I error; one- (1) or two-sided (2) Wald test.
    median_os        : baseline (biomarker at its mean) median OS, months.
    censor_rate      : marginal censoring probability (default 0.20); ignored
                       when accrual_months / followup_months are given.
    accrual_months,
    followup_months  : uniform accrual period and minimum follow-up after the
                       last enrolment; censoring is then derived with
                       `censor_rate_from_design`.
    dropout_annual   : annual loss-to-follow-up fraction (accrual design only).
    non_evaluable    : fraction of enrolled patients without a usable
                       biomarker value (missing tissue, assay failure, ...).
    n_covariates, r2 : adjustment covariates and the multiple R^2 of the
                       biomarker on them.
    biomarker_sd     : if set, user-facing HRs are per raw unit and convert
                       to per-SD as HR ** biomarker_sd.
    c_estimator      : "harrell" or "uno" for concordance-based questions.
    c_tau            : truncation horizon (months) for Uno's C; default is
                       the 90th percentile of observed follow-up.
    engine           : "fast" or "lifelines" for univariate Cox fits.
    n_sims           : replicates for power estimates / confirmation runs.
    n_sims_grid      : replicates per grid point when solving for N or HR.
    """

    alpha: float = 0.05
    sides: int = 2
    median_os: float = DEFAULT_MEDIAN_OS_MONTHS
    censor_rate: float | None = None
    accrual_months: float | None = None
    followup_months: float | None = None
    dropout_annual: float = 0.0
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

    @property
    def lambda0(self) -> float:
        return median_os_to_lambda0(self.median_os)

    @property
    def r2_eff(self) -> float:
        return self.r2 if self.n_covariates > 0 else 0.0

    def censor_rate_for(self, hr_per_sd: float) -> float:
        """Marginal censoring probability at a given per-SD HR."""
        if self.accrual_months is None:
            return DEFAULT_CENSOR_RATE if self.censor_rate is None else self.censor_rate
        return censor_rate_from_design(
            hr_per_sd, self.accrual_months, self.followup_months,
            dropout_annual=self.dropout_annual, lambda0=self.lambda0,
        )

    def lambda_c_for(self, hr_per_sd: float) -> float:
        """Calibrated exponential censoring rate (per month)."""
        return calibrate_censoring_rate(np.log(hr_per_sd), self.censor_rate_for(hr_per_sd),
                                        lambda0=self.lambda0)

    def evaluable(self, n_enrolled: int) -> int:
        return int(np.floor(n_enrolled * (1 - self.non_evaluable)))

    def enrolled(self, n_evaluable: int) -> int:
        return int(np.ceil(n_evaluable / (1 - self.non_evaluable)))

    def to_per_sd(self, hr: float) -> float:
        return hr if self.biomarker_sd is None else hr ** self.biomarker_sd

    def to_per_unit(self, hr_per_sd: float) -> float:
        return hr_per_sd if self.biomarker_sd is None else hr_per_sd ** (1 / self.biomarker_sd)

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
        return "Harrell's C" if self.c_estimator == "harrell" else "Uno's C"
