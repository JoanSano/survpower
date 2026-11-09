"""Censoring models (#16): calibrated Weibull and piecewise shapes, the trial calendar
simulated directly, and a cohort's own censoring (reverse Kaplan-Meier)."""

import dataclasses

import numpy as np
import pytest
from scipy import integrate

from survpower import planning
from survpower.baseline import Weibull, median_os_to_lambda0
from survpower.censoring import (calibrate_cumhaz_scale, censor_rate_from_design,
                                 dropout_scale, event_probability_exponential,
                                 piecewise_cumhaz, piecewise_inv_cumhaz)
from survpower.design import StudyDesign
from survpower.planning import run_power_mode, sample_size
from survpower.scenario import (CalendarCensoring, Covariates, CoxWald, Effect,
                                EmpiricalCensoring, ExponentialCensoring, ExponentialPH, PHModel,
                                PiecewiseCensoring, Scenario, StandardNormal, WeibullCensoring,
                                scenario_for, simulate)
from survpower.simulate import simulate_power, simulate_waning_effect

LAMBDA0 = median_os_to_lambda0(15)


def _realized(scenario, n, n_reps, seed):
    """Censored fraction of each of `n_reps` cohorts of size n (no analysis)."""
    sc = scenario.calibrated(seed + 999)
    rng = np.random.default_rng(seed)
    out = np.empty(n_reps)
    for i in range(n_reps):
        _, event, _ = sc.draw(n, rng)
        out[i] = 1 - event.mean()
    return out


def _within(values, target, k=4):
    mcse = values.std(ddof=1) / np.sqrt(len(values))
    assert abs(values.mean() - target) < k * mcse, (values.mean(), target, mcse)


# --- calibrated shapes -------------------------------------------------------------------

CALIBRATED = {
    "exponential": lambda r: ExponentialCensoring(target_rate=r),
    "weibull 0.5": lambda r: WeibullCensoring(target_rate=r, shape=0.5),
    "weibull 2": lambda r: WeibullCensoring(target_rate=r, shape=2.0),
    "piecewise": lambda r: PiecewiseCensoring(target_rate=r, breaks=(12.0, 24.0),
                                              relative_rates=(0.5, 1.0, 3.0)),
}
BASELINES = {
    "exponential": lambda hr: ExponentialPH(log_hr=np.log(hr), lambda0=LAMBDA0),
    "weibull 1.5": lambda hr: PHModel(Weibull(median_os=15, shape=1.5), log_hr=np.log(hr)),
}


@pytest.mark.parametrize("baseline", BASELINES)
@pytest.mark.parametrize("model", CALIBRATED)
def test_realized_censoring_hits_the_target(model, baseline):
    target = 0.3
    sc = Scenario(StandardNormal(), BASELINES[baseline](1.3), CALIBRATED[model](target))
    _within(_realized(sc, n=1000, n_reps=400, seed=11), target)


@pytest.mark.parametrize("hr", [1.0, 1.5])
@pytest.mark.parametrize("shape", [0.5, 2.0])
def test_weibull_censoring_matches_quadrature(shape, hr):
    """P(C < T) = int S_T(t) h_C(t) S_C(t) dt, exponential T marginalized over X ~ N(0, 1),
    against the probe estimate at a known lambda_c."""
    lam_c = 0.02 if shape == 0.5 else 3e-4
    x, w = np.polynomial.hermite_e.hermegauss(40)
    w = w / np.sqrt(2 * np.pi)

    def integrand(t):
        s_t = w @ np.exp(-LAMBDA0 * hr ** x * t)
        return s_t * lam_c * shape * t ** (shape - 1) * np.exp(-lam_c * t ** shape)

    exact = integrate.quad(integrand, 0, np.inf, limit=200)[0]
    model = WeibullCensoring(target_rate=0.5, shape=shape, rate=lam_c)
    n = 500_000
    rng = np.random.default_rng(3)
    T = rng.exponential(1 / (LAMBDA0 * hr ** rng.standard_normal(n)))
    C = model.draw(Covariates(np.zeros(n)), T, rng)
    p = np.mean(C < T)
    assert abs(p - exact) < 4 * np.sqrt(p * (1 - p) / n)
    # and the exact conditional probability the calibration uses
    mcse = np.sqrt(p * (1 - p) / n)
    assert np.mean(-np.expm1(-lam_c * T ** shape)) == pytest.approx(exact, abs=4 * mcse)


def test_calibrated_scale_is_monotone_and_unique():
    T = np.random.default_rng(0).exponential(1 / LAMBDA0, 200_000)
    scales = [calibrate_cumhaz_scale(T, lambda t: t ** 2, r) for r in (0.1, 0.3, 0.5, 0.7)]
    assert np.all(np.diff(scales) > 0)
    lam = scales[1]
    assert np.mean(-np.expm1(-lam * T ** 2)) == pytest.approx(0.3, abs=1e-9)


def test_calibration_refuses_a_target_below_the_always_censored():
    T = np.concatenate([np.full(300, np.inf), np.ones(700)])  # 30% cured
    with pytest.raises(ValueError, match="no censoring scale"):
        calibrate_cumhaz_scale(T, lambda t: t, 0.2)


def test_weibull_shape_one_is_exponential_censoring():
    sc = Scenario(event_times=ExponentialPH(log_hr=np.log(1.3), lambda0=LAMBDA0))
    lam_exp = sc.censoring.calibrated(sc.sample_latent, 1).rate
    lam_wei = WeibullCensoring(target_rate=0.2, shape=1.0).calibrated(sc.sample_latent, 1).rate
    assert lam_wei == pytest.approx(lam_exp, rel=0.01)


def test_piecewise_cumhaz_round_trips():
    breaks, rates = (6.0, 18.0), (0.5, 2.0, 1.0)
    t = np.array([0.0, 3.0, 6.0, 10.0, 18.0, 40.0])
    H = piecewise_cumhaz(t, breaks, rates)
    np.testing.assert_allclose(H, [0, 1.5, 3, 11, 27, 49])
    np.testing.assert_allclose(piecewise_inv_cumhaz(H, breaks, rates), t)


@pytest.mark.parametrize("kwargs, match", [
    (dict(target_rate=1.0), "target_rate"),
    (dict(shape=0.0), "shape"),
    (dict(rate=-1.0), "rate"),
])
def test_weibull_censoring_validates(kwargs, match):
    with pytest.raises(ValueError, match=match):
        WeibullCensoring(**kwargs)


@pytest.mark.parametrize("kwargs, match", [
    (dict(breaks=(12.0,), relative_rates=(1.0,)), "one more"),
    (dict(breaks=(12.0, 6.0), relative_rates=(1.0, 1.0, 1.0)), "increasing"),
    (dict(breaks=(12.0,), relative_rates=(1.0, 0.0)), "positive"),
])
def test_piecewise_censoring_validates(kwargs, match):
    with pytest.raises(ValueError, match=match):
        PiecewiseCensoring(**kwargs)


def test_uncalibrated_draw_raises():
    with pytest.raises(ValueError, match="calibrated"):
        WeibullCensoring().draw(Covariates(np.zeros(3)), np.ones(3), np.random.default_rng(0))


# --- the calendar ------------------------------------------------------------------------

@pytest.mark.parametrize("hr", [1.0, 1.15, 1.5])
def test_calendar_matches_the_design_rate_and_the_closed_form(hr):
    A, F, d = 36, 12, 0.05
    sc = Scenario(StandardNormal(), ExponentialPH(log_hr=np.log(hr), lambda0=LAMBDA0),
                  CalendarCensoring(A, F, dropout_annual=d))
    realized = _realized(sc, n=2000, n_reps=200, seed=2)
    _within(realized, censor_rate_from_design(hr, A, F, d, lambda0=LAMBDA0))
    _within(realized, 1 - event_probability_exponential(A, F, d, median_os=15, hr_per_sd=hr))


@pytest.mark.parametrize("shape", [0.5, 1.0, 2.0])
def test_weibull_dropout_loses_the_annual_fraction_at_twelve_months(shape):
    d = 0.1
    assert -np.expm1(-dropout_scale(d, shape) * 12 ** shape) == pytest.approx(d)
    model = CalendarCensoring(accrual_months=0, followup_months=1000, dropout_annual=d,
                              dropout_shape=shape)
    C = model.draw(Covariates(np.zeros(400_000)), None, np.random.default_rng(1))
    assert np.mean(C < 12) == pytest.approx(d, abs=4 * np.sqrt(d * (1 - d) / 400_000))


def test_calendar_is_not_calibrated():
    model = CalendarCensoring(24, 12)
    assert model.calibrated(None, 0) is model
    C = model.draw(Covariates(np.zeros(1000)), None, np.random.default_rng(0))
    assert C.min() >= 12 and C.max() <= 36


@pytest.mark.parametrize("kwargs, match", [
    (dict(accrual_months=0, followup_months=0), "accrual_months"),
    (dict(accrual_months=12, followup_months=12, dropout_annual=1.0), "dropout_annual"),
    (dict(accrual_months=12, followup_months=12, dropout_shape=0), "dropout_shape"),
])
def test_calendar_validates(kwargs, match):
    with pytest.raises(ValueError, match=match):
        CalendarCensoring(**kwargs)


# --- a cohort's censoring --------------------------------------------------------------

def _cohort(n=2000, seed=4):
    rng = np.random.default_rng(seed)
    T = rng.exponential(1 / LAMBDA0, n)
    C = rng.uniform(12, 48, n)
    return np.minimum(T, C), (T <= C).astype(int)


def test_reverse_km_matches_lifelines():
    from lifelines import KaplanMeierFitter
    time, event = _cohort()
    model = EmpiricalCensoring.from_data(time, event)
    km = KaplanMeierFitter().fit(time, 1 - event)
    t = np.asarray(model.times[:-1])
    np.testing.assert_allclose(model.probability_before(t + 1e-9),
                               1 - km.survival_function_at_times(t).to_numpy(), atol=1e-12)
    assert model.cdf[-1] == 1.0 and model.times[-1] == pytest.approx(time.max())


def test_reverse_km_counts_deaths_before_censoring_at_a_tie():
    # times 1 (death), 1 (censored), 2 (censored): one censoring among the two not dead
    # at 1, then the last one
    model = EmpiricalCensoring.from_data([1, 1, 2], [1, 0, 0])
    assert model.times == (1.0, 2.0)
    assert model.cdf == pytest.approx((0.5, 1.0))


def test_empirical_censoring_reproduces_the_cohort_fraction():
    """Under the model that generated the cohort, simulating its estimated censoring gives
    back the cohort's censored fraction."""
    time, event = _cohort(n=4000)
    model = EmpiricalCensoring.from_data(time, event)
    sc = Scenario(StandardNormal(), ExponentialPH(log_hr=0.0, lambda0=LAMBDA0), model)
    realized = _realized(sc, n=2000, n_reps=100, seed=5)
    observed = 1 - event.mean()
    se = np.sqrt(observed * (1 - observed) / len(event))  # the cohort's own sampling error
    assert abs(realized.mean() - observed) < 4 * se
    # the expected fraction the design reports agrees with the draws
    design = StudyDesign(censoring_model="empirical", censoring_data=model)
    _within(realized, design.censor_rate_for(1.0), k=5)


def test_empirical_draws_follow_the_cdf():
    model = EmpiricalCensoring(times=(5.0, 10.0, 20.0), cdf=(0.2, 0.5, 1.0))
    C = model.draw(Covariates(np.zeros(200_000)), None, np.random.default_rng(0))
    assert set(np.unique(C)) == {5.0, 10.0, 20.0}
    np.testing.assert_allclose([np.mean(C <= 5), np.mean(C <= 10)], [0.2, 0.5], atol=0.005)
    np.testing.assert_allclose(model.probability_before([5.0, 5.1, 25.0]), [0.0, 0.2, 1.0])


@pytest.mark.parametrize("time, event, match", [
    ([1, 2, 3], [1, 1, 1], "no censored"),
    ([0, 2, 3], [0, 1, 0], "positive"),
    ([1, 2, 3], [0, 2, 0], "0 or 1"),
])
def test_empirical_censoring_validates(time, event, match):
    with pytest.raises(ValueError, match=match):
        EmpiricalCensoring.from_data(time, event)


# --- StudyDesign and the planning paths ----------------------------------------------------

def test_default_design_keeps_exponential_censoring():
    design = StudyDesign()
    assert design.non_default_models() == ()
    assert isinstance(design.censoring_for(0.2), ExponentialCensoring)
    sc = scenario_for(StudyDesign(n_sims=50, seed=3), Effect.hr(1.3))
    assert sc.censoring == ExponentialCensoring(target_rate=0.2)


def test_design_builds_each_censoring_model():
    cal = StudyDesign(accrual_months=36, followup_months=12, dropout_annual=0.05,
                      censoring_model="calendar")
    assert cal.censoring_for(0.3) == CalendarCensoring(36, 12, dropout_annual=0.05)
    assert cal.non_default_models() == ("censoring",)
    assert cal.censor_rate_for(1.2) == StudyDesign(
        accrual_months=36, followup_months=12, dropout_annual=0.05).censor_rate_for(1.2)
    wei = StudyDesign(censor_rate=0.3, censoring_shape=2.0)
    assert wei.censoring_for(0.3) == WeibullCensoring(target_rate=0.3, shape=2.0)
    emp = EmpiricalCensoring(times=(5.0, 10.0), cdf=(0.5, 1.0))
    assert StudyDesign(censoring_model="empirical", censoring_data=emp).censoring_for(0.2) is emp


@pytest.mark.parametrize("kwargs, match", [
    (dict(censoring_model="weibull"), "censoring_model must be one of"),
    (dict(censoring_model="calendar"), "requires accrual_months"),
    (dict(censoring_model="empirical"), "given together"),
    (dict(censoring_data=EmpiricalCensoring(times=(1.0,), cdf=(1.0,))), "given together"),
    (dict(censoring_model="empirical", censoring_data=EmpiricalCensoring(times=(1.0,), cdf=(1.0,)),
          censor_rate=0.2), "give neither"),
    (dict(censoring_shape=0.0), "censoring_shape must be positive"),
    (dict(censoring_shape=2.0, accrual_months=12, followup_months=12), "fixed censor_rate"),
    (dict(censoring_shape=2.0, censor_rate=0.0), "above 0"),
])
def test_design_validates_censoring(kwargs, match):
    with pytest.raises(ValueError, match=match):
        StudyDesign(**kwargs)


CALENDAR = dict(accrual_months=36, followup_months=12, dropout_annual=0.05,
                censoring_model="calendar", n_sims=60, n_sims_grid=40, seed=5)


def test_calendar_design_runs_the_planning_paths():
    design = StudyDesign(**CALENDAR)
    row = run_power_mode(design, [1.3], [200]).iloc[0]
    assert 0 < row["power_sim"] <= 1
    assert sample_size(hr=1.4, design=design).n_evaluable > 0


def test_calendar_design_draws_from_the_calendar():
    design = StudyDesign(**CALENDAR)
    sc = scenario_for(design, Effect.hr(1.3))
    assert isinstance(sc.censoring, CalendarCensoring)
    row = simulate(sc, n=500, n_sims=40, seed=1).iloc[0]
    assert abs(row["censor_rate_realized"] - design.censor_rate_for(1.3)) < \
        4 * row["censor_rate_realized_mcse"]
    assert "lambda_c" not in row


@pytest.mark.parametrize("run", [
    lambda d: run_power_mode(dataclasses.replace(d, n_covariates=2), [1.3], [200]),
    lambda d: planning.run_cindex_precision_mode(d, [0.7], ns=[100]),
    lambda d: planning.run_cindex_benchmark_mode(d, [0.7], [0.6], ns=[100]),
    lambda d: d.lambda_c_for(1.3),
])
def test_paths_without_censoring_models_raise(run):
    with pytest.raises(ValueError, match="censoring"):
        run(StudyDesign(**CALENDAR))


# --- realized censoring in every output ------------------------------------------------

def test_engines_report_realized_censoring():
    out = simulate_power(n=300, hr=1.3, censor_rate=0.25, n_sims=40, seed=1, details=True)
    assert abs(out["censor_rate_realized"] - 0.25) < 4 * out["censor_rate_realized_mcse"]
    wan = simulate_waning_effect(n=200, hr_early=1.5, tau=12, censor_rate=0.25, n_sims=10)
    assert {"censor_rate_realized", "censor_rate_realized_mcse"} <= set(wan)
    df = simulate(Scenario(analyses=(CoxWald(),)), n=200, n_sims=20, seed=0)
    assert {"censor_rate_realized", "censor_rate_realized_mcse"} <= set(df.columns)
