"""Baseline-hazard families calibrated to a median OS (#14).

Every generator is checked against an analytic result: closed-form survival
functions (scipy's where one exists), the asymptotic MCSE of a sample median,
binomial MCSEs at analytic quantiles, and Cox's invariance to monotone maps of time."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, optimize, stats

import survpower
from survpower.baseline import (
    LN2,
    Exponential,
    Gompertz,
    EmpiricalBaseline,
    LogLogistic,
    NumericBaseline,
    PiecewiseExponential,
    Weibull,
    _normal_rule,
    baseline_from_dict,
    marginal_cure_fraction,
)
from survpower.censoring import censor_rate_from_design
from survpower.concordance import c_of_beta, harrell_limit, uno_limit
from survpower.design import StudyDesign
from survpower.fitting import cox_fit_1d
from survpower.planning import (
    run_cindex_benchmark_mode,
    run_cindex_precision_mode,
    run_detectable_effect_mode,
    run_power_mode,
    sample_size,
)
from survpower.scenario import (
    Covariates,
    Effect,
    EventTimeModel,
    ExponentialCensoring,
    ExponentialPH,
    PHModel,
    Scenario,
    run_replicates,
    scenario_for,
    simulate,
)

M = 15.0
BETA = np.log(1.3)
def _weibull_cohort(n, beta, seed, censor_scale=40.0):
    """(time, event, biomarker) from Weibull(15, 1.4) PH in a raw biomarker with mean 5 and
    SD 2, beta per SD; times rounded to 0.1 month, so there are ties."""
    rng = np.random.default_rng(seed)
    x = rng.normal(5.0, 2.0, n)
    z = (x - 5.0) / 2.0
    T = Weibull(M, 1.4).inv_cumhaz(rng.standard_exponential(n) * np.exp(-beta * z))
    C = rng.exponential(censor_scale, n)
    return np.maximum(np.round(np.minimum(T, C), 1), 0.1), (T <= C).astype(int), x


FAMILIES = {
    "exponential": Exponential(M),
    "weibull_rising": Weibull(M, 1.5),
    "weibull_falling": Weibull(M, 0.7),
    "gompertz": Gompertz(M, 0.05),
    "gompertz_cure": Gompertz(M, -0.05),
    "piecewise": PiecewiseExponential(M, (6, 12), (1.0, 1.5, 2.0)),
    "loglogistic": LogLogistic(M, 1.5),
    "numeric": NumericBaseline(lambda t: 1.0 + 0.05 * t, M),
    "empirical": EmpiricalBaseline.from_data(*_weibull_cohort(400, 0.3, seed=11)),
}
PROPER = {k: v for k, v in FAMILIES.items() if v.cure_fraction == 0}
FEW = dict(n_sims=30, n_sims_grid=20, seed=5)


def _draw(baseline, x, seed):
    """T for covariate values x from PHModel(baseline, BETA)."""
    model = PHModel(baseline, BETA)
    return model.draw(Covariates(np.asarray(x, dtype=float)), np.random.default_rng(seed))


# --- calibration and inversion --------------------------------------------------------

@pytest.mark.parametrize("name", FAMILIES)
def test_inv_cumhaz_inverts_cumhaz(name):
    b = FAMILIES[name]
    t = np.geomspace(1e-2, 60, 300)
    np.testing.assert_allclose(b.inv_cumhaz(b.cumhaz(t)), t, rtol=1e-10, atol=0)


@pytest.mark.parametrize("name", FAMILIES)
def test_the_baseline_median_is_median_os(name):
    b = FAMILIES[name]
    m = M if name != "empirical" else b.median  # an estimated median
    assert b.median == pytest.approx(m, rel=1e-15)
    assert b.cumhaz(np.array(m)) == pytest.approx(LN2, rel=1e-12)
    assert b.inv_cumhaz(np.array(LN2)) == pytest.approx(m, rel=1e-10)


@pytest.mark.parametrize("b, ref", [
    (Exponential(M), stats.expon(scale=M / LN2)),
    (Weibull(M, 1.5), stats.weibull_min(c=1.5, scale=M / LN2 ** (1 / 1.5))),
    (Weibull(M, 0.7), stats.weibull_min(c=0.7, scale=M / LN2 ** (1 / 0.7))),
    (LogLogistic(M, 1.5), stats.fisk(c=1.5, scale=M)),
    (Gompertz(M, 0.05), stats.gompertz(c=Gompertz(M, 0.05).lam / 0.05, scale=1 / 0.05)),
], ids=["exponential", "weibull_rising", "weibull_falling", "loglogistic", "gompertz"])
def test_cumhaz_matches_scipy(b, ref):
    t = np.linspace(0.1, 80, 200)
    np.testing.assert_allclose(b.cumhaz(t), -ref.logsf(t), rtol=1e-10)
    np.testing.assert_allclose(b.density(t), ref.pdf(t), rtol=1e-9)


def test_piecewise_and_numeric_cumhaz_in_closed_form():
    pw = FAMILIES["piecewise"]  # H_shape(15) = 6 * 1 + 6 * 1.5 + 3 * 2 = 21
    np.testing.assert_allclose(pw.cumhaz(np.array([3.0, 6.0, 9.0, 30.0])),
                               LN2 / 21 * np.array([3.0, 6.0, 10.5, 51.0]), rtol=1e-13)
    num = FAMILIES["numeric"]  # H_shape(t) = t + 0.025 t^2
    t = np.array([0.5, 7.0, 40.0])
    np.testing.assert_allclose(num.cumhaz(t), LN2 * (t + 0.025 * t ** 2) / (M + 0.025 * M ** 2),
                               rtol=1e-12)


def test_gompertz_shape_zero_is_exponential_and_annual_hr():
    t = np.linspace(0, 50, 11)
    np.testing.assert_allclose(Gompertz(M, 0.0).cumhaz(t), Exponential(M).cumhaz(t), rtol=1e-15)
    g = Gompertz.from_annual_hr(M, 1.5)
    assert g.annual_hr == pytest.approx(1.5)
    assert g.hazard(np.array(18.0)) / g.hazard(np.array(6.0)) == pytest.approx(1.5)


@pytest.mark.parametrize("make", [
    lambda: Weibull(M, 0), lambda: Weibull(-1, 1.5), lambda: LogLogistic(M, np.inf),
    lambda: Gompertz(M, np.nan), lambda: Gompertz(M, 50.0),
    lambda: PiecewiseExponential(M, (6, 12), (1, 2)),
    lambda: PiecewiseExponential(M, (12, 6), (1, 2, 3)),
    lambda: PiecewiseExponential(M, (6,), (1, 0)),
    lambda: NumericBaseline(None, M), lambda: NumericBaseline(lambda t: 0 * t, M),
])
def test_invalid_parameters_raise(make):
    with pytest.raises(ValueError):
        make()


# --- draws: medians and distribution ---------------------------------------------------

@pytest.mark.parametrize("name", FAMILIES)
def test_sample_median_at_beta_zero(name):
    """Within 4 MCSE of median_os, with the MCSE of a sample median sqrt(0.25 / n) / f(m)."""
    b, n = FAMILIES[name], 200_000
    T = PHModel(b, 0.0).draw(Covariates(np.zeros(n)), np.random.default_rng(1))
    mcse = np.sqrt(0.25 / n) / b.density(np.array(b.median))
    assert abs(np.median(T) - b.median) < 4 * mcse


def _marginal_density(b, t, beta):
    x, w = _normal_rule()
    r = np.exp(beta * x)
    return float(w @ (b.hazard(np.array(t)) * r * np.exp(-b.cumhaz(np.array(t)) * r)))


@pytest.mark.parametrize("name", FAMILIES)
def test_marginal_median(name):
    """With beta = log 1.5 and X ~ N(0, 1), the sample median matches the root of
    E_X[S0(t)^exp(beta X)] = 1/2 within 4 MCSE."""
    b, n, beta = FAMILIES[name], 200_000, np.log(1.5)
    model = PHModel(b, beta)
    target = model.marginal_median()
    x, w = _normal_rule()
    assert w @ np.exp(-b.cumhaz(np.array(target)) * np.exp(beta * x)) == pytest.approx(0.5)
    T = model.draw(Covariates(np.random.default_rng(2).standard_normal(n)),
                   np.random.default_rng(3))
    mcse = np.sqrt(0.25 / n) / _marginal_density(b, target, beta)
    assert abs(np.median(T) - target) < 4 * mcse


@pytest.mark.parametrize("name", FAMILIES)
@pytest.mark.parametrize("x", [-1.0, 0.0, 1.0])
def test_distribution_at_fixed_x(name, x):
    """Empirical survival at the 10/25/50/75/90% analytic quantiles (found from the
    cumulative hazard by root-finding, not from inv_cumhaz) is within 4 binomial MCSEs.
    Quantiles beyond a cure fraction do not exist: the share never failing is checked."""
    b, n = FAMILIES[name], 50_000
    T = _draw(b, np.full(n, x), seed=4)
    r = np.exp(BETA * x)
    for p in (0.10, 0.25, 0.50, 0.75, 0.90):
        h = -np.log1p(-p) / r  # S(t | x) = exp(-H0(t) r) = 1 - p
        if h >= b.cumhaz_max:
            continue
        q = optimize.brentq(lambda t: b.cumhaz(np.array(t)) - h, 1e-9, 1e4, xtol=1e-12)
        s_emp = np.mean(T > q)
        assert abs(s_emp - (1 - p)) < 4 * np.sqrt(p * (1 - p) / n), p
    cure = np.exp(-b.cumhaz_max * r)
    assert abs(np.mean(np.isinf(T)) - cure) < 4 * np.sqrt(cure * (1 - cure) / n) + 1e-12


@pytest.mark.parametrize("name", PROPER)
def test_cox_is_invariant_to_the_baseline(name):
    """Mapping exponential T and C alike through t -> H_W^{-1}(H_E(t)) leaves every risk
    set unchanged, so the Cox fit is identical (report Section 2.2)."""
    rng = np.random.default_rng(6)
    X = rng.standard_normal(400)
    T = ExponentialPH(BETA, LN2 / M).draw(Covariates(X), rng)
    C = rng.exponential(scale=30.0, size=400)
    to_b = lambda t: FAMILIES[name].inv_cumhaz(Exponential(M).cumhaz(t))
    b0, se0 = cox_fit_1d(np.minimum(T, C), (T <= C).astype(int), X)
    Tb, Cb = to_b(T), to_b(C)
    b1, se1 = cox_fit_1d(np.minimum(Tb, Cb), (Tb <= Cb).astype(int), X)
    assert abs(b1 - b0) < 1e-12 and abs(se1 - se0) < 1e-12


def test_numeric_baseline_matches_weibull_draws():
    """Weibull(15, 1.5) has hazard proportional to sqrt(t); same seed, same draws."""
    x = np.random.default_rng(7).standard_normal(20_000)
    num = _draw(NumericBaseline(lambda t: np.sqrt(t), M), x, seed=8)
    np.testing.assert_allclose(num, _draw(Weibull(M, 1.5), x, seed=8), rtol=1e-6)


# --- the engine: censoring, HR recovery, MCSE columns -----------------------------------

def _scenario(b, censor_rate):
    return Scenario(event_times=PHModel(b, BETA),
                    censoring=ExponentialCensoring(target_rate=censor_rate, n_probe=200_000))


@pytest.mark.parametrize("name", FAMILIES)
def test_realized_censoring_hits_the_target(name):
    target = 0.45 if FAMILIES[name].cure_fraction else 0.30
    reps = run_replicates(_scenario(FAMILIES[name], target), n=500, n_sims=200, seed=9)
    cens = 1 - reps.events / 500
    assert abs(cens.mean() - target) < 4 * cens.std(ddof=1) / np.sqrt(200)


@pytest.mark.slow
@pytest.mark.parametrize("name", FAMILIES)
def test_hr_recovery(name):
    target = 0.45 if FAMILIES[name].cure_fraction else 0.20
    row = simulate(_scenario(FAMILIES[name], target), n=20_000, n_sims=200, seed=10).iloc[0]
    assert abs(row["bias_beta"]) < 3 * row["bias_beta_mcse"]


def test_outputs_carry_mcse_columns():
    sc = _scenario(Weibull(M, 1.5), 0.2)
    assert isinstance(sc.event_times, EventTimeModel) and sc.has_closed_form
    df = simulate(sc, n=100, n_sims=20, seed=1)
    assert {"power_mcse", "bias_beta_mcse", "coverage_mcse", "emp_se_mcse"} <= set(df.columns)


# --- defective (cure-fraction) Gompertz --------------------------------------------------

def test_gompertz_cure_fraction():
    g = FAMILIES["gompertz_cure"]
    assert g.cumhaz_max == pytest.approx(LN2 / (1 - np.exp(-0.05 * M)))
    assert 0 < g.cure_fraction < 0.5
    assert np.isinf(g.inv_cumhaz(np.array([g.cumhaz_max * 1.01]))).all()
    assert marginal_cure_fraction(g, 0.0) == pytest.approx(g.cure_fraction)
    assert marginal_cure_fraction(Weibull(M, 2.0), BETA) == 0


def test_censoring_below_the_cure_fraction_raises():
    g = FAMILIES["gompertz_cure"]
    design = StudyDesign(baseline=g, censor_rate=0.2)
    with pytest.raises(ValueError, match="cure fraction"):
        scenario_for(design, Effect.hr(1.3))
    assert scenario_for(design, Effect.hr(1.3), censor_rate=0.5).event_times.baseline is g


# --- StudyDesign ----------------------------------------------------------------------

def test_default_design_is_unchanged():
    for design in (StudyDesign(), StudyDesign(baseline=Exponential(M))):
        assert design.non_default_models() == ()
        assert design.lambda0 == LN2 / M
        assert scenario_for(design, Effect.hr(1.2)).event_times == ExponentialPH(np.log(1.2),
                                                                                 LN2 / M)
    a = run_power_mode(StudyDesign(**FEW), [1.3], [150])
    b = run_power_mode(StudyDesign(baseline=Exponential(M), **FEW), [1.3], [150])
    assert a.equals(b)


def test_design_validates_the_baseline():
    with pytest.raises(ValueError, match="median_os=20"):
        StudyDesign(baseline=Weibull(20, 1.5))
    with pytest.raises(ValueError, match="survpower.baseline"):
        StudyDesign(baseline="weibull")
    design = StudyDesign(median_os=20, baseline=Weibull(20, 1.5))
    assert design.non_default_models() == ("baseline",)
    assert design.baseline_kwargs() == dict(baseline=Weibull(20, 1.5))
    assert design.baseline_hazard is design.baseline
    assert hash(design) == hash(StudyDesign(median_os=20, baseline=Weibull(20.0, 1.5)))
    with pytest.raises(ValueError, match="use design.baseline"):
        design.lambda0


@pytest.mark.parametrize("name", [k for k in FAMILIES if k != "numeric"])
def test_baselines_round_trip_through_json(name):
    b = FAMILIES[name]
    assert baseline_from_dict(json.loads(json.dumps(b.to_dict()))) == b


def test_numeric_baseline_is_api_only():
    with pytest.raises(TypeError, match="API-only"):
        FAMILIES["numeric"].to_dict()
    with pytest.raises(ValueError, match="unknown baseline family"):
        baseline_from_dict({"family": "numeric", "median_os": 15})


def test_families_are_exported():
    assert survpower.Weibull is Weibull and survpower.PiecewiseExponential is PiecewiseExponential


# --- censoring from an accrual calendar and concordance limits ---------------------------

def test_accrual_censoring_under_a_baseline():
    """P(C < T) = E_X[(1/A) int_F^{A+F} S0(c)^exp(beta X) dc] for uniform C_adm."""
    b, A, F, n = Weibull(M, 1.5), 24.0, 12.0, 500_000
    x, w = _normal_rule()
    s_avg = [integrate.quad(lambda c: np.exp(-b.cumhaz(np.array(c)) * r), F, A + F)[0] / A
             for r in np.exp(BETA * x)]
    expected = float(w @ np.array(s_avg))
    got = censor_rate_from_design(1.3, A, F, baseline=b, n_probe=n)
    assert abs(got - expected) < 4 * np.sqrt(expected * (1 - expected) / n)
    assert censor_rate_from_design(1.3, A, F, baseline=Exponential(M)) == \
        censor_rate_from_design(1.3, A, F, lambda0=LN2 / M)


def test_harrell_limit_monte_carlo_matches_the_exponential_form():
    constant = PiecewiseExponential(M, (10.0,), (1.0, 1.0))  # exponential, by the MC route
    exact = harrell_limit(0.5, 0.03, lambda0=LN2 / M)
    assert harrell_limit(0.5, 0.03, baseline=constant) == pytest.approx(exact, abs=1.5e-3)


def test_uno_limit_without_truncation_is_the_population_concordance():
    """Under PH the uncensored concordance is K(beta) whatever the baseline."""
    assert uno_limit(0.5, 1e9, baseline=Weibull(M, 0.7)) == pytest.approx(c_of_beta(0.5),
                                                                          abs=1.5e-3)


# --- planning paths ----------------------------------------------------------------------

WEIBULL = StudyDesign(baseline=Weibull(M, 1.5), **FEW)


def test_power_under_a_weibull_baseline():
    row = run_power_mode(WEIBULL, [1.3], [150]).iloc[0]
    exp_row = run_power_mode(StudyDesign(**FEW), [1.3], [150]).iloc[0]
    assert abs(row["power_sim"] - exp_row["power_sim"]) < 0.25
    assert np.isfinite(row["power_closed_form"])  # still PH
    med_low, med_high = row["split_median_os_low"], row["split_median_os_high"]
    assert med_high < M < med_low
    assert "median OS" in row["summary"]


def test_median_split_medians_map_through_the_baseline():
    """The group medians are the exponential ones mapped through H_W^{-1}(H_E(t))."""
    exp_row = run_power_mode(StudyDesign(**FEW), [1.3], [150]).iloc[0]
    row = run_power_mode(WEIBULL, [1.3], [150]).iloc[0]
    w, e = Weibull(M, 1.5), Exponential(M)
    for g in ("low", "high"):
        assert row[f"split_median_os_{g}"] == pytest.approx(
            float(w.inv_cumhaz(e.cumhaz(np.array(exp_row[f"split_median_os_{g}"])))), abs=0.06)


def test_other_paths_take_the_baseline():
    sample_size(hr=1.5, design=WEIBULL)
    run_detectable_effect_mode(WEIBULL, [150], [0.8])
    run_power_mode(StudyDesign(n_covariates=2, r2=0.2, baseline=Weibull(M, 1.5), **FEW),
                   [1.4], [150])
    run_cindex_benchmark_mode(WEIBULL, [0.7], [0.6], ns=[100])
    run_cindex_precision_mode(StudyDesign(baseline=Weibull(M, 1.5), c_estimator="uno", **FEW),
                              [0.7], ns=[100])


def test_accrual_design_with_a_baseline():
    design = StudyDesign(accrual_months=24, followup_months=12, baseline=Gompertz(M, 0.03))
    assert design.censor_rate_for(1.3) != StudyDesign(accrual_months=24,
                                                      followup_months=12).censor_rate_for(1.3)


# --- the Breslow baseline of a cohort (#84) -----------------------------------------------

def _no_ties(n=300, seed=12):
    t, d, x = _weibull_cohort(n, 0.3, seed)
    return t + np.random.default_rng(seed).uniform(0, 1e-3, n), d, x  # break the ties


def test_breslow_matches_lifelines():
    from lifelines import CoxPHFitter
    t, d, x = _no_ties()
    b, info = EmpiricalBaseline.from_data_info(t, d, x)
    z = (x - x.mean()) / x.std(ddof=1)
    cph = CoxPHFitter().fit(pd.DataFrame(dict(time=t, event=d, z=z)), "time", "event")
    expected = cph.baseline_cumulative_hazard_.iloc[:, 0].loc[list(b.times)].to_numpy()
    np.testing.assert_allclose(b.cumhaz_values, expected, rtol=1e-10)
    assert info["beta_hat"] == pytest.approx(cph.params_["z"])
    assert (info["n"], info["events"]) == (300, int(d.sum()))


def test_without_a_biomarker_it_is_nelson_aalen():
    from lifelines import NelsonAalenFitter
    t, d, _ = _weibull_cohort(300, 0.3, seed=13)  # with ties
    b = EmpiricalBaseline.from_data(t, d)
    na = NelsonAalenFitter(nelson_aalen_smoothing=False).fit(t, d).cumulative_hazard_.iloc[:, 0]
    np.testing.assert_allclose(b.cumhaz_values, na.loc[list(b.times)].to_numpy(), rtol=1e-12)


def test_interpolation_and_tail():
    b = EmpiricalBaseline(times=(2.0, 5.0), cumhaz_values=(0.2, 0.5), tail_hazard=0.25)
    t = np.array([1.0, 2.0, 3.5, 5.0, 9.0])
    np.testing.assert_allclose(b.cumhaz(t), [0.1, 0.2, 0.35, 0.5, 1.5])
    np.testing.assert_allclose(b.hazard(t), [0.1, 0.1, 0.1, 0.25, 0.25])
    np.testing.assert_allclose(b.inv_cumhaz(b.cumhaz(t)), t)
    assert b.median == pytest.approx(5 + (LN2 - 0.5) / 0.25) and not b.median_estimated


def test_default_tail_is_the_hazard_over_the_last_events():
    t = np.arange(1.0, 31.0)
    b = EmpiricalBaseline.from_data(t, np.ones(30))
    k = 10  # the last 10 events are at 21..30: the window starts at the event time 20
    H = np.array(b.cumhaz_values)
    assert b.tail_hazard == pytest.approx((H[-1] - H[-1 - k]) / (30 - 20))
    assert EmpiricalBaseline.from_data(t, np.ones(30), tail_hazard=0.3).tail_hazard == 0.3


@pytest.mark.slow
def test_breslow_recovers_the_true_baseline():
    """A large Weibull PH cohort: H0 and the per-SD HR within sampling error of the truth."""
    t, d, x = _weibull_cohort(20_000, 0.3, seed=14, censor_scale=60.0)
    b, info = EmpiricalBaseline.from_data_info(t, d, x)
    assert abs(info["beta_hat"] - 0.3) < 4 * info["se"]
    for s in (3.0, 10.0, 15.0, 25.0):
        assert b.cumhaz(np.array(s)) == pytest.approx(Weibull(M, 1.4).cumhaz(np.array(s)),
                                                        rel=0.05)
    assert b.median == pytest.approx(M, rel=0.03)


@pytest.mark.parametrize("kwargs, match", [
    (dict(time=[1.0, 2.0, -1.0], event=[1, 1, 0]), "positive"),
    (dict(time=[1.0, 2.0, 3.0], event=[1, 2, 0]), "0 or 1"),
    (dict(time=[1.0, 2.0, 3.0], event=[1, 0, 0]), "two distinct event times"),
    (dict(time=[1.0, 2.0, 3.0], event=[1, 1, 0], biomarker=[2.0, 2.0, 2.0]), "constant"),
])
def test_from_data_validates(kwargs, match):
    with pytest.raises(ValueError, match=match):
        EmpiricalBaseline.from_data(**kwargs)


def test_missing_rows_are_dropped():
    t, d, x = _no_ties(100)
    x = x.copy(); x[:5] = np.nan
    assert EmpiricalBaseline.from_data_info(t, d, x)[1]["n"] == 95


def test_planning_with_an_empirical_baseline():
    b = EmpiricalBaseline.from_data(*_weibull_cohort(400, 0.3, seed=15))
    design = StudyDesign(median_os=b.median, baseline=b, **FEW)
    row = run_power_mode(design, [1.3], [150]).iloc[0]
    assert 0 < row["power_sim"] <= 1 and np.isfinite(row["split_median_os_low"])
    assert hash(design) and baseline_from_dict(json.loads(json.dumps(b.to_dict()))) == b
