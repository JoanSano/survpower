"""The composable scenario engine (`survpower.scenario`, #9)."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

import survpower
from survpower.censoring import calibrate_censoring_for, calibrate_censoring_rate
from survpower.dgp import gen_exponential_ph
from survpower.scenario import (
    CoxWald,
    Covariates,
    CensoringModel,
    EventTimeModel,
    ExponentialCensoring,
    ExponentialPH,
    LogRank,
    Scenario,
    run_replicates,
    simulate,
)
from survpower.simulate import simulate_continuous_vs_dichotomized, simulate_power


def _ph(hr=1.2, censor_rate=0.3, analyses=(CoxWald(),)):
    return Scenario(event_times=ExponentialPH(log_hr=np.log(hr)),
                    censoring=ExponentialCensoring(target_rate=censor_rate),
                    analyses=analyses)


def test_same_seed_same_result_and_different_seed_differs():
    a = simulate(_ph(), n=120, n_sims=40, seed=3)
    b = simulate(_ph(), n=120, n_sims=40, seed=3)
    c = simulate(_ph(), n=120, n_sims=40, seed=4)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)
    assert a.loc[0, "lambda_c"] != c.loc[0, "lambda_c"]  # calibration seed moves too


@pytest.mark.parametrize("engine", ["fast", "lifelines"])
@pytest.mark.parametrize("sides", [1, 2])
@pytest.mark.parametrize("hr", [1.25, 0.8])
def test_simulate_matches_simulate_power_bit_for_bit(engine, sides, hr):
    kw = dict(n=150, n_sims=30, seed=7, sides=sides)
    row = simulate(_ph(hr, analyses=(CoxWald(engine=engine),)), **kw).iloc[0]
    ref = simulate_power(hr=hr, censor_rate=0.3, engine=engine, details=True, **kw)
    for key, value in ref.items():
        assert row[key] == value or (np.isnan(row[key]) and np.isnan(value)), key
    power, events, lam_c = simulate_power(hr=hr, censor_rate=0.3, engine=engine, **kw)
    assert (row["power"], row["mean_events"], row["lambda_c"]) == (power, events, lam_c)


def test_continuous_vs_dichotomized_shares_cohorts():
    analyses = (CoxWald(), CoxWald(dichotomize=True), LogRank())
    reps = run_replicates(_ph(1.3, 0.25, analyses), n=200, n_sims=50, seed=2)
    ref = simulate_continuous_vs_dichotomized(200, 1.3, 0.25, n_sims=50, seed=2)
    assert ref["power_cont"] == round(reps.values["cox_wald"]["reject"].mean(), 3)
    assert ref["power_dich"] == round(reps.values["cox_wald_median_split"]["reject"].mean(), 3)
    assert ref["power_dich_logrank"] == round(reps.values["logrank_median_split"]["reject"].mean(), 3)
    assert ref["lam_c"] == reps.scenario.censoring.rate


def test_one_row_per_analysis_with_mcse_columns():
    df = simulate(_ph(analyses=(CoxWald(), LogRank())), n=100, n_sims=20, seed=1)
    assert list(df["analysis"]) == ["cox_wald", "logrank_median_split"]
    assert {"power", "power_mcse", "n_sims", "n_failed", "mean_events"} <= set(df.columns)
    assert {"bias_beta", "coverage_mcse"} <= set(df.columns)  # ExponentialPH has a log_hr
    assert np.isnan(df.loc[1, "bias_beta"])  # log-rank has no coefficient


def test_duplicate_analysis_names_rejected():
    with pytest.raises(ValueError, match="unique"):
        Scenario(analyses=(CoxWald(), CoxWald()))


@dataclass(frozen=True)
class _StepTimes:
    """Toy model defined only here: T = 1 above the cohort median of X, else 3
    (no randomness)."""

    is_ph: bool = field(default=False, init=False)

    def draw(self, cov, rng):
        return np.where(cov.X > np.median(cov.X), 1.0, 3.0)

    def survival(self, t, x):
        return (np.asarray(t) < np.where(np.asarray(x) > 0, 1.0, 3.0)).astype(float)


@dataclass(frozen=True)
class _FixedCensoring:
    at: float = 2.0

    def calibrated(self, sampler, seed):
        return self

    def draw(self, cov, rng):
        return np.full(cov.n, self.at)


def test_toy_models_run_without_library_changes():
    assert isinstance(_StepTimes(), EventTimeModel)
    assert isinstance(_FixedCensoring(), CensoringModel)
    sc = Scenario(event_times=_StepTimes(), censoring=_FixedCensoring(),
                  analyses=(LogRank(dichotomize=True),))
    reps = run_replicates(sc, n=60, n_sims=5, seed=0)
    # censoring at 2 leaves exactly the 30 high-X patients as events
    assert np.all(reps.events == 30)
    assert reps.values["logrank_median_split"]["reject"].all()
    row = simulate(sc, n=60, n_sims=5, seed=0).iloc[0]
    assert row["power"] == 1.0 and "bias_beta" not in row and "lambda_c" not in row


def test_generic_calibrator_matches_wrapper():
    lam = calibrate_censoring_for(lambda n, rng: gen_exponential_ph(n, 0.1, 0.05, rng)[0],
                                  0.2, seed=5, n_probe=50_000)
    assert lam == calibrate_censoring_rate(0.1, 0.2, lambda0=0.05, n_probe=50_000, seed=5)


def test_exponential_ph_survival_and_median():
    m = ExponentialPH.from_median_os(log_hr=0.0, median_os=12.0)
    assert m.survival(12.0, 0.0) == pytest.approx(0.5)
    assert m.survival(12.0, 1.0) == pytest.approx(0.5)  # log_hr = 0


def test_uncalibrated_censoring_needs_rate():
    with pytest.raises(ValueError, match="calibrated"):
        ExponentialCensoring().draw(Covariates(np.zeros(3)), np.random.default_rng(0))
    fixed = ExponentialCensoring(rate=0.1)
    assert fixed.calibrated(None, seed=0) is fixed


def test_exported_as_provisional():
    assert survpower.Scenario is Scenario
    assert survpower.simulate_scenario is simulate
    assert hasattr(survpower.simulate, "simulate_power")  # the module is not shadowed
