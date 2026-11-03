import numpy as np
import pandas as pd
import pytest
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index as lifelines_c

from survpower.censoring import calibrate_censoring_rate
from survpower.concordance import (
    benchmark_sample_size,
    beta_of_c,
    beta_of_c_approx,
    c_of_beta,
    censoring_survival_left,
    harrell_c_with_se,
    harrell_limit,
    precision_half_width,
    precision_sample_size,
    simulate_cindex_power,
    uno_c_with_se,
    uno_limit,
)
from survpower.fitting import cox_fit_1d

LAMBDA0 = np.log(2) / 15


def _cohort(n=400, beta=0.8, lam_c=0.01, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(n)
    T = rng.exponential(1 / (LAMBDA0 * np.exp(beta * X)))
    C = rng.exponential(1 / lam_c, n)
    return np.minimum(T, C), T <= C, X


@pytest.mark.parametrize("c, hr", [(0.55, 1.20), (0.60, 1.45), (0.65, 1.80), (0.70, 2.33), (0.75, 3.24)])
def test_table_b1_mapping(c, hr):
    assert np.exp(beta_of_c(c)) == pytest.approx(hr, abs=0.01)


def test_report_hrs_as_concordance():
    assert c_of_beta(np.log(1.10)) == pytest.approx(0.527, abs=5e-4)
    assert c_of_beta(np.log(1.15)) == pytest.approx(0.539, abs=5e-4)


def test_mapping_round_trip_and_symmetry():
    for b in (0.05, 0.4, 1.3):
        assert beta_of_c(c_of_beta(b)) == pytest.approx(b, rel=1e-8)
        assert c_of_beta(-b) == pytest.approx(c_of_beta(b))


def test_rule_of_thumb_within_3_percent():
    for c in np.linspace(0.55, 0.75, 9):
        assert beta_of_c_approx(c) == pytest.approx(beta_of_c(c), rel=0.03)


def test_harrell_limit_table_b1():
    b = beta_of_c(0.70)
    lam_c = calibrate_censoring_rate(b, 0.2)
    assert lam_c == pytest.approx(0.00954, abs=2e-5)
    assert harrell_limit(b, lam_c) == pytest.approx(0.7032, abs=5e-4)


def test_harrell_matches_lifelines():
    t, d, x = _cohort()
    c, se = harrell_c_with_se(t, d, x)
    assert c == pytest.approx(lifelines_c(t, -x, d), abs=1e-12)
    assert 0 < se < 0.05


def test_uno_equals_harrell_without_censoring():
    t, _, x = _cohort(lam_c=1e-9)
    d = np.ones_like(t, dtype=bool)
    assert uno_c_with_se(t, d, x, tau=np.inf) == pytest.approx(harrell_c_with_se(t, d, x))


def test_censoring_km_left_limit():
    t = np.array([1.0, 2.0, 3.0, 4.0])
    d = np.array([True, False, True, False])
    # censoring at t=2 (3 at risk): G(t-) = 1, 1, 2/3, 2/3
    assert censoring_survival_left(t, d) == pytest.approx([1, 1, 2 / 3, 2 / 3])


@pytest.mark.parametrize("method", ["harrell", "uno"])
def test_se_matches_simulation_sd(method):
    b = beta_of_c(0.70)
    lam_c = calibrate_censoring_rate(b, 0.3)
    r = simulate_cindex_power(300, b, lam_c, n_sims=300, method=method, seed=1)
    assert r["mean_se_c"] == pytest.approx(r["sd_c"], rel=0.15)


def test_uno_converges_to_uno_limit():
    b = beta_of_c(0.70)
    lam_c = calibrate_censoring_rate(b, 0.4)
    tau = 30.0
    r = simulate_cindex_power(400, b, lam_c, n_sims=150, method="uno", tau=tau, seed=2)
    assert r["mean_c"] == pytest.approx(uno_limit(b, tau), abs=0.006)


def test_fast_cox_matches_lifelines():
    t, d, x = _cohort(seed=4)
    b, se = cox_fit_1d(t, d, x)
    cph = CoxPHFitter().fit(pd.DataFrame(dict(time=t, event=d, X=x)), "time", "event")
    assert b == pytest.approx(cph.params_["X"], abs=1e-6)
    assert se == pytest.approx(cph.standard_errors_["X"], rel=1e-5)


def test_benchmark_and_precision_closed_forms():
    # Table B3 / B.4 (the report used the unrounded sigma_C ~ 0.323): +/-1 subject
    assert abs(benchmark_sample_size(0.7032, 0.65, 0.323) - 228) <= 1
    assert abs(benchmark_sample_size(0.70, 0.65, 0.323) - 258) <= 1
    assert abs(precision_sample_size(0.323, 0.05) - 160) <= 1
    assert precision_half_width(161, 0.323) <= 0.05
    with pytest.raises(ValueError):
        benchmark_sample_size(0.64, 0.65, 0.3)


def test_type_one_error_near_alpha():
    lam_c = calibrate_censoring_rate(0.0, 0.2)
    r = simulate_cindex_power(120, 0.0, lam_c, n_sims=1500, seed=3)
    assert r["power_cox"] == pytest.approx(0.05, abs=0.02)
