import numpy as np
import pandas as pd
import pytest
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index as lifelines_c
from scipy import stats
from scipy.special import expit

from survpower.censoring import calibrate_censoring_rate
from survpower.concordance import (
    benchmark_sample_size,
    beta_of_c,
    beta_of_c_approx,
    c_of_beta,
    censoring_survival_left,
    concordance_index,
    concordance_limit,
    gonen_heller_c_with_se,
    harrell_c_with_se,
    harrell_limit,
    mapping_table,
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


@pytest.mark.parametrize("method", ["harrell", "uno", "gonen-heller"])
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


# --- Gonen-Heller -------------------------------------------------------------

def test_gonen_heller_matches_pairwise_definition():
    # Gonen & Heller (2005), eq. 3: I(b'x_ji < 0) / (1 + exp(b'x_ji)) summed over both orders
    rng = np.random.default_rng(0)
    X, b = rng.standard_normal((40, 2)), np.array([0.7, -0.4])
    eta = X @ b
    tot = 0.0
    for i in range(40):
        for j in range(i + 1, 40):
            d = eta[j] - eta[i]
            tot += (d < 0) / (1 + np.exp(d)) + (-d < 0) / (1 + np.exp(-d))
    assert gonen_heller_c_with_se(X, b)[0] == pytest.approx(tot / (40 * 39 / 2), rel=1e-12)


def test_gonen_heller_gradient_term_matches_finite_differences():
    rng = np.random.default_rng(1)
    n, X, b = 150, rng.standard_normal((150, 3)), np.array([0.5, -0.3, 0.2])
    h = 0.5 * np.std(X @ b, ddof=1) * n ** (-1 / 3)

    def smoothed(beta):
        D = (X @ beta)[None, :] - (X @ beta)[:, None]
        F = stats.norm.cdf(-D / h) * expit(-D) + stats.norm.cdf(D / h) * expit(D)
        np.fill_diagonal(F, 0.0)
        return F.sum() / (n * (n - 1))

    se0 = gonen_heller_c_with_se(X, b, bandwidth=h)[1]
    for k, e in enumerate(np.eye(3) * 1e-6):
        g = (smoothed(b + e) - smoothed(b - e)) / 2e-6
        cov = np.zeros((3, 3)); cov[k, k] = 1.0
        se = gonen_heller_c_with_se(X, b, cov, bandwidth=h)[1]
        assert se ** 2 - se0 ** 2 == pytest.approx(g ** 2, rel=1e-6)


@pytest.mark.parametrize("censor_rate", [0.2, 0.6])
def test_gonen_heller_unbiased_for_population_c_under_censoring(censor_rate):
    b = beta_of_c(0.70)
    lam_c = calibrate_censoring_rate(b, censor_rate)
    r = simulate_cindex_power(300, b, lam_c, n_sims=200, method="gonen-heller", seed=6)
    assert concordance_limit(b, lam_c, "gonen-heller") == pytest.approx(0.70, abs=1e-9)
    assert r["mean_c"] == pytest.approx(0.70, abs=4 * r["mean_c_mcse"] + 1e-3)


def test_gonen_heller_multivariable_fits_cox_model():
    rng = np.random.default_rng(2)
    n = 300
    X = rng.standard_normal((n, 2))
    T = rng.exponential(1 / (LAMBDA0 * np.exp(X @ np.array([0.8, 0.4]))))
    C = rng.exponential(1 / 0.01, n)
    t, d = np.minimum(T, C), T <= C
    cph = CoxPHFitter().fit(pd.DataFrame(dict(time=t, event=d, a=X[:, 0], b=X[:, 1])),
                            "time", "event")
    c, se = concordance_index(t, d, X, method="gonen-heller")
    c2, se2 = gonen_heller_c_with_se(X, cph.params_, cph.variance_matrix_)
    assert c == pytest.approx(c2, abs=1e-6) and se == pytest.approx(se2, rel=1e-4)
    # the two-covariate model discriminates better than the biomarker alone
    assert c > concordance_index(t, d, X[:, 0], method="gonen-heller")[0]
    assert 0 < se < 0.05


def test_gonen_heller_univariate_uses_fast_fit_and_is_orientation_free():
    t, d, x = _cohort(seed=5)
    c, se = concordance_index(t, d, x, method="gonen-heller")
    b, se_b = cox_fit_1d(t, d, x)
    assert (c, se) == pytest.approx(gonen_heller_c_with_se(x, b, se_b ** 2))
    assert concordance_index(t, d, -x, method="gonen-heller") == pytest.approx((c, se))
    with pytest.raises(ValueError):
        gonen_heller_c_with_se(np.ones((10, 2)), [1.0])


def test_mapping_table_matches_table8():
    df = mapping_table()
    assert np.round(df["beta"], 3).tolist() == [0.179, 0.370, 0.585, 0.844, 1.177]
    assert df["hr"].round(2).tolist() == [1.20, 1.45, 1.80, 2.33, 3.24]
    assert df["harrell_limit"].round(4).tolist() == [0.5501, 0.6006, 0.6517, 0.7032, 0.7548]
    assert df["N80_closed_form"].tolist() == [306, 72, 29, 14, 8]
