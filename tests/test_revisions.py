"""Checks for the analyses added with the revised report (November 2026)."""

import numpy as np
import pytest
from scipy.stats import norm

from survpower.calibration import bootstrap_probit_inverse, estimate_required_n_by_simulation
from survpower.closed_form import (
    dichotomized_design,
    dichotomized_power,
    dichotomized_sample_size,
)
from survpower.simulate import (
    simulate_adjusted_vs_unadjusted,
    simulate_continuous_vs_dichotomized,
    simulate_epv_diagnostic,
    simulate_power,
)


@pytest.mark.parametrize("n, hr, freedman, schoenfeld", [
    (1200, 1.10, 0.652, 0.654), (1200, 1.15, 0.931, 0.933),
    (250, 1.10, 0.188, 0.188), (250, 1.15, 0.348, 0.351),
])
def test_table1_closed_forms(n, hr, freedman, schoenfeld):
    assert dichotomized_power(n, hr, 0.2, method="freedman") == pytest.approx(freedman, abs=6e-4)
    assert dichotomized_power(n, hr, 0.2, method="schoenfeld") == pytest.approx(schoenfeld, abs=6e-4)


def test_schoenfeld_equals_are_prediction():
    # with the plug-in group HR, Schoenfeld = Phi(|beta| sqrt((2/pi) m) - z)
    hr, n, cr = 1.15, 1200, 0.2
    ghr, p_con, p_exp = dichotomized_design(hr, cr)
    m = n / 2 * (p_con + p_exp)
    are = norm.cdf(np.log(hr) * np.sqrt(2 / np.pi * m) - norm.ppf(0.975))
    assert dichotomized_power(n, hr, cr, method="schoenfeld") == pytest.approx(are, abs=1e-4)
    assert dichotomized_power(n, hr, cr, method="freedman") < dichotomized_power(n, hr, cr, method="schoenfeld")


def test_schoenfeld_sample_size_reaches_target():
    n = dichotomized_sample_size(0.8, 1.2, 0.2, method="schoenfeld")
    assert dichotomized_power(n, 1.2, 0.2, method="schoenfeld") >= 0.8
    assert dichotomized_power(n - 4, 1.2, 0.2, method="schoenfeld") < 0.8


def test_same_data_continuous_decisions_match_simulate_power():
    r = simulate_continuous_vs_dichotomized(250, 1.2, 0.2, n_sims=80, seed=5)
    assert r["power_cont"] == pytest.approx(simulate_power(250, 1.2, 0.2, n_sims=80, seed=5)[0], abs=1e-9)
    assert r["power_dich"] < r["power_cont"]
    assert r["group_hr_fitted"] == pytest.approx(1.2 ** 1.5958, abs=0.05)


def test_bootstrap_interval_brackets_n_star():
    out = estimate_required_n_by_simulation(1.2, 0.8, n_sims_per_point=300, confirm_n_sims=200,
                                            seed=1, scan_below=0)
    assert out["n_ci_low"] <= out["n_calibrated"] <= out["n_ci_high"]
    # interval width shrinks with more replicates per point
    x = np.sqrt([200, 260, 320, 380, 440])
    p = [0.55, 0.65, 0.74, 0.81, 0.86]
    w300 = np.diff(bootstrap_probit_inverse(x, p, 300, 0.8, n_boot=800))[0]
    w3000 = np.diff(bootstrap_probit_inverse(x, p, 3000, 0.8, n_boot=800))[0]
    assert w3000 < w300 / 2


def test_epv_reports_null_and_size_adjusted_power():
    r = simulate_epv_diagnostic(150, 1.2, 0.2, p=3, r2_target=0.3, n_sims=40, seed=2)
    for key in ("type1_error", "size_adj_power", "bias_beta", "bias_beta_mcse", "mean_se"):
        assert key in r
    assert 0 <= r["type1_error"] <= 1


def test_omitting_prognostic_covariates_attenuates_and_costs_power():
    r = simulate_adjusted_vs_unadjusted(600, 1.15, 0.2, p=5, sigma_z=1.0, n_sims=40, seed=3)
    assert r["mean_HR_unadjusted"] < r["mean_HR_adjusted"]
    assert r["power_unadjusted"] < r["power_adjusted"]


def test_logrank_1d_matches_lifelines():
    from lifelines.statistics import logrank_test
    from survpower.fitting import logrank_1d
    rng = np.random.default_rng(3)
    for _ in range(20):
        G = (rng.random(150) < 0.4).astype(float)
        T = rng.exponential(1 / np.exp(0.4 * G))
        C = rng.exponential(2.0, 150)
        t, e = np.minimum(T, C), T <= C
        chi2 = logrank_test(t[G == 1], t[G == 0], e[G == 1], e[G == 0]).test_statistic
        assert logrank_1d(t, e, G) ** 2 == pytest.approx(chi2, rel=1e-10)


def test_schoenfeld_predicts_median_split_logrank_power():
    # Schoenfeld's formula is the log-rank power; the Wald test on G matches it
    r = simulate_continuous_vs_dichotomized(500, 1.15, 0.2, n_sims=2000, seed=1)
    closed = dichotomized_power(500, 1.15, 0.2, method="schoenfeld")
    mcse = r["power_dich_logrank_mcse"]
    assert abs(r["power_dich_logrank"] - closed) < max(3 * mcse, 0.02)
    assert abs(r["power_dich_logrank"] - r["power_dich"]) < 3 * mcse
