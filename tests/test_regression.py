"""Regression against the original report script (cox_ph_power_simulations.py).

The reference values below were produced by the original, single-file script
with the same arguments and seeds. With the lifelines engine the package must
reproduce them exactly -- this is what guarantees `survpower report`
regenerates the PDF's numbers.
"""

import pytest

from survpower.calibration import estimate_required_n_by_simulation
from survpower.censoring import calibrate_censoring_rate
from survpower.simulate import (
    simulate_crossing_hazards,
    simulate_power,
    simulate_power_with_adjustment,
    simulate_power_with_heterogeneous_adjustment,
    simulate_waning_effect,
)


def test_censoring_calibration():
    assert calibrate_censoring_rate(0.1, 0.2) == pytest.approx(0.01151631300276183, rel=1e-12)


def test_simulate_power():
    power, events, lam_c = simulate_power(250, 1.15, 0.2, n_sims=60, seed=3, engine="lifelines")
    assert power == pytest.approx(0.5166666666666667)
    assert events == pytest.approx(199.26666666666668)
    assert lam_c == pytest.approx(0.01154045917969199)


def test_fast_engine_matches_lifelines_decisions():
    fast = simulate_power(250, 1.15, 0.2, n_sims=60, seed=3, engine="fast")
    assert fast[0] == pytest.approx(0.5166666666666667)


def test_required_n():
    out = estimate_required_n_by_simulation(1.15, 0.8, n_sims_per_point=40, confirm_n_sims=60,
                                            seed=42, engine="lifelines", scan_below=0)
    assert out["n_closed_form"] == 503
    assert out["n_calibrated"] == 527
    assert out["a_hat"] == pytest.approx(0.19226)
    assert out["b_hat"] == pytest.approx(-3.56949)
    assert [g["n"] for g in out["grid"]] == [302, 402, 503, 604, 704]
    assert [g["power_hat"] for g in out["grid"]] == [0.425, 0.575, 0.775, 1.0, 0.925]
    assert out["confirmed_power"] == pytest.approx(0.883)
    assert out["confirmed_mean_events"] == pytest.approx(420.9)


def test_adjustment():
    out = simulate_power_with_adjustment(250, 1.1, 0.2, 2, 0.3, n_sims=25, seed=5)
    expected = {"p": 2, "r2_target": 0.3, "power": 0.12, "mean_events": 201.1, "mean_HR_hat": 1.08}
    assert {k: out[k] for k in expected} == expected  # MCSE columns are extra keys


def test_heterogeneous_adjustment():
    out = simulate_power_with_heterogeneous_adjustment(200, 1.1, 0.2, 3, 0.3, sigma_z=0.3,
                                                       n_sims=15, seed=1)
    assert out["power"] == 0.0
    assert out["mean_events"] == pytest.approx(158.3)
    assert out["mean_HR_hat"] == pytest.approx(1.083)
    assert out["median_abs_HRz_realized"] == pytest.approx(1.205)


def test_waning():
    out = simulate_waning_effect(250, 1.3, 6.0, 0.2, n_sims=20, seed=7)
    assert out["power_naive"] == pytest.approx(0.05)
    assert out["power_zph_detect"] == pytest.approx(0.2)
    assert out["mean_HR_hat"] == pytest.approx(1.054)
    assert out["mean_events"] == pytest.approx(202.9)


def test_crossing():
    out = simulate_crossing_hazards(200, 0.75, 1.4, 6.0, 0.2, n_sims=15, seed=11)
    assert out["power_naive"] == pytest.approx(0.133)
    assert out["power_zph_detect"] == pytest.approx(0.867)
    assert out["mean_HR_hat"] == pytest.approx(1.114)
    assert out["mean_events"] == pytest.approx(160.2)
