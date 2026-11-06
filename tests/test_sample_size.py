"""The unified `sample_size()` entry point (#11)."""

import numpy as np
import pytest

from survpower import SampleSizeResult, StudyDesign, hr_of_c, sample_size
from survpower.closed_form import sample_size_continuous_cph
from survpower.simulate import simulate_power


@pytest.mark.parametrize("hr, events, n", [(1.10, 864.0, 1081), (1.15, 401.8, 503)])
def test_closed_form_reference_values(hr, events, n):
    r = sample_size(hr=hr, method="closed_form")
    assert isinstance(r, SampleSizeResult)
    assert (r.events, r.n_evaluable, r.n_enrolled, r.method) == (events, n, n, "closed_form")
    assert r.confirmed_power is None and r.n_ci is None


@pytest.mark.parametrize("c", [0.55, 0.60, 0.70])
def test_c_index_equals_its_hazard_ratio_exactly(c):
    assert sample_size(c_index=c, method="closed_form") == sample_size(hr=hr_of_c(c),
                                                                       method="closed_form")


def test_c_index_070_closed_form():
    r = sample_size(c_index=0.70, method="closed_form")
    assert (r.events, r.n_evaluable) == (11.0, 14)
    assert r.c_index == pytest.approx(0.70, abs=1e-9)


# Hsieh & Lavori (2000), multiple-myeloma example (log BUN, SD 0.3126, log-HR 1 per unit,
# one-sided alpha = 0.05, power 0.80), as printed in the Stata 13 manual,
# [ST] stpower cox, Examples 2-4: E/N = 64/64; R^2 = 0.1837 -> 78/78;
# P(event) = 0.738 -> 64/86; both -> 78/106.
@pytest.mark.parametrize("r2, event_prob, events, n", [
    (0.0, 1.0, 64, 64), (0.1837, 1.0, 78, 78), (0.0, 0.738, 64, 86), (0.1837, 0.738, 78, 106)])
def test_hsieh_lavori_2000_myeloma_example(r2, event_prob, events, n):
    D, N = sample_size_continuous_cph(np.e, 0.80, sigma_x=0.3126, alpha=0.05,
                                      event_prob=event_prob, rho_sq=r2, sides=1)
    assert (int(np.ceil(D)), N) == (events, n)
    assert sample_size_continuous_cph(np.e, 0.80, var_x=0.3126 ** 2, alpha=0.05,
                                      event_prob=event_prob, rho_sq=r2, sides=1) == (D, N)


def test_hsieh_lavori_example_through_the_design():
    design = StudyDesign(sides=1, censor_rate=1 - 0.738, biomarker_sd=0.3126,
                         n_covariates=8, r2=0.1837)
    r = sample_size(hr=np.e, design=design, method="closed_form")  # HR per unit of log BUN
    assert (int(np.ceil(r.events)), r.n_evaluable) == (78, 106)


def test_var_x_and_sigma_x_are_exclusive():
    with pytest.raises(ValueError, match="not both"):
        sample_size_continuous_cph(1.2, 0.8, sigma_x=2.0, var_x=4.0)


@pytest.mark.parametrize("kw", [dict(), dict(hr=1.2, c_index=0.6)])
def test_exactly_one_effect(kw):
    with pytest.raises(ValueError, match="exactly one"):
        sample_size(**kw)


def test_unknown_and_unavailable_methods():
    with pytest.raises(ValueError, match="method"):
        sample_size(hr=1.2, method="bogus")
    with pytest.raises(NotImplementedError, match="#24"):
        sample_size(hr=1.2, method="robbins_monro")


def test_calibrated_has_interval_and_mcse():
    d = StudyDesign(n_sims=300, n_sims_grid=200)
    r = sample_size(hr=1.2, design=d, method="calibrated", seed=3)
    assert r.method == "probit" and r.n_ci is not None
    assert r.n_ci[0] <= r.n_evaluable <= r.n_ci[1]
    assert r.confirmed_power.mcse == pytest.approx(
        np.sqrt(r.confirmed_power.value * (1 - r.confirmed_power.value) / 300), abs=1e-4)
    assert r.details["grid"]


@pytest.mark.slow
def test_closed_form_fails_at_large_effects():
    """C = 0.70: the closed form gives N = 14, where simulated power is well
    below 0.70; "auto" finds N of 20-24."""
    n0 = sample_size(c_index=0.70, method="closed_form").n_evaluable
    assert n0 == 14
    assert simulate_power(n0, hr_of_c(0.70), 0.20, n_sims=4000, seed=1)[0] < 0.70
    r = sample_size(c_index=0.70, method="auto")
    assert 20 <= r.n_evaluable <= 24
    assert r.method == "scan" and r.confirmed_power.value >= 0.80


def test_run_sample_size_mode_is_built_on_sample_size():
    from survpower.planning import run_sample_size_mode
    d = StudyDesign(n_sims=200, n_sims_grid=150, non_evaluable=0.1)
    df, details = run_sample_size_mode(d, hrs=[1.3], powers=[0.8])
    r = sample_size(hr=1.3, power=0.8, design=d, seed=d.seed)
    row = df.iloc[0]
    assert (row["n_calibrated_evaluable"], row["n_enrolled"]) == (r.n_evaluable, r.n_enrolled)
    assert row["confirmed_power"] == r.confirmed_power.value
    assert row["summary"].startswith(r.summary)
    assert details[0] == r.details
