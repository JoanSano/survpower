import numpy as np
import pytest

from survpower.censoring import censor_rate_from_design, event_probability_exponential
from survpower.closed_form import detectable_hr_continuous_cph
from survpower.design import StudyDesign
from survpower.planning import detectable_censor_rate


@pytest.mark.parametrize("hr", [1.0, 1.5])
@pytest.mark.parametrize("dropout", [0.0, 0.05, 0.2])
@pytest.mark.parametrize("F", [6, 12, 24])
@pytest.mark.parametrize("A", [0, 12, 36])
def test_closed_form_matches_monte_carlo(A, F, dropout, hr):
    n_probe = 500_000
    p = event_probability_exponential(A, F, dropout, hr_per_sd=hr)
    p_mc = 1 - censor_rate_from_design(hr, A, F, dropout, n_probe=n_probe)
    mcse = np.sqrt(p * (1 - p) / n_probe)
    assert abs(p - p_mc) < 4 * mcse


@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_zero_accrual_is_the_limit(dropout):
    lam = np.log(2) / 15
    r = lam - np.log1p(-dropout) / 12
    limit = lam / r * (1 - np.exp(-r * 12))
    assert event_probability_exponential(0, 12, dropout) == pytest.approx(limit, abs=1e-10)
    assert event_probability_exponential(1e-9, 12, dropout) == pytest.approx(limit, abs=1e-10)


def test_docs_example():
    # docs/concepts/censoring-and-accrual.md: median OS 15, A = 36, F = 12, 5%/yr dropout
    censored = 1 - event_probability_exponential(36, 12, 0.05)
    assert censored == pytest.approx(0.315, abs=0.001)


def test_longer_followup_beats_longer_accrual():
    # at fixed A + F, moving months from accrual to follow-up lowers censoring
    p = [event_probability_exponential(A, 48 - A, 0.05) for A in (0, 12, 24, 36, 48)]
    assert np.all(np.diff(p) < 0)


# Lachin & Foulkes (1986), Biometrics 42, 507-519. Their R (accrual) and T (total
# study time) are in years with hazards per year; here A = 12 R and F = 12 (T - R)
# months, median_os = 12 ln 2 / lambda, and a loss hazard eta per year is an annual
# loss fraction 1 - exp(-eta). With hr_per_sd = 1 there is no spread in X, so the
# function returns their single-hazard E(delta | lambda, eta), eq. (4.1).
def _lf_event_probability(lam, eta, R=3, T=5):
    return event_probability_exponential(
        12 * R, 12 * (T - R), dropout_annual=-np.expm1(-eta), median_os=12 * np.log(2) / lam,
    )


def test_docs_example_is_lachin_foulkes_eq_4_1():
    # the docs example in their units: R = 3, T = 4 years, lambda = 12 ln 2 / 15 and
    # eta = -ln(0.95) per year; eq. (4.1) gives E(delta), p. 511 gives E(xi) = (eta / lambda) E(delta)
    lam, eta, R, T = 12 * np.log(2) / 15, -np.log(0.95), 3, 4
    r = lam + eta
    death = lam / r * (1 - (np.exp(-(T - R) * r) - np.exp(-T * r)) / (R * r))
    assert event_probability_exponential(36, 12, 0.05) == pytest.approx(death, abs=1e-12)
    assert death == pytest.approx(0.685, abs=5e-4)
    assert eta / lam * death == pytest.approx(0.063, abs=5e-4)  # lost to follow-up


def test_lachin_foulkes_section_2_example():
    # Section 2 (p. 509): R = 3, T = 5, E(delta | .20) = .4959 and E(delta | .30) = .6381
    assert _lf_event_probability(0.20, 0.0) == pytest.approx(0.4959, abs=5e-5)
    assert _lf_event_probability(0.30, 0.0) == pytest.approx(0.6381, abs=5e-5)


# Table 2 (p. 512): eta, E(delta | .20), E(delta | .30), E(xi | .20, eta), E(xi | .30, eta)
LF_TABLE_2 = [
    (0.000, 0.496, 0.638, 0.000, 0.000),
    (0.025, 0.477, 0.615, 0.060, 0.051),
    (0.050, 0.459, 0.594, 0.115, 0.099),
    (0.075, 0.442, 0.573, 0.166, 0.143),
    (0.100, 0.425, 0.554, 0.213, 0.185),
    (0.125, 0.410, 0.535, 0.256, 0.223),
    (0.150, 0.396, 0.518, 0.297, 0.259),
    (0.175, 0.382, 0.501, 0.334, 0.292),
    (0.200, 0.369, 0.486, 0.369, 0.324),
]


@pytest.mark.parametrize("eta, death_e, death_c, loss_e, loss_c", LF_TABLE_2)
def test_lachin_foulkes_table_2(eta, death_e, death_c, loss_e, loss_c):
    for lam, death, loss in [(0.20, death_e, loss_e), (0.30, death_c, loss_c)]:
        p = _lf_event_probability(lam, eta)
        assert p == pytest.approx(death, abs=5e-4)
        # their E(xi | lambda, eta) = (eta / lambda) E(delta | lambda, eta)
        assert eta / lam * p == pytest.approx(loss, abs=5e-4)


@pytest.mark.parametrize("A, F, dropout, median_os, n", [
    (36, 12, 0.05, 15, 100), (36, 12, 0.05, 15, 300), (12, 0.5, 0.2, 60, 100),
])
def test_detectable_censor_rate_is_a_fixed_point(A, F, dropout, median_os, n):
    # the censoring used for the detectable HR is the censoring at that HR
    design = StudyDesign(accrual_months=A, followup_months=F, dropout_annual=dropout,
                         median_os=median_os)
    cr = detectable_censor_rate(design, n, 0.8)
    hr = detectable_hr_continuous_cph(n, 0.8, event_prob=1 - cr)
    assert design.censor_rate_for(hr) == pytest.approx(cr, abs=2e-3)


def test_detectable_censor_rate_fixed_rate_ignores_hr():
    assert detectable_censor_rate(StudyDesign(censor_rate=0.3), 100, 0.8) == 0.3
