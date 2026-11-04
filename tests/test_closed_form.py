import numpy as np
import pytest

from survpower.closed_form import (
    detectable_hr_continuous_cph,
    dichotomized_power,
    group_hr_from_persd_hr,
    median_split_expected_ci,
    median_split_median_survival,
    sample_size_continuous_cph,
    solve_p_con,
    theoretical_power_with_r2,
)


def test_sample_size_matches_report_section_8_2():
    # report: D = 864.0 (N = 1081) for HR 1.10; D = 401.8 (N = 503) for HR 1.15
    D, N = sample_size_continuous_cph(1.10, 0.80)
    assert D == pytest.approx(864.0, abs=0.1) and N == 1081
    D, N = sample_size_continuous_cph(1.15, 0.80)
    assert D == pytest.approx(401.8, abs=0.1) and N == 503


@pytest.mark.parametrize("sides", [1, 2])
@pytest.mark.parametrize("rho_sq", [0.0, 0.3])
def test_power_n_hr_round_trip(sides, rho_sq):
    D, N = sample_size_continuous_cph(1.2, 0.85, rho_sq=rho_sq, sides=sides)
    hr = detectable_hr_continuous_cph(D / 0.8, 0.85, rho_sq=rho_sq, sides=sides)
    assert hr == pytest.approx(1.2, rel=1e-9)
    power = theoretical_power_with_r2(D / 0.8, 1.2, 0.2, r2_target=rho_sq, sides=sides)
    assert power == pytest.approx(0.85, abs=1e-9)


def test_adjustment_inflates_events():
    D0, _ = sample_size_continuous_cph(1.2, 0.8)
    D3, _ = sample_size_continuous_cph(1.2, 0.8, rho_sq=0.3)
    assert D3 == pytest.approx(D0 / 0.7)


def test_one_sided_needs_fewer_events():
    assert sample_size_continuous_cph(1.2, 0.8, sides=1)[0] < sample_size_continuous_cph(1.2, 0.8)[0]


def test_dichotomized_matches_report_table_1():
    assert group_hr_from_persd_hr(1.10) == pytest.approx(1.164, abs=1e-3)
    assert dichotomized_power(1200, 1.10, 0.2) == pytest.approx(0.652, abs=1e-3)
    assert dichotomized_power(250, 1.15, 0.2) == pytest.approx(0.348, abs=1e-3)


def test_solve_p_con_pooled_probability():
    ghr = 1.25
    p_con = solve_p_con(ghr, 0.8)
    p_exp = 1 - (1 - p_con) ** ghr
    assert 0.5 * (p_con + p_exp) == pytest.approx(0.8)
    assert p_exp > p_con


def test_median_split_median_survival():
    lam0 = np.log(2) / 15
    assert median_split_median_survival(1.0, lam0) == pytest.approx((15.0, 15.0), rel=1e-8)
    low, high = median_split_median_survival(1.3, lam0)
    assert high < 15 < low
    # check S_high(median) = 1/2 by Monte Carlo
    x = np.random.default_rng(0).standard_normal(2_000_000)
    x = x[x > 0]
    assert np.mean(np.exp(-lam0 * 1.3 ** x * high)) == pytest.approx(0.5, abs=1e-3)


def test_median_split_expected_ci():
    lo, hi, d_low, d_high = median_split_expected_ci(200, 1.3, 0.2)
    ghr = group_hr_from_persd_hr(1.3)
    assert d_low + d_high == pytest.approx(160)
    assert np.sqrt(lo * hi) == pytest.approx(ghr)  # symmetric on the log scale
    assert np.log(hi / lo) == pytest.approx(2 * 1.96 * np.sqrt(1 / d_low + 1 / d_high), rel=1e-3)
