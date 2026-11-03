"""Monte Carlo SEs: the `results` helpers on hand-computed inputs, a registry
check that every simulation output carries `n_sims` and its MCSE columns,
and a calibration check that the reported power MCSE matches the actual
replicate-to-replicate spread."""

import numpy as np
import pandas as pd
import pytest

from survpower.concordance import simulate_cindex_power
from survpower.design import StudyDesign
from survpower.results import (
    exp_mean_mcse,
    mean_mcse,
    proportion_mcse,
    sd_mcse,
    summarize_replicates,
)
from survpower.sensitivity import (
    run_collinearity_study,
    run_epv_stress_test,
    run_heterogeneous_adjustment_study,
    run_omitted_covariates_study,
    run_sensitivity,
)
from survpower.simulate import (
    simulate_adjusted_vs_unadjusted,
    simulate_continuous_vs_dichotomized,
    simulate_crossing_hazards,
    simulate_epv_diagnostic,
    simulate_power,
    simulate_power_design,
    simulate_power_with_adjustment,
    simulate_power_with_heterogeneous_adjustment,
    simulate_waning_effect,
)


# --- helpers on hand-computed inputs ---------------------------------------

def test_proportion_mcse():
    assert proportion_mcse(0.75, 4) == pytest.approx(np.sqrt(0.75 * 0.25 / 4))
    assert proportion_mcse(0.0, 100) == 0.0
    assert proportion_mcse(1.0, 100) == 0.0


def test_moment_mcses():
    x = [0.0, 0.2, 0.4, np.nan]  # non-finite values are dropped: SD = 0.2, n = 3
    assert mean_mcse(x) == pytest.approx(0.2 / np.sqrt(3))
    assert sd_mcse(x) == pytest.approx(0.2 / np.sqrt(4))
    assert exp_mean_mcse(x) == pytest.approx(np.exp(0.2) * 0.2 / np.sqrt(3))
    assert np.isnan(mean_mcse([1.0]))


def test_summarize_replicates_all_columns():
    out = summarize_replicates(
        reject=[True, False, True, True],
        beta_hat=[0.0, 0.2, 0.4, np.nan],
        se_hat=[0.1, 0.1, 0.1, 0.1],
        beta_true=0.1,
        events=[10, 20, 30, 40],
    )
    expected = dict(
        power=0.75, power_mcse=np.sqrt(0.75 * 0.25 / 4), n_sims=4, n_failed=1,
        bias_beta=0.1, bias_beta_mcse=0.2 / np.sqrt(3),
        emp_se=0.2, emp_se_mcse=0.2 / np.sqrt(4),
        mean_model_se=0.1, rel_err_model_se=-0.5,
        # |beta_hat - 0.1| = 0.1, 0.1, 0.3 against 1.96 * 0.1
        coverage=2 / 3, coverage_mcse=np.sqrt(2 / 3 * 1 / 3 / 3),
        mean_events=25.0,
    )
    assert set(out) == set(expected)
    for k, v in expected.items():
        assert out[k] == pytest.approx(v), k


def test_summarize_replicates_omits_columns_without_inputs():
    out = summarize_replicates([True, False])
    assert set(out) == {"power", "power_mcse", "n_sims", "n_failed"}
    out = summarize_replicates([True, False], beta_hat=[0.1, 0.3])
    assert "emp_se" in out and "bias_beta" not in out and "coverage" not in out


# --- every simulation output reports n_sims and MCSEs ----------------------

_SIM = dict(n=120, censor_rate=0.2, n_sims=6, seed=1)

# (name, callable returning a dict or DataFrame, estimates that need an *_mcse column)
ENGINE_REGISTRY = [
    ("simulate_power", lambda: simulate_power(hr=1.3, details=True, **_SIM),
     ["power", "bias_beta", "emp_se", "coverage"]),
    ("simulate_power_design_adjusted",
     lambda: simulate_power_design(hr=1.3, p=2, r2_target=0.3, details=True, **_SIM),
     ["power", "mean_HR_hat", "bias_beta", "emp_se", "coverage"]),
    ("simulate_continuous_vs_dichotomized",
     lambda: simulate_continuous_vs_dichotomized(hr=1.3, **_SIM),
     ["power_cont", "power_dich", "group_hr_fitted"]),
    ("simulate_power_with_adjustment",
     lambda: simulate_power_with_adjustment(hr=1.3, p=2, r2_target=0.3, **_SIM),
     ["power", "mean_HR_hat", "bias_beta", "emp_se", "coverage"]),
    ("simulate_epv_diagnostic",
     lambda: simulate_epv_diagnostic(hr=1.3, p=2, r2_target=0.3, **_SIM),
     ["power", "bias_beta", "mean_HR_hat", "sd_beta_hat", "mean_se", "type1_error",
      "size_adj_power"]),
    ("simulate_power_with_heterogeneous_adjustment",
     lambda: simulate_power_with_heterogeneous_adjustment(hr=1.3, p=2, r2_target=0.3,
                                                         sigma_z=0.3, **_SIM),
     ["power", "mean_HR_hat", "bias_beta", "emp_se", "coverage"]),
    ("simulate_adjusted_vs_unadjusted",
     lambda: simulate_adjusted_vs_unadjusted(hr=1.3, p=2, **_SIM),
     ["power_adjusted", "power_unadjusted", "mean_HR_adjusted", "mean_HR_unadjusted",
      "sd_beta_adjusted", "sd_beta_unadjusted"]),
    ("simulate_waning_effect",
     lambda: simulate_waning_effect(hr_early=1.3, tau=6.0, **_SIM),
     ["power_naive", "power_zph_detect", "mean_HR_hat"]),
    ("simulate_crossing_hazards",
     lambda: simulate_crossing_hazards(hr_early=0.8, hr_late=1.3, tau=6.0, **_SIM),
     ["power_naive", "power_zph_detect", "mean_HR_hat"]),
    ("simulate_cindex_power",
     lambda: simulate_cindex_power(120, np.log(1.5), 0.01, n_sims=6, c0=0.55, seed=1),
     ["power_cox", "power_c05", "power_cc0", "mean_c", "sd_c", "mean_se_c", "mean_hr"]),
    ("run_collinearity_study",
     lambda: run_collinearity_study(hr=1.3, r2_grid=(0.3,), p_grid=(0, 2), **_SIM),
     ["power", "mean_HR_hat", "bias_beta", "emp_se", "coverage"]),
    ("run_epv_stress_test",
     lambda: run_epv_stress_test(hr=1.3, p_grid=(0, 2), **_SIM),
     ["power", "type1_error", "size_adj_power", "bias_beta", "mean_HR_hat", "sd_beta_hat",
      "mean_se"]),
    ("run_omitted_covariates_study",
     lambda: run_omitted_covariates_study(hr=1.3, p=2, sigma_z_grid=(0.0, 0.5), **_SIM),
     ["power_adjusted", "power_unadjusted", "mean_HR_adjusted", "mean_HR_unadjusted"]),
    ("run_heterogeneous_adjustment_study",
     lambda: run_heterogeneous_adjustment_study(hr=1.3, p_grid=(2,), sigma_z_grid=(0.3,),
                                                **_SIM),
     ["power", "mean_HR_hat", "bias_beta", "emp_se", "coverage"]),
    ("run_sensitivity_non_ph",
     lambda: pd.concat(run_sensitivity(StudyDesign(seed=1), n=120, hr=1.3,
                                       which=["waning", "crossing"], n_sims=6,
                                       plots=False).values()),
     ["power_naive", "power_zph_detect", "mean_HR_hat"]),
]


@pytest.mark.parametrize("name,run,estimates", ENGINE_REGISTRY,
                         ids=[e[0] for e in ENGINE_REGISTRY])
def test_every_output_reports_mcse(name, run, estimates):
    out = run()
    cols = set(out.columns if isinstance(out, pd.DataFrame) else out)
    assert "n_sims" in cols
    missing = [f"{e}_mcse" for e in estimates if f"{e}_mcse" not in cols]
    assert not missing, f"{name} lacks {missing}"


# --- the reported MCSE matches the actual spread ----------------------------

@pytest.mark.slow
def test_power_mcse_matches_replicate_spread():
    runs = [simulate_power(250, 1.15, 0.2, n_sims=200, seed=1000 + i, details=True)
            for i in range(200)]
    sd_power = np.std([r["power"] for r in runs], ddof=1)
    mean_mcse_reported = np.mean([r["power_mcse"] for r in runs])
    assert sd_power == pytest.approx(mean_mcse_reported, rel=0.15)
