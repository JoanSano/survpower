"""End-to-end smoke tests: every planning mode through the CLI, with few
replicates. They check plumbing and outputs, not Monte Carlo accuracy."""

import os

import pandas as pd
import pytest

from survpower.cli import main

FEW = ["--n-sims", "60", "--n-sims-grid", "40", "--no-plots"]


def run(tmp_path, *args):
    main([*args, *FEW, "--outdir", str(tmp_path)])
    return tmp_path


def test_power_mode(tmp_path):
    out = run(tmp_path, "--hr", "1.25", "--n", "200", "400")
    df = pd.read_csv(out / "power_analysis.csv")
    assert list(df["n_enrolled"]) == [200, 400]
    assert df["power_sim"].between(0, 1).all()
    assert df["summary"].str.contains("power").all()


def test_sample_size_mode_with_design(tmp_path):
    out = run(tmp_path, "--hr", "1.3", "--accrual-months", "24", "--followup-months", "12",
              "--dropout", "0.05", "--non-evaluable", "0.1", "--sides", "1")
    df = pd.read_csv(out / "sample_size.csv")
    row = df.iloc[0]
    assert row["n_enrolled"] >= row["n_calibrated_evaluable"] / 0.9 - 1


def test_detectable_effect_mode_adjusted(tmp_path):
    out = run(tmp_path, "--n", "300", "--n-covariates", "2", "--r2", "0.2")
    df = pd.read_csv(out / "detectable_effect.csv")
    assert df["hr_detectable_per_sd"].iloc[0] > 1
    assert 0.5 < df["c_detectable"].iloc[0] < 1


def test_cindex_association_sample_size(tmp_path):
    out = run(tmp_path, "--c-index", "0.65")
    df = pd.read_csv(out / "sample_size.csv")
    assert df["c_index"].iloc[0] == pytest.approx(0.65, abs=1e-3)


def test_cindex_benchmark_power_uno(tmp_path):
    out = run(tmp_path, "--c-index", "0.70", "--c0", "0.65", "--n", "200",
              "--c-estimator", "uno")
    df = pd.read_csv(out / "cindex_benchmark.csv")
    assert df["estimator"].iloc[0] == "uno" and "tau_months" in df


def test_cindex_precision(tmp_path):
    out = run(tmp_path, "--c-index", "0.70", "--precision", "0.05", "0.08")
    df = pd.read_csv(out / "cindex_precision.csv")
    assert df["n_evaluable"].iloc[0] > df["n_evaluable"].iloc[1]


def test_sensitivity_outputs(tmp_path):
    out = run(tmp_path, "--hr", "1.3", "--n", "150", "--sensitivity", "waning",
              "--n-sims-sensitivity", "15")
    assert os.path.exists(out / "sensitivity_waning_n150_hr1.3.csv")


@pytest.mark.parametrize("args", [
    [],
    ["--hr", "1.2", "--c-index", "0.6"],
    ["--hr", "1.2", "--c0", "0.6"],
    ["--c-index", "0.6", "--c0", "0.65"],
    ["--n", "100", "--r2", "0.3"],
    ["--c-index", "0.7", "--c0", "0.6", "--n-covariates", "2"],
    ["--hr", "1.2", "--censor-rate", "0.3", "--accrual-months", "24", "--followup-months", "6"],
])
def test_invalid_inputs_exit(args):
    with pytest.raises(SystemExit):
        main(args)


def test_planning_tables_report_mcse(tmp_path):
    ss = pd.read_csv(run(tmp_path / "ss", "--hr", "1.3") / "sample_size.csv")
    de = pd.read_csv(run(tmp_path / "de", "--n", "300") / "detectable_effect.csv")
    for df in (ss, de):
        assert {"confirmed_power_mcse", "n_sims"} <= set(df.columns)
        assert (df["n_sims"] == 60).all()


def test_protective_interval_is_reciprocal(tmp_path):
    out = run(tmp_path, "--n", "300", "--biomarker-sd", "2")
    row = pd.read_csv(out / "detectable_effect.csv").iloc[0]
    assert row["hr_protective_ci_low"] == pytest.approx(1 / row["hr_ci_high"], abs=1e-4)
    assert row["hr_protective_ci_high"] == pytest.approx(1 / row["hr_ci_low"], abs=1e-4)
    assert (row["hr_protective_ci_low"] <= row["hr_detectable_protective_per_sd"]
            <= row["hr_protective_ci_high"])
    assert (row["hr_protective_per_unit_ci_low"] <= row["hr_detectable_protective_per_unit"]
            <= row["hr_protective_per_unit_ci_high"])
    assert "Monte Carlo intervals" in row["summary"]
