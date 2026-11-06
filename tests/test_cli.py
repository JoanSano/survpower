"""End-to-end smoke tests: every planning mode through the CLI, with few
replicates. They check plumbing and outputs, not Monte Carlo accuracy."""

import importlib
import os

import pandas as pd
import pytest

import survpower
from survpower.cli import main

FEW = ["--n-sims", "60", "--no-plots"]
GRID = ["--n-sims-grid", "40"]
SOLVES = {"sample-size", "detectable-effect", "benchmark"}


def run(tmp_path, command, *args):
    grid = GRID if command in SOLVES and "--n" not in args else []
    main([command, *args, *FEW, *grid, "--outdir", str(tmp_path)])
    return tmp_path


def test_power_mode(tmp_path):
    out = run(tmp_path, "power", "--hr", "1.25", "--n", "200", "400")
    df = pd.read_csv(out / "power_analysis.csv")
    assert list(df["n_enrolled"]) == [200, 400]
    assert df["power_sim"].between(0, 1).all()
    assert df["summary"].str.contains("power").all()
    assert df["summary"].str.contains("A median-split analysis of the same cohorts").all()
    assert (df["split_hr_ci_low"] < df["split_hr"]).all() and (df["split_hr"] < df["split_hr_ci_high"]).all()
    assert (df["split_events_high"] + df["split_events_low"]).round(0).equals(df["mean_events"].round(0))
    assert (df["split_median_os_high"] < df["split_median_os_low"]).all()  # HR > 1: high = worse


def test_median_split_in_sample_size_and_detectable_modes(tmp_path):
    ss = pd.read_csv(run(tmp_path / "ss", "sample-size", "--hr", "1.3") / "sample_size.csv").iloc[0]
    assert "at that N it would have" in ss["summary"] and ss["split_source"] == "simulated"
    de = pd.read_csv(run(tmp_path / "de", "detectable-effect", "--n", "200")
                     / "detectable_effect.csv").iloc[0]
    assert "At this N and HR, a median-split analysis" in de["summary"]
    assert de["split_power"] < de["confirmed_power"] + 0.1  # the split loses power


def test_power_mode_median_split_closed_form_and_adjusted(tmp_path):
    df = pd.read_csv(run(tmp_path / "ll", "power", "--hr", "1.25", "--n", "200",
                         "--engine", "lifelines") / "power_analysis.csv")
    assert "Schoenfeld closed form)" in df["summary"].iloc[0]
    assert df["split_source"].iloc[0] == "closed form"
    df = pd.read_csv(run(tmp_path / "adj", "power", "--hr", "1.25", "--n", "200",
                         "--n-covariates", "2", "--r2", "0.1") / "power_analysis.csv")
    assert "median-split" not in df["summary"].iloc[0]


def test_sample_size_mode_with_design(tmp_path):
    out = run(tmp_path, "sample-size", "--hr", "1.3", "--accrual-months", "24",
              "--followup-months", "12", "--dropout", "0.05", "--non-evaluable", "0.1",
              "--sides", "1")
    df = pd.read_csv(out / "sample_size.csv")
    row = df.iloc[0]
    assert row["n_enrolled"] >= row["n_calibrated_evaluable"] / 0.9 - 1


def test_detectable_effect_mode_adjusted(tmp_path):
    out = run(tmp_path, "detectable-effect", "--n", "300", "--n-covariates", "2", "--r2", "0.2")
    df = pd.read_csv(out / "detectable_effect.csv")
    assert df["hr_detectable_per_sd"].iloc[0] > 1
    assert 0.5 < df["c_detectable"].iloc[0] < 1


def test_cindex_association_sample_size(tmp_path):
    out = run(tmp_path, "sample-size", "--c-index", "0.65")
    df = pd.read_csv(out / "sample_size.csv")
    assert df["c_index"].iloc[0] == pytest.approx(0.65, abs=1e-3)


def test_cindex_benchmark_power_uno(tmp_path):
    out = run(tmp_path, "benchmark", "--c-index", "0.70", "--c0", "0.65", "--n", "200",
              "--c-estimator", "uno")
    df = pd.read_csv(out / "cindex_benchmark.csv")
    assert df["estimator"].iloc[0] == "uno" and "tau_months" in df
    assert f"HR {df['hr_per_sd'].iloc[0]:.3g} per SD" in df["summary"].iloc[0]


def test_cindex_precision(tmp_path):
    out = run(tmp_path, "precision", "--c-index", "0.70", "--half-width", "0.05", "0.08",
              "--biomarker-sd", "2")
    df = pd.read_csv(out / "cindex_precision.csv")
    assert df["n_evaluable"].iloc[0] > df["n_evaluable"].iloc[1]
    hr = df["hr_per_sd"].iloc[0]
    assert f"HR {hr ** 0.5:.3g} per biomarker unit or {hr:.3g} per SD" in df["summary"].iloc[0]


def test_precision_at_n_returns_the_half_width_without_a_dummy_flag(tmp_path):
    out = run(tmp_path, "precision", "--c-index", "0.7", "--n", "200")
    row = pd.read_csv(out / "cindex_precision.csv").iloc[0]
    assert row["n_enrolled"] == 200 and 0 < row["half_width_closed_form"] < 0.5
    assert "target_half_width" not in row


def test_sensitivity_outputs(tmp_path):
    out = run(tmp_path, "power", "--hr", "1.3", "--n", "150", "--sensitivity", "waning",
              "--n-sims-sensitivity", "15")
    assert os.path.exists(out / "sensitivity_waning_n150_hr1.3.csv")


@pytest.mark.parametrize("args", [
    ["power", "--hr", "1.2", "--c-index", "0.6", "--n", "100"],
    ["sample-size"],
    ["detectable-effect"],
    ["benchmark", "--c-index", "0.7"],
    ["benchmark", "--c0", "0.6"],
    ["benchmark", "--c-index", "0.6", "--c0", "0.65"],
    ["precision", "--c-index", "0.7"],
    ["precision", "--c-index", "0.7", "--n", "200", "--half-width", "0.05"],
    ["precision", "--c-index", "0.7", "--half-width", "0.6"],
    ["detectable-effect", "--n", "100", "--r2", "0.3"],
    ["sample-size", "--hr", "1.2", "--censor-rate", "0.3", "--accrual-months", "24",
     "--followup-months", "6"],
    ["sample-size", "--hr", "1"],
    ["power", "--hr", "1.2", "--n", "5"],
    ["nonsense"],
])
def test_invalid_inputs_exit(args):
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2


@pytest.mark.parametrize("args", [
    ["power", "--hr", "1.2"],  # a missing flag no longer changes the question
    ["sample-size", "--hr", "1.2", "--n", "100"],
    ["benchmark", "--c-index", "0.7", "--c0", "0.6", "--n", "200", "--power", "0.8"],
])
def test_each_command_answers_one_question(args):
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2


@pytest.mark.parametrize("args, flag", [
    (["benchmark", "--c-index", "0.7", "--c0", "0.6", "--sides", "1"], "--sides"),
    (["power", "--hr", "1.2", "--n", "100", "--c-estimator", "uno"], "--c-estimator"),
    (["precision", "--c-index", "0.7", "--n", "100", "--sensitivity", "waning"], "--sensitivity"),
    (["power", "--hr", "1.2", "--n", "100", "--power", "0.8"], "--power"),
    (["precision", "--c-index", "0.7", "--n", "100", "--alpha", "0.01"], "--alpha"),
    (["power", "--hr", "1.2", "--n", "100", "--n-sims-grid", "5"], "--n-sims-grid"),
    (["precision", "--c-index", "0.7", "--n", "100", "--c-tau", "24"], "--c-tau"),
    (["benchmark", "--c-index", "0.7", "--c0", "0.6", "--n", "100", "--n-sims-grid", "5"],
     "--n-sims-grid"),
    (["power", "--hr", "1.2", "--n", "100", "--tau", "4"], "--tau"),
    (["power", "--hr", "1.2", "--n", "100", "--sensitivity", "crossing", "--waning-hr", "1.2"],
     "--waning-hr"),
    (["power", "--hr", "1.2", "--n", "100", "--sensitivity", "waning", "--crossing-hr", "1", "2"],
     "--crossing-hr"),
    (["power", "--hr", "1.2", "--n", "100", "--n-sims-sensitivity", "5"], "--n-sims-sensitivity"),
])
def test_nothing_accepted_and_ignored(args, flag, capsys):
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    assert flag in capsys.readouterr().err


@pytest.mark.parametrize("command", ["benchmark", "precision"])
def test_concordance_questions_explain_they_are_univariate(command, capsys):
    target = ["--c0", "0.6"] if command == "benchmark" else ["--n", "100"]
    with pytest.raises(SystemExit) as exc:
        main([command, "--c-index", "0.7", *target, "--n-covariates", "2"])
    assert exc.value.code == 2
    assert "power the adjusted HR" in capsys.readouterr().err


def test_flags_cannot_be_abbreviated(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["sample-size", "--hr", "1.2", "--n-sim", "10"])
    assert exc.value.code == 2


def test_no_command_prints_help_and_exits_2(capsys):
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
    assert "sample-size" in capsys.readouterr().out


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"survpower {survpower.__version__}"


@pytest.mark.parametrize("command", ["power", "sample-size", "detectable-effect", "benchmark",
                                     "precision"])
def test_every_command_has_help_with_examples(command, capsys):
    with pytest.raises(SystemExit) as exc:
        main([command, "--help"])
    assert exc.value.code == 0
    assert f"survpower {command} --" in capsys.readouterr().out


def test_quick_only_sets_replicates_the_command_has(tmp_path):
    main(["precision", "--c-index", "0.7", "--n", "100", "--quick", "--no-plots",
          "--outdir", str(tmp_path)])
    assert pd.read_csv(tmp_path / "cindex_precision.csv")["n_sims"].iloc[0] == 200


def test_pinned_results_at_a_fixed_seed(tmp_path):
    """A small guard on the numbers: seeds and the per-combination offset stay as they are."""
    df = pd.read_csv(run(tmp_path, "power", "--hr", "1.25", "1.4", "--n", "150", "--seed", "11")
                     / "power_analysis.csv")
    assert df["power_sim"].tolist() == [0.75, 0.95]
    assert df["mean_events"].tolist() == [120.8, 120.3]


# --- The flag-only syntax, removed in 0.4.1 (#66) --------------------------------------

LEGACY = [
    (["--c-index", "0.7", "--c0", "0.65", "--n", "100"],
     ["benchmark", "--c-index", "0.7", "--c0", "0.65", "--n", "100"], "cindex_benchmark"),
    (["--c-index", "0.7", "--c0", "0.65"],
     ["benchmark", "--c-index", "0.7", "--c0", "0.65"], "cindex_benchmark"),
    (["--c-index", "0.7", "--precision", "0.05"],
     ["precision", "--c-index", "0.7", "--half-width", "0.05"], "cindex_precision"),
    (["--c-index", "0.7", "--precision", "0.05", "--n", "100"],
     ["precision", "--c-index", "0.7", "--n", "100"], "cindex_precision"),
    (["--hr", "1.5", "--n", "100"], ["power", "--hr", "1.5", "--n", "100"], "power_analysis"),
    (["--c-index", "0.6", "--n", "100"], ["power", "--c-index", "0.6", "--n", "100"], "power_analysis"),
    (["--hr", "1.5"], ["sample-size", "--hr", "1.5"], "sample_size"),
    (["--c-index", "0.65"], ["sample-size", "--c-index", "0.65"], "sample_size"),
    (["--n", "100"], ["detectable-effect", "--n", "100"], "detectable_effect"),
]


@pytest.mark.parametrize("old, new, csv", LEGACY)
def test_legacy_syntax_gives_the_same_csv_and_warns(tmp_path, old, new, csv, capsys):
    grid = GRID if new[0] in SOLVES and "--n" not in new else []
    main([*old, *FEW, *grid, "--outdir", str(tmp_path / "old")])
    err = capsys.readouterr().err
    assert "deprecated" in err and f"survpower {' '.join(new)}" in err
    main([*new, *FEW, *grid, "--outdir", str(tmp_path / "new")])
    assert capsys.readouterr().err == ""
    assert (tmp_path / "old" / f"{csv}.csv").read_bytes() == (tmp_path / "new" / f"{csv}.csv").read_bytes()


@pytest.mark.parametrize("old, dropped, csv", [
    (["--hr", "1.5", "--n", "100", "--power", "0.9"], "--power", "power_analysis"),
    (["--c-index", "0.7", "--c0", "0.65", "--n", "100", "--power", "0.9"], "--power", "cindex_benchmark"),
    (["--c-index", "0.7", "--c0", "0.65", "--n", "100", "--sides", "1"], "--sides", "cindex_benchmark"),
    (["--c-index", "0.7", "--precision", "0.05", "--sensitivity", "waning"], "--sensitivity", "cindex_precision"),
    (["--hr", "1.5", "--n", "100", "--c-estimator", "uno"], "--c-estimator", "power_analysis"),
    (["--hr", "1.5", "--n", "100", "--tau", "4"], "--tau", "power_analysis"),
    (["--c-index", "0.7", "--precision", "0.05", "--c-tau", "24"], "--c-tau", "cindex_precision"),
])
def test_legacy_drops_what_was_ignored_and_says_so(tmp_path, old, dropped, csv, capsys):
    main([*old, *FEW, "--outdir", str(tmp_path)])
    err = capsys.readouterr().err
    assert "dropped" in err and dropped in err.split("dropped")[1]
    assert (tmp_path / f"{csv}.csv").exists()


@pytest.mark.parametrize("old", [
    ["--hr", "1.2", "--c-index", "0.6"],
    ["--hr", "1.2", "--c0", "0.6"],
    ["--hr", "1.2", "--precision", "0.05"],
    ["--c0", "0.6"],
    ["--precision", "0.05"],
    ["--c-index", "0.6", "--c0", "0.65"],
    ["--c-index", "0.7", "--c0", "0.6", "--precision", "0.05"],
    ["--c-index", "0.7", "--c0", "0.6", "--n-covariates", "2"],
    ["--n", "100", "--r2", "0.3"],
    ["--alpha", "0.01"],
])
def test_legacy_still_rejects_what_it_rejected(old):
    with pytest.raises(SystemExit) as exc:
        main(old)
    assert exc.value.code == 2


def test_legacy_accepts_abbreviations_as_it_did(tmp_path, capsys):
    main(["--hr", "1.5", "--n", "100", "--no-pl", "--n-sims", "20", "--outdir", str(tmp_path)])
    assert "survpower power --hr 1.5 --n 100" in capsys.readouterr().err


@pytest.mark.parametrize("args", [["report"], ["report", "--appendix-b", "--outdir", "x"]])
def test_report_command_explains_its_removal(args, capsys):
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "removed in 0.4.0" in err and "v0.2.0" in err and "Tutorials" in err


def test_report_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("survpower.report")
