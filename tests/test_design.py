"""`StudyDesign` rejects invalid designs at construction (#50)."""

import dataclasses
import itertools
import math

import pytest

from survpower import StudyDesign, sample_size
from survpower.cli import DESIGN_FLAGS, main

ACCRUAL = dict(accrual_months=12, followup_months=12)
NAN, INF = math.nan, math.inf


@pytest.mark.parametrize("kwargs, field", [
    (dict(alpha=0), "alpha"), (dict(alpha=0.5), "alpha"), (dict(alpha=2), "alpha"),
    (dict(alpha=NAN), "alpha"),
    (dict(sides=0), "sides"), (dict(sides=3), "sides"),
    (dict(median_os=0), "median_os"), (dict(median_os=-1), "median_os"),
    (dict(median_os=INF), "median_os"), (dict(median_os=NAN), "median_os"),
    (dict(censor_rate=-0.1), "censor_rate"), (dict(censor_rate=1), "censor_rate"),
    (dict(censor_rate=1.2), "censor_rate"), (dict(censor_rate=NAN), "censor_rate"),
    (dict(censor_rate=0.2, **ACCRUAL), "censor_rate"),
    (dict(accrual_months=12), "followup_months"), (dict(followup_months=12), "accrual_months"),
    (dict(accrual_months=-1, followup_months=12), "accrual_months"),
    (dict(accrual_months=12, followup_months=-1), "followup_months"),
    (dict(accrual_months=INF, followup_months=12), "accrual_months"),
    (dict(accrual_months=0, followup_months=0), "followup_months"),
    (dict(dropout_annual=0.1), "dropout_annual"),
    (dict(dropout_annual=-0.1, **ACCRUAL), "dropout_annual"),
    (dict(dropout_annual=1, **ACCRUAL), "dropout_annual"),
    (dict(non_evaluable=-0.1), "non_evaluable"), (dict(non_evaluable=1), "non_evaluable"),
    (dict(r2=-0.1, n_covariates=2), "r2"), (dict(r2=1, n_covariates=2), "r2"),
    (dict(r2=1.5, n_covariates=2), "r2"), (dict(r2=0.3), "r2"),
    (dict(n_covariates=-1), "n_covariates"), (dict(n_covariates=1.5), "n_covariates"),
    (dict(biomarker_sd=0), "biomarker_sd"), (dict(biomarker_sd=-1), "biomarker_sd"),
    (dict(biomarker_sd=INF), "biomarker_sd"),
    (dict(c_estimator="x"), "c_estimator"),
    (dict(c_tau=0), "c_tau"), (dict(c_tau=-1), "c_tau"),
    (dict(engine="x"), "engine"),
    (dict(n_sims=0), "n_sims"), (dict(n_sims=1.5), "n_sims"),
    (dict(n_sims_grid=0), "n_sims_grid"),
])
def test_invalid_design_raises_naming_the_field(kwargs, field):
    with pytest.raises(ValueError, match=rf"\b{field}\b"):
        StudyDesign(**kwargs)


@pytest.mark.parametrize("kwargs", [
    dict(),
    dict(alpha=0.001), dict(alpha=0.499), dict(sides=1),
    dict(censor_rate=0), dict(censor_rate=0.999),
    dict(accrual_months=0, followup_months=12), dict(accrual_months=12, followup_months=0),
    dict(dropout_annual=0.999, **ACCRUAL),
    dict(non_evaluable=0), dict(non_evaluable=0.999),
    dict(r2=0.999, n_covariates=1), dict(r2=0, n_covariates=0),
    dict(biomarker_sd=0.001), dict(c_tau=0.001), dict(c_estimator="gonen-heller"),
    dict(engine="lifelines"), dict(n_sims=1, n_sims_grid=1),
])
def test_boundary_values_are_valid(kwargs):
    StudyDesign(**kwargs)


def test_replace_validates():
    with pytest.raises(ValueError, match="alpha"):
        dataclasses.replace(StudyDesign(), alpha=2)


def test_design_is_frozen():
    design = StudyDesign()
    with pytest.raises(dataclasses.FrozenInstanceError):
        design.alpha = 2
    assert dataclasses.replace(design, n_sims=5000).n_sims == 5000
    assert design.n_sims == 1000


def test_design_is_hashable_by_value():
    assert hash(StudyDesign(seed=1)) == hash(StudyDesign(seed=1))
    assert len({StudyDesign(seed=1), StudyDesign(seed=1), StudyDesign(seed=2)}) == 2


@pytest.mark.parametrize("censoring, adjustment, scale, non_evaluable", itertools.product(
    [dict(), dict(censor_rate=0), dict(censor_rate=0.999),
     dict(accrual_months=0, followup_months=1), dict(dropout_annual=0.5, **ACCRUAL)],
    [dict(), dict(n_covariates=1, r2=0.999)],
    [dict(), dict(biomarker_sd=0.01), dict(biomarker_sd=100)],
    [0, 0.9],
))
def test_closed_form_n_is_at_least_one(censoring, adjustment, scale, non_evaluable):
    design = StudyDesign(**censoring, **adjustment, **scale, non_evaluable=non_evaluable)
    r = sample_size(hr=1.3, design=design, method="closed_form")
    assert r.n_evaluable >= 1 and r.n_enrolled >= r.n_evaluable


def test_every_field_has_a_cli_flag():
    assert set(DESIGN_FLAGS) == {f.name for f in dataclasses.fields(StudyDesign)}


@pytest.mark.parametrize("args, message", [
    (["--alpha", "2"], "--alpha must be in (0, 0.5)"),
    (["--alpha", "0.6", "--power", "0.55"], "--alpha must be in (0, 0.5)"),
    (["--censor-rate", "1.2"], "--censor-rate must be in [0, 1)"),
    (["--accrual-months", "12"], "--accrual-months and --followup-months must be given together"),
    (["--censor-rate", "0.2", "--accrual-months", "12", "--followup-months", "12"],
     "give either --censor-rate or an accrual/follow-up design, not both"),
    (["--accrual-months", "-1", "--followup-months", "12"],
     "--accrual-months and --followup-months must be non-negative"),
    (["--accrual-months", "0", "--followup-months", "0"],
     "--accrual-months and --followup-months cannot both be zero"),
    (["--dropout", "0.1"], "--dropout requires --accrual-months/--followup-months "
                           "(with --censor-rate, include dropout in the censoring rate)"),
    (["--non-evaluable", "1"], "--non-evaluable must be in [0, 1)"),
    (["--r2", "0.3"], "--r2 requires --n-covariates >= 1"),
    (["--r2", "1.5", "--n-covariates", "2"], "--r2 must be in [0, 1)"),
    (["--biomarker-sd", "0"], "--biomarker-sd must be positive"),
    (["--median-os", "0"], "--median-os must be positive"),
    (["--n-sims", "0"], "--n-sims must be a positive integer"),
    (["--n-sims-grid", "0"], "--n-sims-grid must be a positive integer"),
])
def test_cli_reports_design_errors_in_its_own_terms(args, message, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["sample-size", "--hr", "1.2", *args])
    assert exc.value.code == 2  # argparse's usage error
    err = capsys.readouterr().err
    assert message in err and "usage:" in err


def test_cli_quick_leaves_explicit_invalid_counts_to_be_rejected(capsys):
    with pytest.raises(SystemExit):
        main(["sample-size", "--hr", "1.2", "--quick", "--n-sims", "0"])
    assert "--n-sims must be a positive integer" in capsys.readouterr().err


@pytest.mark.parametrize("command, flags, offered", [
    (["power", "--hr", "1.2", "--n", "100"], ["--median-os", "0"], "--median-os"),
    (["sample-size", "--hr", "1.2"], ["--dropout", "0.1"], "--dropout"),
    (["detectable-effect", "--n", "100"], ["--n-covariates", "-1"], "--n-covariates"),
    (["benchmark", "--c-index", "0.7", "--c0", "0.6"], ["--alpha", "2"], "--alpha"),
    (["benchmark", "--c-index", "0.7", "--c0", "0.6"], ["--n-sims-grid", "0"], "--n-sims-grid"),
    (["precision", "--c-index", "0.7", "--n", "100"], ["--c-estimator", "uno", "--c-tau", "-1"], "--c-tau"),
    (["precision", "--c-index", "0.7", "--n", "100"], ["--censor-rate", "2"], "--censor-rate"),
    (["precision", "--c-index", "0.7", "--n", "100"], ["--n-sims", "0"], "--n-sims"),
])
def test_design_errors_name_the_flag_on_every_command(command, flags, offered, capsys):
    with pytest.raises(SystemExit) as exc:
        main([*command, *flags])
    assert exc.value.code == 2
    assert f"error: {offered}" in capsys.readouterr().err
