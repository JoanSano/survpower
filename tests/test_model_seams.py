"""One route from a StudyDesign to a Scenario, and a guard on every other path (#62).

`StudyDesign` has no model fields yet (#14 and #16 add them), so a test model enters by
replacing `planning.scenario_for`, and a design with a non-default model is a subclass."""

import dataclasses
import inspect
from dataclasses import dataclass, field

import numpy as np
import pytest

import survpower
from survpower import planning, sensitivity
from survpower.design import StudyDesign
from survpower.planning import (
    run_cindex_benchmark_mode,
    run_cindex_precision_mode,
    run_detectable_effect_mode,
    run_power_mode,
    run_sample_size_mode,
    sample_size,
)
from survpower.scenario import (
    CoxWald,
    Effect,
    ExponentialCensoring,
    ExponentialPH,
    Scenario,
    StandardNormal,
    run_replicates,
    scenario_for,
    simulate,
)
from survpower.sensitivity import run_sensitivity
from survpower.simulate import simulate_power

FEW = dict(n_sims=30, n_sims_grid=20, seed=5)
DRAWN = []  # how many times the test event-time models were drawn from


@dataclass(frozen=True)
class _CountingPH:
    """Exponential PH that records its use. `is_ph` is settable, to act as a non-PH model
    with no closed form."""

    log_hr: float = 0.0
    is_ph: bool = True

    def draw(self, cov, rng):
        DRAWN.append(cov.n)
        return rng.exponential(scale=15.0 / np.exp(self.log_hr * cov.X))

    def survival(self, t, x):
        return np.exp(-np.asarray(t) * np.exp(self.log_hr * np.asarray(x)) / 15.0)


def _use_test_model(monkeypatch, is_ph=True):
    def fake(design, effect, analyses=None, censor_rate=None):
        return Scenario(StandardNormal(), _CountingPH(np.log(effect.per_sd), is_ph),
                        ExponentialCensoring(target_rate=0.2 if censor_rate is None else censor_rate),
                        analyses or (CoxWald(engine=design.engine),))

    def legacy_engine_must_not_run(*args, **kwargs):
        raise AssertionError("an unadjusted path bypassed scenario_for")

    monkeypatch.setattr(planning, "scenario_for", fake)
    monkeypatch.setattr(planning, "simulate_power_design", legacy_engine_must_not_run)
    DRAWN.clear()


@dataclass(frozen=True)
class _DesignWithModel(StudyDesign):
    """What a design looks like once #14 adds a field."""

    def non_default_models(self):
        return ("baseline",)


# --- scenario_for ------------------------------------------------------------------

def test_scenario_for_builds_the_default_models():
    design = StudyDesign(median_os=20, censor_rate=0.3)
    sc = scenario_for(design, Effect.hr(1.2))
    assert sc == Scenario(StandardNormal(), ExponentialPH(np.log(1.2), design.lambda0),
                          ExponentialCensoring(target_rate=0.3), (CoxWald(engine="fast"),))
    assert sc.has_closed_form


def test_scenario_for_matches_the_legacy_engine_bit_for_bit():
    design = StudyDesign(censor_rate=0.3, engine="fast")
    row = simulate(scenario_for(design, Effect.hr(1.25)), n=150, n_sims=30, seed=7).iloc[0]
    power, events, lam_c = simulate_power(150, 1.25, 0.3, lambda0=design.lambda0, n_sims=30, seed=7)
    assert (row["power"], row["mean_events"], row["lambda_c"]) == (power, events, lam_c)


def test_scenario_for_takes_the_solved_censoring_and_analyses():
    sc = scenario_for(StudyDesign(), Effect.hr(1.2), analyses=(CoxWald(dichotomize=True),),
                      censor_rate=0.45)
    assert sc.censoring.target_rate == 0.45
    assert [a.name for a in sc.analyses] == ["cox_wald_median_split"]


def test_scenario_for_uses_the_censoring_at_the_effect():
    design = StudyDesign(accrual_months=24, followup_months=12)
    sc = scenario_for(design, Effect.hr(1.5))
    assert sc.censoring.target_rate == design.censor_rate_for(1.5)


def test_scenario_for_refuses_what_it_cannot_build():
    with pytest.raises(ValueError, match="n_covariates"):
        scenario_for(StudyDesign(n_covariates=2), Effect.hr(1.2))
    with pytest.raises(ValueError, match="baseline"):  # a new field cannot have no effect
        scenario_for(_DesignWithModel(), Effect.hr(1.2))


@pytest.mark.parametrize("kwargs", [dict(scale="odds", per_sd=1.2), dict(scale="hr", per_sd=0.0),
                                    dict(scale="hr", per_sd=-1.0), dict(scale="hr", per_sd=np.inf),
                                    dict(scale="hr", per_sd=np.nan)])
def test_effect_validates(kwargs):
    with pytest.raises(ValueError):
        Effect(**kwargs)


def test_effect_is_frozen_hashable_and_exported():
    effect = Effect.hr(1.2)
    assert effect == Effect("hr", 1.2) and hash(effect) == hash(Effect("hr", 1.2))
    with pytest.raises(dataclasses.FrozenInstanceError):
        effect.per_sd = 2.0
    assert survpower.Effect is Effect and survpower.scenario_for is scenario_for


# --- a test model reaches every unadjusted path ------------------------------------------

@pytest.mark.parametrize("engine", ["fast", "lifelines"])
@pytest.mark.parametrize("run", [
    lambda d: run_power_mode(d, [1.4], [120]),
    lambda d: sample_size(hr=1.5, design=d),
    lambda d: sample_size(hr=1.5, design=d, method="calibrated"),
    lambda d: run_sample_size_mode(d, [1.5], [0.8]),
    lambda d: run_detectable_effect_mode(d, [120], [0.8]),
], ids=["power", "sample_size_auto", "sample_size_calibrated", "sample_size_mode", "detectable"])
def test_a_test_model_reaches_the_simulation(monkeypatch, engine, run):
    _use_test_model(monkeypatch)
    run(StudyDesign(engine=engine, **FEW))
    assert DRAWN  # and `legacy_engine_must_not_run` did not fire


def test_a_test_model_changes_the_answer(monkeypatch):
    """Power follows the model that was plugged in, not the design's exponential default."""
    design = StudyDesign(**FEW)
    high = run_power_mode(design, [2.0], [100])["power_sim"].iloc[0]

    def null_model(design, effect, analyses=None, censor_rate=None):
        return Scenario(StandardNormal(), _CountingPH(0.0),
                        ExponentialCensoring(target_rate=0.2), analyses or (CoxWald(),))

    monkeypatch.setattr(planning, "scenario_for", null_model)
    null = run_power_mode(design, [2.0], [100])["power_sim"].iloc[0]
    assert high > 0.9 and null < 0.3


# --- models with no closed form ---------------------------------------------------------

def test_no_closed_form_raises_and_auto_labels_the_start(monkeypatch):
    _use_test_model(monkeypatch, is_ph=False)
    design = StudyDesign(**FEW)
    with pytest.raises(ValueError, match="no closed form"):
        sample_size(hr=1.5, design=design, method="closed_form")
    res = sample_size(hr=1.5, design=design)
    assert res.details["start"] == "ph_approximation"
    assert "PH approximation used as the starting N" in res.summary
    assert "closed-form" not in res.summary


def test_no_closed_form_leaves_those_columns_empty(monkeypatch):
    _use_test_model(monkeypatch, is_ph=False)
    design = StudyDesign(**FEW)
    power = run_power_mode(design, [1.4], [120]).iloc[0]
    assert np.isnan(power["power_closed_form"]) and np.isnan(power["power_dich_schoenfeld"])
    assert np.isnan(power["split_hr_plugin"]) and power["split_source"] == "simulated"
    assert "plug-in" not in power["summary"] and "median OS" not in power["summary"]
    ss = run_sample_size_mode(design, [1.5], [0.8])[0].iloc[0]
    assert np.isnan(ss["n_closed_form_evaluable"]) and np.isnan(ss["n_dichotomized_enrolled"])
    assert "A median-split analysis would need" not in ss["summary"]
    de = run_detectable_effect_mode(design, [120], [0.8]).iloc[0]
    assert np.isnan(de["hr_closed_form_per_sd"])


def test_no_closed_form_is_simulated_with_lifelines_too(monkeypatch):
    _use_test_model(monkeypatch, is_ph=False)
    row = run_power_mode(StudyDesign(engine="lifelines", **FEW), [1.4], [120]).iloc[0]
    assert row["split_source"] == "simulated"


# --- every path supports a non-default model, or raises ------------------------------------

SUPPORTS = {
    "run_power_mode": lambda d: run_power_mode(d, [1.4], [120]),
    "sample_size": lambda d: sample_size(hr=1.5, design=d),
    "run_sample_size_mode": lambda d: run_sample_size_mode(d, [1.5], [0.8]),
    "run_detectable_effect_mode": lambda d: run_detectable_effect_mode(d, [120], [0.8]),
}
RAISES = {
    "adjusted power": (lambda d: run_power_mode(d, [1.4], [120]), dict(n_covariates=2)),
    "adjusted sample_size": (lambda d: sample_size(hr=1.5, design=d), dict(n_covariates=2)),
    "adjusted closed form": (lambda d: sample_size(hr=1.5, design=d, method="closed_form"),
                             dict(n_covariates=2)),
    "adjusted sample-size mode": (lambda d: run_sample_size_mode(d, [1.5], [0.8]),
                                  dict(n_covariates=2)),
    "adjusted detectable": (lambda d: run_detectable_effect_mode(d, [120], [0.8]),
                            dict(n_covariates=2)),
    "run_cindex_benchmark_mode": (lambda d: run_cindex_benchmark_mode(d, [0.7], [0.6], ns=[100]), {}),
    "run_cindex_precision_mode": (lambda d: run_cindex_precision_mode(d, [0.7], ns=[100]), {}),
    "run_sensitivity": (lambda d: run_sensitivity(d, n=100, hr=1.3, which=["waning"], plots=False), {}),
}
#: These take a design but are steps of the paths above.
STEPS = {"median_split_summary", "detectable_censor_rate"}


def test_the_lists_above_cover_every_path_that_takes_a_design():
    found = set()
    for module in (planning, sensitivity):
        for name, fn in inspect.getmembers(module, inspect.isfunction):
            if (fn.__module__ == module.__name__ and not name.startswith("_")
                    and "design" in inspect.signature(fn).parameters):
                found.add(name)
    assert found == set(SUPPORTS) | {n for n in RAISES if n.startswith("run_")} | STEPS


@pytest.mark.parametrize("name", SUPPORTS)
def test_unadjusted_paths_support_a_non_default_model(monkeypatch, name):
    _use_test_model(monkeypatch)
    SUPPORTS[name](_DesignWithModel(**FEW))
    assert DRAWN


@pytest.mark.parametrize("name", RAISES)
def test_other_paths_raise_naming_the_option(name):
    run, kwargs = RAISES[name]
    with pytest.raises(ValueError, match="baseline"):
        run(_DesignWithModel(**FEW, **kwargs))


def test_unadjusted_paths_raise_until_scenario_for_builds_the_model():
    """Without the monkeypatch, no path silently gives an exponential answer."""
    for run in SUPPORTS.values():
        with pytest.raises(ValueError, match="baseline"):
            run(_DesignWithModel(**FEW))


# --- censoring sees the cohort ----------------------------------------------------------

@dataclass(frozen=True)
class _CutAtEvents:
    """An analysis at the d-th event: everyone still alive is censored then."""

    d: int
    calls: list = field(default_factory=list, compare=False)

    def calibrated(self, sampler, seed):
        return self

    def draw(self, cov, event_times, rng):
        self.calls.append(len(event_times))
        return np.full(cov.n, np.sort(event_times)[self.d - 1])


def test_a_censoring_model_can_use_the_event_times():
    model = _CutAtEvents(d=40)
    sc = Scenario(event_times=ExponentialPH(log_hr=np.log(1.3)), censoring=model)
    reps = run_replicates(sc, n=100, n_sims=5, seed=0)
    assert np.all(reps.events == 40)
    assert model.calls == [100] * 5


# --- no numeric change ------------------------------------------------------------------

PINNED = StudyDesign(n_sims=60, n_sims_grid=40, seed=5)


def test_pinned_power():
    row = run_power_mode(PINNED, [1.3], [150]).iloc[0]
    assert [row["power_sim"], row["mean_events"], row["power_dichotomized_sim"]] == [0.8, 119.8, 0.633]


def test_pinned_sample_size():
    row = run_sample_size_mode(PINNED, [1.4], [0.8])[0].iloc[0]
    assert [row["n_calibrated_evaluable"], row["confirmed_power"]] == [115, 0.867]


def test_pinned_detectable_effect():
    row = run_detectable_effect_mode(PINNED, [150], [0.8]).iloc[0]
    assert [row["hr_detectable_per_sd"], row["confirmed_power"]] == [1.3176, 0.85]
