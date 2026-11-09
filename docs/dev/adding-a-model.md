# Adding an event-time or censoring model

For contributors. A new data-generating option is a new model class plus one branch in
`scenario_for`. It is not a new simulation loop or a new CLI command. This page covers what
#14, #16, #63 and #64 need. The scenario API is provisional.

## How a design reaches the simulation

```
StudyDesign + Effect  --scenario_for-->  Scenario  --run_replicates-->  results
```

`scenario.scenario_for(design, effect, analyses, censor_rate=None)` is the only place that
turns a design into models. The unadjusted `run_power_mode`, `sample_size`,
`run_sample_size_mode` and `run_detectable_effect_mode` all go through it. `median_split_summary`
does too.

Every other path has its own loop and calls `design.require_default_models(path, supports=...)`
first. `supports` names the model fields that the path passes on to its engines; any other
non-default field raises:

| Path | Guard message starts with | Honours |
|---|---|---|
| The same modes with `n_covariates > 0` | `adjusted analyses` | `baseline` |
| `run_cindex_benchmark_mode`, `run_cindex_precision_mode` | `concordance questions` | `baseline` |
| `run_sensitivity` | `sensitivity analyses` | nothing |

`scenario_for` lists the fields it builds in `scenario.SCENARIO_FOR_SUPPORTS`. The paths with
their own loop take a PH baseline through `design.baseline_kwargs()`: `lambda0=` for the
exponential baseline, which keeps their random stream, and `baseline=` otherwise. They draw
event times with `dgp.ph_event_times`.

Moving one of these onto the engine is separate work, done when a feature needs it.

## Steps

1. **Write the model** in `scenario.py`, as a frozen dataclass that satisfies the protocol:

    - `EventTimeModel`: `is_ph`, `draw(cov, rng)`, `survival(t, x)`, and `log_hr` if the
      model has one true log-HR for X. `simulate` then reports bias and coverage against it,
      and it sets the direction of one-sided tests.
    - `CensoringModel`: `calibrated(sampler, seed)` and `draw(cov, event_times, rng)`.
      `event_times` are the cohort's latent times. Independent censoring ignores them; an
      analysis cut at a target number of events (#64) needs them. Censoring is drawn after the
      event times, and this order must not change.

2. **Add the design field**, following the rules below, and list it in
   `StudyDesign.non_default_models()` when it is not the default. Every guarded path then
   raises for it, naming the field.

3. **Add its branch in `scenario_for`**, and add the field to `SCENARIO_FOR_SUPPORTS`.
   Until you do, it raises like the guarded paths, so a field cannot silently have no effect.
   `baseline` (#14) is the worked example: `ExponentialPH` when it is exponential, so the
   default stream does not move, and `PHModel` otherwise.

4. **Say whether closed forms apply.** `Scenario.has_closed_form` is `event_times.is_ph`.
   Without one, `method="closed_form"` raises, the `*_closed_form` columns are NaN, and the
   calibration's starting N is reported as a PH approximation (`details["start"]`,
   `"ph_approximation"`). Also check what the closed-form helpers assume beyond PH.
   `median_split_summary` reports median OS only for `ExponentialPH`, and
   `censor_rate_from_design` assumes exponential event times.

5. **Add the CLI flag** to a shared flag group, on the commands whose path supports it
   (#60's rule: assumptions are flags, never commands), and to `cli.DESIGN_FLAGS`.

6. **Test it.** Extend `tests/test_model_seams.py`: the model reaches each path it claims, and
   the guarded paths raise. Add a Monte Carlo check of the model against a known result.

## Rules for model fields on `StudyDesign`

So that provenance (#51), `to_cli` (#54) and the GUI (#56) keep working:

- **Plain and hashable.** A field holds a frozen configuration of plain values, so
  `StudyDesign` stays hashable.
- **Round-trips through JSON,** including the model's type. A saved design has to rebuild the
  same model.
- **Otherwise API-only.** A model that cannot do this, such as `NumericBaseline`, which holds a
  function, is marked as API-only. `to_cli` then raises, and the GUI does not offer it.

## Numbers must not move

The default path keeps its random stream. A new model must not reorder draws for the default
models or change the seed offsets (`seed + 999` for calibration, `101 * combo` per
combination). `tests/test_model_seams.py` pins default results at a fixed seed.
