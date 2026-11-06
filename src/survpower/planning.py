"""Study planning: the "solve for X" entry points behind the CLI.

Power, N and the effect size are linked -- fix any two (plus alpha) and the
third follows:

    effect + N  -> power                  `run_power_mode`
    effect      -> required events and N  `sample_size` (HR or C-index),
                                          `run_sample_size_mode` (tables)
    N           -> minimum detectable HR  `run_detectable_effect_mode`

The effect is a per-SD hazard ratio (or per raw unit with `biomarker_sd`);
a target concordance converts to one with `survpower.concordance.hr_of_c`.
Two concordance-specific questions have their own modes:

    C vs benchmark C0 (+/- N)    `run_cindex_benchmark_mode`
    precision of C   (+/- N)     `run_cindex_precision_mode`

Sample sizes passed in and reported out are ENROLLED patients; simulations
run on the evaluable subset N_eval = floor(N * (1 - non_evaluable)).
Every row carries a `summary` sentence ready for protocol / grant text.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from ._utils import wilson_ci
from .calibration import (
    calibrate_sample_size,
    estimate_detectable_hr_by_simulation,
    estimate_required_n_by_simulation,
)
from .closed_form import (
    detectable_hr_continuous_cph,
    dichotomized_power,
    dichotomized_sample_size,
    group_hr_from_persd_hr,
    median_split_expected_ci,
    median_split_median_survival,
    sample_size_continuous_cph,
    theoretical_power_with_r2,
)
from .concordance import (
    benchmark_power,
    benchmark_sample_size,
    beta_of_c,
    c_of_beta,
    hr_of_c,
    concordance_limit,
    default_uno_tau,
    precision_half_width,
    precision_sample_size,
    sigma_c,
    simulate_cindex_power,
)
from .design import StudyDesign
from .results import Estimate, proportion_mcse
from .scenario import Effect, ExponentialPH, scenario_for
from .simulate import CONT_VS_DICH, continuous_vs_dichotomized, scenario_power, simulate_power_design

#: Below this many evaluable patients the U-statistic SE of C is anti-
#: conservative (Appendix B, Table 9).
MIN_N_FOR_C_TESTS = 60


def _simulated_power(design: StudyDesign, hr_per_sd: float, n: int, censor_rate: float,
                     n_sims: int, seed: int) -> tuple[float, float]:
    """(power, mean_events) of the design's Wald test. Unadjusted designs run the
    scenario they describe; adjusted ones have their own exponential loop."""
    if design.n_covariates == 0:
        return scenario_power(scenario_for(design, Effect.hr(hr_per_sd), censor_rate=censor_rate),
                              n, n_sims, design.alpha, seed, design.sides)
    design.require_default_models("adjusted analyses")
    return simulate_power_design(
        n=n, hr=hr_per_sd, censor_rate=censor_rate, p=design.n_covariates, r2_target=design.r2,
        lambda0=design.lambda0, n_sims=n_sims, alpha=design.alpha, sides=design.sides, seed=seed,
        engine=design.engine,
    )


def _has_closed_form(design: StudyDesign, hr_per_sd: float, censor_rate: float) -> bool:
    """Whether the closed-form formulas describe this design's model. Adjusted
    designs are always exponential PH, by the guard."""
    if design.n_covariates > 0:
        design.require_default_models("adjusted analyses")
        return True
    return scenario_for(design, Effect.hr(hr_per_sd), censor_rate=censor_rate).has_closed_form


def median_split_summary(design: StudyDesign, n_eval: int, hr_per_sd: float, censor_rate: float,
                         seed: int, sim: dict | None = None) -> dict:
    """What a median-split (high vs low biomarker) analysis of an unadjusted
    design gives at this N and per-SD HR: power, the fitted high-vs-low HR with
    the 95% CI a typical trial reports, the optimistic plug-in HR, expected
    events and model-based median OS per group. Simulated on the same kind of
    cohorts with the fast engine (pass `sim` to reuse a run), Schoenfeld closed
    form otherwise. A model with no closed form is always simulated, without the
    plug-in HR; the median OS is only known for the exponential model."""
    scenario = scenario_for(design, Effect.hr(hr_per_sd), CONT_VS_DICH, censor_rate=censor_rate)
    plugin = group_hr_from_persd_hr(hr_per_sd) if scenario.has_closed_form else np.nan
    med_low, med_high = np.nan, np.nan
    if isinstance(scenario.event_times, ExponentialPH):
        med_low, med_high = median_split_median_survival(hr_per_sd, design.lambda0)
    if design.engine == "fast" or not scenario.has_closed_form:
        r = sim or continuous_vs_dichotomized(scenario, n_eval, design.n_sims, design.alpha, seed,
                                              design.sides)
        out = dict(split_power=r["power_dich"], split_hr=r["group_hr_fitted"],
                   split_hr_ci_low=r["group_hr_ci_low"], split_hr_ci_high=r["group_hr_ci_high"],
                   split_events_high=r["events_high"], split_events_low=r["events_low"],
                   split_source="simulated")
    else:
        lo, hi, d_low, d_high = median_split_expected_ci(n_eval, hr_per_sd, censor_rate)
        out = dict(split_power=round(dichotomized_power(n_eval, hr_per_sd, censor_rate, design.alpha,
                                                        design.sides, method="schoenfeld"), 3),
                   split_hr=round(plugin, 3), split_hr_ci_low=round(lo, 3),
                   split_hr_ci_high=round(hi, 3), split_events_high=round(d_high, 1),
                   split_events_low=round(d_low, 1), split_source="closed form")
    out.update(split_hr_plugin=round(plugin, 3), split_median_os_high=round(med_high, 1),
               split_median_os_low=round(med_low, 1))
    return out


def median_split_text(ms: dict) -> str:
    """The median-split clause of a summary: '... would have X% power; ...'."""
    simulated = ms["split_source"] == "simulated"
    if np.isnan(ms["split_hr_plugin"]):
        plugin_txt = ""
    elif simulated:
        plugin_txt = f"; plug-in HR per SD^1.596 = {ms['split_hr_plugin']:.2f}"
    else:
        plugin_txt = "; plug-in HR per SD^1.596, Schoenfeld closed form"
    hr_txt = (f"a high-vs-low HR of {ms['split_hr']:.2f} (typical 95% CI "
              f"{ms['split_hr_ci_low']:.2f} to {ms['split_hr_ci_high']:.2f}{plugin_txt})")
    median_txt = ("" if np.isnan(ms["split_median_os_high"]) else
                  f" and median OS {ms['split_median_os_high']:.1f} vs "
                  f"{ms['split_median_os_low']:.1f} months")
    return (f"{ms['split_power']:.0%} power and would estimate {hr_txt}, with "
            f"~{ms['split_events_high']:.0f} vs ~{ms['split_events_low']:.0f} events{median_txt} "
            f"in the high vs low group")


def run_power_mode(design: StudyDesign, hrs: list, ns: list) -> pd.DataFrame:
    """Effect and N both given: simulated power (with 95% Monte Carlo CI),
    the closed-form prediction and the median-split power, per (N, HR) pair.
    `hrs` are on the user's scale (per SD, or per unit if biomarker_sd)."""
    rows = []
    combo = 0
    for n in ns:
        n_eval = design.evaluable(n)
        for hr in hrs:
            hr_sd = design.to_per_sd(hr)
            cr = design.censor_rate_for(hr_sd)
            seed = design.seed + 101 * combo
            univariate_fast = design.n_covariates == 0 and design.engine == "fast"
            dich_sim = dich_lr_sim = np.nan
            if univariate_fast:  # continuous and median-split fits on the same cohorts
                r = continuous_vs_dichotomized(
                    scenario_for(design, Effect.hr(hr_sd), CONT_VS_DICH, censor_rate=cr),
                    n_eval, design.n_sims, design.alpha, seed, design.sides)
                power, mean_ev, dich_sim = r["power_cont"], r["mean_events"], r["power_dich"]
                dich_lr_sim = r["power_dich_logrank"]
            else:
                power, mean_ev = _simulated_power(design, hr_sd, n_eval, cr, design.n_sims, seed)
            lo, hi = wilson_ci(power, design.n_sims)
            has_closed = _has_closed_form(design, hr_sd, cr)
            closed = np.nan
            if has_closed:
                closed = theoretical_power_with_r2(
                    n=n_eval, hr=hr_sd, censor_rate=cr, r2_target=design.r2_eff,
                    alpha=design.alpha, sides=design.sides,
                )
            univariate = design.n_covariates == 0
            dich = {m: (dichotomized_power(n_eval, hr_sd, cr, design.alpha, design.sides, method=m)
                        if univariate and has_closed else np.nan) for m in ("freedman", "schoenfeld")}
            split = (median_split_summary(design, n_eval, hr_sd, cr, seed,
                                          sim=r if univariate_fast else None)
                     if univariate else {})
            rows.append(dict(
                n_enrolled=n, n_evaluable=n_eval, hr=round(hr, 4), hr_per_sd=round(hr_sd, 4),
                c_index=round(c_of_beta(np.log(hr_sd)), 4),
                censor_rate=round(cr, 3), mean_events=round(mean_ev, 1),
                power_sim=round(power, 3),
                power_sim_mcse=round(proportion_mcse(power, design.n_sims), 4),
                power_ci_low=round(lo, 3), power_ci_high=round(hi, 3), n_sims=design.n_sims,
                power_closed_form=round(closed, 3),
                power_dichotomized_sim=round(float(dich_sim), 3),
                power_dichotomized_logrank_sim=round(float(dich_lr_sim), 3),
                power_dich_freedman=round(float(dich["freedman"]), 3),
                power_dich_schoenfeld=round(float(dich["schoenfeld"]), 3),
                summary=(
                    f"With N = {n} patients enrolled ({n_eval} evaluable; ~{mean_ev:.0f} "
                    f"expected events, {cr:.0%} censored), a {design.test_label()} has {power:.0%} power "
                    f"(95% simulation CI {lo:.0%} to {hi:.0%}) to detect a hazard ratio of "
                    f"{design.hr_label(hr_sd)}{design.adjustment_label()}."
                    + (f" A median-split analysis of the same cohorts would have "
                       f"{median_split_text(split)}." if univariate else "")
                ),
                **split,
            ))
            combo += 1
    return pd.DataFrame(rows)


SAMPLE_SIZE_METHODS = ("auto", "closed_form", "calibrated", "robbins_monro")


@dataclass(frozen=True)
class SampleSizeResult:
    """Output of `sample_size`.

    n_evaluable, n_enrolled : the required N (enrolled = evaluable inflated for
                              `design.non_evaluable`)
    events                  : expected events at that N (closed form: required D)
    method                  : what produced N: "closed_form", "probit" (simulation-
                              calibrated) or "scan" (direct integer scan)
    n_closed_form           : the closed-form N (evaluable), for comparison. A model with
                              no closed form reports the proportional-hazards
                              approximation that started the calibration, and
                              `details["start"]` is "ph_approximation"
    n_ci                    : 95% Monte Carlo interval of N (probit calibration only)
    confirmed_power         : simulated power at N with its MCSE (simulation methods)
    hr_per_sd, c_index      : the effect, on both scales
    details                 : the calibration dict (grid, fit, scan) for plotting
    """

    n_evaluable: int
    n_enrolled: int
    events: float
    method: str
    n_closed_form: int
    events_closed_form: float
    n_ci: tuple | None
    confirmed_power: Estimate | None
    hr_per_sd: float
    c_index: float
    target_power: float
    censor_rate: float
    summary: str
    details: dict = field(default_factory=dict, repr=False)


def sample_size(
    hr: float | None = None,
    c_index: float | None = None,
    power: float = 0.80,
    design: StudyDesign | None = None,
    method: Literal["auto", "closed_form", "calibrated", "robbins_monro"] = "auto",
    seed: int | None = None,
) -> SampleSizeResult:
    """Required sample size for a continuous biomarker, given its hazard ratio
    or its concordance.

    hr       : hazard ratio on the user's scale (per SD, or per raw unit with
               `design.biomarker_sd`); exactly one of `hr` / `c_index`.
    c_index  : the *population* concordance of the biomarker, Gonen & Heller's
               K(beta) (eq. 25), converted with `concordance.hr_of_c`. It is not
               the censoring-dependent limit of Harrell's C: ``c_index=0.70`` means
               "the population concordance is 0.70".
    power    : target power.
    design   : `StudyDesign` (default `StudyDesign()`): alpha, sides, censoring
               or accrual calendar, non-evaluable fraction, adjustment R^2, engine
               and replicate counts.
    method   : "closed_form"  Hsieh & Lavori (2000), report Eq. 16:
                              D = (z_{1-alpha/sides} + z_power)^2 / (beta^2 (1 - R^2)),
                              N = ceil(D / P(event)). Optimistic at large effects.
               "calibrated"   simulation calibration on the probit scale
                              (`calibration.calibrate_sample_size`), no scan.
               "auto"         "calibrated", replaced by a direct integer scan when
                              the calibrated N is below 100 or the probit slope is not
                              positive. Recommended.
               "robbins_monro" not available yet (#24).
    seed     : simulation seed (default `design.seed`).

    At large effects the closed form fails: for ``c_index=0.70`` it gives
    N = 14, where the simulated power is only about 0.6; "auto" finds N ~ 22.
    """
    if (hr is None) == (c_index is None):
        raise ValueError("give exactly one of hr or c_index")
    if method not in SAMPLE_SIZE_METHODS:
        raise ValueError(f"method must be one of {SAMPLE_SIZE_METHODS}")
    if method == "robbins_monro":
        raise NotImplementedError("method='robbins_monro' is planned (#24); use 'auto'")
    design = design if design is not None else StudyDesign()
    seed = design.seed if seed is None else seed
    hr_sd = hr_of_c(c_index) if c_index is not None else design.to_per_sd(hr)
    c = c_of_beta(np.log(hr_sd))
    cr = design.censor_rate_for(hr_sd)
    has_closed = _has_closed_form(design, hr_sd, cr)

    if method == "closed_form":
        if not has_closed:
            raise ValueError("this design's event-time model has no closed form; "
                             "use method='calibrated' or 'auto'")
        D, n0 = sample_size_continuous_cph(hr_sd, power, alpha=design.alpha, event_prob=1 - cr,
                                           rho_sq=design.r2_eff, sides=design.sides)
        n_enrol = design.enrolled(n0)
        summary = (f"To detect a hazard ratio of {design.hr_label(hr_sd)} with {power:.0%} power "
                   f"({design.test_label()}{design.adjustment_label()}), {np.ceil(D):.0f} events "
                   f"are required ({cr:.0%} censored): N = {n0} evaluable / {n_enrol} enrolled "
                   f"patients (closed form, Hsieh & Lavori 2000; optimistic at large effects).")
        return SampleSizeResult(
            n_evaluable=n0, n_enrolled=n_enrol, events=round(D, 1), method="closed_form",
            n_closed_form=n0, events_closed_form=round(D, 1), n_ci=None, confirmed_power=None,
            hr_per_sd=hr_sd, c_index=c, target_power=power, censor_rate=cr, summary=summary,
            details=dict(events_closed_form=D, n_closed_form=n0, start="closed_form"))

    auto = method == "auto"
    out = estimate_required_n_by_simulation(
        hr=hr_sd, target_power=power, censor_rate=cr, lambda0=design.lambda0,
        alpha=design.alpha, n_sims_per_point=design.n_sims_grid,
        confirm_n_sims=design.n_sims, seed=seed,
        p=design.n_covariates, r2_target=design.r2, sides=design.sides,
        engine=design.engine, scan_below=100 if auto else 0, scan_if_flat=auto,
        power_fn=lambda n, n_sims, s: _simulated_power(design, hr_sd, n, cr, n_sims, s),
    )
    out["start"] = "closed_form" if has_closed else "ph_approximation"
    n_enrol = design.enrolled(out["n_calibrated"])
    lo, hi = out["n_ci_low"], out["n_ci_high"]
    has_ci = bool(np.isfinite(lo))
    ci_txt = (f"; 95% Monte Carlo interval {int(lo)}-{int(hi)} evaluable" if has_ci else "")
    unresolved = has_closed and has_ci and lo <= out["n_closed_form"] <= hi
    start_label = "closed-form N" if has_closed else "PH approximation used as the starting N"
    summary = (
        f"To detect a hazard ratio of {design.hr_label(hr_sd)} with {power:.0%} power "
        f"({design.test_label()}{design.adjustment_label()}), about "
        f"{out['confirmed_mean_events']:.0f} events are required ({cr:.0%} censored): "
        f"N = {out['n_calibrated']} evaluable / {n_enrol} enrolled patients "
        f"(simulation-calibrated{ci_txt}; confirmed power "
        f"{out['confirmed_power']:.0%}; {start_label} = "
        f"{design.enrolled(out['n_closed_form'])} enrolled)."
        + (" The interval contains the closed-form N, so the calibration does not "
           "resolve a difference from it; increase --n-sims-grid to do so."
           if unresolved else "")
    )
    return SampleSizeResult(
        n_evaluable=out["n_calibrated"], n_enrolled=n_enrol,
        events=out["confirmed_mean_events"], method=out["method"],
        n_closed_form=out["n_closed_form"], events_closed_form=out["events_closed_form"],
        n_ci=(lo, hi) if has_ci else None,
        confirmed_power=Estimate(out["confirmed_power"],
                                 round(proportion_mcse(out["confirmed_power"], design.n_sims), 4)),
        hr_per_sd=hr_sd, c_index=c, target_power=power, censor_rate=cr, summary=summary,
        details=out)


def run_sample_size_mode(design: StudyDesign, hrs: list, powers: list) -> tuple[pd.DataFrame, list]:
    """Effect only: required events and N per (HR, target power) -- closed
    form and simulation-calibrated -- inflated for non-evaluable patients,
    plus the N a median-split analysis would need. Also returns the full
    calibration dicts (for `plotting.plot_calibration_curves`)."""
    rows, details = [], []
    combo = 0
    for hr in hrs:
        hr_sd = design.to_per_sd(hr)
        cr = design.censor_rate_for(hr_sd)
        for pw in powers:
            res = sample_size(hr=hr, power=pw, design=design, method="auto",
                              seed=design.seed + 101 * combo)
            out = res.details
            details.append(out)
            n_enrol = res.n_enrolled
            split = {}
            n_dich = np.nan
            has_closed = out["start"] == "closed_form"
            show_split = design.n_covariates == 0 and has_closed
            if show_split:
                n_dich_eval = dichotomized_sample_size(pw, hr_sd, cr, design.alpha, design.sides,
                                                       method="schoenfeld")
                n_dich = design.enrolled(n_dich_eval)
                split = median_split_summary(design, n_dich_eval, hr_sd, cr,
                                             seed=design.seed + 101 * combo + 7)
            lo, hi = out["n_ci_low"], out["n_ci_high"]
            rows.append(dict(
                hr=round(hr, 4), hr_per_sd=round(hr_sd, 4), c_index=round(c_of_beta(np.log(hr_sd)), 4),
                target_power=pw, censor_rate=round(cr, 3),
                events_closed_form=int(np.ceil(out["events_closed_form"])) if has_closed else np.nan,
                n_closed_form_evaluable=out["n_closed_form"] if has_closed else np.nan,
                n_calibrated_evaluable=out["n_calibrated"],
                n_calibrated_ci_low=lo, n_calibrated_ci_high=hi,
                n_enrolled=n_enrol,
                expected_events=out["confirmed_mean_events"],
                confirmed_power=out["confirmed_power"],
                confirmed_power_mcse=round(proportion_mcse(out["confirmed_power"], design.n_sims), 4),
                n_sims=design.n_sims,
                calibration=out["method"],
                n_dichotomized_enrolled=n_dich,
                summary=(
                    res.summary
                    + (f" A median-split analysis would need ~{n_dich} enrolled (Schoenfeld "
                       f"closed form); at that N it would have {median_split_text(split)}."
                       if show_split else "")
                ),
                **split,
            ))
            combo += 1
    return pd.DataFrame(rows), details


#: Simulation searches for the detectable HR under an accrual design, each at
#: the censoring of the previous search's HR (the first at the closed-form one).
MAX_DETECTABLE_RUNS = 3


def detectable_censor_rate(design: StudyDesign, n_eval: int, power: float,
                           tol: float = 1e-3, max_iter: int = 10) -> float:
    """Censoring rate at the (unknown) detectable HR. An accrual design's
    censoring depends on the HR and the detectable HR on the censoring, so
    iterate the closed-form detectable HR to a fixed point, starting at HR = 1.
    A fixed --censor-rate does not depend on the HR and is returned as is."""
    cr = design.censor_rate_for(1.0)
    if design.accrual_months is None:
        return cr
    for _ in range(max_iter):
        hr = detectable_hr_continuous_cph(n_eval, power, alpha=design.alpha, event_prob=1 - cr,
                                          rho_sq=design.r2_eff, sides=design.sides)
        cr, cr_prev = design.censor_rate_for(hr), cr
        if abs(cr - cr_prev) < tol:
            break
    return cr


def run_detectable_effect_mode(design: StudyDesign, ns: list, powers: list) -> pd.DataFrame:
    """N only: minimum detectable per-SD HR (and concordance) per (N, power),
    closed form and simulation-calibrated. By symmetry of X, 1/HR is
    detectable with the same power."""
    rows = []
    combo = 0
    for n in ns:
        n_eval = design.evaluable(n)
        for pw in powers:
            cr = detectable_censor_rate(design, n_eval, pw)
            for run in range(MAX_DETECTABLE_RUNS):
                out = estimate_detectable_hr_by_simulation(
                    n=n_eval, target_power=pw, censor_rate=cr, lambda0=design.lambda0,
                    alpha=design.alpha, p=design.n_covariates, r2_target=design.r2,
                    sides=design.sides, n_sims_per_point=design.n_sims_grid,
                    confirm_n_sims=design.n_sims, seed=design.seed + 101 * combo,
                    engine=design.engine,
                    power_at=lambda hr, n_sims, s: _simulated_power(design, hr, n_eval, cr, n_sims, s),
                )
                if design.accrual_months is None:
                    break
                # the simulated HR differs slightly from the closed-form one the censoring
                # was taken at: re-run at the censoring of the simulated HR until they agree
                cr_next = design.censor_rate_for(out["hr_calibrated"])
                if abs(cr_next - cr) < 1e-4 or run == MAX_DETECTABLE_RUNS - 1:
                    break
                cr = cr_next
            hr_sd = out["hr_calibrated"]
            lo, hi = out["hr_ci_low"], out["hr_ci_high"]
            has_ci = np.isfinite(lo)
            row = dict(
                n_enrolled=n, n_evaluable=n_eval, target_power=pw, censor_rate=round(cr, 3),
                hr_closed_form_per_sd=(out["hr_closed_form"] if _has_closed_form(design, hr_sd, cr)
                                       else np.nan),
                hr_detectable_per_sd=hr_sd,
                hr_ci_low=lo, hr_ci_high=hi,
                # by symmetry of X, the protective HR is 1/HR: its interval is the reciprocal one
                hr_detectable_protective_per_sd=round(1 / hr_sd, 4),
                hr_protective_ci_low=round(1 / hi, 4), hr_protective_ci_high=round(1 / lo, 4),
                c_detectable=round(c_of_beta(np.log(hr_sd)), 4),
            )
            if design.biomarker_sd is not None:
                u = design.to_per_unit
                row.update(
                    hr_detectable_per_unit=round(u(hr_sd), 4),
                    hr_per_unit_ci_low=round(u(lo), 4), hr_per_unit_ci_high=round(u(hi), 4),
                    hr_detectable_protective_per_unit=round(1 / u(hr_sd), 4),
                    hr_protective_per_unit_ci_low=round(1 / u(hi), 4),
                    hr_protective_per_unit_ci_high=round(1 / u(lo), 4),
                )
            ci_txt = ""
            if has_ci:
                u = design.to_per_unit
                ci_txt = (f" (95% Monte Carlo intervals {u(lo):.3f}-{u(hi):.3f} and "
                          f"{1 / u(hi):.3f}-{1 / u(lo):.3f})")
            row.update(
                expected_events=out["confirmed_mean_events"],
                confirmed_power=out["confirmed_power"],
                confirmed_power_mcse=round(proportion_mcse(out["confirmed_power"], design.n_sims), 4),
                n_sims=design.n_sims,
                summary=(
                    f"With N = {n} patients enrolled ({n_eval} evaluable; "
                    f"~{out['confirmed_mean_events']:.0f} expected events, {cr:.0%} censored), a "
                    f"{design.test_label()}{design.adjustment_label()} has {pw:.0%} power to "
                    f"detect a hazard ratio of {design.hr_label(hr_sd)} or larger, or "
                    f"{design.to_per_unit(1 / hr_sd):.3g} or smaller for a protective effect"
                    f"{ci_txt}; simulation-confirmed power {out['confirmed_power']:.0%}."
                ),
            )
            if design.n_covariates == 0:
                split = median_split_summary(design, n_eval, hr_sd, cr,
                                             seed=design.seed + 101 * combo + 7)
                row["summary"] += (f" At this N and HR, a median-split analysis would have "
                                   f"{median_split_text(split)}.")
                row.update(split)
            rows.append(row)
            combo += 1
    return pd.DataFrame(rows)


# --- Concordance-specific questions --------------------------------------------

def _cindex_context(design: StudyDesign, c: float) -> dict:
    """beta, censoring, horizon, estimator limit and sigma_C for a target C."""
    design.require_default_models("concordance questions")
    if design.n_covariates > 0:
        raise ValueError("concordance questions are univariate; for a biomarker added to a "
                         "clinical model, power its adjusted hazard ratio instead (Appendix B.5)")
    beta = beta_of_c(c)
    lam_c = design.lambda_c_for(np.exp(beta))
    tau = None
    if design.c_estimator == "uno":
        tau = design.c_tau if design.c_tau is not None else default_uno_tau(beta, lam_c, design.lambda0)
    c_lim = concordance_limit(beta, lam_c, design.c_estimator, tau=tau, lambda0=design.lambda0)
    sig = sigma_c(beta, lam_c, design.c_estimator, tau=tau, lambda0=design.lambda0)
    return dict(beta=beta, lam_c=lam_c, tau=tau, c_limit=c_lim, sigma_c=sig,
                censor_rate=design.censor_rate_for(np.exp(beta)))


def _cindex_columns(design: StudyDesign, c: float, ctx: dict) -> dict:
    cols = dict(c_target=c, hr_per_sd=round(float(np.exp(ctx["beta"])), 4),
                estimator=design.c_estimator, c_limit=round(ctx["c_limit"], 4),
                sigma_c=round(ctx["sigma_c"], 4), censor_rate=round(ctx["censor_rate"], 3))
    if ctx["tau"] is not None:
        cols["tau_months"] = round(ctx["tau"], 1)
    return cols


def run_cindex_benchmark_mode(design: StudyDesign, cs: list, c0s: list,
                              ns: list | None = None, powers: list | None = None) -> pd.DataFrame:
    """Power (given N) or N (given target power) to show that the biomarker's
    concordance exceeds a benchmark: one-sided test of H0: C <= C0 at level
    alpha, using the chosen estimator (Appendix B.4). The closed form uses the
    estimator's limit under the design (not the nominal C); the N is then
    simulation-calibrated."""
    powers = powers or [0.80]
    rows = []
    combo = 0
    for c in cs:
        ctx = _cindex_context(design, c)
        for c0 in c0s:
            if ctx["c_limit"] <= c0:
                raise ValueError(f"benchmark C0 = {c0} is not below the estimator's limit "
                                 f"{ctx['c_limit']:.4f} for target C = {c}")

            def power_fn(n, n_sims, s, _c0=c0):
                r = simulate_cindex_power(n, ctx["beta"], ctx["lam_c"], n_sims=n_sims,
                                          alpha=design.alpha, c0=_c0, method=design.c_estimator,
                                          tau=ctx["tau"], lambda0=design.lambda0, seed=s)
                return r["power_cc0"], r["mean_events"]

            base = dict(_cindex_columns(design, c, ctx), c0=c0)
            question = (f"a one-sided test at alpha = {design.alpha:g} that the biomarker's "
                        f"{design.estimator_label()} exceeds {c0:g} (true C = {c:g}, "
                        f"HR {design.hr_value_label(np.exp(ctx['beta']))})")
            if ns:
                for n in ns:
                    n_eval = design.evaluable(n)
                    pw, ev = power_fn(n_eval, design.n_sims, design.seed + 101 * combo)
                    lo, hi = wilson_ci(pw, design.n_sims)
                    closed = benchmark_power(n_eval, ctx["c_limit"], c0, ctx["sigma_c"], design.alpha)
                    rows.append(dict(
                        base, n_enrolled=n, n_evaluable=n_eval, mean_events=ev,
                        power_sim=round(pw, 3),
                        power_sim_mcse=round(proportion_mcse(pw, design.n_sims), 4),
                        power_ci_low=round(lo, 3), power_ci_high=round(hi, 3),
                        n_sims=design.n_sims, power_closed_form=round(closed, 3),
                        summary=(f"With N = {n} patients enrolled ({n_eval} evaluable; ~{ev:.0f} "
                                 f"events, {ctx['censor_rate']:.0%} censored), {question} has {pw:.0%} power (95% simulation CI "
                                 f"{lo:.0%} to {hi:.0%})."),
                    ))
                    combo += 1
            else:
                for target in powers:
                    n0 = benchmark_sample_size(ctx["c_limit"], c0, ctx["sigma_c"], target, design.alpha)
                    out = calibrate_sample_size(
                        power_fn, n0, target, n_sims_per_point=design.n_sims_grid,
                        confirm_n_sims=design.n_sims, seed=design.seed + 101 * combo, min_n=20,
                    )
                    n_eval = out["n_calibrated"]
                    if n_eval < MIN_N_FOR_C_TESTS:
                        warnings.warn(f"N = {n_eval} is below {MIN_N_FOR_C_TESTS}: the SE of C is "
                                      "anti-conservative there (Appendix B, Table 9)")
                    n_enrol = design.enrolled(n_eval)
                    rows.append(dict(
                        base, target_power=target, n_closed_form_evaluable=n0,
                        n_calibrated_evaluable=n_eval, n_enrolled=n_enrol,
                        expected_events=out["confirmed_mean_events"],
                        confirmed_power=out["confirmed_power"],
                        confirmed_power_mcse=round(
                            proportion_mcse(out["confirmed_power"], design.n_sims), 4),
                        n_sims=design.n_sims, calibration=out["method"],
                        summary=(f"For {target:.0%} power in {question}, N = {n_eval} evaluable / "
                                 f"{n_enrol} enrolled patients are required (~"
                                 f"{out['confirmed_mean_events']:.0f} events, {ctx['censor_rate']:.0%} censored; "
                                 f"simulation-confirmed "
                                 f"power {out['confirmed_power']:.0%}; closed-form N = "
                                 f"{design.enrolled(n0)} enrolled)."),
                    ))
                    combo += 1
    return pd.DataFrame(rows)


def run_cindex_precision_mode(design: StudyDesign, cs: list, half_widths: list | None = None,
                              ns: list | None = None, level: float = 0.95,
                              check_sims: int = 500) -> pd.DataFrame:
    """Precision of the estimated concordance: the N for a target CI
    half-width, or the half-width achieved at a given N. Closed form
    h = z sigma_C / sqrt(N), checked against the empirical SD of C in
    `check_sims` simulated cohorts (skipped above 4000 patients)."""
    rows = []
    combo = 0
    for c in cs:
        ctx = _cindex_context(design, c)
        base = _cindex_columns(design, c, ctx)
        targets = ([(design.evaluable(n), n, None) for n in ns] if ns
                   else [(precision_sample_size(ctx["sigma_c"], h, level), None, h)
                         for h in half_widths])
        for n_eval, n_enrol, h in targets:
            n_enrol = n_enrol if n_enrol is not None else design.enrolled(n_eval)
            hw = precision_half_width(n_eval, ctx["sigma_c"], level)
            sim_hw = sim_hw_mcse = np.nan
            n_check = min(check_sims, design.n_sims)
            if n_eval <= 4000:
                r = simulate_cindex_power(n_eval, ctx["beta"], ctx["lam_c"],
                                          n_sims=n_check,
                                          method=design.c_estimator, tau=ctx["tau"],
                                          lambda0=design.lambda0, seed=design.seed + 101 * combo)
                sim_hw, sim_hw_mcse = 1.96 * r["sd_c"], 1.96 * r["sd_c_mcse"]
            row = dict(base, n_enrolled=n_enrol, n_evaluable=n_eval,
                       half_width_closed_form=round(hw, 4), half_width_simulated=round(sim_hw, 4),
                       half_width_simulated_mcse=round(sim_hw_mcse, 4),
                       n_sims=n_check if n_eval <= 4000 else 0)
            if h is not None:
                row["target_half_width"] = h
            row["summary"] = (
                f"With N = {n_enrol} patients enrolled ({n_eval} evaluable, "
                f"{ctx['censor_rate']:.0%} censored), the {level:.0%} "
                f"confidence interval for {design.estimator_label()} (true C = {c:g}, "
                f"HR {design.hr_value_label(np.exp(ctx['beta']))}) has an "
                f"expected half-width of +/-{hw:.3f}"
                + (f" (simulated: +/-{sim_hw:.3f})" if np.isfinite(sim_hw) else "") + "."
            )
            rows.append(row)
            combo += 1
    return pd.DataFrame(rows)
