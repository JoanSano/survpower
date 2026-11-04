"""Study planning: the "solve for X" entry points behind the CLI.

Power, N and the effect size are linked -- fix any two (plus alpha) and the
third follows:

    effect + N  -> power                  `run_power_mode`
    effect      -> required events and N  `run_sample_size_mode`
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
    theoretical_power_with_r2,
)
from .concordance import (
    benchmark_power,
    benchmark_sample_size,
    beta_of_c,
    c_of_beta,
    concordance_limit,
    default_uno_tau,
    precision_half_width,
    precision_sample_size,
    sigma_c,
    simulate_cindex_power,
)
from .design import StudyDesign
from .results import proportion_mcse
from .simulate import simulate_continuous_vs_dichotomized, simulate_power_design

#: Below this many evaluable patients the U-statistic SE of C is anti-
#: conservative (Appendix B, Table 9).
MIN_N_FOR_C_TESTS = 60


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
            dich_sim = np.nan
            if univariate_fast:  # continuous and median-split fits on the same cohorts
                r = simulate_continuous_vs_dichotomized(
                    n=n_eval, hr=hr_sd, censor_rate=cr, lambda0=design.lambda0,
                    n_sims=design.n_sims, alpha=design.alpha, seed=seed, sides=design.sides)
                power, mean_ev, dich_sim = r["power_cont"], r["mean_events"], r["power_dich"]
            else:
                power, mean_ev = simulate_power_design(
                    n=n_eval, hr=hr_sd, censor_rate=cr, p=design.n_covariates,
                    r2_target=design.r2, lambda0=design.lambda0, n_sims=design.n_sims,
                    alpha=design.alpha, sides=design.sides, seed=seed, engine=design.engine,
                )
            lo, hi = wilson_ci(power, design.n_sims)
            closed = theoretical_power_with_r2(
                n=n_eval, hr=hr_sd, censor_rate=cr, r2_target=design.r2_eff,
                alpha=design.alpha, sides=design.sides,
            )
            univariate = design.n_covariates == 0
            dich = {m: (dichotomized_power(n_eval, hr_sd, cr, design.alpha, design.sides, method=m)
                        if univariate else np.nan) for m in ("freedman", "schoenfeld")}
            rows.append(dict(
                n_enrolled=n, n_evaluable=n_eval, hr=round(hr, 4), hr_per_sd=round(hr_sd, 4),
                c_index=round(c_of_beta(np.log(hr_sd)), 4),
                censor_rate=round(cr, 3), mean_events=round(mean_ev, 1),
                power_sim=round(power, 3),
                power_sim_mcse=round(proportion_mcse(power, design.n_sims), 4),
                power_ci_low=round(lo, 3), power_ci_high=round(hi, 3), n_sims=design.n_sims,
                power_closed_form=round(closed, 3),
                power_dichotomized_sim=round(float(dich_sim), 3),
                power_dich_freedman=round(float(dich["freedman"]), 3),
                power_dich_schoenfeld=round(float(dich["schoenfeld"]), 3),
                summary=(
                    f"With N = {n} patients enrolled ({n_eval} evaluable; ~{mean_ev:.0f} "
                    f"expected events, {cr:.0%} censored), a {design.test_label()} has {power:.0%} power "
                    f"(95% simulation CI {lo:.0%} to {hi:.0%}) to detect a hazard ratio of "
                    f"{design.hr_label(hr_sd)}{design.adjustment_label()}."
                ),
            ))
            combo += 1
    return pd.DataFrame(rows)


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
            out = estimate_required_n_by_simulation(
                hr=hr_sd, target_power=pw, censor_rate=cr, lambda0=design.lambda0,
                alpha=design.alpha, n_sims_per_point=design.n_sims_grid,
                confirm_n_sims=design.n_sims, seed=design.seed + 101 * combo,
                p=design.n_covariates, r2_target=design.r2, sides=design.sides,
                engine=design.engine,
            )
            details.append(out)
            n_enrol = design.enrolled(out["n_calibrated"])
            n_dich = (design.enrolled(dichotomized_sample_size(pw, hr_sd, cr, design.alpha,
                                                               design.sides, method="schoenfeld"))
                      if design.n_covariates == 0 else np.nan)
            lo, hi = out["n_ci_low"], out["n_ci_high"]
            has_ci = np.isfinite(lo)
            ci_txt = (f"; 95% Monte Carlo interval {int(lo)}-{int(hi)} evaluable" if has_ci else "")
            unresolved = has_ci and lo <= out["n_closed_form"] <= hi
            rows.append(dict(
                hr=round(hr, 4), hr_per_sd=round(hr_sd, 4), c_index=round(c_of_beta(np.log(hr_sd)), 4),
                target_power=pw, censor_rate=round(cr, 3),
                events_closed_form=int(np.ceil(out["events_closed_form"])),
                n_closed_form_evaluable=out["n_closed_form"],
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
                    f"To detect a hazard ratio of {design.hr_label(hr_sd)} with {pw:.0%} power "
                    f"({design.test_label()}{design.adjustment_label()}), about "
                    f"{out['confirmed_mean_events']:.0f} events are required ({cr:.0%} censored): "
                    f"N = {out['n_calibrated']} evaluable / {n_enrol} enrolled patients "
                    f"(simulation-calibrated{ci_txt}; confirmed power "
                    f"{out['confirmed_power']:.0%}; closed-form N = "
                    f"{design.enrolled(out['n_closed_form'])} enrolled)."
                    + (" The interval contains the closed-form N, so the calibration does not "
                       "resolve a difference from it; increase --n-sims-grid to do so."
                       if unresolved else "")
                    + (f" A median-split analysis would need ~{n_dich} enrolled."
                       if design.n_covariates == 0 else "")
                ),
            ))
            combo += 1
    return pd.DataFrame(rows), details


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
            out = estimate_detectable_hr_by_simulation(
                n=n_eval, target_power=pw, censor_rate=cr, lambda0=design.lambda0,
                alpha=design.alpha, p=design.n_covariates, r2_target=design.r2,
                sides=design.sides, n_sims_per_point=design.n_sims_grid,
                confirm_n_sims=design.n_sims, seed=design.seed + 101 * combo,
                engine=design.engine,
            )
            hr_sd = out["hr_calibrated"]
            lo, hi = out["hr_ci_low"], out["hr_ci_high"]
            has_ci = np.isfinite(lo)
            row = dict(
                n_enrolled=n, n_evaluable=n_eval, target_power=pw, censor_rate=round(cr, 3),
                hr_closed_form_per_sd=out["hr_closed_form"],
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
            rows.append(row)
            combo += 1
    return pd.DataFrame(rows)


# --- Concordance-specific questions --------------------------------------------

def _cindex_context(design: StudyDesign, c: float) -> dict:
    """beta, censoring, horizon, estimator limit and sigma_C for a target C."""
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
                        f"{design.estimator_label()} exceeds {c0:g} (true C = {c:g})")
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
                f"confidence interval for {design.estimator_label()} (true C = {c:g}) has an "
                f"expected half-width of +/-{hw:.3f}"
                + (f" (simulated: +/-{sim_hw:.3f})" if np.isfinite(sim_hw) else "") + "."
            )
            rows.append(row)
            combo += 1
    return pd.DataFrame(rows)
