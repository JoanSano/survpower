"""Monte Carlo power engines.

Every engine: draws a cohort from a known data-generating process, applies
exponential censoring calibrated to a target marginal rate, fits a Cox model
and records the Wald test on the biomarker X. Seeds make every result exactly
reproducible. Other censoring models (#16) go through `survpower.scenario`.

Every dict output carries `n_sims`, a `*_mcse` column per Monte Carlo
estimate (see `survpower.results`) and the realized censored fraction
`censor_rate_realized`. `mean_HR_hat` = exp(mean(beta_hat)) is
inflated by ~exp(Var(beta_hat)/2) even when beta_hat is unbiased; bias is
reported on the log-HR scale (`bias_beta`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines.statistics import proportional_hazard_test
from scipy.stats import norm

from ._utils import tqdm
from .baseline import DEFAULT_LAMBDA0, is_exponential
from .censoring import (
    calibrate_censoring_crossing,
    calibrate_censoring_rate,
    calibrate_censoring_rate_heterogeneous,
    calibrate_censoring_waning,
)
from .dgp import gen_adjustment_covariates, gen_crossing, gen_waning, ph_event_times
from .fitting import ENGINES, cox_fit_1d, fit_cox, logrank_1d, rejects, wald_rejects
from .results import (censoring_columns, exp_mean_mcse, mean_mcse, proportion_mcse, sd_mcse,
                      summarize_replicates)
from .scenario import (
    CoxWald,
    ExponentialCensoring,
    ExponentialPH,
    LogRank,
    PHModel,
    Scenario,
    StandardNormal,
    design_frame,
    run_replicates,
)


def _r4(x: float) -> float:
    return round(float(x), 4)


def _beta_columns(reject, beta_hats, ses, beta, alpha) -> dict:
    """MCSE columns for one Wald-tested coefficient, rounded for tables."""
    s = summarize_replicates(reject, beta_hats, ses, beta_true=beta, alpha=alpha)
    return dict(
        power_mcse=_r4(s["power_mcse"]),
        bias_beta=_r4(s["bias_beta"]), bias_beta_mcse=_r4(s["bias_beta_mcse"]),
        emp_se=_r4(s["emp_se"]), emp_se_mcse=_r4(s["emp_se_mcse"]),
        mean_model_se=_r4(s["mean_model_se"]),
        coverage=round(s["coverage"], 3), coverage_mcse=_r4(s["coverage_mcse"]),
        mean_HR_hat_mcse=_r4(exp_mean_mcse(beta_hats)),
        n_failed=s["n_failed"],
    )


def _censoring(events, n: int) -> dict:
    """`results.censoring_columns`, rounded for tables."""
    return {k: _r4(v) for k, v in censoring_columns(events, n).items()}


def _censor(T: np.ndarray, lam_c: float, rng: np.random.Generator):
    C = rng.exponential(scale=1.0 / lam_c, size=len(T))
    return np.minimum(T, C), (T <= C).astype(int)


_design_frame = design_frame


def _ph_scenario(hr, censor_rate, lambda0, analyses, baseline=None) -> Scenario:
    """PH with X ~ N(0, 1) and calibrated exponential censoring: exponential
    (`ExponentialPH`, rate `lambda0` or the `Exponential` baseline's) unless
    another `baseline` is given (`PHModel`)."""
    if is_exponential(baseline):
        lam = lambda0 if baseline is None else baseline.rate
        event_times = ExponentialPH(log_hr=np.log(hr), lambda0=lam)
    else:
        event_times = PHModel(baseline, log_hr=np.log(hr))
    return Scenario(StandardNormal(), event_times, ExponentialCensoring(target_rate=censor_rate),
                    analyses)


def simulate_power(
    n: int,
    hr: float,
    censor_rate: float,
    lambda0: float = DEFAULT_LAMBDA0,
    n_sims: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    sides: int = 2,
    engine: str = "fast",
    details: bool = False,
    baseline=None,
) -> tuple[float, float, float] | dict:
    """Monte Carlo power for a continuous covariate in a univariate Cox model.

    Returns (power, mean_events, lambda_c); with `details`, the unrounded
    `summarize_replicates` dict (power, bias, empirical / model SE, coverage,
    each with its MCSE) plus `lambda_c`. `baseline` (`survpower.baseline`)
    replaces the exponential baseline of rate `lambda0`.
    """
    reps = run_replicates(_ph_scenario(hr, censor_rate, lambda0, (CoxWald(engine=engine),),
                                       baseline),
                          n, n_sims=n_sims, seed=seed, alpha=alpha, sides=sides,
                          desc=f"continuous n={n} HR={hr:.3g}")
    lam_c = reps.scenario.censoring.rate
    if details:
        out = reps.summary("cox_wald")
        out["lambda_c"] = lam_c
        return out
    rejections = int(reps.values["cox_wald"]["reject"].sum())
    return rejections / n_sims, float(np.mean(reps.events)), lam_c


def scenario_power(scenario: Scenario, n: int, n_sims: int, alpha: float, seed: int,
                   sides: int) -> tuple[float, float]:
    """(power, mean_events) of a scenario with one analysis."""
    reps = run_replicates(scenario, n, n_sims=n_sims, seed=seed, alpha=alpha, sides=sides,
                          desc=f"continuous n={n}")
    name = scenario.analyses[0].name
    return int(reps.values[name]["reject"].sum()) / n_sims, float(np.mean(reps.events))


def simulate_power_with_adjustment(
    n: int,
    hr: float,
    censor_rate: float,
    p: int,
    r2_target: float,
    lambda0: float = DEFAULT_LAMBDA0,
    n_sims: int = 300,
    alpha: float = 0.05,
    seed: int = 0,
    sides: int = 2,
    baseline=None,
) -> dict:
    """Power for X in a Cox model adjusted for p null covariates whose
    multiple correlation with X is r2_target. The true hazard
    depends only on X; `baseline` replaces the exponential baseline."""
    beta = np.log(hr)
    lam_c = calibrate_censoring_rate(beta, censor_rate, lambda0=lambda0, seed=seed + 999,
                                     baseline=baseline)
    rng = np.random.default_rng(seed)
    rejections, events, beta_hats, ses, reject = 0, [], [], [], []
    for _ in tqdm(range(n_sims), desc=f"p={p} R2={r2_target}", leave=False):
        X, Z = gen_adjustment_covariates(n, r2_target, p, rng)
        T = ph_event_times(beta * X, rng, lambda0, baseline)
        time_, event = _censor(T, lam_c, rng)
        events.append(event.sum())
        cph, _ = fit_cox(_design_frame(time_, event, X, Z))
        r = rejects(cph, beta, alpha, sides)
        rejections += r
        reject.append(r)
        beta_hats.append(cph.summary.loc["X", "coef"])
        ses.append(cph.summary.loc["X", "se(coef)"])
    return dict(
        p=p, r2_target=r2_target,
        power=round(rejections / n_sims, 3),
        mean_events=round(float(np.mean(events)), 1), **_censoring(events, n),
        mean_HR_hat=round(float(np.exp(np.mean(beta_hats))), 3),
        n_sims=n_sims,
        **_beta_columns(reject, beta_hats, ses, beta, alpha),
    )


def simulate_power_design(
    n: int,
    hr: float,
    censor_rate: float,
    p: int = 0,
    r2_target: float = 0.0,
    lambda0: float = DEFAULT_LAMBDA0,
    n_sims: int = 1000,
    alpha: float = 0.05,
    sides: int = 2,
    seed: int = 0,
    engine: str = "fast",
    details: bool = False,
    baseline=None,
) -> tuple[float, float] | dict:
    """Power for X, unadjusted (p = 0, `simulate_power` with `engine`) or
    adjusted (`simulate_power_with_adjustment`, lifelines). Returns
    (power, mean_events); with `details`, the engine's full dict with MCSEs.
    `baseline` replaces the exponential baseline of rate `lambda0`."""
    if p == 0:
        out = simulate_power(
            n=n, hr=hr, censor_rate=censor_rate, lambda0=lambda0, n_sims=n_sims,
            alpha=alpha, sides=sides, seed=seed, engine=engine, details=details,
            baseline=baseline,
        )
        return out if details else (out[0], out[1])
    out = simulate_power_with_adjustment(
        n=n, hr=hr, censor_rate=censor_rate, p=p, r2_target=r2_target,
        lambda0=lambda0, n_sims=n_sims, alpha=alpha, sides=sides, seed=seed, baseline=baseline,
    )
    return out if details else (out["power"], out["mean_events"])


def simulate_continuous_vs_dichotomized(
    n: int,
    hr: float,
    censor_rate: float,
    lambda0: float = DEFAULT_LAMBDA0,
    n_sims: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    sides: int = 2,
    baseline=None,
) -> dict:
    """Continuous and median-dichotomized Cox analyses on the *same*
    simulated cohorts. The dichotomized fit uses
    G = 1{X > median(X)}; its mean coefficient gives the fitted group HR, to
    compare with the plug-in HR ** 1.5958, with the 95% CI a typical trial
    reports and the expected events per group. The groups are also compared
    with the log-rank test (`power_dich_logrank`), whose power Schoenfeld's
    closed form predicts. Uses the fast engine (identical continuous decisions
    to `simulate_power` with the same seed). `baseline` replaces the
    exponential baseline of rate `lambda0`."""
    out = continuous_vs_dichotomized(_ph_scenario(hr, censor_rate, lambda0, CONT_VS_DICH, baseline),
                                     n, n_sims, alpha, seed, sides)
    return dict(n=n, hr=hr, **out)


#: The analyses of `continuous_vs_dichotomized`, applied to the same cohorts.
CONT_VS_DICH = (CoxWald(), CoxWald(dichotomize=True), LogRank(dichotomize=True))


def continuous_vs_dichotomized(scenario: Scenario, n: int, n_sims: int, alpha: float, seed: int,
                               sides: int) -> dict:
    """The `simulate_continuous_vs_dichotomized` columns for any scenario whose
    analyses are `CONT_VS_DICH`."""
    cont, dich, lr = CONT_VS_DICH
    reps = run_replicates(scenario, n, n_sims=n_sims, seed=seed, alpha=alpha, sides=sides,
                          desc=f"cont vs dich n={n}")
    lam_c = getattr(reps.scenario.censoring, "rate", None)
    events = reps.events
    vc, vd = reps.values[cont.name], reps.values[dich.name]
    rej_cont, rej_dich = int(vc["reject"].sum()), int(vd["reject"].sum())
    rej_lr = int(reps.values[lr.name]["reject"].sum())
    bc, bg, se_gs, ev_high = vc["beta_hat"], vd["beta_hat"], vd["se"], vd["events_high"]
    return dict(
        n_sims=n_sims, mean_events=round(float(np.mean(events)), 1), **_censoring(events, n),
        power_cont=round(rej_cont / n_sims, 3), power_dich=round(rej_dich / n_sims, 3),
        group_hr_fitted=round(float(np.exp(np.nanmean(bg))), 3), lam_c=lam_c,
        power_cont_mcse=_r4(proportion_mcse(rej_cont / n_sims, n_sims)),
        power_dich_mcse=_r4(proportion_mcse(rej_dich / n_sims, n_sims)),
        power_dich_logrank=round(rej_lr / n_sims, 3),
        power_dich_logrank_mcse=_r4(proportion_mcse(rej_lr / n_sims, n_sims)),
        group_hr_fitted_mcse=_r4(exp_mean_mcse(bg)),
        # the 95% Wald CI a typical trial reports: mean log-HR +/- 1.96 * mean SE
        group_hr_ci_low=round(float(np.exp(np.nanmean(bg) - 1.96 * np.nanmean(se_gs))), 3),
        group_hr_ci_high=round(float(np.exp(np.nanmean(bg) + 1.96 * np.nanmean(se_gs))), 3),
        events_high=round(float(np.mean(ev_high)), 1),
        events_low=round(float(np.mean(events) - np.mean(ev_high)), 1),
        n_failed=int(np.sum(~np.isfinite(bc))),
    )


def _epv_pass(n, beta, lam_c, p, r2_target, lambda0, n_sims, rng, desc):
    """One EPV simulation pass: coefficients, SEs, z statistics, events and
    instability flags for X in the adjusted Cox model."""
    coefs, ses, zs, events, n_unstable = [], [], [], [], 0
    for _ in tqdm(range(n_sims), desc=desc, leave=False):
        X, Z = gen_adjustment_covariates(n, r2_target, p, rng)
        T = rng.exponential(scale=1.0 / (lambda0 * np.exp(beta * X)))
        time_, event = _censor(T, lam_c, rng)
        events.append(event.sum())
        cph, unstable = fit_cox(_design_frame(time_, event, X, Z))
        coef, se = cph.summary.loc["X", "coef"], cph.summary.loc["X", "se(coef)"]
        if se > 5 or abs(coef) > 5:
            unstable = True
        n_unstable += int(unstable)
        coefs.append(coef); ses.append(se); zs.append(cph.summary.loc["X", "z"])
    return np.array(coefs), np.array(ses), np.array(zs), np.array(events), n_unstable


def simulate_epv_diagnostic(
    n: int,
    hr: float,
    censor_rate: float,
    p: int,
    r2_target: float,
    lambda0: float = DEFAULT_LAMBDA0,
    n_sims: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    sides: int = 2,
    with_null: bool = True,
) -> dict:
    """Low events-per-variable diagnostics for X in a model adjusted for p
    covariates:

    * bias of beta_hat on the log-HR scale, with its MCSE (the mean HR is
      inflated even for an unbiased beta_hat, by ~exp(Var/2));
    * empirical SD of beta_hat vs. the mean model-based SE;
    * nominal power; and, with `with_null`, the type-I error under beta = 0
      and the size-adjusted power (critical value = empirical 1 - alpha
      quantile of the null Wald statistic at the same p).

    At EPV below ~5 the model SE is too small and the Wald test
    anti-conservative, so nominal power overstates what the design delivers.
    """
    beta = np.log(hr)
    s = 1.0 if beta >= 0 else -1.0
    stat = (lambda z: np.abs(z)) if sides == 2 else (lambda z: s * z)
    z_crit = norm.ppf(1 - alpha / sides)

    lam_c = calibrate_censoring_rate(beta, censor_rate, lambda0=lambda0, seed=seed + 999)
    rng = np.random.default_rng(seed)
    coefs, ses, zs, events, n_unstable = _epv_pass(n, beta, lam_c, p, r2_target, lambda0,
                                                    n_sims, rng, f"EPV p={p}")
    power = float(np.mean(stat(zs) > z_crit))
    out = dict(
        p=p, r2_target=r2_target, n=n, n_sims=n_sims,
        epv=round(float(np.mean(events)) / (p + 1), 2),
        mean_events=round(float(np.mean(events)), 1), **_censoring(events, n),
        power=round(power, 3),
        power_mcse=_r4(proportion_mcse(power, n_sims)),
        bias_beta=round(float(np.mean(coefs) - beta), 4),
        bias_beta_mcse=round(float(np.std(coefs, ddof=1) / np.sqrt(n_sims)), 4),
        mean_HR_hat=round(float(np.exp(np.mean(coefs))), 3),
        mean_HR_hat_mcse=_r4(exp_mean_mcse(coefs)),
        median_HR_hat=round(float(np.exp(np.median(coefs))), 3),
        sd_beta_hat=round(float(np.std(coefs)), 3),
        sd_beta_hat_mcse=_r4(sd_mcse(coefs)),
        mean_se=round(float(np.mean(ses)), 3),
        mean_se_mcse=_r4(mean_mcse(ses)),
        frac_unstable=round(n_unstable / n_sims, 3),
    )
    if with_null:
        lam_c0 = calibrate_censoring_rate(0.0, censor_rate, lambda0=lambda0, seed=seed + 1999)
        rng0 = np.random.default_rng(seed + 7_777_777)
        _, _, z0, _, _ = _epv_pass(n, 0.0, lam_c0, p, r2_target, lambda0, n_sims, rng0,
                                   f"EPV null p={p}")
        crit_emp = np.quantile(stat(z0), 1 - alpha)
        # the empirical critical value is treated as fixed in the size-adjusted MCSE
        type1 = float(np.mean(stat(z0) > z_crit))
        size_adj = float(np.mean(stat(zs) > crit_emp))
        out.update(type1_error=round(type1, 3),
                   type1_error_mcse=_r4(proportion_mcse(type1, n_sims)),
                   size_adj_power=round(size_adj, 3),
                   size_adj_power_mcse=_r4(proportion_mcse(size_adj, n_sims)))
    return out


def simulate_adjusted_vs_unadjusted(
    n: int,
    hr: float,
    censor_rate: float,
    p: int = 5,
    sigma_z: float = 0.6,
    lambda0: float = DEFAULT_LAMBDA0,
    n_sims: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    sides: int = 2,
) -> dict:
    """What omitting prognostic covariates costs. Z_1..Z_p are independent of X
    (no confounding) with log-HRs ~ N(0, sigma_z^2) redrawn per replicate. Both the adjusted model
    (X, Z) and the unadjusted model (X only) are fitted to the same data.
    The unadjusted HR is attenuated (non-collapsibility) and loses power;
    adjustment for prognostic covariates is a power *gain*."""
    beta = np.log(hr)
    lam_c = calibrate_censoring_rate_heterogeneous(beta, p, 0.0, sigma_z, censor_rate,
                                                   lambda0=lambda0, seed=seed + 999)
    rng = np.random.default_rng(seed)
    rej_adj = rej_un = 0
    b_adj, b_un, events = [], [], []
    for _ in tqdm(range(n_sims), desc=f"omitted sigma_z={sigma_z}", leave=False):
        X, Z = gen_adjustment_covariates(n, 0.0, p, rng)
        lin_pred = beta * X + (Z @ rng.normal(0.0, sigma_z, size=p) if p else 0.0)
        T = rng.exponential(scale=1.0 / (lambda0 * np.exp(lin_pred)))
        time_, event = _censor(T, lam_c, rng)
        events.append(event.sum())
        cph, _ = fit_cox(_design_frame(time_, event, X, Z))
        rej_adj += rejects(cph, beta, alpha, sides)
        b_adj.append(cph.summary.loc["X", "coef"])
        b, se = cox_fit_1d(time_, event, X)
        rej_un += wald_rejects(b / se, beta, alpha, sides)
        b_un.append(b)
    return dict(
        p=p, sigma_z=sigma_z, n=n, n_sims=n_sims, mean_events=round(float(np.mean(events)), 1),
        **_censoring(events, n),
        power_adjusted=round(rej_adj / n_sims, 3), power_unadjusted=round(rej_un / n_sims, 3),
        mean_HR_adjusted=round(float(np.exp(np.mean(b_adj))), 3),
        mean_HR_unadjusted=round(float(np.exp(np.mean(b_un))), 3),
        sd_beta_adjusted=round(float(np.std(b_adj)), 3),
        sd_beta_unadjusted=round(float(np.std(b_un)), 3),
        power_adjusted_mcse=_r4(proportion_mcse(rej_adj / n_sims, n_sims)),
        power_unadjusted_mcse=_r4(proportion_mcse(rej_un / n_sims, n_sims)),
        mean_HR_adjusted_mcse=_r4(exp_mean_mcse(b_adj)),
        mean_HR_unadjusted_mcse=_r4(exp_mean_mcse(b_un)),
        sd_beta_adjusted_mcse=_r4(sd_mcse(b_adj)),
        sd_beta_unadjusted_mcse=_r4(sd_mcse(b_un)),
        n_failed=int(np.sum(~np.isfinite(b_un))),
    )


def simulate_power_with_heterogeneous_adjustment(
    n: int,
    hr: float,
    censor_rate: float,
    p: int,
    r2_target: float,
    sigma_z: float = 0.0,
    lambda0: float = DEFAULT_LAMBDA0,
    n_sims: int = 300,
    alpha: float = 0.05,
    seed: int = 0,
    sides: int = 2,
) -> dict:
    """Power for X adjusted for p covariates that are themselves prognostic,
    each with log-HR ~ N(0, sigma_z^2) redrawn every replicate.
    sigma_z = 0 recovers the null-adjustment case."""
    beta = np.log(hr)
    lam_c = calibrate_censoring_rate_heterogeneous(
        beta, p, r2_target, sigma_z, censor_rate, lambda0=lambda0, seed=seed + 999
    )
    rng = np.random.default_rng(seed)
    rejections, events, beta_hats, realized_abs_betaz = 0, [], [], []
    ses, reject = [], []
    for _ in tqdm(range(n_sims), desc=f"p={p} sigma_z={sigma_z}", leave=False):
        X, Z = gen_adjustment_covariates(n, r2_target, p, rng)
        if p == 0:
            lin_pred = beta * X
        else:
            beta_z_vec = rng.normal(0.0, sigma_z, size=p)
            lin_pred = beta * X + Z @ beta_z_vec
            realized_abs_betaz.extend(np.abs(beta_z_vec).tolist())
        T = rng.exponential(scale=1.0 / (lambda0 * np.exp(lin_pred)))
        time_, event = _censor(T, lam_c, rng)
        events.append(event.sum())
        cph, _ = fit_cox(_design_frame(time_, event, X, Z))
        r = rejects(cph, beta, alpha, sides)
        rejections += r
        reject.append(r)
        beta_hats.append(cph.summary.loc["X", "coef"])
        ses.append(cph.summary.loc["X", "se(coef)"])
    return dict(
        p=p, r2_target=r2_target, sigma_z=sigma_z, n=n, n_sims=n_sims,
        power=round(rejections / n_sims, 3),
        mean_events=round(float(np.mean(events)), 1), **_censoring(events, n),
        mean_HR_hat=round(float(np.exp(np.mean(beta_hats))), 3),
        median_abs_HRz_realized=(
            round(float(np.exp(np.median(realized_abs_betaz))), 3) if realized_abs_betaz else 1.0
        ),
        **_beta_columns(reject, beta_hats, ses, beta, alpha),
    )


def simulate_waning_effect(
    n: int,
    hr_early: float,
    tau: float,
    censor_rate: float,
    lambda0: float = DEFAULT_LAMBDA0,
    n_sims: int = 400,
    alpha: float = 0.05,
    seed: int = 0,
    sides: int = 2,
) -> dict:
    """Waning effect (HR = hr_early before tau, 1 after) analysed with a
    constant-HR Cox model: naive power and bias, plus the Grambsch-Therneau
    test's power to detect the violation."""
    beta0 = np.log(hr_early)
    lam_c = calibrate_censoring_waning(beta0, tau, censor_rate, lambda0=lambda0, seed=seed + 999)
    rng = np.random.default_rng(seed)
    rejections_naive, rejections_zph, beta_hats, events = 0, 0, [], []
    for _ in tqdm(range(n_sims), desc=f"waning n={n}", leave=False):
        T, X = gen_waning(n, beta0, tau, lambda0, rng)
        time_, event = _censor(T, lam_c, rng)
        events.append(event.sum())
        df = _design_frame(time_, event, X)
        cph, _ = fit_cox(df)
        rejections_naive += rejects(cph, beta0, alpha, sides)
        beta_hats.append(cph.summary.loc["X", "coef"])
        zph = proportional_hazard_test(cph, df, time_transform="rank")
        rejections_zph += zph.summary.loc["X", "p"] < alpha
    return dict(
        n=n, hr_early=hr_early, tau=tau,
        power_naive=round(rejections_naive / n_sims, 3),
        power_zph_detect=round(rejections_zph / n_sims, 3),
        mean_HR_hat=round(float(np.exp(np.mean(beta_hats))), 3),
        mean_events=round(float(np.mean(events)), 1), **_censoring(events, n),
        n_sims=n_sims,
        power_naive_mcse=_r4(proportion_mcse(rejections_naive / n_sims, n_sims)),
        power_zph_detect_mcse=_r4(proportion_mcse(rejections_zph / n_sims, n_sims)),
        mean_HR_hat_mcse=_r4(exp_mean_mcse(beta_hats)),
    )


def simulate_crossing_hazards(
    n: int,
    hr_early: float,
    hr_late: float,
    tau: float,
    censor_rate: float,
    lambda0: float = DEFAULT_LAMBDA0,
    n_sims: int = 400,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Crossing hazards (HR = hr_early before tau, hr_late after), analysed
    with a constant-HR Cox model. Always two-sided: the effect
    has no single direction."""
    beta_early, beta_late = np.log(hr_early), np.log(hr_late)
    lam_c = calibrate_censoring_crossing(beta_early, beta_late, tau, censor_rate,
                                         lambda0=lambda0, seed=seed + 999)
    rng = np.random.default_rng(seed)
    rejections_naive, rejections_zph, beta_hats, events = 0, 0, [], []
    for _ in tqdm(range(n_sims), desc=f"crossing n={n}", leave=False):
        T, X = gen_crossing(n, beta_early, beta_late, tau, lambda0, rng)
        time_, event = _censor(T, lam_c, rng)
        events.append(event.sum())
        df = _design_frame(time_, event, X)
        cph, _ = fit_cox(df)
        rejections_naive += cph.summary.loc["X", "p"] < alpha
        beta_hats.append(cph.summary.loc["X", "coef"])
        zph = proportional_hazard_test(cph, df, time_transform="rank")
        rejections_zph += zph.summary.loc["X", "p"] < alpha
    return dict(
        n=n, hr_early=hr_early, hr_late=hr_late, tau=tau,
        power_naive=round(rejections_naive / n_sims, 3),
        power_zph_detect=round(rejections_zph / n_sims, 3),
        mean_HR_hat=round(float(np.exp(np.mean(beta_hats))), 3),
        mean_events=round(float(np.mean(events)), 1), **_censoring(events, n),
        n_sims=n_sims,
        power_naive_mcse=_r4(proportion_mcse(rejections_naive / n_sims, n_sims)),
        power_zph_detect_mcse=_r4(proportion_mcse(rejections_zph / n_sims, n_sims)),
        mean_HR_hat_mcse=_r4(exp_mean_mcse(beta_hats)),
    )
