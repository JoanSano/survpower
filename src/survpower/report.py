"""Reproduce the tables and figures of the accompanying report.

`reproduce_report` runs Parts 1-3 with the report's original settings (and
the lifelines engine, so numbers match the PDF); `reproduce_appendix_b`
regenerates the concordance-index appendix.

.. deprecated:: 0.3.0
    This module will be removed in 0.4.0 (#27). To regenerate the report's
    numbers exactly, install the ``v0.2.0`` tag; the C <-> HR table (Table 8)
    is now `survpower.concordance.mapping_table`.
"""

from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd
from lifelines.statistics import power_under_cph

from . import plotting
from .baseline import DEFAULT_LAMBDA0, DEFAULT_MEDIAN_OS_MONTHS, DEFAULT_TAU_MONTHS
from .calibration import calibrate_sample_size, estimate_required_n_by_simulation
from .censoring import calibrate_censoring_rate
from .closed_form import group_hr_from_persd_hr, schoenfeld_two_group_power, solve_p_con
from .concordance import (
    benchmark_sample_size,
    beta_of_c,
    c_of_beta,
    harrell_limit,
    mapping_table,
    sigma_c,
    simulate_cindex_power,
)
from .sensitivity import (
    run_collinearity_study_both_cohorts,
    run_epv_stress_test,
    run_heterogeneous_adjustment_study,
    run_omitted_covariates_study,
)
from .simulate import (
    simulate_continuous_vs_dichotomized,
    simulate_crossing_hazards,
    simulate_waning_effect,
)

warnings.warn(
    "survpower.report is deprecated and will be removed in 0.4.0; to reproduce the report, "
    "install the v0.2.0 tag (see https://joansano.github.io/survpower/main/reproducing-the-report/)",
    DeprecationWarning, stacklevel=2,
)

COHORTS = [(1200, "pooled"), (250, "King's")]


def run_part1(n_sims: int = 2000, censor_rate: float = 0.20,
              lambda0: float = DEFAULT_LAMBDA0) -> pd.DataFrame:
    """Table 1: continuous vs. median-dichotomized power, both by Monte Carlo
    on the same cohorts, with the plug-in and fitted group HR and the
    Freedman and Schoenfeld closed forms for the dichotomized analysis."""
    rows = []
    for n, label in COHORTS:
        for hr in [1.10, 1.15]:
            r = simulate_continuous_vs_dichotomized(n=n, hr=hr, censor_rate=censor_rate,
                                                    lambda0=lambda0, n_sims=n_sims, seed=42)
            ghr = group_hr_from_persd_hr(hr)
            p_con = solve_p_con(ghr, marginal_p=1 - censor_rate)
            p_exp = 1 - (1 - p_con) ** ghr
            rows.append(dict(
                label=label, n=n, hr=hr, mean_events=r["mean_events"],
                power_cont=r["power_cont"], power_dich=r["power_dich"],
                group_hr_plugin=round(ghr, 3), group_hr_fitted=r["group_hr_fitted"],
                power_dich_freedman=round(power_under_cph(n // 2, n // 2, p_exp, p_con, ghr), 3),
                power_dich_schoenfeld=round(schoenfeld_two_group_power(n // 2, n // 2, p_exp, p_con, ghr), 3),
                lam_c_per_month=round(r["lam_c"], 5),
            ))
    return pd.DataFrame(rows)


def run_part1b(target_power: float = 0.80, censor_rate: float = 0.20,
               lambda0: float = DEFAULT_LAMBDA0, n_sims_per_point: int = 2000,
               confirm_n_sims: int = 10_000, return_details: bool = False):
    """Table 2: closed-form vs. simulation-calibrated N for the continuous
    covariate, with the 95% parametric-bootstrap interval for N*."""
    rows, details = [], []
    for hr in [1.10, 1.15]:
        out = estimate_required_n_by_simulation(
            hr=hr, target_power=target_power, censor_rate=censor_rate, lambda0=lambda0,
            n_sims_per_point=n_sims_per_point, confirm_n_sims=confirm_n_sims,
            seed=42, scan_below=0,
        )
        details.append(out)
        rows.append({k: out[k] for k in ("hr", "n_closed_form", "n_calibrated", "n_ci_low",
                                         "n_ci_high", "confirmed_power", "confirmed_mean_events")})
    df = pd.DataFrame(rows)
    return (df, details) if return_details else df


def run_part2(n_sims: int = 400, hr_early: float = 1.30, tau: float = DEFAULT_TAU_MONTHS,
              censor_rate: float = 0.20, lambda0: float = DEFAULT_LAMBDA0) -> pd.DataFrame:
    """Table 6: waning-effect scenario for both cohorts."""
    rows = []
    for n, label in COHORTS:
        out = simulate_waning_effect(n=n, hr_early=hr_early, tau=tau, censor_rate=censor_rate,
                                     lambda0=lambda0, n_sims=n_sims, seed=7)
        out["label"] = label
        rows.append(out)
    cols = ["label", "n", "mean_events", "power_naive", "mean_HR_hat", "power_zph_detect"]
    return pd.DataFrame(rows)[cols]


def run_part3(n_sims: int = 400, hr_early: float = 0.75, hr_late: float = 1.40,
              tau: float = DEFAULT_TAU_MONTHS, censor_rate: float = 0.20,
              lambda0: float = DEFAULT_LAMBDA0) -> pd.DataFrame:
    """Table 7: crossing-hazards scenario for the pooled cohort."""
    out = simulate_crossing_hazards(n=1200, hr_early=hr_early, hr_late=hr_late, tau=tau,
                                    censor_rate=censor_rate, lambda0=lambda0, n_sims=n_sims, seed=11)
    return pd.DataFrame([out])[["n", "mean_events", "power_naive", "mean_HR_hat", "power_zph_detect"]]


def _section(title: str, df: pd.DataFrame) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)
    print(df.to_string(index=False))


def reproduce_report(outdir: str = ".") -> dict:
    """Run every part of the main report with its original settings; write
    CSV tables and PDF figures to `outdir`. Takes ~10-15 minutes."""
    os.makedirs(outdir, exist_ok=True)
    out = lambda name: os.path.join(outdir, name)
    t0 = time.time()
    print(f"lambda0 = {DEFAULT_LAMBDA0:.5f} / month "
          f"(median OS = {DEFAULT_MEDIAN_OS_MONTHS:.1f} months at X = 0)")
    tables = {}
    tables["table1_continuous_vs_dichotomized"] = run_part1()
    _section("Table 1 -- continuous vs. dichotomized power", tables["table1_continuous_vs_dichotomized"])
    part1b, details = run_part1b(return_details=True)
    tables["table2_calibrated_sample_size"] = part1b
    _section("Table 2 -- simulation-calibrated N (95% MC interval)", part1b)
    tables["fig2_collinearity_study"] = run_collinearity_study_both_cohorts()
    _section("Figure 2 data -- correlated adjustment covariates", tables["fig2_collinearity_study"])
    tables["table3_epv_stress_test"] = run_epv_stress_test()
    _section("Table 3 -- low events-per-variable", tables["table3_epv_stress_test"])
    tables["table4_heterogeneous_adjustment"] = run_heterogeneous_adjustment_study()
    _section("Table 4 -- prognostic adjustment covariates", tables["table4_heterogeneous_adjustment"])
    tables["table5_omitted_covariates"] = run_omitted_covariates_study()
    _section("Table 5 -- adjusted vs. unadjusted (omitted prognostic covariates)",
             tables["table5_omitted_covariates"])
    tables["table6_waning_effect"] = run_part2()
    _section("Table 6 -- waning (early-only) effect", tables["table6_waning_effect"])
    tables["table7_crossing_hazards"] = run_part3()
    _section("Table 7 -- crossing hazards", tables["table7_crossing_hazards"])

    plotting.plot_calibration_curves(details, n_sims_per_point=2000, outpath=out("fig1_calibration_curve.pdf"))
    plotting.plot_collinearity_power(tables["fig2_collinearity_study"], outpath=out("fig2_collinearity_power.pdf"))
    plotting.plot_epv_bias(tables["table3_epv_stress_test"], true_hr=1.10, outpath=out("fig3_epv.pdf"))
    plotting.plot_heterogeneous_adjustment(tables["table4_heterogeneous_adjustment"],
                                           outpath=out("fig4_informative_adjustment.pdf"))
    for name, df in tables.items():
        df.to_csv(out(f"{name}.csv"), index=False)
    print(f"\nTotal runtime: {time.time() - t0:.1f} s; outputs in {os.path.abspath(outdir)}")
    return tables


# --- Appendix B -----------------------------------------------------------------

C_LEVELS = (0.55, 0.60, 0.65, 0.70, 0.75)
ASSOC_N_GRID = (10, 15, 20, 30, 40, 60, 80, 120, 160, 250, 400)
BENCH_N_GRID = (30, 50, 75, 100, 150, 200, 250, 350, 500, 700)


def run_cindex_mapping(levels=C_LEVELS, censor_rate: float = 0.20,
                       lambda0: float = DEFAULT_LAMBDA0) -> pd.DataFrame:
    """Table 8: `concordance.mapping_table` for the report's HRs (1.10, 1.15)
    and target C levels, rounded as printed."""
    cs = [c_of_beta(np.log(1.10)), c_of_beta(np.log(1.15))] + list(levels)
    design = None
    if lambda0 != DEFAULT_LAMBDA0:
        from .design import StudyDesign
        design = StudyDesign(median_os=np.log(2.0) / lambda0, censor_rate=censor_rate)
    df = mapping_table(cs, censor_rate=censor_rate, design=design)
    return df.round(dict(C=4, beta=4, hr=3, lam_c=5, harrell_limit=4, D80=1))


def run_cindex_association_grid(mapping: pd.DataFrame, n_grid=ASSOC_N_GRID, n_sims: int = 1000,
                                lambda0: float = DEFAULT_LAMBDA0) -> pd.DataFrame:
    """Figure 6: power of the Cox Wald test and of the test of C = 0.5 vs. N
    for each target C (saturated cells skipped)."""
    rows = []
    for k, m in enumerate(mapping[mapping["C"].isin(C_LEVELS)].itertuples()):
        for j, n in enumerate(n_grid):
            if m.D80 / 0.8 * 6 < n and n > 40:
                continue
            r = simulate_cindex_power(n, m.beta, m.lam_c, n_sims=n_sims, lambda0=lambda0,
                                      seed=1000 * k + j)
            rows.append(dict(C=m.C, n=n, mean_events=r["mean_events"], power_cox=r["power_cox"],
                             power_c05=r["power_c05"], mean_c=round(r["mean_c"], 4)))
    return pd.DataFrame(rows)


def run_cindex_null(n_grid=(10, 15, 20, 30, 60, 120, 250), n_sims: int = 4000,
                    lambda0: float = DEFAULT_LAMBDA0) -> pd.DataFrame:
    """Table 9: empirical type-I error of both tests under beta = 0."""
    lam_c = calibrate_censoring_rate(0.0, 0.20, lambda0=lambda0)
    rows = []
    for j, n in enumerate(n_grid):
        r = simulate_cindex_power(n, 0.0, lam_c, n_sims=n_sims, lambda0=lambda0, seed=9000 + j)
        rows.append(dict(n=n, type1_cox=r["power_cox"], type1_c05=r["power_c05"]))
    return pd.DataFrame(rows)


def run_cindex_benchmark(c: float = 0.70, c0s=(0.60, 0.65), n_grid=BENCH_N_GRID,
                         n_sims: int = 1000, method: str = "harrell",
                         lambda0: float = DEFAULT_LAMBDA0) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Figure 7 / Table 10: power to show C > C0 vs. N, and the closed-form
    (nominal C and estimator limit) vs. simulation-calibrated N for 80% power."""
    b = beta_of_c(c)
    lam_c = calibrate_censoring_rate(b, 0.20, lambda0=lambda0)
    c_lim = harrell_limit(b, lam_c, lambda0)
    sig = sigma_c(b, lam_c, method, lambda0=lambda0)
    curve, table = [], []
    for c0 in c0s:
        for j, n in enumerate(n_grid):
            r = simulate_cindex_power(n, b, lam_c, n_sims=n_sims, c0=c0, method=method,
                                      lambda0=lambda0, seed=50_000 + 100 * int(c0 * 100) + j)
            curve.append(dict(c0=c0, n=n, power_cc0=r["power_cc0"], sd_c=r["sd_c"]))

        def power_fn(n, ns, s, _c0=c0):
            r = simulate_cindex_power(n, b, lam_c, n_sims=ns, c0=_c0, method=method,
                                      lambda0=lambda0, seed=s)
            return r["power_cc0"], r["mean_events"]

        n_nom = benchmark_sample_size(c, c0, sig)
        n_lim = benchmark_sample_size(c_lim, c0, sig)
        cal = calibrate_sample_size(power_fn, n_nom, 0.80, seed=30_000 + int(100 * c0))
        table.append(dict(c0=c0, n_closed_form_nominal=n_nom, n_closed_form_limit=n_lim,
                          n_calibrated=cal["n_calibrated"], confirmed_power=cal["confirmed_power"],
                          events=cal["confirmed_mean_events"]))
    return pd.DataFrame(curve), pd.DataFrame(table), dict(c_limit=c_lim, sigma_c=sig)


def reproduce_appendix_b(outdir: str = ".", n_sims: int = 1000) -> dict:
    """Regenerate Appendix B: Tables 8-10 (CSV) and Figures 5-7 (PDF).
    Use n_sims ~ 200 for a quick look (a few minutes at the default)."""
    os.makedirs(outdir, exist_ok=True)
    out = lambda name: os.path.join(outdir, name)
    t0 = time.time()
    tables = {"table8_concordance_mapping": run_cindex_mapping()}
    _section("Table 8 -- concordance <-> per-SD hazard ratio", tables["table8_concordance_mapping"])
    tables["fig6_association_grid"] = run_cindex_association_grid(tables["table8_concordance_mapping"], n_sims=n_sims)
    _section("Figure 6 data -- power to detect association", tables["fig6_association_grid"])
    tables["table9_type1_error"] = run_cindex_null(n_sims=4 * n_sims)
    _section("Table 9 -- type-I error under beta = 0", tables["table9_type1_error"])
    curve, table, info = run_cindex_benchmark(n_sims=n_sims)
    tables["fig7_benchmark_curve"], tables["table10_benchmark_n"] = curve, table
    _section(f"Table 10 -- N for C > C0 (C_H limit {info['c_limit']:.4f}, sigma_C "
             f"{info['sigma_c']:.3f})", table)

    m = tables["table8_concordance_mapping"]
    plotting.plot_concordance_mapping(m[m["C"].isin(C_LEVELS)], outpath=out("fig5_concordance_mapping.pdf"))
    plotting.plot_cindex_association_power(tables["fig6_association_grid"],
                                           outpath=out("fig6_association_power.pdf"))
    plotting.plot_benchmark_power(curve, info["c_limit"], info["sigma_c"],
                                  outpath=out("fig7_benchmark_power.pdf"))
    for name, df in tables.items():
        df.to_csv(out(f"{name}.csv"), index=False)
    print(f"\nTotal runtime: {time.time() - t0:.1f} s; outputs in {os.path.abspath(outdir)}")
    return tables
