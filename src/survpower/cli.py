"""Command-line interface.

    survpower [planning options]      plan a study (see `survpower --help`)
    survpower report [--appendix-b]   reproduce the report (deprecated, removed in 0.4.0)

What is solved for is inferred from what you give:

    --hr/--c-index and --n   -> power at that N
    --hr/--c-index only      -> required N (per --power)
    --n only                 -> minimum detectable HR and C
    --c-index with --c0      -> benchmark question: show C > C0
    --c-index with --precision, or --precision with --n -> precision of C
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

from .baseline import DEFAULT_MEDIAN_OS_MONTHS, DEFAULT_TAU_MONTHS
from .concordance import ESTIMATORS, hr_of_c
from .design import StudyDesign
from .fitting import ENGINES
from .planning import (
    run_cindex_benchmark_mode,
    run_cindex_precision_mode,
    run_detectable_effect_mode,
    run_power_mode,
    run_sample_size_mode,
)
from .sensitivity import SENSITIVITY_CHOICES, run_sensitivity

QUICK_SIMS = {"n_sims": 200, "n_sims_grid": 100, "n_sims_sensitivity": 100}
REPORT_DEPRECATION = (
    "warning: `survpower report` is deprecated and will be removed in 0.4.0. To reproduce "
    "the report's numbers, install the v0.2.0 tag:\n"
    "  pip install \"git+https://github.com/JoanSano/survpower.git@v0.2.0\"\n"
    "See https://joansano.github.io/survpower/main/reproducing-the-report/"
)


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Show defaults in --help, except for options whose default is unset."""

    def _get_help_string(self, action):
        if action.default is None or action.default is False or action.default == []:
            return action.help
        return super()._get_help_string(action)


EPILOG = """examples:
  survpower --hr 1.15 --power 0.8 0.9                 required N
  survpower --n 250 --accrual-months 36 --followup-months 12 --dropout 0.05
                                                     minimum detectable HR / C
  survpower --hr 1.10 1.15 --n 250 1200 --n-covariates 5 --r2 0.3
                                                     power, adjusted model
  survpower --c-index 0.70 --c0 0.65                 N to show C > 0.65
  survpower --c-index 0.70 --precision 0.05          N for a +/-0.05 CI on C
  survpower report --appendix-b                      reproduce Appendix B (deprecated)
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="survpower",
        description=("Power and sample-size planning for a continuous biomarker under the Cox "
                     "model. Give an effect size (--hr or --c-index), a sample size (--n), or "
                     "both; the missing quantity is solved for."),
        epilog=EPILOG, formatter_class=_HelpFormatter,
    )

    g = parser.add_argument_group("effect size and sample size")
    eff = g.add_mutually_exclusive_group()
    eff.add_argument("--hr", type=float, nargs="+",
                     help="hazard ratio(s) per SD of the biomarker (per raw unit with "
                          "--biomarker-sd); <1 for a protective effect")
    eff.add_argument("--c-index", type=float, nargs="+",
                     help="target concordance index(es) of the biomarker, in (0.5, 1); "
                          "converted to a per-SD HR for association questions")
    g.add_argument("--n", type=int, nargs="+", help="sample size(s): patients ENROLLED")
    g.add_argument("--power", type=float, nargs="+", default=None,
                   help="target power(s) when solving for N or the effect [default: 0.80]")
    g.add_argument("--alpha", type=float, default=0.05, help="type I error rate")
    g.add_argument("--sides", type=int, choices=(1, 2), default=2,
                   help="one- or two-sided Wald test (benchmark tests are always one-sided)")
    g.add_argument("--biomarker-sd", type=float, default=None,
                   help="SD of the raw biomarker; if set, --hr is per raw unit (HR**SD per SD)")

    c = parser.add_argument_group("concordance questions")
    c.add_argument("--c0", type=float, nargs="+",
                   help="benchmark concordance(s): power / N to show C > C0 (one-sided)")
    c.add_argument("--precision", type=float, nargs="+",
                   help="target 95%% CI half-width(s) for the estimated C")
    c.add_argument("--c-estimator", choices=ESTIMATORS, default="harrell",
                   help="concordance estimator for benchmark / precision questions")
    c.add_argument("--c-tau", type=float, default=None,
                   help="truncation horizon (months) for Uno's C [default: 90th percentile "
                        "of observed follow-up]")

    s = parser.add_argument_group("survival, follow-up and attrition")
    s.add_argument("--median-os", type=float, default=DEFAULT_MEDIAN_OS_MONTHS,
                   help="baseline (biomarker at its mean) median overall survival, months")
    s.add_argument("--censor-rate", type=float, default=None,
                   help="marginal censoring probability [default: 0.20]; alternative to an "
                        "accrual/follow-up design")
    s.add_argument("--accrual-months", type=float, default=None,
                   help="uniform accrual period, months (requires --followup-months)")
    s.add_argument("--followup-months", type=float, default=None,
                   help="minimum follow-up after the last patient is enrolled, months")
    s.add_argument("--dropout", type=float, default=0.0,
                   help="annual loss-to-follow-up fraction (with the accrual design)")
    s.add_argument("--non-evaluable", type=float, default=0.0,
                   help="fraction of enrolled patients without a usable biomarker value")

    a = parser.add_argument_group("analysis model")
    a.add_argument("--n-covariates", type=int, default=0,
                   help="number of adjustment covariates in the Cox model")
    a.add_argument("--r2", type=float, default=0.0,
                   help="multiple R^2 of the biomarker on the adjustment covariates")

    v = parser.add_argument_group("sensitivity analyses (at the given or solved N / HR)")
    v.add_argument("--sensitivity", nargs="+", default=[], metavar="ANALYSIS",
                   choices=SENSITIVITY_CHOICES + ("all",),
                   help="any of: " + ", ".join(SENSITIVITY_CHOICES) + ", all")
    v.add_argument("--tau", type=float, default=DEFAULT_TAU_MONTHS,
                   help="change-point (months) for the waning / crossing scenarios")
    v.add_argument("--waning-hr", type=float, default=None,
                   help="early (t < tau) HR for the waning scenario [default: planning HR]")
    v.add_argument("--crossing-hr", type=float, nargs=2, default=None, metavar=("EARLY", "LATE"),
                   help="HRs before/after tau for the crossing scenario [default: 1/HR, HR]")

    m = parser.add_argument_group("simulation and output")
    m.add_argument("--engine", choices=ENGINES, default="fast",
                   help="Cox fitter for univariate simulations")
    m.add_argument("--n-sims", type=int, default=1000,
                   help="replicates for power estimates and confirmation runs")
    m.add_argument("--n-sims-grid", type=int, default=1000,
                   help="replicates per grid point when solving for N or the effect")
    m.add_argument("--n-sims-sensitivity", type=int, default=300,
                   help="replicates per scenario in sensitivity analyses")
    m.add_argument("--quick", action="store_true", help="smoke test: 200 / 100 / 100 replicates")
    m.add_argument("--seed", type=int, default=42, help="random seed")
    m.add_argument("--outdir", default="survpower_output", help="directory for CSV / PDF outputs")
    m.add_argument("--no-plots", action="store_true", help="skip figure generation")
    return parser


def build_report_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="survpower report", formatter_class=_HelpFormatter,
        description="Reproduce the report's tables (CSV) and figures (PDF). Deprecated: "
                    "removed in 0.4.0; install the v0.2.0 tag for exact reproduction.")
    parser.add_argument("--appendix-b", action="store_true",
                        help="reproduce Appendix B (concordance index) instead of the main report")
    parser.add_argument("--n-sims", type=int, default=1000,
                        help="replicates per cell (Appendix B only)")
    parser.add_argument("--outdir", default="survpower_report", help="output directory")
    return parser


def _validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    effect = args.hr or args.c_index
    if not effect and not args.n:
        parser.error("give --hr or --c-index, --n, or both")
    if args.hr and any(h <= 0 or h == 1 for h in args.hr):
        parser.error("--hr values must be positive and different from 1")
    if args.c_index and any(not 0.5 < c < 1 for c in args.c_index):
        parser.error("--c-index values must be in (0.5, 1)")
    if (args.c0 or args.precision) and args.hr:
        parser.error("--c0 / --precision are concordance questions: give --c-index, not --hr")
    if args.c0 and not args.c_index:
        parser.error("--c0 requires --c-index")
    if args.c0 and args.precision:
        parser.error("give either --c0 (benchmark) or --precision, not both")
    if args.precision and not args.c_index:
        parser.error("--precision requires --c-index")
    if args.c0 and any(c0 >= c for c0 in args.c0 for c in args.c_index):
        parser.error("every --c0 must be below every --c-index")
    if args.precision and any(not 0 < h < 0.5 for h in args.precision):
        parser.error("--precision half-widths must be in (0, 0.5)")
    if (args.c0 or args.precision) and args.n_covariates:
        parser.error("concordance questions are univariate; to show added value over a clinical "
                     "model, power the adjusted HR (--hr with --n-covariates/--r2) instead")
    if args.n and any(n < 10 for n in args.n):
        parser.error("--n values must be at least 10")
    if not 0 < args.alpha < 0.5:
        parser.error("--alpha must be in (0, 0.5)")
    if args.power and any(not args.alpha < p < 1 for p in args.power):
        parser.error("--power values must be in (alpha, 1)")
    if (args.accrual_months is None) != (args.followup_months is None):
        parser.error("--accrual-months and --followup-months must be given together")
    if args.accrual_months is not None and args.censor_rate is not None:
        parser.error("give either --censor-rate or an accrual/follow-up design, not both")
    if args.accrual_months is not None and (args.accrual_months < 0 or args.followup_months < 0):
        parser.error("--accrual-months and --followup-months must be non-negative")
    if args.dropout and args.accrual_months is None:
        parser.error("--dropout requires --accrual-months/--followup-months "
                     "(with --censor-rate, include dropout in the censoring rate)")
    if args.censor_rate is not None and not 0 <= args.censor_rate < 1:
        parser.error("--censor-rate must be in [0, 1)")
    for name in ("dropout", "non_evaluable", "r2"):
        if not 0 <= getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1)")
    if args.r2 > 0 and args.n_covariates == 0:
        parser.error("--r2 requires --n-covariates >= 1")
    if args.biomarker_sd is not None and args.biomarker_sd <= 0:
        parser.error("--biomarker-sd must be positive")
    if args.n and any(np.floor(n * (1 - args.non_evaluable)) < 10 for n in args.n):
        parser.error("fewer than 10 evaluable patients after --non-evaluable")
    if args.power and effect and args.n and not (args.c0 or args.precision):
        print("note: --power is ignored when both the effect and --n are given (power is the output)")


def _print_assumptions(design: StudyDesign) -> None:
    print("=" * 70 + "\nDesign assumptions\n" + "=" * 70)
    print(f"  baseline median OS : {design.median_os:g} months "
          f"(lambda0 = {design.lambda0:.5f} / month; constant baseline hazard)")
    scale = "SD" if design.biomarker_sd is None else f"raw unit (SD = {design.biomarker_sd:g})"
    print(f"  biomarker          : continuous, ~Normal; effects as HR per {scale}")
    print(f"  test               : {design.test_label()}")
    if design.accrual_months is None:
        print(f"  censoring          : {design.censor_rate_for(1.0):.0%} (fixed)")
    else:
        print(f"  censoring          : from design -- {design.accrual_months:g} mo accrual + "
              f"{design.followup_months:g} mo follow-up, {design.dropout_annual:.0%}/yr dropout "
              f"(~{design.censor_rate_for(1.0):.0%} censored at HR = 1; it depends on the HR, "
              f"see the censor_rate column for the value used)")
    print(f"  non-evaluable      : {design.non_evaluable:.0%} of enrolled")
    print("  adjustment         : " + ("none" if design.n_covariates == 0 else
                                       f"{design.n_covariates} covariate(s), R^2 = {design.r2:g}"))
    print()


def _print_table(title: str, df: pd.DataFrame) -> None:
    print("=" * 70 + f"\n{title}\n" + "=" * 70)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(df.drop(columns="summary", errors="ignore").to_string(index=False))
    if "summary" in df:
        print("\nSummary (for protocol / grant text):")
        for s in df["summary"]:
            print(f"  - {s}")
    print()


def _plan(argv: list) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate(parser, args)
    if args.quick:  # only override replicate counts left at their defaults
        for name, value in QUICK_SIMS.items():
            if getattr(args, name) == parser.get_default(name):
                setattr(args, name, value)

    design = StudyDesign(
        alpha=args.alpha, sides=args.sides, median_os=args.median_os,
        censor_rate=args.censor_rate, accrual_months=args.accrual_months,
        followup_months=args.followup_months, dropout_annual=args.dropout,
        non_evaluable=args.non_evaluable, n_covariates=args.n_covariates, r2=args.r2,
        biomarker_sd=args.biomarker_sd, c_estimator=args.c_estimator, c_tau=args.c_tau,
        engine=args.engine, n_sims=args.n_sims, n_sims_grid=args.n_sims_grid, seed=args.seed,
    )
    powers = args.power or [0.80]
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    _print_assumptions(design)

    def save(df, name):
        df.to_csv(os.path.join(outdir, f"{name}.csv"), index=False)

    if args.c0:
        df = run_cindex_benchmark_mode(design, args.c_index, args.c0, ns=args.n, powers=powers)
        _print_table("CONCORDANCE vs BENCHMARK (H0: C <= C0, one-sided)", df)
        save(df, "cindex_benchmark")
        planning_points = []
    elif args.precision:
        df = run_cindex_precision_mode(design, args.c_index, half_widths=args.precision, ns=args.n)
        _print_table("PRECISION of the estimated concordance", df)
        save(df, "cindex_precision")
        planning_points = []
    else:
        # association: a target concordance is just another way to give the HR
        if args.hr:
            hrs = args.hr
        elif args.c_index:
            hrs = [design.to_per_unit(hr_of_c(c)) for c in args.c_index]
        else:
            hrs = None
        if hrs and args.n:
            df = run_power_mode(design, hrs, args.n)
            _print_table("POWER at the given N and effect", df)
            save(df, "power_analysis")
            planning_points = [(r.n_evaluable, r.hr_per_sd, f"n{r.n_enrolled}_hr{r.hr_per_sd:g}")
                               for r in df.itertuples()]
        elif hrs:
            df, details = run_sample_size_mode(design, hrs, powers)
            _print_table("REQUIRED SAMPLE SIZE for the given effect", df)
            save(df, "sample_size")
            if not args.no_plots:
                from .plotting import plot_calibration_curves
                for pw in powers:
                    sel = [d for d in details if d["target_power"] == pw and d["method"] == "probit"]
                    if sel:
                        plot_calibration_curves(
                            sel, target_power=pw, n_sims_per_point=design.n_sims_grid,
                            outpath=os.path.join(outdir, f"calibration_curve_power{pw:g}.pdf"))
            planning_points = [(r.n_calibrated_evaluable, r.hr_per_sd,
                                f"hr{r.hr_per_sd:g}_power{r.target_power:g}") for r in df.itertuples()]
        else:
            df = run_detectable_effect_mode(design, args.n, powers)
            _print_table("MINIMUM DETECTABLE EFFECT at the given N", df)
            save(df, "detectable_effect")
            planning_points = [(r.n_evaluable, r.hr_detectable_per_sd,
                                f"n{r.n_enrolled}_power{r.target_power:g}") for r in df.itertuples()]

    which = list(SENSITIVITY_CHOICES) if "all" in args.sensitivity else args.sensitivity
    if which and not planning_points:
        print("note: sensitivity analyses apply to hazard-ratio (association) questions; skipped\n")
    if which and planning_points:
        if "crossing" in which and design.sides == 1:
            print("note: the crossing-hazards scenario has no single effect direction and "
                  "always uses a two-sided test\n")
        crossing_hr = tuple(design.to_per_sd(h) for h in args.crossing_hr) if args.crossing_hr else None
        waning_hr = design.to_per_sd(args.waning_hr) if args.waning_hr else None
        for n_eval, hr_sd, tag in planning_points:
            results = run_sensitivity(
                design, n=int(n_eval), hr=float(hr_sd), which=which, tau=args.tau,
                waning_hr=waning_hr, crossing_hr=crossing_hr, n_sims=args.n_sims_sensitivity,
                outdir=outdir, tag=tag, plots=not args.no_plots,
            )
            for name, res in results.items():
                _print_table(f"SENSITIVITY: {name}  (evaluable N = {int(n_eval)}, "
                             f"HR per SD = {hr_sd:.3g})", res)
                save(res, f"sensitivity_{name}_{tag}")

    print(f"Outputs written to {os.path.abspath(outdir)}  (runtime {time.time() - t0:.1f} s)")


def main(argv: list | None = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "report":
        args = build_report_parser().parse_args(argv[1:])
        print(REPORT_DEPRECATION, file=sys.stderr)
        with warnings.catch_warnings():  # the notice above already says it
            warnings.simplefilter("ignore", DeprecationWarning)
            from .report import reproduce_appendix_b, reproduce_report
        if args.appendix_b:
            reproduce_appendix_b(args.outdir, n_sims=args.n_sims)
        else:
            reproduce_report(args.outdir)
        return
    _plan(argv)


if __name__ == "__main__":
    main()
