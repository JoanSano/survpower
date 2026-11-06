"""Command-line interface. The question is the command:

    survpower power              --hr/--c-index and --n   -> power at that N
    survpower sample-size        --hr/--c-index           -> required N (per --power)
    survpower detectable-effect  --n                      -> minimum detectable HR and C
    survpower benchmark          --c-index and --c0       -> show C > C0: N, or power at --n
    survpower precision          --c-index and --half-width or --n
                                                          -> N for a CI half-width, or the
                                                             half-width at N

Each command only offers the flags its path reads; assumptions (censoring, covariates, ...)
are shared flag groups, never a command. The old flag-only syntax (`survpower --hr 1.15`) is
translated to a command with a warning until 0.4.1 (#66).
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import sys
import time

import pandas as pd

from . import __version__
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
DEFAULT_SIMS_SENSITIVITY = 300
REPORT_REMOVED = (
    "the report command was removed in 0.4.0; install survpower==0.2.x (tag v0.2.0) to "
    "reproduce the report, or see the Tutorials:\n"
    "  pip install \"git+https://github.com/JoanSano/survpower.git@v0.2.0\""
)
UNIVARIATE = ("concordance questions are univariate; to show added value over a clinical "
              "model, power the adjusted HR (survpower power or sample-size, with --hr, "
              "--n-covariates and --r2) instead")


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Show defaults in --help, except for options whose default is unset."""

    def _get_help_string(self, action):
        if action.default is None or action.default is False or action.default == []:
            return action.help
        return super()._get_help_string(action)


def _group(title: str, *args: tuple) -> argparse.ArgumentParser:
    """A parent parser holding one titled group of flags; each flag is (names, kwargs)."""
    parent = argparse.ArgumentParser(add_help=False)
    group = parent.add_argument_group(title)
    for names, kwargs in args:
        group.add_argument(*names, **kwargs)
    return parent


_HR = dict(type=float, nargs="+",
           help="hazard ratio(s) per SD of the biomarker (per raw unit with --biomarker-sd); "
                "<1 for a protective effect")
_C_ASSOCIATION = dict(type=float, nargs="+",
                      help="target concordance index(es) of the biomarker, in (0.5, 1); "
                           "converted to a per-SD HR")
_C_TARGET = dict(type=float, nargs="+", help="target concordance index(es), in (0.5, 1)")
_N = dict(type=int, nargs="+", help="sample size(s): patients ENROLLED")
_POWER = dict(type=float, nargs="+", default=None, help="target power(s) [default: 0.80]")


def _flag_groups() -> dict:
    """The shared flag groups, one parent parser each. The effect and N flags are not here,
    because which of them are required is what distinguishes the commands."""
    return dict(
        alpha=_group("test", (["--alpha"], dict(type=float, default=0.05,
                                                help="type I error rate"))),
        sides=_group("test", (["--sides"], dict(type=int, choices=(1, 2), default=2,
                                                help="one- or two-sided Wald test"))),
        population=_group(
            "population",
            (["--median-os"], dict(type=float, default=DEFAULT_MEDIAN_OS_MONTHS,
                                   help="baseline (biomarker at its mean) median overall "
                                        "survival, months")),
            (["--biomarker-sd"], dict(type=float, default=None,
                                      help="SD of the raw biomarker; if set, HRs are per raw "
                                           "unit (HR**SD per SD)")),
            (["--non-evaluable"], dict(type=float, default=0.0,
                                       help="fraction of enrolled patients without a usable "
                                            "biomarker value")),
        ),
        censoring=_group(
            "censoring",
            (["--censor-rate"], dict(type=float, default=None,
                                     help="marginal censoring probability [default: 0.20]; "
                                          "alternative to an accrual/follow-up design")),
            (["--accrual-months"], dict(type=float, default=None,
                                        help="uniform accrual period, months (requires "
                                             "--followup-months)")),
            (["--followup-months"], dict(type=float, default=None,
                                         help="minimum follow-up after the last patient is "
                                              "enrolled, months")),
            (["--dropout"], dict(type=float, default=0.0,
                                 help="annual loss-to-follow-up fraction (with the accrual "
                                      "design)")),
        ),
        adjustment=_group(
            "adjustment",
            (["--n-covariates"], dict(type=int, default=0,
                                      help="number of adjustment covariates in the Cox model")),
            (["--r2"], dict(type=float, default=0.0,
                            help="multiple R^2 of the biomarker on the adjustment covariates")),
        ),
        sensitivity=_group(
            "sensitivity analyses (at the given or solved N and HR)",
            (["--sensitivity"], dict(nargs="+", default=[], metavar="ANALYSIS",
                                     choices=SENSITIVITY_CHOICES + ("all",),
                                     help="any of: " + ", ".join(SENSITIVITY_CHOICES) + ", all")),
            (["--tau"], dict(type=float, default=DEFAULT_TAU_MONTHS,
                             help="change-point (months) for the waning / crossing scenarios")),
            (["--waning-hr"], dict(type=float, default=None,
                                   help="early (t < tau) HR for the waning scenario "
                                        "[default: planning HR]")),
            (["--crossing-hr"], dict(type=float, nargs=2, default=None, metavar=("EARLY", "LATE"),
                                     help="HRs before/after tau for the crossing scenario "
                                          "[default: 1/HR, HR]")),
            (["--n-sims-sensitivity"], dict(type=int, default=DEFAULT_SIMS_SENSITIVITY,
                                            help="replicates per scenario")),
        ),
        concordance=_group(
            "concordance estimator",
            (["--c-estimator"], dict(choices=ESTIMATORS, default="harrell",
                                     help="gonen-heller is model-based, unaffected by censoring")),
            (["--c-tau"], dict(type=float, default=None,
                               help="truncation horizon (months) for --c-estimator uno "
                                    "[default: 90th percentile of observed follow-up]")),
        ),
        simulation=_group(
            "simulation",
            (["--n-sims"], dict(type=int, default=1000,
                                help="replicates for power estimates and confirmation runs")),
            (["--seed"], dict(type=int, default=42, help="random seed")),
            (["--quick"], dict(action="store_true",
                               help="smoke test: 200 replicates, 100 per grid point or scenario")),
        ),
        engine=_group("simulation", (["--engine"], dict(choices=ENGINES, default="fast",
                                                        help="Cox fitter for univariate "
                                                             "simulations"))),
        grid=_group("simulation", (["--n-sims-grid"], dict(type=int, default=1000,
                                                           help="replicates per grid point when "
                                                                "solving"))),
        output=_group(
            "output",
            (["--outdir"], dict(default="survpower_output",
                                help="directory for CSV / PDF outputs")),
            (["--no-plots"], dict(action="store_true", help="skip figure generation")),
        ),
    )


def _add_effect(parser: argparse.ArgumentParser) -> None:
    """The effect scale: one mutually exclusive, required group."""
    group = parser.add_argument_group("effect")
    effect = group.add_mutually_exclusive_group(required=True)
    effect.add_argument("--hr", **_HR)
    effect.add_argument("--c-index", **_C_ASSOCIATION)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="survpower", formatter_class=_HelpFormatter,
        description="Power and sample-size planning for a continuous biomarker under the Cox "
                    "model. The command is the question; see `survpower COMMAND --help`.",
    )
    parser.add_argument("--version", action="version", version=f"survpower {__version__}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    g = _flag_groups()
    common = [g["population"], g["censoring"], g["simulation"]]
    association = [g["alpha"], g["sides"], *common, g["adjustment"], g["sensitivity"], g["engine"]]
    concordance = [*common, g["concordance"]]

    def add(name, summary, run, parents, examples):
        sub = commands.add_parser(
            name, help=summary.replace("%", "%%"), description=summary + ".", parents=parents,
            formatter_class=_HelpFormatter, allow_abbrev=False,
            epilog="examples:\n" + "\n".join(f"  survpower {name} {e}" for e in examples))
        sub.set_defaults(run=run, subparser=sub)
        return sub

    sub = add("power", "Power at a given N and effect", _run_power,
              [*association, g["output"]],
              ["--hr 1.15 --n 250 1200", "--hr 1.10 --n 1200 --n-covariates 5 --r2 0.3",
               "--c-index 0.60 --n 250", "--hr 1.30 --n 400 --sensitivity waning crossing"])
    _add_effect(sub)
    sub.add_argument_group("sample size").add_argument("--n", required=True, **_N)

    sub = add("sample-size", "Required N for a given effect", _run_sample_size,
              [*association, g["grid"], g["output"]],
              ["--hr 1.15 --power 0.8 0.9", "--c-index 0.55 0.60 0.70",
               "--hr 1.20 --n-covariates 5 --r2 0.3"])
    _add_effect(sub)
    sub.add_argument_group("target").add_argument("--power", **_POWER)

    sub = add("detectable-effect", "Minimum detectable HR and C at a given N", _run_detectable,
              [*association, g["grid"], g["output"]],
              ["--n 250", "--n 250 --accrual-months 36 --followup-months 12 --dropout 0.05"])
    target = sub.add_argument_group("sample size and target")
    target.add_argument("--n", required=True, **_N)
    target.add_argument("--power", **_POWER)

    sub = add("benchmark", "Show that C exceeds a benchmark C0: N, or power at --n "
                           "(one-sided test)", _run_benchmark,
              [g["alpha"], *concordance, g["grid"], g["output"]],
              ["--c-index 0.70 --c0 0.65", "--c-index 0.70 --c0 0.65 --n 200",
               "--c-index 0.70 --c0 0.65 --c-estimator uno --c-tau 24"])
    target = sub.add_argument_group("effect and benchmark")
    target.add_argument("--c-index", required=True, **_C_TARGET)
    target.add_argument("--c0", type=float, nargs="+", required=True,
                        help="benchmark concordance(s): one-sided test of H0: C <= C0")
    solve = sub.add_argument_group("solve for N (default) or for power")
    solve = solve.add_mutually_exclusive_group()
    solve.add_argument("--n", **_N)
    solve.add_argument("--power", **_POWER)

    sub = add("precision", "Precision of the estimated C: N for a 95% CI half-width, or the "
                           "half-width at --n", _run_precision,
              [*concordance, g["output"]],
              ["--c-index 0.70 --half-width 0.05", "--c-index 0.70 --n 200"])
    sub.add_argument_group("effect").add_argument("--c-index", required=True, **_C_TARGET)
    goal = sub.add_argument_group("target half-width or sample size")
    goal = goal.add_mutually_exclusive_group(required=True)
    goal.add_argument("--half-width", type=float, nargs="+",
                      help="target 95%% CI half-width(s) for the estimated C")
    goal.add_argument("--n", **_N)

    # a stub so the removal explains itself; #66 deletes it
    commands.add_parser("report", help="removed in 0.4.0", add_help=False)
    return parser


#: The option for each `StudyDesign` field, to report its errors in CLI terms.
DESIGN_FLAGS = {
    "alpha": "--alpha", "sides": "--sides", "median_os": "--median-os",
    "censor_rate": "--censor-rate", "accrual_months": "--accrual-months",
    "followup_months": "--followup-months", "dropout_annual": "--dropout",
    "non_evaluable": "--non-evaluable", "n_covariates": "--n-covariates", "r2": "--r2",
    "biomarker_sd": "--biomarker-sd", "c_estimator": "--c-estimator", "c_tau": "--c-tau",
    "engine": "--engine", "n_sims": "--n-sims", "n_sims_grid": "--n-sims-grid", "seed": "--seed",
}


def _build_design(args: argparse.Namespace) -> StudyDesign:
    """The design from the flags this command has; the rest keep `StudyDesign`'s defaults."""
    dests = {field: flag[2:].replace("-", "_") for field, flag in DESIGN_FLAGS.items()}
    try:
        return StudyDesign(**{field: getattr(args, dest) for field, dest in dests.items()
                              if hasattr(args, dest)})
    except ValueError as exc:
        fields = r"\b(" + "|".join(DESIGN_FLAGS) + r")\b"
        args.subparser.error(re.sub(fields, lambda m: DESIGN_FLAGS[m.group(1)], str(exc)))


def _inert_flags(args: argparse.Namespace) -> dict:
    """Flags a command offers that this particular run would not read, with why. An explicit
    default value is not detected, and does no harm."""
    which = getattr(args, "sensitivity", [])
    every = "all" in which
    inert = {}
    if getattr(args, "c_tau", None) is not None and args.c_estimator != "uno":
        inert["--c-tau"] = "it only applies to --c-estimator uno"
    if args.command == "benchmark" and args.n and getattr(args, "n_sims_grid", 1000) != 1000:
        inert["--n-sims-grid"] = "nothing is solved for when --n is given"
    if getattr(args, "tau", DEFAULT_TAU_MONTHS) != DEFAULT_TAU_MONTHS \
            and not (every or {"waning", "crossing"} & set(which)):
        inert["--tau"] = "it only applies to --sensitivity waning or crossing"
    if getattr(args, "waning_hr", None) is not None and not (every or "waning" in which):
        inert["--waning-hr"] = "it only applies to --sensitivity waning"
    if getattr(args, "crossing_hr", None) is not None and not (every or "crossing" in which):
        inert["--crossing-hr"] = "it only applies to --sensitivity crossing"
    if getattr(args, "n_sims_sensitivity", DEFAULT_SIMS_SENSITIVITY) != DEFAULT_SIMS_SENSITIVITY \
            and not which:
        inert["--n-sims-sensitivity"] = "it only applies with --sensitivity"
    return inert


def _validate(args: argparse.Namespace) -> StudyDesign:
    """Check what `StudyDesign` cannot (the effect, N and power) and return the design."""
    fail = args.subparser.error
    if args.hr and any(h <= 0 or h == 1 for h in args.hr):
        fail("--hr values must be positive and different from 1")
    if args.c_index and any(not 0.5 < c < 1 for c in args.c_index):
        fail("--c-index values must be in (0.5, 1)")
    if args.c0 and any(c0 >= c for c0 in args.c0 for c in args.c_index):
        fail("every --c0 must be below every --c-index")
    if args.half_width and any(not 0 < h < 0.5 for h in args.half_width):
        fail("--half-width values must be in (0, 0.5)")
    if args.n and any(n < 10 for n in args.n):
        fail("--n values must be at least 10")
    design = _build_design(args)
    if args.power and any(not design.alpha < p < 1 for p in args.power):
        fail("--power values must be in (alpha, 1)")
    if args.n and any(design.evaluable(n) < 10 for n in args.n):
        fail("fewer than 10 evaluable patients after --non-evaluable")
    return design


def _print_assumptions(design: StudyDesign, command: str) -> None:
    print("=" * 70 + "\nDesign assumptions\n" + "=" * 70)
    print(f"  baseline median OS : {design.median_os:g} months "
          f"(lambda0 = {design.lambda0:.5f} / month; constant baseline hazard)")
    scale = "SD" if design.biomarker_sd is None else f"raw unit (SD = {design.biomarker_sd:g})"
    print(f"  biomarker          : continuous, ~Normal; effects as HR per {scale}")
    if command == "benchmark":
        print(f"  test               : one-sided test at alpha = {design.alpha:g}")
    elif command != "precision":
        print(f"  test               : {design.test_label()}")
    if design.accrual_months is None:
        print(f"  censoring          : {design.censor_rate_for(1.0):.0%} (fixed)")
    else:
        print(f"  censoring          : from design -- {design.accrual_months:g} mo accrual + "
              f"{design.followup_months:g} mo follow-up, {design.dropout_annual:.0%}/yr dropout\n"
              f"                       the censored fraction depends on the HR; each result "
              f"states the one it used")
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


def _save(args: argparse.Namespace, df: pd.DataFrame, name: str) -> None:
    df.to_csv(os.path.join(args.outdir, f"{name}.csv"), index=False)


def _hrs(args: argparse.Namespace, design: StudyDesign) -> list:
    """The planning HRs: a target concordance is just another way to give the HR."""
    return args.hr or [design.to_per_unit(hr_of_c(c)) for c in args.c_index]


# Each question returns its planning points (evaluable N, per-SD HR, file tag), which the
# sensitivity analyses run at.

def _run_power(args: argparse.Namespace, design: StudyDesign) -> list:
    df = run_power_mode(design, _hrs(args, design), args.n)
    _print_table("POWER at the given N and effect", df)
    _save(args, df, "power_analysis")
    return [(r.n_evaluable, r.hr_per_sd, f"n{r.n_enrolled}_hr{r.hr_per_sd:g}")
            for r in df.itertuples()]


def _run_sample_size(args: argparse.Namespace, design: StudyDesign) -> list:
    powers = args.power or [0.80]
    df, details = run_sample_size_mode(design, _hrs(args, design), powers)
    _print_table("REQUIRED SAMPLE SIZE for the given effect", df)
    _save(args, df, "sample_size")
    if not args.no_plots:
        from .plotting import plot_calibration_curves
        for pw in powers:
            sel = [d for d in details if d["target_power"] == pw and d["method"] == "probit"]
            if sel:
                plot_calibration_curves(
                    sel, target_power=pw, n_sims_per_point=design.n_sims_grid,
                    outpath=os.path.join(args.outdir, f"calibration_curve_power{pw:g}.pdf"))
    return [(r.n_calibrated_evaluable, r.hr_per_sd, f"hr{r.hr_per_sd:g}_power{r.target_power:g}")
            for r in df.itertuples()]


def _run_detectable(args: argparse.Namespace, design: StudyDesign) -> list:
    df = run_detectable_effect_mode(design, args.n, args.power or [0.80])
    _print_table("MINIMUM DETECTABLE EFFECT at the given N", df)
    _save(args, df, "detectable_effect")
    return [(r.n_evaluable, r.hr_detectable_per_sd, f"n{r.n_enrolled}_power{r.target_power:g}")
            for r in df.itertuples()]


def _run_benchmark(args: argparse.Namespace, design: StudyDesign) -> list:
    df = run_cindex_benchmark_mode(design, args.c_index, args.c0, ns=args.n,
                                   powers=args.power or [0.80])
    _print_table("CONCORDANCE vs BENCHMARK (H0: C <= C0, one-sided)", df)
    _save(args, df, "cindex_benchmark")
    return []


def _run_precision(args: argparse.Namespace, design: StudyDesign) -> list:
    df = run_cindex_precision_mode(design, args.c_index, half_widths=args.half_width, ns=args.n)
    _print_table("PRECISION of the estimated concordance", df)
    _save(args, df, "cindex_precision")
    return []


def _run_sensitivity(args: argparse.Namespace, design: StudyDesign, planning_points: list) -> None:
    which = list(SENSITIVITY_CHOICES) if "all" in args.sensitivity else args.sensitivity
    if "crossing" in which and design.sides == 1:
        print("note: the crossing-hazards scenario has no single effect direction and "
              "always uses a two-sided test\n")
    crossing_hr = tuple(design.to_per_sd(h) for h in args.crossing_hr) if args.crossing_hr else None
    waning_hr = design.to_per_sd(args.waning_hr) if args.waning_hr else None
    for n_eval, hr_sd, tag in planning_points:
        results = run_sensitivity(
            design, n=int(n_eval), hr=float(hr_sd), which=which, tau=args.tau,
            waning_hr=waning_hr, crossing_hr=crossing_hr, n_sims=args.n_sims_sensitivity,
            outdir=args.outdir, tag=tag, plots=not args.no_plots,
        )
        for name, res in results.items():
            _print_table(f"SENSITIVITY: {name}  (evaluable N = {int(n_eval)}, "
                         f"HR per SD = {hr_sd:.3g})", res)
            _save(args, res, f"sensitivity_{name}_{tag}")


# --- The flag-only syntax, until 0.4.1 (#66) --------------------------------------------

#: Flags the old syntax rejected on some questions rather than ignored. They are passed on,
#: so that the new command rejects them as well.
_LEGACY_ALWAYS_PASSED = {"hr", "c_index", "n_covariates", "r2"}
_LEGACY_QUESTION_FLAGS = {"hr", "c_index", "c0", "precision", "n", "power"}


def _translate_legacy(argv: list) -> list:
    """The command that `survpower --flags` means, with the flags it ignored removed."""
    groups = _flag_groups()
    legacy = argparse.ArgumentParser(prog="survpower", parents=list(groups.values()))
    effect = legacy.add_mutually_exclusive_group()
    effect.add_argument("--hr", **_HR)
    effect.add_argument("--c-index", **_C_ASSOCIATION)
    legacy.add_argument("--c0", type=float, nargs="+")
    legacy.add_argument("--precision", type=float, nargs="+")
    legacy.add_argument("--n", **_N)
    legacy.add_argument("--power", **_POWER)
    old = legacy.parse_args(argv)

    if old.c0 and old.precision:
        legacy.error("give either --c0 (benchmark) or --precision, not both")
    if old.c0:
        old.command = "benchmark"
    elif old.precision:
        old.command = "precision"
    elif (old.hr or old.c_index) and old.n:
        old.command = "power"
    elif old.hr or old.c_index:
        old.command = "sample-size"
    elif old.n:
        old.command = "detectable-effect"
    else:
        legacy.error("give --hr or --c-index, --n, or both")

    commands = next(a for a in build_parser()._actions
                    if isinstance(a, argparse._SubParsersAction)).choices
    accepted = {a.dest for a in commands[old.command]._actions}
    inert = _inert_flags(old)
    if old.command == "benchmark" and old.n and old.power:
        inert["--power"] = "--n wins"  # the new command makes the two exclusive
    new, dropped = [old.command], []
    # the question's flags first (`sorted` is stable), so the suggested command reads naturally
    for action in sorted(legacy._actions, key=lambda a: a.dest not in _LEGACY_QUESTION_FLAGS):
        value = getattr(old, action.dest, None)
        if action.dest == "help" or value == action.default:
            continue
        flag = action.option_strings[0]
        if action.dest == "precision" and not old.n:
            flag = "--half-width"  # with --n, --n wins and the half-width was dropped
        elif (flag in inert or action.dest not in accepted) \
                and action.dest not in _LEGACY_ALWAYS_PASSED:
            dropped.append(flag)
            continue
        new.append(flag)
        if value is not True:
            new.extend(str(v) for v in (value if isinstance(value, list) else [value]))

    print("warning: this syntax is deprecated and will be removed in 0.4.1. Use:\n"
          f"  survpower {shlex.join(new)}", file=sys.stderr)
    if dropped:
        print(f"  dropped, because they had no effect here: {', '.join(dropped)}", file=sys.stderr)
    return new


def main(argv: list | None = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0].startswith("--") and argv[0] not in ("--help", "--version"):
        argv = _translate_legacy(argv)
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    if args.command is None:
        if unknown:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
        parser.print_help()
        sys.exit(2)
    if args.command == "report":
        parser.exit(2, f"survpower: error: {REPORT_REMOVED}\n")
    if unknown:
        if args.command in ("benchmark", "precision") and {"--n-covariates", "--r2"} & set(unknown):
            args.subparser.error(UNIVARIATE)
        args.subparser.error(f"this command does not take: {' '.join(unknown)}")

    for name in ("hr", "c_index", "n", "power", "c0", "half_width"):
        if not hasattr(args, name):
            setattr(args, name, None)
    for flag, why in _inert_flags(args).items():
        args.subparser.error(f"{flag} would have no effect: {why}")
    if args.quick:  # only override replicate counts this command has, left at their defaults
        for name, value in QUICK_SIMS.items():
            if hasattr(args, name) and getattr(args, name) == args.subparser.get_default(name):
                setattr(args, name, value)
    design = _validate(args)
    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.time()
    _print_assumptions(design, args.command)
    planning_points = args.run(args, design)
    if getattr(args, "sensitivity", None):
        _run_sensitivity(args, design, planning_points)
    print(f"Outputs written to {os.path.abspath(args.outdir)}  (runtime {time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
