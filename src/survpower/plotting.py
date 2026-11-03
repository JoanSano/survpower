"""Figures (matplotlib). Each function takes the DataFrames / dicts returned
by the corresponding analysis function -- no hard-coded numbers -- and saves
a figure (PDF or any matplotlib format, by extension)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # file output only; safe headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from .concordance import c_of_beta

_PLOT_STYLE = {
    "font.size": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
}
PALETTE = ["#2c5f8a", "#b45a28", "#6a3d9a", "#2e8b57", "#6b8e23"]


def plot_calibration_curves(results: list, target_power: float = 0.80,
                            n_sims_per_point: int = 300,
                            outpath: str = "calibration_curve.pdf") -> None:
    """Simulated power on the probit scale vs. sqrt(N) with the WLS
    calibration line, one panel per HR (report Figure 1). `results` are the
    dicts from `estimate_required_n_by_simulation`."""
    plt.rcParams.update(_PLOT_STYLE)
    fig, axes = plt.subplots(1, len(results), figsize=(4.7 * len(results), 4.0))
    axes = np.atleast_1d(axes)
    for ax, res in zip(axes, results):
        grid_n = np.array([g["n"] for g in res["grid"]])
        grid_power = np.array([g["power_hat"] for g in res["grid"]])
        sqrt_n = np.sqrt(grid_n)
        p_tilde = (grid_power * n_sims_per_point + 0.5) / (n_sims_per_point + 1)
        ax.scatter(sqrt_n, norm.ppf(p_tilde), color=PALETTE[0], zorder=3, s=42,
                   label="simulated (probit scale)")
        xs = np.linspace(sqrt_n.min() * 0.92, sqrt_n.max() * 1.05, 100)
        ax.plot(xs, res["a_hat"] * xs + res["b_hat"], color=PALETTE[0], lw=1.4, alpha=0.85,
                label=r"WLS fit: $\hat a\sqrt{N}+\hat b$")
        z_target = norm.ppf(target_power)
        ax.axhline(z_target, color=PALETTE[1], lw=1.1, ls="--", alpha=0.8,
                   label=f"target power = {target_power:.2f}")
        sqrt_n_star = np.sqrt(res["n_calibrated"])
        ax.axvline(sqrt_n_star, color=PALETTE[1], lw=1.1, ls="--", alpha=0.8)
        ax.scatter([sqrt_n_star], [z_target], color=PALETTE[1], zorder=4, s=55,
                   marker="D", label=fr"$N^*={res['n_calibrated']}$")
        ax.set_xlabel(r"$\sqrt{N}$")
        ax.set_ylabel(r"$\Phi^{-1}(\widehat{\mathrm{Power}})$")
        ax.set_title(f"HR = {res['hr']:.3g}  (closed-form $N_0$={res['n_closed_form']})")
        ax.legend(fontsize=8, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_collinearity_power(df: pd.DataFrame, outpath: str = "collinearity_power.pdf") -> None:
    """Power vs. number of adjustment covariates p, by R^2, one panel per N
    (report Figure 2). `df` from `run_collinearity_study(_both_cohorts)`."""
    plt.rcParams.update(_PLOT_STYLE)
    r2_levels = sorted(df["r2_target"].unique())
    n_levels = sorted(df["n"].unique(), reverse=True)
    colors = dict(zip(r2_levels, PALETTE))
    fig, axes = plt.subplots(1, len(n_levels), figsize=(4.8 * len(n_levels), 4.2))
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, n_levels):
        sub_n = df[df["n"] == n]
        for r2 in r2_levels:
            sub = sub_n[sub_n["r2_target"] == r2].sort_values("p")
            ax.plot(sub["p"], sub["power"], marker="o", color=colors[r2], lw=1.3, ms=5,
                    label=fr"$R^2={r2}$", zorder=3)
            ax.axhline(sub["theoretical_power"].iloc[0], color=colors[r2], lw=1.0, ls="--",
                       alpha=0.5, zorder=1)
        ax.set_xlabel("number of adjustment covariates $p$")
        ax.set_ylabel("power")
        ax.set_title(f"$N={n}$, mean events $\\approx${sub_n['mean_events'].mean():.0f}")
        ax.set_xticks(sorted(df["p"].unique()))
        ax.set_ylim(0, 1.0)
    axes[0].legend(fontsize=8.5, loc="lower left", frameon=True, facecolor="white",
                   framealpha=0.92, edgecolor="none",
                   title="solid = simulated\ndashed = closed form", title_fontsize=7.5)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_epv_bias(df: pd.DataFrame, true_hr: float, outpath: str = "epv_bias.pdf") -> None:
    """Bias of the recovered HR and SD of the log-HR vs. p, with EPV on a
    twin axis (report Figure 3). `df` from `run_epv_stress_test`."""
    plt.rcParams.update(_PLOT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    epv_labels = [f"{e:.0f}" if e >= 10 else f"{e:.1f}" for e in df["epv"].values]
    panels = [("mean_HR_hat", PALETTE[1], "mean recovered HR for $X$", "Bias in the recovered effect"),
              ("sd_beta_hat", PALETTE[0], r"SD of $\hat\beta_X$ across replicates",
               "Growing estimation variance")]
    for ax, (col, color, ylabel, title) in zip(axes, panels):
        ax.plot(df["p"], df[col], marker="o", color=color, lw=1.4, ms=5)
        ax.set_xlabel("number of adjustment covariates $p$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        twin = ax.twiny()
        twin.set_xlim(ax.get_xlim())
        twin.set_xticks(df["p"].values)
        twin.set_xticklabels(epv_labels, fontsize=7.5)
        twin.set_xlabel("events per adjustment parameter (EPV)", fontsize=8.5)
    axes[0].axhline(true_hr, color="#444444", lw=1.0, ls="--", alpha=0.7, label=f"true HR = {true_hr}")
    axes[0].legend(fontsize=8.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_heterogeneous_adjustment(df: pd.DataFrame,
                                  outpath: str = "informative_adjustment.pdf") -> None:
    """Power for X vs. sigma_z with 95% CIs, by p, against the closed form
    (report Figure 4). `df` from `run_heterogeneous_adjustment_study`."""
    plt.rcParams.update(_PLOT_STYLE)
    df = df.copy()
    df["se"] = np.sqrt(df["power"] * (1 - df["power"]) / df["n_sims"])
    r2_target = df["r2_target"].iloc[0]
    p_levels = sorted(df["p"].unique())
    colors = dict(zip(p_levels, PALETTE))
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    for p in p_levels:
        sub = df[df["p"] == p].sort_values("sigma_z")
        ax.errorbar(sub["sigma_z"], sub["power"], yerr=1.96 * sub["se"], marker="o",
                    color=colors[p], lw=1.3, ms=5, capsize=3, label=f"$p={p}$ adjustment covariates")
    ax.axhline(df["theoretical_power"].iloc[0], color="#444444", lw=1.1, ls="--", alpha=0.7,
               label=fr"closed-form prediction ($R^2={r2_target}$ only)")
    ax.set_xlabel(r"SD of each covariate's true log-hazard ratio, $\sigma_Z$")
    ax.set_ylabel("power for $X$")
    ax.set_title(f"$N={df['n'].iloc[0]}$, $R^2(X,Z)={r2_target}$", fontsize=10)
    ax.legend(fontsize=8.5, frameon=False, loc="lower left")
    ax.set_xticks(sorted(df["sigma_z"].unique()))
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# --- Appendix B ---------------------------------------------------------------

def plot_concordance_mapping(mapping: pd.DataFrame | None = None, highlight: float = 0.70,
                             outpath: str = "figB1_concordance_mapping.pdf") -> None:
    """Exact mapping between per-SD HR and population concordance (Figure B1).
    `mapping` (optional) has columns C and hr to mark."""
    plt.rcParams.update({**_PLOT_STYLE, "font.size": 9})
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    hr = np.exp(np.linspace(0, np.log(4.5), 200))
    ax.plot(hr, [c_of_beta(np.log(h)) for h in hr], color=PALETTE[0])
    if mapping is not None:
        for _, m in mapping.iterrows():
            hot = abs(m["C"] - highlight) < 1e-9
            ax.plot(m["hr"], m["C"], "o", ms=5, zorder=3,
                    color=PALETTE[1] if hot else PALETTE[0])
            ax.annotate(f"C={m['C']:.3g}\nHR={m['hr']:.2f}", (m["hr"], m["C"]),
                        textcoords="offset points", xytext=(7, -16), fontsize=7)
    ax.set_xscale("log")
    ax.set_xticks([1, 1.5, 2, 3, 4])
    ax.set_xticklabels(["1", "1.5", "2", "3", "4"])
    ax.set_xlabel(r"hazard ratio per SD of $X$, $e^{\beta}$")
    ax.set_ylabel(r"concordance $K(\beta)$")
    ax.set_title(r"Concordance implied by a per-SD hazard ratio ($X\sim\mathcal{N}(0,1)$)", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_cindex_association_power(grid: pd.DataFrame,
                                  outpath: str = "figB2_association_power.pdf") -> None:
    """Power to detect association vs. N for several target C: Cox Wald test
    and the test of C = 0.5 (Figure B2). `grid` has columns C, n, power_cox,
    power_c05."""
    plt.rcParams.update({**_PLOT_STYLE, "font.size": 9})
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for k, (c, sub) in enumerate(grid.groupby("C")):
        sub = sub.sort_values("n")
        for ax, col in zip(axs, ["power_cox", "power_c05"]):
            ax.plot(sub["n"], sub[col], "-o", ms=3, color=PALETTE[k % len(PALETTE)], label=f"C = {c:.2f}")
    for ax, title in zip(axs, [r"Cox Wald test of $\beta=0$", "Concordance test of $C=0.5$"]):
        ax.set_xscale("log")
        ax.axhline(0.8, ls="--", lw=0.8, color="grey")
        ax.set_xlabel("sample size $N$")
        ax.set_title(title, fontsize=9)
    axs[0].set_ylabel("power")
    axs[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_benchmark_power(bench: pd.DataFrame, c_limit: float, sigma: float, alpha: float = 0.05,
                         outpath: str = "figB3_benchmark_power.pdf") -> None:
    """Simulated power to show C > C0 vs. N against the closed form (eq. B.6)
    (Figure B3). `bench` has columns c0, n, power_cc0."""
    plt.rcParams.update({**_PLOT_STYLE, "font.size": 9})
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    z = norm.ppf(1 - alpha)
    for k, (c0, sub) in enumerate(bench.groupby("c0")):
        sub = sub.sort_values("n")
        color = PALETTE[k % len(PALETTE)]
        ax.plot(sub["n"], sub["power_cc0"], "o", color=color, ms=4,
                label=fr"simulated, $H_0: C\leq{c0:.2f}$")
        nn = np.linspace(max(10, sub["n"].min() * 0.8), sub["n"].max() * 1.05, 300)
        ax.plot(nn, norm.cdf((c_limit - c0) * np.sqrt(nn) / sigma - z), "-", color=color, lw=1,
                label="closed form (B.6)")
    ax.axhline(0.8, ls="--", lw=0.8, color="grey")
    ax.set_xlabel("sample size $N$")
    ax.set_ylabel(f"power (one-sided, $\\alpha={alpha:g}$)")
    ax.set_title("Power to show the biomarker beats a benchmark $C_0$", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
