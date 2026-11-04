"""Figures for the docs site, drawn from the package at build time (never committed).

As an MkDocs hook (``hooks:`` in mkdocs.yml) every figure is added to the site as a
generated file; run directly, it writes them to a folder:

    python scripts/docs_figures.py [out_dir]       # default: docs_figures_out
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from survpower.censoring import event_probability_exponential


def accrual_censoring_contour(out) -> None:
    """Censoring fraction over (accrual A, follow-up F) at HR = 1, median OS 15
    months, 5%/yr dropout, from the closed form."""
    A = np.linspace(0, 48, 97)
    F = np.linspace(0, 36, 73)
    cens = np.array([[1 - event_probability_exponential(a, f, 0.05) if a + f > 0 else 1.0
                      for a in A] for f in F])
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    cf = ax.contourf(A, F, cens, levels=np.arange(0.1, 1.01, 0.05), cmap="Blues")
    cs = ax.contour(A, F, cens, levels=np.arange(0.2, 0.9, 0.1), colors="k", linewidths=0.7)
    ax.clabel(cs, fmt=lambda v: f"{v:.0%}", fontsize=8)
    for total in (24, 48, 72):
        ax.plot([0, total], [total, 0], ls=":", lw=0.9, color="#b45a28")
    ax.plot([36], [12], marker="D", color="#b45a28", ms=6)
    ax.annotate("docs example\n(31.5%)", (36, 12), xytext=(35, 16), ha="right", fontsize=8)
    ax.set_xlim(0, 48)
    ax.set_ylim(0, 36)
    ax.set_xlabel("accrual A (months)")
    ax.set_ylabel("follow-up after last enrolment F (months)")
    ax.set_title("Censored fraction, HR = 1, median OS 15 mo, 5%/yr dropout", fontsize=10)
    fig.colorbar(cf, ax=ax, label="P(censored)", format=lambda v, _: f"{v:.0%}")
    fig.tight_layout()
    fig.savefig(out, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


# published path (relative to docs/) -> figure function
FIGURES = {
    "concepts/img/accrual-censoring-contour.svg": accrual_censoring_contour,
}


def on_files(files, config):
    """MkDocs hook: add every figure to the site as a generated file."""
    from mkdocs.structure.files import File

    for path, draw in FIGURES.items():
        buf = io.BytesIO()
        draw(buf)
        files.append(File.generated(config, path, content=buf.getvalue()))
    return files


if __name__ == "__main__":
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "docs_figures_out")
    for path, draw in FIGURES.items():
        target = out_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        draw(target)
        print(target)
