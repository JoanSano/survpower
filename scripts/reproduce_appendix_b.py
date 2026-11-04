"""Regenerate Appendix B (concordance index): python scripts/reproduce_appendix_b.py [outdir] [n_sims]

Deprecated: survpower.report is removed in 0.4.0. For exact reproduction install the
v0.2.0 tag; these scripts will be replaced by tutorials (#12, #26).
"""
import sys

from survpower.report import reproduce_appendix_b

if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "survpower_report_appB"
    n_sims = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    reproduce_appendix_b(outdir, n_sims=n_sims)
