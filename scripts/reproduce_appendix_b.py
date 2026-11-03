"""Regenerate Appendix B (concordance index): python scripts/reproduce_appendix_b.py [outdir] [n_sims]"""
import sys

from survpower.report import reproduce_appendix_b

if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "survpower_report_appB"
    n_sims = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    reproduce_appendix_b(outdir, n_sims=n_sims)
