"""Regenerate the main report's tables and figures: python scripts/reproduce_report.py [outdir]

Deprecated: survpower.report is removed in 0.4.0. For exact reproduction install the
v0.2.0 tag; these scripts will be replaced by tutorials (#12, #26).
"""
import sys

from survpower.report import reproduce_report

if __name__ == "__main__":
    reproduce_report(sys.argv[1] if len(sys.argv) > 1 else "survpower_report")
