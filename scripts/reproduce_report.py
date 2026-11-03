"""Regenerate the main report's tables and figures: python scripts/reproduce_report.py [outdir]"""
import sys

from survpower.report import reproduce_report

if __name__ == "__main__":
    reproduce_report(sys.argv[1] if len(sys.argv) > 1 else "survpower_report")
