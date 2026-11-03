#!/usr/bin/env bash
# Copy-paste recipes for grant and trial planning. See docs/quickstart.md.
set -euo pipefail

# Required N for a per-SD HR, at 80% and 90% power
survpower --hr 1.20 --power 0.80 0.90 --outdir out/sample_size

# Power of existing cohorts
survpower --hr 1.10 1.15 --n 250 1200 --outdir out/power

# Minimum detectable effect, realistic calendar, 10% assay failures
survpower --n 250 --accrual-months 36 --followup-months 12 --dropout 0.05 \
          --non-evaluable 0.10 --outdir out/mde

# Adjusted analysis with a sensitivity check on collinearity / EPV
survpower --hr 1.20 --n-covariates 5 --r2 0.3 --sensitivity collinearity epv --outdir out/adjusted

# Concordance targets: association, benchmark (Uno's C) and precision
survpower --c-index 0.70 --outdir out/c_assoc
survpower --c-index 0.70 --c0 0.65 --c-estimator uno --c-tau 24 --outdir out/c_bench
survpower --c-index 0.70 --precision 0.05 --outdir out/c_precision

# Non-proportional hazards at the planning point
survpower --hr 1.30 --n 400 --sensitivity waning crossing --tau 6 --outdir out/nonph
