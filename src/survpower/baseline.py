"""Baseline hazard calibration.

The data-generating process throughout is a constant ("exponential") baseline
hazard h0(t) = lambda0, calibrated so the baseline (X = 0) median survival
equals a realistic median overall survival:

    lambda0 = ln(2) / median_os

lambda0 = 1 recovers an arbitrary-time-unit model and lambda0 -> 0 the
degenerate "no baseline risk" limit. Power and required-event results for a
PH model are exactly invariant to lambda0 (Cox regression only uses the
ordering of times); only time-denominated quantities (censoring rates, fixed
clinical windows such as tau) change.
"""

from __future__ import annotations

import numpy as np

DEFAULT_MEDIAN_OS_MONTHS = 15.0  # typical glioblastoma median overall survival
DEFAULT_TAU_MONTHS = 6.0         # early-risk window for the non-PH scenarios
DEFAULT_CENSOR_RATE = 0.20


def median_os_to_lambda0(median_os: float) -> float:
    """Rate of a constant baseline hazard with the given median survival
    (same time unit as `median_os`; months throughout this package)."""
    return float(np.log(2.0) / median_os)


DEFAULT_LAMBDA0 = median_os_to_lambda0(DEFAULT_MEDIAN_OS_MONTHS)  # ~0.04621 / month


def median_survival(hr: float, lambda0: float = DEFAULT_LAMBDA0) -> float:
    """Median survival for a subject whose hazard is `hr` times the baseline."""
    return float(np.log(2.0) / (lambda0 * hr))
