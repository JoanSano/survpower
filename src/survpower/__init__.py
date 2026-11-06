"""survpower: power and sample-size planning for continuous biomarkers under
the Cox proportional-hazards model, with hazard-ratio or concordance-index
effect sizes.

Quick start::

    from survpower import StudyDesign, sample_size
    print(sample_size(hr=1.15, power=0.80).summary)       # or c_index=0.60

or from the shell: ``survpower sample-size --hr 1.15``.
"""

from .baseline import (
    DEFAULT_LAMBDA0,
    DEFAULT_MEDIAN_OS_MONTHS,
    DEFAULT_TAU_MONTHS,
    median_os_to_lambda0,
    median_survival,
)
from .calibration import (
    calibrate_sample_size,
    estimate_detectable_hr_by_simulation,
    estimate_required_n_by_simulation,
)
from .censoring import calibrate_censoring_rate, censor_rate_from_design, event_probability_exponential
from .closed_form import (
    detectable_hr_continuous_cph,
    dichotomized_power,
    dichotomized_sample_size,
    sample_size_continuous_cph,
    theoretical_power_with_r2,
)
from .concordance import (
    benchmark_power,
    benchmark_sample_size,
    beta_of_c,
    c_of_beta,
    concordance_index,
    concordance_limit,
    fit_cox_coefficients,
    gonen_heller_c_with_se,
    harrell_c_with_se,
    harrell_limit,
    hr_of_c,
    precision_half_width,
    precision_sample_size,
    sigma_c,
    simulate_cindex_power,
    uno_c_with_se,
    uno_limit,
)
from .design import StudyDesign
from .fitting import cox_fit_1d
from .planning import (
    run_cindex_benchmark_mode,
    run_cindex_precision_mode,
    run_detectable_effect_mode,
    run_power_mode,
    run_sample_size_mode,
    SampleSizeResult,
    sample_size,
)
from .simulate import simulate_power, simulate_power_design
from . import scenario
# provisional (may change before 1.0); `simulate_scenario` because `survpower.simulate`
# is the module of the legacy engines
from .scenario import Scenario
from .scenario import simulate as simulate_scenario

__version__ = "0.3.0"

__all__ = [name for name in dir() if not name.startswith("_")]
