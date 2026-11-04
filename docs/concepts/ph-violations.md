# Non-proportional hazards

Every power figure above assumes the HR is constant over follow-up. When the
biomarker's effect plausibly changes with time, check the design under a
realistic alternative.

```bash
survpower --hr 1.30 --n 400 --sensitivity waning crossing --tau 6
```

## Scenarios

**Waning effect** (`waning`): HR = `--waning-hr` (default: planning HR) before
τ, then 1. Event times use inverse-cumulative-hazard sampling of

\[
H(t\mid X) = \begin{cases} \lambda_0 t\, e^{\beta_0 X}, & t < \tau \\
\lambda_0 \tau\, e^{\beta_0 X} + \lambda_0 (t - \tau), & t \ge \tau. \end{cases}
\]

**Crossing hazards** (`crossing`): HR = EARLY before τ and LATE after
(`--crossing-hr`, default 1/HR then HR).

Each replicate is analysed with the standard constant-HR Cox model, and the
Grambsch–Therneau test of the scaled Schoenfeld residuals is applied to it.

## What to look at

| Column | Meaning |
|---|---|
| `power_naive` | power of the constant-HR Wald test: power to detect the *averaged* effect |
| `mean_HR_hat` | what the naive model estimates, an event-weighted average of β(t) |
| `power_zph_detect` | how often the PH diagnostic would flag the violation |

With τ = 6 months and median OS 15 months, only ~25% of events occur before τ.
A strong early effect (HR 1.30) is estimated as ≈1.08, and at N = 250 the PH
test catches the violation in under a quarter of studies.

How much opposite-signed effects cancel depends on **where τ falls in the
survival distribution**. Moving τ from 6 to 12 months can turn high-but-biased
power into almost none.

## If non-PH is plausible a priori

Pre-specify a PH-robust primary or co-primary analysis, and power it by
simulation under the anticipated alternative:

* restricted mean survival time (RMST) at a fixed horizon;
* weighted log-rank tests (Fleming–Harrington G^ρ,γ);
* time-varying coefficients (X × log t, or a step function at τ);
* landmark / milestone survival;
* weighted Cox regression for a well-defined average HR (Schemper et al., 2009),
  with robust sandwich variances (Lin & Wei, 1989).

Their power is not evaluated by `survpower`; their relative merits depend on
the pattern of non-proportionality (Lin et al., 2020).

!!! note "Competing risks"
    For overall survival, death from any cause is the event and there are no
    competing risks. They arise for cause-specific endpoints and for
    progression. There, a Cox model that censors competing events estimates the
    cause-specific hazard. A Fine–Gray model estimates the subdistribution hazard.
    Cumulative incidence should use the Aalen–Johansen estimator, not 1 − KM.

The concordance remains well defined under non-PH, but it depends on follow-up.
Use Uno's C with a fixed horizon (`--c-estimator uno --c-tau ...`).
Do not use Gönen–Heller's C here: it is computed from the fitted Cox model, so
under non-PH it reports the concordance the misspecified model implies.
