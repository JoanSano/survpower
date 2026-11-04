# Censoring, accrual and attrition

## Events drive power

For a Cox model the information about \(\beta\) accrues with the number of
**events** \(D\), not subjects:

\[
\mathcal I(\beta) \approx D\,\sigma_X^2 (1 - R^2), \qquad
D = \frac{(z_{1-\alpha/2} + z_{1-\gamma})^2}{\beta^2 \sigma_X^2 (1 - R^2)}, \qquad
N = \lceil D / P(\text{event}) \rceil .
\]

So the censoring assumption matters as much as the effect size. Every table
reports `expected_events`. Quote it in the protocol.

## How censoring is generated

Every simulation censors the same way, whichever option below is used. Each
patient gets two independent clocks, both measured from that patient's entry:

\[
T \sim \text{Exp}\!\left(\lambda_0 e^{\beta X}\right), \qquad
C \sim \text{Exp}(\lambda_c),
\]

and we observe \(\min(T, C)\), with an event when \(T \le C\). \(C\) has the
same rate \(\lambda_c\) for every patient and does not depend on \(T\) or
\(X\), so censoring is independent and non-informative. Whichever clock runs
out first wins. For one patient,

\[
P(\text{censored} \mid X) = \frac{\lambda_c}{\lambda_c + \lambda_0 e^{\beta X}},
\]

and the censored fraction of the trial is the average of this over
\(X \sim N(0, 1)\).

The user does not give \(\lambda_c\). They give a **censored fraction**
\(p_c\), and \(\lambda_c\) is found by bisection so that \(P(C < T) = p_c\)
under the planned effect (`calibrate_censoring_rate`). The two options below
differ only in how \(p_c\) is chosen. **Censoring in the simulated data is
always exponential.**

## Two ways to choose the censored fraction

**Directly.** `--censor-rate 0.2` sets \(p_c = 0.2\): 20% of patients are
censored overall.

**From the trial calendar.** Give `--accrual-months A --followup-months F`
(optionally `--dropout d`), and \(p_c\) is the fraction this calendar would
censor:

* Patients enter uniformly over the first \(A\) months. The analysis takes
  place \(F\) months after the last enrolment, at calendar month \(A + F\).
  So \(F\) is the **minimum** follow-up, not a window in which censoring
  happens.
* A patient entering at calendar month \(e\) can be followed for
  \(C_{\text{adm}} = A + F - e\) months before the analysis date. This is the
  *administrative censoring time*, measured from that patient's entry. With
  uniform entry, \(C_{\text{adm}} \sim U(F, A + F)\): the first patient
  enrolled can be followed for \(A + F\) months, the last for \(F\).
* Independently, patients can be lost to follow-up at any time after entry,
  including during accrual. This dropout is exponential with rate
  \(\lambda_d = -\ln(1 - d)/12\) per month. The logarithm turns the annual
  loss fraction \(d\) into the constant rate that loses exactly that fraction
  in a year: \(e^{-12\lambda_d} = 1 - d\). All other independent censoring
  (withdrawal, moving away) belongs in \(d\).
* The calendar's censoring time is \(\min(C_{\text{adm}}, \text{dropout})\).

`censor_rate_from_design` computes the fraction this censors by Monte Carlo,
and `event_probability_exponential` computes it in closed form (below). Only
that fraction is passed to the simulations, which then use exponential
censoring as described above.

!!! note "Same fraction, different timing"
    The simulated trials match the calendar's **number** of censored patients,
    not **when** they are censored. Under the calendar nobody is censored
    administratively before \(F\) months. Exponential censoring is most likely
    right after entry. Cox power depends mostly on the number of events, so
    matching the fraction is expected to be enough for the Wald test. But the
    timing changes which patients have events, and it affects Harrell's C
    (whose limit depends on the censoring distribution) and Uno's C with a
    fixed horizon more. That this is second order has not yet been checked;
    simulating the calendar censoring itself is planned in
    [#16](https://github.com/JoanSano/survpower/issues/16).

Example: median OS 15 months, 36 months accrual, 12 months follow-up and 5%
dropout per year give 31.5% censoring at HR = 1. This is Lachin and Foulkes's
eq. 4.1 with R = 3 and T = 4 years (see the cross-check below).

### Which HR sets the censored fraction

Under a calendar the censored fraction depends on the HR. A biomarker effect
spreads the patients' hazards, and averaging over that spread changes how many
have their event before the analysis date. In the example it is 31.5% at HR = 1,
31.6% at HR = 1.15 and 32.2% at HR = 1.5 (per SD). The value at HR = 1 is only
a reference, which the CLI prints in its design summary. Each planning mode
uses the fraction at the HR it is planning for, and reports it in the
`censor_rate` column:

| Question | HR used for the censored fraction |
|---|---|
| Power (`--hr` and `--n`) | each HR given |
| Sample size (`--hr`) | each HR given |
| Minimum detectable HR (`--n`) | the detectable HR itself. It is unknown in advance, so the closed-form detectable HR and the fraction are iterated together until they agree, starting at HR = 1 |
| Concordance questions (`--c`, `--c0`, `--precision`) | the HR implied by the target C |

The simulation-calibrated detectable HR can differ slightly from the
closed-form one where the fraction is fixed. Within that gap the fraction
changes by about 0.1 percentage points in typical designs.

A fixed `--censor-rate` does not depend on the HR. The same fraction is used at
every HR, and \(\lambda_c\) is recalibrated to hit it.

### The event probability in closed form

With exponential survival the calendar design has a closed form, which
`event_probability_exponential` evaluates. Take one subject with hazard
\(\lambda\) and dropout hazard \(\delta\), and let \(r = \lambda + \delta\).
Given an administrative censoring time \(c\), the subject has the event if it
comes before both \(c\) and dropout:

\[
P(\text{event} \mid \lambda, c) = \int_0^c \lambda e^{-r t}\,dt
= \frac{\lambda}{r}\left(1 - e^{-r c}\right).
\]

Averaging over \(c \sim U(F, A + F)\):

\[
P(\text{event} \mid \lambda)
= \frac{\lambda}{r}\left[1 - \frac{e^{-rF} - e^{-r(A+F)}}{rA}\right],
\qquad
\lim_{A \to 0} P(\text{event} \mid \lambda) = \frac{\lambda}{r}\left(1 - e^{-rF}\right).
\]

Under the Cox model each subject's hazard is \(\lambda = \lambda_0 e^{\beta X}\)
with \(X \sim N(0, 1)\), so the trial-level probability is

\[
P(\text{event}) = E_X\!\left[P(\text{event} \mid \lambda_0 e^{\beta X})\right],
\]

which the function computes by Gauss–Hermite quadrature. The censored fraction
is \(1 - P(\text{event})\). It agrees with the Monte Carlo
`censor_rate_from_design` to within Monte Carlo error.

### Accrual vs follow-up

![Censored fraction over accrual and follow-up](img/accrual-censoring-contour.svg)

*Censored fraction at HR = 1 (median OS 15 months, 5% dropout per year). Dotted
lines mark constant total study time A + F = 24, 48 and 72 months.*

The average potential follow-up is \(F + A/2\), so a month of follow-up after
the last enrolment is worth about two months of accrual. At a fixed total study
time, moving months from accrual into follow-up lowers censoring. Same settings,
A + F = 48 months:

| Accrual A | Follow-up F | Censored |
|---:|---:|---:|
| 0 | 48 | 16.6% |
| 12 | 36 | 19.6% |
| 24 | 24 | 24.3% |
| 36 | 12 | 31.5% |
| 48 | 0 | 42.9% |

Starting from the example (A = 36, F = 12, 31.5%), 12 more months of
follow-up bring censoring down to 21.0%, while 12 more months of accrual give
27.3%. The extra accrual also enrols more patients, so whether it is the better
use of the time depends on the enrolment rate.

### Cross-check against Lachin & Foulkes (1986)

Lachin and Foulkes give the same quantity for uniform entry over \(R\) years,
total study time \(T\) and exponential losses with hazard \(\eta\) (their
eq. 4.1):

\[
E(\delta \mid \lambda, \eta) = \frac{\lambda}{\lambda + \eta}
\left[1 - \frac{e^{-(T-R)(\lambda+\eta)} - e^{-T(\lambda+\eta)}}{R(\lambda + \eta)}\right].
\]

This is the formula above with \(R = A\), \(T = A + F\) and \(\eta = \delta\).
Here dropout is given as an annual loss fraction \(d\), with
\(\delta = -\ln(1 - d)/12\) per month, so their \(\eta\) per year is
\(d = 1 - e^{-\eta}\). The tests reproduce their worked example
(\(R = 3\), \(T = 5\), \(\lambda = 0.20\) and \(0.30\) per year:
\(E(\delta) = 0.4959\) and \(0.6381\), Section 2) and every entry of their
Table 2, which adds losses \(\eta = 0\) to \(0.20\) per year.

## Non-evaluable patients

`--non-evaluable f` is the fraction of enrolled patients **without a usable
biomarker value** (no tissue, assay failure, imaging not analysable). It is
separate from censoring: those patients contribute nothing. Simulations use
\(N_{\text{eval}} = \lfloor N (1 - f) \rfloor\), and solved sample sizes are
inflated to \(N_{\text{enrolled}} = \lceil N_{\text{eval}} / (1 - f) \rceil\).

## Baseline median survival

`--median-os` sets \(\lambda_0 = \ln 2 / \text{median OS}\). For PH analyses,
power and N are **exactly invariant** to it: Cox regression uses only the
ordering of times, and the censoring calibration rescales with \(\lambda_0\).
It matters when:

* censoring comes from an accrual calendar, where calendar time interacts with
  survival time;
* you run the waning / crossing scenarios with a fixed clinical window τ;
* you use Uno's C with a fixed horizon `--c-tau`.
