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

## Two ways to specify censoring

**Directly.** `--censor-rate 0.2` means 20% of patients are censored overall.
The simulation uses exponential censoring \(C \sim \text{Exp}(\lambda_c)\),
with \(\lambda_c\) found by bisection so that \(P(C < T) = 0.2\) under the
planned effect.

**From the trial calendar.** Give `--accrual-months A --followup-months F`
(optionally `--dropout d`):

* entry is uniform over \([0, A]\), and the analysis is \(F\) months after the
  last enrolment, so administrative censoring is \(C_{\text{adm}} \sim U(F, A + F)\);
* dropout is exponential with \(\lambda_d = -\ln(1 - d)/12\) per month.

`censor_rate_from_design` computes the implied \(P(C < T)\) by Monte Carlo,
and the simulations then use exponential censoring at that rate. Matching the
event probability is what matters for power; the shape of the censoring
distribution is second order.

Example: median OS 15 months, 36 months accrual, 12 months follow-up and 5%
dropout per year give 31.5% censoring at HR = 1.

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
