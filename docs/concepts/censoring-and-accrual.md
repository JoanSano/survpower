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

`censor_rate_from_design` computes the implied \(P(C < T)\), and the
simulations then use exponential censoring at that rate. Matching the event
probability is what matters for power; the shape of the censoring distribution
is second order.

Example: median OS 15 months, 36 months accrual, 12 months follow-up and 5%
dropout per year give about 31% censoring at HR = 1.

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
