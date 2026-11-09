# Effect sizes: hazard ratio and concordance

## The model

The biomarker \(X\) is standardized, \(X \sim \mathcal N(0,1)\), and the hazard is

\[
h(t \mid X) = h_0(t)\, e^{\beta X}, \qquad \mathrm{HR} = e^{\beta}\ \text{per SD of } X .
\]

By default, data are simulated with a constant baseline hazard
\(h_0 = \lambda_0 = \ln 2 / \text{median OS}\), so \(T \mid X \sim \text{Exponential}(\lambda_0 e^{\beta X})\)
is an exact Cox model. Any other [baseline hazard](censoring-and-accrual.md#shape-of-the-baseline-hazard)
keeps the model an exact Cox model with the same HR: only \(h_0(t)\) changes.

## Per SD vs. per raw unit

Literature often reports an HR per raw unit (per mm³, per 10 points of a
score). With `--biomarker-sd s`, `--hr` is read per raw unit and converted:

\[
\mathrm{HR}_{\text{per SD}} = \mathrm{HR}_{\text{per unit}}^{\,s}.
\]

Outputs show both scales.

## From concordance to hazard ratio

Under PH, with *any* baseline hazard, a pair's probability of being ordered
correctly is \(P(T_i < T_j \mid X_i, X_j) = \operatorname{expit}\{\beta (X_i - X_j)\}\).
Averaging over \(D = X_i - X_j \sim \mathcal N(0, 2)\) gives the population
concordance:

\[
K(\beta) = 2 \int_0^\infty \operatorname{expit}(\beta d)\, \tfrac{1}{\sqrt 2}\,\varphi\!\left(\tfrac{d}{\sqrt 2}\right) \mathrm{d}d,
\]

computed by `c_of_beta` and inverted exactly by `beta_of_c` / `hr_of_c`. As a
closed-form rule of thumb, accurate to 3% over C ∈ [0.55, 0.75]:
\(\beta \approx 1.16 \tan\{\pi (C - \tfrac12)\}\) (`beta_of_c_approx`).

| Target C | HR per SD | Events for 80% power (closed form) | N, closed form | N, simulated |
|---|---|---|---|---|
| 0.527 | 1.10 | 864 | 1081 | ~1107 |
| 0.539 | 1.15 | 402 | 503 | ~514 |
| 0.55 | 1.20 | 245 | 306 | ~312 |
| 0.60 | 1.45 | 57 | 72 | ~84 |
| 0.65 | 1.80 | 23 | 29 | ~36 |
| 0.70 | 2.33 | 11 | 14 | ~21 |
| 0.75 | 3.24 | 6 | 8 | ~15 |

20% censoring, two-sided α = 0.05. Every output table includes a `c_index`
column, so you can see the effect on both scales.

!!! warning "A C of 0.70 is a very large effect for one biomarker"
    C = 0.70 corresponds to HR = 2.33 **per SD**, more than ten times larger on
    the log scale than HR = 1.10 (C = 0.527). A design assuming HR = 1.10 and a
    design assuming C = 0.70 are different hypotheses, not two phrasings of one.
    Pre-register one of them.

### In Python: `sample_size(c_index=...)`

`sample_size` takes either effect and converts a concordance with `hr_of_c`:

```python
from survpower import sample_size
sample_size(c_index=0.70, method="closed_form").n_evaluable   # 14
sample_size(c_index=0.70).n_evaluable                          # ~21 ("auto")
```

`c_index` is the **population** concordance \(K(\beta)\) above (Gönen–Heller),
not the limit of Harrell's C, which depends on censoring. So `c_index=0.70` means
"the population concordance is 0.70". To plan a test against a benchmark C₀
with a specific estimator, use `run_cindex_benchmark_mode`.

!!! warning "The closed form fails at large effects"
    The closed form treats \(\hat\beta\) as normal with its asymptotic
    variance. That holds when there are many events, not at a dozen. For
    C = 0.70 it gives N = 14 (11 events). The simulated power at N = 14 is
    about 0.55–0.64, not 0.80. `method="auto"` (the default) calibrates by
    simulation and, below 100 patients, scans N directly. It finds N ≈ 21–22.
    The last column of the table above shows the same gap at C ≥ 0.60.

## Skewed biomarkers

The mapping assumes \(X\) is normal. For skewed biomarkers (volumes, densities),
either plan on a normalizing transform, which is also what you will analyse, or
recompute \(K\) from the empirical distribution of pairwise differences. The
concordance itself is rank-based and unaffected by monotone transforms; the HR
per SD is not.
