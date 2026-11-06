# GUI feasibility: Streamlit, Panel, Gradio, marimo

Spike for #20. Assessed 2026-11-05 against survpower 0.3.0.

> **Later changes (not otherwise reflected in this dated note):** the `to_cli` prototype in
> appendix A uses the flag-only CLI syntax, which #61 replaced with subcommands (#54 is written
> against those). #51 moved to milestone M5.5, although the follow-up table says M6.

## Summary

**Go, with Streamlit, for a local app launched by `survpower gui`.** Build it after the
core gaps in section 2 are closed. The estimate is about 17.5 working days: 11 for the
core and 6.5 for the GUI. The recommendation was accepted, and the work is tracked in
milestone M6.

Decisions taken for this spike:

- **Target user:** the maintainer and collaborators running it locally. Hosting and
  WebAssembly were assessed but did not drive the choice.
- **Parallel execution:** in scope as a core feature, proposed as a follow-up issue. The
  estimate assumes it exists.

Main findings:

- **No framework is blocked by the core,** and the core can drive all four. Progress and
  cooperative cancellation worked in every one using a thin hook on the replicate loop.
- **Cancel behaviour differs most.** With the obvious wiring, Cancel did not stop a running
  job in Gradio or in a marimo cell.
- **Pyodide works, with a caveat.** survpower gave the same result as native on three
  Pyodide releases, but the current release fails to install because `lifelines` pins
  `pandas<3`.
- **Parallelism matters less than #20 assumed on the fast engine.** A default `sample_size`
  takes about 3 s without it. It matters for lifelines and adjusted models, which would take
  an estimated 4 to 6 minutes at default budgets.

**How claims are backed:** "Tested" means run for this spike; everything else links to the
framework's docs. Throwaway prototypes were not merged, and appendix A has the setup and the
key snippets.

## 1. Framework comparison

Versions tested: Streamlit 1.66.1, Panel 1.10.0, Gradio 6.31.0, marimo 0.25.2, on Python 3.12.

| Criterion | Streamlit | Panel | Gradio | marimo |
|---|---|---|---|---|
| **Execution model** | [Reruns the script on each interaction; fragments rerun only themselves](https://docs.streamlit.io/develop/concepts/architecture/fragments) | [Callbacks, dispatched to a thread pool with `nthreads`](https://panel.holoviz.org/how_to/concurrency/threading.html) | Event listeners with [one queue each, one request at a time by default](https://gradio.app/guides/queuing) | [Reactive: dependent cells rerun](https://docs.marimo.io/guides/reactivity/); [`mo.stop` and a run button gate slow cells](https://docs.marimo.io/guides/expensive_notebooks/) |
| **Progress** | Tested: `st.progress`, or polling from a fragment | Tested: `Progress` indicator | Tested: [`gr.Progress`](https://www.gradio.app/docs/gradio/progress), which shows elapsed time and percent; can also track tqdm | Tested: [`mo.status.progress_bar`](https://docs.marimo.io/api/status/), which shows rate and ETA |
| **Cancel, simplest wiring** (table below) | Tested: stops, but *any* widget click also stops the run | Tested: stops | Tested: **job keeps running**; [documented](https://www.gradio.app/docs/gradio/button) | Tested: **job keeps running**, because the click waits for the cell |
| **Cancel, working pattern** | Tested: thread plus fragment. [App threads are not officially supported, but threads that make no `st.*` calls avoid the problem](https://docs.streamlit.io/develop/concepts/design/multithreading) | Same as simplest wiring | Tested: `cancels=` plus a shared `threading.Event` | Tested: [`mo.Thread`](https://docs.marimo.io/guides/outputs/) with a shared event |
| **Cache on (design, seed)** | Tested: [`st.cache_data`](https://docs.streamlit.io/develop/concepts/architecture/caching), which also hashes the function's code | Tested: [`pn.cache`](https://panel.holoviz.org/how_to/caching/memoization.html), optionally to disk | Tested: [`@gr.cache`](https://gradio.app/guides/caching/) | Tested: [`mo.cache`, `mo.persistent_cache`](https://docs.marimo.io/api/caching/) |
| **Local launch** | Tested: [`python -m streamlit run`](https://docs.streamlit.io/develop/concepts/architecture/run-your-app); needs a subprocess | Tested: `panel serve` | Tested: `demo.launch()`, in process | Tested: `marimo run` |
| **Hosted option and limits** (low weight) | [Community Cloud: up to 2 cores, 2.7 GB; sleeps after 12 h idle (figures dated Feb 2024)](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app) | Self-host | [HF Spaces: free tier is 2 vCPU, 16 GB, but a Gradio Space needs a paid plan and sleeps when idle](https://huggingface.co/docs/hub/spaces-overview) | Self-host, or a WASM export |
| **WebAssembly route** (not tested in a browser) | No first-party route found; third-party stlite not assessed | [`panel convert`](https://panel.holoviz.org/how_to/wasm/convert.html) | Gradio-Lite looks unmaintained: [documented only in archived 4.44.1 docs](https://gradio.app/4.44.1/guides/gradio-lite), `js/lite` is gone from the repo, and the [last npm release](https://www.npmjs.com/package/@gradio/lite) is 5.45.0 (Sep 2025) | [First-party; 2 GB cap and no true CPU parallelism](https://docs.marimo.io/guides/wasm/) |
| **Testing** | Tested: [`AppTest`](https://docs.streamlit.io/develop/concepts/app-testing) runs headless under pytest and covers widgets and cache | Tested: [plain pytest on objects](https://panel.holoviz.org/how_to/test/pytest.html); `inner_text` came back empty in browser tests, though role locators worked | Tested: `gradio_client` against a live local server | Tested: `app.run()` from pytest, and [pytest collects notebook test cells](https://docs.marimo.io/guides/testing/) |
| **Added packages / disk** (over survpower's 25 / 306 MB) | +25 / +215 MB | +21 / +131 MB | +43 / +143 MB | +22 / +203 MB |
| **Warm import** | 0.2 s | 0.6 s | 1.6 s | 0.4 s |
| **Python floor, latest release** | ≥3.11 | ≥3.12 | ≥3.10 | ≥3.10 |
| **Resolves on Python 3.9 to** | 1.50.0 | 1.4.5 | 4.44.1 | 0.17.6 |
| **Stars / open issues and PRs** | 46k / 1190 | 5.8k / 822 | 44k / 114 | 23k / 625 |
| **Latest release; licence** | 2026-10-21; Apache-2.0 | 2026-10-23; BSD-3 | 2026-10-28; Apache-2.0 | 2026-10-19; Apache-2.0 (pre-1.0) |

### Tested: what Cancel does to a running job

Each prototype ran an 18,000-replicate `sample_size` in headless Brave, with Cancel clicked
about 6 s in. The figures are from the second of two runs; both runs gave the same outcome
for each variant.

| Variant | Replicates at click | Final replicates | Outcome |
|---|---|---|---|
| Streamlit, blocking script | 13,200 | 13,400 | Stopped within 0.13 s; the page returned to idle |
| Streamlit, thread plus fragment | 12,400 | 12,600 | Stopped within 0.09 s; the page showed the cancelled state |
| Panel, `nthreads=2` | 3,800 | 3,800 | Stopped within 0.04 s |
| Gradio, `cancels=` only | 12,000 | **18,000** | **Ran to completion** |
| Gradio, `cancels=` plus event | 11,000 | 11,000 | Stopped within 0.04 s |
| marimo, blocking cell | 12,800 | **18,000** | **Ran to completion and showed the result** |
| marimo, `mo.Thread` | 12,600 | 12,800 | Stopped within 0.03 s |

Run worked again after Cancel in all seven. Replicates are logged every 200, so the counts
are rounded. Stop latency is about one replicate of the fast engine; adjusted models will
be slower to stop.

**Unresolved Panel slowdown.** The Panel job ran far slower than the others: about 630
replicates/s with progress updates and about 270 without, against roughly 2,000 in the
other frameworks' prototypes. One unbounded run without updates did not finish within the
harness's 250 s limit. A stack dump showed the event loop idle and the job thread simply
running slowly. The root cause was not found and could be in the prototype, so this did not
count against Panel. Anyone choosing Panel should resolve it first.

### Tested: Pyodide

The install was attempted under Node 22 with `micropip`, using the `lifelines` wheel from
PyPI and a locally built survpower wheel. It was not run in a browser.

| Pyodide (Python, pandas) | Install | `sample_size(hr=1.15)` at default budgets |
|---|---|---|
| 0.27.7 (3.12, 2.2.3) | Works | N = 513, CI 499–528, power 0.774, about 13 s |
| 0.28.3 (3.13, 2.3.1) | Works | Same, about 12 s |
| 0.29.5 (3.13, 2.3.3) | Works | Same, about 14 s |
| 314.0.7 (3.14, 3.0.2) | **Fails**: `lifelines` 0.30.3 requires `pandas<3.0` | Same, about 14 s, once the pin is bypassed with `deps=False` |

- **Results:** N, interval and power match the native run exactly, on these checks.
- **Speed:** about 4× slower than native (3.3 s). Those runs shared the machine, so this is
  rough.
- **Concurrency:** on all four, `threading.Thread` failed with "can't start new thread" and
  `multiprocessing.Pool` failed with a missing `_multiprocessing`, matching
  [Pyodide's constraints](https://pyodide.org/en/stable/usage/wasm-constraints.html). A
  WASM deployment gets no parallel speedup from this design.
- **Distribution:** survpower is not on PyPI (404), so WASM needs a self-hosted wheel. The
  same applies to `pip install survpower[gui]` as written in #20; until it is published, the
  extra installs from Git.

## 2. API gaps in the core

| # | Gap | Evidence | Issue | Size |
|---|---|---|---|---|
| G1 | No progress callback or cancellation | #9 closed without it. Patching tqdm proved the hook works, but only `run_replicates` is covered: the adjusted path has its own loop, as do five other functions in `simulate.py`. A scan's total work is unknown until the grid finishes, so progress needs stages | #53 | M |
| G2 | `StudyDesign` does not validate | Tested: invalid designs are accepted, and some return impossible answers: `censor_rate=1.2` gives N = -570 and `r2=1.5` gives N = -285 from the closed form. Rules live in `cli._validate` and depend on argparse | #50 | S |
| G3 | Results lack provenance and strict JSON | Tested: `asdict` then `json.dumps(allow_nan=False)` fails on the scan path (NaN). Results carry no design, seed or version. `details` omits the scan table and `n_sims`, so a power curve cannot show correct intervals | #51 | S |
| G4 | Figures are saved, not returned | Existing plot functions write files and return `None`, and #15 deliberately leaves them so. #15 removes the import-time Agg backend and supplies the KM preview; it is still open | #52, after #15 | S |
| G5 | No path from a result back to the CLI | Tested: `to_cli` (appendix A) reproduced N, enrolled N and power on 5 of 5 designs. But there is no `--version`, and the CSV and summary record no seed, version or replicate count | #54 | S |
| G6 | Single process only | Tested: see below | #55 | M |

### Tested: parallelism (G6)

On an 8-core M1 Pro with BLAS limited to one thread per process:

| Workload | Sequential | Parallel | Speedup |
|---|---|---|---|
| Grid, fast engine, default budget, new pool (2 / 4 / 8 processes) | 2.4 s | 2.8 / 2.4 / 2.0 s | 0.9× / 1.0× / 1.2× |
| Same, warm pool, 5 processes (1.3 s to warm) | 2.4 s | 0.56 s | 4.3× |
| Grid, fast engine, 10× budget, warm pool, 5 processes | 13.1 s | 3.2 s | 4.1× |
| Grid, adjusted p=5, 100 replicates, new pool, 5 processes | 23.5 s | 7.5 s | 3.1× |

Implications for the design:

- **Keep the pool warm.** Worker start-up costs about as much as a default fast run. A pool
  held with `st.cache_resource` worked inside Streamlit and took 4.0 s for 30,000 replicates
  on both runs. It logged `missing ScriptRunContext` warnings.
- **Parallelize grid points first.** Their seeds are already independent, and the results
  were identical to sequential in every test, so no published number changes. The
  confirmation run still follows the grid, which limits the overall gain to about 3× when
  each stage costs the same.
- **Treat replicate-level splitting as a later, breaking step.** Fixed chunks with spawned
  streams give the same result for any worker count, but not today's numbers (0.7995 against
  0.7858 at 8,000 replicates, a gap within about 2 standard errors of the difference). It
  would need a changelog entry. Chunks must also share one censoring calibration.
- **Plan for cancellation in workers.** `Future.cancel()` cannot stop a running task, so
  stopping depends on task size or a shared flag.

## 3. Reproducibility

Every result panel shows, and exports as JSON:

- survpower version, plus numpy, scipy, pandas and lifelines versions;
- the full `StudyDesign`, the effect, the target power and N;
- seed, method (`closed_form`, `probit` or `scan`) and replicate counts;
- the equivalent CLI command, behind an "Export as CLI command" button.

Rules:

- **The cache key includes the survpower version:** canonical JSON of (design, effect, power,
  seed, version). `st.cache_data` hashes the decorated function's own code, and its docs do
  not promise to detect changes in functions it calls.
- **The GUI takes one effect and one power at a time.** The CLI offsets the seed by
  `101 * combo` for each extra combination, so a single-value run is what matches
  `sample_size()` exactly.
- **Compare numbers, not sentences.** The CLI appends a median-split sentence to the summary,
  so its text differs from `sample_size().summary` even when the numbers match.

## 4. MVP scope and estimate

**Inputs:** `StudyDesign` fields, HR or C-index, and N or target power, using G2's validation
so that the GUI offers nothing the core rejects.

**Outputs:**

- power curve with Monte Carlo intervals;
- N\* with its interval;
- the summary sentence;
- a KM preview of one simulated cohort.

**One change from #20:** "N\* with its bootstrap interval" only exists on the probit path.
The scan path has `n_ci = None` (tested: `c_index=0.70`), so the GUI must show the scanned
powers and their MCSEs there rather than an empty interval.

**Preview runs.** The closed-form N appears at once, and simulation refines it. Below the
default 1,000 replicates:

- a persistent "Preview" banner states the replicate count;
- every power is shown with its MCSE (at power 0.80: ±5.7 points at 50 replicates, ±2.8 at
  200, ±1.3 at 1,000);
- the "for protocol text" copy button is disabled.

Scan results are the reason for this rule. In testing, `c_index=0.70` at 100 replicates gave
N = 18, where the `sample_size` docstring reports about 22. A run with 40 replicates reported
power 0.925 against a 0.80 target, with an interval of 142–248.

**Estimate.** These are judgement calls, not measurements. They assume one developer who knows
the codebase, a local single-user app on a desktop browser, and required-N and power modes only.

| Work | Days |
|---|---|
| G1 progress and cancel | 2.5 |
| G2 validation | 1 |
| G3 serialization and provenance | 1.5 |
| G4 figures, with #15 | 2 |
| G5 `to_cli`, `--version` | 1 |
| G6 `n_jobs` | 3 |
| **Core subtotal** | **11** |
| Input form and validation messages | 1.5 |
| Job runner: thread, polling, cancel, single-job lock | 1 |
| Outputs: power curve, KM preview, summary | 1.5 |
| Preview guard, provenance, export | 1 |
| `survpower gui`, `[gui]` extra, packaging, docs | 0.5 |
| `AppTest` suite and CI | 1 |
| **GUI subtotal** | **6.5** |
| **Total** | **17.5** |

**Not in the MVP:** sensitivity analyses, concordance benchmark and precision questions,
multi-value grids, hosting.

**Not tested, to settle during the build:**

- what happens to a running job when the tab closes;
- `st.fragment` on Streamlit versions older than 1.66.1;
- real run times for adjusted models at default budgets (the 4 to 6 minutes is extrapolated
  from 100 replicates).

## 5. Recommendation

**Go with Streamlit, using the thread-plus-fragment pattern, local first.**

Why Streamlit:

- `AppTest` gave the most direct test story, covering widgets and caching without a browser
  or a server.
- The thread-plus-fragment pattern worked for progress and for Cancel, and the app stays in
  one short script.
- Its project health is not a concern: it is widely used and released within the last month,
  though the same is true of Gradio, so this does not separate them.

What this accepts:

- **Threads:** Streamlit does not officially support multithreading in app code. The job
  thread makes no `st.*` calls, which is the option its docs describe as avoiding the problem.
- **Blocking runs:** the blocking pattern is unsuitable, because any widget click aborts it.
- **Python floor:** the `[gui]` extra needs Python ≥3.11 at the tested version, while the core
  stays at ≥3.9.
- **Weight:** it adds the most disk (+215 MB), though it is optional.
- **Launch:** `survpower gui` wraps a subprocess, and the app file must ship as package data.

Why not the others:

- **Gradio** is the runner-up, with an in-process launch and a built-in progress display. It
  adds the most packages (43, including `huggingface-hub` and `pydub`) and imports slowest,
  and its `cancels=` does not stop a running function.
- **marimo** is the choice if WASM becomes a priority, since its route is first-party. It is
  pre-1.0, and a cell blocks the click that would cancel it.
- **Panel** has the smallest added disk, but needs Python ≥3.12 at its latest release, and its
  slowdown is unresolved.

**Revisit if** the target user becomes clinicians on a hosted page. That points to marimo with
WASM, either pinned to Pyodide 0.29 or after `lifelines` lifts its `pandas<3` pin.

### Follow-up issues

All are in milestone M6.

| Issue | Title | Size | Depends on |
|---|---|---|---|
| #50 | `StudyDesign` accepts invalid designs and can return impossible sample sizes | S | |
| #51 | Add `to_dict`/`to_json` with provenance to result objects | S | |
| #52 | Return figures from the existing plotting functions | S | #15 |
| #53 | Add progress reporting and cooperative cancellation to all simulation loops | M | |
| #54 | Add `to_cli`, `--version` and provenance in CLI outputs | S | #51 |
| #55 | Add `n_jobs` to simulation calibration, keeping results identical | M | #53 |
| #56 | Build the Streamlit MVP behind `survpower gui` | L | All of the above, and #15 |

## Appendix A: test setup and snippets

**Setup:** Apple M1 Pro (8 cores), macOS, Python 3.12.16, one virtualenv per framework, headless
Brave driven by Playwright. Load average was about 5 to 7 during the runs, so timings are rough
and comparisons between frameworks are more reliable than absolute values.

**Job hook.** This emulates the hook proposed in G1 by replacing the loop's `tqdm`. It covers
`run_replicates` only and patches a global, so it is for prototypes only.

```python
class Cancelled(Exception): ...

def run_job(on_progress, should_stop):
    done = [0]
    def hooked(it, **kw):
        for x in it:
            if should_stop():
                raise Cancelled()
            yield x
            done[0] += 1
            if done[0] % 200 == 0:
                on_progress(done[0], TOTAL)
    survpower.scenario.tqdm = hooked
    return sample_size(hr=1.10, design=StudyDesign(n_sims=3000, n_sims_grid=3000))
```

**Streamlit, thread plus fragment.** The recommended pattern; the thread makes no `st.*` calls.

```python
s = st.session_state
if "job" not in s:
    s.job = dict(status="idle", cancel=threading.Event())
job = s.job

def worker(job):
    try:
        job["status"] = run_job(lambda d, t: job.update(status=f"progress {d}/{t}"),
                                job["cancel"].is_set)
    except Cancelled:
        job["status"] = "CANCELLED"

if st.button("Run"):
    job["cancel"].clear()
    threading.Thread(target=worker, args=(job,), daemon=True).start()
if st.button("Cancel"):
    job["cancel"].set()

@st.fragment(run_every=0.5)
def show():
    st.write(job["status"])
show()
```

**Gradio: `cancels=` needs an event as well.**

```python
ev = run.click(go, outputs=out)
stop.click(lambda: cancel.set(), cancels=[ev])   # without the fn, go() runs to completion
```

**marimo: use a thread so that the click is handled during the run.**

```python
def start(_):
    cancel.clear()
    mo.Thread(target=worker, daemon=True).start()
run = mo.ui.button(label="Run", on_click=start)
stop = mo.ui.button(label="Cancel", on_click=lambda _: cancel.set())
```

**Export as CLI command.** Every `StudyDesign` field has a flag. Round-trip tested on 5 designs
covering default and fixed censoring, non-evaluable patients, an accrual calendar with dropout
and `biomarker_sd`, adjusted one-sided tests, and the scan path. All matched on N, enrolled N
and confirmed power.

```python
FLAGS = dict(alpha="--alpha", sides="--sides", median_os="--median-os",
             censor_rate="--censor-rate", accrual_months="--accrual-months",
             followup_months="--followup-months", dropout_annual="--dropout",
             non_evaluable="--non-evaluable", n_covariates="--n-covariates", r2="--r2",
             biomarker_sd="--biomarker-sd", c_estimator="--c-estimator", c_tau="--c-tau",
             engine="--engine", n_sims="--n-sims", n_sims_grid="--n-sims-grid", seed="--seed")

def to_cli(design, *, hr=None, c_index=None, power=0.80):
    effect = ["--hr", repr(hr)] if hr is not None else ["--c-index", repr(c_index)]
    args = ["survpower", *effect, "--power", repr(power)]
    for f in dataclasses.fields(design):
        v = getattr(design, f.name)
        if v != f.default and v is not None:
            args += [FLAGS[f.name], str(v)]
    return shlex.join(args)
```
