# Writing a tutorial

This note is for contributors. It is not published (`exclude_docs` in `mkdocs.yml`).

Tutorials are [Jupytext "percent"](https://jupytext.readthedocs.io/en/latest/formats-scripts.html)
`.py` files: they diff cleanly, run as plain Python, and `mkdocs-jupyter` renders them
as executed notebooks.

## Adding one

1. Create `docs/tutorials/NN_short_name.py`.
2. Add it to the "Tutorials" section of `nav` in `mkdocs.yml`.
3. Run it: `SURVPOWER_TUTORIAL_SIMS=50 MPLBACKEND=Agg python docs/tutorials/NN_short_name.py`.
4. Build the docs: `mkdocs build --strict`.

## The budget variable

Every tutorial reads:

```python
N_SIMS = int(os.environ.get("SURVPOWER_TUTORIAL_SIMS", 1000))
```

| Where | Budget | What it is for |
|---|---|---|
| PR CI (`tutorials` job) | 50 | smoke test; fails if a tutorial raises |
| Docs build of `main` | default (1000) | published numbers |
| Docs build of other branches | 50 (`scripts/build_docs_site.sh`) | keeps the multi-branch build fast |

Prose must not quote a simulated number, since the number changes with the budget.
Print it from code instead. Checks of simulated values run only when `N_SIMS >= 1000`.

## Style template

Each tutorial has these parts, in this order:

1. **Question**
2. **Design assumptions**
3. Code
4. A table with MCSEs or intervals (every simulated number carries one)
5. Interpretation
6. Agreement with an analytic result (an `assert`, so a tutorial cannot quietly contradict it)
7. "What would change this"

Do not import `survpower.plotting` in a tutorial: it sets the Agg backend on import, which
stops figures from rendering inline until #15 removes that call. Draw with `matplotlib.pyplot`.
