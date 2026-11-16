# Synthetic study conventions

This repository implements the controlled-data phase of the DIA3 project.
Keep deliverables in English and separate assumed truth from empirical estimates.

## Run and verify

Use Python 3.14 and install `requirements-dev.txt` in a virtual environment.
`python -m src.experiment` runs all three cohorts and twelve scenarios.
`python scripts/execute_notebook.py` validates source fingerprints and saves
executed notebook outputs. `RERUN_EXPERIMENTS=1` recomputes experiments first.

Before delivery: `python -m ruff check src tests scripts`,
`python -m ruff format --check src tests scripts`, `python -m mypy src scripts`,
`python -m pytest -q`, notebook execution, and `git diff --check`.
Use `MPLBACKEND=Agg` for headless experiments.

## Invariants and pitfalls

- Preserve the supplied clean-cohort distributions and logistic equation unless
  the user explicitly asks to change them. Their prevalence is about 19%, not 14%.
- Coefficients are simulation assumptions, not epidemiological estimates.
- Never include true probabilities, outcomes or clean backup measurements as inputs.
- Replace noisy measurements under the same names, keeping the feature schema clear.
- Pair splits and perturbations across scenarios; use independent cohort seeds.
- Corrupt development labels only; keep test outcomes clean.
- Tune using available development labels, including their noise. Freeze every
  scenario before evaluating the cohort's test set.
- Fit every scaler in its training fold. Compare coefficients in raw input units.
- The centered intercept is -4.8; the raw-coordinate intercept is -7.4.
- Sex is a known null feature. Count it separately from added noise questions.
- AP-selected L1 regularization does not guarantee perfect variable selection.
- SD across three seeds is descriptive, not a confidence interval.
- Recompute outputs after source changes so notebook fingerprints remain valid.

Contribute on a feature/fix branch using conventional commits and PRs to main.
