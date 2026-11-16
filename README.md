# Synthetic data: parameter recovery and noise robustness

Phase 2 of the ESILV A4 DIA3 machine learning project. The supplied generator is
preserved, and the completed experiment measures coefficient recovery and
robustness to controlled survey errors. The assumptions are experimental and are
not claimed to be epidemiological estimates.

## Results

Three independent cohorts (seeds 42, 43, 44), each with 50,000 observations, are
split into 40,000 development and 10,000 test observations. The 12 scenarios use
the same split within each cohort. Five-fold development validation selects L1
logistic strength and tree depth before test evaluation. Unpenalized logistic
regression provides a correctly specified clean-model baseline.

| Scenario | Logistic test AP, mean ± SD | Brier, mean | Latent-probability MSE, mean |
| --- | ---: | ---: | ---: |
| Clean | 0.4470 ± 0.0187 | 0.1281 | 0.000052 |
| 20 irrelevant questions | 0.4459 ± 0.0192 | 0.1282 | 0.000135 |
| 20% development label flips | 0.4456 ± 0.0193 | 0.1494 | 0.021386 |
| Combined severity 0.20 + 10 questions | 0.4423 ± 0.0172 | 0.1495 | 0.021558 |

Clean logistic signal-coefficient RMSE is 0.0297 on average across the three
cohorts. L1 selected 2, 2 and 19 of 20 irrelevant questions in the corresponding
question-only experiments: predictive selection does not guarantee exact support
recovery. Uncertainty is descriptive SD across three seeds, not a 95% interval.

The unchanged generator yields 18.89–19.17% positives, rather than the 14% stated
in the starter prose. Its intercept is not recalibrated to the CDC dataset.

Read the [executed notebook](notebooks/01_synthetic_data_starter.ipynb) for the full
protocol, results, equations and limitations. The historical filename is retained;
the notebook now contains the completed study. Evidence is in [results](results).

## Noise protocol

- Clean: original features and labels, no extra questions.
- Measurement: severity 0.05, 0.10 or 0.20. For BMI above 27, subtract
  `Uniform(0.5, 3.0) × severity / 0.2` and round to one decimal. A fraction
  `severity` of smokers report no smoking. Both development and test inputs
  receive measurement error.
- Questions: append 10 or 20 independent alternating Bernoulli/Gaussian features.
- Labels: flip 5%, 10% or 20% of development labels symmetrically. Test outcomes
  remain clean; development validation uses the same available noisy labels.
- Combined: matching measurement severity and label-flip rate, plus 10 questions.

The true probability, outcome and clean backup measurements never enter predictors.
`Sex` is an additional known null feature. Noise draws are paired across severity
levels. The privileged oracle evaluates known clean probabilities, and a frozen
clean logistic model is also evaluated on corrupted measurements to distinguish
measurement shift from training-label corruption.

The known equation is logistic with centered BMI, general health and age terms.
Its centered intercept is −4.80; its raw-coordinate intercept is −7.40. Estimated
weights and intercept are transformed back from standardized training coordinates
before comparing with truth. Trees are compared using feature importances, not
interpreted as estimates of signed logistic coefficients.

## Reproduce

Tested with Python 3.14.10. No external dataset is required.

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
MPLBACKEND=Agg .venv/bin/python -m src.experiment
.venv/bin/python scripts/execute_notebook.py
```

The full experiment took about 30 seconds on the development machine. Results
include 1,080 development fits (six configurations, five folds, twelve scenarios,
three cohorts), frozen selections per seed, held-out metrics, coefficient recovery,
noise support, tree importances, environment details and figures.

The notebook checks saved source hashes and displays saved results by default.
To recompute all experiments and notebook outputs in one command:

```sh
RERUN_EXPERIMENTS=1 MPLBACKEND=Agg .venv/bin/python scripts/execute_notebook.py
```

## Verification

```sh
.venv/bin/python -m ruff check src tests scripts
.venv/bin/python -m ruff format --check src tests scripts
.venv/bin/python -m mypy src scripts
.venv/bin/python -m pytest -q
git diff --check
```

Tests check deterministic generation, the known logit equation, absence of oracle
and clean-feature leakage, nonmutation, nested label flips, coefficient coordinate
conversion and clean-cohort recovery.

## Interpretation and limitations

For symmetric label-flip rate `r`, the observed probability is
`r + (1 − 2r) × p`. This compresses probabilities toward 0.5 but preserves their
population ranking for `r < 0.5`, helping explain why AP changes less than Brier
loss and probability MSE. Brier loss includes outcome randomness; latent MSE
compares directly with the known probability.

The true model is logistic, many lifestyle variables are independent, errors are
stylized, and three cohorts provide limited replication. This is not a realistic
patient simulator or evidence of real-world causal relationships. Different
prevalence and target construction prevent direct performance ranking against CDC.
Do not infer disease prevention from the negative assumed alcohol coefficient.

The implementation follows scikit-learn's
[pipeline guidance](https://scikit-learn.org/stable/common_pitfalls.html) and
[logistic-regression interface](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
The population generator and coefficient assumptions originate in this repository's
supplied starter notebook; their original distributions and equation are preserved.
