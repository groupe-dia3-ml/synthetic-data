# Phase 2: Synthetic Data Generation and Noise Robustness

Phase 2 of the Machine Learning final project (ESILV, group DIA3).

This repository contains the synthetic data generator where ground-truth epidemiological risk relationships are mathematically defined, paired with controlled survey noise injection to evaluate model parameter recovery and robustness.

## Structure

```
synthetic-data/
├── notebooks/
│   └── 01_synthetic_data_starter.ipynb   # Main notebook outlining the synthetic design and experiments
├── .gitignore
├── requirements.txt
└── README.md
```

## Methodology

1. Mathematical Ground Truth: True logistic risk function with known weights linking BMI, blood pressure, cholesterol, general health, and lifestyle to diabetes probability.
2. Synthetic Cohort: Generating 50,000 synthetic patient profiles with realistic correlation structures.
3. Noise Protocols: Injecting self-reporting errors (BMI under-reporting, smoking concealment) and adding uninformative noise questions.
4. Model Recovery: Assessing whether simple models (Logistic Regression, Decision Trees) can accurately recover the original weights despite noise.
