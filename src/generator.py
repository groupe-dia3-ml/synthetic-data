"""Preserve the starter generator; its coefficients are experimental assumptions."""

import numpy as np
import pandas as pd

TRUE_BETAS = {
    "intercept": -4.80,
    "BMI": 0.08,
    "HighBP": 0.75,
    "HighChol": 0.55,
    "GenHlth": 0.45,
    "Age": 0.15,
    "PhysActivity": -0.30,
    "Smoker": 0.20,
    "HvyAlcoholConsump": -0.15,
    "Fruits": -0.10,
    "Veggies": -0.10,
}
FEATURES = [name for name in TRUE_BETAS if name != "intercept"] + ["Sex"]
CENTERS = {"BMI": 25.0, "GenHlth": 1.0, "Age": 1.0}
RAW_INTERCEPT = TRUE_BETAS["intercept"] - sum(TRUE_BETAS[k] * v for k, v in CENTERS.items())


def true_probabilities(features: pd.DataFrame) -> np.ndarray:
    logit = np.full(len(features), TRUE_BETAS["intercept"])
    for feature in FEATURES:
        logit += TRUE_BETAS.get(feature, 0.0) * (features[feature].to_numpy() - CENTERS.get(feature, 0.0))
    return 1.0 / (1.0 + np.exp(-logit))


def generate_clean_cohort(n_samples: int = 50000, seed: int = 42) -> pd.DataFrame:
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    r = np.random.RandomState(seed)
    age = r.choice(
        np.arange(1, 14),
        size=n_samples,
        p=[
            0.05,
            0.06,
            0.07,
            0.08,
            0.09,
            0.10,
            0.11,
            0.11,
            0.10,
            0.09,
            0.06,
            0.05,
            0.03,
        ],
    )
    sex = r.binomial(1, 0.48, size=n_samples)
    bmi = np.clip(r.normal(loc=27.5, scale=5.5, size=n_samples), 15.0, 55.0)
    phys = r.binomial(1, 0.75, size=n_samples)
    fruits = r.binomial(1, 0.60, size=n_samples)
    veggies = r.binomial(1, 0.80, size=n_samples)
    smoker = r.binomial(1, 0.40, size=n_samples)
    alcohol = r.binomial(1, 0.06, size=n_samples)
    prob_bp = 1 / (1 + np.exp(-(-4.0 + 0.10 * bmi + 0.25 * age)))
    high_bp = r.binomial(1, prob_bp)
    prob_chol = 1 / (1 + np.exp(-(-3.0 + 0.06 * bmi + 0.20 * age)))
    high_chol = r.binomial(1, prob_chol)
    gen_hlth = np.clip(
        np.round(1.5 + 0.05 * bmi + 0.15 * age + 0.5 * high_bp + r.normal(0, 0.5, n_samples)), 1, 5
    ).astype(int)
    frame = pd.DataFrame(
        {
            "Age": age,
            "Sex": sex,
            "BMI": bmi,
            "HighBP": high_bp,
            "HighChol": high_chol,
            "GenHlth": gen_hlth,
            "PhysActivity": phys,
            "Smoker": smoker,
            "HvyAlcoholConsump": alcohol,
            "Fruits": fruits,
            "Veggies": veggies,
        }
    )
    probability = true_probabilities(frame)
    frame["True_Probability"] = probability
    frame["Diabetes_Status"] = r.binomial(1, probability)
    return frame


def observed_features(
    clean: pd.DataFrame, measurement: float = 0, n_noise: int = 0, seed: int = 10042
) -> pd.DataFrame:
    """Return only observed inputs, replacing rather than appending clean measurements."""
    if not 0 <= measurement <= 0.2 or n_noise < 0:
        raise ValueError("Expected measurement in [0, .2] and nonnegative noise count")
    r = np.random.RandomState(seed)
    observed = clean[FEATURES].copy()
    # Pre-draw every perturbation to pair the random errors across severity levels.
    bmi_error = r.uniform(0.5, 3.0, len(clean)) * (measurement / 0.2)
    concealment = r.uniform(size=len(clean))
    if measurement > 0:
        overweight = clean["BMI"] > 27.0
        observed.loc[overweight, "BMI"] = np.round(clean.loc[overweight, "BMI"] - bmi_error[overweight], 1)
        observed.loc[(clean["Smoker"] == 1) & (concealment < measurement), "Smoker"] = 0
    for i in range(1, n_noise + 1):
        observed[f"Noise_Q{i}"] = r.normal(size=len(clean)) if i % 2 == 0 else r.binomial(1, 0.5, len(clean))
    return observed


def corrupt_training_labels(labels, rate: float, seed: int = 20042) -> np.ndarray:
    if not 0 <= rate <= 0.5:
        raise ValueError("Expected label flip rate in [0, .5]")
    labels = np.asarray(labels, dtype=int)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Expected binary labels")
    flips = np.random.RandomState(seed).uniform(size=len(labels)) < rate
    return np.where(flips, 1 - labels, labels)


def scenarios():
    yield {"name": "clean", "family": "clean", "level": 0.0, "measurement": 0.0, "labels": 0.0, "noise": 0}
    for rate in (0.05, 0.10, 0.20):
        yield {
            "name": f"measurement_{rate:.2f}",
            "family": "measurement",
            "level": rate,
            "measurement": rate,
            "labels": 0.0,
            "noise": 0,
        }
    for count in (10, 20):
        yield {
            "name": f"questions_{count}",
            "family": "questions",
            "level": float(count),
            "measurement": 0.0,
            "labels": 0.0,
            "noise": count,
        }
    for rate in (0.05, 0.10, 0.20):
        yield {
            "name": f"labels_{rate:.2f}",
            "family": "labels",
            "level": rate,
            "measurement": 0.0,
            "labels": rate,
            "noise": 0,
        }
    for rate in (0.05, 0.10, 0.20):
        yield {
            "name": f"combined_{rate:.2f}",
            "family": "combined",
            "level": rate,
            "measurement": rate,
            "labels": rate,
            "noise": 10,
        }
