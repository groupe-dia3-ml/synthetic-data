import numpy as np
import pytest

from src.experiment import candidates, raw_coefficients
from src.generator import (
    FEATURES,
    RAW_INTERCEPT,
    TRUE_BETAS,
    corrupt_training_labels,
    generate_clean_cohort,
    observed_features,
    true_probabilities,
)


def test_generator_reproducibility_and_known_equation():
    data = generate_clean_cohort(1000, 42)
    assert data.equals(generate_clean_cohort(1000, 42))
    weights = np.array([TRUE_BETAS.get(f, 0) for f in FEATURES])
    logits = RAW_INTERCEPT + data[FEATURES].to_numpy() @ weights
    np.testing.assert_allclose(data["True_Probability"], 1 / (1 + np.exp(-logits)))
    np.testing.assert_allclose(true_probabilities(data), data["True_Probability"])
    assert data["Age"].between(1, 13).all()
    assert data["GenHlth"].between(1, 5).all()


def test_observed_inputs_cannot_leak_clean_measurements_or_truth():
    data = generate_clean_cohort(1000)
    before = data.copy(deep=True)
    noisy = observed_features(data, 0.2, 20)
    assert data.equals(before)
    assert set(noisy.columns) == set(FEATURES + [f"Noise_Q{i}" for i in range(1, 21)])
    assert "True_Probability" not in noisy and "Diabetes_Status" not in noisy
    assert (noisy["Smoker"] <= data["Smoker"]).all()
    assert (noisy.loc[data["BMI"] > 27, "BMI"] < data.loc[data["BMI"] > 27, "BMI"]).all()
    assert observed_features(data).equals(data[FEATURES])


def test_label_noise_is_nested_and_does_not_mutate_labels():
    clean = np.tile([0, 0, 1], 1000)
    before = clean.copy()
    low = corrupt_training_labels(clean, 0.05)
    high = corrupt_training_labels(clean, 0.2)
    assert np.all((low != clean) <= (high != clean))
    np.testing.assert_array_equal(clean, before)
    np.testing.assert_array_equal(corrupt_training_labels(clean, 0), clean)
    with pytest.raises(ValueError):
        corrupt_training_labels(clean, 0.6)


def test_standardized_coefficients_are_back_transformed_correctly():
    data = generate_clean_cohort(5000)
    model = candidates()["logistic_unpenalized"][1].fit(data[FEATURES], data["Diabetes_Status"])
    weights, intercept = raw_coefficients(model)
    raw_probability = 1 / (1 + np.exp(-(data[FEATURES].to_numpy() @ weights + intercept)))
    np.testing.assert_allclose(model.predict_proba(data[FEATURES])[:, 1], raw_probability, atol=1e-10)


def test_clean_parameter_recovery_on_large_independent_cohort():
    data = generate_clean_cohort(50000, 123)
    model = candidates()["logistic_unpenalized"][1].fit(data[FEATURES], data["Diabetes_Status"])
    estimated, _ = raw_coefficients(model)
    truth = np.array([TRUE_BETAS.get(f, 0) for f in FEATURES])
    assert np.sqrt(np.mean((estimated - truth) ** 2)) < 0.08
