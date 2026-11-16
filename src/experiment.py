"""Paired noise comparisons with clean test outcomes and known population weights."""

import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from .generator import (
    FEATURES,
    RAW_INTERCEPT,
    TRUE_BETAS,
    corrupt_training_labels,
    generate_clean_cohort,
    observed_features,
    scenarios,
)

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def candidates():
    models = {
        "logistic_unpenalized": (
            "logistic",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("clf", LogisticRegression(C=np.inf, max_iter=2000, random_state=42)),
                ]
            ),
        )
    }
    for c in (0.01, 0.1, 1.0):
        models[f"l1_C{c:g}"] = (
            "l1",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=c, solver="liblinear", l1_ratio=1, max_iter=2000, random_state=42
                        ),
                    ),
                ]
            ),
        )
    for depth in (3, 5):
        models[f"tree_depth{depth}"] = (
            "tree",
            Pipeline(
                [
                    ("clf", DecisionTreeClassifier(max_depth=depth, min_samples_leaf=50, random_state=42)),
                ]
            ),
        )
    return models


def raw_coefficients(fitted):
    """Undo standardization, including the raw-coordinate intercept."""
    scaler = fitted.named_steps["scale"]
    model = fitted.named_steps["clf"]
    weights = model.coef_[0] / scaler.scale_
    intercept = float(model.intercept_[0] - np.dot(weights, scaler.mean_))
    return weights, intercept


def predictive_metrics(y, probability, true_probability):
    return {
        "average_precision": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "latent_probability_mse": float(np.mean((probability - true_probability) ** 2)),
    }


def run(output_dir: str | Path | None = None, n_samples: int = 50000, seeds=(42, 43, 44)):
    with threadpool_limits(limits=2):
        return _run(Path(output_dir) if output_dir else ROOT / "results", n_samples, seeds)


def _run(out, n_samples, seeds):
    out.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    all_cv, records, recovery, cohort_audits = [], [], [], []
    coefficients: list[dict[str, Any]] = []
    tree_importances: list[dict[str, Any]] = []
    protocol = list(scenarios())
    source_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / "src").glob("*.py"))
    }
    for seed in seeds:
        clean = generate_clean_cohort(n_samples, seed)
        train, test = train_test_split(
            np.arange(n_samples), test_size=0.2, stratify=clean["Diabetes_Status"], random_state=seed
        )
        labels = clean["Diabetes_Status"].to_numpy()
        folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(train, labels[train]))
        cohort_audits.append(
            {
                "seed": seed,
                "rows": n_samples,
                "observed_prevalence": float(labels.mean()),
                "mean_true_probability": float(clean["True_Probability"].mean()),
                "train_rows": len(train),
                "test_rows": len(test),
                "train_index_sha256": hashlib.sha256(train.tobytes()).hexdigest(),
                "test_index_sha256": hashlib.sha256(test.tobytes()).hexdigest(),
            }
        )
        if seed == seeds[0]:
            clean.iloc[train].describe().T.to_csv(out / "development_describe.csv")
            clean.iloc[train][FEATURES].corr().to_csv(out / "development_correlations.csv")
        selections = {}
        for scenario in protocol:
            observed = observed_features(clean, scenario["measurement"], scenario["noise"], seed + 10000)
            y_training = corrupt_training_labels(labels[train], scenario["labels"], seed + 20000)
            candidates_for_scenario = []
            for name, (family, template) in candidates().items():
                fold_scores, fit_seconds = [], 0.0
                for fold, (fit, valid) in enumerate(folds, 1):
                    model = clone(template)
                    tick = perf_counter()
                    model.fit(observed.iloc[train[fit]], y_training[fit])
                    fit_seconds += perf_counter() - tick
                    prob = model.predict_proba(observed.iloc[train[valid]])[:, 1]
                    # Validation uses the labels available to training, including their corruption.
                    score = float(average_precision_score(y_training[valid], prob))
                    fold_scores.append(score)
                    all_cv.append(
                        {
                            "seed": seed,
                            "scenario": scenario["name"],
                            "model": name,
                            "family": family,
                            "fold": fold,
                            "average_precision": score,
                        }
                    )
                candidates_for_scenario.append(
                    {
                        "model": name,
                        "family": family,
                        "cv_ap_mean": float(np.mean(fold_scores)),
                        "cv_ap_sd": float(np.std(fold_scores, ddof=1)),
                        "fit_seconds": fit_seconds,
                    }
                )
            selections[scenario["name"]] = [
                max(
                    (item for item in candidates_for_scenario if item["family"] == family),
                    key=lambda item: item["cv_ap_mean"],
                )
                for family in ("logistic", "l1", "tree")
            ]
            print(f"Seed {seed}: selected {scenario['name']} on development data", flush=True)
        # Freeze every noise scenario for this cohort before any test evaluation.
        write_json(
            out / f"frozen_seed{seed}.json",
            {
                "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "seed": seed,
                "scenario_protocol": protocol,
                "selections": selections,
                "selection_metric": "five-fold AP on available (possibly corrupted) development labels",
                "source_sha256": source_hashes,
            },
        )
        true_probability = clean["True_Probability"].to_numpy()[test]
        y_test = labels[test]
        clean_fit = clone(candidates()["logistic_unpenalized"][1]).fit(
            clean.iloc[train][FEATURES], labels[train]
        )
        for scenario in protocol:
            observed = observed_features(clean, scenario["measurement"], scenario["noise"], seed + 10000)
            y_training = corrupt_training_labels(labels[train], scenario["labels"], seed + 20000)
            common = {
                "seed": seed,
                "scenario": scenario["name"],
                "noise_family": scenario["family"],
                "level": scenario["level"],
            }
            records.append(
                {
                    **common,
                    "model": "oracle",
                    "selected_configuration": "known_latent_probability",
                    **predictive_metrics(y_test, true_probability, true_probability),
                }
            )
            shifted = clean_fit.predict_proba(observed.iloc[test][FEATURES])[:, 1]
            records.append(
                {
                    **common,
                    "model": "clean_logistic_transfer",
                    "selected_configuration": "frozen_clean_fit",
                    **predictive_metrics(y_test, shifted, true_probability),
                }
            )
            for selection in selections[scenario["name"]]:
                template = candidates()[selection["model"]][1]
                tick = perf_counter()
                fitted = clone(template).fit(observed.iloc[train], y_training)
                fit_seconds = perf_counter() - tick
                prob = fitted.predict_proba(observed.iloc[test])[:, 1]
                family = selection["family"]
                records.append(
                    {
                        **common,
                        "model": family,
                        "selected_configuration": selection["model"],
                        "cv_ap_mean": selection["cv_ap_mean"],
                        "cv_ap_sd": selection["cv_ap_sd"],
                        "fit_seconds": fit_seconds,
                        **predictive_metrics(y_test, prob, true_probability),
                    }
                )
                if family == "tree":
                    tree_importances.extend(
                        {**common, "feature": feature, "importance": float(importance)}
                        for feature, importance in zip(
                            observed.columns, fitted.named_steps["clf"].feature_importances_, strict=True
                        )
                    )
                    continue
                estimated, intercept = raw_coefficients(fitted)
                truth = np.array([TRUE_BETAS.get(feature, 0.0) for feature in observed.columns])
                active, true_active = np.abs(estimated) > 1e-6, truth != 0
                recovery.append(
                    {
                        **common,
                        "model": family,
                        "signal_coefficient_rmse": float(
                            np.sqrt(np.mean((estimated[true_active] - truth[true_active]) ** 2))
                        ),
                        "intercept_estimate": intercept,
                        "raw_intercept_truth": RAW_INTERCEPT,
                        "intercept_error": intercept - RAW_INTERCEPT,
                        "selected_signal_count": int(np.sum(active & true_active)),
                        "selected_null_count": int(np.sum(active & ~true_active)),
                        "null_count": int(np.sum(~true_active)),
                        "noise_question_count": scenario["noise"],
                        "selected_noise_questions": int(
                            sum(
                                a and str(f).startswith("Noise_")
                                for a, f in zip(active, observed.columns, strict=True)
                            )
                        ),
                        "selected_feature_count": int(active.sum()),
                    }
                )
                coefficients.extend(
                    {
                        **common,
                        "model": family,
                        "feature": feature,
                        "true_coefficient": float(t),
                        "estimated_coefficient": float(e),
                    }
                    for feature, t, e in zip(observed.columns, truth, estimated, strict=True)
                )
        pd.DataFrame(records).to_csv(out / "test_metrics.csv", index=False)
        pd.DataFrame(recovery).to_csv(out / "recovery.csv", index=False)
    pd.DataFrame(all_cv).to_csv(out / "cv_folds.csv", index=False)
    pd.DataFrame(coefficients).to_csv(out / "coefficients.csv", index=False)
    pd.DataFrame(tree_importances).to_csv(out / "tree_importances.csv", index=False)
    pd.DataFrame(cohort_audits).to_csv(out / "cohort_audit.csv", index=False)
    table = pd.DataFrame(records)
    summary = (
        table.groupby(["scenario", "noise_family", "level", "model"])[
            ["average_precision", "roc_auc", "brier", "latent_probability_mse"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(filter(None, col)) for col in summary.columns]
    summary.to_csv(out / "summary.csv", index=False)
    make_figures(table, pd.DataFrame(coefficients), pd.DataFrame(recovery), out)
    manifest = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "n_samples": n_samples,
        "seeds": list(seeds),
        "scenario_count": len(protocol),
        "elapsed_seconds": perf_counter() - start,
        "source_sha256": source_hashes,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "scikit-learn", "matplotlib"]
        },
        "generator_note": "Unchanged starter population and outcome equation; prevalence is measured, not calibrated to CDC.",
        "uncertainty_note": "Three independent cohorts: SD across seeds is descriptive, not a 95% confidence interval.",
    }
    write_json(out / "run_manifest.json", manifest)
    print(f"Finished synthetic experiments: {out}", flush=True)
    return manifest


def make_figures(table, coefficient_table, recovery_table, out):
    colors = {"logistic": "#40678b", "l1": "#b95e40", "tree": "#638451", "oracle": "#888888"}
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for column, family in enumerate(["measurement", "questions", "labels", "combined"]):
        subset = table[table["noise_family"].isin([family, "clean"])]
        for model, color in colors.items():
            selected = subset[subset["model"] == model]
            for row, metric in enumerate(["average_precision", "latent_probability_mse"]):
                aggregated = selected.groupby("level")[metric].agg(["mean", "std"])
                axes[row, column].errorbar(
                    aggregated.index,
                    aggregated["mean"],
                    yerr=aggregated["std"],
                    marker="o",
                    capsize=3,
                    label=model,
                    color=color,
                )
                axes[row, column].set(
                    xlabel="Noise questions" if family == "questions" else "Noise severity",
                    ylabel=metric.replace("_", " "),
                    title=family,
                )
        axes[0, column].legend(fontsize=8)
    fig.suptitle("Held-out clean outcomes; mean ± SD over three generated cohorts")
    fig.tight_layout()
    fig.savefig(out / "robustness.png", dpi=150)
    plt.close(fig)
    selected = coefficient_table[
        (coefficient_table["scenario"] == "clean") & (coefficient_table["model"] == "logistic")
    ]
    agg = selected.groupby("feature").agg(
        truth=("true_coefficient", "first"),
        mean=("estimated_coefficient", "mean"),
        sd=("estimated_coefficient", "std"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    positions = np.arange(len(agg))
    axes[0].errorbar(agg["mean"], positions, xerr=agg["sd"], fmt="o", label="Estimated mean ± seed SD")
    axes[0].scatter(agg["truth"], positions, marker="x", color="#b95e40", label="Defined truth")
    axes[0].set(
        yticks=positions,
        yticklabels=agg.index,
        xlabel="Coefficient in raw input units",
        title="Clean logistic recovery",
    )
    axes[0].legend(fontsize=8)
    selection = recovery_table[
        (recovery_table["scenario"] == "questions_20") & (recovery_table["model"] == "l1")
    ]
    axes[1].bar(selection["seed"].astype(str), selection["selected_noise_questions"], color="#40678b")
    axes[1].set(
        xlabel="Cohort seed",
        ylabel="Selected noise questions (out of 20)",
        title="L1 does not guarantee perfect support recovery",
        ylim=(0, 20),
    )
    fig.tight_layout()
    fig.savefig(out / "recovery.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run()
