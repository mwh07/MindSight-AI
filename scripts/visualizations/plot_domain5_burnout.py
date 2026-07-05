#!/usr/bin/env python3
"""
MINDSIGHT Domain 5 — Occupational Burnout: SHAP Plotting (self-contained)

Loads the ordinal (3 cumulative classifiers) + quantile (3 regressors) XGBoost
bundle produced by train_domain5_burnout.py and produces:
  - Per-threshold ordinal SHAP summary + mean|SHAP| bar plots
    (">=Moderate", ">=High", ">=Severe")
  - A combined importance panel (mean|SHAP| averaged across the 3 ordinal heads)
  - SHAP summary + bar plot for the median (q50) quantile model, i.e. the
    Burnout Index point-estimate driver plot
  - A SHAP dependence plot for stress_score vs. social_support_score, to
    visualize the buffering interaction directly
  - A single-person SHAP waterfall plot (tier + index) as an example of the
    per-person "why did I get this score" report

No dependency on plot_utils.py -- this script reconstructs the same feature
engineering used at training time (RAW_WORK_FEATURES + engineered features +
one-hot gender) directly from the raw dataset CSV, so it stays correct even
if plot_utils.py's assumptions drift from the training pipeline.
"""

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Config -- mirrors train_domain5_burnout.py exactly
# --------------------------------------------------------------------------

RAW_WORK_FEATURES = [
    "work_hours_per_week", "meetings_per_day",
    "work_life_balance_score", "job_satisfaction_score",
    "deadline_pressure_score", "autonomy_score",
    "stress_score", "social_support_score",
]
TIER_ORDER = ["Low", "Moderate", "High", "Severe"]
SAMPLE_SIZE = 500

DISPLAY_NAMES = {
    "work_hours_per_week": "Work Hours / Week",
    "meetings_per_day": "Meetings / Day",
    "work_life_balance_score": "Work-Life Balance",
    "job_satisfaction_score": "Job Satisfaction",
    "deadline_pressure_score": "Deadline Pressure",
    "autonomy_score": "Autonomy",
    "stress_score": "Stress",
    "social_support_score": "Social Support",
    "age": "Age",
    "stress_x_support": "Stress x Social Support",
    "hours_over_50": "Hours Over 50/Week",
    "meeting_load_ratio": "Meeting Load Ratio",
    "gender_f": "Gender: F",
    "gender_m": "Gender: M",
    "gender_other": "Gender: Other",
}


def display(feature_key):
    return DISPLAY_NAMES.get(feature_key, feature_key)


def project_root():
    """Walks upward from this script's location until it finds a directory
    containing sibling 'models' and 'datasets' folders -- i.e. the actual
    MINDSIGHT project root -- rather than assuming a fixed nesting depth.
    This avoids breaking if the script is moved to a different folder depth
    (e.g. scripts/visualizations/ vs. scripts/), which is exactly what broke
    the previous single-level-up assumption in the Domain 6 plotting script."""
    current = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isdir(os.path.join(current, "models")) and os.path.isdir(os.path.join(current, "datasets")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise RuntimeError(
                "Could not locate the MINDSIGHT project root by walking up from "
                f"{os.path.dirname(os.path.abspath(__file__))}. Expected to find a "
                "directory containing sibling 'models' and 'datasets' folders. "
                "Check that this script hasn't been moved outside the project tree."
            )
        current = parent


def ensure_output_dir():
    out_dir = os.path.join(project_root(), "results", "aggregate_analysis", "domain5_burnout")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# --------------------------------------------------------------------------
# Load models + metadata
# --------------------------------------------------------------------------

def load_domain5_bundle():
    model_dir = os.path.join(project_root(), "models", "saved_states")
    meta_path = os.path.join(model_dir, "domain5_burnout_metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Domain 5 metadata not found at {meta_path}")

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    ordinal_models = {}
    for label, filename in metadata["model_files"]["ordinal"].items():
        bst = xgb.Booster()
        bst.load_model(os.path.join(model_dir, filename))
        ordinal_models[label] = bst

    quantile_models = {}
    for label, filename in metadata["model_files"]["quantile"].items():
        bst = xgb.Booster()
        bst.load_model(os.path.join(model_dir, filename))
        quantile_models[label] = bst

    return ordinal_models, quantile_models, metadata


# --------------------------------------------------------------------------
# Reconstruct features from raw dataset -- must mirror training exactly
# --------------------------------------------------------------------------

def load_and_engineer_features(metadata):
    data_path = os.path.join(project_root(), "datasets", "tech_burnout_2026_clean.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Domain 5 dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    required = RAW_WORK_FEATURES + ["age", "gender"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns for feature reconstruction: {missing}")

    for col in RAW_WORK_FEATURES + ["age"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    df["stress_x_support"] = df["stress_score"] * df["social_support_score"]
    df["hours_over_50"] = (df["work_hours_per_week"] - 50).clip(lower=0)
    df["meeting_load_ratio"] = df["meetings_per_day"] / df["work_hours_per_week"].replace(0, np.nan)
    df["meeting_load_ratio"] = df["meeting_load_ratio"].fillna(df["meeting_load_ratio"].median())

    df["gender"] = df["gender"].astype(str).str.strip().str.lower()
    gender_dummies = pd.get_dummies(df["gender"], prefix="gender", dtype=int)

    feature_order = metadata["feature_order"]
    # Ensure every expected gender dummy column exists, even if this dataset
    # sample happens not to contain every category (e.g. no "other" rows).
    for col in metadata["gender_dummy_columns"]:
        if col not in gender_dummies.columns:
            gender_dummies[col] = 0

    X = pd.concat([df[RAW_WORK_FEATURES + ["age", "stress_x_support", "hours_over_50", "meeting_load_ratio"]],
                   gender_dummies], axis=1)
    X = X.reindex(columns=feature_order, fill_value=0)
    return X, df


def dmatrix_for(booster, X):
    return xgb.DMatrix(X, feature_names=list(X.columns))


# --------------------------------------------------------------------------
# Ordinal head plots
# --------------------------------------------------------------------------

def plot_ordinal_shap(ordinal_models, X_sample, out_dir):
    feature_order = list(X_sample.columns)
    display_labels = [display(f) for f in feature_order]

    all_mean_abs = {}
    for label, booster in ordinal_models.items():
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(X_sample)

        X_display = X_sample.copy()
        X_display.columns = display_labels

        safe_label = label.replace(">=", "ge_")

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_display, show=False)
        plt.title(f"Domain 5 — SHAP Summary — P[tier {label}]")
        plt.tight_layout()
        path = os.path.join(out_dir, f"domain5_ordinal_{safe_label}_shap_summary.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  | Saved: {path}")

        mean_abs = np.mean(np.abs(shap_values), axis=0)
        all_mean_abs[label] = mean_abs
        sorted_idx = np.argsort(mean_abs)[::-1]

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(feature_order)), mean_abs[sorted_idx], color="steelblue")
        plt.yticks(range(len(feature_order)), [display_labels[i] for i in sorted_idx])
        plt.xlabel("Mean |SHAP value|")
        plt.title(f"Domain 5 — Feature Importance — P[tier {label}]")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        path = os.path.join(out_dir, f"domain5_ordinal_{safe_label}_mean_shap_bar.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  | Saved: {path}")

    # Combined importance panel -- averaged across the 3 cumulative heads.
    combined = np.mean(np.vstack(list(all_mean_abs.values())), axis=0)
    sorted_idx = np.argsort(combined)[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_order)), combined[sorted_idx], color="darkslateblue")
    plt.yticks(range(len(feature_order)), [display_labels[i] for i in sorted_idx])
    plt.xlabel("Mean |SHAP value| (averaged across ordinal thresholds)")
    plt.title("Domain 5 — Combined Burnout-Tier Driver Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    path = os.path.join(out_dir, "domain5_ordinal_combined_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  | Saved: {path}")


# --------------------------------------------------------------------------
# Quantile head plots (median model = Burnout Index point-estimate driver)
# --------------------------------------------------------------------------

def plot_quantile_shap(quantile_models, X_sample, out_dir):
    feature_order = list(X_sample.columns)
    display_labels = [display(f) for f in feature_order]

    median_booster = quantile_models.get("q50")
    if median_booster is None:
        print("  | [WARNING] q50 quantile model not found in bundle; skipping quantile SHAP plots.")
        return

    explainer = shap.TreeExplainer(median_booster)
    shap_values = explainer.shap_values(X_sample)

    X_display = X_sample.copy()
    X_display.columns = display_labels

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_display, show=False)
    plt.title("Domain 5 — SHAP Summary — Burnout Index (median estimate)")
    plt.tight_layout()
    path = os.path.join(out_dir, "domain5_quantile_q50_shap_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  | Saved: {path}")

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_order)), mean_abs[sorted_idx], color="teal")
    plt.yticks(range(len(feature_order)), [display_labels[i] for i in sorted_idx])
    plt.xlabel("Mean |SHAP value|")
    plt.title("Domain 5 — Burnout Index Driver Importance (median model)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    path = os.path.join(out_dir, "domain5_quantile_q50_mean_shap_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  | Saved: {path}")

    # Dependence plot: surfaces the stress x social-support buffering effect
    # directly, colored by social_support_score.
    if "stress_score" in feature_order and "social_support_score" in feature_order:
        plt.figure(figsize=(9, 7))
        shap.dependence_plot(
            "stress_score", shap_values, X_sample,
            interaction_index="social_support_score",
            show=False,
        )
        plt.title("Domain 5 — Stress SHAP vs. Social Support (buffering effect)")
        plt.tight_layout()
        path = os.path.join(out_dir, "domain5_stress_support_dependence.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  | Saved: {path}")

    return explainer, shap_values


# --------------------------------------------------------------------------
# Single-person example: waterfall plot for one respondent's Burnout Index
# --------------------------------------------------------------------------

def plot_example_person_waterfall(quantile_models, X_sample, out_dir, row_idx=0):
    median_booster = quantile_models.get("q50")
    if median_booster is None or len(X_sample) == 0:
        return

    explainer = shap.TreeExplainer(median_booster)
    person_X = X_sample.iloc[[row_idx]]
    shap_values = explainer(person_X)  # Explanation object for waterfall API

    display_labels = [display(f) for f in X_sample.columns]
    shap_values.feature_names = display_labels

    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title("Domain 5 — Example: One Respondent's Burnout Index Drivers")
    plt.tight_layout()
    path = os.path.join(out_dir, "domain5_example_person_waterfall.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  | Saved: {path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    print("\n[Domain 5] Occupational Burnout — Ordinal + Quantile SHAP Plots")
    out_dir = ensure_output_dir()
    try:
        ordinal_models, quantile_models, metadata = load_domain5_bundle()
        X, _ = load_and_engineer_features(metadata)

        if len(X) > SAMPLE_SIZE:
            X_sample = X.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        else:
            X_sample = X.reset_index(drop=True)

        print(f"  | Using {len(X_sample)} rows for SHAP computation "
              f"(features: {list(X_sample.columns)})")

        print("  | Computing SHAP for ordinal (tier) heads...")
        plot_ordinal_shap(ordinal_models, X_sample, out_dir)

        print("  | Computing SHAP for quantile (Burnout Index) head...")
        plot_quantile_shap(quantile_models, X_sample, out_dir)

        print("  | Generating example single-person waterfall plot...")
        plot_example_person_waterfall(quantile_models, X_sample, out_dir)

        print("  | Domain 5 plots generated successfully.")
    except Exception as e:
        print(f"  | [FAILED] Domain 5 plotting failed: {e}")
        raise


if __name__ == "__main__":
    main()