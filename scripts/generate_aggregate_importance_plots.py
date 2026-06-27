#!/usr/bin/env python3
"""
MINDSIGHT — Aggregate Feature-Importance Plots (Domains 4, 5, 6)

This script loads the saved models for domains 4, 5, and 6, then computes
SHAP values (for tree-based models) or coefficient magnitudes (for LogisticRegression)
across the entire training dataset. It generates population-level summary plots
(saved as image files in results/aggregate_analysis/) for use in the project report.

It does not modify any saved model artifacts, inference_wrappers.py, or profile_aggregator.py.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

# Add project root to path if needed
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import feature mapping (without modifying inference_wrappers)
try:
    from models.feature_mappings import FEATURE_TRANSLATION_MAP
except ImportError:
    FEATURE_TRANSLATION_MAP = {}

def get_display_name(feature_key):
    """Return human-readable name from mapping, fallback to feature key."""
    return FEATURE_TRANSLATION_MAP.get(feature_key, feature_key)

def ensure_output_dir():
    """Create results/aggregate_analysis/ if it doesn't exist."""
    out_dir = os.path.join(PROJECT_ROOT, "results", "aggregate_analysis")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def load_domain4_models():
    """Load the two RandomForest models for domain 4, and their feature lists."""
    model_path = os.path.join(PROJECT_ROOT, "models", "saved_states", "domain4_digital_social.pkl")
    meta_path = os.path.join(PROJECT_ROOT, "models", "saved_states", "domain4_digital_social_metadata.json")
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        # Fallback to older naming if needed
        alt_model = os.path.join(PROJECT_ROOT, "models", "saved_states", "domain4_multitask.pkl")
        alt_meta = os.path.join(PROJECT_ROOT, "models", "saved_states", "domain4_multitask_metadata.json")
        if os.path.exists(alt_model) and os.path.exists(alt_meta):
            model_path, meta_path = alt_model, alt_meta
        else:
            raise FileNotFoundError("Domain 4 model files not found.")

    with open(model_path, "rb") as f:
        model_payload = pickle.load(f)
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    # Extract the two models
    addiction_model = None
    loneliness_model = None
    if isinstance(model_payload, dict):
        for key in ["addiction_model", "iat_model", "internet_addiction_model", "model_addiction", "rf_iat"]:
            if key in model_payload:
                addiction_model = model_payload[key]
                break
        for key in ["loneliness_model", "lone_model", "model_loneliness", "rf_lone"]:
            if key in model_payload:
                loneliness_model = model_payload[key]
                break
        # If still None, try to infer from feature_names_in_
        if addiction_model is None or loneliness_model is None:
            candidates = [v for v in model_payload.values() if hasattr(v, "predict")]
            for cand in candidates:
                fit_features = list(getattr(cand, "feature_names_in_", []))
                if any(f.startswith("IAT") for f in fit_features) and addiction_model is None:
                    addiction_model = cand
                elif any(f.startswith("loneliness") for f in fit_features) and loneliness_model is None:
                    loneliness_model = cand
    else:
        raise ValueError("Model payload is not a dict; cannot extract two models.")

    if addiction_model is None or loneliness_model is None:
        raise ValueError("Could not resolve both addiction and loneliness models.")

    return addiction_model, loneliness_model, metadata

def load_domain5_model():
    """Load XGBoost model for domain 5 and its metadata."""
    model_path = os.path.join(PROJECT_ROOT, "models", "saved_states", "domain5_burnout.json")
    meta_path = os.path.join(PROJECT_ROOT, "models", "saved_states", "domain5_burnout_metadata.json")
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Domain 5 model files not found.")
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    bst = xgb.Booster()
    bst.load_model(model_path)
    return bst, metadata

def load_domain6_model():
    """Load LogisticRegression + IsolationForest for domain 6."""
    model_path = os.path.join(PROJECT_ROOT, "models", "saved_states", "domain6_clinical.pkl")
    meta_path = os.path.join(PROJECT_ROOT, "models", "saved_states", "domain6_clinical_metadata.json")
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Domain 6 model files not found.")
    with open(model_path, "rb") as f:
        model_payload = pickle.load(f)
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    classifier = model_payload["classifier"]
    # In inference_wrappers, they store coef_ and intercept_ manually after fit
    # but here we just need the classifier itself
    return classifier, metadata

def get_dataset(domain_name):
    """Load the appropriate dataset for a domain."""
    if domain_name == "domain4":
        path = os.path.join(PROJECT_ROOT, "datasets", "internet_phq_loneliness_clean.csv")
    elif domain_name == "domain5":
        path = os.path.join(PROJECT_ROOT, "datasets", "tech_burnout_2026_clean.csv")
    elif domain_name == "domain6":
        path = os.path.join(PROJECT_ROOT, "datasets", "ocd_symptoms_clean.csv")
    else:
        raise ValueError(f"Unknown domain: {domain_name}")
    if not os.path.exists(path):
        # If missing, we can still generate synthetic? But per prompt, they should exist.
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    # Clean up column names to match schema (remove spaces, etc.)
    df.columns = df.columns.str.strip()
    return df

def generate_shap_summary_and_bar(model, X, feature_names, model_name, out_dir, sample_size=500):
    """
    For a tree-based model (RandomForest or XGBoost), compute SHAP values
    and produce two plots: a beeswarm summary and a mean absolute bar chart.
    """
    # Ensure X is a DataFrame with proper column names
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=feature_names)
    else:
        # Ensure columns match feature_names
        if list(X.columns) != feature_names:
            X = X[feature_names]

    # Use a subset if dataset is large
    if sample_size is not None and len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X

    # Create SHAP explainer
    if isinstance(model, (RandomForestRegressor, xgb.XGBRegressor)):
        explainer = shap.TreeExplainer(model)
    else:
        # For XGBoost booster, use TreeExplainer directly
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_sample)

    # 1. Beeswarm summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"{model_name}_shap_summary.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved SHAP summary plot: {save_path}")

    # 2. Mean absolute SHAP bar chart
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_vals = mean_abs_shap[sorted_idx]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(len(sorted_features)), sorted_vals, color='steelblue')
    plt.yticks(range(len(sorted_features)), [get_display_name(f) for f in sorted_features])
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"Mean Absolute SHAP Value per Feature\n{model_name}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    save_path_bar = os.path.join(out_dir, f"{model_name}_mean_shap_bar.png")
    plt.savefig(save_path_bar, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved mean SHAP bar chart: {save_path_bar}")

    return save_path, save_path_bar

def generate_coefficient_plots(classifier, X, feature_names, out_dir):
    """
    For LogisticRegression (one-vs-rest), produce per-class coefficient bar charts
    and a mean absolute coefficient magnitude bar chart.
    """
    # Extract coefficients. classifier is OneVsRestClassifier; we need to get estimators.
    # In our saved model, we attached coef_ and intercept_ manually,
    # but the actual estimator list is in classifier.estimators_
    if hasattr(classifier, "estimators_"):
        coefs = np.vstack([est.coef_[0] for est in classifier.estimators_])
        classes = classifier.classes_
    else:
        # Fallback: if it's a single LogisticRegression
        coefs = classifier.coef_
        classes = np.arange(coefs.shape[0])

    # 1. Per-class bar charts
    for i, class_label in enumerate(classes):
        plt.figure(figsize=(10, 6))
        coef = coefs[i]
        sorted_idx = np.argsort(np.abs(coef))[::-1]
        sorted_features = [feature_names[j] for j in sorted_idx]
        sorted_vals = coef[sorted_idx]
        colors = ['green' if v > 0 else 'red' for v in sorted_vals]
        plt.barh(range(len(sorted_features)), sorted_vals, color=colors)
        plt.yticks(range(len(sorted_features)), [get_display_name(f) for f in sorted_features])
        plt.xlabel("Coefficient value")
        plt.title(f"Class {class_label} — Symptom Coefficients")
        plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
        plt.tight_layout()
        save_path = os.path.join(out_dir, f"domain6_class_{class_label}_coefficients.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved class {class_label} coefficient plot: {save_path}")

    # 2. Mean absolute coefficient magnitude (overall importance)
    mean_abs = np.mean(np.abs(coefs), axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_vals = mean_abs[sorted_idx]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_features)), sorted_vals, color='purple')
    plt.yticks(range(len(sorted_features)), [get_display_name(f) for f in sorted_features])
    plt.xlabel("Mean |Coefficient| across classes")
    plt.title("Average Coefficient Magnitude per Symptom")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    save_path = os.path.join(out_dir, "domain6_mean_abs_coefficients.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved mean absolute coefficient plot: {save_path}")

    return save_path

def main():
    print("🚀 Generating aggregate importance plots for Domains 4, 5, 6...")
    out_dir = ensure_output_dir()
    print(f"📁 Output directory: {out_dir}")

    # ---------- Domain 4 ----------
    print("\n📊 Domain 4 — Digital & Social (RandomForest)")
    try:
        addiction_model, loneliness_model, _ = load_domain4_models()
        df4 = get_dataset("domain4")
        # Build feature matrices exactly as inference_wrappers does
        # For addiction: IAT features only; for loneliness: loneliness features only
        iat_features = [f"IAT{i}" for i in range(1, 11)]
        loneliness_features = [f"loneliness{i}" for i in range(1, 7)]
        # Ensure columns exist
        X_iat = df4[iat_features].copy()
        X_lone = df4[loneliness_features].copy()
        # Drop rows with missing values (should be minimal)
        X_iat = X_iat.dropna()
        X_lone = X_lone.dropna()

        print("  ⏳ Computing SHAP for Internet Addiction model...")
        generate_shap_summary_and_bar(
            addiction_model,
            X_iat,
            iat_features,
            "domain4_addiction",
            out_dir,
            sample_size=500
        )

        print("  ⏳ Computing SHAP for Loneliness model...")
        generate_shap_summary_and_bar(
            loneliness_model,
            X_lone,
            loneliness_features,
            "domain4_loneliness",
            out_dir,
            sample_size=500
        )
        print("  ✅ Domain 4 plots generated.")
    except Exception as e:
        print(f"  ❌ Domain 4 failed: {e}")

    # ---------- Domain 5 ----------
    print("\n📊 Domain 5 — Occupational Burnout (XGBoost)")
    try:
        bst, meta5 = load_domain5_model()
        df5 = get_dataset("domain5")
        features5 = meta5["features"]  # exactly the 8 work-related features
        X5 = df5[features5].dropna()
        print("  ⏳ Computing SHAP for XGBoost model...")
        generate_shap_summary_and_bar(
            bst,
            X5,
            features5,
            "domain5_burnout",
            out_dir,
            sample_size=500
        )
        print("  ✅ Domain 5 plots generated.")
    except Exception as e:
        print(f"  ❌ Domain 5 failed: {e}")

    # ---------- Domain 6 ----------
    print("\n📊 Domain 6 — Severe Clinical Screening (LogisticRegression)")
    try:
        classifier, meta6 = load_domain6_model()
        df6 = get_dataset("domain6")
        features6 = meta6["features"]
        X6 = df6[features6].dropna()
        print("  ⏳ Computing coefficient plots...")
        generate_coefficient_plots(classifier, X6, features6, out_dir)
        print("  ✅ Domain 6 plots generated.")
    except Exception as e:
        print(f"  ❌ Domain 6 failed: {e}")

    print("\n✅ All plots completed. Check the output directory:")
    print(f"   {out_dir}")

if __name__ == "__main__":
    main()