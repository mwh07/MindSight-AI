#!/usr/bin/env python3
"""
MINDSIGHT — Aggregate Feature-Importance Plot for Domain 3 (Mood & Sleep)

This script loads the trained LightGBM model for Domain 3 and the
nhanes_joined_mood_sleep.csv dataset, computes SHAP values across the
full dataset (or a representative sample), and generates:
- A beeswarm summary plot (shap.summary_plot)
- A mean absolute SHAP value bar chart

Saves plots to: results/aggregate_analysis/
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import lightgbm as lgb

# Add project root to path if needed
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import feature mapping for human‑readable labels
try:
    from models.feature_mappings import FEATURE_TRANSLATION_MAP
except ImportError:
    FEATURE_TRANSLATION_MAP = {}

def get_display_name(feature_key):
    """Return human‑readable name from mapping, fallback to feature key."""
    return FEATURE_TRANSLATION_MAP.get(feature_key, feature_key)

def parse_time_to_hours(time_str):
    """Convert HH:MM string to fractional hours."""
    try:
        if pd.isna(time_str):
            return 12.0
        time_str = str(time_str).strip()
        parts = time_str.split(':')
        if len(parts) != 2:
            return 12.0
        return int(parts[0]) + (int(parts[1]) / 60.0)
    except Exception:
        return 12.0

def ensure_output_dir():
    """Create results/aggregate_analysis/ if it doesn't exist."""
    out_dir = os.path.join(PROJECT_ROOT, "results", "aggregate_analysis")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def load_model_and_metadata():
    """Load LightGBM booster and metadata."""
    model_path = os.path.join(PROJECT_ROOT, "models", "saved_states", "domain3_mood_sleep.txt")
    meta_path = os.path.join(PROJECT_ROOT, "models", "saved_states", "domain3_mood_sleep_metadata.json")
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Domain 3 model or metadata not found. Run train_domain3_mood_sleep.py first.")

    with open(meta_path, "r") as f:
        metadata = json.load(f)
    booster = lgb.Booster(model_file=model_path)
    return booster, metadata

def load_dataset():
    """Load the NHANES joined mood & sleep dataset."""
    data_path = os.path.join(PROJECT_ROOT, "datasets", "nhanes_joined_mood_sleep.csv")
    if not os.path.exists(data_path):
        # Fallback: synthetic data (for development)
        print("⚠️ Real dataset not found. Generating synthetic data for demonstration.")
        np.random.seed(42)
        n = 500
        data = {}
        for i in range(1, 10):
            data[f"DPQ0{i}0"] = np.random.randint(0, 4, n)
        data["DPQ100"] = np.random.randint(0, 4, n)
        data["SLQ300"] = ["23:00"] * n
        data["SLQ310"] = ["07:00"] * n
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(data_path)
        required = ["DPQ010", "DPQ020", "DPQ030", "DPQ040", "DPQ050", "DPQ060", "DPQ070", "DPQ080", "DPQ090", "SLQ300", "SLQ310"]
        for col in required:
            if col not in df.columns:
                df[col] = 0
    return df

def build_feature_matrix(df, metadata, booster):
    """
    Replicate feature engineering and align columns with the model's feature order.
    """
    model_features = booster.feature_name()
    df["bed_hours"] = df["SLQ300"].apply(parse_time_to_hours)
    df["wake_hours"] = df["SLQ310"].apply(parse_time_to_hours)
    df["calculated_sleep_duration"] = (df["wake_hours"] - df["bed_hours"]) % 24.0

    X = pd.DataFrame()
    for f in model_features:
        if f in df.columns:
            X[f] = df[f]
        else:
            X[f] = 0
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(X.median() if not X.empty else 0)
    return X

def plot_shap_summary(shap_values, X_sample, feature_names, out_dir):
    """Generate beeswarm summary plot."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False,
        max_display=len(feature_names)
    )
    plt.tight_layout()
    save_path = os.path.join(out_dir, "domain3_mood_shap_summary.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved SHAP summary plot: {save_path}")
    return save_path

def plot_mean_shap_bar(mean_abs_shap, feature_names, out_dir):
    """
    Generate mean absolute SHAP bar chart using numpy indexing.
    mean_abs_shap: 1D array of length n_features.
    feature_names: list of strings of same length.
    """
    mean_abs_shap = np.asarray(mean_abs_shap).flatten()
    # Ensure feature_names is a list of strings
    if not isinstance(feature_names, (list, np.ndarray)):
        feature_names = list(feature_names)
    else:
        feature_names = list(feature_names)
    
    if len(mean_abs_shap) != len(feature_names):
        print(f"⚠️ Length mismatch: mean_abs_shap={len(mean_abs_shap)}, feature_names={len(feature_names)}")
        min_len = min(len(mean_abs_shap), len(feature_names))
        mean_abs_shap = mean_abs_shap[:min_len]
        feature_names = feature_names[:min_len]
        print(f"   Truncated to {min_len} features.")

    # Sort descending
    sorted_idx = np.argsort(mean_abs_shap)[::-1]  # numpy array of ints
    sorted_features = np.array(feature_names)[sorted_idx].tolist()
    sorted_vals = mean_abs_shap[sorted_idx]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_features)), sorted_vals, color='teal')
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel("Mean |SHAP value|")
    plt.title("Domain 3 – Mean Absolute SHAP Value per Feature")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    save_path = os.path.join(out_dir, "domain3_mood_mean_shap_bar.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved mean SHAP bar chart: {save_path}")
    return save_path

def main():
    print("🚀 Generating aggregate importance plots for Domain 3 (Mood & Sleep)...")
    out_dir = ensure_output_dir()
    print(f"📁 Output directory: {out_dir}")

    # Load model and metadata
    booster, metadata = load_model_and_metadata()
    model_features = booster.feature_name()
    print(f"📋 Model has {len(model_features)} features.")

    # Load dataset and build feature matrix
    df = load_dataset()
    X = build_feature_matrix(df, metadata, booster)
    print(f"📊 Feature matrix shape: {X.shape}")

    # Sample if dataset is large (>1000 rows) to speed up
    sample_size = min(500, len(X))
    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
        print(f"📊 Using a random sample of {sample_size} rows (full dataset has {len(X)} rows).")
    else:
        X_sample = X
        print(f"📊 Using full dataset ({len(X)} rows).")

    # Compute SHAP values
    print("⏳ Computing SHAP values (this may take a moment)...")
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_sample)

    # Multi-class handling
    if isinstance(shap_values, list):
        shap_3d = np.array(shap_values)  # (n_classes, n_samples, n_features)
        shap_avg_across_classes = np.mean(shap_3d, axis=0)
        mean_abs_across_all = np.mean(np.abs(shap_3d), axis=(0, 1))
        shap_vals_for_plot = shap_avg_across_classes
        print("   (Multi-class model: using average SHAP across classes for summary, and overall mean absolute for bar chart.)")
    else:
        shap_vals_for_plot = shap_values
        mean_abs_across_all = np.mean(np.abs(shap_values), axis=0)

    # Feature names
    feature_names = [get_display_name(f) for f in model_features]

    # Generate plots
    print("⏳ Generating summary plot...")
    plot_shap_summary(shap_vals_for_plot, X_sample, feature_names, out_dir)

    print("⏳ Generating mean SHAP bar chart...")
    plot_mean_shap_bar(mean_abs_across_all, feature_names, out_dir)

    print("\n✅ All Domain 3 plots completed.")
    print(f"   Saved to: {out_dir}")

if __name__ == "__main__":
    main()