#!/usr/bin/env python3
"""
MINDSIGHT Domain 6 — Symptom Cluster Screener + Atypical Presentation Detector: Plotting

Loads the Model 1 (multinomial logistic regression cluster classifier) +
Model 2 (Isolation Forest) bundle produced by train_domain6_clinical.py and
produces:
  - Per-cluster coefficient plots (one per of the 8 classes)
  - A combined mean-|coefficient| importance panel across clusters
  - A confusion matrix heatmap (from evaluation_metrics.json)
  - A symptom-count-by-anomaly-flag comparison plot -- this directly
    visualizes the sanity check the training script already validated
    numerically (flagged rows should show a higher symptom count than
    non-flagged rows if the detector is tracking the intended construct)
  - A PCA scatter of the population, colored by predicted cluster, with
    Isolation-Forest-flagged rows marked distinctly

No dependency on plot_utils.py. Coefficients are read directly from
domain6_clinical_metadata.json; only the fitted Isolation Forest and
cluster classifier objects are unpickled (needed to score the real dataset
for the scatter/count plots -- metadata alone can't reproduce predictions).
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

BINARY_FEATURES = [
    "unwanted_thoughts", "repetitive_behaviors", "overthinking",
    "mind_going_blank", "avoidance_social_activity", "panic",
    "hypervigilance", "sleep_disturbances", "low_energy",
]

DISPLAY_NAMES = {
    "unwanted_thoughts": "Unwanted Thoughts",
    "repetitive_behaviors": "Repetitive Behaviors",
    "overthinking": "Overthinking",
    "mind_going_blank": "Mind Going Blank",
    "avoidance_social_activity": "Social Avoidance",
    "panic": "Panic",
    "hypervigilance": "Hypervigilance",
    "sleep_disturbances": "Sleep Disturbances",
    "low_energy": "Low Energy",
}


def display(feature_key):
    return DISPLAY_NAMES.get(feature_key, feature_key)


def project_root():
    """Walks upward from this script's location until it finds a directory
    containing sibling 'models' and 'datasets' folders -- i.e. the actual
    MINDSIGHT project root -- rather than assuming a fixed nesting depth.
    This avoids breaking if the script is moved to a different folder depth
    (e.g. scripts/visualizations/ vs. scripts/), which is exactly what broke
    the previous single-level-up assumption."""
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
    out_dir = os.path.join(project_root(), "results", "aggregate_analysis", "domain6_clinical")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# --------------------------------------------------------------------------
# Load metadata + pickled models
# --------------------------------------------------------------------------

def load_domain6_bundle():
    model_dir = os.path.join(project_root(), "models", "saved_states")
    meta_path = os.path.join(model_dir, "domain6_clinical_metadata.json")
    model_path = os.path.join(model_dir, "domain6_clinical.pkl")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Domain 6 metadata not found at {meta_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Domain 6 model payload not found at {model_path}")

    with open(meta_path, "r") as f:
        metadata = json.load(f)
    with open(model_path, "rb") as f:
        payload = pickle.load(f)

    cluster_classifier = payload["cluster_classifier"]
    isolation_forest = payload["isolation_forest"]
    return cluster_classifier, isolation_forest, metadata


def load_dataset(metadata):
    data_path = os.path.join(project_root(), "datasets", "ocd_symptoms_clean.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Domain 6 dataset not found at {data_path}")
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    missing = [c for c in BINARY_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required feature columns: {missing}")

    X = df[BINARY_FEATURES].copy()
    for col in BINARY_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0).astype(int)

    cluster_map = metadata["cluster_map"]
    df["cluster_label"] = df["Disease"].map(cluster_map).fillna("No_Detectable_Signal") \
        if "Disease" in df.columns else pd.Series(["Unknown"] * len(df))

    return X, df


# --------------------------------------------------------------------------
# Coefficient plots (per cluster + combined)
# --------------------------------------------------------------------------

def plot_coefficients(metadata, out_dir):
    coefficients = metadata["coefficients"]  # {class_name: {"coef": [...], "intercept": f}}
    feature_order = metadata["features"]
    display_labels = [display(f) for f in feature_order]
    class_order = metadata["class_order"]

    coef_matrix = []
    for class_name in class_order:
        coef = np.array(coefficients[class_name]["coef"])
        coef_matrix.append(coef)

        sorted_idx = np.argsort(np.abs(coef))[::-1]
        sorted_vals = coef[sorted_idx]
        colors = ["crimson" if v > 0 else "steelblue" for v in sorted_vals]

        plt.figure(figsize=(9, 5.5))
        plt.barh(range(len(feature_order)), sorted_vals, color=colors)
        plt.yticks(range(len(feature_order)), [display_labels[i] for i in sorted_idx])
        plt.xlabel("Coefficient value")
        plt.title(f"Domain 6 — Symptom Coefficients — {class_name}")
        plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
        plt.tight_layout()
        path = os.path.join(out_dir, f"domain6_{class_name}_coefficients.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  | Saved: {path}")

    coef_matrix = np.vstack(coef_matrix)
    mean_abs = np.mean(np.abs(coef_matrix), axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]

    plt.figure(figsize=(9, 5.5))
    plt.barh(range(len(feature_order)), mean_abs[sorted_idx], color="purple")
    plt.yticks(range(len(feature_order)), [display_labels[i] for i in sorted_idx])
    plt.xlabel("Mean |coefficient| across all 8 clusters")
    plt.title("Domain 6 — Combined Symptom Importance Across Clusters")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    path = os.path.join(out_dir, "domain6_combined_mean_abs_coefficients.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  | Saved: {path}")


# --------------------------------------------------------------------------
# Confusion matrix heatmap (from evaluation_metrics.json)
# --------------------------------------------------------------------------

def plot_confusion_matrix(metadata, out_dir):
    eval_path = os.path.join(project_root(), "models", "saved_states", "evaluation_metrics.json")
    if not os.path.exists(eval_path):
        print("  | [WARNING] evaluation_metrics.json not found; skipping confusion matrix plot.")
        return
    with open(eval_path, "r") as f:
        all_metrics = json.load(f)

    domain_metrics = all_metrics.get("domain_6_severe_clinical", {})
    cm_data = domain_metrics.get("cluster_classifier", {})
    cm = np.array(cm_data.get("confusion_matrix", []))
    class_labels = cm_data.get("class_labels", metadata["class_order"])

    if cm.size == 0:
        print("  | [WARNING] No confusion matrix found in evaluation_metrics.json; skipping.")
        return

    cm_normalized = cm / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(9, 8))
    plt.imshow(cm_normalized, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(label="Fraction of true class")
    plt.xticks(range(len(class_labels)), class_labels, rotation=45, ha="right")
    plt.yticks(range(len(class_labels)), class_labels)
    plt.xlabel("Predicted cluster")
    plt.ylabel("True cluster")
    plt.title(f"Domain 6 — Confusion Matrix (Test Set, n={cm_data.get('test_set_size', '?')})\n"
              f"Accuracy={cm_data.get('accuracy', '?')}, Macro-F1={cm_data.get('macro_f1', '?')}")
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            plt.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                     color="white" if cm_normalized[i, j] > 0.5 else "black", fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "domain6_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  | Saved: {path}")


# --------------------------------------------------------------------------
# Symptom-count-by-anomaly-flag comparison (visualizes the training-time
# sanity check: does the flag track unusual/multi-cluster symptom load?)
# --------------------------------------------------------------------------

def plot_symptom_count_by_flag(isolation_forest, X, out_dir):
    preds = isolation_forest.predict(X)
    flagged = (preds == -1)
    symptom_count = X.sum(axis=1)

    max_count = int(symptom_count.max())
    bins = range(0, max_count + 2)

    plt.figure(figsize=(9, 6))
    plt.hist(symptom_count[~flagged], bins=bins, alpha=0.6, label="Not flagged", color="steelblue", density=True)
    plt.hist(symptom_count[flagged], bins=bins, alpha=0.6, label="Flagged (atypical)", color="crimson", density=True)
    plt.xlabel("Number of symptoms present")
    plt.ylabel("Proportion within group")
    plt.title("Domain 6 — Symptom Count Distribution by Anomaly Flag\n"
              "(validates that flagged rows genuinely show higher symptom load)")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "domain6_symptom_count_by_anomaly_flag.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  | Saved: {path}")

    mean_flagged = symptom_count[flagged].mean() if flagged.any() else float("nan")
    mean_not_flagged = symptom_count[~flagged].mean() if (~flagged).any() else float("nan")
    print(f"      Mean symptom count -- flagged: {mean_flagged:.3f}, not flagged: {mean_not_flagged:.3f}")


# --------------------------------------------------------------------------
# PCA scatter: population colored by predicted cluster, flagged rows marked
# --------------------------------------------------------------------------

def plot_pca_scatter(cluster_classifier, isolation_forest, X, metadata, out_dir):
    preds_cluster = cluster_classifier.predict(X)
    preds_anomaly = isolation_forest.predict(X)  # -1 = flagged
    class_order = metadata["class_order"]

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    # Jitter -- binary feature space collapses many rows onto identical
    # coordinates without it.
    rng = np.random.default_rng(42)
    jitter = rng.normal(0, 0.06, X_pca.shape)
    X_pca_j = X_pca + jitter

    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(11, 9))
    for i, cluster_name in enumerate(class_order):
        idx = (preds_cluster == i) & (preds_anomaly == 1)
        plt.scatter(X_pca_j[idx, 0], X_pca_j[idx, 1], s=18, alpha=0.5,
                    color=cmap(i % 10), label=cluster_name)
    flagged_idx = (preds_anomaly == -1)
    plt.scatter(X_pca_j[flagged_idx, 0], X_pca_j[flagged_idx, 1], s=70, alpha=0.9,
                color="black", marker="*", label="Flagged (atypical)", edgecolors="white", linewidths=0.5)

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    plt.title("Domain 6 — Predicted Cluster + Atypical-Presentation Flags (PCA, jittered)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "domain6_cluster_anomaly_pca_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  | Saved: {path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    print("\n[Domain 6] Symptom Cluster Screener + Atypical Presentation Detector — Plots")
    out_dir = ensure_output_dir()
    try:
        cluster_classifier, isolation_forest, metadata = load_domain6_bundle()
        X, df = load_dataset(metadata)

        print("  | Generating per-cluster + combined coefficient plots...")
        plot_coefficients(metadata, out_dir)

        print("  | Generating confusion matrix plot...")
        plot_confusion_matrix(metadata, out_dir)

        print("  | Generating symptom-count-by-anomaly-flag validation plot...")
        plot_symptom_count_by_flag(isolation_forest, X, out_dir)

        print("  | Generating PCA scatter (cluster + anomaly flag)...")
        plot_pca_scatter(cluster_classifier, isolation_forest, X, metadata, out_dir)

        print("  | Domain 6 plots generated successfully.")
    except Exception as e:
        print(f"  | [FAILED] Domain 6 plotting failed: {e}")
        raise


if __name__ == "__main__":
    main()