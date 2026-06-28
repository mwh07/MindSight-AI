#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import (
    load_domain6_model, get_dataset, ensure_output_dir, get_display_name
)
from sklearn.decomposition import PCA

def generate_coefficient_plots(classifier, X, feature_names, out_dir):
    if hasattr(classifier, "estimators_"):
        coefs = np.vstack([est.coef_[0] for est in classifier.estimators_])
        classes = classifier.classes_
    else:
        coefs = classifier.coef_
        classes = np.arange(coefs.shape[0])

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
        print(f"  | Saved class {class_label} coefficient plot: {save_path}")

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
    print(f"  | Saved mean absolute coefficient plot: {save_path}")
    return save_path

def generate_anomaly_scatter_plot(anomaly_detector, X, out_dir):
    preds = anomaly_detector.predict(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Apply JITTER FIX to prevent 8,000 dots from stacking onto 37 coordinates
    # Adding tiny random noise to spread the identical points out slightly
    np.random.seed(42)
    jitter = np.random.normal(0, 0.08, X_pca.shape)
    X_pca_jittered = X_pca + jitter

    plt.figure(figsize=(10, 8))
    normal_idx = (preds == 1)
    anomaly_idx = (preds == -1)
    
    plt.scatter(X_pca_jittered[normal_idx, 0], X_pca_jittered[normal_idx, 1], 
                c='blue', alpha=0.3, label='Normal (Inliers)', s=20, edgecolors='none')
    plt.scatter(X_pca_jittered[anomaly_idx, 0], X_pca_jittered[anomaly_idx, 1], 
                c='red', alpha=0.8, marker='*', label='Atypical (Anomalies)', s=80, edgecolors='black')
                
    plt.title("Isolation Forest Anomaly Detection (PCA with Jitter)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(out_dir, "domain6_anomaly_scatter.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  | Saved anomaly scatter plot: {save_path}")

def main():
    print("\n📊 Domain 6 — Severe Clinical Screening (LogisticRegression)")
    out_dir = ensure_output_dir()
    try:
        classifier, anomaly_detector, meta6 = load_domain6_model()
        df6 = get_dataset("domain6")
        features6 = meta6["features"]
        X6 = df6[features6].dropna()
        
        print("  ⏳ Computing coefficient plots...")
        generate_coefficient_plots(classifier, X6, features6, out_dir)
        
        if anomaly_detector:
            print("  ⏳ Computing anomaly scatter plot (with Jitter Fix)...")
            generate_anomaly_scatter_plot(anomaly_detector, X6, out_dir)
            
        print("  ✅ Domain 6 plots generated.")
    except Exception as e:
        print(f"  ❌ Domain 6 failed: {e}")

if __name__ == "__main__":
    main()
