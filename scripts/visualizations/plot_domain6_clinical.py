#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import (
    load_domain6_model, get_dataset, ensure_output_dir, get_display_name
)

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

def main():
    print("\n📊 Domain 6 — Severe Clinical Screening (LogisticRegression)")
    out_dir = ensure_output_dir()
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

if __name__ == "__main__":
    main()
