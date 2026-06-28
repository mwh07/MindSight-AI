#!/usr/bin/env python3
import sys
from plot_utils import (
    load_domain5_model, get_dataset, ensure_output_dir, generate_shap_summary_and_bar
)

def main():
    print("\n📊 Domain 5 — Occupational Burnout (XGBoost)")
    out_dir = ensure_output_dir()
    try:
        bst, meta5 = load_domain5_model()
        df5 = get_dataset("domain5")
        features5 = meta5["features"]
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

if __name__ == "__main__":
    main()
