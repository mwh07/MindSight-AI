#!/usr/bin/env python3
import sys
from plot_utils import (
    load_domain4_models, get_dataset, ensure_output_dir, generate_shap_summary_and_bar
)

def main():
    print("\n📊 Domain 4 — Digital & Social (RandomForest)")
    out_dir = ensure_output_dir()
    try:
        addiction_model, loneliness_model, _ = load_domain4_models()
        df4 = get_dataset("domain4")
        
        iat_features = [f"IAT{i}" for i in range(1, 11)]
        loneliness_features = [f"loneliness{i}" for i in range(1, 7)]
        
        X_iat = df4[iat_features].dropna()
        X_lone = df4[loneliness_features].dropna()

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

if __name__ == "__main__":
    main()
