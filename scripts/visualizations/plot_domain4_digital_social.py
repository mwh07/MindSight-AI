#!/usr/bin/env python3
import sys
from plot_utils import (
    load_domain4_model, get_dataset, ensure_output_dir, generate_shap_summary_and_bar
)

def main():
    print("\nDomain 4 - Digital & Social Cross-Impact (RandomForest)")
    out_dir = ensure_output_dir()
    try:
        model, metadata = load_domain4_model()
        df4 = get_dataset("domain4")
        
        features = metadata.get("features", [])
        if not features:
            iat_features = [f"IAT{i}" for i in range(1, 11)]
            loneliness_features = [f"loneliness{i}" for i in range(1, 7)]
            features = ["age", "gender"] + iat_features + loneliness_features

        # Ensure we have the same features the model was trained on
        X = df4[features].dropna().copy()
        if 'gender' in X.columns and X['gender'].dtype == object:
            X['gender'] = X['gender'].map({'Male': 1, 'Female': 0, 'Other': 2}).fillna(0)

        print("  Computing SHAP for Cross-Impact (Depression Risk) model...")
        generate_shap_summary_and_bar(
            model,
            X,
            features,
            "domain4_cross_impact",
            out_dir,
            sample_size=500
        )

        print("  [SUCCESS] Domain 4 cross-impact plots generated.")
    except Exception as e:
        print(f"  [ERROR] Domain 4 failed: {e}")

if __name__ == "__main__":
    main()
