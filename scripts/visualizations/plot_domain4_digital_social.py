#!/usr/bin/env python3
import sys
from pathlib import Path
from plot_utils import (
    load_domain4_model, get_dataset, generate_shap_summary_and_bar
)

def main():
    print("\nDomain 4 - Digital & Social Cross-Impact (RandomForest)")

    # Define the desired output folder
    # Assumes this script lives in: ROOT/scripts/visualizations/
    # So ROOT is two levels up from this file.
    root_dir = Path(__file__).resolve().parent.parent.parent
    out_dir = root_dir / "results" / "aggregate_analysis" / "domain4_digital_social"
    out_dir.mkdir(parents=True, exist_ok=True)

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
            out_dir,          # <-- Pass the new dedicated folder
            sample_size=500
        )

        print(f"  [SUCCESS] Domain 4 cross-impact plots generated in:\n    {out_dir}")
    except Exception as e:
        print(f"  [ERROR] Domain 4 failed: {e}")

if __name__ == "__main__":
    main()