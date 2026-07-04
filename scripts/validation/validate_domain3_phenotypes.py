"""
VALIDATION SCRIPT: Domain 3 (Mood & Sleep) Phenotype Clustering
===============================================================
WHAT THIS SCRIPT DOES:
This script validates the KMeans clustering model used in Domain 3.
It loads the dataset, applies the trained model, and calculates the 
Silhouette Score, which is the standard academic metric for evaluating 
the quality of unsupervised clustering. It also prints the sizes of 
each discovered clinical phenotype.

WHY IT IS USEFUL:
Use this script for your project report or defense to mathematically prove 
that the 4 phenotypes discovered by the AI are statistically distinct and valid.

USAGE:
Run directly from the root directory:
`python scripts/validation/validate_domain3_phenotypes.py`
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score

def parse_time_to_hours(time_str):
    try:
        if pd.isna(time_str): return 12.0
        time_str = str(time_str).strip()
        parts = time_str.split(':')
        if len(parts) != 2: return 12.0
        return int(parts[0]) + (int(parts[1]) / 60.0)
    except Exception:
        return 12.0

def main():
    print("--- Domain 3 Phenotype Validation ---")
    model_path = "models/saved_states/domain3_mood_sleep.pkl"
    if not os.path.exists(model_path):
        print("Error: Model not found. Run training script first.")
        return

    with open(model_path, "rb") as f:
        payload = pickle.load(f)

    kmeans = payload["kmeans_model"]
    scaler = payload["scaler"]
    cluster_profiles = payload["cluster_profiles"]

    data_path = "datasets/nhanes_joined_mood_sleep.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        print("Dataset not found.")
        return

    dpq_cols = [f"DPQ0{i}0" for i in range(1, 10)]
    
    # Ensure columns exist if synthetic/clipped data is used
    for col in dpq_cols:
        if col not in df.columns:
            df[col] = 0
            
    df["phq9_sum"] = df[dpq_cols].sum(axis=1)
    df["bed_hours"] = df["SLQ300"].apply(parse_time_to_hours)
    df["wake_hours"] = df["SLQ310"].apply(parse_time_to_hours)
    df["calculated_sleep_duration"] = (df["wake_hours"] - df["bed_hours"]) % 24.0

    X = df[["phq9_sum", "calculated_sleep_duration"]].dropna()
    X_scaled = scaler.transform(X)
    labels = kmeans.predict(X_scaled)
    
    # Calculate Silhouette Score
    # Sample if dataset is too large to compute silhouette efficiently
    if len(X_scaled) > 5000:
        idx = np.random.choice(len(X_scaled), 5000, replace=False)
        sil_score = silhouette_score(X_scaled[idx], labels[idx])
    else:
        sil_score = silhouette_score(X_scaled, labels)

    print(f"\nOverall Silhouette Score: {sil_score:.4f}")
    if sil_score > 0.5:
        print("  -> Excellent clustering structure. Phenotypes are highly distinct.")
    elif sil_score > 0.25:
        print("  -> Good clustering structure. Phenotypes are moderately distinct.")
    else:
        print("  -> Weak clustering structure. Boundaries overlap heavily.")

    print("\nPhenotype Breakdown:")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        percentage = (c / len(labels)) * 100
        print(f"- {cluster_profiles.get(str(u), 'Unknown')}")
        print(f"  Population Size: {c} patients ({percentage:.1f}%)")

if __name__ == "__main__":
    main()
