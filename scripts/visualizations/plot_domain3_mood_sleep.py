#!/usr/bin/env python3
"""
MINDSIGHT — Clinical Phenotype Cluster Plot for Domain 3 (Mood & Sleep)

This script loads the unsupervised KMeans model for Domain 3 and the
nhanes_joined_mood_sleep.csv dataset. It visualizes the AI-discovered 
clinical phenotypes by plotting PHQ-9 Severity vs Sleep Duration.

Saves plots to: results/aggregate_analysis/domain3_mood_sleep/
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path if needed
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
    print("Generating Phenotype Cluster Plot for Domain 3 (Mood & Sleep)...")
    
    # ========== NEW: Define dedicated output subfolder ==========
    out_dir = os.path.join(PROJECT_ROOT, "results", "aggregate_analysis", "domain3_mood_sleep")
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(PROJECT_ROOT, "models", "saved_states", "domain3_mood_sleep.pkl")
    if not os.path.exists(model_path):
        print("Error: domain3_mood_sleep.pkl not found. Run train_domain3_mood_sleep.py first.")
        return

    with open(model_path, "rb") as f:
        payload = pickle.load(f)

    kmeans = payload["kmeans_model"]
    scaler = payload["scaler"]
    cluster_profiles = payload["cluster_profiles"]

    data_path = os.path.join(PROJECT_ROOT, "datasets", "nhanes_joined_mood_sleep.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        print("Warning: Real dataset not found. Generating synthetic data for visualization.")
        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            "DPQ010": np.random.randint(0, 4, n),
            "SLQ300": ["23:00"] * n,
            "SLQ310": ["07:00"] * n
        })

    # STRICTLY sum only the 9 core PHQ-9 items to ensure max score is 27.
    # (Exclude DPQ100 which is functional impairment, not symptom severity).
    dpq_cols = [f"DPQ0{i}0" for i in range(1, 10)]
    
    # If using synthetic data with only DPQ010, ensure the columns exist before summing
    for col in dpq_cols:
        if col not in df.columns:
            df[col] = 0
            
    df["phq9_sum"] = df[dpq_cols].sum(axis=1)

    if "SLQ300" in df.columns and "SLQ310" in df.columns:
        df["bed_hours"] = df["SLQ300"].apply(parse_time_to_hours)
        df["wake_hours"] = df["SLQ310"].apply(parse_time_to_hours)
        df["calculated_sleep_duration"] = (df["wake_hours"] - df["bed_hours"]) % 24.0
    else:
        df["calculated_sleep_duration"] = 8.0

    X = df[["phq9_sum", "calculated_sleep_duration"]].dropna()
    
    # Predict clusters
    X_scaled = scaler.transform(X)
    X["Cluster"] = kmeans.predict(X_scaled)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i in range(kmeans.n_clusters):
        cluster_data = X[X["Cluster"] == i]
        label = cluster_profiles.get(str(i), f"Cluster {i}")
        
        # Add jitter to reveal data density (prevent thousands of points overlapping on integers)
        jitter_x = cluster_data["phq9_sum"] + np.random.uniform(-0.3, 0.3, size=len(cluster_data))
        jitter_y = cluster_data["calculated_sleep_duration"] + np.random.uniform(-0.3, 0.3, size=len(cluster_data))
        
        plt.scatter(
            jitter_x, 
            jitter_y, 
            alpha=0.4, 
            s=20,  # Slightly smaller dots
            c=colors[i % len(colors)],
            label=label,
            edgecolors='none'
        )
        
        # Plot centroid
        centroid = scaler.inverse_transform([kmeans.cluster_centers_[i]])[0]
        plt.scatter(
            centroid[0], centroid[1], 
            s=250, c=colors[i % len(colors)], 
            marker='X', edgecolors='black', linewidth=1.5
        )

    plt.title("Domain 3: AI-Discovered User Profiles (Mood & Sleep)", fontsize=14, pad=15)
    plt.xlabel("Depression Severity (PHQ-9 Sum)", fontsize=12)
    plt.ylabel("Calculated Sleep Duration (Hours)", fontsize=12)
    plt.axhline(y=7, color='gray', linestyle='--', alpha=0.5, label="Optimal Sleep Lower Bound (7h)")
    plt.axhline(y=9, color='gray', linestyle='--', alpha=0.5, label="Optimal Sleep Upper Bound (9h)")
    plt.axvline(x=10, color='red', linestyle='--', alpha=0.3, label="Moderate Depression Cutoff (10)")
    
    plt.legend(title="Phenotypes (Centroids marked with X)", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Add a caption mentioning the clustering algorithm
    caption = "Figure: Unsupervised K-Means clustering naturally segments users into distinct behavioral profiles based on clinical severity and sleep rhythm."
    plt.figtext(0.5, -0.05, caption, wrap=True, horizontalalignment='center', fontsize=10, style='italic', color='gray')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # ========== Save to the dedicated folder ==========
    save_path = os.path.join(out_dir, "domain3_phenotype_clusters.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved Phenotype Cluster Plot: {save_path}")

if __name__ == "__main__":
    main()