#!/usr/bin/env python3
"""
MINDSIGHT Domain 2 Calibration Engine (v3.1 - Corrected)
Processes empirical Rosenberg Self-Esteem Scale (Q1-Q10) datasets,
stratifies responses into demographic cohorts, and exports empirical
percentile lookup tables for normative runtime evaluations.
Aligns with frontend (reverse items [3,5,8,9,10]) and 0-4 scale.
ADDED: Evaluation metrics note (deterministic, no ML).
"""

import os
import json
import pickle
import numpy as np
import pandas as pd

def get_age_band(age):
    """Categorizes numerical age into stratified cohort bands matching runtime."""
    if age < 18:
        return "under_18"
    elif age <= 25:
        return "18_25"
    elif age <= 35:
        return "26_35"
    elif age <= 50:
        return "36_50"
    elif age <= 65:
        return "51_65"
    else:
        return "over_65"

def compute_rse_score(row, reverse_items):
    """Compute RSE total score (0-40) using correct reverse items on 0-4 scale."""
    score = 0
    for i in range(1, 11):
        col = f"Q{i}"
        val = row[col]  # already 0-4
        if i in reverse_items:
            score += (4 - val)   # reverse
        else:
            score += val
    return score

def calibrate_self_esteem_norms():
    print("[RUNNING] Initializing Domain 2: Self-Esteem Normative Calibration...")
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_path = os.path.join(project_root, "datasets", "rosenberg_self_esteem_clean.csv")
    rses_cols = [f"Q{i}" for i in range(1, 11)]
    reverse_items = [3, 5, 8, 9, 10]  # consistent with frontend and inference
    
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        print(f"  | Successfully ingested clean self-esteem dataset: {dataset_path}")
        # Filter out extreme age outliers
        df = df[(df['age'] >= 10) & (df['age'] <= 100)]
        print(f"  | Filtered age to 10-100, remaining rows: {len(df)}")
    else:
        print(f"  | [WARNING] Dataset {dataset_path} absent. Generating synthetic normative population matrix.")
        np.random.seed(42)
        row_count = 2000
        # Generate synthetic 0-4 responses (matching dataset scale)
        synthetic_data = {col: np.random.randint(0, 5, row_count) for col in rses_cols}
        synthetic_data["age"] = np.random.randint(18, 70, row_count)
        synthetic_data["gender"] = np.random.choice([0, 1, 2], row_count, p=[0.48, 0.48, 0.04])
        df = pd.DataFrame(synthetic_data)

    # Ensure numeric and clip to 0-4 (handle any out-of-range)
    for col in rses_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(2).clip(0, 4).astype(int)

    # Align demographic variables
    df["gender"] = pd.to_numeric(df.get("gender", 0), errors='coerce').fillna(0).astype(int)
    # Map gender 3 to 2 (non-binary/other) if present
    df["gender"] = df["gender"].replace(3, 2)
    df["age"] = pd.to_numeric(df.get("age", 25), errors='coerce').fillna(25).astype(int)
    df["age_band"] = df["age"].apply(get_age_band)

    # Compute RSE total using consistent logic
    df["rses_total"] = df.apply(lambda row: compute_rse_score(row, reverse_items), axis=1)

    # Initialize complete percentile lookup table (scores 0-40)
    percentile_lookup = {}
    age_bands = ["under_18", "18_25", "26_35", "36_50", "51_65", "over_65"]
    gender_ids = [0, 1, 2]  # 0: Male, 1: Female, 2: Non-binary/Other
    
    # Pre-seed with 50% for all possible scores (0-40)
    for g_id in gender_ids:
        for a_band in age_bands:
            cohort_key = f"g{g_id}_{a_band}"
            percentile_lookup[cohort_key] = {str(score): 50.0 for score in range(0, 41)}
    
    # Calculate cumulative empirical distribution functions from data
    grouped = df.groupby(["gender", "age_band"])
    print("  | Calculating cumulative empirical distribution functions for cohorts...")
    
    for (gender_id, age_band), cohort_df in grouped:
        if gender_id not in gender_ids or age_band not in age_bands:
            continue
        cohort_key = f"g{gender_id}_{age_band}"
        scores_in_cohort = cohort_df["rses_total"].values
        
        cohort_map = {}
        for possible_score in range(0, 41):
            if len(scores_in_cohort) > 0:
                percentile = (np.sum(scores_in_cohort <= possible_score) / len(scores_in_cohort)) * 100.0
                cohort_map[str(possible_score)] = round(float(percentile), 2)
            else:
                cohort_map[str(possible_score)] = 50.0
                
        percentile_lookup[cohort_key] = cohort_map

    # Save outputs securely
    output_dir = os.path.join(project_root, "models", "saved_states")
    os.makedirs(output_dir, exist_ok=True)
    
    pkl_path = os.path.join(output_dir, "domain2_self_esteem.pkl")
    json_path = os.path.join(output_dir, "domain2_self_esteem_percentiles.json")
    meta_path = os.path.join(output_dir, "domain2_self_esteem_metadata.json")
    
    with open(pkl_path, "wb") as f:
        pickle.dump(percentile_lookup, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open(json_path, "w") as f:
        json.dump(percentile_lookup, f, indent=4)
        
    # Export structural verification layout contract
    metadata = {
        "schema_version": "3.1",
        "domain": "domain_2_self_esteem",
        "features": ["age", "gender"] + rses_cols,
        "scoring_config": {
            "reverse_items": reverse_items,
            "scale_bounds": [0, 4],
            "max_theoretical_score": 40
        },
        "demographic_mapping": {
            "gender_codes": {"0": "Male", "1": "Female", "2": "Non-binary/Other"},
            "age_bands": age_bands
        },
        "preprocessing": {
            "age_filter_min": 10,
            "age_filter_max": 100
        }
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"[SUCCESS] Normative cohort percentile matrices successfully serialized.")
    print(f"    +-- Pickle Destination: {pkl_path}")
    print(f"    +-- Metadata Contract Destination: {meta_path}\n")

    # --- Save evaluation metrics (deterministic) ---
    eval_metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    domain_metrics = {
        "metric_type": "N/A - deterministic scoring, no learned parameters",
        "note": "RSE score is a fixed clinical formula; percentile table is empirical lookup, not a fitted model."
    }
    if os.path.exists(eval_metrics_path):
        with open(eval_metrics_path, "r") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}
    all_metrics["domain_2_self_esteem"] = domain_metrics
    with open(eval_metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[SUCCESS] Evaluation metrics saved to -> {eval_metrics_path}")

if __name__ == "__main__":
    calibrate_self_esteem_norms()