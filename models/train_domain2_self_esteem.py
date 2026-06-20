#!/usr/bin/env python3
"""
MINDSIGHT Domain 2 Calibration Engine (v3.0 - Standardized)
Processes empirical Rosenberg Self-Esteem Scale (Q1-Q10) datasets,
stratifies responses into demographic cohorts, and exports empirical
percentile lookup tables for normative runtime evaluations.
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

def calibrate_self_esteem_norms():
    print("[RUNNING] Initializing Domain 2: Self-Esteem Normative Calibration...")
    
    dataset_path = "datasets/rosenberg_self_esteem_clean.csv"
    rses_cols = [f"Q{i}" for i in range(1, 11)]
    
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        print(f"  │ Successfully ingested clean self-esteem dataset: {dataset_path}")
    else:
        print(f"  │ [WARNING] Dataset {dataset_path} absent. Generating synthetic normative population matrix.")
        np.random.seed(42)
        row_count = 2000
        # Rosenberg responses scale from 1 to 4 matching assessment inputs
        synthetic_data = {col: np.random.randint(1, 5, row_count) for col in rses_cols}
        synthetic_data["age"] = np.random.randint(16, 70, row_count)
        synthetic_data["gender"] = np.random.choice([0, 1, 2], row_count, p=[0.48, 0.48, 0.04])
        df = pd.DataFrame(synthetic_data)

    # Clean and enforce valid entry ranges (1-4 Likert format)
    for col in rses_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(2).astype(int)
        df = df[df[col].isin([1, 2, 3, 4])]

    # Align demographic variables
    df["gender"] = pd.to_numeric(df.get("gender", 0), errors='coerce').fillna(0).astype(int)
    df["age"] = pd.to_numeric(df.get("age", 25), errors='coerce').fillna(25).astype(int)
    df["age_band"] = df["age"].apply(get_age_band)

    # ✅ FIXED: Correct clinical grouping based on FEATURE_TRANSLATION_MAP valence
    pos_items = ["Q1", "Q3", "Q4", "Q5", "Q7", "Q10"]
    neg_items = ["Q2", "Q6", "Q8", "Q9"]
    
    raw_scores = []
    for _, row in df.iterrows():
        score = 0
        for item in pos_items:
            score += int(row[item])
        for item in neg_items:
            # 5 - val maintains 1-4 scale mapping symmetry (Max possible = 40)
            score += (5 - int(row[item]))
        raw_scores.append(score)
        
    df["rses_total"] = raw_scores

    # Initialize complete percentile lookup table across all expected structural categories
    percentile_lookup = {}
    age_bands = ["under_18", "18_25", "26_35", "36_50", "51_65", "over_65"]
    gender_ids = [0, 1, 2] # 0: Male/Default, 1: Female, 2: Non-binary/Other
    
    # Pre-seed the complete matrix space to ensure zero runtime KeyErrors
    for g_id in gender_ids:
        for a_band in age_bands:
            cohort_key = f"g{g_id}_{a_band}"
            percentile_lookup[cohort_key] = {str(score): 50.0 for score in range(0, 45)}
    
    # Calculate cumulative empirical distribution functions from data
    grouped = df.groupby(["gender", "age_band"])
    print("  │ Calculating cumulative empirical distribution functions for cohorts...")
    
    for (gender_id, age_band), cohort_df in grouped:
        if gender_id not in gender_ids or age_band not in age_bands:
            continue
        cohort_key = f"g{gender_id}_{age_band}"
        scores_in_cohort = cohort_df["rses_total"].values
        
        cohort_map = {}
        for possible_score in range(0, 45):
            if len(scores_in_cohort) > 0:
                percentile = (np.sum(scores_in_cohort <= possible_score) / len(scores_in_cohort)) * 100.0
                cohort_map[str(possible_score)] = round(float(percentile), 2)
            else:
                cohort_map[str(possible_score)] = 50.0
                
        percentile_lookup[cohort_key] = cohort_map

    # Save outputs securely
    output_dir = "models/saved_states"
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
        "schema_version": "3.0",
        "domain": "domain_2_self_esteem",
        "features": ["age", "gender"] + rses_cols,
        "scoring_config": {
            "positive_items": pos_items,
            "negative_items": neg_items,
            "scale_bounds": [1, 4],
            "max_theoretical_score": 40
        },
        "demographic_mapping": {
            "gender_codes": {"0": "Male/Default", "1": "Female", "2": "Non-binary/Other"},
            "age_bands": age_bands
        }
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"[SUCCESS] Normative cohort percentile matrices successfully serialized.")
    print(f"    └── Pickle Destination: {pkl_path}")
    print(f"    └── Metadata Contract Destination: {meta_path}\n")

if __name__ == "__main__":
    calibrate_self_esteem_norms()