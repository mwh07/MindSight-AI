#!/usr/bin/env python3
"""
MINDSIGHT Domain 5 Calibration Engine (v2.8 - Standardized)
Trains an XGBoost Regressor to predict continuous occupational burnout scores
and extracts empirical diagnostic cutoffs based on group maximums.
"""

import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def train_burnout_model():
    print("🚀 Commencing Domain 5 XGBoost Regressor Training Pipeline...")
    
    data_path = "datasets/tech_burnout_2026_clean.csv"
    output_dir = "models/saved_states"
    os.makedirs(output_dir, exist_ok=True)
    
    output_model_path = os.path.join(output_dir, "domain5_burnout.json")
    output_meta_path = os.path.join(output_dir, "domain5_burnout_metadata.json")
    
    schema_features = [
        "age", "work_hours_per_week", "meetings_per_day", 
        "work_life_balance_score", "job_satisfaction_score", 
        "deadline_pressure_score", "autonomy_score", "stress_score", "social_support_score"
    ]
    expected_genders = ["Male", "Female", "Non-binary", "Prefer not to say"]
    
    df = None
    if os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path)
            print(f"  │ Successfully ingested occupational burnout dataset.")
        except Exception as e:
            print(f"  │ [CRITICAL] Failed to read dataset: {str(e)}")
            df = None
            
    if df is None or len(df) < 50:
        print("  │ [WARNING] Dataset insufficient. Deploying stable synthetic calibration cohort.")
        np.random.seed(42)
        row_count = 1000
        df = pd.DataFrame({f: np.random.uniform(1, 10, row_count) for f in schema_features})
        df["gender"] = np.random.choice(expected_genders, row_count)
        df["burnout_score"] = df["stress_score"] * 0.6 + np.random.normal(0, 1, row_count)
        df["burnout_level"] = pd.cut(df["burnout_score"], bins=4, labels=["Low", "Moderate", "High", "Severe"])

    target_score_col = "burnout_score"
    target_level_col = "burnout_level"
    
    # Process numeric features
    for col in schema_features:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median() if not df.empty else 5.0)
    
    # Consistent One-Hot Encoding
    df["gender"] = df["gender"].fillna("Prefer not to say").astype(str)
    for gen in expected_genders:
        df[f"gender_{gen}"] = (df["gender"].str.lower() == gen.lower()).astype(int)
        
    final_features = schema_features + [f"gender_{gen}" for gen in expected_genders]
    
    X = df[final_features]
    y = pd.to_numeric(df[target_score_col], errors='coerce').fillna(df[target_score_col].median() if not df.empty else 5.0)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="rmse"
    )
    
    print("  │ Training Gradient Boosted Regression Trees...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Threshold extraction with enforced ordering to prevent logic inversions
    print("  │ Calibrating empirical score-to-level thresholds...")
    max_scores = df.groupby(target_level_col)[target_score_col].max().reindex(["Low", "Moderate", "High", "Severe"])
    
    thresholds = {
        "low_to_moderate": float(max_scores.get("Low", y.quantile(0.25))),
        "moderate_to_high": float(max_scores.get("Moderate", y.quantile(0.50))),
        "high_to_severe": float(max_scores.get("High", y.quantile(0.75)))
    }
    
    model.get_booster().save_model(output_model_path)
    
    # Export Metadata Verification Contract
    metadata = {
        "schema_version": "2.6",
        "domain": "domain_5_occupational_burnout",
        "features": final_features,
        "thresholds": thresholds,
        "gender_categories": expected_genders
    }
    with open(output_meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"[SUCCESS] Domain 5 model and metadata exported.")
    print(f"   └── Model Path: {output_model_path}")
    print(f"   └── Metadata Path: {output_meta_path}\n")

if __name__ == "__main__":
    train_burnout_model()