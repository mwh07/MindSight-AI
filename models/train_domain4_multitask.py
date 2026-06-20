#!/usr/bin/env python3
"""
MINDSIGHT Domain 4 Calibration Engine (v2.8 - Standardized)
Trains independent Random Forest Regressors for Internet Addiction (IAT)
and Loneliness scales. Outputs unified schemas and clinical severity boundaries.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def train_digital_social_models():
    print("🤖 Starting Domain 4: Digital & Social Random Forest Regressor Pipeline...")
    
    data_path = "datasets/internet_phq_loneliness_clean.csv"
    output_dir = "models/saved_states"
    output_model_path = os.path.join(output_dir, "domain4_digital_social.pkl")
    output_meta_path = os.path.join(output_dir, "domain4_digital_social_metadata.json")
    os.makedirs(output_dir, exist_ok=True)
    
    iat_features = [f"IAT{i}" for i in range(1, 11)]
    loneliness_features = [f"loneliness{i}" for i in range(1, 7)]
    
    # Enforce strict layout order matching schema_config.json: age, gender, IAT, loneliness
    input_schema_features = ["age", "gender"] + iat_features + loneliness_features
    
    iat_target = "TotalIA"
    loneliness_target = "lonelinesstotal"
    
    df = None
    if os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path)
            print(f"📊 Loaded {df.shape[0]} rows for training from demographic cohort database.")
        except Exception as e:
            print(f"⚠️ [CRITICAL] Failed to read dataset: {str(e)}")
            df = None

    if df is None or len(df) < 30:
        print("⚠️ Notice: Dataset absent or insufficient. Generating standard calibration cohort...")
        np.random.seed(42)
        row_count = 1000
        synthetic_data = {}
        synthetic_data["age"] = np.random.randint(18, 65, row_count)
        synthetic_data["gender"] = np.random.choice([0, 1, 2], row_count, p=[0.48, 0.48, 0.04])
        
        for col in iat_features:
            synthetic_data[col] = np.random.randint(1, 6, row_count)  # IAT typical scale: 1-5
        for col in loneliness_features:
            synthetic_data[col] = np.random.randint(1, 5, row_count)  # UCLA Loneliness scale items
            
        df = pd.DataFrame(synthetic_data)
        df[iat_target] = df[iat_features].sum(axis=1) + np.random.randint(-3, 3, row_count)
        df[loneliness_target] = df[loneliness_features].sum(axis=1) * 2 + np.random.randint(-2, 2, row_count)

    # Clean and fill scale matrices
    for col in input_schema_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median() if not df.empty else 0)
            
    df[iat_target] = pd.to_numeric(df[iat_target], errors='coerce').fillna(df[iat_target].median() if not df.empty else 25)
    df[loneliness_target] = pd.to_numeric(df[loneliness_target], errors='coerce').fillna(df[loneliness_target].median() if not df.empty else 30)
    
    # Isolate feature tracking blocks for training
    X_iat = df[iat_features]
    y_iat = df[iat_target]
    
    X_lone = df[loneliness_features]
    y_lone = df[loneliness_target]
    
    # Train Model 1: Internet Addiction
    print("   Fitting Internet Addiction Scale Regressor...")
    rf_iat = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=15,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    rf_iat.fit(X_iat, y_iat)
    print(f"     ✅ Internet Addiction OOB R² Score: {rf_iat.oob_score_:.4f}")
    
    # Train Model 2: Loneliness
    print("   Fitting Loneliness Scale Regressor...")
    rf_lone = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=15,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    rf_lone.fit(X_lone, y_lone)
    print(f"     ✅ Loneliness OOB R² Score: {rf_lone.oob_score_:.4f}")
    
    iat_importances = {feat: float(imp) for feat, imp in zip(iat_features, rf_iat.feature_importances_)}
    lone_importances = {feat: float(imp) for feat, imp in zip(loneliness_features, rf_lone.feature_importances_)}
    
    # Package twin-model payloads clearly
    model_payload = {
        "rf_iat": rf_iat,
        "rf_lone": rf_lone
    }
    with open(output_model_path, "wb") as f:
        pickle.dump(model_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"🎉 Estimator binaries serialized to -> {output_model_path}")
    
    # Enrich metadata contract rules
    metadata = {
        "schema_version": "2.6",
        "domain": "domain_4_digital_and_social",
        "features": input_schema_features,  # Complete tracking definition matching master schema
        "iat_features": iat_features,
        "loneliness_features": loneliness_features,
        "global_importances": {
            **iat_importances,
            **lone_importances
        },
        "clinical_cutoffs": {
            "internet_addiction": {
                "Normal": [0, 30],
                "Mild": [31, 49],
                "Moderate": [50, 79],
                "Severe": [80, 100]
            },
            "loneliness": {
                "Low Loneliness": [0, 24],
                "Moderate Loneliness": [25, 44],
                "High Loneliness": [45, 80]
            }
        }
    }
    
    with open(output_meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"🎉 Balanced feature schema metadata exported to -> {output_meta_path}\n")

if __name__ == "__main__":
    train_digital_social_models()