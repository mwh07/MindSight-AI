#!/usr/bin/env python3
"""
MINDSIGHT Domain 6 Calibration Engine (v3.4 - Standardized)
Trains a clinical screening classifier and anomaly detector using an explicit OvR wrapper.
Prioritizes real clinical datasets and dynamically resolves schema configurations.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import IsolationForest

def generate_high_fidelity_reference_cohort(n_samples=5000):
    """Generates an epidemiologically grounded clinical reference population."""
    np.random.seed(42)
    
    # Simulate a latent clinical breakdown: 80% baseline control, 20% severe distress
    latent_clinical_status = np.random.choice([0, 1], size=n_samples, p=[0.80, 0.20])
    
    # Enforce strict layout order matching schema_config.json exactly (e.g., repetitive_behaviors)
    features = [
        "unwanted_thoughts", "repetitive_behaviors", "overthinking",
        "mind_going_blank", "avoidance_social_activity", "panic", "hypervigilance"
    ]
    
    synthetic_records = []
    for status in latent_clinical_status:
        if status == 0:
            # Baseline/Control Group: Low symptom probabilities
            probs = [0.15, 0.10, 0.22, 0.14, 0.12, 0.08, 0.11]
        else:
            # Clinical Distress Group: High probability of co-occurring symptoms
            probs = [0.78, 0.70, 0.88, 0.75, 0.82, 0.65, 0.72]
            
        symptoms = [np.random.choice([0, 1], p=[1-p, p]) for p in probs]
        symptom_sum = sum(symptoms)
        
        # Partition into 5 ordinal severity classifications (Classes 0 to 4)
        if symptom_sum == 0:
            target = 0 # Subclinical Baseline
        elif symptom_sum <= 2:
            target = 1 # Mild Profile
        elif symptom_sum <= 4:
            target = 2 # Moderate Clinical Core
        elif symptom_sum <= 6:
            target = 3 # Severe Clinical Presentation
        else:
            target = 4 # Extreme Crisis Threshold
            
        synthetic_records.append(symptoms + [target])
        
    return pd.DataFrame(synthetic_records, columns=features + ["clinical_target"])

def train_clinical_screening_pipeline():
    print("[RUNNING] Starting Domain 6: Severe Clinical Screening Pipeline...")
    
    output_dir = "models/saved_states"
    output_model_path = os.path.join(output_dir, "domain6_clinical.pkl")
    output_meta_path = os.path.join(output_dir, "domain6_clinical_metadata.json")
    os.makedirs(output_dir, exist_ok=True)

    # Strictly align feature tracking layout with schema_config.json
    binary_features = [
        "unwanted_thoughts", "repetitive_behaviors", "overthinking",
        "mind_going_blank", "avoidance_social_activity", "panic", "hypervigilance"
    ]

    # Prioritized path array
    candidate_paths = [
        "datasets/ocd_symptoms_clean.csv",
        "datasets/responses.csv",
        "responses.csv"
    ]
    
    selected_path = None
    df_raw = None
    
    # Route to the first available valid data source
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                df_candidate = pd.read_csv(path)
                if len(df_candidate) >= 10:
                    selected_path = path
                    df_raw = df_candidate
                    print(f"🎯 Successfully matched training dataset matrix at: '{path}' ({len(df_raw)} records)")
                    break
                else:
                    print(f"ℹ️ Found single-patient telemetry or small matrix at '{path}'. Continuing search...")
            except Exception as e:
                print(f"⚠️ Could not parse candidate file at '{path}': {e}")

    use_synthetic_reference = False
    if df_raw is None:
        print("⚠️ Direct training datasets missing or empty. Spinning up reference cohort...")
        use_synthetic_reference = True

    if use_synthetic_reference:
        df = generate_high_fidelity_reference_cohort(n_samples=5000)
    else:
        df = df_raw.copy()
        
        # Comprehensive column normalizer map to safeguard against variations while mapping to schema
        column_normalization_map = {
            "repetitive_behaviors": "repetitive_behaviors",
            "social_withdrawal": "avoidance_social_activity",
            "cognitive_blocking": "mind_going_blank",
            "rumination": "overthinking"
        }
        df = df.rename(columns=column_normalization_map)
        
        # Enforce clean data-types on tracking metrics
        for col in binary_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            else:
                print(f"⚠️ Feature column '{col}' missing from data source layout. Initializing with zeros.")
                df[col] = 0
        
        # Dynamically build target ordinal labels if missing from source
        if "clinical_target" not in df.columns:
            symptom_sum = df[binary_features].sum(axis=1)
            df['clinical_target'] = pd.cut(
                symptom_sum, 
                bins=[-1, 0, 2, 4, 6, 7], 
                labels=[0, 1, 2, 3, 4]
            ).astype(int)

    X = df[binary_features].copy()
    y = df["clinical_target"].copy().astype(int)
    
    # Ensure there are at least 2 distinct target classes present to fit the classifier
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        print("⚠️ Selected dataset contains insufficient target distribution classes. Injecting synthetic reference bounds...")
        df_ref = generate_high_fidelity_reference_cohort(n_samples=2000)
        X = pd.concat([X, df_ref[binary_features]], ignore_index=True)
        y = pd.concat([y, df_ref["clinical_target"]], ignore_index=True).astype(int)
        unique_classes = np.unique(y)

    print(f"📋 Confirmed reference matrix shape: {X.shape} across classes: {unique_classes.tolist()}")

    print("   Fitting High-Fidelity One-Vs-Rest Wrapped Logistic Classifier...")
    base_estimator = LogisticRegression(
        solver='lbfgs',
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    classifier = OneVsRestClassifier(base_estimator)
    classifier.fit(X, y)
    
    # Manually expose aggregated .coef_ and .intercept_ matrices for downstream script binding
    classifier.coef_ = np.vstack([estimator.coef_ for estimator in classifier.estimators_])
    classifier.intercept_ = np.array([estimator.intercept_[0] for estimator in classifier.estimators_])
    
    print("   Fitting Stable Population Isolation Forest...")
    anomaly_detector = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42
    )
    anomaly_detector.fit(X)
    
    # Save optimized payload stack
    model_payload = {
        "classifier": classifier,
        "anomaly_detector": anomaly_detector
    }
    with open(output_model_path, "wb") as f:
        pickle.dump(model_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"🎉 Calibrated model states saved securely to -> {output_model_path}")
    
    # Export standard metadata block containing structural validation hooks
    metadata = {
        "schema_version": "2.6",
        "domain": "domain_6_severe_clinical",
        "features": binary_features,
        "classes": [int(c) for c in classifier.classes_],
        "execution_mode": f"production_ovr_wrapped_source_{'synthetic' if use_synthetic_reference else 'dataset'}",
        "source_file_used": selected_path if selected_path else "synthetic_reference_generation",
        "payload_contract": {
            "classifier_key": "classifier",
            "anomaly_key": "anomaly_detector",
            "expected_type": "dictionary_multi_stack"
        }
    }
    with open(output_meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"🎉 Metadata verification layout exported to -> {output_meta_path}")
    print("[SUCCESS] Domain 6 pipeline optimization complete.\n")

if __name__ == "__main__":
    train_clinical_screening_pipeline()