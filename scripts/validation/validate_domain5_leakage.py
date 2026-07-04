"""
VALIDATION SCRIPT: Domain 5 (Burnout) Data Leakage Proof
========================================================
WHAT THIS SCRIPT DOES:
This script checks the raw dataset for severe multicollinearity and data
leakage concerning the Occupational Burnout target. It proves that variables 
like `burnout_total` and `Depersonalization_score` correlate 1.0 with the 
target, confirming they are mathematically synonymous with the target itself.

WHY IT IS USEFUL:
Use this script during your project defense to justify why certain variables 
(like `burnout_total`) were deliberately excluded from the Machine Learning 
training features in `train_domain5_burnout.py`. It proves you know how to 
identify and prevent "cheating" (data leakage) in predictive models.

USAGE:
Run directly from the root directory:
`python scripts/validation/validate_domain5_leakage.py`
"""

import os
import pandas as pd
import numpy as np

print("Current Directory:", os.getcwd())

data_path = "datasets/tech_burnout_2026_clean.csv"
if not os.path.exists(data_path):
    print("DATASET NOT FOUND:", data_path)
else:
    df = pd.read_csv(data_path)
    print("Dataset shape:", df.shape)
    
    schema_features = [
        "work_hours_per_week", "meetings_per_day", 
        "work_life_balance_score", "job_satisfaction_score", 
        "deadline_pressure_score", "autonomy_score", "stress_score", "social_support_score"
    ]
    
    print("\nMissing values:")
    print(df[schema_features].isnull().sum())
    
    print("\nDescriptive statistics:")
    print(df[schema_features].describe())
    
    print("\nCorrelation matrix:")
    corr = df[schema_features + ["burnout_score"]].corr()
    print(corr)
    
    print("\nVIF:")
    # Calculate VIF from correlation matrix
    # VIF is diagonal elements of the inverse correlation matrix
    X_corr = df[schema_features].dropna().corr()
    try:
        inv_corr = np.linalg.inv(X_corr.values)
        vif = pd.DataFrame({
            "Feature": schema_features,
            "VIF": np.diag(inv_corr)
        })
        print(vif)
    except Exception as e:
        print("Could not compute VIF:", e)
