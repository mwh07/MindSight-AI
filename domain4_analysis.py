import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

data_path = "datasets/internet_phq_loneliness_clean.csv"
if not os.path.exists(data_path):
    print("Dataset not found.")
else:
    df = pd.read_csv(data_path)
    print("Dataset shape:", df.shape)
    
    iat_features = [f"IAT{i}" for i in range(1, 11)]
    loneliness_features = [f"loneliness{i}" for i in range(1, 7)]
    features = ["age", "gender"] + iat_features + loneliness_features
    
    # Clean data
    X = df[features].copy()
    if 'gender' in X.columns and X['gender'].dtype == object:
        X['gender'] = X['gender'].map({'Male': 1, 'Female': 0, 'Other': 2}).fillna(0)
    X = X.fillna(0)
    y = df["totalphq"].fillna(0)
    
    print("\n--- Cross-Impact Model Analysis (Target: Depression) ---")
    
    # 1. Linear Regression Baseline
    lr_model = LinearRegression()
    lr_model.fit(X, y)
    print(f"\nLinear Regression R-squared: {lr_model.score(X, y):.4f}")
    
    print("\nTop Linear Coefficients (Impact on Depression):")
    coeffs = pd.Series(lr_model.coef_, index=features).sort_values(ascending=False)
    print(coeffs.head(10))
    
    # 2. Random Forest Baseline
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    print(f"\nRandom Forest R-squared: {rf_model.score(X, y):.4f}")
    
    print("\nTop Random Forest Importances:")
    importances = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
    print(importances.head(10))
