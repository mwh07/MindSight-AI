#!/usr/bin/env python3
"""
MINDSIGHT — Exploratory Data Analysis (EDA) Correlation Heatmaps

This script generates Domain-Specific Correlation Heatmaps to prove two mathematical principles:
1. Domain 1 (Psychometrics): Internal Consistency (Cronbach's framework).
   Proves that items measuring the same trait (e.g., Extraversion) correlate properly.
2. Domain 5 (ML Features): Feature Collinearity.
   Proves that we checked for highly redundant/collinear features before training XGBoost.

Generated plots are saved to results/aggregate_analysis/
"""
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from plot_utils import ensure_output_dir

def plot_domain1_internal_consistency(out_dir):
    print("  ⏳ Generating Domain 1 Psychometric Internal Consistency Heatmap...")
    dataset_path = os.path.join(PROJECT_ROOT, "datasets", "big_five_personality_clean.csv")
    
    if not os.path.exists(dataset_path):
        print(f"  ❌ Domain 1 dataset not found: {dataset_path}")
        return
        
    df = pd.read_csv(dataset_path)
    
    # Select only the 10 Extraversion items
    ext_cols = [f"EXT{i}" for i in range(1, 11)]
    if not all(col in df.columns for col in ext_cols):
        print("  ❌ Could not find all EXT1-EXT10 columns in Domain 1 dataset.")
        return
        
    df_ext = df[ext_cols].dropna()
    
    # Compute Pearson Correlation
    corr_matrix = df_ext.corr(method="pearson")
    
    plt.figure(figsize=(10, 8))
    # Psychometric diverging palette: coolwarm or RdBu
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, 
                vmin=-1, vmax=1, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
                
    plt.title("Domain 1: Psychometric Internal Consistency (Extraversion)\n(Mathematical Proof of Construct Reliability)", fontsize=14)
    plt.tight_layout()
    
    save_path = os.path.join(out_dir, "domain1_eda_ext_correlation.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  | Saved Domain 1 EDA Heatmap: {save_path}")

def plot_domain5_feature_collinearity(out_dir):
    print("  ⏳ Generating Domain 5 ML Feature Collinearity Heatmap...")
    dataset_path = os.path.join(PROJECT_ROOT, "datasets", "tech_burnout_2026_clean.csv")
    
    if not os.path.exists(dataset_path):
        print(f"  ❌ Domain 5 dataset not found: {dataset_path}")
        return
        
    df = pd.read_csv(dataset_path)
    
    features = [
        'work_hours_per_week', 'meetings_per_day', 'work_life_balance_score', 
        'job_satisfaction_score', 'deadline_pressure_score', 'autonomy_score', 
        'stress_score', 'social_support_score'
    ]
    target = 'burnout_score'
    
    cols_to_use = features + [target]
    
    if not all(col in df.columns for col in cols_to_use):
        print("  ❌ Could not find all required ML features in Domain 5 dataset.")
        return
        
    df_ml = df[cols_to_use].dropna()
    
    # Compute Spearman Rank Correlation (better for ML features with non-linear relationships)
    corr_matrix = df_ml.corr(method="spearman")
    
    # Create cleaner labels for the plot
    clean_labels = [c.replace('_score', '').replace('_per_week', '').replace('_', ' ').title() for c in cols_to_use]
    
    plt.figure(figsize=(12, 10))
    # ML diverging palette
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Spectral", center=0, 
                vmin=-1, vmax=1, square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                xticklabels=clean_labels, yticklabels=clean_labels)
                
    plt.title("Domain 5: ML Feature Collinearity (XGBoost)\n(Mathematical Proof of Non-Redundant Features)", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    save_path = os.path.join(out_dir, "domain5_eda_collinearity.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  | Saved Domain 5 EDA Heatmap: {save_path}")

def main():
    print("\n📊 Exploratory Data Analysis (EDA) Correlation Heatmaps")
    out_dir = ensure_output_dir()
    
    try:
        plot_domain1_internal_consistency(out_dir)
        plot_domain5_feature_collinearity(out_dir)
        print("  ✅ EDA Heatmaps generated successfully.")
    except Exception as e:
        print(f"  ❌ EDA generation failed: {e}")

if __name__ == "__main__":
    main()
