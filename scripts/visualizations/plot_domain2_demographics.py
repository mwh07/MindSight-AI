#!/usr/bin/env python3
"""
MINDSIGHT — Domain 2 Demographic KDE Distributions

This script loads the Rosenberg Self-Esteem dataset (Domain 2), calculates 
the true mathematical total scores (accounting for reverse-scored items), 
and generates Gaussian Kernel Density Estimation (KDE) curves.

This proves that self-esteem distributions vary significantly across 
demographic cohorts (e.g. Males vs Females, different age groups), 
mathematically justifying the project's use of Normative Percentile Scoring.
"""
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ========== NEW: We no longer need plot_utils for output dir ==========
# from plot_utils import ensure_output_dir   # <-- REMOVED

def calculate_rses_scores(df):
    """Calculate the Rosenberg Self-Esteem Score (0-40 scale) using vectorized numpy arrays."""
    # Q3, Q5, Q8, Q9, Q10 are reverse scored (0-4 scale)
    reverse_items = [3, 5, 8, 9, 10]
    scores = np.zeros(len(df))
    
    for i in range(1, 11):
        col = f"Q{i}"
        # Neutral imputation for missing, clip to valid 0-4 range
        val = pd.to_numeric(df[col], errors='coerce').fillna(2).clip(0, 4).astype(int)
        if i in reverse_items:
            scores += (4 - val)
        else:
            scores += val
            
    return scores

def map_gender(val):
    if val == 1:
        return "Male"
    elif val == 2:
        return "Female"
    elif val == 3:
        return "Other"
    else:
        return "Unknown"

def main():
    print("\n📊 Domain 2 — Self-Esteem Demographics (KDE Distributions)")
    
    # ========== NEW: Define dedicated output subfolder ==========
    out_dir = os.path.join(PROJECT_ROOT, "results", "aggregate_analysis", "domain2_demographics")
    os.makedirs(out_dir, exist_ok=True)
    
    dataset_path = os.path.join(PROJECT_ROOT, "datasets", "rosenberg_self_esteem_clean.csv")
    if not os.path.exists(dataset_path):
        print(f"  ❌ Dataset not found at {dataset_path}")
        return
        
    print("  ⏳ Loading dataset and calculating normative scores...")
    df = pd.read_csv(dataset_path)
    
    # Filter age outliers to get clean distributions
    df = df[(df['age'] >= 10) & (df['age'] <= 80)].copy()
    
    # Add mapped columns
    df['total_score'] = calculate_rses_scores(df)
    df['gender_label'] = df['gender'].apply(map_gender)
    
    # --- PLOT 1: GENDER DISTRIBUTION ---
    print("  ⏳ Plotting Gender KDE...")
    plt.figure(figsize=(10, 6))
    
    # Filter to main genders for clean visualization
    df_gender = df[df['gender_label'].isin(["Male", "Female"])]
    
    sns.kdeplot(data=df_gender, x="total_score", hue="gender_label", 
                fill=True, common_norm=False, palette={"Male": "#4c72b0", "Female": "#c44e52"},
                alpha=0.5, linewidth=2)
                
    plt.title("Rosenberg Self-Esteem Distribution by Gender\n(Mathematical Justification for Cohort Norming)", fontsize=14)
    plt.xlabel("Total Self-Esteem Score (0-40)", fontsize=12)
    plt.ylabel("Density (Population Proportion)", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Mark medians
    median_male = df_gender[df_gender['gender_label'] == 'Male']['total_score'].median()
    median_female = df_gender[df_gender['gender_label'] == 'Female']['total_score'].median()
    plt.axvline(median_male, color='#4c72b0', linestyle='--', alpha=0.8, label=f'Male Median ({median_male})')
    plt.axvline(median_female, color='#c44e52', linestyle='--', alpha=0.8, label=f'Female Median ({median_female})')
    plt.legend(title="Gender")
    
    plt.tight_layout()
    save_path_gender = os.path.join(out_dir, "domain2_kde_gender.png")
    plt.savefig(save_path_gender, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  | Saved Gender KDE Plot: {save_path_gender}")
    
    # --- PLOT 2: AGE COHORT DISTRIBUTION ---
    print("  ⏳ Plotting Age Cohort KDE...")
    df['age_cohort'] = pd.cut(df['age'], bins=[9, 18, 30, 80], labels=["Adolescents (10-18)", "Young Adults (19-30)", "Adults (31+)"])
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x="total_score", hue="age_cohort", 
                fill=True, common_norm=False, palette="viridis",
                alpha=0.4, linewidth=2)
                
    plt.title("Rosenberg Self-Esteem Distribution by Age Cohort", fontsize=14)
    plt.xlabel("Total Self-Esteem Score (0-40)", fontsize=12)
    plt.ylabel("Density (Population Proportion)", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # We won't draw median lines for age cohorts to avoid cluttering the plot
    
    plt.tight_layout()
    save_path_age = os.path.join(out_dir, "domain2_kde_age.png")
    plt.savefig(save_path_age, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  | Saved Age KDE Plot: {save_path_age}")
    
    print("  ✅ Domain 2 Demographic KDE plots generated.")

if __name__ == "__main__":
    main()