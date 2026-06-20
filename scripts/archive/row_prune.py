import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# === COMMAND-LINE ARGUMENTS ===
def parse_args():
    parser = argparse.ArgumentParser(
        description="Prune datasets by age: 80% age 15-30, 15% age 31-50, 5% others (if age column exists)."
    )
    parser.add_argument("folder", help="Path to the folder containing CSV files")
    return parser.parse_args()

# === AGE-BASED STRATIFIED SAMPLING ===
def sample_by_age(df, age_col):
    """
    Returns a DataFrame with:
      - 80% of rows where age between 15-30
      - 15% of rows where age between 31-50
      - 5% of rows where age <15 or >50
    """
    # Ensure age is numeric, drop NaN age rows
    df[age_col] = pd.to_numeric(df[age_col], errors='coerce')
    df = df.dropna(subset=[age_col])
    
    # Define bins
    core_mask = (df[age_col] >= 12) & (df[age_col] <= 30)
    adult_mask = (df[age_col] >= 31) & (df[age_col] <= 50)
    other_mask = ~(core_mask | adult_mask)
    
    # Count rows per bin
    core_df = df[core_mask]
    adult_df = df[adult_mask]
    other_df = df[other_mask]
    
    # Target sizes (80/15/5 of total after dropping NAs)
    total = len(df)
    n_core = int(0.80 * total)
    n_adult = int(0.15 * total)
    n_other = total - n_core - n_adult  # 5% (or any remainder)
    
    # Sample with replacement if needed (should be fine for large datasets)
    if len(core_df) == 0:
        print("  ⚠️ No rows in 15-30 range – skipping age sampling for this file")
        return df  # fallback: return all rows
    
    # Sample each group (if group is smaller than target, take all)
    core_sampled = core_df.sample(n=min(n_core, len(core_df)), random_state=42)
    adult_sampled = adult_df.sample(n=min(n_adult, len(adult_df)), random_state=42)
    other_sampled = other_df.sample(n=min(n_other, len(other_df)), random_state=42)
    
    # Combine
    return pd.concat([core_sampled, adult_sampled, other_sampled], ignore_index=True)

# === MAIN FUNCTION ===
def prune_folder(folder_path):
    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        print(f"❌ Error: '{folder_path}' is not a valid directory.")
        return
    
    # Create output folder with '_pruned' suffix
    output_folder = folder_path.parent / (folder_path.name + "_pruned")
    output_folder.mkdir(exist_ok=True)
    
    # Find all CSV files
    csv_files = list(folder_path.glob("*.csv"))
    if not csv_files:
        print(f"⚠️ No CSV files found in '{folder_path}'.")
        return
    
    print(f"📂 Processing {len(csv_files)} CSV files...")
    
    for file in csv_files:
        print(f"  📄 {file.name}...")
        df = pd.read_csv(file)
        original_rows = len(df)
        
        # Check for age column (case-insensitive)
        age_col = None
        for col in df.columns:
            if col.lower() == 'age':
                age_col = col
                break
        
        if age_col is None:
            print(f"    ⏩ No age column found – keeping all rows ({original_rows})")
            pruned_df = df
        else:
            print(f"    🎯 Age column: '{age_col}' – applying stratified sampling...")
            pruned_df = sample_by_age(df, age_col)
            print(f"    ✅ Original: {original_rows} → Pruned: {len(pruned_df)}")
        
        # Save to output folder
        output_path = output_folder / file.name
        pruned_df.to_csv(output_path, index=False)
    
    print(f"\n✅ All pruned datasets saved to: {output_folder}")

# === ENTRY POINT ===
if __name__ == "__main__":
    args = parse_args()
    prune_folder(args.folder)