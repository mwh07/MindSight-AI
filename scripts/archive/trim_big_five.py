import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\Lenovo\Desktop\MINDSIGHT\datasets\big_five_personality_clean.csv")
print(f"Original shape: {df.shape}")

# --- Step 1: Drop rows with more than 10 missing values ---
df = df.dropna(thresh=40)
print(f"After NaN drop: {df.shape}")

# --- Step 2: Remove straightliners ---
row_std = df.std(axis=1)
df = df[row_std >= 0.5]
print(f"After straightliner removal: {df.shape}")

# --- Step 3: Compute Big Five trait scores ---
df['EXT'] = df[['EXT1','EXT2','EXT3','EXT4','EXT5',
                 'EXT6','EXT7','EXT8','EXT9','EXT10']].mean(axis=1)
df['EST'] = df[['EST1','EST2','EST3','EST4','EST5',
                 'EST6','EST7','EST8','EST9','EST10']].mean(axis=1)
df['AGR'] = df[['AGR1','AGR2','AGR3','AGR4','AGR5',
                 'AGR6','AGR7','AGR8','AGR9','AGR10']].mean(axis=1)
df['CSN'] = df[['CSN1','CSN2','CSN3','CSN4','CSN5',
                 'CSN6','CSN7','CSN8','CSN9','CSN10']].mean(axis=1)
df['OPN'] = df[['OPN1','OPN2','OPN3','OPN4','OPN5',
                 'OPN6','OPN7','OPN8','OPN9','OPN10']].mean(axis=1)

# --- Step 4: Bin each trait into Low / Mid / High ---
for trait in ['EXT', 'EST', 'AGR', 'CSN', 'OPN']:
    df[f'{trait}_bin'] = pd.cut(df[trait], bins=3, labels=['Low', 'Mid', 'High'])

df['strata'] = (df['EXT_bin'].astype(str) + '_' +
                df['EST_bin'].astype(str) + '_' +
                df['AGR_bin'].astype(str))

# --- Step 5: Stratified sample to ~200K rows ---
TARGET = 200_000
df_sampled = df.groupby('strata', group_keys=False).apply(
    lambda x: x.sample(frac=TARGET/len(df), random_state=42)
    if len(x) > 1 else x
)

# --- Step 6: Keep only original 50 columns ---
df_final = df_sampled[df.columns[:50]].reset_index(drop=True)
print(f"Final shape: {df_final.shape}")

# --- Save as a NEW file ---
df_final.to_csv(
    r"C:\Users\Lenovo\Desktop\MINDSIGHT\datasets\big_five_personality_pruned.csv",
    index=False
)
print("Saved as big_five_personality_pruned.csv")