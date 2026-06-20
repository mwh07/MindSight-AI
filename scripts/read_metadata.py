import os
import sys
import json
import pandas as pd
import argparse
from datetime import datetime

# === MAIN FUNCTION ===
def generate_metadata(folder_path):
    # Validate folder
    if not os.path.isdir(folder_path):
        print(f"❌ Error: '{folder_path}' is not a valid directory.")
        return

    # Prepare metadata structure
    metadata = {
        "folder": folder_path,
        "generated_at": datetime.now().isoformat(),
        "datasets": {}
    }

    # Find all CSV files
    csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]

    if not csv_files:
        print(f"⚠️ No CSV files found in '{folder_path}'.")
        return

    print(f"📂 Found {len(csv_files)} CSV file(s) in '{folder_path}'...")

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        try:
            df = pd.read_csv(file_path)

            # Basic info
            rows, cols = df.shape
            missing_counts = df.isnull().sum().to_dict()
            missing_percent = (df.isnull().sum() / len(df) * 100).to_dict()

            # Numeric stats
            numeric_stats = df.describe(include='number').to_dict()

            # Categorical info (top 5 unique values)
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            categorical_sample = {}
            for col in cat_cols:
                val_counts = df[col].value_counts().head(5).to_dict()
                categorical_sample[col] = val_counts

            # Column data types
            dtypes = df.dtypes.astype(str).to_dict()

            metadata["datasets"][file] = {
                "rows": rows,
                "columns": cols,
                "column_names": df.columns.tolist(),
                "data_types": dtypes,
                "missing_values_count": missing_counts,
                "missing_values_percent": missing_percent,
                "numeric_stats": numeric_stats,
                "categorical_top_5": categorical_sample
            }
            print(f"✅ Processed: {file} ({rows} rows, {cols} cols)")

        except Exception as e:
            metadata["datasets"][file] = {"error": str(e)}
            print(f"❌ Error processing {file}: {e}")

    # === Save JSON ===
    folder_basename = os.path.basename(folder_path.rstrip('/\\'))
    json_file = os.path.join(folder_path, f"{folder_basename}_metadata.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, default=str)
    print(f"\n✅ JSON metadata saved: {json_file}")

    # === Save TXT (human readable) ===
    txt_file = os.path.join(folder_path, f"{folder_basename}_metadata.txt")
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(f"=== METADATA REPORT: {folder_basename} ===\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for filename, info in metadata["datasets"].items():
            f.write(f"--- {filename} ---\n")
            if "error" in info:
                f.write(f"ERROR: {info['error']}\n\n")
                continue

            f.write(f"Rows: {info['rows']}\n")
            f.write(f"Columns: {info['columns']}\n")
            f.write(f"Column names: {', '.join(info['column_names'])}\n")
            f.write(f"Data types: {info['data_types']}\n\n")

            # Missing values (top 5 columns by missing %)
            missing_percent = info['missing_values_percent']
            sorted_missing = sorted(missing_percent.items(), key=lambda x: x[1], reverse=True)[:5]
            f.write("Top 5 columns by missing %:\n")
            for col, pct in sorted_missing:
                if pct > 0:
                    f.write(f"  - {col}: {pct:.2f}%\n")
            if not any(pct > 0 for _, pct in sorted_missing):
                f.write("  (No missing values)\n")
            f.write("\n")

            # Numeric stats (first 5 numeric columns)
            numeric_stats = info['numeric_stats']
            if numeric_stats:
                f.write("Numeric stats (first 5 numeric columns):\n")
                count = 0
                for col, stats in numeric_stats.items():
                    if count >= 5:
                        break
                    mean = stats.get('mean', 'N/A')
                    min_ = stats.get('min', 'N/A')
                    max_ = stats.get('max', 'N/A')
                    f.write(f"  - {col}: mean={mean:.2f}, min={min_}, max={max_}\n")
                    count += 1
            else:
                f.write("No numeric columns.\n")
            f.write("\n")

            # Categorical top 5 (first 3 categorical columns)
            categorical = info['categorical_top_5']
            if categorical:
                f.write("Categorical top 5 values (first 3 columns):\n")
                count = 0
                for col, vals in categorical.items():
                    if count >= 3:
                        break
                    f.write(f"  - {col}: {vals}\n")
                    count += 1
            else:
                f.write("No categorical columns.\n")
            f.write("\n" + "="*50 + "\n\n")

    print(f"✅ TXT metadata saved: {txt_file}")

# === COMMAND-LINE ENTRY ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metadata for all CSV files in a folder.")
    parser.add_argument("folder", help="Path to the folder containing CSV files")
    args = parser.parse_args()

    generate_metadata(args.folder)