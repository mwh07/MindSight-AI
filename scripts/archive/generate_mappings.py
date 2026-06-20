# scripts/generate_mappings.py
import os
import json
import pandas as pd

def generate_feature_mappings():
    # Resolve absolute paths from scripts/ directory to root level
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    schema_path = os.path.join(base_dir, "schema_config.json")
    output_path = os.path.join(base_dir, "feature_mappings.txt")
    
    if not os.path.exists(schema_path):
        print(f"❌ Error: schema_config.json not found at expected root path: {schema_path}")
        return

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    mapping_lines = []
    mapping_lines.append("=== MINDSIGHT AUTOMATED FEATURE MAPPING REPORT ===")
    mapping_lines.append(f"Source Schema Version: {schema.get('schema_version', 'Unknown')}\n")

    # 1. Map Exogenous Demographics
    mapping_lines.append("## EXOGENOUS DEMOGRAPHICS")
    demo_features = schema.get("exogenous_demographics", {}).get("features", [])
    
    for feat in demo_features:
        mapping_lines.append(f"{feat} -> (Live Intake Parameter / Dynamic Model Injection)")
    mapping_lines.append("")

    # 2. Map Structural Domains
    mapping_lines.append("## DOMAIN FEATURE SCHEMAS")
    domains = schema.get("domains", {})
    
    for domain_id, domain_cfg in domains.items():
        mapping_lines.append(f"### {domain_id.upper()}")
        
        # Handle single vs split source keys
        sources = {}
        if "dataset_source" in domain_cfg:
            sources["features"] = domain_cfg["dataset_source"]
        else:
            for key, val in domain_cfg.items():
                if key.endswith("_source"):
                    prefix = key.replace("_source", "")
                    sources[f"{prefix}_features"] = val

        for feature_key_attr, csv_filename in sources.items():
            # Smart Auto-Detection Paths
            possible_paths = [
                os.path.join(base_dir, csv_filename),                  # Root level
                os.path.join(base_dir, "datasets", csv_filename),      # datasets/ subfolder
                os.path.join(base_dir, "data", csv_filename)           # data/ subfolder
            ]
            
            csv_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    csv_path = path
                    break
            
            features_list = domain_cfg.get(feature_key_attr, [])
            
            if csv_path:
                try:
                    available_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
                    status_flag = "VERIFIED"
                    display_name = os.path.relpath(csv_path, base_dir).replace("\\", "/")
                except Exception:
                    available_cols = []
                    status_flag = "UNREADABLE"
                    display_name = csv_filename
            else:
                available_cols = []
                status_flag = "MISSING FILE"
                display_name = csv_filename

            mapping_lines.append(f"Source File: {display_name} [{status_flag}]")
            
            for feat in features_list:
                if available_cols and feat in available_cols:
                    mapping_lines.append(f"  {feat} -> ({display_name}/col '{feat}')")
                else:
                    mapping_lines.append(f"  {feat} -> ({display_name}/col '{feat}') ⚠️ NOT FOUND IN FILE DETECTED")
            mapping_lines.append("")

    # Write output to the root directory
    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write("\n".join(mapping_lines))

    print(f"🚀 Success! Feature mapping output file successfully generated at: {output_path}")

if __name__ == "__main__":
    generate_feature_mappings()