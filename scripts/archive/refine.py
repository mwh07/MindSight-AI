import os
import pandas as pd
from pathlib import Path

def identify_dataset_by_footprint(file_path):
    """
    Safely inspects the header of a file using an exact-match validation grid
    to completely prevent delimiter mismatches on tab-separated files.
    Returns: (dataset_key, detected_separator, detected_encoding)
    """
    encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'utf-16']
    delimiters_to_try = [',', '\t', ';']
    
    for encoding in encodings_to_try:
        for sep in delimiters_to_try:
            try:
                # Read just a 5-row matrix slice to identify layout structure safely
                df_preview = pd.read_csv(file_path, sep=sep, encoding=encoding, nrows=5, skip_blank_lines=True)
                
                # Strip out potential BOM characters (\ufeff) and whitespaces, then lowercase everything
                actual_cols = [str(c).replace('\ufeff', '').strip().lower() for c in df_preview.columns]
                
                # --- EXACT FOOTPRINT MATCHING ENGINE ---
                if 'ext1' in actual_cols:
                    return "big_five", sep, encoding
                    
                if 'dpq010' in actual_cols:
                    return "nhanes_depression", sep, encoding
                    
                if 'slq300' in actual_cols:
                    return "nhanes_sleep", sep, encoding
                    
                if 'burnout_score' in actual_cols or 'burnout_level' in actual_cols:
                    return "tech_burnout", sep, encoding
                    
                if 'unwanted_thoughts' in actual_cols:
                    return "ocd_symptoms", sep, encoding
                    
                if 'tot_da' in actual_cols:
                    return "digital_addiction", sep, encoding
                    
                if 'lonelinesstotal' in actual_cols or 'totalphq' in actual_cols:
                    return "internet_loneliness", sep, encoding
                    
                if 'stress_level' in actual_cols or 'anxiety_score' in actual_cols:
                    return "global_mental_health", sep, encoding
                    
                if 'q1' in actual_cols and 'q10' in actual_cols:
                    return "rosenberg_self_esteem", sep, encoding
                    
            except Exception:
                continue
                
    return None, None, None

def run_refinement_pipeline(source_root_dir):
    """
    Recursively scans all subfolders, accurately matches profiles via exact-match footprints,
    prunes unneeded vectors, and saves output to a folder relative to the script location.
    """
    # Establish relative directory boundaries dynamically
    script_directory = Path(__file__).resolve().parent
    output_directory = script_directory / "Refined_Datasets"
    output_directory.mkdir(parents=True, exist_ok=True)
    
    print("="*75)
    print(f"🚀 ROBUST PIPELINE INITIALIZED")
    print(f"📂 Clean Data Target Destination: {output_directory}")
    print("="*75)

    # Pre-calculated structural target definitions
    big_five_targets = [f"{t}{i}" for t in ['EXT', 'EST', 'AGR', 'CSN', 'OPN'] for i in range(1, 11)]
    mendeley_ia_targets = [f"IA_{i}" for i in range(1, 13)] + [f"BSMA_{i}" for i in range(1, 7)]
    loneliness_phq_targets = [f"IAT{i}" for i in range(1, 21)] + [f"phq{i}" for i in range(1, 10)] + [f"loneliness{i}" for i in range(1, 21)]
    rosenberg_targets = [f"Q{i}" for i in range(1, 11)]

    manifest_spec = {
        "big_five": {
            "output_name": "big_five_personality_clean.csv",
            "required_cols": big_five_targets
        },
        "rosenberg_self_esteem": {
            "output_name": "rosenberg_self_esteem_clean.csv",
            "required_cols": ['gender', 'age'] + rosenberg_targets
        },
        "nhanes_depression": {
            "output_name": "nhanes_depression_clean.csv",
            "required_cols": ['SEQN', 'DPQ010', 'DPQ020', 'DPQ030', 'DPQ040', 'DPQ050', 'DPQ060', 'DPQ070', 'DPQ080', 'DPQ090', 'DPQ100']
        },
        "nhanes_sleep": {
            "output_name": "nhanes_sleep_clean.csv",
            "required_cols": ['SEQN', 'SLQ300', 'SLQ310', 'SLD012', 'SLQ320', 'SLQ330', 'SLD013']
        },
        "global_mental_health": {
            "output_name": "global_mental_health_2025_clean.csv",
            "required_cols": ['Patient_ID', 'Age', 'Gender', 'Depression_Score', 'Anxiety_Score', 'Stress_Level', 'Sleep_Hours', 'Work_Status']
        },
        "internet_loneliness": {
            "output_name": "internet_phq_loneliness_clean.csv",
            "required_cols": ['age', 'gender', 'TotalIA', 'totalphq', 'categoryphq', 'lonelinesstotal', 'Lonelinesscategory'] + loneliness_phq_targets
        },
        "tech_burnout": {
            "output_name": "tech_burnout_2026_clean.csv",
            "required_cols": ['age', 'gender', 'work_hours_per_week', 'meetings_per_day', 'work_life_balance_score', 'job_satisfaction_score', 'social_support_score', 'deadline_pressure_score', 'autonomy_score', 'stress_score', 'burnout_score', 'burnout_level']
        },
        "ocd_symptoms": {
            "output_name": "ocd_symptoms_clean.csv",
            "required_cols": ['Disease', 'unwanted_thoughts', 'repetitive_behaviors', 'overthinking', 'mind_going_blank', 'avoidance_social_activity', 'panic', 'hypervigilance', 'sleep_disturbances', 'low_energy']
        },
        "digital_addiction": {
            "output_name": "digital_addiction_mendeley_clean.csv",
            "required_cols": ['AGE', 'Gender', 'tot_IA', 'tot_DA', 'tot_BSMA'] + mendeley_ia_targets
        }
    }

    processed_counter = 0

    # Crawl files recursively through all primary folders and deep subfolders
    for path_item in Path(source_root_dir).rglob("*.csv"):
        # Prevent processing recursive loops inside our own output folder
        if output_directory in path_item.parents:
            continue
            
        detected_key, sep, enc = identify_dataset_by_footprint(path_item)
        
        if detected_key:
            spec = manifest_spec[detected_key]
            print(f"🎯 MATCH CONFIRMED!")
            print(f"   📍 File: ...{os.sep}{os.path.join(*path_item.parts[-2:])}")
            print(f"   🧬 Type: {detected_key.upper()}")
            print(f"   ⚙️  Specs: Delimiter=[{repr(sep)}], Encoding=[{enc}]")
            
            try:
                # Load using the parsed verification layout configurations
                df = pd.read_csv(path_item, sep=sep, encoding=enc, low_memory=False)
                
                # Strip out potential system BOM anomalies and structural whitespaces from headers
                df.columns = [str(c).replace('\ufeff', '').strip() for c in df.columns]
                
                # Map available columns case-insensitively 
                file_columns_lower_map = {c.lower(): c for c in df.columns}
                final_extraction_list = []
                
                for target_col in spec["required_cols"]:
                    target_lower = target_col.lower()
                    if target_lower in file_columns_lower_map:
                        final_extraction_list.append(file_columns_lower_map[target_lower])
                
                # Slice down to targeted columns
                pruned_df = df[final_extraction_list].copy()
                
                # Drop rows containing missing values safely
                pruned_df.dropna(inplace=True)
                
                # Export as uniform comma-separated CSV directly to execution directory
                export_destination = output_directory / spec["output_name"]
                pruned_df.to_csv(export_destination, index=False, sep=",")
                
                print(f"   💾 Saved: {spec['output_name']}")
                print(f"   📊 Matrix: {pruned_df.shape[0]} rows x {pruned_df.shape[1]} columns")
                processed_counter += 1
                
            except Exception as e:
                print(f"   ❌ Processing Error on this element block: {str(e)}")
            print("-" * 75)

    print(f"🎉 Pipeline Executed Successfully. Completely refined {processed_counter} datasets.")

if __name__ == "__main__":
    # --- FILL IN THE ABSOLUTE PATH TO YOUR TOP-LEVEL 'Datasets' FOLDER ---
    TARGET_RAW_DATASETS_FOLDER = r"C:\Users\Lenovo\Desktop\Datasets"
    
    run_refinement_pipeline(TARGET_RAW_DATASETS_FOLDER)