#!/usr/bin/env python3
"""
MINDSIGHT Central Control System & Automated Pipeline Orchestrator (v3.5)
Project Core Command Line Process Routing Hub.

Routes standalone or sequential operational directives via high-fidelity CLI arguments:
handles live survey intake loops, manual calibration suites, granular file system 
purges, and the full cross-domain evaluation pipeline.
"""

import os
import sys
import time
import csv
import json
import shutil
import warnings
import argparse
import subprocess
import importlib
from datetime import datetime

# Guarantee project root directory visibility across all relative imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

class MindsightNumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to downcast NumPy primitives safely during serialization."""
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)

def clear_terminal():
    """Clears the console screen across major operating systems."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    """Renders the standard system administrative dashboard interface banner."""
    print("=" * 80)
    print("      MINDSIGHT COGNITIVE PIPELINE & PSYCHOMETRIC ORCHESTRATOR")
    print("=" * 80)
    print(" Automated Engine: Version 3.5 | Advanced Argument Process Architecture")
    print("=" * 80)

def flush_directory_contents(target_dir, label="Directory"):
    """Wipes all files/folders inside a path while strictly preserving '.gitkeep'."""
    if not os.path.exists(target_dir):
        return
    print(f" Flushing active workspace -> {label}")
    for item in os.listdir(target_dir):
        item_path = os.path.join(target_dir, item)
        if item == ".gitkeep":
            continue
        try:
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
        except Exception as e:
            print(f"    Failed to purge item {item}: {str(e)}")

def are_models_trained():
    """Evaluates whether models are trained and validates against .schema_hash."""
    saved_states_dir = os.path.join(PROJECT_ROOT, "models", "saved_states")
    if not os.path.exists(saved_states_dir):
        os.makedirs(saved_states_dir, exist_ok=True)
        return False
    items = os.listdir(saved_states_dir)
    if len([item for item in items if item != ".gitkeep"]) == 0:
        return False
        
    try:
        import hashlib
        schema_path = os.path.join(PROJECT_ROOT, "schema_config.json")
        hash_path = os.path.join(PROJECT_ROOT, "models", ".schema_hash")
        with open(schema_path, "rb") as f:
            current_hash = hashlib.md5(f.read()).hexdigest()
        if not os.path.exists(hash_path):
            return False
        with open(hash_path, "r") as f:
            saved_hash = f.read().strip()
        if current_hash != saved_hash:
            return False
    except Exception:
        pass
        
    return True

def execute_training_submodule(script_relative_path, step_description):
    """Executes an individual domain training script as an isolated subprocess."""
    full_script_path = os.path.normpath(os.path.join(PROJECT_ROOT, script_relative_path))
    print(f"   [LAUNCHING] {step_description}")
    
    if not os.path.exists(full_script_path):
        print(f"   ERROR: Script missing at: {full_script_path}")
        return False
        
    execution_env = os.environ.copy()
    execution_env["PYTHONUNBUFFERED"] = "1"
    execution_env["PYTHONIOENCODING"] = "utf-8"
    
    try:
        process = subprocess.Popen(
            [sys.executable, full_script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            env=execution_env
        )
        while True:
            output_line = process.stdout.readline()
            if output_line == '' and process.poll() is not None:
                break
        return process.poll() == 0
    except Exception as e:
        print(f"   SYSTEM EXCEPTION: Calibration fault: {str(e)}")
        return False

def run_standalone_survey():
    """Spawns the interactive psychometric survey test runner."""
    print("\n DIRECTIVE: INTERACTIVE DIAGNOSTIC SURVEY INTAKE")
    print("Handing control context over to: scripts/take_assessment.py\n" + "-" * 80)
    target_script = os.path.normpath(os.path.join(PROJECT_ROOT, "scripts", "take_assessment.py"))
    
    if not os.path.exists(target_script):
        print(f" CRITICAL ERROR: Survey script missing at: {target_script}")
        return
    try:
        subprocess.run([sys.executable, target_script], check=True)
    except Exception as e:
        print(f"\n RUNTIME ERROR: Survey console failure: {str(e)}")

def run_standalone_training_suite():
    """Sequentially calibrates all six multi-domain model pipelines."""
    print("\n  DIRECTIVE: MULTI-DOMAIN MODEL CALIBRATION SUITE")
    training_manifest = [
        ("models/train_domain1_personality.py", "Domain 1: Personality Trait Vectors"),
        ("models/train_domain2_self_esteem.py", "Domain 2: Self-Esteem Empirical Mapping"),
        ("models/train_domain3_mood_sleep.py", "Domain 3: Mood Severity & Sleep Timing"),
        ("models/train_domain4_multitask.py", "Domain 4: Attachment & Loneliness Forest"),
        ("models/train_domain5_burnout.py", "Domain 5: Stress & Burnout Engine"),
        ("models/train_domain6_clinical.py", "Domain 6: Severe Clinical Screening Matrix")
    ]
    for script_path, description in training_manifest:
        if not execute_training_submodule(script_path, description):
            print(f"\n Build failed during processing of {description}.")
            sys.exit(1)
            
    try:
        import hashlib
        schema_path = os.path.join(PROJECT_ROOT, "schema_config.json")
        hash_path = os.path.join(PROJECT_ROOT, "models", ".schema_hash")
        with open(schema_path, "rb") as f:
            current_hash = hashlib.md5(f.read()).hexdigest()
        with open(hash_path, "w") as f:
            f.write(current_hash)
        print(f" Schema hash updated to {current_hash}")
    except Exception as e:
        print(f" Failed to update schema hash: {e}")
        
    print(" All 6 structural domains calibrated successfully.")

def run_evaluation_pipeline(custom_csv_path=None):
    """Runs the core deterministic 5-phase evaluation and packaging sequence."""
    individual_eval_dir = os.path.normpath(os.path.join(PROJECT_ROOT, "individual_eval"))
    os.makedirs(individual_eval_dir, exist_ok=True)
    
    print("\n[PHASE 1] INITIALIZING RUNTIME WORKSPACE")
    flush_directory_contents(individual_eval_dir, "individual_eval/")
    
    print("\n[PHASE 2] VALIDATING MODEL PACKAGES")
    if not are_models_trained():
        print(" Models flagged as UNTRAINED. Auto-triggering calibration suite...")
        run_standalone_training_suite()
    else:
        print(" Pre-calibrated model weights discovered.")

    print("\n[PHASE 3] INGESTING INTERACTIVE SURVEY RECORD")
    csv_path = custom_csv_path if custom_csv_path else os.path.normpath(os.path.join(PROJECT_ROOT, "tests", "responses.csv"))
    if not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0:
        print(f" CRITICAL CONFIGURATION FAULT: Ledger missing at: {csv_path}")
        sys.exit(1)
        
    raw_payload = {}
    try:
        with open(csv_path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                raise ValueError("Spreadsheet tracker contains 0 record rows.")
            raw_payload = rows[-1]
    except Exception as e:
        print(f" RECOVERY FAULT: Processing ledger failed: {str(e)}")
        sys.exit(1)

    # --- NEW: Append the current response to tests/all_responses.csv ---
    all_responses_path = os.path.join(PROJECT_ROOT, "tests", "all_responses.csv")
    
    ordered_headers = [
        'age', 'gender',
        'EXT1', 'EXT2', 'EXT3', 'EST1', 'EST2', 'EST3', 'AGR1', 'AGR2', 'AGR3', 'CSN1', 'CSN2', 'CSN3', 'OPN1', 'OPN2', 'OPN3',
        'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10',
        'DPQ010', 'DPQ020', 'DPQ030', 'DPQ040', 'DPQ050', 'DPQ060', 'DPQ070', 'DPQ080', 'DPQ090', 'DPQ100',
        'SLQ300', 'SLQ310',
        'IAT1', 'IAT2', 'IAT3', 'IAT4', 'IAT5', 'IAT6', 'IAT7', 'IAT8', 'IAT9', 'IAT10',
        'loneliness1', 'loneliness2', 'loneliness3', 'loneliness4', 'loneliness5', 'loneliness6',
        'work_hours_per_week', 'meetings_per_day', 'work_life_balance_score', 'job_satisfaction_score', 
        'deadline_pressure_score', 'autonomy_score', 'stress_score', 'social_support_score',
        'unwanted_thoughts', 'repetitive_behaviors', 'overthinking', 'mind_going_blank', 
        'avoidance_social_activity', 'panic', 'hypervigilance'
    ]
    
    file_exists = os.path.isfile(all_responses_path)
    with open(all_responses_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=ordered_headers, extrasaction='ignore', restval=0)
        if not file_exists or os.stat(all_responses_path).st_size == 0:
            writer.writeheader()
        writer.writerow(raw_payload)
    print(f"   Appended response to {all_responses_path}")

    sanitized_payload = {}
    for column_id, string_value in raw_payload.items():
        val_stripped = string_value.strip() if string_value else ""
        if column_id in ['SLQ300', 'SLQ310']:
            sanitized_payload[column_id] = val_stripped
            continue
        try:
            if '.' in val_stripped:
                sanitized_payload[column_id] = float(val_stripped)
            else:
                sanitized_payload[column_id] = int(val_stripped)
        except ValueError:
            sanitized_payload[column_id] = val_stripped

    print("\n[PHASE 4] RUNNING MULTI-DOMAIN MULTITASK INFERENCE")
    timestamp_token = datetime.now().strftime("%Y%m%d_%H%M%S")
    sandbox_dir_name = f"eval_{timestamp_token}"
    target_output_dir = os.path.normpath(os.path.join(individual_eval_dir, sandbox_dir_name))
    os.makedirs(target_output_dir, exist_ok=True)
    
    try:
        from models.profile_aggregator import generate_full_profile
        
        compiled_profile = generate_full_profile(sanitized_payload)
        
        output_json_path = os.path.join(target_output_dir, f"compiled_profile_{sandbox_dir_name}.json")
        
        with open(output_json_path, "w", encoding='utf-8') as json_file:
            json.dump(compiled_profile, json_file, indent=2, cls=MindsightNumpyEncoder)
        print(f"   Captured structural data payload -> {output_json_path}")
    except Exception as e:
        print(f" PIPELINE ERROR: Analytics evaluation run crashed: {str(e)}")
        sys.exit(1)

    print("\n[PHASE 5] PACKAGING SYSTEM ACCOUNTABILITY ARCHIVES")
    archive_reports_dir = os.path.normpath(os.path.join(PROJECT_ROOT, "reports", f"report_{timestamp_token}"))
    os.makedirs(archive_reports_dir, exist_ok=True)
    
    # Copy compiled JSON and CSV to the timestamped archive
    shutil.copy(output_json_path, os.path.join(archive_reports_dir, f"compiled_profile_{sandbox_dir_name}.json"))
    print(f"   Archived profile JSON -> {archive_reports_dir}")
    
    dest_csv_path = os.path.join(archive_reports_dir, "responses.csv")
    shutil.copy(csv_path, dest_csv_path)
    print(f"   Archived responses CSV -> {archive_reports_dir}")
    
    # Prepare current_report inside individual_eval/ (overwrite if exists)
    current_report_dir = os.path.join(individual_eval_dir, "current_report")
    if os.path.exists(current_report_dir):
        shutil.rmtree(current_report_dir)
    os.makedirs(current_report_dir, exist_ok=True)
    
    shutil.copy(output_json_path, os.path.join(current_report_dir, f"compiled_profile_{sandbox_dir_name}.json"))
    shutil.copy(csv_path, os.path.join(current_report_dir, "responses.csv"))
    print(f"   Saved as current_report -> {current_report_dir}")
    
    # Clean up the intermediate eval_* sandbox directory
    shutil.rmtree(target_output_dir)
    print(f"   Cleaned intermediate sandbox -> {target_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MINDSIGHT Orchestrator - Production Driver Matrix",
        add_help=False
    )
    # Mode Selectors
    parser.add_argument('-s', '--survey', action='store_true', help="Spawn live survey intake interface (take_assessment.py)")
    parser.add_argument('-t', '--train', action='store_true', help="Manually retrain all 6 machine learning domains")
    parser.add_argument('-e', '--eval', action='store_true', help="Processes latest response ledger (Default action)")
    
    # Purging Controls
    parser.add_argument('--fl', action='store_true', help="Alias for flushing training data weights")
    parser.add_argument('--flush-train', action='store_true', help="Clear weights under models/saved_states/")
    parser.add_argument('--flush-reports', action='store_true', help="Erase historical archive packages under reports/")
    parser.add_argument('--flush-eval', action='store_true', help="Clean working directories under individual_eval/")
    parser.add_argument('--flush-all', action='store_true', help="Structural reset across all three zones")
    
    # Advanced Diagnostics Configuration
    parser.add_argument('--csv-path', type=str, default=None, help="Direct the intake script to parse a customized input CSV path.")
    
    # Documentation Info
    parser.add_argument('-m', '--manual', action='store_true', help="Display manual schematic showing all args and workflow details.")
    parser.add_argument('-h', '--help', action='store_true', help="Show help dialog.")
    
    args = parser.parse_args()
    
    if args.help or args.manual:
        clear_terminal()
        print("=" * 80)
        print("              MINDSIGHT ARCHITECTURE INTERFACE MANUAL CLI CODES")
        print("=" * 80)
        print("OPERATIONAL MODES:")
        print("  -s, --survey      : Spawn live survey intake interface (take_assessment.py)")
        print("  -t, --train       : Manually retrain all 6 machine learning domains")
        print("  -e, --eval        : Processes latest response ledger (Default action)")
        print("\nPURGING CONTROLS:")
        print("  --fl              : Alias for flushing training data weights")
        print("  --flush-train     : Clear weights under models/saved_states/")
        print("  --flush-reports   : Erase historical archive packages under reports/")
        print("  --flush-eval      : Clean working directories under individual_eval/")
        print("  --flush-all       : Structural reset across all three zones")
        print("\nADVANCED DIAGNOSTICS CONFIGURATION:")
        print("  --csv-path PATH   : Direct the intake script to parse a customized input CSV path.")
        print("\nSYSTEM WORKFLOW SCHEMATIC DETAILS:")
        print("  [Phase 1] Initial Workspace Clean: Flushes everything inside individual_eval/ folder.")
        print("  [Phase 2] Weight Validation Check: Assesses models/saved_states/. Runs setup if empty.")
        print("  [Phase 3] Intake File Parsing     : Extracts the last row index from the target CSV file.")
        print("  [Phase 4] Inference Processing    : Resolves model states, writes timestamped JSON (PDF disabled).")
        print("  [Phase 5] Structural Archival     : Packages JSON + CSV to reports/report_[timestamp]/ and as current_report in individual_eval/, then removes intermediate sandbox.")
        print("=" * 80)
        sys.exit(0)
        
    try:
        clear_terminal()
        print_banner()
        
        # Execute Maintenance Purges first if called
        if args.flush_all:
            flush_directory_contents(os.path.join(PROJECT_ROOT, "models", "saved_states"), "models/saved_states/")
            flush_directory_contents(os.path.join(PROJECT_ROOT, "reports"), "reports/")
            flush_directory_contents(os.path.join(PROJECT_ROOT, "individual_eval"), "individual_eval/")
        else:
            if args.fl or args.flush_train:
                flush_directory_contents(os.path.join(PROJECT_ROOT, "models", "saved_states"), "models/saved_states/")
            if args.flush_reports:
                flush_directory_contents(os.path.join(PROJECT_ROOT, "reports"), "reports/")
            if args.flush_eval:
                flush_directory_contents(os.path.join(PROJECT_ROOT, "individual_eval"), "individual_eval/")
        
        # Route Operational Tasks
        warnings.filterwarnings("ignore", category=UserWarning)
        if args.survey:
            run_standalone_survey()
        elif args.train:
            run_standalone_training_suite()
        elif args.eval:
            run_evaluation_pipeline(custom_csv_path=args.csv_path)
        else:
            # Default action if no operational mode parameter is explicitly passed
            if not any([args.survey, args.train]):
                print("\n No operational selector assigned. Running standard automated evaluation cycle...")
                run_evaluation_pipeline(custom_csv_path=args.csv_path)
                
        print("\n" + "=" * 80)
        print(" MINDSIGHT PROCESS ROUTER OVERSEE COMPLETED SMOOTHLY")
        print("=" * 80 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n Core execution override signal caught. Stopping master router cleanly.")
        sys.exit(0)