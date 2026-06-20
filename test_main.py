#!/usr/bin/env python3
"""
MINDSIGHT Central Control System & Cognitive Pipeline Orchestrator (v2.8)
Project Core Master Dashboard and Analytical Process Router.

Maintains strict structural compliance across intake processing, localized directory 
isolation under individual_eval/, and direct multi-domain model training loops.
"""

import os
import sys
import time
import csv
import json
import warnings
import subprocess
import importlib  # Required for clean runtime module cache flushing
from datetime import datetime

# Guarantee project root directory visibility across all relative imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

class MindsightNumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to automatically downcast NumPy data types 
    (like int64, float64, and ndarrays) into standard Python primitives
    so that json.dump doesn't crash after ML inference runs.
    """
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
    print(" Core Engine: Version 2.8 | Workspace Sandbox Isolation Architecture")
    print("=" * 80)

def execute_training_submodule(script_relative_path, step_description):
    """
    Executes an individual domain training script as an isolated subprocess,
    streaming runtime build logs directly to the console in real-time.
    """
    full_script_path = os.path.normpath(os.path.join(PROJECT_ROOT, script_relative_path))
    
    print(f"\n🚀 [LAUNCHING] {step_description}...")
    print(f"👉 Target: {full_script_path}")
    print("-" * 80)
    
    if not os.path.exists(full_script_path):
        print(f"❌ ERROR: Script file missing at target path.")
        return False
        
    start_time = time.time()
    
    # Configure unbuffered system streams to mirror real-time progress outputs
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
        
        # Real-time console buffer line reading routine
        while True:
            output_line = process.stdout.readline()
            if output_line == '' and process.poll() is not None:
                break
            if output_line:
                print(f"  │ {output_line.strip()}")
                
        return_code = process.poll()
        elapsed_time = time.time() - start_time
        print("-" * 80)
        
        if return_code == 0:
            print(f"✅ [SUCCESS] Finalized smoothly in {elapsed_time:.2f}s.")
            return True
        else:
            print(f"❌ [FAILURE] Terminated prematurely with exit code {return_code}.")
            return False
            
    except Exception as e:
        print(f"❌ SYSTEM EXCEPTION: Runtime fault caught during execution: {str(e)}")
        return False

def run_flush_training_data():
    """Executes the training data flush script."""
    print("\n🧹 Initiating training data flush procedure...")
    script_path = os.path.join(PROJECT_ROOT, "scripts", "flush_training_data.py")
    if os.path.exists(script_path):
        subprocess.run([sys.executable, script_path])
    else:
        print(f"⚠️ Training data flush script not found at: {script_path}")
    time.sleep(1.5)

def run_flush_reports():
    """Executes the evaluation reports flush script."""
    print("\n🧹 Initiating evaluation reports flush procedure...")
    script_path = os.path.join(PROJECT_ROOT, "scripts", "flush_reports.py")
    if os.path.exists(script_path):
        subprocess.run([sys.executable, script_path])
    else:
        print(f"⚠️ Reports flush script not found at: {script_path}")
    time.sleep(1.5)

def run_flush_both():
    """Sequentially executes both training data and reports flush scripts."""
    print("\n🔄 Initiating comprehensive system flush (Training Data & Reports)...")
    run_flush_training_data()
    run_flush_reports()

def run_live_questionnaire():
    """
    [Directive 1] Spawns the interactive psychometric survey test runner.
    Hands over input/output controls entirely to allow native keyboard interaction.
    """
    clear_terminal()
    print_banner()
    print("\n📝 DIRECTIVE 1: INTERACTIVE DIAGNOSTIC SURVEY INTAKE")
    print("Spawning intake window wrapper stream...")
    print("Handing control context over to: scripts/take_assessment.py\n")
    print("-" * 80)
    
    target_script = os.path.normpath(os.path.join(PROJECT_ROOT, "scripts", "take_assessment.py"))
    
    if not os.path.exists(target_script):
        print(f"❌ CRITICAL ERROR: Survey script missing at: {target_script}")
        input("\nPress [ENTER] to return to the main dashboard...")
        return

    try:
        # Launching with native sys.stdin/stdout context for keyboard responses
        subprocess.run([sys.executable, target_script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n⚠️ Assessment run halted: Subprocess exited with status code {e.returncode}")
    except Exception as e:
        print(f"\n❌ RUNTIME ERROR: Failed to instantiate live survey console: {str(e)}")
        
    input("\nPress [ENTER] to return to the main dashboard menu...")

def run_evaluation_pipeline():
    """
    [Directive 2] Directly processes the single active intake response ledger.
    Parses values with smart type-casting, establishes an isolated sandboxed workspace, 
    and drives cross-domain ML synthesis to output JSON summaries and PDF layout grids.
    """
    # Mute scikit-learn feature name UserWarnings to keep console pristine
    warnings.filterwarnings("ignore", category=UserWarning)
    
    clear_terminal()
    print_banner()
    print("\n📊 DIRECTIVE 2: PSYCHOMETRIC EVALUATION PIPELINE RUNNER")
    print("Target Ledger Context: tests/responses.csv (Active Single-Session Row)\n")
    print("-" * 80)
    
    csv_path = os.path.normpath(os.path.join(PROJECT_ROOT, "tests", "responses.csv"))
    
    if not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0:
        print("❌ CRITICAL ERROR: Active assessment row registry file cannot be found.")
        print(f"👉 Expected Path: {csv_path}")
        print("💡 Please run option [1] first to generate an active assessment session entry.")
        input("\nPress [ENTER] to abort and head back to the main menu...")
        return
        
    print(f"📖 Ingesting active questionnaire entries from path: {csv_path}...")
    
    raw_payload = {}
    try:
        with open(csv_path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                raise ValueError("The target responses.csv exists but contains no active data rows.")
            
            # FIX 1: Focus strictly on the LATEST appended intake session (rows[-1]) instead of the first historical row (rows[0])
            raw_payload = rows[-1] 
    except Exception as e:
        print(f"❌ DATA INGESTION ERROR: Failed parsing spreadsheet: {str(e)}")
        input("\nPress [ENTER] to return to the main dashboard...")
        return

    # --- High-Fidelity Type-Casting Core Matrix Engine ---
    sanitized_payload = {}
    for column_id, string_value in raw_payload.items():
        val_stripped = string_value.strip() if string_value else ""
        
        # Enforce structural exceptions: Sleep timestamps must remain strings
        if column_id in ['SLQ300', 'SLQ310']:
            sanitized_payload[column_id] = val_stripped
            continue
            
        # Dynamically cast numeric distributions without losing accuracy
        try:
            if '.' in val_stripped:
                sanitized_payload[column_id] = float(val_stripped)
            else:
                sanitized_payload[column_id] = int(val_stripped)
        except ValueError:
            # Fallback for unexpected characters or categorical labels
            sanitized_payload[column_id] = val_stripped

    print("✅ Schema parsing and data-type matrix casting completed successfully.")

    try:
        print("🤖 Instantiating and flushing multi-domain inference engines...")
        
        # FIX 2: Evict stale script references from memory to force-load code modifications
        import models.inference_wrappers
        import models.profile_aggregator
        import models.pdf_generator
        
        importlib.reload(models.inference_wrappers)
        importlib.reload(models.profile_aggregator)
        importlib.reload(models.pdf_generator)
        
        from models.profile_aggregator import generate_full_profile
        from models.pdf_generator import compile_pdf_report
        
        print("🧠 Processing backend cross-domain interactions and synthesis metrics...")
        compiled_profile = generate_full_profile(sanitized_payload)
        
        # --- Strict Directory Isolation Protocol Implementation ---
        timestamp_token = datetime.now().strftime("%Y%m%d_%H%M%S")
        sandbox_dir_name = f"eval_{timestamp_token}"
        target_output_dir = os.path.normpath(os.path.join(PROJECT_ROOT, "individual_eval", sandbox_dir_name))
        
        os.makedirs(target_output_dir, exist_ok=True)
        
        output_json_path = os.path.join(target_output_dir, f"compiled_profile_{sandbox_dir_name}.json")
        output_pdf_path = os.path.join(target_output_dir, f"Mindsight_Report_{sandbox_dir_name}.pdf")
        
        # Flush structured structural metadata profile using custom NumPy Encoder
        with open(output_json_path, "w", encoding='utf-8') as json_file:
            json.dump(compiled_profile, json_file, indent=2, cls=MindsightNumpyEncoder)
        print(f"  ➡️  High-fidelity profile structure saved to -> {output_json_path}")
        
        # Flush visual ReportLab layout presentation graph
        print("🎨 Constructing ReportLab flowable layout grids and rendering PDF dashboard...")
        compile_pdf_report(compiled_profile, output_pdf_path)
        print(f"  ➡️  Publication-quality visual report compiled to -> {output_pdf_path}")
        
        print("\n" + "=" * 80)
        print("✨ CORE INTEGRATED ASSESSMENT PIPELINE RECONCILIATION COMPLETED")
        print(f"Isolated Sandbox Workspace: {target_output_dir}{os.sep}")
        print("=" * 80)
        
    except ImportError as ie:
        print("\n❌ ARCHITECTURE ERROR: Failed importing multi-domain core models.")
        print(f"Details: {str(ie)}")
        print("👉 Please run Directive [3] first to guarantee saved state serialization.")
    except Exception as e:
        print(f"\n❌ PIPELINE CRASH: Severe anomaly hit during pipeline run: {str(e)}")
        import traceback
        traceback.print_exc()
        
    input("\nPress [ENTER] to return to the main dashboard menu...")

def run_automated_training_suite():
    """
    [Directive 3] Sequentially steps through and triggers all six multi-domain 
    model training pipelines found directly inside the models/ directory.
    """
    clear_terminal()
    print_banner()
    print("\n🛠️  DIRECTIVE 3: MULTI-DOMAIN MODEL CALIBRATION SUITE")
    print("This suite executes all 6 domain modules sequentially to build and serialize")
    print("the required models and empirical lookup tables.\n")
    
    confirm = input("Proceed with training all 6 analytical backends? (y/n): ").strip().lower()
    if confirm != 'y':
        return

    # Define execution manifests mapping exactly to your verified repository structure
    training_manifest = [
        ("models/train_domain1_personality.py", "Domain 1: GRM-IRT Personality Trait Vectors"),
        ("models/train_domain2_self_esteem.py", "Domain 2: Rosenberg Self-Esteem Empirical Mapping"),
        ("models/train_domain3_mood_sleep.py", "Domain 3: Mood Severity & Sleep Timing Estimator"),
        ("models/train_domain4_multitask.py", "Domain 4: Digital Attachment & Loneliness Forest"),
        ("models/train_domain5_burnout.py", "Domain 5: Occupational Stress & Burnout Engine"),
        ("models/train_domain6_clinical.py", "Domain 6: Severe Clinical Screening Matrix")
    ]
    
    success_count = 0
    for script_path, description in training_manifest:
        if execute_training_submodule(script_path, description):
            success_count += 1
        else:
            print(f"\n⚠️ Pipeline execution halted: Build failed during processing of {description}.")
            break
            
    print("\n" + "=" * 80)
    print(f"🎉 PIPELINE TRACKER: {success_count}/6 Domain Modules Calibrated Successfully.")
    print("=" * 80)
    input("\nPress [ENTER] to return to the main dashboard menu...")

def main_dashboard():
    """Main administrative execution dispatch switchboard engine loop."""
    while True:
        clear_terminal()
        print_banner()
        print("\n Operational Directives Dashboard Options:")
        print(" [1] 📝 Run Diagnostic Questionnaire (Process Live Survey Intake)")
        print(" [2] 📊 Execute Evaluation Pipeline  (Process tests/responses.csv directly)")
        print(" [3] 🛠️  Retrain Multi-Domain Backends (Sequential Calibration Suite)")
        print(" [4] 🗑️  Flush Training Data")
        print(" [5] 🗑️  Flush Reports")
        print(" [6] 🔄 Flush Both (Training Data & Reports)")
        print(" [7] ❌ Close Console Connection")
        print("-" * 80)
        
        user_selection = input("👉 Select Directive Option (1-7): ").strip()
        
        if user_selection == '1':
            run_live_questionnaire()
        elif user_selection == '2':
            run_evaluation_pipeline()
        elif user_selection == '3':
            run_automated_training_suite()
        elif user_selection == '4':
            run_flush_training_data()
        elif user_selection == '5':
            run_flush_reports()
        elif user_selection == '6':
            run_flush_both()
        elif user_selection == '7':
            clear_terminal()
            print("\n👋 MINDSIGHT core system pipeline safely decoupled. Orchestrator offline.\n")
            break
        else:
            print("⚠️ Selection parameter invalid. Please supply an integer indexing token between 1 and 7.")
            time.sleep(1.5)

if __name__ == "__main__":
    try:
        main_dashboard()
    except KeyboardInterrupt:
        print("\n\n⚠️ Core hardware execution override signal caught. Stopping master router cleanly.")
        sys.exit(0)
