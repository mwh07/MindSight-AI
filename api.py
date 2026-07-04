#!/usr/bin/env python3
"""
MINDSIGHT REST API - Flask Wrapper for the Orchestrator System
"""

import os
import sys
import json
import tempfile
import csv
import shutil
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

app = Flask(__name__)

# Enable CORS for specific origins to prevent CSRF and external API abuse
CORS(app, origins=['http://localhost:5173', 'http://127.0.0.1:5173'])

DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() == "true"

def debug_print(message, level="INFO"):
    """Custom debug printer"""
    if DEBUG:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        print(f"[{timestamp}] [{level}] {message}")
        sys.stdout.flush()

# Rest of your code remains the same...
@app.route('/test', methods=['GET'])
def test():
    """Simple test endpoint"""
    debug_print(" Test endpoint called", "INFO")
    return jsonify({
        'message': 'Code running perfectly here! ',
        'status': 'Server is working',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    debug_print(" Health check called", "INFO")
    
    # Check if models are trained
    saved_states_dir = os.path.join(PROJECT_ROOT, "models", "saved_states")
    models_trained = False
    if os.path.exists(saved_states_dir):
        items = os.listdir(saved_states_dir)
        if len([item for item in items if item != ".gitkeep"]) > 0:
            try:
                import hashlib
                schema_path = os.path.join(PROJECT_ROOT, "schema_config.json")
                hash_path = os.path.join(PROJECT_ROOT, "models", ".schema_hash")
                with open(schema_path, "rb") as f:
                    current_hash = hashlib.md5(f.read()).hexdigest()
                if os.path.exists(hash_path):
                    with open(hash_path, "r") as f:
                        saved_hash = f.read().strip()
                    if current_hash == saved_hash:
                        models_trained = True
            except Exception:
                pass
    
    return jsonify({
        'status': 'healthy',
        'service': 'MINDSIGHT API',
        'timestamp': datetime.now().isoformat(),
        'models_trained': models_trained,
        'debug_mode': DEBUG,
        'project_root': PROJECT_ROOT
    })

@app.route('/assess', methods=['POST', 'OPTIONS'])
def assess():


    debug_print("=" * 60, "START")
    debug_print(" Received POST request to /assess", "INFO")

    try:
        # Step 1: Get JSON data
        debug_print(" Step 1: Extracting JSON data", "INFO")
        data = request.get_json()
        if not data:
            debug_print(" No data provided", "ERROR")
            return jsonify({'error': 'No data provided'}), 400
        debug_print(f" Received {len(data)} fields", "SUCCESS")


        # --- NEW: Save response to tests/responses.csv and append to tests/all_responses.csv ---
        debug_print(" Step 1.5: Saving response to tests/", "INFO")
        responses_path = os.path.join(PROJECT_ROOT, "tests", "responses.csv")
        all_responses_path = os.path.join(PROJECT_ROOT, "tests", "all_responses.csv")
        os.makedirs(os.path.dirname(responses_path), exist_ok=True)

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

        # Write to responses.csv (overwrite)
        with open(responses_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=ordered_headers, extrasaction='ignore', restval=0)
            writer.writeheader()
            writer.writerow(data)
        debug_print(f" Overwrote {responses_path} with new response", "SUCCESS")

        # Append to all_responses.csv
        file_exists = os.path.isfile(all_responses_path)
        with open(all_responses_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=ordered_headers, extrasaction='ignore', restval=0)
            if not file_exists or os.stat(all_responses_path).st_size == 0:
                writer.writeheader()
            writer.writerow(data)
        debug_print(f" Appended to {all_responses_path}", "SUCCESS")

        # Step 2: Create temporary CSV
        debug_print(" Step 2: Creating temporary CSV", "INFO")
        df = pd.DataFrame([data])
        temp_csv = tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False, newline='', encoding='utf-8'
        )
        temp_csv_path = temp_csv.name
        df.to_csv(temp_csv_path, index=False)
        temp_csv.close()
        debug_print(f" CSV created: {temp_csv_path}", "SUCCESS")

        # Step 3: Set up archive and current report directories
        debug_print(" Step 3: Setting up output directories", "INFO")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Archive: reports/report_{timestamp}/
        reports_dir = os.path.join(PROJECT_ROOT, "reports", f"report_{timestamp}")
        os.makedirs(reports_dir, exist_ok=True)
        debug_print(f" Archive dir: {reports_dir}", "SUCCESS")

        # Current report: individual_eval/current_report/ (overwrites)
        current_report_dir = os.path.join(PROJECT_ROOT, "individual_eval", "current_report")
        if os.path.exists(current_report_dir):
            shutil.rmtree(current_report_dir)
        os.makedirs(current_report_dir, exist_ok=True)
        debug_print(f" Current report dir: {current_report_dir}", "SUCCESS")

        # Step 4: Run inference pipeline
        debug_print(" Step 4: Running inference pipeline", "INFO")
        with open(temp_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                raise ValueError("CSV is empty")
            raw_payload = rows[0]

        # Sanitize payload
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
        debug_print(f" Payload sanitized with {len(sanitized_payload)} fields", "SUCCESS")
        try:
            from models.profile_aggregator import generate_full_profile

            debug_print(" Running profile generation...", "INFO")
            compiled_profile = generate_full_profile(sanitized_payload)
            debug_print(" Profile generated successfully", "SUCCESS")
        except Exception as e:
            debug_print(f" Inference error: {str(e)}", "ERROR")
            import traceback
            debug_print(traceback.format_exc(), "ERROR")
            raise

        # Step 5: Save JSON and CSV to both archive and current_report
        debug_print(" Step 5: Saving results", "INFO")

        # Custom JSON encoder
        class NumpyEncoder(json.JSONEncoder):
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

        json_filename = f"compiled_profile_{timestamp}.json"
        csv_filename = "responses.csv"

        # Save to archive (reports/report_{timestamp}/)
        json_path_archive = os.path.join(reports_dir, json_filename)
        csv_path_archive = os.path.join(reports_dir, csv_filename)
        with open(json_path_archive, 'w', encoding='utf-8') as f:
            json.dump(compiled_profile, f, indent=2, cls=NumpyEncoder)
        shutil.copy(temp_csv_path, csv_path_archive)
        debug_print(f" Archived to: {reports_dir}", "SUCCESS")

        # Save to current_report (overwrite)
        json_path_current = os.path.join(current_report_dir, json_filename)
        csv_path_current = os.path.join(current_report_dir, csv_filename)
        shutil.copy(json_path_archive, json_path_current)
        shutil.copy(csv_path_archive, csv_path_current)
        debug_print(f" Updated current_report: {current_report_dir}", "SUCCESS")

        # Step 6: Clean up temporary CSV
        debug_print(" Step 6: Cleaning up", "INFO")
        os.unlink(temp_csv_path)
        debug_print(" Temporary files cleaned", "SUCCESS")

        # Step 7: Prepare response
        debug_print(" Step 7: Preparing response", "INFO")
        summary = {
            'domains_processed': len(compiled_profile.get('domain_scores', {})),
            'profile_version': compiled_profile.get('version', 'unknown'),
            'timestamp': timestamp
        }

        response_data = {
            'status': 'success',
            'message': 'Assessment completed successfully ',
            'timestamp': timestamp,
            'archive_dir': reports_dir,
            'current_report_dir': current_report_dir,
            'summary': summary,
            'profile_preview': {k: v for k, v in list(compiled_profile.items())[:5]}
        }

        debug_print(" Response prepared successfully", "SUCCESS")
        debug_print("=" * 60, "END")
        return jsonify(response_data), 200

    except Exception as e:
        debug_print(f" Error in assessment: {str(e)}", "ERROR")
        import traceback
        debug_print(traceback.format_exc(), "ERROR")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc() if DEBUG else None
        }), 500
    
@app.route('/train', methods=['POST', 'OPTIONS'])
def train_models():
    """Trigger model training"""
    if os.environ.get("ALLOW_ADMIN_ENDPOINTS", "False").lower() != "true":
        return jsonify({'error': 'Admin endpoints disabled'}), 403
    
    debug_print(" Training endpoint called", "INFO")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, 'main.py', '--train'],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )
        
        return jsonify({
            'status': 'success' if result.returncode == 0 else 'error',
            'output': result.stdout,
            'error': result.stderr
        }), 200 if result.returncode == 0 else 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/flush', methods=['POST', 'OPTIONS'])
def flush_data():
    """Flush training data, reports, or eval directories"""
    if os.environ.get("ALLOW_ADMIN_ENDPOINTS", "False").lower() != "true":
        return jsonify({'error': 'Admin endpoints disabled'}), 403
    
    debug_print(" Flush endpoint called", "INFO")
    try:
        target = request.json.get('target', 'all')
        
        if target in ['train', 'all']:
            train_dir = os.path.join(PROJECT_ROOT, "models", "saved_states")
            if os.path.exists(train_dir):
                for item in os.listdir(train_dir):
                    if item != ".gitkeep":
                        item_path = os.path.join(train_dir, item)
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        else:
                            os.remove(item_path)
                debug_print(" Flushed training data", "SUCCESS")
        
        if target in ['reports', 'all']:
            reports_dir = os.path.join(PROJECT_ROOT, "reports")
            if os.path.exists(reports_dir):
                for item in os.listdir(reports_dir):
                    if item != ".gitkeep":
                        item_path = os.path.join(reports_dir, item)
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        else:
                            os.remove(item_path)
                debug_print(" Flushed reports", "SUCCESS")
        
        if target in ['eval', 'all']:
            eval_dir = os.path.join(PROJECT_ROOT, "individual_eval")
            if os.path.exists(eval_dir):
                for item in os.listdir(eval_dir):
                    if item != ".gitkeep":
                        item_path = os.path.join(eval_dir, item)
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                        else:
                            os.remove(item_path)
                debug_print(" Flushed eval directory", "SUCCESS")
        
        return jsonify({
            'status': 'success',
            'message': f'Flushed {target} successfully'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/latest-report', methods=['GET'])
def latest_report():
    debug_print(" Fetching latest report", "INFO")
    current_report_dir = os.path.join(PROJECT_ROOT, "individual_eval", "current_report")
    if not os.path.exists(current_report_dir):
        return jsonify({'error': 'No report found'}), 404
    json_files = [f for f in os.listdir(current_report_dir) if f.endswith('.json')]
    if not json_files:
        return jsonify({'error': 'No report JSON found'}), 404
    json_files.sort(reverse=True)
    latest_json = json_files[0]
    with open(os.path.join(current_report_dir, latest_json), 'r', encoding='utf-8') as f:
        data = json.load(f)
    return jsonify(data), 200

if __name__ == "__main__":
    print("=" * 80)
    print(" MINDSIGHT REST API SERVER")
    print("=" * 80)
    print(f" Server: http://localhost:5000")
    print(f" Debug Mode: {DEBUG}")
    print(f" Project Root: {PROJECT_ROOT}")
    print("\n Available Endpoints:")
    print("   GET  /test     - Simple test endpoint")
    print("   GET  /health   - Health check with model status")
    print("   GET  /latest-report - Fetch the latest report")
    print("   POST /assess   - Submit questionnaire data")
    print("   POST /train    - Trigger model retraining")
    print("   POST /flush    - Flush data (train/reports/eval/all)")
    print("=" * 80)
    print(" Server is ready! ")
    print("=" * 80)
    
    app.run(debug=DEBUG, port=5000, host='0.0.0.0')