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

# Enable CORS for all routes and all origins
CORS(app)  # This allows all origins by default

# Or if you want more specific configuration:
# CORS(app, origins=['http://localhost:5173', 'http://127.0.0.1:5173'])

DEBUG = True

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
    debug_print("🧪 Test endpoint called", "INFO")
    return jsonify({
        'message': 'Code running perfectly here! ✅',
        'status': 'Server is working',
        'timestamp': datetime.now().isoformat(),
        'project_root': PROJECT_ROOT
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    debug_print("💚 Health check called", "INFO")
    
    # Check if models are trained
    saved_states_dir = os.path.join(PROJECT_ROOT, "models", "saved_states")
    models_trained = False
    if os.path.exists(saved_states_dir):
        items = os.listdir(saved_states_dir)
        models_trained = len([item for item in items if item != ".gitkeep"]) > 0
    
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
    """Main assessment endpoint"""
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    debug_print("=" * 60, "START")
    debug_print("🚀 Received POST request to /assess", "INFO")
    
    try:
        # Step 1: Get JSON data
        debug_print("📥 Step 1: Extracting JSON data", "INFO")
        data = request.get_json()
        
        if not data:
            debug_print("❌ No data provided", "ERROR")
            return jsonify({'error': 'No data provided'}), 400
        
        debug_print(f"✅ Received {len(data)} fields", "SUCCESS")
        
        # Step 2: Create temporary CSV in the correct format
        debug_print("📝 Step 2: Creating temporary CSV", "INFO")
        
        # Convert to DataFrame (single row)
        df = pd.DataFrame([data])
        
        # Create temp file
        temp_csv = tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.csv', 
            delete=False, 
            newline='',
            encoding='utf-8'
        )
        temp_csv_path = temp_csv.name
        df.to_csv(temp_csv_path, index=False)
        temp_csv.close()
        debug_print(f"✅ CSV created: {temp_csv_path}", "SUCCESS")
        
        # Step 3: Create individual_eval directory structure
        debug_print("📁 Step 3: Setting up evaluation workspace", "INFO")
        individual_eval_dir = os.path.join(PROJECT_ROOT, "individual_eval")
        os.makedirs(individual_eval_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        eval_dir_name = f"eval_{timestamp}"
        eval_dir = os.path.join(individual_eval_dir, eval_dir_name)
        os.makedirs(eval_dir, exist_ok=True)
        debug_print(f"✅ Workspace created: {eval_dir}", "SUCCESS")
        
        # Step 4: Run inference using the orchestrator's logic
        debug_print("⚙️ Step 4: Running inference pipeline", "INFO")
        
        # Read the CSV row
        with open(temp_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                raise ValueError("CSV is empty")
            raw_payload = rows[0]
        
        # Sanitize payload (convert types)
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
        
        debug_print(f"✅ Payload sanitized with {len(sanitized_payload)} fields", "SUCCESS")
        
        # Import and run inference
        try:
            import models.inference_wrappers
            import models.profile_aggregator
            import importlib
            
            # Reload to ensure fresh state
            importlib.reload(models.inference_wrappers)
            importlib.reload(models.profile_aggregator)
            
            from models.profile_aggregator import generate_full_profile
            
            debug_print("🔮 Running profile generation...", "INFO")
            compiled_profile = generate_full_profile(sanitized_payload)
            debug_print("✅ Profile generated successfully", "SUCCESS")
            
        except Exception as e:
            debug_print(f"❌ Inference error: {str(e)}", "ERROR")
            import traceback
            debug_print(traceback.format_exc(), "ERROR")
            raise
        
        # Step 5: Save results
        debug_print("💾 Step 5: Saving results", "INFO")
        
        # Save JSON
        output_json_path = os.path.join(eval_dir, f"compiled_profile_{timestamp}.json")
        
        # Custom JSON encoder for numpy types
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
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(compiled_profile, f, indent=2, cls=NumpyEncoder)
        debug_print(f"✅ JSON saved: {output_json_path}", "SUCCESS")
        
        # Copy CSV to eval directory
        shutil.copy(temp_csv_path, os.path.join(eval_dir, "responses.csv"))
        debug_print("✅ CSV copied to eval directory", "SUCCESS")
        
        # Step 6: Create archive in reports
        debug_print("📦 Step 6: Archiving to reports", "INFO")
        reports_dir = os.path.join(PROJECT_ROOT, "reports", f"report_{timestamp}")
        os.makedirs(reports_dir, exist_ok=True)
        
        shutil.copy(output_json_path, os.path.join(reports_dir, f"compiled_profile_{timestamp}.json"))
        shutil.copy(temp_csv_path, os.path.join(reports_dir, "responses.csv"))
        debug_print(f"✅ Archived to: {reports_dir}", "SUCCESS")
        
        # Step 7: Create current_report
        debug_print("📂 Step 7: Creating current_report", "INFO")
        current_report_dir = os.path.join(individual_eval_dir, "current_report")
        if os.path.exists(current_report_dir):
            shutil.rmtree(current_report_dir)
        os.makedirs(current_report_dir, exist_ok=True)
        
        shutil.copy(output_json_path, os.path.join(current_report_dir, f"compiled_profile_{timestamp}.json"))
        shutil.copy(temp_csv_path, os.path.join(current_report_dir, "responses.csv"))
        debug_print("✅ current_report created", "SUCCESS")
        
        # Step 8: Cleanup
        debug_print("🧹 Step 8: Cleaning up", "INFO")
        os.unlink(temp_csv_path)
        debug_print("✅ Temporary files cleaned", "SUCCESS")
        
        # Step 9: Prepare response
        debug_print("📤 Step 9: Preparing response", "INFO")
        
        # Extract summary from profile
        summary = {
            'domains_processed': len(compiled_profile.get('domains', {})),
            'profile_version': compiled_profile.get('version', 'unknown'),
            'timestamp': timestamp
        }
        
        response_data = {
            'status': 'success',
            'message': 'Assessment completed successfully ✅',
            'timestamp': timestamp,
            'output_dir': eval_dir,
            'archive_dir': reports_dir,
            'summary': summary,
            'profile_preview': {k: v for k, v in list(compiled_profile.items())[:5]}
        }
        
        debug_print("✅ Response prepared successfully", "SUCCESS")
        debug_print("=" * 60, "END")
        return jsonify(response_data), 200
        
    except Exception as e:
        debug_print(f"❌ Error in assessment: {str(e)}", "ERROR")
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
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    debug_print("🛠️ Training endpoint called", "INFO")
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
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    debug_print("🧹 Flush endpoint called", "INFO")
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
                debug_print("✅ Flushed training data", "SUCCESS")
        
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
                debug_print("✅ Flushed reports", "SUCCESS")
        
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
                debug_print("✅ Flushed eval directory", "SUCCESS")
        
        return jsonify({
            'status': 'success',
            'message': f'Flushed {target} successfully'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 MINDSIGHT REST API SERVER")
    print("=" * 80)
    print(f"📍 Server: http://localhost:5000")
    print(f"🔧 Debug Mode: {DEBUG}")
    print(f"📁 Project Root: {PROJECT_ROOT}")
    print("\n📋 Available Endpoints:")
    print("   GET  /test     - Simple test endpoint")
    print("   GET  /health   - Health check with model status")
    print("   POST /assess   - Submit questionnaire data")
    print("   POST /train    - Trigger model retraining")
    print("   POST /flush    - Flush data (train/reports/eval/all)")
    print("=" * 80)
    print("✅ Server is ready! 🎯")
    print("=" * 80)
    
    app.run(debug=True, port=5000, host='0.0.0.0')