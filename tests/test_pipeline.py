#!/usr/bin/env python3
import os
import traceback
import unittest
import pandas as pd

# Core framework module imports
from models.profile_aggregator import generate_full_profile
from models.inference_wrappers import get_domain_prediction

# Import your real ReportLab compiler function from models/pdf_generator.py
try:
    from models.pdf_generator import compile_pdf_report
except ImportError:
    raise ImportError("❌ Linkage Broken: Could not find 'compile_pdf_report' inside 'models/pdf_generator.py'.")

class TestMindsightPipeline(unittest.TestCase):

    def setUp(self):
        """
        Establishes target directories and isolates your source spreadsheet asset.
        """
        self.csv_path = "tests/responses.csv"
        self.output_dir = "individual_eval"
        
        # Ensure the evaluation target workspace folder exists physically
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Guard clause verifying the source file exists before executing pipeline
        if not os.path.exists(self.csv_path):
            self.fail(f"❌ Execution Blocked: Source data spreadsheet missing at '{self.csv_path}'.")

    def test_production_csv_batch_evaluation_and_pdf_generation(self):
        """
        Iterates directly through tests/responses.csv, evaluates each row 
        against all ML domains, and compiles genuine ReportLab PDFs into individual_eval/
        """
        print(f"\n📥 Loading evaluation datasets from: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        print(f"📊 Found {len(df)} user response profiles to evaluate.")
        
        self.assertGreater(len(df), 0, "❌ The responses.csv file does not contain any rows.")

        # Loop through every row in the spreadsheet
        for index, row in df.iterrows():
            # Convert row to native dictionary, discarding empty/NaN features
            user_responses = row.dropna().to_dict()
            
            # Identify name token or fallback to row counter sequence
            user_id = user_responses.get("user_id", user_responses.get("name", f"user_{index + 1}"))
            print(f"\n🔄 [Processing Profile {index + 1}/{len(df)}] Engine Execution ID: {user_id}")
            
            try:
                # 1. Execute all 6 Machine Learning Inference pipelines concurrently
                full_profile = generate_full_profile(user_responses)
                self.assertIsInstance(full_profile, dict, "Aggregator matrix must output a master dictionary object.")
                
                # 2. Build explicit file path tokens targeting the individual_eval/ workspace
                safe_filename = f"report_{user_id}.pdf".replace(" ", "_").lower()
                output_pdf_path = os.path.join(self.output_dir, safe_filename)
                
                # 3. Fire the processed structural matrix data payload straight into your ReportLab compiler
                compile_pdf_report(full_profile, output_pdf_path)
                
                # 4. Enforce check verifying that a physical binary asset was successfully compiled to disk
                self.assertTrue(
                    os.path.exists(output_pdf_path), 
                    f"❌ PDF verification failure: Asset was not written to drive for {user_id}"
                )
                print(f"✅ Success -> Report saved permanently to: {output_pdf_path}")
                
            except Exception as e:
                self.fail(
                    f"❌ Pipeline crashed on profile '{user_id}' at row sequence index {index}.\n"
                    f"Details: {str(e)}\n{traceback.format_exc()}"
                )

if __name__ == "__main__":
    unittest.main()