import os
import sys
import unittest
import numpy as np
import pandas as pd

# Append project root directory to the python path to prevent import resolution failures
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.inference_wrappers import PersonalityVAEInference, SelfEsteemInferenceWrapper

class TestModelInferencePipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """
        Executes once before test execution. Generates a robust, realistic 
        dummy responses.csv file containing multi-domain user response strings.
        """
        cls.dummy_csv_path = os.path.join(PROJECT_ROOT, "tests", "dummy_responses.csv")
        
        # 1. Build Domain 1 item pools (50 items scaled 1.0 to 5.0)
        prefixes = ['EXT', 'EST', 'AGR', 'CSN', 'OPN']
        d1_cols = [f"{prefix}{i}" for prefix in prefixes for i in range(1, 11)]
        
        # 2. Build Domain 2 item pools (Demographics + 10 items scaled 0 to 4)
        d2_cols = ['gender', 'age'] + [f"Q{i}" for i in range(1, 11)]
        
        # Assemble composite dataframe schema
        all_cols = d1_cols + d2_cols
        
        # Construct 3 mock users representing diverse psychometric bounds
        mock_data = []
        
        # User 0: Normal median values, valid demographics
        user_normal = [3.0] * 50 + [1, 28.0] + [2] * 10
        # User 1: Extreme high values, handles age outlier condition (90210.0)
        user_high = [5.0] * 50 + [2, 90210.0] + [4] * 10
        # User 2: Lower floor responses, handles zero age outlier
        user_low = [1.0] * 50 + [0, 0.0] + [0] * 10
        
        mock_data.append(user_normal)
        mock_data.append(user_high)
        mock_data.append(user_low)
        
        df_dummy = pd.DataFrame(mock_data, columns=all_cols)
        df_dummy.to_csv(cls.dummy_csv_path, index=False)
        print(f"\n[SETUP] Dynamically generated integration test vector matrix at: {cls.dummy_csv_path}")

    @classmethod
    def tearDownClass(cls):
        """Clean up generated mock artifacts after testing completes."""
        if os.path.exists(cls.dummy_csv_path):
            os.remove(cls.dummy_csv_path)
            print(f"[TEARDOWN] Purged localized testing artifact: {cls.dummy_csv_path}")

    def test_domain1_personality_vae(self):
        """Verifies Domain 1 VAE safely extracts exactly 5 continuous orthogonal factors."""
        print("\n[RUNNING] Testing Domain 1: Personality Beta-VAE Latent Extraction...")
        
        try:
            inference_engine = PersonalityVAEInference()
        except FileNotFoundError as e:
            self.fail(f"Initialization Failed: Are the model weights missing? Error details: {e}")
            
        df = pd.read_csv(self.dummy_csv_path)
        prefixes = ['EXT', 'EST', 'AGR', 'CSN', 'OPN']
        d1_cols = [f"{prefix}{i}" for prefix in prefixes for i in range(1, 11)]
        
        for idx, row in df.iterrows():
            vector = row[d1_cols].values.astype(float)
            latent_traits = inference_engine.extract_latent_traits(vector)
            
            # Assert latent factor mapping array matches dimensions exactly
            self.assertEqual(latent_traits.shape, (5,), f"Latent vector size mismatched for user row index {idx}")
            self.assertTrue(np.isinstance(latent_traits, np.ndarray), "Output structure type must be a numpy vector array.")
            self.assertFalse(np.isnan(latent_traits).any(), "Model inference pipeline returned invalid NaN fields.")

    def test_domain2_self_esteem_catboost(self):
        """Verifies Domain 2 CatBoost handles age outliers and outputs scores within valid bounds."""
        print("\n[RUNNING] Testing Domain 2: Self-Esteem CatBoost Prediction Index...")
        
        try:
            inference_engine = SelfEsteemInferenceWrapper()
        except FileNotFoundError as e:
            self.fail(f"Initialization Failed: Missing joblib files. Error details: {e}")
            
        df = pd.read_csv(self.dummy_csv_path)
        item_cols = [f"Q{i}" for i in range(1, 11)]
        
        for idx, row in df.iterrows():
            gender = row['gender']
            age = row['age']
            q_responses = row[item_cols].values.astype(int)
            
            predicted_score = inference_engine.predict_score(gender, age, q_responses)
            
            # Assert data types and scoring ranges
            self.assertIsInstance(predicted_score, float, "Predicted score metric should return as an isolated float value.")
            # Rosenberg scores can mathematically range between 0 and 40 (10 items * 4 points max)
            self.assertTrue(0.0 <= predicted_score <= 40.0, f"Predicted score {predicted_score} out of psychometric bounds.")

if __name__ == "__main__":
    unittest.main()