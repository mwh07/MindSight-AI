import os
import sys
import unittest
import pandas as pd
import json

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Core Pipeline Inference Framework Imports
from models.inference_wrappers import (
    PersonalityVAEInference, 
    SelfEsteemInferenceWrapper, 
    MoodSleepInferenceWrapper,
    DigitalLonelinessInferenceWrapper,
    TechBurnoutInferenceWrapper,
    ClinicalBaselinesInferenceWrapper
)

# Core Holistic Representation Engine
from models.profile_aggregator import MindsightProfileAggregator

# Executive PDF Generation Engine
from models.pdf_generator import MindsightPDFReporter

class UserAssessmentSuite(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.responses_path = os.path.join(PROJECT_ROOT, "tests", "responses.csv")
        cls.individual_eval_dir = os.path.join(PROJECT_ROOT, "individual_eval")
        
        # Initialize all 6 Domain Inference Engines
        cls.vae = PersonalityVAEInference()
        cls.rses = SelfEsteemInferenceWrapper()
        cls.mood_sleep = MoodSleepInferenceWrapper()
        cls.digital_lone = DigitalLonelinessInferenceWrapper()
        cls.burnout_engine = TechBurnoutInferenceWrapper()
        cls.clinical_engine = ClinicalBaselinesInferenceWrapper()

    def test_generate_individual_assessments(self):
        """Processes responses.csv and isolates individual JSON and personalized multi-domain PDF assets."""
        if not os.path.exists(self.responses_path):
            self.skipTest("responses.csv targets missing from execution workspace.")

        df = pd.read_csv(self.responses_path)
        
        aggregator = MindsightProfileAggregator(results_dir=self.individual_eval_dir)
        pdf_engine = MindsightPDFReporter(output_dir=self.individual_eval_dir)
        
        # Structural Layout Map Configurations
        d1_cols = [f"{p}{i}" for p in ['EXT', 'EST', 'AGR', 'CSN', 'OPN'] for i in range(1, 11)]
        d2_q_cols = [f"Q{i}" for i in range(1, 11)]
        d3_phq_cols = [f"DPQ0{i}0" for i in range(1, 10)]
        d4_cols = [f"DIG_{i}" for i in range(1, 18)]
        d6_cols = [f"CLIN_{i}" for i in range(1, 9)]

        print(f"\n[LAUNCH] Starting automated psychometric compilation for {len(df)} user records...")

        for idx, row in df.iterrows():
            user_target_id = f"person_{idx}"
            
            # --- DOMAIN 1: PERSONALITY VAE ---
            latent_traits = self.vae.extract_latent_traits(row[d1_cols].values.astype(float))
            
            # --- DOMAIN 2: SELF ESTEEM CATBOOST ---
            rses_score = self.rses.predict_score(row['gender'], row['age'], row[d2_q_cols].values.astype(int))
            
            # --- DOMAIN 3: MOOD & SLEEP LIGHTGBM ---
            phq9_vals = row[d3_phq_cols].values.astype(int) if d3_phq_cols[0] in df.columns else [1]*9
            weekday_s = float(row['SLD012']) if 'SLD012' in df.columns else 7.5
            weekend_s = float(row['SLD013']) if 'SLD013' in df.columns else 8.5
            clock_stamps = [row['SLQ300'], row['SLQ310'], row['SLQ320'], row['SLQ330']] if 'SLQ300' in df.columns else ['22:30', '06:30', '23:30', '07:30']
            mood_results = self.mood_sleep.predict_metrics(phq9_vals, weekday_s, weekend_s, clock_stamps)
            
            # --- DOMAIN 4: DIGITAL LONELINESS NEURAL NET ---
            d4_vector = row[d4_cols].values.astype(float) if d4_cols[0] in df.columns else [3.0]*18
            digital_results = self.digital_lone.predict_profile(d4_vector)
            
            # --- DOMAIN 5: WORKPLACE STRESS & BURNOUT XGBOOST ---
            gender_str = "Male" if row['gender'] == 1 else "Female" if row['gender'] == 2 else "Non-binary"
            work_hours = float(row['work_hours_per_week']) if 'work_hours_per_week' in df.columns else 44.0
            meetings = float(row['meetings_per_day']) if 'meetings_per_day' in df.columns else 3.0
            wlb = float(row['work_life_balance_score']) if 'work_life_balance_score' in df.columns else 5.0
            job_sat = float(row['job_satisfaction_score']) if 'job_satisfaction_score' in df.columns else 6.0
            social_sup = float(row['social_support_score']) if 'social_support_score' in df.columns else 6.0
            deadline = float(row['deadline_pressure_score']) if 'deadline_pressure_score' in df.columns else 4.0
            autonomy = float(row['autonomy_score']) if 'autonomy_score' in df.columns else 6.0
            burnout_results = self.burnout_engine.predict_occupational_strain(
                gender_str, row['age'], work_hours, meetings, wlb, job_sat, social_sup, deadline, autonomy
            )
            
            # --- DOMAIN 6: CLINICAL BASELINES DETECTOR ---
            d6_vector = row[d6_cols].values.astype(int) if d6_cols[0] in df.columns else [0, 0, 1, 0, 0, 0, 0, 0]
            clinical_results = self.clinical_engine.calculate_clinical_baselines(
                row['age'], mood_results['predicted_phq9_severity'], 
                mood_results['predicted_phq9_severity'] * 0.85, weekday_s, d6_vector
            )
            
            isolated_report = {
                "user_id": user_target_id,
                "demographics": {"gender_code": int(row['gender']), "age": float(row['age'])},
                "domain_1_personality": {"latent_factors": latent_traits.tolist()},
                "domain_2_self_esteem": {"predicted_rses_index": rses_score},
                "domain_3_mood_and_sleep": mood_results,
                "domain_4_digital_and_loneliness": digital_results,
                "domain_5_workplace_burnout_and_stress": burnout_results,
                "domain_6_clinical_baselines_and_ocd": clinical_results,
                "status": "COMPLETED_SUCCESSFULLY"
            }
            
            # Run compiler
            holistic_profile = aggregator.compile_holistic_profile(
                user_id=user_target_id, d1_traits=latent_traits, d2_score=rses_score,
                d3_metrics=mood_results, d4_profile=digital_results, d5_strain=burnout_results,
                d6_baselines=clinical_results
            )
            
            # --- DIRECT NESTED ROUTING AND FILE HOUSEKEEPING ---
            subfolder_name = f"Person_{idx}_eval"
            person_destination_dir = os.path.join(self.individual_eval_dir, subfolder_name)
            os.makedirs(person_destination_dir, exist_ok=True)
            
            # 1. Export Isolated JSON inside the subfolder
            json_file_path = os.path.join(person_destination_dir, f"assessment_{user_target_id}.json")
            with open(json_file_path, 'w') as f:
                json.dump(isolated_report, f, indent=4)
                
            # 2. Print live dashboard panel to stdout console
            aggregator.display_terminal_dashboard(holistic_profile)
            
            # 3. Generate dynamic PDF dossier inside the subfolder
            pdf_engine.build_pdf_report(holistic_profile, custom_dir=person_destination_dir)
            
            # 4. HOUSEKEEPING SWEEP: Instantly delete the unwanted root-level file dropped by the aggregator
            unwanted_summary_file = os.path.join(self.individual_eval_dir, f"profile_summary_{user_target_id}.json")
            if os.path.exists(unwanted_summary_file):
                os.remove(unwanted_summary_file)

        print(f"\n[SUCCESS] Pipeline analysis completed.")
        print(f"[INFO] Diagnostics exported to nested structures inside: {self.individual_eval_dir}/")

if __name__ == "__main__":
    unittest.main()