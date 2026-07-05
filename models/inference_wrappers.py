#!/usr/bin/env python3
"""
MINDSIGHT Live Runtime Inference Engine (v3.9 - Production Calibrated)
Authoritative inference manager providing strict contract alignment across
all 6 evaluation domains. Fixes structural scaling inversions, local attributions,
and drop-out vector arrays.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import shap

# Single source of truth for every feature's human-readable display name.
try:
    from models.feature_mappings import FEATURE_TRANSLATION_MAP
except ImportError:
    try:
        from feature_mappings import FEATURE_TRANSLATION_MAP
    except ImportError:
        FEATURE_TRANSLATION_MAP = {}

# Reverse-coded items in Domain 1 (as per IMP-70 questionnaire)
REVERSE_ITEMS = ["EXT2", "EST2", "CSN2"]

def get_display_name(feature_key, fallback=None):
    """
    Single helper every domain below uses to resolve a feature's
    human-readable label. Looks up FEATURE_TRANSLATION_MAP first; if a key
    is genuinely absent from that map, falls back to a title-cased version.
    """
    if feature_key in FEATURE_TRANSLATION_MAP:
        return FEATURE_TRANSLATION_MAP[feature_key]
    if fallback is not None:
        return fallback
    return feature_key.replace("_", " ").title()

CLINICAL_SEVERITY_MAP = {
    0: "Baseline Healthy Profile",
    1: "Mild Symptomatic Profile",
    2: "Moderate Distress Phenotype",
    3: "Severe Clinical Screening Indication"
}

def parse_time_to_hours(time_str):
    """Safely converts HH:MM string representations to fractional numeric hours."""
    try:
        if pd.isna(time_str):
            return 12.0
        time_str = str(time_str).strip()
        parts = time_str.split(':')
        if len(parts) != 2:
            return 12.0
        return int(parts[0]) + (int(parts[1]) / 60.0)
    except Exception:
        return 12.0

# =====================================================================
# DOMAIN 1: BIG FIVE PERSONALITY TRAIT VECTORS (AUTHENTIC IRT SCORING)
# =====================================================================
def evaluate_domain1_personality(raw_payload):
    """
    Computes all 5 Big Five personality vector dimensions using true GRM-IRT parameter estimation.
    Falls back to a normalized mean-based estimate ONLY if the fitted parameter file is genuinely
    missing or unreadable -- per-trait scoring failures are now isolated.
    """
    model_path = os.path.join(os.path.dirname(__file__), "saved_states", "domain1_grm_parameters.pkl")

    # Static fallback traits layer if the pickle file is not found.
    traits_fallback = {
        "extraversion": ["EXT1", "EXT2", "EXT3"],
        "emotional_stability": ["EST1", "EST2", "EST3"],
        "agreeableness": ["AGR1", "AGR2", "AGR3"],
        "conscientiousness": ["CSN1", "CSN2", "CSN3"],
        "openness": ["OPN1", "OPN2", "OPN3"]
    }

    placement = {}

    trait_mapping = {
        "EXT": "extraversion",
        "EST": "emotional_stability",
        "AGR": "agreeableness",
        "CSN": "conscientiousness",
        "OPN": "openness"
    }

    if os.path.exists(model_path):
        grm_registry = None
        try:
            with open(model_path, "rb") as f:
                grm_registry = pickle.load(f)
        except Exception:
            grm_registry = None

        if grm_registry:
            # Establish the theta evaluation grid matching authentic IRT scales (-4.0 to +4.0)
            grid = np.linspace(-4.0, 4.0, 81)

            for trait_name in grm_registry.keys():
                # Each trait is scored independently
                try:
                    item_params = grm_registry[trait_name].get("items", {})

                    # Initialize total log-likelihood array for the candidate theta grid
                    log_lik = np.zeros_like(grid)

                    for item_id, params in item_params.items():
                        try:
                            raw_val = float(raw_payload.get(item_id, 3.0))
                            # Reverse-code if needed
                            if item_id in REVERSE_ITEMS:
                                raw_val = 6 - raw_val   # 1->5, 2->4, 3->3, 4->2, 5->1
                            # Constrain Likert responses to standard 1 to 5 index range
                            resp = int(np.clip(round(raw_val), 1, 5))
                        except (ValueError, TypeError):
                            resp = 3

                        # Extract item discrimination (a) and threshold parameters (b)
                        a = float(params.get("a", 1.0))
                        b = np.array(params.get("b", [-1.5, -0.5, 0.5, 1.5]))

                        # Compute category boundary probabilities across the entire theta grid
                        p_star = 1.0 / (1.0 + np.exp(-a * (grid[:, None] - b[None, :])))

                        # Anchor boundary matrices: P*(0) = 1.0, P*(K) = 0.0
                        ones = np.ones((len(grid), 1))
                        zeros = np.zeros((len(grid), 1))
                        p_star_full = np.hstack([ones, p_star, zeros])

                        # Extract the true specific option probability mass: P(k) = P*(k-1) - P*(k)
                        p_cat = p_star_full[:, resp - 1] - p_star_full[:, resp]
                        log_lik += np.log(np.clip(p_cat, 1e-9, 1.0))

                    # Apply Bayesian Standard Normal Prior (MAP estimation) to regularize extreme responses
                    log_lik += -0.5 * (grid ** 2)
                    best_idx = np.argmax(log_lik)
                    mapped_name = trait_mapping.get(trait_name, trait_name)
                    placement[mapped_name] = round(float(grid[best_idx]), 3)
                except Exception:
                    # Only this single trait falls through to its mean-based fallback below;
                    continue

    # Fill in ONLY the traits that real GRM scoring did not produce above
    for trait_name, keys in traits_fallback.items():
        if trait_name in placement:
            continue
        vals = []
        for k in keys:
            try:
                raw_val = float(raw_payload.get(k, 3.0))
                if k in REVERSE_ITEMS:
                    raw_val = 6 - raw_val
                vals.append(raw_val)
            except (ValueError, TypeError):
                vals.append(3.0)
        mean_score = np.mean(vals)
        placement[trait_name] = round(float((mean_score - 3.0) * 1.33), 3)

    contributors = []
    for trait_name, score in placement.items():
        contributors.append({
            "feature": trait_name.title(),
            "display_name": f"{trait_name.title()} Vector",
            "contribution": round(float(abs(score)), 4),
            "direction": "+" if score >= 0 else "-"
        })

    return {
        "domain": "domain_1_personality",
        "placement": placement,
        "top_contributors": sorted(contributors, key=lambda x: x["contribution"], reverse=True)[:3]
    }

# =====================================================================
# DOMAIN 2: ROSENBERG SELF-ESTEEM MATRIX (REALIGNED SCHEMA)
# =====================================================================
def evaluate_domain2_self_esteem(raw_payload):
    """
    Processes the Rosenberg Self-Esteem (RSE) Scale across a 0-40 maximum score boundary.
    Converts raw 1-5 Likert to 0-4 scale, applies reverse-scoring to items 3,5,8,9,10.
    Appends normative percentile lookup based on demographic cohort.
    """
    reverse_scoring_items = [3, 5, 8, 9, 10]
    total_score = 0
    item_contributions = []
    
    for i in range(1, 11):
        raw_val = raw_payload.get(f"Q{i}", raw_payload.get(f"RSE{i}", 2.0))
        try:
            val = int(float(raw_val))
            val = val - 1                    # 1→0, 2→1, ... 5→4
            val = max(0, min(4, val))        # clamp
        except (ValueError, TypeError):
            val = 2
            
        if i in reverse_scoring_items:
            processed_val = 4 - val
        else:
            processed_val = val
            
        total_score += processed_val
        
        item_contributions.append({
            "feature": f"Q{i}",
            "display_name": get_display_name(f"Q{i}", fallback=f"Self-Esteem Item {i}"),
            "contribution": round(float(abs(processed_val - 2.0)), 4),
            "direction": "+" if processed_val >= 2.0 else "-"
        })
        
    item_contributions = sorted(item_contributions, key=lambda x: x["contribution"], reverse=True)

    # Calculate empirical normative percentile
    normative_percentile = None
    try:
        model_path = os.path.join(os.path.dirname(__file__), "saved_states", "domain2_self_esteem.pkl")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                percentile_lookup = pickle.load(f)
            
            age = float(raw_payload.get("age", 25))
            if age < 18:
                a_band = "under_18"
            elif age <= 25:
                a_band = "18_25"
            elif age <= 35:
                a_band = "26_35"
            elif age <= 50:
                a_band = "36_50"
            elif age <= 65:
                a_band = "51_65"
            else:
                a_band = "over_65"
            
            raw_gender = str(raw_payload.get("gender", "0")).strip().lower()
            if raw_gender in ["1", "1.0", "f", "female"]:
                g_id = 1
            elif raw_gender in ["2", "2.0", "nb", "non-binary", "other"]:
                g_id = 2
            else:
                g_id = 0
                
            cohort_key = f"g{g_id}_{a_band}"
            
            if cohort_key in percentile_lookup:
                normative_percentile = percentile_lookup[cohort_key].get(str(int(total_score)), 50.0)
    except Exception as e:
        print(f"Domain 2 Percentile Lookup Failed: {e}")

    return {
        "domain": "domain_2_self_esteem",
        "placement": {
            "score": int(total_score),
            "max_possible_score": 40,
            "classification": "High Self-Esteem" if total_score >= 30 else ("Normal" if total_score >= 19 else "Low Self-Esteem"),
            "normative_percentile": normative_percentile
        },
        "top_contributors": item_contributions[:3]
    }

# =====================================================================
# DOMAIN 3: MOOD SEVERITY & SLEEP METRICS (ML + SHAP)
# =====================================================================
def evaluate_domain3_mood_sleep(raw_payload):
    """
    Evaluates clinical PHQ-9 tracking matrices and localized sleep timings.
    Now uses Unsupervised KMeans clustering to map the user to a Clinical Phenotype,
    while returning the standard deterministic PHQ-9 clinical score for API safety.
    """
    # --- Deterministic calculations (always computed, kept for reference) ---
    phq_total = 0
    dpq_scores = {}
    for i in range(1, 10):
        item_key = f"DPQ0{i}0"
        try:
            val = int(float(raw_payload.get(item_key, 0)))
            if val in (7, 9):
                val = 0
            score = max(0, min(3, val))
            phq_total += score
            dpq_scores[item_key] = score
        except (ValueError, TypeError):
            dpq_scores[item_key] = 0

    def parse_time(time_str, default=12.0):
        try:
            if pd.isna(time_str): return default
            time_str = str(time_str).strip()
            parts = time_str.split(':')
            if len(parts) != 2: return default
            return int(parts[0]) + (int(parts[1]) / 60.0)
        except Exception:
            return default

    sleep_onset = parse_time(raw_payload.get("SLQ300", "23:00"))
    sleep_wakeup = parse_time(raw_payload.get("SLQ310", "07:00"))
    duration = sleep_wakeup - sleep_onset
    if duration < 0:
        duration += 24.0

    # --- Deterministic severity ---
    if phq_total >= 20:
        deterministic_severity = "Severe Depression"
    elif phq_total >= 15:
        deterministic_severity = "Moderately Severe Depression"
    elif phq_total >= 10:
        deterministic_severity = "Moderate Depression"
    elif phq_total >= 5:
        deterministic_severity = "Mild Depression"
    else:
        deterministic_severity = "Minimal or Baseline Profile"

    # Deterministic top contributors (top 3 highest scoring DPQ items)
    sorted_dpq = sorted(dpq_scores.items(), key=lambda x: x[1], reverse=True)
    contributors = []
    for item_key, score in sorted_dpq:
        if score > 0:
            contributors.append({
                "feature": item_key,
                "display_name": get_display_name(item_key, fallback=item_key),
                "contribution": float(score),
                "direction": "+"
            })
    if not contributors:
        contributors = [
            {"feature": "PHQ_Core", "display_name": "Symptom Burden Summation", "contribution": float(phq_total), "direction": "+"}
        ]
        
    placement = {
        "phq9_sum": int(phq_total),
        "severity_label": deterministic_severity,
        "calculated_sleep_duration_hours": round(float(duration), 2)
    }

    # --- Load KMeans model and scaler ---
    model_path = os.path.join(os.path.dirname(__file__), "saved_states", "domain3_mood_sleep.pkl")
    meta_path = os.path.join(os.path.dirname(__file__), "saved_states", "domain3_mood_sleep_metadata.json")

    try:
        if os.path.exists(model_path) and os.path.exists(meta_path):
            with open(model_path, "rb") as f:
                payload = pickle.load(f)
            
            kmeans = payload.get("kmeans_model")
            scaler = payload.get("scaler")
            cluster_profiles = payload.get("cluster_profiles", {})
            
            if kmeans and scaler:
                X_input = pd.DataFrame([{"phq9_sum": phq_total, "calculated_sleep_duration": duration}])
                X_scaled = scaler.transform(X_input)
                cluster_idx = kmeans.predict(X_scaled)[0]
                
                phenotype_name = cluster_profiles.get(str(cluster_idx), f"Phenotype {cluster_idx}")
                placement["clinical_phenotype"] = phenotype_name
    except Exception as e:
        print(f"Clustering Error in Domain 3: {e}")
        pass

    return {
        "domain": "domain_3_mood_sleep",
        "placement": placement,
        "top_contributors": contributors[:3]
    }

# =====================================================================
# DOMAIN 4: ATTACHMENT & LONELINESS FOREST PIPELINE (FIXED & ISOLATED)
# =====================================================================
def evaluate_domain4_multitask(raw_payload):
    """
    Evaluates attachment, loneliness, and relationship dynamics via a single unified model.
    Scores for Internet Addiction and Loneliness are calculated deterministically to satisfy
    the clinical contract. The ML model predicts Depression Risk, and its SHAP values
    are used to isolate the cross-impact drivers of distress.
    """
    model_path = os.path.join(os.path.dirname(__file__), "saved_states", "domain4_digital_social.pkl")
    meta_path = os.path.join(os.path.dirname(__file__), "saved_states", "domain4_digital_social_metadata.json")

    def get_static_scores():
        iat_vals = []
        for i in range(1, 11):
            try:
                iat_vals.append(float(raw_payload.get(f"IAT{i}", 3.0)))
            except (ValueError, TypeError):
                iat_vals.append(3.0)
                
        lone_vals = []
        for i in range(1, 7):
            try:
                lone_vals.append(float(raw_payload.get(f"loneliness{i}", 3.0)))
            except (ValueError, TypeError):
                lone_vals.append(3.0)
                
        pred_iat = float(np.sum(iat_vals))
        pred_lone = float(np.sum(lone_vals))
        return pred_iat, pred_lone

    pred_iat, pred_lone = get_static_scores()
    base_placement = {
        "predicted_total_internet_addiction": round(pred_iat, 3),
        "predicted_total_loneliness": round(pred_lone, 3),
        "loneliness_score": round(pred_lone * 3.33, 3),
        "classification": "Elevated Distress Profile" if pred_lone > 18 else "Baseline Cohort Profile"
    }

    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        base_placement["data_source"] = "fallback_raw_sum"
        return {
            "domain": "domain_4_digital_and_social",
            "placement": base_placement,
            "top_contributors": []
        }

    try:
        with open(model_path, "rb") as f:
            model_payload = pickle.load(f)
        with open(meta_path, "r") as f:
            metadata = json.load(f)

        if "depression_risk_model" in model_payload:
            unified_model = model_payload["depression_risk_model"]
        else:
            raise ValueError("domain4_digital_social.pkl missing 'depression_risk_model'")

        features = metadata.get("features", [])
        if not features:
            features = list(getattr(unified_model, "feature_names_in_", []))
        
        row_dict = {}
        for f_name in features:
            row_dict[f_name] = float(raw_payload.get(f_name, 2.0 if "gender" not in f_name else 0.0))
        
        df_row = pd.DataFrame([row_dict], columns=features)
        pred_depression = float(unified_model.predict(df_row)[0])

        contributors = []
        try:
            explainer = shap.TreeExplainer(unified_model)
            shap_values = explainer.shap_values(df_row)
            
            if isinstance(shap_values, list):
                row_shap = shap_values[0][0]
            elif len(np.array(shap_values).shape) == 3:
                row_shap = np.array(shap_values)[0, :, 0]
            else:
                row_shap = shap_values[0]
                
            for idx, f_name in enumerate(features):
                if not (f_name.startswith("IAT") or f_name.startswith("loneliness")):
                    continue
                try:
                    val_weight = float(row_shap[idx]) if idx < len(row_shap) else 0.0
                except Exception:
                    val_weight = 0.0
                contributors.append({
                    "feature": f_name,
                    "display_name": get_display_name(f_name),
                    "contribution": round(float(abs(val_weight)), 4),
                    "direction": "+" if val_weight >= 0 else "-"
                })
        except Exception as e:
            print(f"SHAP Error in Domain 4: {e}")
            pass

        base_placement["predicted_depression_risk"] = round(pred_depression, 3)
        base_placement["data_source"] = "unified_ml_cross_impact"
        
        # Override baseline classification using the ML predictions
        if pred_depression >= 15.0:
            base_placement["classification"] = "High Cross-Impact Depression Risk"
        elif pred_iat >= 50.0:
            base_placement["classification"] = "Elevated Digital Dependency"
        elif pred_lone > 18.0:
            base_placement["classification"] = "Elevated Social Isolation"
        else:
            base_placement["classification"] = "Baseline Cohort Profile"

        return {
            "domain": "domain_4_digital_and_social",
            "placement": base_placement,
            "top_contributors": sorted(contributors, key=lambda x: x["contribution"], reverse=True)[:3]
        }
        
    except Exception as e:
        base_placement["data_source"] = "fallback_raw_sum_error"
        return {
            "domain": "domain_4_digital_and_social",
            "placement": base_placement,
            "top_contributors": []
        }

        
# =====================================================================
# DOMAIN 5: OCCUPATIONAL BURNOUT GRADIENT ENGINE
# =====================================================================
def evaluate_domain5_burnout(raw_payload):
    """
    Evaluates occupational burnout using Ordinal and Quantile XGBoost models.
    """
    meta_path = os.path.join(os.path.dirname(__file__), "saved_states", "domain5_burnout_metadata.json")
    
    if not os.path.exists(meta_path):
        return {
            "domain": "domain_5_occupational_burnout",
            "placement": {"burnout_index": 5.0, "burnout_tier_label": "Unavailable", "tier_thresholds": None},
            "top_contributors": []
        }
        
    with open(meta_path, "r") as f:
        metadata = json.load(f)
        
    # Prepare features as per metadata
    # Numeric features
    row_dict = {}
    for col in metadata["raw_work_features"] + ["age"]:
        raw_val = raw_payload.get(col, 3.0 if "score" in col else (30.0 if col == "age" else 0.0))
        try:
            row_dict[col] = float(raw_val)
        except (ValueError, TypeError):
            row_dict[col] = 3.0 if "score" in col else (30.0 if col == "age" else 0.0)

    # Engineered features
    row_dict["stress_x_support"] = row_dict.get("stress_score", 3.0) * row_dict.get("social_support_score", 3.0)
    work_hrs = row_dict.get("work_hours_per_week", 40.0)
    row_dict["hours_over_50"] = max(0.0, work_hrs - 50.0)
    row_dict["meeting_load_ratio"] = (row_dict.get("meetings_per_day", 0.0) / work_hrs) if work_hrs > 0 else 0.0
    
    # Gender dummies
    gender_val = str(raw_payload.get("gender", "")).strip().lower()
    gender_map = {"0": "male", "1": "female", "2": "non-binary", "m": "male", "f": "female", "male": "male", "female": "female"}
    gender_val = gender_map.get(gender_val, gender_val)
    
    for col in metadata["gender_dummy_columns"]:
        if gender_val in col.lower():
            row_dict[col] = 1.0
        else:
            row_dict[col] = 0.0

    # Ensure correct feature order
    input_row = [row_dict.get(f, 0.0) for f in metadata["feature_order"]]
    df_row = pd.DataFrame([input_row], columns=metadata["feature_order"])
    
    # Load point estimate (quantile 0.50) model
    q50_filename = metadata["model_files"]["quantile"].get("q50")
    if not q50_filename:
        return {
            "domain": "domain_5_occupational_burnout",
            "placement": {"burnout_index": 5.0, "burnout_tier_label": "Unavailable", "tier_thresholds": None},
            "top_contributors": []
        }
    
    q50_path = os.path.join(os.path.dirname(__file__), "saved_states", q50_filename)
    if not os.path.exists(q50_path):
        return {
            "domain": "domain_5_occupational_burnout",
            "placement": {"burnout_index": 5.0, "burnout_tier_label": "Unavailable", "tier_thresholds": None},
            "top_contributors": []
        }
        
    bst_median = xgb.Booster()
    bst_median.load_model(q50_path)
    dmat = xgb.DMatrix(df_row)
    pred_score = float(bst_median.predict(dmat)[0])

    # Load ordinal heads to determine tier probabilities
    tier_probs = {}
    for label, filename in metadata["model_files"]["ordinal"].items():
        bst_ord = xgb.Booster()
        bst_ord.load_model(os.path.join(os.path.dirname(__file__), "saved_states", filename))
        tier_probs[label] = float(bst_ord.predict(dmat)[0])

    p_ge_mod = tier_probs.get("ge_Moderate", 0.0)
    p_ge_high = tier_probs.get("ge_High", 0.0)
    p_ge_severe = tier_probs.get("ge_Severe", 0.0)
    
    # Cumulative to class probs
    p_low = 1.0 - p_ge_mod
    p_mod = p_ge_mod - p_ge_high
    p_high = p_ge_high - p_ge_severe
    p_severe = p_ge_severe
    
    # Clamp and normalize
    probs = np.clip(np.array([p_low, p_mod, p_high, p_severe]), 0, None)
    if np.sum(probs) > 0:
        probs = probs / np.sum(probs)
    
    best_tier_idx = np.argmax(probs)
    tier_labels = [
        "Low / Controlled Engagement Profile", 
        "Moderate Burnout Profile", 
        "High Burnout Risk", 
        "Severe Burnout Indication"
    ]
    lvl = tier_labels[best_tier_idx]

    contributors = []
    try:
        explainer = shap.TreeExplainer(bst_median)
        shap_values = explainer.shap_values(df_row)
        row_shap = shap_values[0] if not isinstance(shap_values, list) else shap_values[0][0]

        for idx, f_name in enumerate(metadata["feature_order"]):
            try:
                weight = float(row_shap[idx])
            except (IndexError, TypeError, ValueError):
                weight = 0.0
            contributors.append({
                "feature": f_name,
                "display_name": get_display_name(f_name),
                "contribution": round(float(abs(weight)), 4),
                "direction": "+" if weight >= 0 else "-"
            })
    except Exception as e:
        print(f"SHAP Error in Domain 5: {e}")
        for f_name in metadata["feature_order"]:
            contributors.append({
                "feature": f_name,
                "display_name": get_display_name(f_name),
                "contribution": 0.0,
                "direction": "?"
            })

    return {
        "domain": "domain_5_occupational_burnout",
        "placement": {
            "burnout_index": round(pred_score, 3),
            "burnout_tier_label": lvl,
            "tier_thresholds": metadata.get("thresholds", None)
        },
        "top_contributors": sorted(contributors, key=lambda x: x["contribution"], reverse=True)[:3]
    }

# =====================================================================
# DOMAIN 6: SEVERE CLINICAL SCREENING MATRIX (LOG-ODDS ATTRIBUTION)
# =====================================================================
def evaluate_domain6_clinical(raw_payload):
    """
    Evaluates clinical risk categorization and checks for profile anomalies.
    """
    model_path = os.path.join(os.path.dirname(__file__), "saved_states", "domain6_clinical.pkl")
    meta_path = os.path.join(os.path.dirname(__file__), "saved_states", "domain6_clinical_metadata.json")
    
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        return {"domain": "domain_6_severe_clinical", "placement": {"predicted_condition_code": 0}, "top_contributors": []}
        
    with open(model_path, "rb") as f:
        model_payload = pickle.load(f)
    with open(meta_path, "r") as f:
        metadata = json.load(f)
        
    binary_features = metadata["features"]
    
    input_vector = []
    for feat_name in binary_features:
        raw_val = raw_payload.get(feat_name, 0)
        try:
            input_vector.append(1 if int(float(raw_val)) > 0 else 0)
        except (ValueError, TypeError):
            input_vector.append(0)
            
    input_arr = np.array([input_vector])
    
    classifier = model_payload["cluster_classifier"]
    anomaly_detector = model_payload["isolation_forest"]
    
    predicted_condition = int(classifier.predict(input_arr)[0])
    anomaly_score = anomaly_detector.predict(input_arr)[0]
    anomaly_flag = True if anomaly_score == -1 else False
    
    if hasattr(classifier, "coef_"):
        if len(classifier.coef_.shape) > 1:
            class_coefficients = classifier.coef_[min(predicted_condition, classifier.coef_.shape[0] - 1)]
            intercept = classifier.intercept_[min(predicted_condition, classifier.intercept_.shape[0] - 1)]
        else:
            class_coefficients = classifier.coef_[0]
            intercept = classifier.intercept_[0]
    else:
        class_coefficients = np.zeros(len(binary_features))
        intercept = 0.0

    contributors = []
    for idx, feat_name in enumerate(binary_features):
        if input_vector[idx] == 0:
            continue
            
        coeff_weight = float(class_coefficients[idx])
        contribution_value = coeff_weight * input_vector[idx]
        
        contributors.append({
            "feature": feat_name,
            "display_name": get_display_name(feat_name),
            "contribution": round(float(abs(contribution_value)), 4),
            "direction": "+" if contribution_value >= 0 else "-"
        })
        
    contributors = sorted(contributors, key=lambda x: x["contribution"], reverse=True)
    
    try:
        class_order = metadata.get("class_order", [])
        if 0 <= predicted_condition < len(class_order):
            condition_label = class_order[predicted_condition]
        else:
            condition_label = "Unknown Cluster"
    except Exception:
        condition_label = "Evaluation Pending"
        
    return {
        "domain": "domain_6_severe_clinical",
        "placement": {
            "predicted_condition_code": int(predicted_condition),
            "predicted_condition_label": condition_label,
            "anomaly_review_flag": bool(anomaly_flag)
        },
        "top_contributors": contributors[:3]
    }