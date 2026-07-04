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

    return {
        "domain": "domain_2_self_esteem",
        "placement": {
            "score": int(total_score),
            "max_possible_score": 40,
            "classification": "High Self-Esteem" if total_score >= 30 else ("Normal" if total_score >= 19 else "Low Self-Esteem")
        },
        "top_contributors": item_contributions[:3]
    }

# =====================================================================
# DOMAIN 3: MOOD SEVERITY & SLEEP METRICS (ML + SHAP)
# =====================================================================
def evaluate_domain3_mood_sleep(raw_payload):
    """
    Evaluates clinical PHQ-9 tracking matrices and localized sleep timings.
    NOW USES THE TRAINED LIGHTGBM CLASSIFIER for severity label and SHAP-based contributors,
    while still reporting the deterministic phq9_sum and calculated sleep duration.

    Falls back to deterministic-only logic if model files are missing.
    """
    # --- Deterministic calculations (always computed, kept for reference) ---
    phq_total = 0
    for i in range(1, 10):
        item_key = f"DPQ0{i}0"
        try:
            val = int(float(raw_payload.get(item_key, 0)))
            if val in (7, 9):
                continue
            phq_total += max(0, min(3, val))
        except (ValueError, TypeError):
            pass

    sleep_onset = parse_time_to_hours(raw_payload.get("SLQ300", "23:00"))
    sleep_wakeup = parse_time_to_hours(raw_payload.get("SLQ310", "07:00"))
    duration = sleep_wakeup - sleep_onset
    if duration < 0:
        duration += 24.0

    # --- Deterministic severity (fallback) ---
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

    # --- Try to load ML model and metadata ---
    model_path = os.path.join(os.path.dirname(__file__), "saved_states", "domain3_mood_sleep.txt")
    meta_path = os.path.join(os.path.dirname(__file__), "saved_states", "domain3_mood_sleep_metadata.json")

    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        # Fallback: use deterministic severity and hardcoded contributors
        return {
            "domain": "domain_3_mood_sleep",
            "placement": {
                "phq9_sum": int(phq_total),
                "severity_label": deterministic_severity,
                "calculated_sleep_duration_hours": round(float(duration), 2)
            },
            "top_contributors": [
                {"feature": "PHQ_Core", "display_name": "Symptom Burden Summation", "contribution": float(phq_total), "direction": "+"},
                {"feature": "Sleep_Duration", "display_name": "Calculated Sleep Duration", "contribution": round(float(abs(7.5 - duration)), 4), "direction": "+" if duration >= 7.0 else "-"}
            ]
        }

    # --- Load model and metadata ---
    try:
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        booster = lgb.Booster(model_file=model_path)
    except Exception:
        # If loading fails, fall back to deterministic
        return {
            "domain": "domain_3_mood_sleep",
            "placement": {
                "phq9_sum": int(phq_total),
                "severity_label": deterministic_severity,
                "calculated_sleep_duration_hours": round(float(duration), 2)
            },
            "top_contributors": [
                {"feature": "PHQ_Core", "display_name": "Symptom Burden Summation", "contribution": float(phq_total), "direction": "+"},
                {"feature": "Sleep_Duration", "display_name": "Calculated Sleep Duration", "contribution": round(float(abs(7.5 - duration)), 4), "direction": "+" if duration >= 7.0 else "-"}
            ]
        }

    # --- Build feature matrix exactly as in training ---
    # Features are stored in metadata["features"] in the correct order
    # These include DPQ010-DPQ090, bed_hours, wake_hours, calculated_sleep_duration
    features = metadata["features"]
    # Compute derived features (same as training script)
    bed_hours = parse_time_to_hours(raw_payload.get("SLQ300", "23:00"))
    wake_hours = parse_time_to_hours(raw_payload.get("SLQ310", "07:00"))
    calc_sleep = (wake_hours - bed_hours) % 24.0

    # Build a row dict with all features in the correct order
    row_dict = {}
    for f in features:
        if f == "bed_hours":
            row_dict[f] = bed_hours
        elif f == "wake_hours":
            row_dict[f] = wake_hours
        elif f == "calculated_sleep_duration":
            row_dict[f] = calc_sleep
        else:
            # It's a DPQ item
            row_dict[f] = float(raw_payload.get(f, 0))
    # Convert to DataFrame with one row
    df_row = pd.DataFrame([row_dict], columns=features)

    # --- Predict severity class ---
    pred_class = int(booster.predict(df_row, raw_score=False).argmax(axis=1)[0])
    classes = metadata.get("classes", ["Minimal", "Mild", "Moderate", "Moderately Severe", "Severe"])
    ml_severity_label = classes[pred_class] if pred_class < len(classes) else deterministic_severity

    # --- Compute SHAP values for contributors ---
    try:
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(df_row)
        # shap_values is a list of arrays (one per class), we want the predicted class
        if isinstance(shap_values, list):
            shap_row = shap_values[pred_class][0]
        elif hasattr(shap_values, "shape") and len(shap_values.shape) == 3:
            shap_row = shap_values[0, :, pred_class]
        else:
            shap_row = shap_values[0]
    except Exception:
        shap_row = [0.0] * len(features)

    # Build contributors from SHAP (only features with non‑zero contribution)
    contributors = []
    for idx, f_name in enumerate(features):
        val = float(shap_row[idx]) if idx < len(shap_row) else 0.0
        if abs(val) > 1e-6:
            contributors.append({
                "feature": f_name,
                "display_name": get_display_name(f_name, fallback=f_name.replace("_", " ").title()),
                "contribution": round(abs(val), 4),
                "direction": "+" if val >= 0 else "-"
            })

    # Sort by absolute contribution descending
    contributors = sorted(contributors, key=lambda x: x["contribution"], reverse=True)

    # If no contributors (e.g., all zero), add fallback
    if not contributors:
        contributors = [
            {"feature": "PHQ_Core", "display_name": "Symptom Burden Summation", "contribution": float(phq_total), "direction": "+"},
            {"feature": "Sleep_Duration", "display_name": "Calculated Sleep Duration", "contribution": round(float(abs(7.5 - duration)), 4), "direction": "+" if duration >= 7.0 else "-"}
        ]

    return {
        "domain": "domain_3_mood_sleep",
        "placement": {
            "phq9_sum": int(phq_total),
            "severity_label": ml_severity_label,   # ML-derived
            "calculated_sleep_duration_hours": round(float(duration), 2)
        },
        "top_contributors": contributors[:3]   # Top 3 SHAP drivers
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
# DOMAIN 5: OCCUPATIONAL BURNOUT GRADIENT ENGINE (UNTOUCHED)
# =====================================================================
def evaluate_domain5_burnout(raw_payload):
    """
    Evaluates continuous occupational burnout indices using XGBoost.
    """
    model_path = os.path.join(os.path.dirname(__file__), "saved_states", "domain5_burnout.json")
    meta_path = os.path.join(os.path.dirname(__file__), "saved_states", "domain5_burnout_metadata.json")
    
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        return {
            "domain": "domain_5_occupational_burnout",
            "placement": {"burnout_index": 5.0, "burnout_tier_label": "Unavailable", "tier_thresholds": None},
            "top_contributors": []
        }
        
    with open(meta_path, "r") as f:
        metadata = json.load(f)
        
    features = metadata["features"]
    thresholds = metadata["thresholds"]
    
    input_row = []
    for f_name in features:
        input_row.append(float(raw_payload.get(f_name, 3.0 if "score" in f_name else 30.0)))
        
    df_row = pd.DataFrame([input_row], columns=features)
    
    bst = xgb.Booster()
    bst.load_model(model_path)
    dmat = xgb.DMatrix(df_row)
    pred_score = float(bst.predict(dmat)[0])
    
    if pred_score >= thresholds["high_to_severe"]:
        lvl = "Severe Burnout Indication"
    elif pred_score >= thresholds["moderate_to_high"]:
        lvl = "High Burnout Risk"
    elif pred_score >= thresholds["low_to_moderate"]:
        lvl = "Moderate Burnout Profile"
    else:
        lvl = "Low / Controlled Engagement Profile"

    contributors = []
    try:
        explainer = shap.TreeExplainer(bst)
        shap_values = explainer.shap_values(df_row)
        row_shap = shap_values[0] if not isinstance(shap_values, list) else shap_values[0][0]

        for idx, f_name in enumerate(features):
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
    except Exception:
        for f_name in features:
            raw_val = float(raw_payload.get(f_name, 3.0 if "score" in f_name else 30.0))
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
            "tier_thresholds": {
                "low_to_moderate": thresholds["low_to_moderate"],
                "moderate_to_high": thresholds["moderate_to_high"],
                "high_to_severe": thresholds["high_to_severe"]
            }
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
    
    classifier = model_payload["classifier"]
    anomaly_detector = model_payload["anomaly_detector"]
    
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
    
    return {
        "domain": "domain_6_severe_clinical",
        "placement": {
            "predicted_condition_code": predicted_condition,
            "predicted_condition_label": CLINICAL_SEVERITY_MAP.get(predicted_condition, "Evaluation Pending"),
            "anomaly_review_flag": anomaly_flag
        },
        "top_contributors": contributors[:3]
    }