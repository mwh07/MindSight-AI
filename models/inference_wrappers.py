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
# This used to also be shared with pdf_generator.py, but PDF generation has
# been discarded -- the JSON output of generate_full_profile() is now the
# sole presentation contract, consumed directly by the frontend. Kept this
# map here (rather than inlining labels per-domain again) because a single
# shared source of truth is still correct even with only one consumer.
try:
    from models.feature_mappings import FEATURE_TRANSLATION_MAP
except ImportError:
    try:
        from feature_mappings import FEATURE_TRANSLATION_MAP
    except ImportError:
        FEATURE_TRANSLATION_MAP = {}


def get_display_name(feature_key, fallback=None):
    """
    Single helper every domain below uses to resolve a feature's
    human-readable label. Looks up FEATURE_TRANSLATION_MAP first; if a key
    is genuinely absent from that map (e.g. a synthetic/derived feature like
    "PHQ_Core" that isn't a raw questionnaire item), falls back to a
    title-cased version of the raw key rather than ever leaving an
    all-caps/raw code like "IAT2" in front of the person reading their own
    report.
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
    missing or unreadable -- per-trait scoring failures are now isolated so one bad trait cannot
    discard valid scores already computed for the other traits.
    """
    model_path = os.path.join(os.path.dirname(__file__), "saved_states", "domain1_grm_parameters.pkl")

    # Static fallback traits layer if the pickle file is not found.
    # Matches schema_config.json exactly: 3 items per trait, not 4.
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
                # Each trait is scored independently -- a failure scoring ONE trait must not
                # discard correctly-computed scores for the other traits.
                try:
                    item_params = grm_registry[trait_name].get("items", {})

                    # Initialize total log-likelihood array for the candidate theta grid
                    log_lik = np.zeros_like(grid)

                    for item_id, params in item_params.items():
                        try:
                            raw_val = float(raw_payload.get(item_id, 3.0))
                            # Constrain Likert responses to standard 1 to 5 index range
                            resp = int(np.clip(round(raw_val), 1, 5))
                        except (ValueError, TypeError):
                            resp = 3

                        # Extract item discrimination (a) and threshold parameters (b)
                        a = float(params.get("a", 1.0))
                        b = np.array(params.get("b", [-1.5, -0.5, 0.5, 1.5]))

                        # Compute category boundary probabilities across the entire theta grid
                        # Shape: (len(grid), len(b))
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
                    # every other trait's already-computed real theta is preserved.
                    continue

    # Fill in ONLY the traits that real GRM scoring did not produce above --
    # this no longer wipes out traits that scored successfully.
    for trait_name, keys in traits_fallback.items():
        if trait_name in placement:
            continue
        vals = []
        for k in keys:
            try:
                vals.append(float(raw_payload.get(k, 3.0)))
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
    FIX: Realigns input key routing to check for 'Q1'-'Q10' before falling back to 'RSE1'-'RSE10',
         and corrects reverse-scoring to map to the verified 0-4 numeric range.
    """
    reverse_scoring_items = [3, 5, 8, 9, 10]
    total_score = 0
    item_contributions = []
    
    for i in range(1, 11):
        # Dynamically support both 'Q1' style and 'RSE1' style payload signatures
        raw_val = raw_payload.get(f"Q{i}", raw_payload.get(f"RSE{i}", 2.0))
        try:
            val = int(float(raw_val))
            val = max(0, min(4, val))  # Strict scale boundaries [0, 4]
        except (ValueError, TypeError):
            val = 2
            
        if i in reverse_scoring_items:
            processed_val = 4 - val  # Reverse score on a 0-4 point scale
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
# DOMAIN 3: MOOD SEVERITY & SLEEP METRICS (UNTOUCHED)
# =====================================================================
def evaluate_domain3_mood_sleep(raw_payload):
    """
    Evaluates clinical PHQ-9 tracking matrices and localized sleep timings.
    Reads the real schema_config.json key names (DPQ010-DPQ090) and excludes
    NHANES refusal/"don't know" codes (7, 9) from the symptom sum, per spec.
    DPQ100 is intentionally excluded -- it is an impairment/difficulty item,
    not one of the 9 PHQ-9 symptom-frequency items.
    """
    phq_total = 0
    for i in range(1, 10):
        item_key = f"DPQ0{i}0"
        try:
            val = int(float(raw_payload.get(item_key, 0)))
            if val in (7, 9):
                # Refusal / "don't know" -- excluded from the symptom sum entirely
                continue
            phq_total += max(0, min(3, val))
        except (ValueError, TypeError):
            pass
            
    sleep_onset = parse_time_to_hours(raw_payload.get("SLQ300", "23:00"))
    sleep_wakeup = parse_time_to_hours(raw_payload.get("SLQ310", "07:00"))
    
    duration = sleep_wakeup - sleep_onset
    if duration < 0:
        duration += 24.0
        
    if phq_total >= 20:
        severity = "Severe Depression"
    elif phq_total >= 15:
        severity = "Moderately Severe Depression"
    elif phq_total >= 10:
        severity = "Moderate Depression"
    elif phq_total >= 5:
        severity = "Mild Depression"
    else:
        severity = "Minimal or Baseline Profile"

    return {
        "domain": "domain_3_mood_sleep",
        "placement": {
            "phq9_sum": int(phq_total),
            "severity_label": severity,
            "calculated_sleep_duration_hours": round(float(duration), 2)
        },
        "top_contributors": [
            {"feature": "PHQ_Core", "display_name": "Symptom Burden Summation", "contribution": float(phq_total), "direction": "+"},
            {"feature": "Sleep_Duration", "display_name": "Calculated Sleep Duration", "contribution": round(float(abs(7.5 - duration)), 4), "direction": "+" if duration >= 7.0 else "-"}
        ]
    }

# =====================================================================
# DOMAIN 4: ATTACHMENT & LONELINESS FOREST PIPELINE (FIXED & ISOLATED)
# =====================================================================
def evaluate_domain4_multitask(raw_payload):
    """
    Evaluates attachment, loneliness, and relationship dynamics via the trained model.
    Drivers are now sourced ONLY from the real model's SHAP attributions -- the
    previously hardcoded feature_importance_map has been removed entirely, since it
    was a fixed dictionary of invented numbers, not derived from any trained model.
    Strictly limits inputs and drivers to IAT and Loneliness indices; domain 5
    leakage remains excluded.
    """
    # 1. Resolve asset paths dynamically
    model_path = os.path.join(os.path.dirname(__file__), "saved_states", "domain4_digital_social.pkl")
    meta_path = os.path.join(os.path.dirname(__file__), "saved_states", "domain4_digital_social_metadata.json")
    
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        model_path = os.path.join(os.path.dirname(__file__), "saved_states", "domain4_multitask.pkl")
        meta_path = os.path.join(os.path.dirname(__file__), "saved_states", "domain4_multitask_metadata.json")

    # 2. Fallback generator using ONLY native Domain 4 metrics -- used ONLY when the
    #    real trained model/metadata cannot be loaded at all. This path computes raw
    #    item sums, NOT model predictions, and must never be confused with real output.
    def execute_contract_fallback():
        iat_vals = []
        for i in range(1, 11):
            val = raw_payload.get(f"IAT{i}", 3.0)
            try:
                iat_vals.append(float(val))
            except (ValueError, TypeError):
                iat_vals.append(3.0)
                
        lone_vals = []
        for i in range(1, 7):
            val = raw_payload.get(f"loneliness{i}", 3.0)
            try:
                lone_vals.append(float(val))
            except (ValueError, TypeError):
                lone_vals.append(3.0)
                
        pred_iat = float(np.sum(iat_vals))
        pred_lone = float(np.sum(lone_vals))
        
        contributors = []
        for i, val in enumerate(iat_vals):
            f_name = f"IAT{i+1}"
            variance = val - 3.0
            if variance != 0:
                contributors.append({
                    "feature": f_name,
                    "display_name": get_display_name(f_name),
                    "contribution": round(abs(variance), 4),
                    "direction": "+" if variance >= 0 else "-"
                })
                
        for i, val in enumerate(lone_vals):
            f_name = f"loneliness{i+1}"
            variance = val - 3.0
            if variance != 0:
                contributors.append({
                    "feature": f_name,
                    "display_name": get_display_name(f_name, fallback=f"Loneliness{i+1}"),
                    "contribution": round(abs(variance), 4),
                    "direction": "+" if variance >= 0 else "-"
                })
                
        if not contributors:
            contributors = [
                {"feature": "IAT1", "display_name": get_display_name("IAT1"), "contribution": 0.0, "direction": "+"},
                {"feature": "loneliness1", "display_name": get_display_name("loneliness1", fallback="Loneliness1"), "contribution": 0.0, "direction": "+"}
            ]
            
        return {
            "domain": "domain_4_digital_and_social",
            "placement": {
                # Fallback uses raw sums (not model predictions)
                "predicted_total_internet_addiction": round(pred_iat, 3),
                "predicted_total_loneliness": round(pred_lone, 3),
                "loneliness_score": round(pred_lone * 3.33, 3),  # rough conversion to 0-100
                "classification": "Elevated Distress Profile" if pred_lone > 18 else "Baseline Cohort Profile",
                "data_source": "fallback_raw_sum"
            },
            "top_contributors": sorted(contributors, key=lambda x: x["contribution"], reverse=True)[:3]
        }

    # If assets are physically missing, trigger the contract fallback immediately
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        return execute_contract_fallback()
        
    try:
        # 3. Load model state and metadata layout
        with open(model_path, "rb") as f:
            model_payload = pickle.load(f)
        with open(meta_path, "r") as f:
            metadata = json.load(f)

        # 4. Extract BOTH model objects. The design is two independent
        #    RandomForestRegressor models (addiction, loneliness), each fit on
        #    its own feature subset -- NOT one shared 18-feature dataframe fed
        #    to a single extracted model. Feeding the full schema feature list
        #    to a model trained on a smaller subset raises a hard sklearn
        #    ValueError ("feature names unseen at fit time"), which is exactly
        #    what was silently sending every call here into the fallback path.
        addiction_model = None
        loneliness_model = None

        if isinstance(model_payload, dict):
            for key in ["addiction_model", "iat_model", "internet_addiction_model", "model_addiction"]:
                if key in model_payload:
                    addiction_model = model_payload[key]
                    break
            for key in ["loneliness_model", "lone_model", "model_loneliness"]:
                if key in model_payload:
                    loneliness_model = model_payload[key]
                    break

            # If named keys weren't found, fall back to inspecting each fitted
            # model's own feature_names_in_ to tell them apart -- this works
            # regardless of what the dict's keys happen to be named, since
            # scikit-learn stores the real fit-time feature names on the
            # estimator itself, not on the dict wrapper around it.
            if addiction_model is None or loneliness_model is None:
                candidates = [v for v in model_payload.values() if hasattr(v, "predict")]
                for cand in candidates:
                    fit_features = list(getattr(cand, "feature_names_in_", []))
                    if any(f.startswith("IAT") for f in fit_features) and addiction_model is None:
                        addiction_model = cand
                    elif any(f.startswith("loneliness") for f in fit_features) and loneliness_model is None:
                        loneliness_model = cand
                # Last resort: if still unresolved and exactly two candidates exist,
                # assign by order (addiction first) per the documented design.
                if addiction_model is None and loneliness_model is None and len(candidates) >= 2:
                    addiction_model, loneliness_model = candidates[0], candidates[1]
        else:
            # Single object in the pickle -- cannot run two-model inference.
            raise ValueError("domain4_digital_social.pkl does not contain two separate models")

        if addiction_model is None or loneliness_model is None:
            raise ValueError("could not resolve both addiction and loneliness models from saved_states pickle")

        # 5. Build EACH model's input row from THAT model's own fit-time feature
        #    names (feature_names_in_), not from metadata["features"] -- this is
        #    correct regardless of which exact subset each model was trained on,
        #    without needing to hardcode or guess the split.
        def build_row_for(model):
            fit_features = list(getattr(model, "feature_names_in_", []))
            if not fit_features:
                # Model wasn't fit on a DataFrame (no stored feature names) --
                # fall back to metadata's declared feature list as a last resort.
                fit_features = metadata.get("features", [])
            row = [float(raw_payload.get(f_name, 2.0)) for f_name in fit_features]
            return pd.DataFrame([row], columns=fit_features), fit_features

        df_addiction, addiction_features = build_row_for(addiction_model)
        df_loneliness, loneliness_features = build_row_for(loneliness_model)

        # 6. Run each model on ONLY its own feature subset.
        pred_addiction = float(addiction_model.predict(df_addiction)[0])
        pred_loneliness_score = float(loneliness_model.predict(df_loneliness)[0])

        # 7. Extract real SHAP attributions from each model separately, then
        #    merge into one combined top-contributors list for this domain.
        contributors = []
        for model, df_row, feat_list in [
            (addiction_model, df_addiction, addiction_features),
            (loneliness_model, df_loneliness, loneliness_features),
        ]:
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(df_row)
                if isinstance(shap_values, list):
                    row_shap = shap_values[0][0]
                elif len(np.array(shap_values).shape) == 3:
                    row_shap = np.array(shap_values)[0, :, 0]
                else:
                    row_shap = shap_values[0]
            except Exception:
                row_shap = [0.0] * len(feat_list)

            for idx, f_name in enumerate(feat_list):
                # GUARD: Prevent any leaked external features (e.g. domain 5,
                # or age/gender) from surfacing in top drivers for this domain.
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

        if not contributors:
            for i in range(1, 4):
                contributors.append({"feature": f"IAT{i}", "display_name": get_display_name(f"IAT{i}"), "contribution": 0.0, "direction": "+"})

        # -------------------------------------------------------------------------
        # FIX: USE MODEL PREDICTIONS INSTEAD OF RAW SUMS
        # -------------------------------------------------------------------------
        # The raw sums are NOT used for the main placement fields.
        # We keep them only for reference but do not store them.
        # All three predicted fields now come from the trained models.
        return {
            "domain": "domain_4_digital_and_social",
            "placement": {
                "predicted_total_internet_addiction": round(pred_addiction, 3),
                "predicted_total_loneliness": round(pred_loneliness_score, 3),
                "loneliness_score": round(pred_loneliness_score, 3),
                "classification": "Elevated Distress Profile" if pred_loneliness_score > 50 else "Baseline Cohort Profile",
                "data_source": "model_prediction"
            },
            "top_contributors": sorted(contributors, key=lambda x: x["contribution"], reverse=True)[:3]
        }
        
    except Exception:
        # Catch-all safety net returns the completely clean domain-isolated fallback
        return execute_contract_fallback()
        
# =====================================================================
# DOMAIN 5: OCCUPATIONAL BURNOUT GRADIENT ENGINE (UNTOUCHED)
# =====================================================================
def evaluate_domain5_burnout(raw_payload):
    """
    Evaluates continuous occupational burnout indices using XGBoost.
    Driver contributions now come from real SHAP attributions against the trained
    model, rather than a flat (raw_value * 0.12) scaling -- the previous formula
    derived both magnitude and direction directly from the input value alone,
    which meant a driver's reported direction could never disagree with whether
    the raw score was above or below the scale midpoint, regardless of what the
    trained model actually learned about that feature.
    """
    model_path = os.path.join(os.path.dirname(__file__), "saved_states", "domain5_burnout.json")
    meta_path = os.path.join(os.path.dirname(__file__), "saved_states", "domain5_burnout_metadata.json")
    
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        # No real metadata available -- explicitly state tier_thresholds as
        # None rather than omitting the key or inventing placeholder numbers,
        # so a frontend can distinguish "model unavailable" from "model ran,
        # here are its real thresholds."
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
        # If SHAP attribution genuinely cannot run, fall back to reporting raw input
        # magnitude only -- explicitly without a fabricated direction inferred from
        # the input value, since that direction is not a model-derived signal.
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
            # Real, already-loaded tier boundaries from this model's own
            # metadata -- not invented display bounds. XGBoost regression
            # output has no natural 0-1 or 0-10 range, so without these a
            # frontend has no honest way to size a bar/gauge for this value.
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
    FIX: Filters out inactive features to stop zero-contribution elements 
         from leaking into and shuffling the top driver slots.
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
        # Only evaluate features that were actively triggered (value == 1)
        if input_vector[idx] == 0:
            continue
            
        coeff_weight = float(class_coefficients[idx])
        contribution_value = coeff_weight * input_vector[idx]
        
        # Keep track of true directionality based on underlying weight sign
        contributors.append({
            "feature": feat_name,
            "display_name": get_display_name(feat_name),
            "contribution": round(float(abs(contribution_value)), 4),
            "direction": "+" if contribution_value >= 0 else "-"
        })
        
    # Sort purely by absolute impact magnitude
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