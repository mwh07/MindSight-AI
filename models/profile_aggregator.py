#!/usr/bin/env python3
"""
MINDSIGHT Profile Aggregator Engine (v3.9 - Production Hardened)
Orchestrates multi-domain inference execution and runs deterministic cross-domain 
clinical synthesis logic to map global profile states. Aligned with v3.9 Runtime contract signatures.
"""

import os
import json
import datetime
from models.inference_wrappers import (
    evaluate_domain1_personality,
    evaluate_domain2_self_esteem,
    evaluate_domain3_mood_sleep,
    evaluate_domain4_multitask,
    evaluate_domain5_burnout,
    evaluate_domain6_clinical
)

def evaluate_cross_domain_synthesis(domain_outputs):
    """
    Evaluates multi-domain results using a tiered clinical severity matrix 
    to build an objective, streamlined overall mental health profile paragraph.
    
    Returns a single concise summary narrative string.
    """
    # -------------------------------------------------------------------------
    # 1. Feature Extraction & Stratification (Defensive Key Parsing)
    # -------------------------------------------------------------------------
    # Domain 1
    d1_placement = domain_outputs.get("domain_1_personality", {}).get("placement", {})
    p_est = float(d1_placement.get("emotional_stability", 0.0))
    
    # Domain 2
    d2_placement = domain_outputs.get("domain_2_self_esteem", {}).get("placement", {})
    rse_score = float(d2_placement.get("score", 15.0))
    # Read the REAL max score that domain 2 actually used (40, per its own placement
    # output) rather than hardcoding 30 -- the two values give materially different
    # percentiles and previously shifted every percentile-based threshold below.
    rse_max = float(d2_placement.get("max_possible_score", 40.0))
    rse_pct = (rse_score / rse_max) * 100.0 if rse_max > 0 else 0.0
    
    # Domain 3 (Flexible Route Catching)
    d3_key = "domain_3_mood_sleep" if "domain_3_mood_sleep" in domain_outputs else "domain_3_mood_and_sleep"
    d3_placement = domain_outputs.get(d3_key, {}).get("placement", {})
    mood_class = str(d3_placement.get("severity_label", d3_placement.get("assigned_severity_class", "Minimal")))
    phq_sum = int(d3_placement.get("phq9_sum", d3_placement.get("deterministic_phq9_sum", 0)))
    
    # Domain 4 -- outer key is now consistently "domain_4_digital_and_social"
    # (see generate_full_profile). The "domain_4_multitask" fallback is kept only
    # for backward compatibility with any stale cached payloads from before this fix.
    d4_key = "domain_4_digital_and_social" if "domain_4_digital_and_social" in domain_outputs else "domain_4_multitask"
    d4_placement = domain_outputs.get(d4_key, {}).get("placement", {})
    lone_score = float(d4_placement.get("loneliness_score", d4_placement.get("predicted_total_loneliness", 30.0)))
    
    # Domain 5
    d5_placement = domain_outputs.get("domain_5_occupational_burnout", {}).get("placement", {})
    burnout_lvl = str(d5_placement.get("burnout_tier_label", d5_placement.get("burnout_level", "Low")))
    
    # Domain 6
    d6_placement = domain_outputs.get("domain_6_severe_clinical", {}).get("placement", {})
    clinical_cond = str(d6_placement.get("predicted_condition_code", d6_placement.get("predicted_condition", "0")))
    anomaly_flag = bool(d6_placement.get("anomaly_review_flag", False))

    # Real-time extraction of live clinical label to block semantic logic drift
    friendly_condition = d6_placement.get("predicted_condition_label", None)
    if not friendly_condition:
        # This fallback map must mirror CLINICAL_SEVERITY_MAP in inference_wrappers.py
        # exactly. It previously used a completely different, fabricated set of labels
        # (e.g. "Generalized Anxiety Phenotype") that domain 6's actual model never
        # produces -- if this path were ever reached (such as when the domain 6 model
        # files are missing and the early-return stub omits predicted_condition_label),
        # it would have silently displayed a clinically incorrect diagnosis label.
        CONDITION_MAP = {
            "0": "Baseline Healthy Profile",
            "1": "Mild Symptomatic Profile",
            "2": "Moderate Distress Phenotype",
            "3": "Severe Clinical Screening Indication"
        }
        cond_key = clinical_cond.split('.')[0] if '.' in clinical_cond else clinical_cond
        friendly_condition = CONDITION_MAP.get(cond_key, "Evaluation Pending Profile")

    # Flexible matching layers accounting for descriptive string labels
    is_severe_mood = phq_sum >= 15 or any(x in mood_class for x in ["Severe", "Moderately Severe"])
    is_moderate_mood = phq_sum >= 10 or any(x in mood_class for x in ["Moderate", "Severe", "Moderately Severe"])
    is_high_burnout = any(x in burnout_lvl for x in ["High", "Severe"])
    is_elevated_burnout = any(x in burnout_lvl for x in ["Moderate", "High", "Severe"])

    # -------------------------------------------------------------------------
    # 2. Tiered Matrix Evaluation Logic (Calibrated Strings)
    # -------------------------------------------------------------------------
    
    # Tier A: Genuinely Severe Clinical Presentation / High Co-occurrence
    if is_severe_mood and (is_high_burnout or p_est < -1.5):
        narrative = (
            f"ACUTE PROFILE EVALUATION: Multi-domain tracking reveals a dense intersection of severe mood indicators "
            f"({mood_class}, PHQ-9 Sum: {phq_sum}) running concurrently with significant systemic exhaustion "
            f"({burnout_lvl}). This presentation is characteristic of severe burnout-depression crossover, "
            f"indicating that environmental pressures have exceeded active coping capacities. Formal clinical guidance "
            f"and structured workload mitigation are recommended to support stabilization."
        )

    # =========================================================================
    # FIXED: Tier B Anomaly Route expanded to catch moderate/elevated clinical indicators
    # =========================================================================
    # Tier B: Atypical / Anomaly Sub-pathway (Balanced & Objective)
    elif anomaly_flag:
        if is_severe_mood or is_moderate_mood or is_high_burnout or is_elevated_burnout:
            narrative = (
                f"COMPLEX PROFILE EVALUATION: Unsupervised screening models flag an atypical response configuration. "
                f"While tracking markers align closely with the {friendly_condition}, the overall structure falls outside "
                f"standard baseline distributions concurrently with elevated clinical metrics (PHQ-9 Sum: {phq_sum}). "
                f"A comprehensive professional differential evaluation is recommended to reconcile these mixed indicators."
            )
        else:
            narrative = (
                f"COMPLEX PROFILE EVALUATION: Input data forms an uncommon psychological signature that deviates slightly "
                f"from standard population archetypes. While baseline scores map closest to the {friendly_condition}, "
                f"underlying acute indicators remain low to moderate, suggesting this pattern may reflect localized situational "
                f"stressors rather than an active clinical condition."
            )

    # Tier C: Standard Sub-Acute Clinical Signal
    elif is_moderate_mood:
        narrative = (
            f"CLINICAL MOOD SIGNAL: Assessment reports active, moderate symptoms of mood disruption falling within the "
            f"{mood_class} classification, with secondary checks mapping close to the {friendly_condition}. While general "
            f"adaptive functioning remains intact, implementing proactive stress-management strategies and routine wellness "
            f"monitoring is recommended to prevent symptom escalation."
        )

    # Tier D: Elevated Environmental Distress (High Burnout / High Loneliness)
    elif is_elevated_burnout or lone_score >= 45.0 or rse_pct <= 30.0:
        if is_high_burnout and (rse_pct <= 25.0 or lone_score >= 50.0):
            narrative = (
                f"ENVIRONMENTAL DISTRESS FOCUS: Primary distress vectors are situated in the immediate environment, "
                f"showing prominent professional exhaustion ({burnout_lvl}) interacting with vulnerable social and "
                f"self-appraisal scales. Workplace pressures appear to be amplifying personal vulnerabilities, and interventions "
                f"should prioritize establishing firm administrative boundaries and intentional social re-engagement."
            )
        else:
            narrative = (
                f"OCCUPATIONAL EXHAUSTION FOCUS: Primary distress vectors are isolated within the occupational domain, where a "
                f"{burnout_lvl} is detected. Crucially, underlying clinical mood and severe clinical markers "
                f"remain stable and well-contained. Interventions should focus on professional boundary-setting, task-autonomy updates, "
                f"and proactive stress-mitigation loops."
            )

    # Tier E: Completely Stable, Resilient, & Homeostatic
    else:
        narrative = (
            "STABLE FUNCTIONAL PROFILE: Multi-domain clinical evaluation reveals strong psychological resilience and homeostatic "
            "equilibrium. Neuro-emotional variables, self-appraisal parameters, and occupational metrics all align healthy "
            "normative population distributions with no active clinical or environmental distress markers detected."
        )

    return narrative

def generate_full_profile(user_responses):
    """
    Consumes raw questionnaire features, extracts diagnostic demographics,
    and executes the fully unified, auditable MINDSIGHT profile payload block.
    """
    # Defensive copy to avoid leaking or mutating state references upstream
    normalized_inputs = dict(user_responses)

    # NOTE: the two translation blocks below are retained for defensive compatibility
    # but are confirmed no-ops against the real schema_config.json input format.
    # evaluate_domain3_mood_sleep() now reads DPQ010-DPQ090 directly (see
    # inference_wrappers.py), so it no longer needs PHQ1-PHQ9 aliases.
    # evaluate_domain2_self_esteem() already checks Q{i} before falling back to
    # RSE{i}, so it never needs the RSE{i} aliases either. Both blocks are safe to
    # remove once confirmed no other caller depends on the aliased keys.

    # Fix 1: Map Domain 3 NHANES Database Codes (DPQ010-DPQ090) to standard Model-ready PHQ tokens
    dpq_to_phq_map = {
        "DPQ010": "PHQ1", "DPQ020": "PHQ2", "DPQ030": "PHQ3",
        "DPQ040": "PHQ4", "DPQ050": "PHQ5", "DPQ060": "PHQ6",
        "DPQ070": "PHQ7", "DPQ080": "PHQ8", "DPQ090": "PHQ9"
    }
    for dpq_key, phq_key in dpq_to_phq_map.items():
        if dpq_key in normalized_inputs and phq_key not in normalized_inputs:
            normalized_inputs[phq_key] = normalized_inputs[dpq_key]

    # Fix 2: Map Domain 2 Survey Intake Codes (Q1-Q10) to standard Model-ready Rosenberg (RSE) tokens
    for idx in range(1, 11):
        raw_q_key = f"Q{idx}"
        target_rse_key = f"RSE{idx}"
        if raw_q_key in normalized_inputs and target_rse_key not in normalized_inputs:
            normalized_inputs[target_rse_key] = normalized_inputs[raw_q_key]

    # Explicit mapping execution block ensuring exact v3.9 inference function linkages.
    # NOTE: outer key renamed from "domain_4_multitask" to "domain_4_digital_and_social"
    # to match schema_config.json and the inner "domain" field that
    # evaluate_domain4_multitask() already returns -- the previous mismatch (outer key
    # said "multitask", inner field said "digital_and_social") forced every downstream
    # consumer to guess which name to look up, and was the root cause of the flexible
    # d4_key fallback below.
    final_domain_outputs = {
        "domain_1_personality": evaluate_domain1_personality(normalized_inputs),
        "domain_2_self_esteem": evaluate_domain2_self_esteem(normalized_inputs),
        "domain_3_mood_sleep": evaluate_domain3_mood_sleep(normalized_inputs),
        "domain_4_digital_and_social": evaluate_domain4_multitask(normalized_inputs),
        "domain_5_occupational_burnout": evaluate_domain5_burnout(normalized_inputs),
        "domain_6_severe_clinical": evaluate_domain6_clinical(normalized_inputs)
    }
        
    # Execute cross-domain synthesis mapping pass
    global_summary = evaluate_cross_domain_synthesis(final_domain_outputs)
    
    # -------------------------------------------------------------------------
    # LIVE DEMOGRAPHIC EXTRACTION LAYER
    # -------------------------------------------------------------------------
    live_age = normalized_inputs.get("age", normalized_inputs.get("AGE", "N/A"))
    raw_gender = normalized_inputs.get("gender", normalized_inputs.get("GENDER", normalized_inputs.get("sex", "N/A")))
    
    gender_map = {
        "0": "Male", "0.0": "Male",
        "1": "Female", "1.0": "Female",
        "2": "Non-binary", "2.0": "Non-binary",
        "m": "Male", "f": "Female", "nb": "Non-binary",
        "male": "Male", "female": "Female", "non-binary": "Non-binary",
        "prefer not to say": "Prefer not to say"
    }
    
    normalized_gender_key = str(raw_gender).strip().lower()
    live_sex = gender_map.get(normalized_gender_key, str(raw_gender))
        
    # Generate dynamic report generation timestamp (Format: YYYYMMDD_HHMMSS)
    live_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    live_id = normalized_inputs.get("id_no", normalized_inputs.get("id", normalized_inputs.get("ID", f"MS-{live_timestamp}-ANONYMOUS")))
    
    return {
        "schema_version": "3.9",
        "id_no": live_id,
        "age": live_age,
        "sex": live_sex,
        "domain_scores": final_domain_outputs,
        "global_synthesis": global_summary
    }

if __name__ == "__main__":
    sample_payload = {
        "EXT1": 4, "EXT2": 3, "EXT3": 5, "EST1": 2, "EST2": 2, "EST3": 3,
        "AGR1": 4, "AGR2": 4, "AGR3": 4, "CSN1": 5, "CSN2": 4, "CSN3": 5,
        "OPN1": 4, "OPN2": 5, "OPN3": 4, "age": 28, "gender": 0,
        "DPQ010": 3, "DPQ020": 3, "DPQ030": 2, "DPQ040": 2, "DPQ050": 2, "DPQ060": 2, "DPQ070": 1, "DPQ080": 1, "DPQ090": 1,
        "Q1": 3, "Q2": 1, "Q3": 2, "Q4": 3, "Q5": 1, "Q6": 0, "Q7": 3, "Q8": 1, "Q9": 0, "Q10": 2,
        "SLQ300": "23:15", "SLQ310": "06:45",
        "work_hours_per_week": 45, "meetings_per_day": 4, "work_life_balance_score": 2, "job_satisfaction_score": 3,
        "deadline_pressure_score": 4, "autonomy_score": 2, "stress_score": 4, "social_support_score": 3,
        "unwanted_thoughts": 1, "repetitve_behaviors": 0, "overthinking": 1, "mind_going_blank": 0,
        "avoidance_of_social_activity": 0, "panic": 1, "hypervigilance": 0
    }
    print("🧪 Running profile aggregator verification loop against v3.9 updates...")
    test_output = generate_full_profile(sample_payload)
    print("\n✅ Success! Synthesis Result Sub-Keys:")
    print(f"   [Patient Sex] -> {test_output['sex']}")
    print(f"   [Global Synthesis Narrative] -> {test_output['global_synthesis']}")