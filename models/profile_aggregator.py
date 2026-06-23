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

def extract_domain_signals(domain_outputs):
    """
    Single shared extraction point for all cross-domain signals. Both
    evaluate_cross_domain_synthesis() and generate_plain_language_summary()
    call this, so the two narratives can never disagree about the underlying
    numbers -- they're reading the exact same parsed values.
    """
    # Domain 1
    d1_placement = domain_outputs.get("domain_1_personality", {}).get("placement", {})
    p_est = float(d1_placement.get("emotional_stability", 0.0))
    p_ext = float(d1_placement.get("extraversion", 0.0))
    p_agr = float(d1_placement.get("agreeableness", 0.0))
    p_csn = float(d1_placement.get("conscientiousness", 0.0))
    p_opn = float(d1_placement.get("openness", 0.0))

    # Domain 2
    d2_placement = domain_outputs.get("domain_2_self_esteem", {}).get("placement", {})
    rse_score = float(d2_placement.get("score", 15.0))
    # Read the REAL max score that domain 2 actually used (40, per its own placement
    # output) rather than hardcoding 30 -- the two values give materially different
    # percentiles and previously shifted every percentile-based threshold below.
    rse_max = float(d2_placement.get("max_possible_score", 40.0))
    rse_pct = (rse_score / rse_max) * 100.0 if rse_max > 0 else 0.0
    rse_classification = str(d2_placement.get("classification", "Normal"))

    # Domain 3 (Flexible Route Catching)
    d3_key = "domain_3_mood_sleep" if "domain_3_mood_sleep" in domain_outputs else "domain_3_mood_and_sleep"
    d3_placement = domain_outputs.get(d3_key, {}).get("placement", {})
    mood_class = str(d3_placement.get("severity_label", d3_placement.get("assigned_severity_class", "Minimal")))
    phq_sum = int(d3_placement.get("phq9_sum", d3_placement.get("deterministic_phq9_sum", 0)))
    sleep_hours = float(d3_placement.get("calculated_sleep_duration_hours", 7.5))

    # Domain 4 -- outer key is now consistently "domain_4_digital_and_social"
    # (see generate_full_profile). The "domain_4_multitask" fallback is kept only
    # for backward compatibility with any stale cached payloads from before this fix.
    d4_key = "domain_4_digital_and_social" if "domain_4_digital_and_social" in domain_outputs else "domain_4_multitask"
    d4_placement = domain_outputs.get(d4_key, {}).get("placement", {})
    lone_score = float(d4_placement.get("loneliness_score", d4_placement.get("predicted_total_loneliness", 30.0)))
    addiction_score = float(d4_placement.get("predicted_total_internet_addiction", 25.0))

    # Domain 5
    d5_placement = domain_outputs.get("domain_5_occupational_burnout", {}).get("placement", {})
    burnout_lvl = str(d5_placement.get("burnout_tier_label", d5_placement.get("burnout_level", "Low")))
    burnout_index = float(d5_placement.get("burnout_index", 0.0))

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
    # Domain 6 is "highly abnormal" when its own severity code/label indicates the
    # most severe clinical tier, OR when the isolation-forest anomaly flag fires
    # alongside at least moderate signal elsewhere -- a single anomaly flag in
    # isolation, with everything else low, is treated as situational, not severe.
    cond_key_normalized = clinical_cond.split('.')[0] if '.' in clinical_cond else clinical_cond
    is_severe_clinical = (
        cond_key_normalized == "3"
        or "Severe" in friendly_condition
        or (anomaly_flag and (is_severe_mood or is_high_burnout))
    )

    return {
        "p_est": p_est, "p_ext": p_ext, "p_agr": p_agr, "p_csn": p_csn, "p_opn": p_opn,
        "rse_score": rse_score, "rse_max": rse_max, "rse_pct": rse_pct, "rse_classification": rse_classification,
        "mood_class": mood_class, "phq_sum": phq_sum, "sleep_hours": sleep_hours,
        "lone_score": lone_score, "addiction_score": addiction_score,
        "burnout_lvl": burnout_lvl, "burnout_index": burnout_index,
        "friendly_condition": friendly_condition, "anomaly_flag": anomaly_flag,
        "is_severe_mood": is_severe_mood, "is_moderate_mood": is_moderate_mood,
        "is_high_burnout": is_high_burnout, "is_elevated_burnout": is_elevated_burnout,
        "is_severe_clinical": is_severe_clinical
    }


def _top_driver_phrase(top_contributors, n=2):
    """Joins the top N driver display names into a short natural-language phrase,
    e.g. 'agreeableness and extraversion' -- used to build each domain's one-line
    domain_summary without repeating the full driver list verbatim."""
    names = []
    for c in top_contributors[:n]:
        label = c.get("display_name") or c.get("feature", "")
        # Strip common suffixes that read awkwardly mid-sentence
        for suffix in [" Vector", " Index", " Score"]:
            if label.endswith(suffix):
                label = label[: -len(suffix)]
        names.append(label.strip())
    if not names:
        return None
    if len(names) == 1:
        return names[0].lower()
    return f"{names[0].lower()} and {names[1].lower()}"


def _build_scale_reference(domain_key, placement):
    """
    Returns scale context for the frontend to size bars/gauges honestly,
    sourced only from real, verifiable facts -- never invented numbers.
    Returns None for any value whose true range isn't actually knowable
    from the model/design itself (e.g. a regression model's raw output),
    rather than guessing a plausible-looking bound. A frontend receiving
    None here should render that value as a plain number, not a bar.
    """
    if domain_key == "domain_1_personality":
        # Real bound: inference_wrappers.py's GRM scoring grid is
        # np.linspace(-4.0, 4.0, 81) -- this is the actual evaluation range
        # used to fit theta, not an estimate.
        return {"type": "fixed_range", "min": -4.0, "max": 4.0}

    if domain_key == "domain_2_self_esteem":
        # Already self-describing via max_possible_score; min is always 0
        # since the RSE reverse-coded sum cannot go negative.
        return {"type": "fixed_range", "min": 0, "max": placement.get("max_possible_score", 40)}

    if domain_key in ("domain_3_mood_sleep", "domain_3_mood_and_sleep"):
        # PHQ-9 sum: 9 items, each scored 0-3 after refusal-code filtering --
        # a textbook, NHANES-standard bound, not a guess.
        return {
            "phq9_sum": {"type": "fixed_range", "min": 0, "max": 27},
            # Sleep duration has no universal hard min/max -- this is a
            # commonly cited clinical NORMAL RANGE for adults, presented
            # explicitly as a reference band rather than a scale bound, so
            # the frontend doesn't mistake "outside this band" for "invalid".
            "sleep_duration_hours": {"type": "reference_band", "low": 7.0, "high": 9.0}
        }

    if domain_key in ("domain_4_digital_and_social", "domain_4_multitask"):
        return {
            # Raw sums over a 1-5 Likert scale per item -- a verifiable
            # arithmetic bound given the known item count, not a guess.
            "predicted_total_internet_addiction": {"type": "fixed_range", "min": 10, "max": 50},
            "predicted_total_loneliness": {"type": "fixed_range", "min": 6, "max": 30},
            # loneliness_score is the trained model's own regression output,
            # not a raw sum -- its true range depends on the training target
            # distribution, which isn't available here. Stating an invented
            # bound for this would repeat the exact mistake already caught
            # once on this same field (the PDF's hardcoded /80.0 scale).
            "loneliness_score": None
        }

    if domain_key == "domain_5_occupational_burnout":
        # Sourced directly from inference_wrappers.py's own placement output
        # (tier_thresholds), which reads the real metadata thresholds at
        # inference time -- nothing invented here, just passed through.
        thresholds = placement.get("tier_thresholds")
        return {"type": "tier_thresholds", "thresholds": thresholds} if thresholds else None

    # Domain 6 needs no numeric scale -- predicted_condition_label and
    # anomaly_review_flag are already self-contained categorical signals.
    return None


def enrich_domain_outputs(domain_outputs, signals):
    """
    Adds four frontend-facing fields to each domain's output, computed once
    here so every consumer gets the same values rather than re-deriving them
    independently:

    - severity_tier: a normalized enum so the frontend can color-code/sort
      across all six domains without a per-domain lookup table of magic
      label strings. Domain 1 (personality) intentionally gets "descriptive"
      rather than a risk tier, since trait position is not a severity scale
      and labeling it as such would misrepresent what the domain measures.
    - domain_summary: a single plain-language sentence naming this domain's
      actual top driver(s) by name, so a frontend can show real insight
      immediately next to the raw driver list rather than just numbers.
    - relative_magnitude (0-100) on each item in top_contributors: that
      driver's contribution scaled against the largest contribution within
      this domain's own top-3, so a frontend can size a bar directly without
      re-implementing per-domain normalization itself.
    - scale_reference: explicit, sourced scale bounds (or an honest None
      when no real bound is knowable) so the frontend never has to guess
      axis/bar limits the way the old PDF generator did by hand each time.

    This is the sole presentation contract now that PDF generation has been
    discarded -- everything a frontend needs to render each domain without
    guessing belongs here.
    """
    enriched = {}
    for domain_key, d_data in domain_outputs.items():
        placement = d_data.get("placement", {})
        top_contribs = d_data.get("top_contributors", [])

        # --- relative_magnitude per driver ---
        max_val = max((abs(float(c.get("contribution", 0.0))) for c in top_contribs), default=0.0)
        enriched_contribs = []
        for c in top_contribs:
            c_copy = dict(c)
            try:
                contrib = abs(float(c.get("contribution", 0.0)))
                c_copy["relative_magnitude"] = round((contrib / max_val) * 100.0, 1) if max_val > 0 else 0.0
            except (ValueError, TypeError):
                c_copy["relative_magnitude"] = 0.0
            enriched_contribs.append(c_copy)

        # --- severity_tier + domain_summary, per domain ---
        driver_phrase = _top_driver_phrase(top_contribs)

        if domain_key == "domain_1_personality":
            severity_tier = "descriptive"
            domain_summary = (
                f"Most shaped by {driver_phrase}." if driver_phrase
                else "No single trait stands out strongly from the others."
            )

        elif domain_key == "domain_2_self_esteem":
            pct = signals["rse_pct"]
            if pct <= 25.0:
                severity_tier = "high"
            elif pct <= 45.0:
                severity_tier = "moderate"
            else:
                severity_tier = "low"
            domain_summary = (
                f"Self-esteem ({signals['rse_classification'].lower()}) was most influenced by {driver_phrase}."
                if driver_phrase else f"Self-esteem classified as {signals['rse_classification'].lower()}."
            )

        elif domain_key in ("domain_3_mood_sleep", "domain_3_mood_and_sleep"):
            if signals["is_severe_mood"]:
                severity_tier = "severe" if "Severe" in signals["mood_class"] and "Moderately" not in signals["mood_class"] else "high"
            elif signals["is_moderate_mood"]:
                severity_tier = "moderate"
            else:
                severity_tier = "low"
            domain_summary = (
                f"{signals['mood_class']} mood signal, driven mainly by {driver_phrase}."
                if driver_phrase else f"{signals['mood_class']} mood signal."
            )

        elif domain_key in ("domain_4_digital_and_social", "domain_4_multitask"):
            if signals["lone_score"] >= 50.0:
                severity_tier = "high"
            elif signals["lone_score"] >= 35.0:
                severity_tier = "moderate"
            else:
                severity_tier = "low"
            domain_summary = (
                f"Loneliness and digital-use signals were most influenced by {driver_phrase}."
                if driver_phrase else "Digital and social patterns are within a typical range."
            )

        elif domain_key == "domain_5_occupational_burnout":
            if signals["is_high_burnout"]:
                severity_tier = "severe" if "Severe" in signals["burnout_lvl"] else "high"
            elif signals["is_elevated_burnout"]:
                severity_tier = "moderate"
            else:
                severity_tier = "low"
            domain_summary = (
                f"{signals['burnout_lvl']}, driven mainly by {driver_phrase}."
                if driver_phrase else f"{signals['burnout_lvl']}."
            )

        elif domain_key == "domain_6_severe_clinical":
            if signals["is_severe_clinical"]:
                severity_tier = "severe"
            elif signals["anomaly_flag"]:
                severity_tier = "high"
            elif "Mild" in signals["friendly_condition"] or "Moderate" in signals["friendly_condition"]:
                severity_tier = "moderate"
            else:
                severity_tier = "low"
            anomaly_note = " An atypical response pattern was also flagged for review." if signals["anomaly_flag"] else ""
            domain_summary = (
                f"{signals['friendly_condition']}, most associated with {driver_phrase}.{anomaly_note}"
                if driver_phrase else f"{signals['friendly_condition']}.{anomaly_note}"
            )

        else:
            severity_tier = "unknown"
            domain_summary = None

        enriched_data = dict(d_data)
        enriched_data["top_contributors"] = enriched_contribs
        enriched_data["severity_tier"] = severity_tier
        enriched_data["domain_summary"] = domain_summary
        enriched_data["scale_reference"] = _build_scale_reference(domain_key, placement)
        enriched[domain_key] = enriched_data

    return enriched


def evaluate_cross_domain_synthesis(domain_outputs):
    """
    Evaluates multi-domain results using a tiered clinical severity matrix 
    to build an objective, streamlined overall mental health profile paragraph.
    
    Returns a single concise summary narrative string.
    """
    s = extract_domain_signals(domain_outputs)
    p_est = s["p_est"]
    rse_pct = s["rse_pct"]
    mood_class = s["mood_class"]
    phq_sum = s["phq_sum"]
    lone_score = s["lone_score"]
    burnout_lvl = s["burnout_lvl"]
    friendly_condition = s["friendly_condition"]
    anomaly_flag = s["anomaly_flag"]
    is_severe_mood = s["is_severe_mood"]
    is_moderate_mood = s["is_moderate_mood"]
    is_high_burnout = s["is_high_burnout"]
    is_elevated_burnout = s["is_elevated_burnout"]

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


def generate_plain_language_summary(domain_outputs):
    """
    Produces a short, plain-language overall wellbeing summary (a few sentences)
    intended for the person taking the assessment to read directly -- distinct
    from evaluate_cross_domain_synthesis()'s clinical-register narrative above.

    Both functions read from extract_domain_signals() so they can never disagree
    about the underlying numbers, even though they're written in different tones
    for different audiences.

    Important: this function describes patterns across six self-report
    questionnaires. It does not diagnose any condition. When domain 6's signal
    is genuinely severe (not just an isolated anomaly flag with everything else
    low), it adds an explicit, unambiguous recommendation to seek professional
    clinical support rather than softening or omitting that recommendation.
    """
    s = extract_domain_signals(domain_outputs)

    # --- Personality, in plain terms ---
    trait_scores = {
        "extraversion": s["p_ext"], "emotional stability": s["p_est"],
        "agreeableness": s["p_agr"], "conscientiousness": s["p_csn"], "openness": s["p_opn"]
    }
    standout_trait = max(trait_scores, key=lambda k: abs(trait_scores[k]))
    standout_value = trait_scores[standout_trait]
    if abs(standout_value) >= 0.3:
        direction_word = "notably higher" if standout_value > 0 else "notably lower"
        personality_line = f"their {standout_trait} stands out as {direction_word} than their other traits"
    else:
        personality_line = "their five personality traits are fairly balanced relative to one another"

    # --- Self-esteem ---
    self_esteem_line = f"self-esteem scored {int(s['rse_score'])} out of {int(s['rse_max'])} ({s['rse_classification'].lower()})"

    # --- Mood and sleep ---
    mood_line = f"a PHQ-9 mood screening score of {s['phq_sum']} ({s['mood_class'].lower()}), with about {s['sleep_hours']:.1f} hours of sleep"

    # --- Digital and social ---
    if s["lone_score"] >= 45.0:
        social_tone = "elevated"
    elif s["lone_score"] >= 30.0:
        social_tone = "moderate"
    else:
        social_tone = "low"
    social_line = f"{social_tone} loneliness signals alongside an internet-use pattern score of {s['addiction_score']:.0f}"

    # --- Burnout ---
    burnout_line = f"a {s['burnout_lvl'].lower()} (index {s['burnout_index']:.1f}) in their work life"

    # --- Assemble the body sentence ---
    body = (
        f"Across the six areas measured, {personality_line}; {self_esteem_line}; "
        f"{mood_line}; {social_line}; and {burnout_line}."
    )

    # --- Opening framing sentence, set by overall severity tier ---
    if s["is_severe_clinical"]:
        opening = (
            "This assessment surfaced some signals worth taking seriously, particularly on the clinical "
            f"screening section, where responses point toward a {s['friendly_condition'].lower()}."
        )
    elif s["anomaly_flag"] or s["is_severe_mood"] or s["is_high_burnout"]:
        opening = (
            "This assessment shows a mixed picture, with a few areas that look more strained than others."
        )
    elif s["is_moderate_mood"] or s["is_elevated_burnout"]:
        opening = (
            "Overall, this looks like a generally steady profile with one or two areas worth keeping an eye on."
        )
    else:
        opening = "Overall, this profile looks stable and well-balanced across the areas measured."

    # --- Closing line: escalate clearly when domain 6 is genuinely severe ---
    recommend_professional_help = bool(s["is_severe_clinical"])
    if s["is_severe_clinical"]:
        closing = (
            "Given these clinical screening results, we'd strongly encourage speaking with a mental health "
            "professional -- a doctor, therapist, or counselor -- for a proper evaluation. This summary is "
            "based on self-reported questionnaire patterns and is not a diagnosis."
        )
    elif s["anomaly_flag"]:
        closing = (
            "A professional check-in could help clarify a couple of these mixed signals, though nothing here "
            "points to an urgent concern on its own."
        )
    else:
        closing = "No immediate concerns stand out, but routine check-ins on mood, sleep, and workload are always worthwhile."

    return {
        "opening": opening,
        "domain_recap": body,
        "closing": closing,
        "recommend_professional_help": recommend_professional_help,
        # Full text retained for any consumer (e.g. the PDF) that wants one
        # ready-to-display paragraph rather than assembling the three parts
        # itself -- both forms are always generated from the same fields,
        # so they can never drift apart from each other.
        "full_text": f"{opening} {body} {closing}"
    }


# Backwards-compatible alias, in case any caller already imports the older name.
evaluate_plain_language_synthesis = generate_plain_language_summary

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
        
    # Execute cross-domain synthesis mapping pass (clinical-register narrative).
    # IMPORTANT: synthesis and enrichment both read from extract_domain_signals()
    # against the RAW domain outputs (before enrichment), so enrichment never
    # has a chance to feed back into the severity logic it depends on.
    global_summary = evaluate_cross_domain_synthesis(final_domain_outputs)

    # Execute plain-language summary pass (person-facing wellbeing snapshot).
    # Reads the exact same extracted signals as global_summary above, so the
    # two will never describe the person's results inconsistently with each
    # other, even though they're written for different audiences.
    plain_summary = generate_plain_language_summary(final_domain_outputs)

    # Enrich each domain's output with severity_tier, a one-line domain_summary,
    # and relative_magnitude on each driver -- computed once here so the
    # frontend (the primary consumer of this JSON) never has to re-derive
    # them from raw numbers itself.
    signals = extract_domain_signals(final_domain_outputs)
    enriched_domain_outputs = enrich_domain_outputs(final_domain_outputs, signals)
    
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
        "schema_version": "4.0",
        "id_no": live_id,
        "age": live_age,
        "sex": live_sex,
        "domain_scores": enriched_domain_outputs,
        "global_synthesis": global_summary,
        "plain_language_summary": plain_summary
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
    print("🧪 Running profile aggregator verification loop against v4.0 updates...")
    test_output = generate_full_profile(sample_payload)
    print("\n✅ Success! Synthesis Result Sub-Keys:")
    print(f"   [Patient Sex] -> {test_output['sex']}")
    print(f"   [Global Synthesis Narrative] -> {test_output['global_synthesis']}")
    print(f"   [Plain Language - Opening] -> {test_output['plain_language_summary']['opening']}")
    print(f"   [Plain Language - Recommend Professional Help] -> {test_output['plain_language_summary']['recommend_professional_help']}")
    print(f"   [Domain 1 severity_tier] -> {test_output['domain_scores']['domain_1_personality']['severity_tier']}")
    print(f"   [Domain 1 domain_summary] -> {test_output['domain_scores']['domain_1_personality']['domain_summary']}")
    print(f"   [Domain 6 severity_tier] -> {test_output['domain_scores']['domain_6_severe_clinical']['severity_tier']}")