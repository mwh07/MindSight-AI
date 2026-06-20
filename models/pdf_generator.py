#!/usr/bin/env python3
"""
MINDSIGHT Single-Page PDF Report Generator Engine (v3.8 - Production Calibrated)
Renders multi-domain placements, graphic reference scales, and cross-domain clinical synthesis
completely onto exactly one page using highly defensive layout metrics. Aligned with v3.8 Inference.
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

try:
    from models.feature_mappings import FEATURE_TRANSLATION_MAP
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from models.feature_mappings import FEATURE_TRANSLATION_MAP
    except ImportError:
        FEATURE_TRANSLATION_MAP = {}

# Production fallback dictionary guaranteeing clean, human-readable labels across all evaluation facets
LOCAL_TRANSLATION_FALLBACK = {
    # Domain 1 Mappings
    "extraversion": "Extraversion Vector",
    "emotional_stability": "Emotional Stability Vector",
    "agreeableness": "Agreeableness Vector",
    "conscientiousness": "Conscientiousness Vector",
    "openness": "Openness Vector",
    
    # Domain 3 Mappings
    "PHQ_Core": "Symptom Burden Summation",
    "Sleep_Duration": "Calculated Sleep Duration",

    # Domain 5: Occupational Burnout Features
    "age": "Age",
    "work_hours_per_week": "Weekly Work Hours",
    "meetings_per_day": "Daily Meetings Count",
    "work_life_balance_score": "Work-Life Balance Rating",
    "job_satisfaction_score": "Job Satisfaction Index",
    "deadline_pressure_score": "Perceived Deadline Pressure",
    "autonomy_score": "Workplace Autonomy",
    "stress_score": "Reported Stress Level",
    "social_support_score": "Social Support Index",
    "gender_Male": "Gender: Male",
    "gender_Female": "Gender: Female",
    "gender_Non-binary": "Gender: Non-binary",
    "gender_Prefer not to say": "Gender: Undisclosed",
    
    # Domain 6: Severe Clinical Screening Features
    "unwanted_thoughts": "Intrusive Thoughts",
    "repetitve_behaviors": "Repetitive Behaviors",
    "overthinking": "Cognitive Rumination",
    "mind_going_blank": "Cognitive Blocking",
    "avoidance_of_social_activity": "Social Withdrawal & Avoidance",
    "panic": "Panic Symptoms",
    "hypervigilance": "Hypervigilance & Physical Arousal"
}

TITLE_OVERRIDES = {
    "domain_1_personality": "Domain 1: Personality Vectors",
    "domain_2_self_esteem": "Domain 2: Self-Esteem Matrix",
    "domain_3_mood_sleep": "Domain 3: Mood & Sleep Dynamics",
    "domain_3_mood_and_sleep": "Domain 3: Mood & Sleep Dynamics",
    "domain_4_digital_and_social": "Domain 4: Digital & Social Dynamics",
    "domain_4_multitask": "Domain 4: Digital & Social Dynamics",
    "domain_5_occupational_burnout": "Domain 5: Occupational Burnout Index",
    "domain_6_severe_clinical": "Domain 6: Severe Clinical Screening"
}


def create_visual_scale(percentage, width=140, filled_color='#2C5282'):
    """Generates a compact linear graphic scale to show domain placements visually."""
    try:
        percentage = max(0.0, min(100.0, float(percentage)))
    except (ValueError, TypeError):
        percentage = 50.0

    w_filled = max(0.1, (percentage / 100.0) * width)
    w_empty = max(0.1, width - w_filled)
    
    t = Table([["", ""]], colWidths=[w_filled, w_empty], rowHeights=[5])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), colors.HexColor(filled_color)),
        ('BACKGROUND', (1,0), (1,0), colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    return t


def compile_pdf_report(profile_data, target_pdf_path):
    """
    Transforms unified profile metadata, demographic descriptors, and visual metrics 
    into a high-contrast corporate-grade single-page diagnostic report.
    """
    margin = 36 
    doc = SimpleDocTemplate(
        target_pdf_path,
        pagesize=letter,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=20, leading=24,
        textColor=colors.HexColor('#1A365D')
    )
    section_heading = ParagraphStyle(
        'SecHeading', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=10, leading=13,
        textColor=colors.HexColor('#2C5282')
    )
    body_text = ParagraphStyle(
        'BodyTxt', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8.5, leading=11,
        textColor=colors.HexColor('#2D3748')
    )
    attrib_text = ParagraphStyle(
        'AttribTxt', parent=styles['Normal'],
        fontName='Helvetica-Oblique', fontSize=7.5, leading=9.5,
        textColor=colors.HexColor('#4A5568')
    )
    synthesis_style = ParagraphStyle(
        'SynthTxt', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8.5, leading=13.0,
        textColor=colors.HexColor('#1A202C')
    )

    story = []
    
    # --- HEADER BLOCK ---
    story.append(Paragraph("MINDSIGHT CLINICAL ASSESSMENT REPORT", title_style))
    story.append(Paragraph("System Architecture Version 3.8 · Unified Multi-Domain Diagnostic Profile", attrib_text))
    story.append(Spacer(1, 6))
    
    # --- BASIC DETAILS BLOCK ---
    id_no = profile_data.get("id_no", profile_data.get("metadata", {}).get("id_no", "MS-2026-X"))
    age = profile_data.get("age", profile_data.get("metadata", {}).get("age", "N/A"))
    sex = profile_data.get("sex", profile_data.get("metadata", {}).get("sex", "N/A"))
    
    details_data = [[
        Paragraph(f"<b>Patient ID:</b> {id_no}", body_text),
        Paragraph(f"<b>Age:</b> {age} yrs", body_text),
        Paragraph(f"<b>Sex / Demographic Cohort:</b> {sex}", body_text)
    ]]
    
    details_table = Table(details_data, colWidths=[220, 80, 240])
    details_table.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F7FAFC')),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(details_table)
    story.append(Spacer(1, 8))
    
    # --- DOMAINS GRID CONSOLE CONSTRUCTION ---
    scores = profile_data.get("domain_scores", {})
    domain_keys = list(scores.keys())
    grid_cells = []
    
    for d_key in domain_keys:
        d_data = scores[d_key]
        display_title = TITLE_OVERRIDES.get(d_key, d_key.replace("domain_", "").replace("_", " ").title())
        
        bar_color = '#2C5282' 
        scale_pct = 50.0
        placement_summary = "Data Unspecified"
        
        placement = d_data.get("placement", {})
        
        if d_key == "domain_1_personality":
            ext = placement.get('extraversion', 0.0)
            est = placement.get('emotional_stability', 0.0)
            agr = placement.get('agreeableness', 0.0)
            csn = placement.get('conscientiousness', 0.0)
            opn = placement.get('openness', 0.0)
            # FIX: Restored full 5-dimensional vector array projection to layout engine
            placement_summary = f"EXT: {ext} | EST: {est} | AGR: {agr} | CSN: {csn} | OPN: {opn}"
            try:
                scale_pct = ((float(est) + 4.0) / 8.0) * 100.0
            except (ValueError, TypeError):
                scale_pct = 50.0
            
        elif d_key == "domain_2_self_esteem":
            # FIX: Realigned to read strict contract signatures from runtime scoring
            score = placement.get('score', placement.get('raw_score', 0))
            classification = placement.get('classification', 'Normal Profile')
            placement_summary = f"RSE Score: {score}/30 ({classification})"
            try:
                scale_pct = (float(score) / 30.0) * 100.0
            except (ValueError, TypeError):
                scale_pct = 50.0
            if scale_pct <= 45.0 or "Low" in classification: 
                bar_color = '#C53030'
            
        # =========================================================================
        # FIXED: Harmonized Domain 3 color space to match universal multi-domain color semantics
        # =========================================================================
        elif d_key in ["domain_3_mood_sleep", "domain_3_mood_and_sleep"]:
            # Bypassed independent overrides to parse genuine summation values
            phq_sum = placement.get('phq9_sum', placement.get('deterministic_phq9_sum', 0))
            sev_class = placement.get('severity_label', placement.get('assigned_severity_class', 'Unknown'))
            sleep_duration = placement.get('calculated_sleep_duration_hours', None)
            
            sleep_str = f" | Sleep: {sleep_duration}h" if sleep_duration is not None else ""
            placement_summary = f"PHQ-9 Sum: {phq_sum} ({sev_class}){sleep_str}"
            try:
                scale_pct = (float(phq_sum) / 27.0) * 100.0
            except (ValueError, TypeError):
                scale_pct = 50.0
                
            # Calibrate 4-tier standardized clinical bucket colors
            if int(phq_sum) >= 15: 
                bar_color = '#C53030'  # Severe / Moderately Severe (Red)
            elif int(phq_sum) >= 10: 
                bar_color = '#DD6B20'  # Moderate (Orange)
            elif int(phq_sum) >= 5:
                bar_color = '#D69E2E'  # Mild / Sub-clinical (Yellow)
            else:
                bar_color = '#2F855A'  # Minimal / Baseline Healthy (Green)
            
        elif d_key in ["domain_4_digital_and_social", "domain_4_multitask"]:
            # FIX: Supports elastic routing parameters for custom Tree Explainer layers
            lone = placement.get('loneliness_score', placement.get('predicted_total_loneliness', 0.0))
            addict = placement.get('predicted_total_internet_addiction', 0.0)
            classification = placement.get('classification', '')
            class_suffix = f" [{classification}]" if classification else ""
            
            placement_summary = f"Loneliness Index: {lone}{class_suffix}" if addict == 0.0 else f"Addiction: {addict} | Loneliness: {lone}"
            try:
                scale_pct = (float(lone) / 80.0) * 100.0
            except (ValueError, TypeError):
                scale_pct = 50.0
            if scale_pct >= 65.0: 
                bar_color = '#DD6B20'
            
        elif d_key == "domain_5_occupational_burnout":
            # FIX: Mapped explicit continuous variables from gradient boosting registers
            score = placement.get('burnout_index', placement.get('burnout_score', 0.0))
            lvl = placement.get('burnout_tier_label', placement.get('burnout_level', 'Low'))
            placement_summary = f"Index: {score} ({lvl})"
            
            lvl_map = {
                "Low": 15.0, "Moderate": 45.0, "High": 75.0, "Severe": 95.0,
                "Low / Controlled Engagement Profile": 15.0,
                "Moderate Burnout Profile": 45.0,
                "High Burnout Risk": 75.0,
                "Severe Burnout Indication": 95.0
            }
            scale_pct = lvl_map.get(lvl, 50.0)
            
            if "Low" in lvl:
                bar_color = '#2F855A'
            elif "Moderate" in lvl:
                bar_color = '#DD6B20'
            else:
                bar_color = '#C53030'
            
        elif d_key == "domain_6_severe_clinical":
            is_anomaly = bool(placement.get('anomaly_review_flag', False))
            anomaly_str = "⚠️ ATYPICAL" if is_anomaly else "Standard"
            
            # FIX: Inherits live runtime-generated strings directly to prevent semantic drift
            resolved_label = placement.get('predicted_condition_label', None)
            raw_condition = placement.get('predicted_condition_code', placement.get('predicted_condition', 0))
            try:
                cond_idx = int(raw_condition)
            except (ValueError, TypeError):
                cond_idx = 0
                
            if not resolved_label:
                clinical_labels = {
                    0: "Baseline Healthy Profile",
                    1: "Mild Symptomatic Profile",
                    2: "Moderate Distress Phenotype",
                    3: "Severe Clinical Screening Indication"
                }
                resolved_label = clinical_labels.get(cond_idx, f"Phenotype Code {cond_idx}")
                
            placement_summary = f"{resolved_label} [{anomaly_str}]"
            
            severity_scale_map = {0: 15.0, 1: 35.0, 2: 55.0, 3: 75.0, 4: 95.0}
            scale_pct = severity_scale_map.get(cond_idx, 15.0)
            if is_anomaly:
                scale_pct = max(scale_pct, 85.0)
                
            if cond_idx >= 3 or is_anomaly:
                bar_color = '#C53030'
            elif cond_idx == 2:
                bar_color = '#DD6B20'
            elif cond_idx == 1:
                bar_color = '#D69E2E'
            else:
                bar_color = '#2F855A'
            
        top_contribs = d_data.get("top_contributors", [])
        contrib_strings = []
        for c in top_contribs:
            feat_name = c.get('feature', 'unknown_feature')
            friendly_label = FEATURE_TRANSLATION_MAP.get(feat_name, LOCAL_TRANSLATION_FALLBACK.get(feat_name, feat_name))
            direction = c.get('direction', '')
            contrib_val = c.get('contribution', 0.0)
            contrib_strings.append(f"{friendly_label} ({direction}{contrib_val})")
            
        drivers_line = "Drivers: " + (", ".join(contrib_strings) if contrib_strings else "None Identified")
        graphic_scale_flowable = create_visual_scale(scale_pct, width=246, filled_color=bar_color)
        
        cell_elements = [
            Paragraph(display_title, section_heading),
            Spacer(1, 2),
            Paragraph(f"<b>Placement:</b> {placement_summary}", body_text),
            Spacer(1, 2),
            graphic_scale_flowable,
            Spacer(1, 2),
            Paragraph(drivers_line, attrib_text),
            Spacer(1, 1)
        ]
        grid_cells.append(cell_elements)
        
    # Chunk cells safely into standard 2-column matrix formatting
    grid_data = []
    for i in range(0, len(grid_cells), 2):
        row = grid_cells[i:i+2]
        if len(row) == 1:
            row.append("")  
        grid_data.append(row)
    
    if grid_data:
        matrix_table = Table(grid_data, colWidths=[266, 266])
        matrix_table.setStyle(TableStyle([
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F7FAFC')),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ]))
        story.append(matrix_table)
        story.append(Spacer(1, 10))
    
    # --- GLOBAL SYSTEM SYNTHESIS BLOCK ---
    story.append(Paragraph("SYNTHESIZED GLOBAL PROFILE OVERVIEW", section_heading))
    story.append(Spacer(1, 4))
    
    synthesis_narrative_text = profile_data.get("global_synthesis", "No synthesis provided.")
    synth_elements = [
        Paragraph(synthesis_narrative_text, synthesis_style)
    ]
    
    synthesis_box = Table([[synth_elements]], colWidths=[540])
    synthesis_box.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#2C5282')),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#EBF8FF')),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(synthesis_box)
    
    doc.build(story)


if __name__ == "__main__":
    mock_payload = {
        "id_no": "MS-2026-9042",
        "age": 31,
        "sex": "Female",
        "domain_scores": {
            "domain_1_personality": {"placement": {"extraversion": 0.376, "emotional_stability": -0.095, "agreeableness": 0.16, "conscientiousness": 1.12, "openness": -0.45}, "top_contributors": []},
            "domain_2_self_esteem": {"placement": {"score": 25, "classification": "High Self-Esteem"}, "top_contributors": []},
            "domain_3_mood_sleep": {"placement": {"phq9_sum": 17, "severity_label": "Moderately Severe Depression"}, "top_contributors": []},
            "domain_4_multitask": {"placement": {"loneliness_score": 50.32}, "top_contributors": []},
            "domain_5_occupational_burnout": {"placement": {"burnout_index": 7.084, "burnout_tier_label": "High Burnout Risk"}, "top_contributors": []},
            "domain_6_severe_clinical": {"placement": {"predicted_condition_code": 1, "predicted_condition_label": "Mild Symptomatic Profile", "anomaly_review_flag": False}, "top_contributors": []}
        },
        "global_synthesis": "COMPLEX PROFILE EVALUATION: Layout structure stabilized."
    }
    print("📋 Testing single page PDF compiler tool output against accurate engine structure...")
    compile_pdf_report(mock_payload, "test_mindsight_report.pdf")
    print("✨ Production alignment verification complete!")