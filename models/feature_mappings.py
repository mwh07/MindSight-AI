#!/usr/bin/env python3
"""
MINDSIGHT Psychometric & Clinical Feature Translation Matrix
Maps database column feature identifiers to human-readable clinical strings.
"""

# MAKE SURE THIS EXACT NAME IS SPELLED CORRECTLY:
FEATURE_TRANSLATION_MAP = {
    # --- Domain 1: Big Five Personality Traits ---
    "EXT1": "Social Engagement & Outgoingness",
    "EXT2": "Preference for Solitude / Reserved Nature",
    "EXT3": "Expressive Vivacity & Assertiveness",
    "EST1": "Stress Vulnerability / Anxiety Reactivity",
    "EST2": "Emotional Resilience & Calmness",
    "EST3": "Tendency to Worry / Rumination",
    "AGR1": "Interpersonal Confrontation / Skepticism",
    "AGR2": "Empathy & Social Sympathy",
    "AGR3": "Prosocial Consideration & Altruism",
    "CSN1": "Task Preparedness & Organization",
    "CSN2": "Spontaneous / Low-Structure Task Management",
    "CSN3": "Precision & Attention to Detail",
    "OPN1": "Vivid Imagination & Creativity",
    "OPN2": "Conceptual Complexity & Abstract Thinking",
    "OPN3": "Intellectual Curiosity & Deep Insight",

    # --- Domain 2: Rosenberg Self-Esteem (RSE) ---
    "Q1": "General Self-Worth",
    "Q2": "Acknowledgment of Personal Strengths",
    "Q3": "Inclination to Feel Like a Failure",
    "Q4": "Perceived Competence Relative to Peers",
    "Q5": "Lack of Self-Pride",
    "Q6": "Positive Self-Attitude",
    "Q7": "Overall Self-Satisfaction",
    "Q8": "Deficit in Self-Respect",
    "Q9": "Feelings of Uselessness",
    "Q10": "Negative Self-Evaluation",

    # --- Domain 3: Mood (PHQ-9) & Sleep Inventories ---
    "DPQ010": "Anhedonia / Lack of Pleasure in Activities",
    "DPQ020": "Acute Depressive Mood / Hopelessness",
    "DPQ030": "Sleep Disruption (Insomnia / Hypersomnia)",
    "DPQ040": "Fatigue / Chronic Low Energy",
    "DPQ050": "Appetite Dysregulation (Under / Overeating)",
    "DPQ060": "Negative Self-Evaluation / Guilt",
    "DPQ070": "Cognitive Slowing / Impaired Concentration",
    "DPQ080": "Psychomotor Agitation or Retardation",
    "DPQ090": "Suicidal Ideation / Self-Harm Cognitions",
    "DPQ100": "Functional Impairment in Daily Operations",
    "SLQ300": "Weekday Sleep Initiation Time",
    "SLQ310": "Weekday Wake Time",

    # --- Domain 4: Digital & Social Systems ---
    "IAT1": "Compulsive Online Duration", 
    "IAT2": "Neglect of Domestic Duties for Internet",
    "IAT3": "Preference for Virtual Relationships", 
    "IAT4": "Emotional Distress when Offline",
    "IAT5": "Fear of Missing Out / Digital Dependence", 
    "IAT6": "Academic or Work Output Decline",
    "IAT7": "Defensive Secretiveness Regarding Usage", 
    "IAT8": "Social Over-Allocation Online",
    "IAT9": "Use of Internet as Emotional Escapism", 
    "IAT10": "Preoccupation with Reconnecting",
    "loneliness1": "Perceived Lack of Companionship", 
    "loneliness2": "Feeling Left Out / Excluded",
    "loneliness3": "Social Isolation / Distance from Peers", 
    "loneliness4": "Feeling Superficially Connected",
    "loneliness5": "Deficit in Meaningful Relationships", 
    "loneliness6": "Feeling Misunderstood by Surroundings",

    # --- Domain 5: Occupational Stress & Burnout ---
    "work_hours_per_week": "Weekly Professional Output Hours",
    "meetings_per_day": "Daily Meeting & Communication Load",
    "work_life_balance_score": "Subjective Work-Life Equilibrium",
    "job_satisfaction_score": "Occupational Fulfillment Level",
    "deadline_pressure_score": "Imminent Milestone / Schedule Stress",
    "autonomy_score": "Operational Choice & Task Agency",
    "stress_score": "General Perceived Stress Load",
    "social_support_score": "Workplace Peer & Structural Support",

    # --- Domain 6: Severe Clinical Screening Matrix ---
    "unwanted_thoughts": "Intrusive Cognitive Distortions",
    "repetitive_behaviors": "Compulsive Behavioral Patterns",
    "overthinking": "Maladaptive Rumination Cycle",
    "mind_going_blank": "Acute Cognitive Blocking under Stress",
    "avoidance_social_activity": "Active Social Withdrawal / Avoidance",
    "panic": "Somatic Panic Sensations",
    "hypervigilance": "Autonomic Hyperarousal & Guardedness"
}