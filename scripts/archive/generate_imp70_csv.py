# generate_imp68_csv.py
import os
import pandas as pd

def build_imp68_questionnaire_csv():
    print("[IMP-70] Compiling Mathematically Optimized Psychometric Grid...")
    
    questions = []
    
    # -------------------------------------------------------------
    # FOUNDATIONAL DEMOGRAPHICS (2 Anchor Items)
    # -------------------------------------------------------------
    questions.append({
        "Question_ID": "age", "Domain": "Demographic Baseline", 
        "Question_Text": "What is your current chronological age in years?", 
        "Response_Type": "Continuous Numeric Integer Entry", "Scoring_Direction": "Standard"
    })
    questions.append({
        "Question_ID": "sex", "Domain": "Demographic Baseline", 
        "Question_Text": "What is your biological sex assigned at birth?", 
        "Response_Type": "Discrete Selection (0 = Male, 1 = Female)", "Scoring_Direction": "Standard"
    })
    
    # -------------------------------------------------------------
    # DOMAIN 1: Core Personality (Big Five Scale Matrix - 10 Items)
    # -------------------------------------------------------------
    d1_items = [
        ("EXT1", "I am the life of the party.", "1-5 Likert Scale (1=Disagree, 5=Agree)", "Standard"),
        ("EXT2", "I don't talk a lot.", "1-5 Likert Scale (1=Disagree, 5=Agree)", "Reverse"),
        ("EST1", "I get stressed out easily.", "1-5 Likert Scale (1=Disagree, 5=Agree)", "Standard"),
        ("EST2", "I am relaxed most of the time.", "1-5 Likert Scale (1=Disagree, 5=Agree)", "Reverse"),
        ("AGR1", "I feel little concern for others.", "1-5 Likert Scale (1=Disagree, 5=Agree)", "Reverse"),
        ("AGR2", "I am interested in people.", "1-5 Likert Scale (1=Disagree, 5=Agree)", "Standard"),
        ("CSN1", "I am always prepared.", "1-5 Likert Scale (1=Disagree, 5=Agree)", "Standard"),
        ("CSN2", "I leave my duties undone.", "1-5 Likert Scale (1=Disagree, 5=Agree)", "Reverse"),
        ("OPN1", "I have a rich vocabulary.", "1-5 Likert Scale (1=Disagree, 5=Agree)", "Standard"),
        ("OPN2", "I have difficulty understanding abstract ideas.", "1-5 Likert Scale (1=Disagree, 5=Agree)", "Reverse"),
    ]
    for q_id, text, r_type, direction in d1_items:
        questions.append({"Question_ID": q_id, "Domain": "Domain 1: Personality", "Question_Text": text, "Response_Type": r_type, "Scoring_Direction": direction})

    # -------------------------------------------------------------
    # DOMAIN 2: Global Self-Esteem (Standardized Rosenberg Matrix - 10 Items)
    # Fix: Enforced 4-point scale to remove neutral midpoint bias
    # -------------------------------------------------------------
    d2_items = [
        ("Q1", "I feel that I am a person of worth, at least on an equal plane with others.", "1-4 Likert (1=Strongly Disagree, 4=Strongly Agree)", "Standard"),
        ("Q2", "I feel that I have a number of good qualities.", "1-4 Likert (1=Strongly Disagree, 4=Strongly Agree)", "Standard"),
        ("Q3", "All in all, I am inclined to feel that I am a failure.", "1-4 Likert (1=Strongly Disagree, 4=Strongly Agree)", "Reverse"),
        ("Q4", "I am able to do things as well as most other people.", "1-4 Likert (1=Strongly Disagree, 4=Strongly Agree)", "Standard"),
        ("Q5", "I feel I do not have much to be proud of.", "1-4 Likert (1=Strongly Disagree, 4=Strongly Agree)", "Reverse"),
        ("Q6", "I take a positive attitude toward myself.", "1-4 Likert (1=Strongly Disagree, 4=Strongly Agree)", "Standard"),
        ("Q7", "On the whole, I am satisfied with myself.", "1-4 Likert (1=Strongly Disagree, 4=Strongly Agree)", "Standard"),
        ("Q8", "I wish I could have more respect for myself.", "1-4 Likert (1=Strongly Disagree, 4=Strongly Agree)", "Reverse"),
        ("Q9", "I certainly feel useless at times.", "1-4 Likert (1=Strongly Disagree, 4=Strongly Agree)", "Reverse"),
        ("Q10", "At times I think I am no good at all.", "1-4 Likert (1=Strongly Disagree, 4=Strongly Agree)", "Reverse"),
    ]
    for q_id, text, r_type, direction in d2_items:
        questions.append({"Question_ID": q_id, "Domain": "Domain 2: Self-Esteem", "Question_Text": text, "Response_Type": r_type, "Scoring_Direction": direction})

    # -------------------------------------------------------------
    # DOMAIN 3: Mood & Circadian Strain (NHANES Block - 11 Items)
    # Fix: Clear notation that time fields require cyclic transformations
    # -------------------------------------------------------------
    d3_items = [
        ("DPQ010", "Little interest or pleasure in doing things over the past 2 weeks.", "0-3 Frequency Scale (0=Not at all, 3=Nearly every day)", "Standard"),
        ("DPQ020", "Feeling down, depressed, or hopeless over the past 2 weeks.", "0-3 Frequency Scale (0=Not at all, 3=Nearly every day)", "Standard"),
        ("DPQ030", "Trouble falling or staying asleep, or sleeping too much.", "0-3 Frequency Scale (0=Not at all, 3=Nearly every day)", "Standard"),
        ("DPQ040", "Feeling tired or having little energy.", "0-3 Frequency Scale (0=Not at all, 3=Nearly every day)", "Standard"),
        ("DPQ050", "Poor appetite or overeating patterns.", "0-3 Frequency Scale (0=Not at all, 3=Nearly every day)", "Standard"),
        ("DPQ060", "Feeling bad about yourself, or that you are a failure.", "0-3 Frequency Scale (0=Not at all, 3=Nearly every day)", "Standard"),
        ("DPQ070", "Trouble concentrating on basic tasks (e.g., reading or watching TV).", "0-3 Frequency Scale (0=Not at all, 3=Nearly every day)", "Standard"),
        ("DPQ080", "Moving/speaking so slowly or quickly that others noticed.", "0-3 Frequency Scale (0=Not at all, 3=Nearly every day)", "Standard"),
        ("DPQ090", "Thoughts that you would be better off dead, or of hurting yourself.", "0-3 Frequency Scale (0=Not at all, 3=Nearly every day)", "Standard"),
        ("SLQ300", "What time do you usually go to bed on workdays/weekdays?", "24hr Time String (HH:MM) [Cyclic Modeled]", "Standard"),
        ("SLQ310", "What time do you usually wake up on workdays/weekdays?", "24hr Time String (HH:MM) [Cyclic Modeled]", "Standard"),
    ]
    for q_id, text, r_type, direction in d3_items:
        questions.append({"Question_ID": q_id, "Domain": "Domain 3: Mood and Sleep", "Question_Text": text, "Response_Type": r_type, "Scoring_Direction": direction})

    # -------------------------------------------------------------
    # DOMAIN 4: Digital Dependency & Isolation (IAT/UCLA - 20 Items)
    # -------------------------------------------------------------
    iat_texts = [
        "", "How often do you stay online longer than you originally intended?",
        "How often do you neglect household/daily tasks to spend more time online?",
        "How often do you prefer the excitement of the internet over real-world relationships?",
        "How often do you form new digital relationships with online users?",
        "How often do close connections complain to you about your internet usage?",
        "How often do your grades, productivity, or work responsibilities suffer from screen time?",
        "How often do you check electronic communication channels before executing required tasks?",
        "How often does your job performance diminish due to distracting online activity?",
        "How often do you become defensive or highly secretive regarding your browser behaviors?",
        "How often do you mask disturbing real-life thoughts using soothing internet media?"
    ]
    for i in range(1, 11):
        questions.append({"Question_ID": f"IAT{i}", "Domain": "Domain 4: Digital and Social Risk", "Question_Text": iat_texts[i], "Response_Type": "1-5 Frequency Scale (1=Rarely, 5=Always)", "Scoring_Direction": "Standard"})

    ucla_texts = [
        "", "How often do you feel completely 'in tune' with the people around you?",
        "How often do you feel that you severely lack deep companionship?",
        "How often do you feel that there is truly no one you can turn to?",
        "How often do you feel isolated and entirely alone?",
        "How often do you feel strongly integrated as part of a group of friends?",
        "How often do you feel that you have a significant amount in common with those around you?",
        "How often do you feel that you are no longer close or connected to anyone?",
        "How often do you feel that your interests or ideas are completely unshared by peers?",
        "How often do you feel naturally outgoing, expressive, and friendly?",
        "How often do you feel emotionally close to people in your immediate environment?"
    ]
    for i in range(1, 11):
        direction = "Reverse" if i in [1, 5, 6, 9, 10] else "Standard"
        questions.append({"Question_ID": f"loneliness{i}", "Domain": "Domain 4: Digital and Social Risk", "Question_Text": ucla_texts[i], "Response_Type": "1-4 Frequency Scale (1=Never, 4=Often)", "Scoring_Direction": direction})

    # -------------------------------------------------------------
    # DOMAIN 5: Occupational Burnout & Fatigue Indicators (8 Items)
    # Fix: Compressed subjective metrics into standardized, anchored scales
    # -------------------------------------------------------------
    d5_items = [
        ("work_hours_per_week", "Average total number of active working hours per week.", "Continuous Numeric Integer Entry", "Standard"),
        ("meetings_per_day", "Average total number of corporate meetings attended daily.", "Continuous Float/Numeric Value", "Standard"),
        ("work_life_balance_score", "Rate your subjective overall work-life balance satisfaction level.", "1-5 Anchored Scale (1=Highly Disrupted, 5=Perfectly Balanced)", "Reverse"),
        ("job_satisfaction_score", "Rate your subjective professional and career fulfillment level.", "1-5 Anchored Scale (1=Completely Unfulfilled, 5=Highly Fulfilled)", "Reverse"),
        ("deadline_pressure_score", "Rate the frequency/severity of time constraints and urgency pressures.", "1-5 Intensity Scale (1=Low/Rare, 5=Extreme/Constant)", "Standard"),
        ("autonomy_score", "Rate your level of control and decision freedom over execution tasks.", "1-5 Intensity Scale (1=Micro-managed, 5=Total Freedom)", "Reverse"),
        ("stress_score", "Rate the baseline cumulative stress experienced over the past month.", "1-5 Intensity Scale (1=Low/Manageable, 5=Overwhelming/Severe)", "Standard"),
        ("social_support_score", "Rate the perceived strength of your immediate workplace support framework.", "1-5 Strength Scale (1=Isolated, 5=Highly Supportive)", "Reverse"),
    ]
    for q_id, text, r_type, direction in d5_items:
        questions.append({"Question_ID": q_id, "Domain": "Domain 5: Occupational Burnout", "Question_Text": text, "Response_Type": r_type, "Scoring_Direction": direction})

    # -------------------------------------------------------------
    # DOMAIN 6: Severe Clinical & Somatic Anomalies (9 Items)
    # Fix: Replaced flat binaries with 0-4 frequency metrics for the Isolation Forest
    # -------------------------------------------------------------
    d6_items = [
        ("unwanted_thoughts", "Experiencing recurrent, distressing, intrusive thoughts or images.", "0-4 Frequency Scale (0=Never, 4=Constant/Severe)", "Standard"),
        ("repetitive_behaviors", "Feeling compelled to repeat physical actions or rigid mental rituals.", "0-4 Frequency Scale (0=Never, 4=Constant/Severe)", "Standard"),
        ("overthinking", "Excessive rumination over insignificant daily micro-interactions.", "0-4 Frequency Scale (0=Never, 4=Constant/Severe)", "Standard"),
        ("mind_going_blank", "Cognitive paralysis or loss of memory continuity during stress situations.", "0-4 Frequency Scale (0=Never, 4=Constant/Severe)", "Standard"),
        ("avoidance_social_activity", "Active avoidance of social events, crowds, or public areas out of distress.", "0-4 Frequency Scale (0=Never, 4=Constant/Severe)", "Standard"),
        ("panic", "Sudden, unprovoked surges of overwhelming physical terror or heart palpitations.", "0-4 Frequency Scale (0=Never, 4=Constant/Severe)", "Standard"),
        ("hypervigilance", "Continuous high-alert monitoring of surroundings to guard against threats.", "0-4 Frequency Scale (0=Never, 4=Constant/Severe)", "Standard"),
        ("severe_mood_swings", "Rapid, unprovoked emotional transitions detached from circumstances.", "0-4 Frequency Scale (0=Never, 4=Constant/Severe)", "Standard"),
        ("dissociation_episodes", "Experiencing detached feelings from your body, identity, or reality.", "0-4 Frequency Scale (0=Never, 4=Constant/Severe)", "Standard"),
    ]
    for q_id, text, r_type, direction in d6_items:
        questions.append({"Question_ID": q_id, "Domain": "Domain 6: Severe Clinical", "Question_Text": text, "Response_Type": r_type, "Scoring_Direction": direction})

    # Structural assertions to guarantee matrix consistency
    assert len(questions) == 70, f"Error: Inventory size mismatch. Expected 70 nodes, found {len(questions)}"
    
    df = pd.DataFrame(questions)
    df.to_csv("imp68_questionnaire.csv", index=False)
    print("🎉 Success! Optimization complete. 'imp68_questionnaire.csv' written to root.")

if __name__ == "__main__":
    build_imp68_questionnaire_csv()