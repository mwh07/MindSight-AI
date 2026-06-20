#!/usr/bin/env python3
import os
import sys
import csv

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def prompt_validated_input(prompt_text, val_type, valid_range=None, choices=None):
    """
    Ensures user entry strictly conforms to psychometric scale rules.
    """
    while True:
        try:
            user_input = input(prompt_text).strip()
            if not user_input:
                print("⚠️ This field is mandatory. Value cannot be left blank.")
                continue
                
            if choices:
                # Handle categorical text options mapping to integer values
                matched = [k for k in choices.keys() if k.lower() == user_input.lower()]
                if matched:
                    return choices[matched[0]]
                print(f"❌ Invalid entry. Must be exactly one of: {list(choices.keys())}")
                continue
            
            # Numeric parsing validation
            val = val_type(user_input)
            if valid_range:
                min_v, max_v = valid_range
                if not (min_v <= val <= max_v):
                    print(f"❌ Value out of bounds! Must be between {min_v} and {max_v}.")
                    continue
            return val
        except ValueError:
            print(f"❌ Invalid format. Please enter a valid {val_type.__name__} value.")

def prompt_validated_time(prompt_text):
    """
    Ensures the sleep inputs strictly follow the HH:MM time format string.
    """
    while True:
        user_input = input(prompt_text).strip()
        if len(user_input) == 5 and ":" in user_input:
            parts = user_input.split(":")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                hours, minutes = int(parts[0]), int(parts[1])
                if 0 <= hours <= 23 and 0 <= minutes <= 59:
                    return user_input
        print("❌ Invalid time format. Please use 24-hour HH:MM format (e.g., 22:30 or 07:15).")

def run_imp70_assessment():
    clear_terminal()
    print("=" * 75)
    print("         IMP-70 COMPREHENSIVE INTEGRATED CLINICAL BATTERY SUITE")
    print("=" * 75)
    print("Instructions: Please respond to each item honestly.")
    print("Your data will be structured and logged into the 'tests/' directory.")
    input("\nPress [ENTER] to initialize the assessment window...")
    
    response_payload = {}

    # -------------------------------------------------------------
    # SECTION 1: EXOGENOUS DEMOGRAPHICS (2 Items)
    # -------------------------------------------------------------
    clear_terminal()
    print("--- SECTION 1: EXOGENOUS DEMOGRAPHICS ---")
    response_payload['age'] = prompt_validated_input(
        "👉 What is your current chronological age in years?\n[Continuous Numeric Integer Entry]\nAge: ", 
        val_type=int, valid_range=(13, 100)
    )
    print()
    
    gender_choices = {"Male": 0, "Female": 1}
    response_payload['gender'] = prompt_validated_input(
        "👉 What is your biological sex assigned at birth?\n[Discrete Selection: Male / Female]\nChoice: ", 
        val_type=int, choices=gender_choices
    )

    # -------------------------------------------------------------
    # SECTION 2: DOMAIN 1 - PERSONALITY BASELINE (15 Items)
    # -------------------------------------------------------------
    clear_terminal()
    print("--- SECTION 2: DOMAIN 1 - PERSONALITY BASELINE ---")
    print("Scale: 1 (Strongly Disagree), 2 (Disagree), 3 (Neutral), 4 (Agree), 5 (Strongly Agree)\n")
    
    personality_items = [
        ("EXT1", "I am the life of the party."),
        ("EXT2", "I don't talk a lot."),
        ("EXT3", "I feel comfortable around people."),
        ("EST1", "I get stressed out easily."),
        ("EST2", "I am relaxed most of the time."),
        ("EST3", "I worry about things frequently."),
        ("AGR1", "I feel little concern for others."),
        ("AGR2", "I am interested in people."),
        ("AGR3", "I feel sympathy for others' feelings."),
        ("CSN1", "I am always prepared."),
        ("CSN2", "I leave my duties undone."),
        ("CSN3", "I follow a planned schedule."),
        ("OPN1", "I have a rich vocabulary."),
        ("OPN2", "I have difficulty understanding abstract ideas."),
        ("OPN3", "I have a vivid and active imagination.")
    ]
    
    for item_code, question in personality_items:
        response_payload[item_code] = prompt_validated_input(
            f"❓ {item_code} - {question}\n👉 Choice (1-5): ", val_type=int, valid_range=(1, 5)
        )
        print()

    # -------------------------------------------------------------
    # SECTION 3: DOMAIN 2 - SELF-ESTEEM ASSESSMENT (10 Items)
    # -------------------------------------------------------------
    clear_terminal()
    print("--- SECTION 3: DOMAIN 2 - SELF-ESTEEM ASSESSMENT (ROSENBERG RSES) ---")
    print("Scale: 1 (Strongly Disagree), 2 (Disagree), 3 (Neutral), 4 (Agree), 5 (Strongly Agree)\n")
    
    rses_items = [
        ("Q1", "I feel that I am a person of worth, at least on an equal plane with others."),
        ("Q2", "I feel that I have a number of good qualities."),
        ("Q3", "All in all, I am inclined to feel that I am a failure."),
        ("Q4", "I am able to do things as well as most other people."),
        ("Q5", "I feel I do not have much to be proud of."),
        ("Q6", "I take a positive attitude toward myself."),
        ("Q7", "On the whole, I am satisfied with myself."),
        ("Q8", "I wish I could have more respect for myself."),
        ("Q9", "I certainly feel useless at times."),
        ("Q10", "At times I think I am no good at all.")
    ]
    
    for item_code, question in rses_items:
        response_payload[item_code] = prompt_validated_input(
            f"❓ {item_code} - {question}\n👉 Choice (1-5): ", val_type=int, valid_range=(1, 5)
        )
        print()

    # -------------------------------------------------------------
    # SECTION 4: DOMAIN 3 - MOOD & SLEEP ARCHITECTURE (12 Items)
    # -------------------------------------------------------------
    clear_terminal()
    print("--- SECTION 4: DOMAIN 3 - MOOD & DEPRESSIVE SYMPTOMS ---")
    print("Over the past 2 weeks, how often have you been bothered by any of the following problems?")
    print("Scale: 0 (Not at all), 1 (Several days), 2 (More than half the days), 3 (Nearly every day)\n")
    
    phq_items = [
        ("DPQ010", "Little interest or pleasure in doing things over the past 2 weeks."),
        ("DPQ020", "Feeling down, depressed, or hopeless over the past 2 weeks."),
        ("DPQ030", "Trouble falling or staying asleep, or sleeping too much."),
        ("DPQ040", "Feeling tired or having little energy."),
        ("DPQ050", "Poor appetite or overeating patterns."),
        ("DPQ060", "Feeling bad about yourself, or that you are a failure."),
        ("DPQ070", "Trouble concentrating on basic tasks (e.g., reading or watching TV)."),
        ("DPQ080", "Moving/speaking so slowly or quickly that others noticed."),
        ("DPQ090", "Thoughts that you would be better off dead, or of hurting yourself."),
        ("DPQ100", "How difficult have these problems made it for you to work, handle things at home, or get along with people?")
    ]
    
    for item_code, question in phq_items:
        response_payload[item_code] = prompt_validated_input(
            f"❓ {item_code} - {question}\n👉 Choice (0-3): ", val_type=int, valid_range=(0, 3)
        )
        print()

    print("\n--- SLEEP METRICS ---")
    response_payload['SLQ300'] = prompt_validated_time("❓ SLQ300 - What time do you usually go to bed on workdays/weekdays? (24 hr clock used)\n👉 Time (HH:MM): ")
    print()
    response_payload['SLQ310'] = prompt_validated_time("❓ SLQ310 - What time do you usually wake up on workdays/weekdays? (24 hr clock used)\n👉 Time (HH:MM): ")

    # -------------------------------------------------------------
    # SECTION 5: DOMAIN 4 - DIGITAL DEPENDENCY & LONELINESS (16 Items)
    # -------------------------------------------------------------
    clear_terminal()
    print("--- SECTION 5: DOMAIN 4 - DIGITAL & SOCIAL RISK (INTERNET ADDICTION) ---")
    print("Scale: 1 (Rarely), 2 (Seldom), 3 (Often), 4 (Very Often), 5 (Always)\n")
    
    iat_items = [
        ("IAT1", "How often do you stay online longer than you originally intended?"),
        ("IAT2", "How often do you neglect household/daily tasks to spend more time online?"),
        ("IAT3", "How often do you prefer the excitement of the internet over real-world relationships?"),
        ("IAT4", "How often do you form new digital relationships with online users?"),
        ("IAT5", "How often do close connections complain to you about your internet usage?"),
        ("IAT6", "How often do your grades, productivity, or work responsibilities suffer from screen time?"),
        ("IAT7", "How often do you check electronic communication channels before executing required tasks?"),
        ("IAT8", "How often does your job performance diminish due to distracting online activity?"),
        ("IAT9", "How often do you become defensive or highly secretive regarding your browser behaviors?"),
        ("IAT10", "How often do you mask disturbing real-life thoughts using soothing internet media?")
    ]
    
    for item_code, question in iat_items:
        response_payload[item_code] = prompt_validated_input(
            f"❓ {item_code} - {question}\n👉 Choice (1-5): ", val_type=int, valid_range=(1, 5)
        )
        print()

    print("--- LONELINESS PARAMETERS ---")
    print("Scale: 1 (Never), 2 (Rarely), 3 (Sometimes), 4 (Often)\n")
    
    loneliness_items = [
        ("loneliness1", "How often do you feel completely 'in tune' with the people around you?"),
        ("loneliness2", "How often do you feel that you severely lack deep companionship?"),
        ("loneliness3", "How often do you feel that there is truly no one you can turn to?"),
        ("loneliness4", "How often do you feel isolated and entirely alone?"),
        ("loneliness5", "How often do you feel strongly integrated as part of a group of friends?"),
        ("loneliness6", "How often do you feel that you have a significant amount in common with those around you?")
    ]
    
    for item_code, question in loneliness_items:
        response_payload[item_code] = prompt_validated_input(
            f"❓ {item_code} - {question}\n👉 Choice (1-4): ", val_type=int, valid_range=(1, 4)
        )
        print()

    # -------------------------------------------------------------
    # SECTION 6: DOMAIN 5 - OCCUPATIONAL BURNOUT (8 Items)
    # -------------------------------------------------------------
    clear_terminal()
    print("--- SECTION 6: DOMAIN 5 - OCCUPATIONAL BURNOUT ---")
    response_payload['work_hours_per_week'] = prompt_validated_input(
        "❓ work_hours_per_week - Average total number of active working hours per week.\n👉 Hours (0-168): ", 
        val_type=int, valid_range=(0, 168)
    )
    print()
    response_payload['meetings_per_day'] = prompt_validated_input(
        "❓ meetings_per_day - Average total number of corporate meetings attended daily.\n👉 Meetings Count: ", 
        val_type=float, valid_range=(0.0, 50.0)
    )
    print()

    print("Scale: Slider continuous range from 1.0 (Low/Poor) to 10.0 (High/Extreme/Excellent)\n")
    burnout_scores = [
        ('work_life_balance_score', "Rate your subjective overall work-life balance satisfaction level."),
        ('job_satisfaction_score', "Rate your subjective professional and career fulfillment level."),
        ('deadline_pressure_score', "Rate the frequency/severity of time constraints and urgency pressures."),
        ('autonomy_score', "Rate your level of control and decision freedom over execution tasks."),
        ('stress_score', "Rate the baseline cumulative stress experienced over the past month."),
        ('social_support_score', "Rate the perceived strength of your immediate workplace support framework.")
    ]
    
    for feature_id, question in burnout_scores:
        response_payload[feature_id] = prompt_validated_input(
            f"❓ {feature_id} - {question}\n👉 Score (1.0 - 10.0): ", val_type=float, valid_range=(1.0, 10.0)
        )
        print()

    # -------------------------------------------------------------
    # SECTION 7: DOMAIN 6 - SEVERE CLINICAL SYMPTOMS (7 Items)
    # -------------------------------------------------------------
    clear_terminal()
    print("--- SECTION 7: DOMAIN 6 - SEVERE CLINICAL SYMPTOMS ---")
    print("Scale: Binary Selection [0 = No, 1 = Yes]\n")
    
    clinical_items = [
        ("unwanted_thoughts", "Experiencing recurrent, distressing, intrusive thoughts or images."),
        ("repetitve_behaviors", "Feeling compelled to repeat physical actions or rigid mental rituals."),
        ("overthinking", "Excessive rumination over insignificant daily micro-interactions."),
        ("mind_going_blank", "Cognitive paralysis or loss of memory continuity during stress situations."),
        ("avoidance_of_social_activity", "Active avoidance of social events, crowds, or public areas out of distress."),
        ("panic", "Sudden, unprovoked surges of overwhelming physical terror or heart palpitations."),
        ("hypervigilance", "Continuous high-alert monitoring of surroundings to guard against threats.")
    ]
    
    for col_id, question in clinical_items:
        response_payload[col_id] = prompt_validated_input(
            f"❓ {col_id} - {question}\n👉 Choice [0 for No / 1 for Yes]: ", val_type=int, valid_range=(0, 1)
        )
        print()

    # -------------------------------------------------------------
    # PERSISTENCE ENGINE: SEPARATE TARGET CSV CHANNELS (FIXED PATHS)
    # -------------------------------------------------------------
    # Step out of 'scripts/' into project root before targeting 'tests/'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    tests_dir = os.path.join(project_root, "tests")
    
    # Guarantee target directory existence 
    os.makedirs(tests_dir, exist_ok=True)
    
    csv_path = os.path.normpath(os.path.join(tests_dir, "responses.csv"))
    all_csv_path = os.path.normpath(os.path.join(tests_dir, "all_responses.csv"))
    
    all_file_exists = os.path.exists(all_csv_path)
    
    # Establish strict, unshifting sequential column order matching the 70 elements
    ordered_headers = [
        'age', 'gender',
        'EXT1', 'EXT2', 'EXT3', 'EST1', 'EST2', 'EST3', 'AGR1', 'AGR2', 'AGR3', 'CSN1', 'CSN2', 'CSN3', 'OPN1', 'OPN2', 'OPN3',
        'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10',
        'DPQ010', 'DPQ020', 'DPQ030', 'DPQ040', 'DPQ050', 'DPQ060', 'DPQ070', 'DPQ080', 'DPQ090', 'DPQ100',
        'SLQ300', 'SLQ310',
        'IAT1', 'IAT2', 'IAT3', 'IAT4', 'IAT5', 'IAT6', 'IAT7', 'IAT8', 'IAT9', 'IAT10',
        'loneliness1', 'loneliness2', 'loneliness3', 'loneliness4', 'loneliness5', 'loneliness6',
        'work_hours_per_week', 'meetings_per_day', 'work_life_balance_score', 'job_satisfaction_score', 
        'deadline_pressure_score', 'autonomy_score', 'stress_score', 'social_support_score',
        'unwanted_thoughts', 'repetitve_behaviors', 'overthinking', 'mind_going_blank', 
        'avoidance_of_social_activity', 'panic', 'hypervigilance'
    ]

    # Dynamically align formatting if checking the long-term history ledger file matrix
    if all_file_exists:
        try:
            with open(all_csv_path, 'r', newline='') as f:
                reader = csv.reader(f)
                existing_headers = next(reader)
            for h in ordered_headers:
                if h not in existing_headers:
                    existing_headers.append(h)
            ordered_headers = existing_headers
        except StopIteration:
            pass

    # CHANNEL 1: Overwrite single active row (responses.csv)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=ordered_headers, extrasaction='ignore', restval=0)
        writer.writeheader()
        writer.writerow(response_payload)

    # CHANNEL 2: Chronologically append history row (all_responses.csv)
    with open(all_csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=ordered_headers, extrasaction='ignore', restval=0)
        if not all_file_exists or os.stat(all_csv_path).st_size == 0:
            writer.writeheader()
        writer.writerow(response_payload)
        
    print("=" * 75)
    print("🎉 SUCCESS: Assessment completed successfully!")
    print(f"🔄 Latest session flushed and saved to: {csv_path}")
    print(f"📚 Historical baseline appended to    : {all_csv_path}")
    print("=" * 75)

if __name__ == "__main__":
    try:
        run_imp70_assessment()
    except KeyboardInterrupt:
        print("\n\n⚠️ Assessment terminated early by user request. Exiting cleanly.")
        sys.exit(0)