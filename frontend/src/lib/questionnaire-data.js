
// MINDSIGHT IMP-70 Questionnaire Data


export const QUESTIONNAIRE_SECTIONS = [
  {
    id: "personality",
    title: "DOMAIN 1: PERSONALITY",
    description: "Indicate how much you agree or disagree with each statement.",
    scale: ["Disagree", "Slightly Disagree", "Neutral", "Slightly Agree", "Agree"],
    scaleValues: [1, 2, 3, 4, 5],
    questions: [
      { id: "EXT1", text: "I am the life of the party." },
      { id: "EXT2", text: "I don't talk a lot.", reverse: true },
      { id: "EXT3", text: "I feel comfortable around people." },
      { id: "EST1", text: "I get stressed out easily." },
      { id: "EST2", text: "I am relaxed most of the time.", reverse: true },
      { id: "EST3", text: "I worry about things." },
      { id: "AGR1", text: "I feel little concern for others.", reverse: true },
      { id: "AGR2", text: "I am interested in people." },
      { id: "AGR3", text: "I feel sympathy for others' feelings." },
      { id: "CSN1", text: "I am always prepared." },
      { id: "CSN2", text: "I leave my duties undone.", reverse: true },
      { id: "CSN3", text: "I pay attention to details." },
      { id: "OPN1", text: "I have a rich vocabulary." },
      { id: "OPN2", text: "I have difficulty understanding abstract ideas.", reverse: true },
      { id: "OPN3", text: "I have a vivid imagination." },
    ],
  },
  {
    id: "self_esteem",
    title: "DOMAIN 2: SELF-ESTEEM",
    description: "Indicate how much you agree or disagree with each statement.",
    scale: ["Disagree", "Slightly Disagree", "Neutral", "Slightly Agree", "Agree"],
    scaleValues: [1, 2, 3, 4, 5],
    questions: [
      { id: "Q1", text: "I feel that I am a person of worth, at least on an equal plane with others." },
      { id: "Q2", text: "I feel that I have a number of good qualities." },
      { id: "Q3", text: "All in all, I am inclined to feel that I am a failure.", reverse: true },
      { id: "Q4", text: "I am able to do things as well as most other people." },
      { id: "Q5", text: "I feel I do not have much to be proud of.", reverse: true },
      { id: "Q6", text: "I take a positive attitude toward myself." },
      { id: "Q7", text: "On the whole, I am satisfied with myself." },
      { id: "Q8", text: "I wish I could have more respect for myself.", reverse: true },
      { id: "Q9", text: "I certainly feel useless at times.", reverse: true },
      { id: "Q10", text: "At times I think I am no good at all.", reverse: true },
    ],
  },
  {
    id: "mood_sleep",
    title: "DOMAIN 3: MOOD AND SLEEP",
    description: "Over the past 2 weeks, how often have you been bothered by the following problems?",
    scale: ["Not at all", "Several days", "More than half the days", "Nearly every day"],
    scaleValues: [0, 1, 2, 3],
    questions: [
      { id: "DPQ010", text: "Little interest or pleasure in doing things over the past 2 weeks." },
      { id: "DPQ020", text: "Feeling down, depressed, or hopeless over the past 2 weeks." },
      { id: "DPQ030", text: "Trouble falling or staying asleep, or sleeping too much." },
      { id: "DPQ040", text: "Feeling tired or having little energy." },
      { id: "DPQ050", text: "Poor appetite or overeating patterns." },
      { id: "DPQ060", text: "Feeling bad about yourself, or that you are a failure." },
      { id: "DPQ070", text: "Trouble concentrating on basic tasks (e.g., reading or watching TV)." },
      { id: "DPQ080", text: "Moving/speaking so slowly or quickly that others noticed." },
      { id: "DPQ090", text: "Thoughts that you would be better off dead, or of hurting yourself." },
      { id: "DPQ100", text: "How difficult have these mood problems made it to work or get along with people?" },
      { 
        id: "SLQ300", 
        text: "What time do you usually go to bed on workdays/weekdays?",
        inputType: "time",
        placeholder: "HH:MM"
      },
      { 
        id: "SLQ310", 
        text: "What time do you usually wake up on workdays/weekdays?",
        inputType: "time",
        placeholder: "HH:MM"
      },
    ],
  },
  {
    id: "digital_social_risk",
    title: "DOMAIN 4: DIGITAL AND SOCIAL RISK",
    description: "Indicate how often each statement applies to you.",
    scale: ["Rarely", "Occasionally", "Frequently", "Often", "Always"],
    scaleValues: [1, 2, 3, 4, 5],
    questions: [
      { id: "IAT1", text: "How often do you stay online longer than you originally intended?" },
      { id: "IAT2", text: "How often do you neglect household/daily tasks to spend more time online?" },
      { id: "IAT3", text: "How often do you prefer the excitement of the internet over real-world relationships?" },
      { id: "IAT4", text: "How often do you form new digital relationships with online users?" },
      { id: "IAT5", text: "How often do close connections complain to you about your internet usage?" },
      { id: "IAT6", text: "How often do your grades, productivity, or work responsibilities suffer from screen time?" },
      { id: "IAT7", text: "How often do you check electronic communication channels before executing required tasks?" },
      { id: "IAT8", text: "How often does your job performance diminish due to distracting online activity?" },
      { id: "IAT9", text: "How often do you become defensive or highly secretive regarding your browser behaviors?" },
      { id: "IAT10", text: "How often do you mask disturbing real-life thoughts using soothing internet media?" },
    ],
  },
  {
    id: "loneliness",
    title: "DOMAIN 4: DIGITAL AND SOCIAL RISK - Loneliness",
    description: "Indicate how often you feel the following.",
    scale: ["Never", "Rarely", "Sometimes", "Often"],
    scaleValues: [0, 1, 2, 3],
    questions: [
      { id: "loneliness1", text: "How often do you feel completely 'in tune' with the people around you?" },
      { id: "loneliness2", text: "How often do you feel that you severely lack deep companionship?" },
      { id: "loneliness3", text: "How often do you feel that there is truly no one you can turn to?" },
      { id: "loneliness4", text: "How often do you feel isolated and entirely alone?" },
      { id: "loneliness5", text: "How often do you feel strongly integrated as part of a group of friends?" },
      { id: "loneliness6", text: "How often do you feel that you have a significant amount in common with those around you?" },
    ],
  },
  {
    id: "occupational_burnout",
    title: "DOMAIN 5: OCCUPATIONAL BURNOUT",
    description: "Please provide information about your work environment and experiences.",
    scale: ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
    scaleValues: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    questions: [
      { 
        id: "work_hours_per_week", 
        text: "Average total number of active working hours per week.",
        inputType: "number",
        placeholder: "Hours",
        min: 0,
        max: 168
      },
      { 
        id: "meetings_per_day", 
        text: "Average total number of corporate meetings attended daily.",
        inputType: "number",
        placeholder: "Meetings",
        min: 0,
        max: 50
      },
      { 
        id: "work_life_balance_score", 
        text: "Rate your subjective overall work-life balance satisfaction level. (1=Poor, 10=Excellent)",
        inputType: "scale"
      },
      { 
        id: "job_satisfaction_score", 
        text: "Rate your subjective professional and career fulfillment level. (1=Poor, 10=Excellent)",
        inputType: "scale"
      },
      { 
        id: "deadline_pressure_score", 
        text: "Rate the frequency/severity of time constraints and urgency pressures. (1=Low, 10=Extreme)",
        inputType: "scale"
      },
      { 
        id: "autonomy_score", 
        text: "Rate your level of control and decision freedom over execution tasks. (1=None, 10=Total)",
        inputType: "scale"
      },
      { 
        id: "stress_score", 
        text: "Rate the baseline cumulative stress experienced over the past month. (1=Low, 10=Extreme)",
        inputType: "scale"
      },
      { 
        id: "social_support_score", 
        text: "Rate the perceived strength of your immediate workplace support framework. (1=Weak, 10=Strong)",
        inputType: "scale"
      },
    ],
  },
  {
    id: "severe_clinical",
    title: "DOMAIN 6: SEVERE CLINICAL",
    description: "Indicate whether you have experienced any of the following.",
    scale: ["No", "Yes"],
    scaleValues: [0, 1],
    questions: [
      { id: "unwanted_thoughts", text: "Experiencing recurrent, distressing, intrusive thoughts or images." },
      { id: "repetitive_behaviors", text: "Feeling compelled to repeat physical actions or rigid mental rituals." },
      { id: "overthinking", text: "Excessive rumination over insignificant daily micro-interactions." },
      { id: "mind_going_blank", text: "Cognitive paralysis or loss of memory continuity during stress situations." },
      { id: "avoidance_social_activity", text: "Active avoidance of social events, crowds, or public areas out of distress." },
      { id: "panic", text: "Sudden, unprovoked surges of overwhelming physical terror or heart palpitations." },
      { id: "hypervigilance", text: "Continuous high-alert monitoring of surroundings to guard against threats." },
    ],
  },

];

