export const output = {
  "schema_version": "3.9",
  "id_no": "MS-20260620_130625-ANONYMOUS",
  "age": 21,
  "sex": "Male",
  "domain_scores": {
    "domain_1_personality": {
      "domain": "domain_1_personality",
      "placement": {
        "extraversion": -0.4,
        "emotional_stability": 0.4,
        "agreeableness": 0.6,
        "conscientiousness": 0.4,
        "openness": 0.0
      },
      "top_contributors": [
        {
          "feature": "Agreeableness",
          "display_name": "Agreeableness Vector",
          "contribution": 0.6,
          "direction": "+"
        },
        {
          "feature": "Extraversion",
          "display_name": "Extraversion Vector",
          "contribution": 0.4,
          "direction": "-"
        },
        {
          "feature": "Emotional_Stability",
          "display_name": "Emotional_Stability Vector",
          "contribution": 0.4,
          "direction": "+"
        }
      ]
    },
    "domain_2_self_esteem": {
      "domain": "domain_2_self_esteem",
      "placement": {
        "score": 21,
        "max_possible_score": 40,
        "classification": "Normal"
      },
      "top_contributors": [
        {
          "feature": "Q1",
          "display_name": "Self-Esteem Item 1",
          "contribution": 2.0,
          "direction": "+"
        },
        {
          "feature": "Q2",
          "display_name": "Self-Esteem Item 2",
          "contribution": 2.0,
          "direction": "-"
        },
        {
          "feature": "Q4",
          "display_name": "Self-Esteem Item 4",
          "contribution": 2.0,
          "direction": "+"
        }
      ]
    },
    "domain_3_mood_sleep": {
      "domain": "domain_3_mood_sleep",
      "placement": {
        "phq9_sum": 7,
        "severity_label": "Mild Depression",
        "calculated_sleep_duration_hours": 8.0
      },
      "top_contributors": [
        {
          "feature": "PHQ_Core",
          "display_name": "Symptom Burden Summation",
          "contribution": 7.0,
          "direction": "+"
        },
        {
          "feature": "Sleep_Duration",
          "display_name": "Calculated Sleep Duration",
          "contribution": 0.5,
          "direction": "+"
        }
      ]
    },
    "domain_4_multitask": {
      "domain": "domain_4_digital_and_social",
      "placement": {
        "predicted_total_internet_addiction": 25.0,
        "predicted_total_loneliness": 16.0,
        "loneliness_score": 53.28,
        "classification": "Baseline Cohort Profile"
      },
      "top_contributors": [
        {
          "feature": "IAT4",
          "display_name": "IAT4",
          "contribution": 0.3723,
          "direction": "-"
        },
        {
          "feature": "IAT5",
          "display_name": "IAT5",
          "contribution": 0.3585,
          "direction": "-"
        },
        {
          "feature": "loneliness4",
          "display_name": "Loneliness4",
          "contribution": 0.2703,
          "direction": "-"
        }
      ]
    },
    "domain_5_occupational_burnout": {
      "domain": "domain_5_occupational_burnout",
      "placement": {
        "burnout_index": 5.341,
        "burnout_tier_label": "Moderate Burnout Profile"
      },
      "top_contributors": [
        {
          "feature": "social_support_score",
          "display_name": "Social Support Resilience",
          "contribution": 0.9,
          "direction": "+"
        },
        {
          "feature": "deadline_pressure_score",
          "display_name": "Perceived Deadline Velocity",
          "contribution": 0.84,
          "direction": "+"
        },
        {
          "feature": "autonomy_score",
          "display_name": "Workplace Autonomy",
          "contribution": 0.84,
          "direction": "+"
        }
      ]
    },
    "domain_6_severe_clinical": {
      "domain": "domain_6_severe_clinical",
      "placement": {
        "predicted_condition_code": 1,
        "predicted_condition_label": "Mild Symptomatic Profile",
        "anomaly_review_flag": true
      },
      "top_contributors": [
        {
          "feature": "avoidance_social_activity",
          "display_name": "Social Withdrawal Vectors",
          "contribution": 8.1305,
          "direction": "+"
        },
        {
          "feature": "mind_going_blank",
          "display_name": "Acute Attentional Dropouts",
          "contribution": 7.3143,
          "direction": "+"
        },
        {
          "feature": "repetitive_behaviors",
          "display_name": "Compulsive Action Patterns",
          "contribution": 7.1883,
          "direction": "+"
        }
      ]
    }
  },
  "global_synthesis": "COMPLEX PROFILE EVALUATION: Unsupervised screening models flag an atypical response configuration. While tracking markers align closely with the Mild Symptomatic Profile, the overall structure falls outside standard baseline distributions concurrently with elevated clinical metrics (PHQ-9 Sum: 7). A comprehensive professional differential evaluation is recommended to reconcile these mixed indicators."
}