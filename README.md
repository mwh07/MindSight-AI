# MindSIGHT

# MINDSIGHT Clinical Assessment System (v3.9)

MINDSIGHT is a production-grade, unified multi-domain diagnostic evaluation framework. The system processes high-dimensional psychological, physiological, occupational, and behavioral payload data across 6 independent diagnostic vectors, running deterministic metrics alongside advanced machine learning ensembles (`XGBoost`, `LightGBM`, `RandomForest`, and `LogisticRegression` + `IsolationForest`). It culminates in a synthesized, cross-domain global overview profile designed for clinical decision-making support.

---

## 1. Datasets & Feature Specifications

The framework ingests raw observation vectors matching a structured schema (e.g., `responses.csv`). The inputs are classified into five primary feature blocks:

### A. Personality Vector Inputs (Big Five / IRT)
* **Features:** 15 item-level sub-features (`EXT1-3`, `EST1-3`, `AGR1-3`, `CSN1-3`, `OPN1-3`).
* **Scale:** Numeric values centered around a neutral baseline anchor point of `2.0`.
* **Application:** Captured to evaluate continuous trait deviations across Extraversion, Emotional Stability, Agreeableness, Conscientiousness, and Openness.

### B. Self-Esteem Scale (Modified RSE)
* **Features:** 10 diagnostic survey indicators (`Q1` through `Q10`).
* **Scale:** Explicit `0` to `4` integer ranges, establishing a mathematical ceiling of `40`.
* **Polarity:** * *Positive Valence Items:* `Q1, Q2, Q4, Q6, Q7` (Direct accumulation).
  * *Inverted Vulnerability Items:* `Q3, Q5, Q8, Q9, Q10` (Inverted via $4.0 - \text{value}$).

### C. Mood & Circadian Metrics (PHQ-9 + Clock Diagnostics)
* **Features:** * 9 clinical psychometric values (`DPQ010` through `DPQ090` inclusive of suicidal ideation indices).
  * 2 string clock tokens (`SLQ300`: Bedtime, `SLQ310`: Wake time) formatted as `HH:MM`.
* **Scale:** PHQ items score from `0` to `3`. Sleep times span a standard 24-hour cyclical clock space.

### D. Digital Engagement & Social Connectivity Spans
* **Features:** 10 Internet Addiction Test metrics (`IAT1-10`) combined with 6 standardized social indicators (`loneliness1-6`).
* **Scale:** Ordinal integer weights tracking compulsive online engagement and perceived relationship depths.

### E. Occupational Load & Severe Symptomatic Markers
* **Administrative Features:** `work_hours_per_week`, `meetings_per_day`, `work_life_balance_score`, `job_satisfaction_score`, `deadline_pressure_score`, `autonomy_score`, `stress_score`, `social_support_score`.
* **Binary Severe Markers:** 7 clinical flags tracking `unwanted_thoughts`, `repetitve_behaviors`, `overthinking`, `mind_going_blank`, `avoidance_of_social_activity`, `panic`, and `hypervigilance`.

---

## 2. Domain-Specific Model Architectures
