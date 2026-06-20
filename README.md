# MINDSIGHT Clinical Assessment System (v3.9)

MINDSIGHT is a production-grade, unified multi-domain diagnostic evaluation framework. The system processes high-dimensional psychological, physiological, occupational, and behavioral payload data across 6 independent diagnostic vectors, running deterministic metrics alongside advanced machine learning ensembles (XGBoost, LightGBM, RandomForest, and LogisticRegression + IsolationForest). It culminates in a synthesized, cross-domain global overview profile designed for clinical decision-making support.

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
* **Polarity:** 
  * *Positive Valence Items:* `Q1, Q2, Q4, Q6, Q7` (Direct accumulation).
  * *Inverted Vulnerability Items:* `Q3, Q5, Q8, Q9, Q10` (Inverted via $4.0 - \text{value}$).

### C. Mood & Circadian Metrics (PHQ-9 + Clock Diagnostics)
* **Features:** 
  * 9 clinical psychometric values (`DPQ010` through `DPQ090` inclusive of suicidal ideation indices).
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

+----------------------------------------------------------------------------------------------------+
|                                    RAW CSV INPUT PAYLOAD ROW                                       |
+----------------------------------------------------------------------------------------------------+
│
+------------------+-----------------------+------------------+-----------------------+
│                  │                       │                  │                       │
▼                  ▼                       ▼                  ▼                       ▼
+--------------+   +--------------+        +--------------+   +--------------+        +--------------+
|   DOMAIN 1   |   |   DOMAIN 2   |        |   DOMAIN 3   |   |   DOMAIN 4   |        |   DOMAIN 5   |
| Personality  |   | Self-Esteem  |        | Mood & Sleep |   | Digital/Soc. |        |   Burnout    |
| Mean Dev.    |   | 0-4 Max 40   |        | InclusiveSum |   | RandomForest |        | XGB/LightGBM |
| Vector Lat.  |   | Inv-Mapping  |        | Wraparound   |   | DataFrames   |        | SHAP Sign-Tr |
+--------------+   +--------------+        +--------------+   +--------------+        +--------------+
│                  │                       │                  │                       │
+------------------+-----------------------+------------------+-----------------------+
│
▼
+--------------------+
|      DOMAIN 6      |
| Severe Clinical LR |
|  + IsolationForest |
+--------------------+
│
▼
+--------------------+
|  PROFILE SYNTHESIS |
| Tiered Cross-Match |
|   Summary Output   |
+--------------------+

### Domain 1: Personality Vectors
* **Type:** Likelihood Deviation Vector Model.
* **Architecture:** Avoids top-level aggregation leakage by calculating the arithmetic mean of item clusters relative to the midpoint. Continuous item attribution tracking maps local feature importance via absolute variance:
  $$\text{Deviation} = |X_i - 2.0|$$
  Surfaces individual sub-items as structural drivers rather than circular trait summaries.

### Domain 2: Self-Esteem Matrix
* **Type:** Calibrated Psychometric Scale scoring.
* **Architecture:** Evaluates total configuration profiles across a maximum scale value of 40 points. Implements strict reverse-scoring algorithms for negative valence questions:
  $$f(x) = 4.0 - x$$
  Categorizes final placements into High Self-Esteem ($\ge 30$), Normal ($\ge 16$), and Low Self-Esteem ($< 16$) bands.

### Domain 3: Mood & Sleep Dynamics
* **Type:** Summatic Clinical Index + Cyclical Clock Modulo Framework.
* **Architecture:** Integrates all 9 categorical tracks of the PHQ-9 instrument to evaluate overall severity. Sleep processing mitigates overnight timeline wraparound traps using a safe modulo 24-hour transform:
  $$\text{Duration} = (\text{Wake Time} - \text{Bed Time}) \pmod{24.0}$$

### Domain 4: Digital & Social Dynamics
* **Type:** Dual Ensemble Random Forest Regressors (RandomForestRegressor).
* **Architecture:** Processes structural data arrays using a matched inference dataframe pipeline to execute independent predictions for `addiction_model` and `loneliness_model`. Contains dynamic fallback scaling thresholds if file state paths are absent.

### Domain 5: Occupational Burnout Index
* **Type:** Gradient Boosted Decision Tree Ensembles (XGBRegressor / LGBMRegressor).
* **Architecture:** Extracts structural target indicators based on institutional loading. Local feature attribution runs through a structural SHAP array that isolates driver values while strictly preserving raw signs ($+$ or $-$), ensuring protective variables (e.g., high autonomy or high social support) are correctly represented as negative pressure vectors instead of risk additions.

### Domain 6: Severe Clinical Screening Matrix
* **Type:** Log-Odds Classifier (LogisticRegression) + Unsupervised Anomaly Detector (IsolationForest).
* **Architecture:** Evaluates clinical presentations while checking response configuration validity. To bypass sigmoid saturation drops ($+0.0$), it performs feature attribution in raw linear log-odds space ($\text{coefficient} \times \text{value}$). The runtime engine actively skips inactive entries (`value == 0`), preventing zero-contribution items from leaking into the top diagnostic driver slots.

---

## 3. Core Base Architecture & File Mappings

### `models/inference_wrappers.py`
The primary operational gateway for individual model scoring. It manages state files, maps raw data matrices into safe numerical configurations, and establishes data integrity contracts across the six domains.
* `evaluate_domain1_personality(raw_payload)`: Extracts 15 sub-items, maps trait variations, and ranks top item-level variances.
* `evaluate_domain2_self_esteem(raw_payload)`: Implements 0-4 polarity inversions, scores items out of 40, and determines categorical classifications.
* `evaluate_domain3_mood_sleep(raw_payload)`: Explicitly sums the full 9-item PHQ matrix and runs cyclical 24-hour sleep duration transformations.
* `evaluate_domain4_multitask(raw_payload)`: Maps behavioral features into input dataframes to drive random forest predictions.
* `evaluate_domain5_burnout(raw_payload)`: Tracks workplace loading via decision trees and handles SHAP vector sign assignments.
* `evaluate_domain6_clinical(raw_payload)`: Runs logistic classifications alongside anomaly detections, utilizing active feature filtering in log-odds spaces.

### `profile_aggregator.py`
The orchestration component that manages multi-domain handoffs and runs cross-domain global profile evaluations.
* `aggregate_assessment_profile(raw_csv_row_dict)`: Ingests flat dictionary rows, executes the `inference_wrappers` pipeline sequentially, and forwards compiled metrics to the narrative builder.
* `evaluate_cross_domain_synthesis(domain_outputs)`: Reads computed pipeline outputs defensively to build an objective, structured summary paragraph based on an unified clinical severity matrix.

---

## 4. General Functional Workflow

The MINDSIGHT engine executes operations via a strict, single-pass pipeline architecture:

[ Ingest Row Payload ]
│
▼
[ Parallel Parsing & Validation ] ──► (Verify Clock Strings & Type Conversions)
│
▼
[ Multi-Domain Evaluation Loop ]  ──► (Execute 6 Core Domain Wrappers)
│
▼
[ Metric Verification Engine ]    ──► (Check PHQ-9 Items, 40-Scale Inversions, SHAP Signs)
│
▼
[ Cross-Domain Synthesis ]        ──► (Apply Tiered Narrative Decision Matrix)
│
▼
[ Consolidated JSON Serialization ] ──► (Output Unified Struct for PDF Generation Layer)


1. **Ingestion & Structuring:** A raw data dictionary is generated from an incoming row source (such as automated CSV parsers) and fed into `aggregate_assessment_profile`.
2. **Payload Extraction:** Demographic identifiers (`age`, `gender`) are isolated, while sub-feature arrays are systematically routed to their respective domain evaluation wrappers.
3. **Isolated Inference Execution:**
   * Continuous data values are centered against baseline metrics.
   * Psychometric questionnaire scales execute inversion logic on negative tracking vectors.
   * Time-series string patterns are transformed into fractional hours via modulo calculations.
   * Ensemble models receive appropriately structured matrices to generate predictive targets.
4. **Local Feature Attribution:** Every individual wrapper extracts local explanation vectors (SHAP arrays, linear coefficients, or baseline deviations), discards inactive features, preserves sign orientation, and sorts the elements to output the top 3 high-impact contributors.
5. **State-Level Synthesis Integration:** The compiled metrics from all 6 vectors are delivered to `evaluate_cross_domain_synthesis`. This component applies a tiered clinical severity matrix to generate an integrated global summary narrative without recomputing base variables.
6. **Contract-Aligned Output Delivery:** The system serializes the complete structural payload (including versioning, metadata, domain parameters, and the synthesized overview text) into a unified object, ready for ingestion by downstream reporting engines and PDF template builders.
