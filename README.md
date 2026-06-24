# MINDSIGHT

**Unified Multi-Domain Psychological Diagnostic Profiling System**
*System Architecture Version 3.8 · Schema Version 2.6*

MINDSIGHT takes a single 70-item questionnaire response and produces a six-domain psychological profile — personality, self-esteem, mood & sleep, digital/social behavior, occupational burnout, and clinical (OCD-spectrum) screening — culminating in a synthesized global profile and a generated PDF report.

---

## Table of Contents

1. [Datasets](#1-datasets)
2. [Domain Model Architectures](#2-domain-model-architectures)
3. [Key Files & Their Roles](#3-key-files--their-roles)
4. [General Workflow](#4-general-workflow-most-important)
5. [Project Structure](#5-project-structure)

---

## 1. Datasets

All datasets live in `datasets/`. Each domain is trained on exactly one source dataset, and the **authoritative feature list per domain is defined in `schema_config.json` (schema v2.6, 70 total features)** — that file, not this README, is the source of truth if the two ever disagree.

| Dataset file | Used by | Feature columns used |
|---|---|---|
| `big_five_personality_clean.csv` | Domain 1 — Personality | `EXT1–EXT3`, `EST1–EST3`, `AGR1–AGR3`, `CSN1–CSN3`, `OPN1–OPN3` (3 items per trait, 15 total) |
| `rosenberg_self_esteem_clean.csv` | Domain 2 — Self-Esteem | `age`, `gender`, `Q1–Q10` (10 RSE items) |
| `nhanes_joined_mood_sleep.csv` | Domain 3 — Mood & Sleep | `DPQ010–DPQ100` (10 PHQ-9 items), `SLQ300` (bedtime), `SLQ310` (wake time) |
| `internet_phq_loneliness_clean.csv` | Domain 4 — Digital & Social | `age`, `gender`, `IAT1–IAT10` (Internet Addiction Test), `loneliness1–loneliness6` |
| `tech_burnout_2026_clean.csv` | Domain 5 — Occupational Burnout | `age`, `gender`, `work_hours_per_week`, `meetings_per_day`, `work_life_balance_score`, `job_satisfaction_score`, `deadline_pressure_score`, `autonomy_score`, `stress_score`, `social_support_score` |
| `ocd_symptoms_clean.csv` | Domain 6 — Severe Clinical Screening | `unwanted_thoughts`, `repetitive_behaviors`, `overthinking`, `mind_going_blank`, `avoidance_social_activity`, `panic`, `hypervigilance` (7 binary symptom flags) |

Supporting files in `datasets/`:
- `datasets_metadata.json` / `datasets_metadata.txt` — column ranges, types, and provenance notes for every dataset above (read via `scripts/read_metadata.py`).

> **Note:** `nhanes_joined_mood_sleep.csv` is the pre-joined output of `scripts/archive/merge_nhanes_mood_sleep.py`, which joined NHANES mood (DPQ) and sleep (SLQ) tables on `SEQN`. The merge script is archived because the join is a one-time data-prep step, not part of the live pipeline.

---

## 2. Domain Model Architectures

Each domain has its own training script (`models/train_domain*_*.py`) and produces artifacts saved into `models/saved_states/`. Architectures differ deliberately by domain — chosen to match the statistical nature of each instrument rather than using one model for everything.

### Domain 1 — Personality (Big Five)
- **Architecture:** Graded Response Model (GRM), Item Response Theory (IRT)
- **Why:** Big Five items are ordinal Likert responses; GRM-IRT models each item's discrimination and threshold parameters per trait, producing a latent trait score (θ) that's more psychometrically valid than a raw item average.
- **One model fit per trait** (5 total: EXT, EST, AGR, CSN, OPN), each on its 3 items.
- **Artifacts:** `domain1_grm_parameters.pkl` (fitted item discrimination/threshold parameters per trait), `domain1_grm_metadata.json`.
- **Output:** θ (theta) placement per trait, plus per-item contribution to that trait's score.

### Domain 2 — Self-Esteem (Rosenberg Scale)
- **Architecture:** Deterministic scoring formula + empirical percentile lookup table (no ML model).
- **Why:** RSE is a validated clinical instrument with a fixed, well-known scoring procedure (reverse-coding negative items, summing to a 0–30 range); there's nothing to "learn" — the value is in correctly computing the standard score and placing it against a normative percentile distribution.
- **Cohort percentiles:** built from `rosenberg_self_esteem_clean.csv`, bucketed by age band and gender; `age` is filtered to a plausible [10, 90] range before building cohort tables to avoid data-entry outliers skewing percentiles.
- **Artifacts:** `domain2_self_esteem.pkl` (scoring/lookup logic), `domain2_self_esteem_percentiles.json` (the percentile table), `domain2_self_esteem_metadata.json`.
- **Output:** RSE total score (out of 30), percentile placement label (e.g. "Normal"), and per-item signed contributions.

### Domain 3 — Mood & Sleep (PHQ-9 + Sleep Duration)
- **Architecture:** LightGBM (gradient-boosted trees) on engineered features, alongside a deterministic PHQ-9 sum and sleep-duration calculation.
- **Why:** PHQ-9 itself has a standard deterministic sum (`DPQ010`–`DPQ100`); LightGBM is used for any secondary modeling once the deterministic mood and sleep features are engineered correctly.
- **Preprocessing requirements (critical):**
  - DPQ rows with refusal/"don't know" codes (`7`, `9`) must be filtered/excluded before summation.
  - `SLQ300` (bedtime) and `SLQ310` (wake time) are parsed as `HH:MM` and converted to sleep duration using **mod-24 wraparound** (since bedtime is typically late-night and wake time is the next morning).
- **Artifacts:** `domain3_mood_sleep.txt` (LightGBM model dump), `domain3_mood_sleep_metadata.json`.
- **Output:** PHQ-9 sum + severity band (e.g. "Moderate Depression"), calculated sleep duration in hours.

### Domain 4 — Digital & Social Dynamics (Internet Addiction + Loneliness)
- **Architecture:** Two independent `RandomForestRegressor` models — one predicting total Internet Addiction score, one predicting total loneliness score.
- **Why:** These are two related but distinct constructs (digital dependency vs. social isolation) measured by separate item sets (`IAT1–10` vs `loneliness1–6`); modeling them as two independent regressors avoids forcing a shared latent structure that doesn't actually exist in the source instruments.
- **Artifacts:** `domain4_digital_social.pkl` (both fitted RF models), `domain4_digital_social_metadata.json`.
- **Output:** Addiction score, Loneliness Index, and top contributing IAT/loneliness items per model (via `.feature_importances_`).

### Domain 5 — Occupational Burnout
- **Architecture:** XGBoost (gradient-boosted trees).
- **Why:** Burnout is a composite outcome influenced by multiple interacting workplace factors (hours, deadline pressure, autonomy, support) with likely non-linear relationships and interactions — XGBoost handles this well and supports SHAP-style per-feature attribution for explainability.
- **Artifacts:** `domain5_burnout.json` (XGBoost model dump), `domain5_burnout_metadata.json`.
- **Output:** Burnout Index (continuous) + severity band (e.g. "Severe Burnout Indication"), top contributing workplace factors.

### Domain 6 — Severe Clinical Screening (OCD-Spectrum Symptoms)
- **Architecture:** `LogisticRegression(multi_class='ovr')` (one-vs-rest) for symptom severity classification, plus an `IsolationForest` for atypical/outlier response-pattern detection.
- **Why:** The 7 features are binary symptom flags; logistic regression gives interpretable per-symptom coefficients, while the Isolation Forest separately flags response patterns that don't fit any typical cluster — surfaced in the report as "ATYPICAL" — which is distinct from symptom severity itself.
- **Artifacts:** `domain6_clinical.pkl` (both fitted models), `domain6_clinical_metadata.json`.
- **Output:** Symptom severity classification (e.g. "Mild Symptomatic Profile"), atypicality flag, and per-symptom contribution (`coefficient × feature_value`).

---

## 3. Key Files & Their Roles

### `models/`
| File | Role |
|---|---|
| `feature_mappings.py` | Central lookup of feature names → human-readable display labels (e.g. `stress_score` → "General Perceived Stress Load") used when rendering driver/contributor labels in reports. Domain-specific — each domain must only reference its own feature names here. |
| `inference_wrappers.py` | Loads each domain's saved artifact from `models/saved_states/` and exposes a uniform `predict(input_row)` interface per domain. Responsible for computing the final placement value and the ranked top-contributors list for every domain. This is the most central file in the live pipeline — nearly all per-domain bugs trace back to logic here rather than to the training scripts. |
| `train_domain1_personality.py` … `train_domain6_clinical.py` | One script per domain; loads the domain's dataset, fits the architecture described in Section 2, and writes the resulting artifact(s) + metadata JSON into `models/saved_states/`. |
| `train_orchestrator.py` | Runs all six domain training scripts in sequence, regenerating every artifact in `saved_states/` from the datasets. |
| `profile_aggregator.py` | Combines all six domains' individual outputs into a single compiled profile (the `compiled_profile_eval_*.json` shape), and produces the synthesized global narrative text. |
| `pdf_generator.py` | Renders the compiled profile into the final `Mindsight_Report_*.pdf`. |
| `.schema_hash` | Hash of `schema_config.json`, used to detect when the feature schema has changed and saved models may be stale. |

### `scripts/`
| File | Role |
|---|---|
| `take_assessment.py` | CLI entry point for taking a single assessment — collects/loads responses, runs them through inference, and triggers report generation. |
| `read_metadata.py` | Reads and prints `datasets_metadata.json`/`.txt` for quick dataset inspection. |
| `flush_training_data.py` | Clears/resets training artifacts (used before a full retrain). |
| `flush_reports.py` | Clears out generated `reports/report_*/` run folders. |
| `check_domains_4_6.py`, `diagonistic_script.py` | Diagnostic utilities for printing raw model internals (e.g. feature importances, regression coefficients) directly against a given input row, to verify a domain's output is using real fitted model values rather than placeholder logic. |
| `archive/` | One-time, already-completed data-prep scripts (NHANES merge, IMP70 questionnaire generation, dataset trimming) — not part of the live pipeline; kept for provenance. |

### `tests/`
| File | Role |
|---|---|
| `run_assessments.py` | Batch-runs multiple responses through the pipeline (e.g. from `all_responses.csv`) for regression testing. |
| `responses.csv` / `all_responses.csv` | Sample response rows used for testing and manual verification. |

### Root
| File | Role |
|---|---|
| `main.py` | Top-level entry point for the project. |
| `schema_config.json` | **Authoritative feature contract** — schema version 2.6, defines exactly which 70 features belong to which domain and which dataset they come from. Any change here requires retraining affected domains. |

---

## 4. General Workflow (MOST IMPORTANT)

This is the end-to-end path data takes through MINDSIGHT, from raw questionnaire response to final PDF report.

```
 ┌─────────────────────┐
 │  70-item response    │   (matches schema_config.json feature list,
 │  (responses.csv /     │    see feature-count note below)
 │   user input)         │
 └──────────┬───────────┘
            │
            ▼
 ┌─────────────────────────────┐
 │  scripts/take_assessment.py  │  Entry point — loads the response row
 └──────────┬───────────────────┘
            │
            ▼
 ┌─────────────────────────────────────────────┐
 │       models/inference_wrappers.py            │
 │  For EACH of the 6 domains:                   │
 │   1. Select that domain's features            │
 │      from the response (per schema_config.json)│
 │   2. Load that domain's saved artifact         │
 │      from models/saved_states/                 │
 │   3. Run domain-specific prediction logic       │
 │      (GRM theta / RSE formula / LightGBM /      │
 │       RandomForest / XGBoost / LogReg+IForest)  │
 │   4. Compute placement value + ranked           │
 │      top-contributors for that domain           │
 └──────────┬─────────────────────────────────────┘
            │  (six independent domain outputs)
            ▼
 ┌─────────────────────────────────┐
 │   models/profile_aggregator.py    │
 │   Combines all 6 domain outputs   │
 │   into one compiled profile JSON  │
 │   + writes a synthesized global   │
 │   narrative paragraph              │
 └──────────┬─────────────────────────┘
            │
            ▼
 ┌─────────────────────────────────┐
 │     models/pdf_generator.py        │
 │   Renders the compiled profile     │
 │   into the final formatted PDF     │
 └──────────┬─────────────────────────┘
            │
            ▼
 ┌──────────────────────────────────────────────────────────┐
 │   reports/report_<timestamp>/                              │
 │   ├── eval_<timestamp>/                                     │
 │   │   ├── compiled_profile_eval_<timestamp>.json             │
 │   │   └── Mindsight_Report_eval_<timestamp>.pdf               │
 │   └── responses.csv   (the exact input that produced this run)│
 └──────────────────────────────────────────────────────────┘
```

**Training workflow (run separately, before any assessment can be taken):**

```
datasets/*.csv  →  models/train_domain{1..6}_*.py  →  models/saved_states/*.pkl|.json|.txt
                         (or models/train_orchestrator.py runs all six at once)
```

Every saved artifact must be regenerated any time:
- `schema_config.json` changes (check `.schema_hash` for drift), or
- a domain's training script logic changes, or
- the underlying dataset is updated.

**Verification workflow (used throughout development to catch silent inference bugs):**

```
Fixed test response row  →  run through take_assessment.py  →  compare report
output by hand against the raw input  →  if a domain's placement/drivers don't
make sense for the given raw values, trace the bug to inference_wrappers.py
(most common) or the domain's train_domain*.py / saved artifact (less common)
```

This hand-verification loop was essential during development — several domains (1, 4, 6) went through multiple rounds of fixes where placement values were silently constant, mislabeled with another domain's feature names, or unverified, despite looking superficially plausible in the rendered PDF.

> **Note on feature counts:** each domain only ever sees its own feature subset as defined in `schema_config.json` — domain 1: 15, domain 2: 12, domain 3: 12, domain 4: 18, domain 5: 10, domain 6: 7. That sums to 74 feature *references*, not 70, because `age` and `gender` are each listed independently in three domains (2, 4, and 5). Counting `age` and `gender` once each instead of three times brings the total back down to exactly 70 unique questionnaire items, matching `schema_config.json`'s `total_features: 70`. Always treat `schema_config.json` as the ground truth for exact per-domain feature lists.

---

## 5. Project Structure

```
MINDSIGHT/
├── datasets/              Source CSVs for all 6 domains + metadata
├── docs/                  Architecture reports, IMP70 questionnaire reference
├── individual_eval/       Standalone per-person evaluation runs
├── models/                Training scripts, inference logic, saved artifacts
│   ├── saved_states/      Fitted model artifacts (one set per domain)
│   └── weights/           (currently unused — reserved for future model weight files)
├── reports/                Timestamped end-to-end assessment runs (input + compiled profile + PDF)
├── scripts/                CLI utilities, diagnostics, archived one-time data-prep scripts
├── tests/                  Sample responses and batch-assessment regression tests
├── main.py                 Project entry point
└── schema_config.json      Authoritative 70-feature schema (v2.6) — source of truth for all domain feature lists
```

---

*For the detailed feature-level data dictionary (column ranges, types, valid codes), see `datasets/datasets_metadata.json` / `.txt`. For the original architecture rationale document, see `docs/mindsight_architecture_report.pdf`.*
