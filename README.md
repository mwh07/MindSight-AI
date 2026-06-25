# MINDSIGHT

**Unified Multi-Domain Psychological Profiling System**
*System Architecture v4.0 · Schema Version 2.6*

MINDSIGHT is a full-stack web application that collects a 70-item questionnaire response and produces a six-domain psychological profile — **Personality**, **Self-Esteem**, **Mood & Sleep**, **Digital & Social Behaviour**, **Occupational Burnout**, and **Severe Clinical Screening** — culminating in a dynamic, frontend-rendered dashboard.

> All terminology throughout the app is aligned with a **research framing, not a clinical-grade diagnostic one**. Nothing MINDSIGHT produces is a diagnosis.

---

## Table of Contents

1. [Datasets](#1-datasets)
2. [Domain Model Architectures](#2-domain-model-architectures)
3. [Key Files & Their Roles](#3-key-files--their-roles)
4. [System Workflow](#4-system-workflow)
5. [Frontend Architecture](#5-frontend-architecture)
6. [Project Structure](#6-project-structure)
7. [How to Run](#7-how-to-run)
8. [Notes on Key Changes](#8-notes-on-key-changes)

---

## 1. Datasets

All datasets live in `datasets/`. Each domain is trained on exactly one source dataset, and the **authoritative feature list per domain is defined in `schema_config.json`** (v2.6, 70 total features) — that file is the single source of truth.

| Dataset file | Used by | Feature columns used |
|---|---|---|
| `big_five_personality_clean.csv` | Domain 1 — Personality | `EXT1–EXT3`, `EST1–EST3`, `AGR1–AGR3`, `CSN1–CSN3`, `OPN1–OPN3` (3 items per trait, 15 total) |
| `rosenberg_self_esteem_clean.csv` | Domain 2 — Self-Esteem | `age`, `gender`, `Q1–Q10` (10 RSE items) |
| `nhanes_joined_mood_sleep.csv` | Domain 3 — Mood & Sleep | `DPQ010–DPQ100` (10 PHQ-9 items), `SLQ300` (bedtime), `SLQ310` (wake time) |
| `internet_phq_loneliness_clean.csv` | Domain 4 — Digital & Social | `age`, `gender`, `IAT1–IAT10`, `loneliness1–loneliness6` |
| `tech_burnout_2026_clean.csv` | Domain 5 — Occupational Burnout | `age`, `gender`, `work_hours_per_week`, `meetings_per_day`, `work_life_balance_score`, `job_satisfaction_score`, `deadline_pressure_score`, `autonomy_score`, `stress_score`, `social_support_score` |
| `ocd_symptoms_clean.csv` | Domain 6 — Severe Clinical Screening | `unwanted_thoughts`, `repetitive_behaviors`, `overthinking`, `mind_going_blank`, `avoidance_of_social_activity`, `panic`, `hypervigilance` (7 binary flags) |

**Supporting files:**
- `datasets_metadata.json` / `.txt` — column ranges, types, and provenance notes.
- `scripts/read_metadata.py` — prints the metadata for quick inspection.

> `nhanes_joined_mood_sleep.csv` is a pre-joined output of `scripts/archive/merge_nhanes_mood_sleep.py` — a one-time data-prep step, not part of the live pipeline.

---

## 2. Domain Model Architectures

Each domain has its own training script (`models/train_domain*_*.py`) and produces artifacts saved into `models/saved_states/`. No single architecture is used everywhere — each one was chosen because it fits how that specific instrument actually behaves, and because the alternatives have specific, real drawbacks for that domain.

### Domain 1 — Personality (Big Five)
- **Architecture:** Graded Response Model (GRM), Item Response Theory — one model fit per trait.
- **Why:** A simple average of a trait's 3 items treats every item as equally informative and assumes responses move in a straight line with the underlying trait — neither is true in practice. Some items discriminate sharply between people near the middle of a trait, others are more informative at the extremes, and the jump between "somewhat agreeable" and "agreeable" isn't the same size as the jump between "agreeable" and "very agreeable." GRM models each item's own discrimination and threshold parameters and estimates a latent trait score (θ) from the *pattern* of responses, not their raw sum — which is also why two traits with similarly-averaged raw items can now correctly land at different θ values instead of collapsing toward the same number, a failure mode the simpler approach was prone to.
- **Artifacts:** `domain1_grm_parameters.pkl`, `domain1_grm_metadata.json`
- **Output:** θ per trait + per-item contribution.

### Domain 2 — Self-Esteem (Rosenberg Scale)
- **Architecture:** Deterministic scoring + empirical percentile lookup (no ML).
- **Why:** The Rosenberg Self-Esteem Scale has a fixed, decades-old scoring procedure with known psychometric behavior — reverse-coding the negatively worded items and summing to a 0–40 range. Training a model to *approximate* a relationship that's already exactly defined would only introduce approximation error on top of a ground truth we already have; machine learning earns its keep when the underlying relationship is unknown, which isn't the case here. What a fixed formula alone can't provide is *context* — the percentile lookup against real cohort data (matched by age/gender) is what tells you where this person's raw score actually sits relative to real respondents, rather than just reporting a number in isolation.
- **Artifacts:** `domain2_self_esteem.pkl`, `domain2_self_esteem_percentiles.json`, `domain2_self_esteem_metadata.json`
- **Output:** RSE total score (0–40), classification, per-item contributions.

### Domain 3 — Mood & Sleep (PHQ-9 + Sleep Duration)
- **Architecture:** Deterministic PHQ-9 sum + sleep-duration calculation.
- **Why:** PHQ-9 is the same situation as domain 2 — a validated clinical screening instrument with a standardized scoring key, not something to be learned from a finite dataset. Sleep duration is arithmetic, not a modeling problem at all: it's the time between a parsed bedtime and wake time, with mod-24 wraparound to correctly handle overnight sleep crossing midnight. The risk here was never "wrong architecture," it was getting the parsing and wraparound logic exactly right — which is exactly where the real bugs in this project turned out to live.
- **Artifacts:** `domain3_mood_sleep.txt`, `domain3_mood_sleep_metadata.json`
- **Output:** PHQ-9 sum, severity band, sleep hours.

### Domain 4 — Digital & Social Dynamics
- **Architecture:** Two independent `RandomForestRegressor` models (Internet Addiction + Loneliness).
- **Why:** Internet Addiction Test items and the loneliness scale's items were never validated together as one combined construct — they're two separately-designed instruments measuring related but distinct things. Fusing them into a single multi-output model would implicitly assume a shared latent structure that was never actually established, so two independent models are used instead, each respecting its own instrument's boundaries. Random forests were chosen over a linear model because Likert-item interactions here are plausibly non-linear (e.g. specific combinations of usage patterns, not just their sum, may matter), and the model's feature importances give a direct, per-person explanation of which specific items drove that person's score.
- **Artifacts:** `domain4_digital_social.pkl`, `domain4_digital_social_metadata.json`
- **Output:** Addiction score, loneliness score, top contributors (via SHAP).

### Domain 5 — Occupational Burnout
- **Architecture:** XGBoost Regressor.
- **Why:** Burnout plausibly depends on *interactions* between workplace factors rather than each factor's isolated effect — a heavy meeting load might only meaningfully predict burnout when paired with low autonomy, not on its own. A linear model would need every such interaction specified by hand in advance; gradient-boosted trees capture these interactions automatically from the data. SHAP-based explainability specifically (not explainability in the abstract) matters here because two people can land at the same burnout index for entirely different underlying reasons, and SHAP traces *this specific person's* result back to *their specific* contributing factors rather than giving a generic, population-level explanation.
- **Artifacts:** `domain5_burnout.json`, `domain5_burnout_metadata.json`
- **Output:** Burnout index (0–10), severity tier, top contributors (SHAP).

### Domain 6 — Severe Clinical Screening
- **Architecture:** `LogisticRegression` (multi-class) + `IsolationForest` (anomaly detection).
- **Why:** This is the domain where interpretability matters most, so a coefficient-based logistic regression was chosen over a higher-capacity black-box classifier — for a clinically-adjacent result, being able to show the exact coefficient behind each symptom's contribution matters more than a marginal accuracy gain from a harder-to-explain model. The Isolation Forest runs as a genuinely separate model answering a genuinely separate question: severity classification asks *how concerning is this response pattern*, while anomaly detection asks *does this response pattern resemble anything the model has seen before at all* — a person can score "mild" on severity while still answering in a combination the model finds structurally unusual, and that's a distinct, real signal that severity scoring alone cannot surface.
- **Artifacts:** `domain6_clinical.pkl`, `domain6_clinical_metadata.json`
- **Output:** Condition code/label, anomaly flag, per-symptom contributions.

---

## 3. Key Files & Their Roles

### Backend (`models/`, `api.py`)

| File | Role |
|---|---|
| `api.py` | Flask REST API — serves the frontend, receives assessment data, runs the inference pipeline, saves results, and serves the latest report. |
| `models/inference_wrappers.py` | Core inference engine — loads each domain's artifact, runs prediction, computes placement values and top-contributors. |
| `models/profile_aggregator.py` | Orchestrator — combines all six domain outputs into a single compiled profile, adds `severity_tier`, `domain_summary`, `relative_magnitude`, and `scale_reference`, and generates the `global_synthesis` and `plain_language_summary`. |
| `models/feature_mappings.py` | Maps feature IDs → human-readable labels (used in the dashboard). |
| `schema_config.json` | Authoritative feature contract — defines exactly which 70 features belong to which domain. |
| `models/train_domain*.py` | One training script per domain — regenerates the saved artifacts from `datasets/`. |
| `main.py` | Legacy CLI entry point — can be used to train models via `python main.py --train` or run assessments. |

### Frontend (`frontend/`)

| File | Role |
|---|---|
| `src/pages/Assessment.jsx` | The 7-step questionnaire UI — collects age, gender, and all 70 responses. |
| `src/pages/Dashboard.jsx` | Main results view — fetches the latest report from `/latest-report`, displays radar charts, gauges, domain summaries, contributors, and the plain-language summary. |
| `src/lib/apiClient.js` | Axios client for all backend API calls (`/assess`, `/latest-report`, etc.). |
| `src/lib/questionnaire-data.js` | Full question bank — 70 items across 6 domains, with scales and metadata. |
| `src/components/results/` | Reusable chart components (`RadarChart`, `BarChartCard`, `ScoreGauge`, `RecommendationCard`). |
| `vite.config.js` | Vite configuration with proxy to `http://localhost:5000` (Flask). |

### Scripts & Utilities

| File | Role |
|---|---|
| `scripts/take_assessment.py` | Legacy CLI entry point — kept for batch testing, but the main pipeline now runs via the Flask API. |
| `scripts/flush_reports.py` | Clears all generated reports. |
| `scripts/flush_training_data.py` | Resets training artifacts. |
| `scripts/check_domains_4_6.py` | Diagnostic utility — prints raw model outputs for a given input row. |
| `tests/run_assessments.py` | Batch-runs multiple responses for regression testing. |

---

## 4. System Workflow

This is the end-to-end path data takes through MINDSIGHT, from user input to dashboard display.

```
User completes the IMP-70 questionnaire
in the React frontend
        │
        ▼
POST /assess (Flask API, api.py)
JSON payload with age, gender, 70 items
        │
        ▼
models/inference_wrappers.py
For EACH of the 6 domains:
  1. Select that domain's features
  2. Load the saved artifact
  3. Run prediction (GRM / RSE formula /
     RandomForest / XGBoost / LogReg)
  4. Compute placement + top contributors
        │ (six independent domain outputs)
        ▼
models/profile_aggregator.py
Combines all 6 domain outputs
Adds severity_tier, domain_summary,
relative_magnitude, scale_reference,
global_synthesis, and
plain_language_summary
        │
        ▼
individual_eval/current_report/
  compiled_profile_<timestamp>.json (latest)
  responses.csv (raw input)
reports/report_<timestamp>/ (archive)
  compiled_profile_<timestamp>.json
  responses.csv
        │
        ▼
Response to frontend
{status, timestamp}
        │
        ▼
Frontend navigates to /dashboard
GET /latest-report
        │
        ▼
Dashboard renders all domains, charts,
summaries, and the plain-language summary
```

### Training Workflow

```
datasets/*.csv → models/train_domain{1..6}_*.py → models/saved_states/*.pkl|.json|.txt
                  (or use python main.py --train to run all six)
```

### Verification Workflow

- Use `scripts/take_assessment.py` with a fixed test row to compare raw inputs against the generated JSON.
- Use `scripts/check_domains_4_6.py` to inspect raw model outputs.

---

## 5. Frontend Architecture

The frontend is a React application built with **Vite**, **Tailwind CSS**, and **Shadcn/ui** components.

### Key Pages

| Page | Route | Description |
|---|---|---|
| Home | `/` | Landing page with hero, features, and dimensions sections. |
| Assessment | `/assessment` | 7-step questionnaire (demographics + 6 domains). |
| Dashboard | `/dashboard` | Full results view — dynamic, powered by the latest report. |
| About | `/about` | Project information. |

### Data Flow

1. **Assessment** → collects responses → `POST /assess` → backend processes → returns timestamp.
2. **Dashboard** → `GET /latest-report` → fetches the most recent `compiled_profile_*.json` → renders all charts, cards, and summaries.
3. All data is fetched dynamically — no static output is hardcoded into the frontend.

### Styling

- Tailwind CSS with a custom dark-mode-ready palette.
- Shadcn/ui components for consistent, accessible UI.
- Framer Motion for smooth transitions and animations.

---

## 6. Project Structure

```
MINDSIGHT/
├── datasets/                   # Source CSVs + metadata
├── docs/                       # IMP70 questionnaire, architecture report
├── frontend/                   # React + Vite frontend
│   ├── public/                 # Static assets
│   └── src/
│       ├── components/         # UI components (landing, assessment, results)
│       ├── lib/                # API client, questionnaire data, utilities
│       ├── pages/               # Home, Assessment, Dashboard, About
│       ├── utils/                # Axios client
│       ├── App.jsx
│       ├── main.jsx
│       └── index.css
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
├── individual_eval/             # Per-assessment outputs
│   └── current_report/          # Latest report (overwritten each time)
├── models/                      # Training scripts, inference, saved states
│   ├── saved_states/             # All fitted artifacts
│   ├── inference_wrappers.py     # Core prediction engine
│   ├── profile_aggregator.py     # Combines domains, adds summaries
│   ├── feature_mappings.py       # Feature -> display name mapping
│   └── train_domain*.py          # One per domain
├── reports/                      # Archived reports (timestamped folders)
├── scripts/                      # CLI utilities, diagnostics, archive
├── tests/                        # Sample responses, batch tests
├── api.py                        # Flask REST API (main entry point)
├── main.py                       # Legacy CLI entry point (can train models)
├── schema_config.json            # Authoritative 70-feature schema
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 7. How to Run

### Prerequisites

- Python 3.8+ (with pip)
- Node.js 16+ (with npm or yarn)

### Backend (Flask API)

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. (Optional) Train the models — only needed once.
# Run each domain's training script individually:
python models/train_domain1_personality.py
python models/train_domain2_self_esteem.py
python models/train_domain3_mood_sleep.py
python models/train_domain4_multitask.py
python models/train_domain5_burnout.py
python models/train_domain6_clinical.py

# OR use the main.py entry point (if it supports training):
python main.py --train

# 3. Start the Flask server
python api.py
```

The API will be available at: `http://localhost:5000`

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check — returns `models_trained` status. |
| `/test` | GET | Simple test endpoint. |
| `/assess` | POST | Submit an assessment — expects JSON with age, gender, and all 70 items. |
| `/latest-report` | GET | Returns the most recent `compiled_profile_*.json` from `individual_eval/current_report/`. |
| `/train` | POST | Trigger model retraining (calls `main.py --train`). |
| `/flush` | POST | Flush training data, reports, or eval directories. |

### Frontend (React + Vite)

```bash
# 1. Navigate to the frontend folder
cd frontend

# 2. Install dependencies
npm install

# 3. Start the development server
npm run dev
```

The frontend will be available at: `http://localhost:5173`

The Vite dev server proxies `/api/*` requests to `http://localhost:5000`, so you do not need to worry about CORS in development.

### Production Build

```bash
# Build the frontend
cd frontend
npm run build

# The built files will be in frontend/dist/
```

You can serve the `frontend/dist/` folder with any static server, or configure the Flask app to serve it directly.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VITE_API_BASE_URL` | `http://localhost:5000` | Backend API base URL (used by the frontend). |

Create a `.env` file in `frontend/` to override the default.

### Running with Docker (optional)

```bash
# Build the Docker image
docker build -t mindsight .

# Run the container
docker run -p 5000:5000 -p 5173:5173 mindsight
```

---

## 8. Notes on Key Changes

- **PDF generation has been removed** — the dashboard is now the sole presentation layer.
- **Frontend/Backend separation** — the codebase is cleanly split into `frontend/` (React) and the Python backend.
- **`repetitive_behaviors` typo has been fixed globally** (frontend, schema, inference, and feature mappings).
- **Relative magnitude** is now computed as the fraction of total absolute contribution (sums to 100% per domain).
- **Risk labels** now display as `Risk: low` / `Risk: severe` (with `Descriptive` for Personality, since trait position is not a severity scale).
- **All terminology is aligned with a research (not clinical-grade) framing.**

For the detailed feature-level data dictionary (column ranges, types, valid codes), see `datasets/datasets_metadata.json`. For the original architecture rationale document, see `docs/mindsight_architecture_report.pdf`.
