# MINDSIGHT — Status, Architecture, & Handoff (dev branch)

Last inspected: dev branch (main.py, api.py, models/*, frontend/*)  
Purpose: a comprehensive snapshot of the repository as-run today, including system architecture, per-domain inference details, input/output schema, visualization contract, operational lifecycle, troubleshooting, and recommended next steps for improvements and AI-driven enhancements.

---

## 1. Quick summary / status snapshot

MINDSIGHT is a local/developer-hosted psychometric profiling system with:
- A Python CLI (main.py) that orchestrates training, evaluation, and archival.
- A Flask REST API (api.py) that accepts questionnaire JSON, routes to the same inference pipeline, and archives outputs.
- A multi-domain inference layer (models/inference_wrappers.py) with six domain evaluators and a profile aggregator (models/profile_aggregator.py) that synthesizes clinical and plain-language narratives.
- A lightweight Vite + React frontend in `frontend/` which is designed to consume the JSON presentation contract produced by the aggregator.
- Model artifacts are expected under `models/saved_states/`. If missing, deterministic fallbacks are available.

Current status:
- Fully wired end-to-end in code (CLI + API + aggregator + domain functions).
- ML artifacts are loaded from disk when present; code contains robust fallbacks.
- Auditability: every assessment is appended to `tests/all_responses.csv` and archived JSON placed into `reports/report_{timestamp}/`.
- Security note: admin endpoints (/train, /flush) are disabled unless `ALLOW_ADMIN_ENDPOINTS=true` is set.

---

## 2. Repo layout (key files & directories)

- `main.py` — CLI orchestrator and full E2E driver.
- `api.py` — Flask REST API (endpoints: `/assess`, `/health`, `/latest-report`, `/train`, `/flush`, `/test`).
- `schema_config.json` — canonical schema mapping used by code.
- `requirements.txt` — Python dependencies.
- `models/`
  - `inference_wrappers.py` — implementations for domain 1..6 (loads model artifacts).
  - `profile_aggregator.py` — generate_full_profile(), enrichment, synthesis, plain-language summary.
  - `feature_mappings.py` — optional human-readable mappings.
  - `saved_states/` — expected model artifacts (pickles, xgb models, metadata JSON).
- `frontend/` — Vite/React app (dev server on 5173 expected).
- `tests/`
  - `responses.csv` — single latest response ledger used by CLI; overwritten on API /assess.
  - `all_responses.csv` — appended ledger for audit trail.
- `reports/` — timestamped archives per run: `reports/report_{timestamp}/compiled_profile_{timestamp}.json`
- `individual_eval/current_report/` — last report copied here for quick download.

---

## 3. Entry points & how to run

Prereqs:
- Python 3.x, system packages for xgboost/lightgbm if used, node/npm for frontend.
- `models/saved_states/*` for ML-powered outputs (optional but recommended).

Install:
```bash
python3 -m pip install -r requirements.txt
```

Run CLI:
- Evaluate latest CSV (default action):
  ```bash
  python3 main.py --eval
  ```
  - Train all domains (manual): python3 main.py --train
  - Launch interactive survey (if script present): python3 main.py --survey

Start API (Flask):
```bash
export FLASK_DEBUG=true       # optional
python3 api.py
# server: http://localhost:5000
```

Frontend (dev):
```bash
cd frontend
npm install
npm run dev    # starts Vite dev server (default localhost:5173)
```

Notes:
- Admin endpoints require `ALLOW_ADMIN_ENDPOINTS=true` environment variable to be set.
- CLI and API both call the same aggregator function: `models.profile_aggregator.generate_full_profile()`.

---

## 4. Data & schema

Primary intake: questionnaire data as CSV or JSON. The canonical ordered headers included in code (used to write responses CSVs) are:

```
age, gender,
EXT1, EXT2, EXT3, EST1, EST2, EST3, AGR1, AGR2, AGR3, CSN1, CSN2, CSN3, OPN1, OPN2, OPN3,
Q1..Q10,
DPQ010, DPQ020, DPQ030, DPQ040, DPQ050, DPQ060, DPQ070, DPQ080, DPQ090, DPQ100,
SLQ300, SLQ310,
IAT1..IAT10,
loneliness1..loneliness6,
work_hours_per_week, meetings_per_day, work_life_balance_score, job_satisfaction_score,
deadline_pressure_score, autonomy_score, stress_score, social_support_score,
unwanted_thoughts, repetitive_behaviors, overthinking, mind_going_blank,
avoidance_social_activity, panic, hypervigilance
```

- `SLQ300`/`SLQ310` are time strings (HH:MM) and are preserved as strings.
- profile_aggregator maps some legacy input codes (e.g., DPQ010 → PHQ1, and Q1..Q10 → RSE1..RSE10) defensively for backward compatibility.

Audit/logging:
- The API writes `tests/responses.csv` (overwrite) and appends rows to `tests/all_responses.csv` (audit ledger).
- Every evaluation creates `reports/report_{timestamp}/compiled_profile_{timestamp}.json` and copies `responses.csv` there.

---

## 5. Compiled profile JSON contract (top-level keys & types)

The aggregator returns a JSON object with this structure (core keys):

```json
{
  "schema_version": "4.0",
  "id_no": "<string>",
  "age": "<int|string>",
  "sex": "<string (normalized)>",
  "domain_scores": {
    "domain_1_personality": { ... },
    "domain_2_self_esteem": { ... },
    "domain_3_mood_sleep": { ... },
    "domain_4_digital_and_social": { ... },
    "domain_5_occupational_burnout": { ... },
    "domain_6_severe_clinical": { ... }
  },
  "global_synthesis": "<string>",         # clinical-register narrative
  "plain_language_summary": {
    "opening": "<string>",
    "domain_recap": "<string>",
    "closing": "<string>",
    "recommend_professional_help": true|false,
    "full_text": "<string>"
  }
}
```

Per-domain `domain_scores[domain_key]` has:
- `domain` (string)
- `placement` (object) — numeric outputs or categorical labels for the domain
- `top_contributors` (array) — list of `{ feature, display_name, contribution, direction }` and aggregator adds `relative_magnitude` (0-100)
- `severity_tier` — normalized enum (e.g., low/moderate/high/severe/descriptive)
- `domain_summary` — one-line human readable sentence
- `scale_reference` — presentation hint (one of):
  - `{"type":"fixed_range","min":X,"max":Y}`
  - `{"type":"reference_band","low":a,"high":b}`
  - `{"type":"tier_thresholds","thresholds":{...}}`
  - `null` (explicitly telling frontend not to invent min/max)

Use these fields directly for frontend charts and alerts.

---

## 6. Per-domain architecture & implementation details

Below is an implementation-level summary for each domain, including expected artifacts, inputs, outputs and fallbacks.

### Domain 1 — Big Five Personality (GRM / IRT)
- Function: `evaluate_domain1_personality(raw_payload)`
- Artifact: `models/saved_states/domain1_grm_parameters.pkl`
  - Expected shape: `grm_registry[trait_name]["items"][item_id]` with item params `"a"` and `"b"` thresholds
- Algorithm:
  - If pickle available: authentic GRM IRT scoring over theta grid `np.linspace(-4.0, 4.0, 81)`.
  - Else fallback: mean of item responses (reverse-code where needed), convert to IRT-like scale `(mean - 3)*1.33`.
- Output:
  - `placement`: trait values (extraversion, emotional_stability, agreeableness, conscientiousness, openness)
  - `top_contributors`: top 3 traits by absolute score
- Presentation:
  - `scale_reference` = `fixed_range` min=-4.0 max=4.0

### Domain 2 — Rosenberg Self-Esteem
- Function: `evaluate_domain2_self_esteem(raw_payload)`
- Artifact: none (deterministic).
- Algorithm: map Q1..Q10 or RSE1..RSE10 from 1..5 → 0..4; reverse-code items 3,5,8,9,10; sum → 0..40.
- Output:
  - `placement.score`, `placement.max_possible_score` = 40, `placement.classification` (High/Normal/Low).
- Presentation:
  - `scale_reference` = fixed_range min=0 max=40

### Domain 3 — Mood & Sleep (PHQ-9 + KMeans phenotypes)
- Function: `evaluate_domain3_mood_sleep(raw_payload)`
- Artifact: optional `models/saved_states/domain3_mood_sleep.pkl`, metadata JSON may be present.
- Algorithm:
  - Deterministic PHQ-9 sum computed from DPQ010..DPQ090 (handles refusal codes), map to severity labels.
  - Parse `SLQ300`/`SLQ310` times → compute sleep duration with midnight wrap logic.
  - Optional KMeans clustering if model present: cluster profile label inserted into placement.
- Output:
  - `placement.phq9_sum`, `placement.severity_label`, `placement.calculated_sleep_duration_hours`, `placement.clinical_phenotype` (optional)
- Presentation:
  - `scale_reference` includes `phq9_sum` fixed_range 0..27 and `sleep_duration_hours` reference_band low=7 high=9

### Domain 4 — Digital & Social (IAT + Loneliness + unified ML)
- Function: `evaluate_domain4_multitask(raw_payload)`
- Artifact: `models/saved_states/domain4_digital_social.pkl`, metadata `domain4_digital_social_metadata.json`
  - Pickle must contain `depression_risk_model`.
  - metadata contains `features` list.
- Algorithm:
  - Deterministic fallback: sum of IAT and loneliness items.
  - If model present: build DataFrame row, `predict` depression risk, use `shap.TreeExplainer` to extract per-feature contributions (extracts IAT* and loneliness* features).
- Output:
  - `placement.predicted_total_internet_addiction`, `placement.predicted_total_loneliness`, `placement.loneliness_score`, `placement.predicted_depression_risk`, `placement.data_source`
- Presentation:
  - `scale_reference` = `null` (frontend must not invent bounds; show plain numbers or rely on classification)

### Domain 5 — Occupational burnout (XGBoost)
- Function: `evaluate_domain5_burnout(raw_payload)`
- Artifact: `models/saved_states/domain5_burnout.json` (xgboost model) and `domain5_burnout_metadata.json` (features + thresholds).
- Algorithm:
  - Build input vector from features in metadata, run xgboost model to predict continuous burnout index.
  - Use thresholds in metadata → map to tier_label.
  - Attempt `shap.TreeExplainer` for feature contributions; fallback to raw values if shap fails.
- Output:
  - `placement.burnout_index`, `placement.burnout_tier_label`, `placement.tier_thresholds` (mirrors metadata thresholds)
- Presentation:
  - Use `tier_thresholds` to draw colored bands on a burnout gauge.

### Domain 6 — Severe Clinical Screening (classifier + anomaly detector)
- Function: `evaluate_domain6_clinical(raw_payload)`
- Artifact: `models/saved_states/domain6_clinical.pkl`, metadata `domain6_clinical_metadata.json`
  - Pickle should contain `classifier` and `anomaly_detector`.
  - Metadata must list binary features used by classifier.
- Algorithm:
  - Convert inputs to binary vector (1 if feature > 0, else 0).
  - classifier.predict → predicted_condition (integer code 0..3)
  - anomaly_detector.predict → anomaly flag (isolation forest style with -1 meaning anomaly).
  - If classifier has `coef_`, compute coefficient-derived contributions for top_contributors.
- Output:
  - `placement.predicted_condition_code`, `placement.predicted_condition_label`, `placement.anomaly_review_flag`
- Presentation:
  - Categorical badge, explicit anomaly banner if `anomaly_review_flag` is true.

---

## 7. Inference & aggregation contract (how data flows)

- Input ingestion (CLI): `main.py` reads `tests/responses.csv` and selects last row; writes to `tests/all_responses.csv`.
- API ingestion: `api.py` accepts JSON POST to `/assess`, writes `tests/responses.csv` (overwrite) and appends to `tests/all_responses.csv`. It creates a temporary CSV and reuses the same sanitation/inference path as CLI.
- Sanitization: numeric-like strings converted to int/float; times preserved.
- Main aggregator flow:
  - `generate_full_profile()` invokes the six `evaluate_domainX_*()` functions.
  - `extract_domain_signals()` reads canonical numbers from raw domain outputs (e.g., phq_sum, rse score, burnout index, loneliness).
  - `evaluate_cross_domain_synthesis()` (clinical narrative) and `generate_plain_language_summary()` (person-facing) both call `extract_domain_signals()` so they derive from identical base numbers.
  - `enrich_domain_outputs()` calculates `relative_magnitude` for contributors; sets `severity_tier`, `domain_summary`, and `scale_reference`.
- Final steps:
  - Save JSON to `reports/report_{timestamp}/compiled_profile_{timestamp}.json`
  - Copy response CSV into the same archive and copy to `individual_eval/current_report/`
  - CLI cleans intermediate eval sandbox.

---

## 8. Visualization mapping (how frontend should use fields)

The aggregator intentionally provides a stable "presentation contract" — prefer these exact fields:

- Primary overview:
  - `plain_language_summary.full_text` (present for the person)
  - `global_synthesis` (present for clinician/reviewer)
  - `plain_language_summary.recommend_professional_help` (boolean) → high-priority alert

- Per domain (each `domain_scores[domain_key]`):
  - Title: use `domain_data["domain"]` or present a friendly label mapping in the UI.
  - One-line: `domain_data["domain_summary"]`.
  - Severity: `domain_data["severity_tier"]`.
  - Primary visualization:
    - If `scale_reference.type == "fixed_range"` → draw a gauge with domain_data.placement value within `[min, max]`.
    - If `scale_reference.type == "reference_band"` → numeric value with shaded normal band `[low, high]`.
    - If `scale_reference.type == "tier_thresholds"` → gauge with colored bands per threshold.
    - If `scale_reference` is null → show plain numeric and classification; avoid bars.
  - Top contributors:
    - Use `top_contributors[].display_name`, `relative_magnitude` (0..100), `direction` for color coding.
    - If SHAP-like weights were computed, show them (but `relative_magnitude` is always available post-aggregation).
  - Clinical flags:
    - For Domain 6 `anomaly_review_flag` → prominent "Review Required" indicator.

- Trend & archive:
  - `reports/` folder contains historical JSONs; UI can allow selecting previous compiled_profile JSONs and draw time-series for numeric placements (burnout_index, phq9_sum, rse_pct).
  - Provide an "Audit" view that displays `tests/all_responses.csv` rows with links to corresponding `reports/report_{timestamp}`.

---

## 9. Model artifact expectations & metadata contracts

Files (expected names & keys):
- `models/saved_states/domain1_grm_parameters.pkl`
  - structure: `grm_registry[trait_name]["items"][item_id]["a"]` and `["b"]` thresholds
- `models/saved_states/domain3_mood_sleep.pkl`
  - object with keys: `"kmeans_model"`, `"scaler"`, `"cluster_profiles"` (dict map str(idx)→name)
- `models/saved_states/domain4_digital_social.pkl`
  - must include `"depression_risk_model"`; metadata JSON `domain4_digital_social_metadata.json` includes `features` list.
- `models/saved_states/domain5_burnout.json`
  - xgboost model file; accompanied by `domain5_burnout_metadata.json` with `features` and `thresholds` = `{ low_to_moderate, moderate_to_high, high_to_severe }`
- `models/saved_states/domain6_clinical.pkl`
  - model_payload with `classifier` and `anomaly_detector`, metadata `domain6_clinical_metadata.json` with `features` list

Metadata JSON conventions:
- `features`: array of string feature names in exact ordering models expect.
- `thresholds` (domain 5): exact numeric thresholds used in mapping continuous predictions to tier labels.

---

## 10. Observability, logs, and health

- `api.py` has `DEBUG` driven by `FLASK_DEBUG` env var. It uses `debug_print()` that prints timestamped messages if `DEBUG` is true.
- `/health` endpoint returns `models_trained` boolean determined by presence of saved_states files and match between `schema_config.json` md5 and `models/.schema_hash`.
- To enable admin functionality:
  ```bash
  export ALLOW_ADMIN_ENDPOINTS=true
  ```
- Admin endpoints:
  - `POST /train` → triggers `python main.py --train` (subprocess).
  - `POST /flush { "target": "train|reports|eval|all" }` → flushes directories.

---

## 11. Troubleshooting & known failure modes

1. Missing model artifacts in `models/saved_states/`:
   - Symptom: many domains return deterministic fallback outputs or indicate `data_source: "fallback_raw_sum"`.
   - Fix: place trained artifacts in `models/saved_states/` with names expected above. If unavailable, accept fallback behavior.

2. Schema drift:
   - Symptom: `models_trained` false in `/health` even if saved_states exists.
   - Cause: `schema_config.json` changed. Main writes `models/.schema_hash` after training. Mismatch forces retraining.
   - Fix: retrain models (run training scripts) or sync `.schema_hash` with current schema if intentional (not recommended without retrain).

3. SHAP explainer errors:
   - Symptom: SHAP-related try/except prints, `top_contributors` empty or fallback.
   - Fix: ensure model is tree-based (xgboost/lightgbm), shap package installed, and model payload compatible with `shap.TreeExplainer`.

4. API returns 500 on `/assess`:
   - Check app logs (enable `FLASK_DEBUG=true`), inspect stacktrace in response if debug is enabled.
   - Confirm `tests/` directory exists and is writable; temp CSV creation may fail on some restricted environments.

5. Time parsing (SLQ300/SLQ310):
   - If time strings are missing or malformed, aggregator uses defaults (`23:00`/`07:00`) or `parse_time` fallback 12.0 depending on function.

---

## 12. Testing & CI suggestions (immediate)

Add these tests and CI steps:

- Unit tests (pytest):
  - `tests/test_profile_aggregator.py`: verify `generate_full_profile()` with the `sample_payload` included in `profile_aggregator.py` returns expected keys and types.
  - Domain-level tests: small fixtures to assert deterministic fallbacks and presence of `scale_reference`.
  - Serialization test: ensure `MindsightNumpyEncoder` serializes numpy ints/floats/arrays correctly.

- Integration tests:
  - `tests/test_api_endpoints.py`: spawn Flask app (test client), POST a known JSON to `/assess` and assert `200` and that files were archived.
  - `tests/test_cli_workflow.py`: test `python main.py --eval` with a temporary tests/responses.csv.

- CI:
  - Run `pip install -r requirements.txt`, run `pytest`, run linting (flake8).
  - Consider adding a GitHub Actions pipeline file that runs tests on PRs.

---

## 13. Security & privacy recommendations

- Personal identifiers:
  - `id_no` fallback can be `MS-{timestamp}-ANONYMOUS` — treat any provided `id`/`id_no` as PII.
  - Ensure `tests/all_responses.csv` and `reports/` are handled per policy: consider redaction or encryption or per-environment retention.
- Admin endpoints:
  - Keep `ALLOW_ADMIN_ENDPOINTS` default `false`. If enabling, expose behind firewall or auth.
- Suggested production practices:
  - Do not run the API with debug mode in production (`FLASK_DEBUG` default false).
  - Add authentication (JWT or API key) for `/assess` and definitely for `/train`, `/flush`.
  - If deployed to cloud, use encrypted storage and secrets manager for model artifact credentials (if any).
  - Consider a data retention policy: e.g. rotate/expire `reports/` after X days or move to secure storage.

---

## 14. Roadmap & prioritized enhancements (recommended)

High priority
1. Add tests & CI (see section 12).
2. Provide a Dockerfile and docker-compose so the entire stack (Flask API + frontend) can be launched reproducibly.
3. Add clear model training scripts or a `train/README.md` describing how to produce artifacts with exact metadata schema.
4. Protect admin endpoints with authentication and move `ALLOW_ADMIN_ENDPOINTS` control to an environment + secrets system.

Medium priority
1. Standardize model artifact format and metadata (single JSON schema for metadata describing `features`, shapes, thresholds, model type).
2. Add model validation on load: a function that verifies presence of expected keys and that feature lists match schema_config.json.
3. Add more defensive logging around SHAP and scaler transforms to capture mismatch issues.
4. Add an endpoint `/reports` to list available archives programmatically for the frontend.

Longer-term / research
1. Introduce a model registry (MLflow or DVC) for artifact versioning and reproducibility.
2. Add A/B experiments for model changes; add evaluation metrics & stored ground-truth where available.
3. Add differential privacy or data minimization for storage if handling real PII.

---

## 15. Checklist for an AI consultant accepting this handoff

Tasks the consultant can start with (ordered):
- Run the test example: call the aggregator `generate_full_profile(sample_payload)` (in `models/profile_aggregator.py` top-level sample).
- Wire automated tests covering each domain fallback path.
- Add Dockerfile + docker-compose for `api` and `frontend` and ensure `reports/` persists with a volume.
- Prepare a model artifact validation script that validates metadata and prints a summary of all `models/saved_states` artifacts (existence, expected keys, shapes).
- Create a minimal frontend page (DomainCard) that renders each `domain_scores` object using the `scale_reference` contract.
- Harden admin endpoints with token-based auth and audit logging.

---

## 16. Quick commands & file references

- Run evaluation (CLI):
  ```bash
  python3 main.py --eval
  ```
- Run API:
  ```bash
  FLASK_DEBUG=true python3 api.py
  ```
- Train via CLI:
  ```bash
  python3 main.py --train
  ```
- Trigger train via API (admin):
  ```bash
  export ALLOW_ADMIN_ENDPOINTS=true
  curl -X POST http://localhost:5000/train
  ```
- Check latest report:
  ```bash
  ls -la reports | tail
  cat individual_eval/current_report/compiled_profile_*.json
  ```

Key files to inspect:
- `main.py`, `api.py`
- `models/profile_aggregator.py`, `models/inference_wrappers.py`, `models/feature_mappings.py`
- `models/saved_states/*`
- `schema_config.json`, `tests/responses.csv`, `tests/all_responses.csv`
- `frontend/src/*` (pages and components will be wired to the JSON contract)

---

## 17. Appendices

A. Example fields & UI wiring excerpt for DomainCard component:
- props:
  - `domainKey` (e.g., "domain_1_personality")
  - `domainData = profile.domain_scores[domainKey]`
- read:
  - `title = domainData.domain`
  - `summary = domainData.domain_summary`
  - `severity = domainData.severity_tier`
  - `placement = domainData.placement` (pick the main numeric to show)
  - `scaleReference = domainData.scale_reference`
  - `contributors = domainData.top_contributors` (use display_name & relative_magnitude)

B. File ownership suggestions (who to contact in team):
- Add a MAINTAINERS or OWNERS file with names and emails for model engineers, backend, frontend.

---

If you want, I can next:
- produce a ready-to-commit file by creating `STATUS.md` in the repository (done), or
- generate sample compiled_profile JSON from the sample payload included in `models/profile_aggregator.py` to help build frontend prototypes, or
- create a Dockerfile + docker-compose.yaml scaffold targeting current code with careful binds for `models/saved_states`, `reports`, and `tests`.

Which one should I do now?
