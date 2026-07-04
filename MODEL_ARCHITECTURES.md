# MINDSIGHT Model Architectures

| Domain | Name | Algorithm | Model Type | Artifact | Input | Output |
|--------|------|-----------|-----------|----------|-------|--------|
| 1 | Big Five Personality | Generalized Rating Model (GRM) IRT Scoring | Parametric IRT | `domain1_grm_parameters.pkl` | EXT1-3, EST1-3, AGR1-3, CSN1-3, OPN1-3 | 5 trait scores (-4.0 to +4.0) |
| 2 | Rosenberg Self-Esteem | Deterministic Sum + Reverse-Coding | None (Deterministic) | None | Q1-10 (or RSE1-10) | RSE Score 0-40 + Classification |
| 3 | Mood & Sleep | PHQ-9 Deterministic + K-Means Clustering | Unsupervised (Optional) | `domain3_mood_sleep.pkl` (optional) | DPQ010-090, SLQ300, SLQ310 | PHQ-9 Sum, Sleep Duration, Clinical Phenotype (optional) |
| 4 | Digital & Social | Tree-Based Regression + SHAP Explainer | XGBoost / LightGBM | `domain4_digital_social.pkl` | IAT1-10, loneliness1-6, age, gender | Internet Addiction Sum, Loneliness Score, Depression Risk |
| 5 | Occupational Burnout | Gradient Boosting Regression | XGBoost | `domain5_burnout.json` (XGB model) | work_hours, meetings, balance_score, etc. (10 features) | Burnout Index, Tier Label (Low/Moderate/High/Severe) |
| 6 | Severe Clinical Screening | Binary Classification + Anomaly Detection | Logistic Regression + Isolation Forest | `domain6_clinical.pkl` | unwanted_thoughts, panic, hypervigilance, etc. (7 binary features) | Condition Code (0-3), Anomaly Flag |

## Key Notes

- **Domain 1**: True IRT theta estimation over grid `[-4.0, +4.0]` (81 points)
- **Domain 2**: Pure deterministic; no ML required
- **Domain 3**: PHQ-9 always computed deterministically; optional K-Means phenotyping if model exists
- **Domain 4**: Fallback to raw sum if model missing; SHAP explains tree-based predictions
- **Domain 5**: XGBoost with metadata-driven thresholds; SHAP TreeExplainer for drivers
- **Domain 6**: Binary features; isolation-forest flagging; coefficient-based attribution
