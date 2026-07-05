# MINDSIGHT — System Architecture

## 1. System Overview

MINDSIGHT scores a person's assessment responses across six independent psychological and occupational domains, each implemented as a self-contained scoring pipeline (training script → saved model artifact + metadata → inference wrapper → aggregator). The domains do not share a model architecture by design: each domain's algorithm was chosen to match the statistical nature of its construct and its underlying instrument, rather than standardizing on a single modeling approach across the system.

Three design principles run across all six domains and are referenced throughout this document:

1. **Validated instruments are scored deterministically, not statistically.** Where a domain is built on a published, validated psychometric instrument with its own fixed scoring rule (PHQ-9 in Domain 3, the Rosenberg Scale in Domain 2), that rule is implemented exactly as published. Machine learning is not used to approximate or replace a validated instrument's own scoring algorithm — doing so would trade a known-valid score for a statistical approximation of it, which is a regression, not an upgrade.
2. **Model transparency scales with clinical stakes.** Domain 6 (severe clinical screening) uses coefficient-transparent logistic regression rather than a harder-to-audit ensemble; Domain 5 (occupational burnout) uses gradient-boosted trees with SHAP attribution; Domain 1 (personality) uses a fully parametric psychometric model. The domain with the most sensitive subject matter has the most auditable architecture.
3. **No silent data fabrication.** Domains 5 and 6, the two most recently redesigned, follow a hard-fail philosophy: missing datasets, missing columns, unmapped categorical values, or insufficient class diversity cause the training pipeline to stop with an explicit error rather than silently substituting synthetic data, zero-filling, or inventing labels. This is called out explicitly in each domain's section below where it applies.

---

## 2. Domain 1 — Big Five Personality

**Construct:** Five-factor personality model (Extraversion, Emotional Stability, Agreeableness, Conscientiousness, Openness).

**Input schema:** 15 items, 3 per trait — `EXT1-3`, `EST1-3`, `AGR1-3`, `CSN1-3`, `OPN1-3` — drawn from `big_five_personality_clean.csv`.

**Architecture: Generalized Rating (Graded Response) Model — parametric Item Response Theory.**

Rather than summing item scores, each trait is estimated as a latent continuous ability (theta, θ) using a Graded Response Model (GRM), the standard IRT approach for ordinal (Likert-type) items:

- Each item is characterized by a **discrimination parameter** (how sharply the item distinguishes between people at different trait levels) and a set of **threshold/difficulty parameters** (the trait level at which a respondent becomes more likely to endorse each successive response category).
- Given a respondent's answers, the most likely θ is estimated by maximizing the likelihood of the observed responses over a fixed evaluation grid: **81 points spanning θ ∈ [-4.0, +4.0]**.
- This is done independently per trait (5 separate GRM item banks, one per Big Five dimension), using only that trait's 3 items.

**Why IRT over deterministic summing:** a raw sum treats every item as equally informative and equally difficult, which is rarely true — some items discriminate between high and low trait levels much better than others. IRT weights each item's contribution by its measured discrimination, and additionally yields a precision estimate (standard error of measurement) that varies across the trait range — typically most precise away from the extremes for a fixed item bank, which matters for any downstream decision that treats scores near a threshold differently from scores in the middle of the distribution.

**Artifact:** `domain1_grm_parameters.pkl` (item discrimination + threshold parameters per trait).

**Output:** 5 trait scores, θ ∈ [-4.0, +4.0].

**Known consideration:** item parameters are calibrated once against the training population and then frozen. If the respondent population shifts over time (different demographics, different response patterns), a frozen calibration can silently drift out of alignment with the population it's scoring. Periodic re-calibration against fresh response data is worth scheduling rather than treating this as a one-time setup step.

---

## 3. Domain 2 — Rosenberg Self-Esteem Scale

**Construct:** Global self-esteem, per the Rosenberg Self-Esteem Scale (RSES).

**Input schema:** `age`, `gender`, and the 10 RSES items (`Q1-10`, sourced from `rosenberg_self_esteem_clean.csv`).

**Architecture: Deterministic sum-scoring with reverse-coding, plus an age/gender-adjusted percentile lookup.**

- The RSES is scored exactly per its published algorithm: items 2, 5, 6, 8, 9 (the negatively-worded items) are reverse-coded, then all 10 items are summed to produce a score in the standard **0–40 range**, with the RSES's own published cutoffs applied for classification (e.g., low / normal / high self-esteem).
- This scoring rule is kept fully deterministic — no ML component — because the RSES is a validated instrument with a fixed, published scoring algorithm; introducing a statistical model here would mean approximating a known-correct calculation, which offers no benefit and only a risk of drift from the validated instrument's actual meaning.
- **Age/gender-adjusted percentile lookup** is the one genuine addition beyond the raw deterministic score. Previously, `age` and `gender` were collected as part of Domain 2's input schema but never used anywhere downstream — the deterministic sum ignored them entirely. This is now corrected: an empirical percentile table (`domain2_self_esteem_percentiles.json`) allows a respondent's raw 0–40 score to be reported not just against an absolute cutoff, but relative to their age/gender peer group ("your self-esteem is at the 65th percentile for your age and gender group," rather than only "your raw score is 28").

**Artifact:** `domain2_self_esteem_percentiles.json`.

**Output:** RSES raw score (0–40), standard classification, and age/gender-adjusted percentile.

**Note on scope:** an item response theory (GRM) redesign — mirroring Domain 1's approach, giving item-level discrimination weighting and a standard error band — was considered and explicitly **not** adopted for this domain. The deterministic-plus-percentile design was chosen instead to keep Domain 2 simple and directly auditable against the RSES's published scoring rule, at the cost of the additional precision an IRT approach would offer near classification boundaries.

---

## 4. Domain 3 — Mood & Sleep

**Construct:** Depressive symptom severity (via PHQ-9) jointly with sleep patterns, plus a combined behavioral phenotype.

**Input schema:** `DPQ010`–`DPQ090` (PHQ-9 items, NHANES coding) and `SLQ300`, `SLQ310` (sleep timing), sourced from `nhanes_joined_mood_sleep.csv`.

**Architecture: PHQ-9 deterministic scoring (mandatory) + K-Means clustering (mandatory, not optional).**

- **PHQ-9 scoring** follows the validated instrument's own fixed algorithm: sum the 9 items, apply the instrument's own published severity cutoffs. As with Domain 2's RSES scoring, this is deliberately kept deterministic — PHQ-9's clinical validity (its published sensitivity/specificity against real depression diagnoses) is a property of using its exact scoring rule, not something a statistical model should be trained to approximate.
- **K-Means clustering** over the PHQ-9 score and sleep features produces a **Clinical Phenotype** label. This is a **mandatory part of Domain 3's output**, not a fallback-if-available feature — every assessment must produce a phenotype assignment; there is no supported code path where phenotyping is skipped because the model happens to be missing.

**Artifact:** `domain3_mood_sleep.pkl`.

**Output:** PHQ-9 sum score, derived sleep duration, and Clinical Phenotype (cluster assignment).

**Known considerations, to be validated before treating phenotype labels as fixed clinical categories:**
- **Cluster identity stability across retrains.** K-Means cluster indices are not inherently stable — the cluster labeled "2" in one training run is not guaranteed to correspond to the same underlying group after retraining on updated data. Anything downstream that references a phenotype by its numeric/label identity needs this checked (e.g. via bootstrap resampling or a fixed-centroid-seeding convention) before that identity can be treated as durable.
- **Clinical validity of the phenotype label itself.** A cluster is only a "clinical phenotype" once it has been validated against known diagnostic subgroups; unsupervised separability in PHQ-9 + sleep space does not, by itself, establish that a cluster corresponds to a clinically meaningful category. The naming should be treated as provisional pending that validation, and *k* (number of clusters) should be justified rather than fixed arbitrarily.

---

## 5. Domain 4 — Digital & Social

**Construct:** Internet/social-media use patterns, loneliness, and downstream depression risk.

**Input schema:** `age`, `gender`, `IAT1-10` (Internet Addiction Test items), `loneliness1-6`, sourced from `internet_phq_loneliness_clean.csv`.

**Architecture: Tree-based gradient boosting (XGBoost/LightGBM) with SHAP explanation.**

- The Internet Addiction Test sum and Loneliness Scale sum are computed from their respective item batteries; if the tree-based model artifact is unavailable, the pipeline falls back to reporting these raw sums directly rather than failing.
- **Depression Risk** is the genuinely inferential output in this domain — a downstream prediction from IAT sum, loneliness sum, and demographics, rather than a re-derivation of either validated scale. This is where the tree-based model and SHAP explanation actually apply.

**Artifact:** `domain4_digital_social.pkl`.

**Output:** Internet Addiction Sum, Loneliness Score, Depression Risk (model-derived).

**Note on scope:** a structural refinement was discussed — since IAT and Loneliness are themselves validated instruments with defined sum-score definitions, there's a case for treating their sums the same way PHQ-9 and RSES are treated elsewhere in this system (deterministic, ground-truth, no ML approximation), reserving the tree-based/SHAP machinery exclusively for the genuinely inferential Depression Risk output. This domain's architecture is retained as currently implemented; the refinement is noted here as a documented future consideration rather than an adopted change.

---

## 6. Domain 5 — Occupational Burnout

**Construct:** Occupational burnout severity, reported both as a continuous index and an ordinal tier.

**Input schema:** `age`, `gender`, `work_hours_per_week`, `meetings_per_day`, `work_life_balance_score`, `job_satisfaction_score`, `deadline_pressure_score`, `autonomy_score`, `stress_score`, `social_support_score` — 10 raw features from `tech_burnout_2026_clean.csv` — plus 3 engineered features computed at training/inference time (see below).

**Architecture: two complementary XGBoost model heads, monotonically constrained.**

This domain was redesigned from a single XGBoost regressor whose continuous output was binned post-hoc into tiers. The redesign addresses two structural issues with that approach: (a) a single regression optimizing squared error has no notion of ordinal tier boundaries, so tier assignment was a side effect rather than a directly optimized target; and (b) a bare point estimate gives no sense of how confidently a given score should be trusted, which matters most exactly at tier boundaries.

**Feature engineering** (computed from the 10 raw features before either model head sees the data):
- `stress_x_support = stress_score × social_support_score` — captures the buffering interaction where high social support offsets the impact of high stress, rather than the two contributing additively.
- `hours_over_50 = max(0, work_hours_per_week - 50)` — captures a known threshold effect (burnout risk accelerates past a heavy-hours threshold rather than rising linearly with hours).
- `meeting_load_ratio = meetings_per_day / work_hours_per_week` — captures meetings crowding out focused work time as a distinct driver from raw hours worked.

**Model 1 — Ordinal tier classification (3 cumulative binary classifiers).** Rather than a single multi-class or regression model, tier probability is modeled as three separate binary classifiers, each predicting P(tier ≥ *k*) for *k* ∈ {Moderate, High, Severe}. Differencing these cumulative probabilities yields a proper probability distribution over the four tiers (Low/Moderate/High/Severe) that respects their natural ordering, unlike a nominal (one-vs-rest) classifier, which would treat "Severe" and "Low" as no more or less related than any other pair of tiers.

**Model 2 — Quantile regression (3 regressors at τ = 0.1, 0.5, 0.9).** The continuous Burnout Index is reported as a median point estimate accompanied by a 10th–90th percentile interval, rather than a bare number — e.g. "Burnout Index: 64 (55–73 range)."

**Monotonic constraints** are applied to both heads on the 8 raw work-related features plus the 3 engineered features, with signed direction fixed a priori from domain knowledge (e.g. `work_hours_per_week` up → burnout up; `social_support_score` up → burnout down). `age` and `gender` (one-hot encoded) carry **no constraint**, since no defensible a priori direction exists for either — this is a deliberate, documented choice, not an oversight.

**Explainability:** SHAP TreeExplainer is applied separately to each of the 6 models (3 ordinal + 3 quantile), producing per-threshold and per-quantile summary/importance plots, a combined cross-threshold importance panel, a stress × social-support dependence plot (visualizing the buffering interaction directly), and single-person waterfall plots for individual-level reporting.

**Artifacts:** `domain5_ordinal_ge_{Moderate,High,Severe}.json` (3 boosters), `domain5_quantile_q{10,50,90}.json` (3 boosters), `domain5_burnout_metadata.json` (feature order, monotonic constraint map, tier/quantile definitions).

**Output:** P(tier ≥ Moderate/High/Severe) → differenced into a 4-way tier probability distribution, and Burnout Index as a median estimate with a 10th–90th percentile interval.

**Data provenance caveat (important for report interpretation):** `tech_burnout_2026_clean.csv` is a **synthetic Kaggle dataset**, not real occupational survey data. `stress_score` correlates with the `burnout_score` label at **r = 0.854 (R² = 0.73)** — far higher than the 0.3–0.6 range typical of two related-but-distinct constructs in genuine self-report data — indicating the label was very likely constructed formulaically with `stress_score` as a dominant input during dataset generation, rather than the two being independently measured and organically correlated. SHAP's strong attribution of importance to `stress_score` (5–6x every other feature) is a correct reflection of this training data, but should be reported as *"how this synthetic dataset's label was constructed"* rather than a validated *"stress is the primary driver of burnout"* clinical or occupational-psychology finding. Any real-world claims from this domain should be treated as provisional pending validation on genuine survey data.

**Training pipeline safeguards:** hard-fails (no silent fallback) on missing dataset files, insufficient row counts, missing required columns, or non-numeric/out-of-range values; a `--allow-synthetic` flag exists strictly for development/testing and is never the default production path.

---

## 7. Domain 6 — Severe Clinical Screening

**Construct:** Detection of symptom-cluster patterns associated with specific psychiatric presentations, plus flagging of atypical/complex symptom profiles warranting closer review.

**Input schema:** 9 binary symptom features — `unwanted_thoughts`, `repetitive_behaviors`, `overthinking`, `mind_going_blank`, `avoidance_social_activity`, `panic`, `hypervigilance`, `sleep_disturbances`, `low_energy` — sourced from `ocd_symptoms_clean.csv` (8,304 rows, confirmed real/non-synthetic data).

This domain underwent the most substantial redesign in the system, driven by direct empirical investigation of the training data rather than a pre-decided algorithm change. The findings that drove the redesign:

- The dataset's raw `Disease` column contains **22 distinct diagnostic labels** (Major Depressive Disorder, OCD, ADHD, Dissociative Amnesia, etc.), each with roughly equal representation (~390 rows) — not a severity scale for any single condition. There is **no severity label anywhere in this data**, which ruled out the originally documented "Condition Code (0–3)" severity-screening design entirely; no choice of algorithm could produce that design from this data, because the required ground truth doesn't exist in it.
- Of the 22 diagnostic categories, only **8 show any detectable symptom signature** — one or two of the 9 features elevated to ~63–66% prevalence, against a ~0.3–2.5% baseline for the other 14 categories plus "No illness," which are statistically indistinguishable from each other and from a healthy baseline.
- Within those 8 detectable categories, **Anxiety and Bipolar Disorder were confirmed non-separable**: both elevate only `sleep_disturbances`, at nearly identical rates (~65–66%), with no other feature differentiating them. This is a genuine identifiability limit in the data, not a modeling shortfall — no classifier can learn to distinguish two classes with identical feature distributions without simply memorizing training-split noise.

**Architecture: two complementary models, reflecting what the data can actually support.**

**Model 1 — Symptom Cluster Screener (multinomial logistic regression).** The raw 22-way `Disease` label is collapsed, using an explicit and total mapping (`CLUSTER_MAP`), into **8 clinically-supportable clusters**:

| Cluster | Signature feature(s) | Source diseases |
|---|---|---|
| OCD_pattern | `unwanted_thoughts` + `repetitive_behaviors` | OCD |
| Depression_pattern | `low_energy` + `sleep_disturbances` | Depression |
| GAD_pattern | `overthinking` + `sleep_disturbances` | GAD |
| PTSD_pattern | `hypervigilance` + `sleep_disturbances` | PTSD |
| Panic_Dissociative_pattern | `panic` | Dissociative Identity Disorder |
| Social_Anxiety_pattern | `avoidance_social_activity` | Social Anxiety Disorder |
| Sleep_Disruption_Nonspecific | `sleep_disturbances` only | Anxiety + Bipolar Disorder (merged; confirmed non-separable) |
| No_Detectable_Signal | — | remaining 14 diseases + "No illness" (statistically flat/noise-level) |

A single multinomial (softmax) logistic regression model is fit against this 8-class target, using the real `Disease` labels as ground truth (not a fabricated sum-based proxy). Per-class coefficients are stored directly and transparently in metadata rather than embedded only in the pickled model object, consistent with this domain's transparency-first design.

**Model 2 — Isolation Forest, reframed as an atypical/multi-cluster presentation detector.** Rather than flagging rare exact symptom combinations (which, given only 9 binary features / 512 possible combinations, an exact frequency table would answer more transparently), Isolation Forest is fit across the full population to flag presentations that combine symptoms from more than one cluster, or show an unusually high symptom count relative to the population (66% of respondents show 0 symptoms; the small tail showing 2+ symptoms is where cross-cluster/comorbid-looking presentations live). This is validated post-hoc, not merely asserted: flagged rows show a mean symptom count of **2.4**, versus **0.38** for non-flagged rows — confirming the detector tracks the intended construct.

**Artifacts:** `domain6_clinical.pkl` (multinomial classifier + Isolation Forest), `domain6_clinical_metadata.json` (cluster map, per-class coefficients, Isolation Forest validation statistics, design rationale).

**Output:** predicted symptom cluster (1 of 8, with per-class probabilities) and an atypical-presentation binary flag.

**Training pipeline safeguards:** hard-fails on missing/insufficient data, missing required columns, and — specific to this domain — **any `Disease` value not present in the explicit cluster map**, since an unmapped category is a genuine schema-drift signal requiring a human clustering decision, not a value to silently absorb into a default bucket. An explicit `--allow-unmapped-as-nodetectable` override exists but is off by default.

---

## 8. Cross-Domain Summary Table

| Domain | Ground truth type | ML necessity | Transparency mechanism |
|---|---|---|---|
| 1 — Personality | Self-report items, no external label | Statistical (IRT) — models latent trait, no label to predict | Parametric model, closed-form theta estimation |
| 2 — Self-Esteem | Validated instrument, fixed scoring rule | None (by design) | Fully deterministic, human-auditable |
| 3 — Mood & Sleep | Validated instrument (PHQ-9) + unsupervised grouping | Deterministic core + mandatory unsupervised layer | PHQ-9 fully deterministic; phenotype requires stability validation |
| 4 — Digital & Social | Validated instrument sums + inferred downstream risk | ML only for the inferential Depression Risk output | SHAP on tree-based Depression Risk model |
| 5 — Burnout | Synthetic label (Kaggle) | ML — no validated instrument exists for this synthetic label | SHAP TreeExplainer per model head, monotonic constraints |
| 6 — Clinical Screening | Real diagnostic labels, collapsed to data-supported granularity | ML — genuine multi-class inference problem | Transparent multinomial coefficients + validated anomaly detector |

---

## 9. Known Limitations Requiring Follow-Up

- **Domain 1:** IRT item-parameter calibration is currently static; needs a re-calibration cadence as the respondent population evolves.
- **Domain 3:** K-Means cluster identity stability across retrains, and clinical validation of the "phenotype" label against known diagnostic subgroups, are both outstanding.
- **Domain 4:** IAT/Loneliness sum-vs-model architecture split is a documented future consideration, not yet adopted.
- **Domain 5:** training data is synthetic; any real-world interpretive claims require validation against genuine occupational survey data before being presented as clinical or occupational-psychology findings.
- **Domain 6:** only 8 of 22 diagnostic categories are separable given the current 9-feature symptom checklist; expanding the feature set (additional symptom items) would be required to meaningfully improve coverage of the remaining 14 categories, rather than any change of model architecture.
