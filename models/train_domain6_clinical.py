#!/usr/bin/env python3
"""
MINDSIGHT Domain 6 Calibration Engine (v5.0 - Symptom Cluster Screener + Atypical Presentation Detector)

This is a full reframe, not a patch. Investigation of the real
ocd_symptoms_clean.csv (8,304 rows, 9 binary symptom features, 22 raw
`Disease` labels) showed:
  - 66.1% of rows have ZERO symptoms present across all 9 features.
  - Only 8 of the 22 raw disease labels have any detectable symptom
    signature (one or two features elevated to ~63-66% prevalence); the
    other 14 are statistically indistinguishable from "No illness" (all
    features at ~0.3%-2.5%, i.e. noise level).
  - Anxiety and Bipolar Disorder are PROVABLY non-separable on this feature
    set -- both elevate only sleep_disturbances, at nearly identical rates,
    with no other differentiating feature. Training a classifier to tell
    them apart would just memorize noise.

Given that, Domain 6 is redesigned as two complementary models rather than
one severity classifier:

  MODEL 1 -- Symptom Cluster Screener (supervised, multinomial logistic
  regression). Target is NOT the raw 22-way Disease label -- it's an
  8-class collapse onto clusters this data can actually support, built
  from a fully explicit Disease -> cluster mapping (see CLUSTER_MAP).
  Anxiety + Bipolar Disorder are merged into one "sleep-disruption,
  nonspecific" class per the confirmed non-separability above. The 14
  no-signal diseases + "No illness" collapse into "No_Detectable_Signal".

  MODEL 2 -- Isolation Forest, reframed. Rather than "is this combination
  rare" (which, with only 9 binary features / 512 possible combinations,
  an exact frequency table answers more transparently), this is fit to
  flag ATYPICAL / MULTI-CLUSTER presentations: profiles that combine
  symptoms from more than one cluster, or show an unusually high symptom
  count for the population (66% of respondents show 0 symptoms, ~23% show
  1 -- the 2+ tail is where cross-cluster/comorbid-looking presentations
  live). This is a genuinely different, complementary signal to Model 1's
  "most likely single cluster" output, and is validated post-hoc against
  symptom count as a sanity check (see evaluation_metrics.json).

Same hard-fail philosophy as the rest of this rewrite: no silent synthetic
fallback, no silent zero-fill, no silent Disease->cluster mapping for
unmapped/unexpected values.
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

BINARY_FEATURES = [
    "unwanted_thoughts", "repetitive_behaviors", "overthinking",
    "mind_going_blank", "avoidance_social_activity", "panic",
    "hypervigilance", "sleep_disturbances", "low_energy",
]

# Explicit, total mapping from every raw Disease value this dataset is known
# to contain, to one of 8 clinically-supportable cluster labels. Any Disease
# value NOT in this map is a schema-drift signal and hard-fails by default
# (see --allow-unmapped-as-nodetectable to override, not recommended).
CLUSTER_MAP = {
    "Obsessive-Compulsive Disorder (OCD)": "OCD_pattern",
    "Depression": "Depression_pattern",
    "Generalized Anxiety Disorder (GAD)": "GAD_pattern",
    "Post-Traumatic Stress Disorder (PTSD)": "PTSD_pattern",
    "Dissociative Identity Disorder": "Panic_Dissociative_pattern",
    "Social Anxiety Disorder": "Social_Anxiety_pattern",
    # Confirmed non-separable on this feature set -- both elevate ONLY
    # sleep_disturbances at ~65-66% with no other differentiating feature.
    "Anxiety": "Sleep_Disruption_Nonspecific",
    "Bipolar Disorder": "Sleep_Disruption_Nonspecific",
    # Statistically indistinguishable from "No illness" on this feature set
    # (all features ~0.3%-2.5%, noise level) -- collapsed to one class.
    "No illness": "No_Detectable_Signal",
    "ADHD (Attention Deficit Hyperactivity Disorder)": "No_Detectable_Signal",
    "Adjustment Disorder": "No_Detectable_Signal",
    "Autism Spectrum Disorder": "No_Detectable_Signal",
    "Borderline Personality Disorder": "No_Detectable_Signal",
    "Dissociative Amnesia": "No_Detectable_Signal",
    "Eating Disorder": "No_Detectable_Signal",
    "Insomnia": "No_Detectable_Signal",
    "Major Depressive Disorder": "No_Detectable_Signal",
    "Panic Disorder": "No_Detectable_Signal",
    "Persistent Depressive Disorder": "No_Detectable_Signal",
    "Schizophrenia": "No_Detectable_Signal",
    "Separation Anxiety Disorder": "No_Detectable_Signal",
    "Substance Use Disorder": "No_Detectable_Signal",
}

CLASS_ORDER = [
    "No_Detectable_Signal",
    "OCD_pattern",
    "Depression_pattern",
    "GAD_pattern",
    "PTSD_pattern",
    "Panic_Dissociative_pattern",
    "Social_Anxiety_pattern",
    "Sleep_Disruption_Nonspecific",
]

ANOMALY_CONTAMINATION = 0.05  # explicit population-share target, documented as such


def fail(msg):
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(1)


def project_root():
    """Walks upward from this script's location until it finds a directory
    containing sibling 'models' and 'datasets' folders, rather than assuming
    a fixed nesting depth."""
    current = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isdir(os.path.join(current, "models")) and os.path.isdir(os.path.join(current, "datasets")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            fail(
                "Could not locate the MINDSIGHT project root by walking up from "
                f"{os.path.dirname(os.path.abspath(__file__))}. Expected a directory "
                "containing sibling 'models' and 'datasets' folders."
            )
        current = parent


# --------------------------------------------------------------------------
# Synthetic cohort -- DEV/TEST ONLY, gated behind --allow-synthetic
# --------------------------------------------------------------------------

def generate_synthetic_cohort(n_samples=5000):
    print("  | [DEV] Generating synthetic reference cohort (NOT for production).")
    np.random.seed(42)
    diseases = list(CLUSTER_MAP.keys())
    records = []
    for _ in range(n_samples):
        disease = np.random.choice(diseases)
        cluster = CLUSTER_MAP[disease]
        base_probs = {f: 0.02 for f in BINARY_FEATURES}
        if cluster == "OCD_pattern":
            base_probs.update({"unwanted_thoughts": 0.63, "repetitive_behaviors": 0.63})
        elif cluster == "Depression_pattern":
            base_probs.update({"low_energy": 0.65, "sleep_disturbances": 0.65})
        elif cluster == "GAD_pattern":
            base_probs.update({"overthinking": 0.64, "sleep_disturbances": 0.65})
        elif cluster == "PTSD_pattern":
            base_probs.update({"hypervigilance": 0.65, "sleep_disturbances": 0.65})
        elif cluster == "Panic_Dissociative_pattern":
            base_probs.update({"panic": 0.63})
        elif cluster == "Social_Anxiety_pattern":
            base_probs.update({"avoidance_social_activity": 0.66})
        elif cluster == "Sleep_Disruption_Nonspecific":
            base_probs.update({"sleep_disturbances": 0.65})
        symptoms = {f: np.random.binomial(1, p) for f, p in base_probs.items()}
        row = {**symptoms, "Disease": disease}
        records.append(row)
    df = pd.DataFrame(records)
    return df[["Disease"] + BINARY_FEATURES]


# --------------------------------------------------------------------------
# Data loading -- hard fail, no silent fallback unless explicitly requested
# --------------------------------------------------------------------------

def load_data(candidate_paths, allow_synthetic):
    for path in candidate_paths:
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"  | [WARNING] Could not parse '{path}': {e}")
            continue
        if len(df) < 50:
            print(f"  | [WARNING] '{path}' has only {len(df)} rows (<50). Skipping.")
            continue
        print(f"  | Matched training dataset at: '{path}' ({len(df)} records)")
        return df, path, False

    if allow_synthetic:
        return generate_synthetic_cohort(), "synthetic_reference_generation", True

    fail(
        f"No usable real dataset found among candidates: {candidate_paths} "
        f"(need >=50 rows). Refusing to fabricate training data. "
        f"Pass --allow-synthetic explicitly for a dev/test run only."
    )


def resolve_columns_and_target(df, allow_unmapped_as_nodetectable):
    df = df.copy()
    df.columns = df.columns.str.strip()

    missing = [c for c in BINARY_FEATURES + ["Disease"] if c not in df.columns]
    if missing:
        fail(f"Required columns missing: {missing}. Refusing to fabricate them.")

    for col in BINARY_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            fail(f"Column '{col}' contains unparseable values.")
        bad_vals = set(df[col].unique()) - {0, 1}
        if bad_vals:
            fail(f"Column '{col}' contains values outside {{0,1}}: {bad_vals}.")
        df[col] = df[col].astype(int)

    unmapped = set(df["Disease"].unique()) - set(CLUSTER_MAP.keys())
    if unmapped:
        if allow_unmapped_as_nodetectable:
            print(f"  | [WARNING] Unmapped Disease values folded into "
                  f"'No_Detectable_Signal' (--allow-unmapped-as-nodetectable set): {unmapped}")
        else:
            fail(
                f"Disease column contains values with no entry in CLUSTER_MAP: {unmapped}. "
                f"This is a schema-drift signal -- either the dataset has new disease "
                f"categories that need their own cluster-mapping decision, or this is "
                f"unexpected data. Add them to CLUSTER_MAP explicitly, or re-run with "
                f"--allow-unmapped-as-nodetectable to fold them into 'No_Detectable_Signal' "
                f"(only appropriate if you've confirmed they truly show no signal)."
            )

    df["cluster_label"] = df["Disease"].map(CLUSTER_MAP).fillna("No_Detectable_Signal")
    df["cluster_idx"] = df["cluster_label"].map({c: i for i, c in enumerate(CLASS_ORDER)})
    return df


# --------------------------------------------------------------------------
# Model 1: Symptom Cluster Screener (multinomial logistic regression)
# --------------------------------------------------------------------------

def train_cluster_classifier(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(
        solver="lbfgs",
        class_weight="balanced", max_iter=2000, random_state=42,
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    cm = confusion_matrix(y_test, preds, labels=list(range(len(CLASS_ORDER)))).tolist()
    return clf, {"accuracy": round(float(acc), 4), "macro_f1": round(float(macro_f1), 4), "confusion_matrix": cm}


# --------------------------------------------------------------------------
# Model 2: Isolation Forest -- atypical / multi-cluster presentation flag
# --------------------------------------------------------------------------

def train_isolation_forest(X, contamination):
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    iso.fit(X)
    preds = iso.predict(X)  # -1 = flagged, 1 = normal
    flagged = (preds == -1)
    realized_flag_rate = float(flagged.mean())

    # Sanity check: does the flag actually track "unusual symptom count",
    # the construct we intend it to capture? Compare mean symptom count for
    # flagged vs. not-flagged rows.
    symptom_count = X.sum(axis=1)
    mean_count_flagged = float(symptom_count[flagged].mean()) if flagged.any() else float("nan")
    mean_count_not_flagged = float(symptom_count[~flagged].mean()) if (~flagged).any() else float("nan")

    return iso, {
        "contamination_target": contamination,
        "realized_flag_rate": round(realized_flag_rate, 4),
        "mean_symptom_count_flagged": round(mean_count_flagged, 3),
        "mean_symptom_count_not_flagged": round(mean_count_not_flagged, 3),
        "sanity_check_note": (
            "If mean_symptom_count_flagged is meaningfully higher than "
            "mean_symptom_count_not_flagged, the anomaly flag is tracking the "
            "intended construct (unusual/multi-cluster symptom load) rather "
            "than something incidental."
        ),
    }


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def train_clinical_screening_pipeline(allow_synthetic=False, allow_unmapped_as_nodetectable=False):
    print("[RUNNING] Domain 6 v5.0: Symptom Cluster Screener + Atypical Presentation Detector")

    project_root_dir = project_root()
    output_dir = os.path.join(project_root_dir, "models", "saved_states")
    os.makedirs(output_dir, exist_ok=True)
    output_model_path = os.path.join(output_dir, "domain6_clinical.pkl")
    output_meta_path = os.path.join(output_dir, "domain6_clinical_metadata.json")

    candidate_paths = [
        os.path.join(project_root_dir, "datasets", "ocd_symptoms_clean.csv"),
    ]

    df_raw, selected_path, is_synthetic = load_data(candidate_paths, allow_synthetic)
    df = resolve_columns_and_target(df_raw, allow_unmapped_as_nodetectable)

    cluster_counts = df["cluster_label"].value_counts().reindex(CLASS_ORDER, fill_value=0)
    print(f"  | Cluster distribution:\n{cluster_counts.to_string()}")

    X = df[BINARY_FEATURES].copy()
    y = df["cluster_idx"].copy()

    if y.nunique() < 2:
        fail(f"Only {y.nunique()} distinct cluster(s) present after mapping -- cannot train.")

    # --- Split for evaluation ---
    min_class_count = y.value_counts().min()
    stratify = y if min_class_count >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    print(f"  | Train/Test split: {len(X_train)} train, {len(X_test)} test")

    print("  | Training Model 1: Symptom Cluster Screener (multinomial logistic regression, eval split)...")
    _, clf_eval_metrics = train_cluster_classifier(X_train, X_test, y_train, y_test)
    print(f"      Test Accuracy: {clf_eval_metrics['accuracy']}, Macro-F1: {clf_eval_metrics['macro_f1']}")

    print("  | Re-fitting Model 1 on full dataset for production...")
    full_clf, _ = train_cluster_classifier(X, X, y, y)  # refit; eval metrics ignored here, reported from split above

    print("  | Training Model 2: Isolation Forest (atypical/multi-cluster presentation flag, full data)...")
    iso, iso_metrics = train_isolation_forest(X, ANOMALY_CONTAMINATION)
    print(f"      Realized flag rate: {iso_metrics['realized_flag_rate']}, "
          f"mean symptom count flagged={iso_metrics['mean_symptom_count_flagged']} "
          f"vs not-flagged={iso_metrics['mean_symptom_count_not_flagged']}")

    # --- Save model payload ---
    model_payload = {"cluster_classifier": full_clf, "isolation_forest": iso}
    with open(output_model_path, "wb") as f:
        pickle.dump(model_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  | Model payload saved -> {output_model_path}")

    # --- Metadata contract ---
    coefficients = {
        CLASS_ORDER[i]: {"coef": full_clf.coef_[i].tolist(), "intercept": float(full_clf.intercept_[i])}
        for i in range(len(CLASS_ORDER))
    }
    metadata = {
        "schema_version": "5.0",
        "domain": "domain_6_severe_clinical",
        "features": BINARY_FEATURES,
        "class_order": CLASS_ORDER,
        "cluster_map": CLUSTER_MAP,
        "design_notes": (
            "Raw 22-way Disease label collapsed to 8 classes based on empirical "
            "signal analysis of ocd_symptoms_clean.csv: 14 diseases + 'No illness' "
            "were statistically indistinguishable (noise-level prevalence on all "
            "9 features) and collapsed to 'No_Detectable_Signal'. Anxiety and "
            "Bipolar Disorder were confirmed non-separable (both elevate only "
            "sleep_disturbances at ~65-66%, no other differentiating feature) and "
            "merged into 'Sleep_Disruption_Nonspecific'. Model 1 predicts the most "
            "likely cluster; Model 2 (Isolation Forest) is a separate, "
            "complementary signal flagging atypical/multi-cluster symptom "
            "presentations, not a severity or rarity score."
        ),
        "source_file_used": selected_path,
        "is_synthetic_source": is_synthetic,
        "coefficients": coefficients,
        "isolation_forest": iso_metrics,
        "payload_contract": {
            "cluster_classifier_key": "cluster_classifier",
            "isolation_forest_key": "isolation_forest",
        },
    }
    with open(output_meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  | Metadata exported -> {output_meta_path}")

    # --- Evaluation metrics ---
    eval_metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    domain_metrics = {
        "cluster_classifier": {**clf_eval_metrics, "class_labels": CLASS_ORDER, "test_set_size": int(len(X_test))},
        "isolation_forest": iso_metrics,
    }
    if os.path.exists(eval_metrics_path):
        with open(eval_metrics_path, "r") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}
    all_metrics["domain_6_severe_clinical"] = domain_metrics
    with open(eval_metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[SUCCESS] Domain 6 pipeline complete. Eval metrics -> {eval_metrics_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-synthetic", action="store_true",
                         help="Allow falling back to a synthetic cohort if real data is "
                              "missing/insufficient. DEV/TEST ONLY.")
    parser.add_argument("--allow-unmapped-as-nodetectable", action="store_true",
                         help="Fold any Disease value not in CLUSTER_MAP into "
                              "'No_Detectable_Signal' instead of hard-failing. Only use "
                              "after confirming those values genuinely show no signal.")
    args = parser.parse_args()
    train_clinical_screening_pipeline(
        allow_synthetic=args.allow_synthetic,
        allow_unmapped_as_nodetectable=args.allow_unmapped_as_nodetectable,
    )