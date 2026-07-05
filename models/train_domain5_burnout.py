#!/usr/bin/env python3
"""
MINDSIGHT Domain 5 Calibration Engine (v3.0 - Ordinal + Quantile Redesign)

Replaces the single XGBRegressor-then-bin approach with:
  - An ORDINAL head: 3 cumulative binary XGBClassifiers
    (P[>=Moderate], P[>=High], P[>=Severe]) combined into tier probabilities,
    so tier assignment is learned directly instead of derived post-hoc from
    a continuous score.
  - A QUANTILE head: 3 XGBRegressors (tau = 0.1, 0.5, 0.9) giving a Burnout
    Index point estimate with an uncertainty interval.
  - Monotonic constraints on the 8 work-related features (signed direction
    known a priori) and NO constraint on age/gender (direction not known
    a priori -- documented explicitly, not an oversight).
  - Engineered interaction/threshold features: stress_x_support,
    hours_over_50, meeting_load_ratio.
  - Hard failure on missing/insufficient data or missing columns -- no
    silent synthetic-data fallback in the production path.
  - Quantile-based (not max-based) tier thresholds.
  - SHAP-ready metadata: feature order, monotonic constraint map, and both
    model bundles saved so TreeExplainer can load them directly at
    inference time.

Run:
    python train_domain5_burnout.py
Optional dev-only synthetic cohort (never used unless explicitly requested):
    python train_domain5_burnout.py --allow-synthetic
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, log_loss
)

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

RAW_WORK_FEATURES = [
    "work_hours_per_week", "meetings_per_day",
    "work_life_balance_score", "job_satisfaction_score",
    "deadline_pressure_score", "autonomy_score",
    "stress_score", "social_support_score",
]
DEMOGRAPHIC_FEATURES = ["age", "gender"]
TIER_ORDER = ["Low", "Moderate", "High", "Severe"]
QUANTILES = [0.1, 0.5, 0.9]

# Signed monotonic direction for work-related features.
# +1 = burnout increases as feature increases, -1 = decreases, 0 = no constraint.
MONOTONE_DIRECTION = {
    "work_hours_per_week": 1,
    "meetings_per_day": 1,
    "work_life_balance_score": -1,
    "job_satisfaction_score": -1,
    "deadline_pressure_score": 1,
    "autonomy_score": -1,
    "stress_score": 1,
    "social_support_score": -1,
    # Engineered features inherit a directional prior too.
    "stress_x_support": 1,     # interaction term; still expected net-positive w.r.t. stress side
    "hours_over_50": 1,
    "meeting_load_ratio": 1,
    # No a priori direction for demographics -- explicit, not an oversight.
    "age": 0,
}
# gender dummy column names are resolved dynamically after one-hot encoding
# and are always assigned monotone constraint 0 (see build_monotone_tuple()).


def fail(msg):
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(1)


def project_root():
    """Walks upward from this script's location until it finds a directory
    containing sibling 'models' and 'datasets' folders, rather than assuming
    a fixed nesting depth. Prevents breakage if this script is later moved
    to a different folder depth (this is exactly what broke the plotting
    scripts' original single-level-up assumption)."""
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
# Data loading -- hard fail, no synthetic fallback unless explicitly requested
# --------------------------------------------------------------------------

def load_data(data_path, allow_synthetic):
    if not os.path.exists(data_path):
        if allow_synthetic:
            print("  | [DEV] --allow-synthetic set: generating synthetic cohort (NOT for production).")
            return _synthetic_cohort()
        fail(f"Dataset not found at {data_path}. Refusing to fabricate training data. "
             f"Pass --allow-synthetic explicitly if you intend to run a dev/test pipeline.")

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        fail(f"Failed to read dataset at {data_path}: {e}")

    if len(df) < 50:
        if allow_synthetic:
            print("  | [DEV] Dataset too small, --allow-synthetic set: using synthetic cohort.")
            return _synthetic_cohort()
        fail(f"Dataset at {data_path} has only {len(df)} rows (<50). Refusing to train "
             f"a production model on insufficient data. Pass --allow-synthetic to override "
             f"for dev/testing only.")

    required = RAW_WORK_FEATURES + DEMOGRAPHIC_FEATURES + ["burnout_score", "burnout_level"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        fail(f"Dataset is missing required columns: {missing}. Refusing to fabricate them. "
             f"Check upstream schema/export against schema_config.json.")

    print(f"  | Successfully ingested {len(df)} rows from {data_path}")
    return df


def _synthetic_cohort():
    """DEV/TEST ONLY. Never reached in production unless --allow-synthetic is passed."""
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({f: np.random.uniform(1, 10, n) for f in RAW_WORK_FEATURES})
    df["age"] = np.random.randint(22, 65, n)
    df["gender"] = np.random.choice(["M", "F", "Other"], n)
    df["burnout_score"] = df["stress_score"] * 0.6 + np.random.normal(0, 1, n)
    df["burnout_level"] = pd.cut(df["burnout_score"], bins=4, labels=TIER_ORDER)
    return df


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------

def engineer_features(df):
    df = df.copy()
    for col in RAW_WORK_FEATURES + DEMOGRAPHIC_FEATURES:
        if col == "gender":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    df["stress_x_support"] = df["stress_score"] * df["social_support_score"]
    df["hours_over_50"] = (df["work_hours_per_week"] - 50).clip(lower=0)
    df["meeting_load_ratio"] = df["meetings_per_day"] / df["work_hours_per_week"].replace(0, np.nan)
    df["meeting_load_ratio"] = df["meeting_load_ratio"].fillna(df["meeting_load_ratio"].median())

    df["gender"] = df["gender"].astype(str).str.strip().str.lower()
    gender_dummies = pd.get_dummies(df["gender"], prefix="gender", dtype=int)

    engineered_features = ["stress_x_support", "hours_over_50", "meeting_load_ratio"]
    base_numeric = RAW_WORK_FEATURES + ["age"] + engineered_features

    X = pd.concat([df[base_numeric], gender_dummies], axis=1)
    gender_cols = list(gender_dummies.columns)
    return X, gender_cols, engineered_features


def build_monotone_tuple(feature_order, gender_cols):
    constraints = []
    for col in feature_order:
        if col in gender_cols:
            constraints.append(0)
        else:
            constraints.append(MONOTONE_DIRECTION.get(col, 0))
    return "(" + ",".join(str(c) for c in constraints) + ")"


# --------------------------------------------------------------------------
# Ordinal head: cumulative binary classifiers
# --------------------------------------------------------------------------

def make_cumulative_targets(levels):
    """levels: categorical series with TIER_ORDER order.
    Returns dict of {'>=Moderate': 0/1 array, '>=High': ..., '>=Severe': ...}"""
    idx = levels.apply(lambda v: TIER_ORDER.index(v))
    targets = {}
    for i, tier in enumerate(TIER_ORDER[1:], start=1):
        targets[f">={tier}"] = (idx >= i).astype(int)
    return targets


def train_ordinal_head(X_train, X_test, y_train_levels, y_test_levels, monotone_str):
    cum_train = make_cumulative_targets(y_train_levels)
    cum_test = make_cumulative_targets(y_test_levels)

    models = {}
    eval_report = {}
    for label, y_tr in cum_train.items():
        y_te = cum_test[label]
        clf = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            monotone_constraints=monotone_str,
            eval_metric="logloss",
            scale_pos_weight=_pos_weight(y_tr),
        )
        clf.fit(X_train, y_tr, eval_set=[(X_test, y_te)], verbose=False)
        proba = clf.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        eval_report[label] = {
            "accuracy": round(float(accuracy_score(y_te, preds)), 4),
            "log_loss": round(float(log_loss(y_te, proba, labels=[0, 1])), 4),
            "positive_rate_train": round(float(y_tr.mean()), 4),
        }
        models[label] = clf
        print(f"      [{label}] acc={eval_report[label]['accuracy']}, "
              f"logloss={eval_report[label]['log_loss']}, "
              f"train_pos_rate={eval_report[label]['positive_rate_train']}")
    return models, eval_report


def _pos_weight(y):
    pos = y.sum()
    neg = len(y) - pos
    return float(neg / pos) if pos > 0 else 1.0


def cumulative_probs_to_tier_probs(p_ge_moderate, p_ge_high, p_ge_severe):
    """Convert monotonically-should-decrease cumulative P[>=tier] into
    per-tier probabilities. Clips to keep them non-negative if the
    cumulative probabilities aren't perfectly monotonic."""
    p_low = 1 - p_ge_moderate
    p_moderate = p_ge_moderate - p_ge_high
    p_high = p_ge_high - p_ge_severe
    p_severe = p_ge_severe
    stacked = np.clip(np.stack([p_low, p_moderate, p_high, p_severe], axis=1), 0, None)
    stacked = stacked / stacked.sum(axis=1, keepdims=True)
    return stacked  # columns align with TIER_ORDER


# --------------------------------------------------------------------------
# Quantile head
# --------------------------------------------------------------------------

def train_quantile_head(X_train, X_test, y_train, y_test, monotone_str):
    models = {}
    eval_report = {}
    for q in QUANTILES:
        reg = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            monotone_constraints=monotone_str,
            objective="reg:quantileerror",
            quantile_alpha=q,
            eval_metric="mae",
        )
        reg.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        preds = reg.predict(X_test)
        pinball = _pinball_loss(y_test.values, preds, q)
        eval_report[f"q{q}"] = {"pinball_loss": round(float(pinball), 4)}
        models[q] = reg
        print(f"      [quantile {q}] pinball_loss={eval_report[f'q{q}']['pinball_loss']}")
    return models, eval_report


def _pinball_loss(y_true, y_pred, q):
    diff = y_true - y_pred
    return np.mean(np.maximum(q * diff, (q - 1) * diff))


# --------------------------------------------------------------------------
# Thresholds -- quantile-based, not max-based
# --------------------------------------------------------------------------

def compute_thresholds(y):
    q = y.quantile([0.25, 0.50, 0.75])
    return {
        "low_to_moderate": float(q.loc[0.25]),
        "moderate_to_high": float(q.loc[0.50]),
        "high_to_severe": float(q.loc[0.75]),
    }


# --------------------------------------------------------------------------
# Fairness check (informational only -- does not block training)
# --------------------------------------------------------------------------

def gender_shap_flag(models_ordinal, X, gender_cols):
    try:
        import shap
    except ImportError:
        print("  | [INFO] shap not installed; skipping gender-attribution fairness check.")
        return None

    flags = {}
    for label, clf in models_ordinal.items():
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X)
        for gcol in gender_cols:
            if gcol in X.columns:
                col_idx = list(X.columns).index(gcol)
                mean_abs_shap = float(np.mean(np.abs(shap_values[:, col_idx])))
                flags[f"{label}__{gcol}"] = round(mean_abs_shap, 5)
    return flags


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def train_burnout_model(allow_synthetic=False):
    print(" Commencing Domain 5 XGBoost Ordinal + Quantile Training Pipeline...")

    project_root_dir = project_root()
    data_path = os.path.join(project_root_dir, "datasets", "tech_burnout_2026_clean.csv")
    output_dir = os.path.join(project_root_dir, "models", "saved_states")
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(data_path, allow_synthetic)

    df["burnout_level"] = df["burnout_level"].astype(str).str.strip()
    bad_levels = set(df["burnout_level"].unique()) - set(TIER_ORDER)
    if bad_levels:
        fail(f"burnout_level contains unexpected values not in {TIER_ORDER}: {bad_levels}")

    X, gender_cols, engineered_features = engineer_features(df)
    feature_order = list(X.columns)
    monotone_str = build_monotone_tuple(feature_order, gender_cols)
    print(f"  | Feature order ({len(feature_order)}): {feature_order}")
    print(f"  | Monotone constraints: {monotone_str}")

    y_score = pd.to_numeric(df["burnout_score"], errors="coerce")
    y_score = y_score.fillna(y_score.median())
    y_level = pd.Categorical(df["burnout_level"], categories=TIER_ORDER, ordered=True)
    y_level = pd.Series(y_level, index=df.index)

    # --- Split for evaluation ---
    X_train, X_test, y_train_score, y_test_score, y_train_level, y_test_level = train_test_split(
        X, y_score, y_level, test_size=0.2, random_state=42, stratify=y_level
    )
    print(f"  | Train/Test split: {len(X_train)} train, {len(X_test)} test")

    print("  | Training ordinal head (cumulative classifiers)...")
    ordinal_models, ordinal_eval = train_ordinal_head(
        X_train, X_test, y_train_level, y_test_level, monotone_str
    )

    print("  | Training quantile head (Burnout Index intervals)...")
    quantile_models, quantile_eval = train_quantile_head(
        X_train, X_test, y_train_score, y_test_score, monotone_str
    )

    # Sanity-check: also report plain point-estimate regression quality (median model)
    median_preds = quantile_models[0.5].predict(X_test)
    r2 = r2_score(y_test_score, median_preds)
    rmse = np.sqrt(mean_squared_error(y_test_score, median_preds))
    mae = mean_absolute_error(y_test_score, median_preds)
    print(f"      [median model as point estimate] R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

    # --- Refit both heads on full dataset for production ---
    print("  | Re-fitting both heads on full dataset for production...")
    full_ordinal_models, _ = train_ordinal_head(X, X, y_level, y_level, monotone_str)
    full_quantile_models, _ = train_quantile_head(X, X, y_score, y_score, monotone_str)

    thresholds = compute_thresholds(y_score)
    print(f"  | Quantile-based thresholds: {thresholds}")

    # --- Fairness flag (informational) ---
    print("  | Running gender-attribution fairness check (informational)...")
    fairness_flags = gender_shap_flag(full_ordinal_models, X, gender_cols)
    if fairness_flags:
        print(f"      mean |SHAP| for gender dummies by cumulative head: {fairness_flags}")

    # --- Save models ---
    for label, clf in full_ordinal_models.items():
        safe_label = label.replace(">=", "ge_")
        path = os.path.join(output_dir, f"domain5_ordinal_{safe_label}.json")
        clf.get_booster().save_model(path)

    for q, reg in full_quantile_models.items():
        path = os.path.join(output_dir, f"domain5_quantile_q{int(q*100)}.json")
        reg.get_booster().save_model(path)

    # --- Metadata contract ---
    metadata = {
        "schema_version": "3.0",
        "domain": "domain_5_occupational_burnout",
        "feature_order": feature_order,
        "raw_work_features": RAW_WORK_FEATURES,
        "demographic_features": DEMOGRAPHIC_FEATURES,
        "gender_dummy_columns": gender_cols,
        "engineered_features": engineered_features,
        "monotone_constraints": dict(zip(feature_order, [int(c) for c in monotone_str.strip("()").split(",")])),
        "tier_order": TIER_ORDER,
        "ordinal_heads": [f">={t}" for t in TIER_ORDER[1:]],
        "quantile_levels": QUANTILES,
        "thresholds": thresholds,
        "model_files": {
            "ordinal": {
                label.replace(">=", "ge_"): f"domain5_ordinal_{label.replace('>=', 'ge_')}.json"
                for label in full_ordinal_models
            },
            "quantile": {
                f"q{int(q*100)}": f"domain5_quantile_q{int(q*100)}.json" for q in full_quantile_models
            },
        },
        "notes": (
            "age and gender are included as features. gender is one-hot encoded; "
            "age and all gender dummy columns carry NO monotonic constraint "
            "(direction not assumed a priori). The 8 work-related raw features plus "
            "the 3 engineered features (stress_x_support, hours_over_50, "
            "meeting_load_ratio) carry signed monotonic constraints. "
            "Tier probabilities at inference time should be derived from the 3 "
            "cumulative P[>=tier] outputs via cumulative_probs_to_tier_probs()-style "
            "differencing, not from a single point regression."
        ),
    }
    output_meta_path = os.path.join(output_dir, "domain5_burnout_metadata.json")
    with open(output_meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # --- Evaluation metrics ---
    eval_metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    domain_metrics = {
        "ordinal_head": ordinal_eval,
        "quantile_head": quantile_eval,
        "median_model_point_estimate": {
            "r2": round(float(r2), 4),
            "rmse": round(float(rmse), 4),
            "mae": round(float(mae), 4),
        },
        "test_set_size": int(len(X_test)),
        "fairness_gender_mean_abs_shap": fairness_flags,
    }
    if os.path.exists(eval_metrics_path):
        with open(eval_metrics_path, "r") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}
    all_metrics["domain_5_occupational_burnout"] = domain_metrics
    with open(eval_metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"[SUCCESS] Domain 5 ordinal + quantile models, metadata, and eval metrics exported.")
    print(f"   +-- Models dir: {output_dir}")
    print(f"   +-- Metadata: {output_meta_path}")
    print(f"   +-- Eval metrics: {eval_metrics_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-synthetic", action="store_true",
                         help="Allow falling back to a synthetic cohort if real data is "
                              "missing/insufficient. DEV/TEST ONLY -- never use in production runs.")
    args = parser.parse_args()
    train_burnout_model(allow_synthetic=args.allow_synthetic)