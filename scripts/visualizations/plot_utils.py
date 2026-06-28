import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from models.feature_mappings import FEATURE_TRANSLATION_MAP
except ImportError:
    FEATURE_TRANSLATION_MAP = {}

def get_display_name(feature_key):
    return FEATURE_TRANSLATION_MAP.get(feature_key, feature_key)

def ensure_output_dir():
    out_dir = os.path.join(PROJECT_ROOT, "results", "aggregate_analysis")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def load_domain4_models():
    model_path = os.path.join(PROJECT_ROOT, "models", "saved_states", "domain4_digital_social.pkl")
    meta_path = os.path.join(PROJECT_ROOT, "models", "saved_states", "domain4_digital_social_metadata.json")
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        alt_model = os.path.join(PROJECT_ROOT, "models", "saved_states", "domain4_multitask.pkl")
        alt_meta = os.path.join(PROJECT_ROOT, "models", "saved_states", "domain4_multitask_metadata.json")
        if os.path.exists(alt_model) and os.path.exists(alt_meta):
            model_path, meta_path = alt_model, alt_meta
        else:
            raise FileNotFoundError("Domain 4 model files not found.")

    with open(model_path, "rb") as f:
        model_payload = pickle.load(f)
    with open(meta_path, "r") as f:
        metadata = json.load(f)

    addiction_model = None
    loneliness_model = None
    if isinstance(model_payload, dict):
        for key in ["addiction_model", "iat_model", "internet_addiction_model", "model_addiction", "rf_iat"]:
            if key in model_payload:
                addiction_model = model_payload[key]
                break
        for key in ["loneliness_model", "lone_model", "model_loneliness", "rf_lone"]:
            if key in model_payload:
                loneliness_model = model_payload[key]
                break
        if addiction_model is None or loneliness_model is None:
            candidates = [v for v in model_payload.values() if hasattr(v, "predict")]
            for cand in candidates:
                fit_features = list(getattr(cand, "feature_names_in_", []))
                if any(f.startswith("IAT") for f in fit_features) and addiction_model is None:
                    addiction_model = cand
                elif any(f.startswith("loneliness") for f in fit_features) and loneliness_model is None:
                    loneliness_model = cand
    else:
        raise ValueError("Model payload is not a dict; cannot extract two models.")

    if addiction_model is None or loneliness_model is None:
        raise ValueError("Could not resolve both addiction and loneliness models.")

    return addiction_model, loneliness_model, metadata

def load_domain5_model():
    model_path = os.path.join(PROJECT_ROOT, "models", "saved_states", "domain5_burnout.json")
    meta_path = os.path.join(PROJECT_ROOT, "models", "saved_states", "domain5_burnout_metadata.json")
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Domain 5 model files not found.")
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    bst = xgb.Booster()
    bst.load_model(model_path)
    return bst, metadata

def load_domain6_model():
    model_path = os.path.join(PROJECT_ROOT, "models", "saved_states", "domain6_clinical.pkl")
    meta_path = os.path.join(PROJECT_ROOT, "models", "saved_states", "domain6_clinical_metadata.json")
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Domain 6 model files not found.")
    with open(model_path, "rb") as f:
        model_payload = pickle.load(f)
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    classifier = model_payload["classifier"]
    return classifier, metadata

def get_dataset(domain_name):
    if domain_name == "domain4":
        path = os.path.join(PROJECT_ROOT, "datasets", "internet_phq_loneliness_clean.csv")
    elif domain_name == "domain5":
        path = os.path.join(PROJECT_ROOT, "datasets", "tech_burnout_2026_clean.csv")
    elif domain_name == "domain6":
        path = os.path.join(PROJECT_ROOT, "datasets", "ocd_symptoms_clean.csv")
    else:
        raise ValueError(f"Unknown domain: {domain_name}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def generate_shap_summary_and_bar(model, X, feature_names, model_name, out_dir, sample_size=500):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=feature_names)
    else:
        if list(X.columns) != feature_names:
            X = X[feature_names]

    if sample_size is not None and len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X

    if isinstance(model, (RandomForestRegressor, xgb.XGBRegressor)):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"{model_name}_shap_summary.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  | Saved SHAP summary plot: {save_path}")

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_vals = mean_abs_shap[sorted_idx]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(len(sorted_features)), sorted_vals, color='steelblue')
    plt.yticks(range(len(sorted_features)), [get_display_name(f) for f in sorted_features])
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"Mean Absolute SHAP Value per Feature\n{model_name}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    save_path_bar = os.path.join(out_dir, f"{model_name}_mean_shap_bar.png")
    plt.savefig(save_path_bar, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  | Saved mean SHAP bar chart: {save_path_bar}")

    return save_path, save_path_bar
