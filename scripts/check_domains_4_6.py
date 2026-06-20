import os
import pickle
import json

# Check Domain 4 files
d4_model = os.path.join("models", "saved_states", "domain4_multitask.pkl")
d4_meta = os.path.join("models", "saved_states", "domain4_multitask_metadata.json")
print(f"Domain 4 Model Exists: {os.path.exists(d4_model)}")
print(f"Domain 4 Metadata Exists: {os.path.exists(d4_meta)}")

# Check Domain 6 weights
d6_model = os.path.join("models", "saved_states", "domain6_clinical.pkl")
if os.path.exists(d6_model):
    with open(d6_model, "rb") as f:
        payload = pickle.load(f)
    clf = payload.get("classifier")
    if hasattr(clf, "coef_"):
        print("Domain 6 Model Coefficients:", clf.coef_)
        print("Domain 6 Intercept:", clf.intercept_)
    else:
        print("Domain 6 Classifier has no coefficients loaded.")
else:
    print("Domain 6 model file not found.")