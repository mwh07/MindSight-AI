import os
import pickle

model_path = os.path.join("models", "saved_states", "domain1_grm_parameters.pkl")
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        grm_registry = pickle.load(f)
    
    print("Traits found in registry:", list(grm_registry.keys()))
    for trait, data in grm_registry.items():
        items = list(data["items"].keys())
        print(f"  -> {trait}: Mapped Items = {items}")
        if items:
            sample_item = items[0]
            print(f"     Sample Parameter Structure ({sample_item}):", data["items"][sample_item])
else:
    print(f"Could not locate serialized state at: {model_path}")