#!/usr/bin/env python3
"""
MINDSIGHT — Domain 1 Item Characteristic Curves (ICC)

This script loads the psychometric Graded Response Model (GRM) parameters for Domain 1 (Personality).
It mathematically extracts the 'alpha' (discrimination) and 'beta' (difficulty thresholds) 
and plots the Item Characteristic Curves (ICC) to visually prove how the model interprets 
patient responses based on their underlying latent trait (Theta).

Generated plots are saved to results/aggregate_analysis/domain1_icc/
"""
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ========== NEW: We no longer need plot_utils for output dir ==========
# from plot_utils import ensure_output_dir   # <-- REMOVED

def load_domain1_grm():
    model_path = os.path.join(PROJECT_ROOT, "models", "saved_states", "domain1_grm_parameters.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Domain 1 GRM parameters not found at {model_path}")
    
    with open(model_path, "rb") as f:
        grm_registry = pickle.load(f)
    return grm_registry

def compute_grm_probabilities(theta, a, b_list):
    """
    Compute Graded Response Model (GRM) category probabilities.
    theta: latent trait value (scalar or numpy array)
    a: discrimination parameter (scalar)
    b_list: list of difficulty parameters (thresholds)
    Returns: Array of shape (len(b_list)+1, len(theta)) containing probabilities for each category
    """
    theta = np.asarray(theta)
    
    # Cumulative probabilities P*(X >= k)
    # k=1 is always 1.0 (everyone answers at least category 1)
    p_star = [np.ones_like(theta)]
    
    for b in b_list:
        # Logistic function for P*(X >= k)
        p = 1.0 / (1.0 + np.exp(-a * (theta - b)))
        p_star.append(p)
        
    # k = K+1 is always 0.0 (no one answers above the max category)
    p_star.append(np.zeros_like(theta))
    
    # Category probabilities P(X = k) = P*(X >= k) - P*(X >= k+1)
    p_cat = []
    for i in range(len(b_list) + 1):
        p_cat.append(p_star[i] - p_star[i+1])
        
    return np.array(p_cat)

def generate_icc_plot(trait_name, item_name, a, b_list, out_dir):
    theta_range = np.linspace(-4, 4, 300)
    probabilities = compute_grm_probabilities(theta_range, a, b_list)
    
    plt.figure(figsize=(10, 6))
    colors = ['#d7191c', '#fdae61', '#ffffbf', '#abdda4', '#2b83ba']
    
    for i in range(probabilities.shape[0]):
        category_label = i + 1
        plt.plot(theta_range, probabilities[i], label=f'Score {category_label}', 
                 color=colors[i], linewidth=2.5)
                 
    plt.title(f"Item Characteristic Curves (ICC) - {trait_name.upper()} ({item_name})\n"
              f"Discrimination (a) = {a:.2f}", fontsize=14)
    plt.xlabel("Latent Trait (θ) Standard Deviations", fontsize=12)
    plt.ylabel("Probability of Selection", fontsize=12)
    plt.legend(title="Response Category", loc="center right", bbox_to_anchor=(1.2, 0.5))
    plt.grid(True, alpha=0.3)
    
    # Add threshold markers
    for b in b_list:
        plt.axvline(x=b, color='gray', linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"domain1_icc_{trait_name}_{item_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  | Saved ICC Plot for {item_name}: {save_path}")
    return save_path

def main():
    print("\n📊 Domain 1 — Personality Psychometrics (GRM-IRT)")
    
    # ========== NEW: Define dedicated output subfolder ==========
    out_dir = os.path.join(PROJECT_ROOT, "results", "aggregate_analysis", "domain1_icc")
    os.makedirs(out_dir, exist_ok=True)
    
    try:
        grm_registry = load_domain1_grm()
        
        # We will plot ICCs for the first item of each trait to demonstrate the math
        traits_to_plot = ["EXT", "EST", "AGR"]
        
        for trait in traits_to_plot:
            if trait not in grm_registry:
                continue
                
            items = list(grm_registry[trait]["items"].keys())
            if not items:
                continue
                
            # Pick the first item (e.g., EXT1)
            target_item = items[0]
            params = grm_registry[trait]["items"][target_item]
            
            print(f"  ⏳ Computing ICC for {trait.upper()} -> {target_item}...")
            generate_icc_plot(
                trait_name=trait,
                item_name=target_item,
                a=params["alpha"],
                b_list=params["beta"],
                out_dir=out_dir
            )
            
        print("  ✅ Domain 1 ICC plots generated.")
        
    except Exception as e:
        print(f"  ❌ Domain 1 failed: {e}")

if __name__ == "__main__":
    main()