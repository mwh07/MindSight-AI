#!/usr/bin/env python3
"""
MINDSIGHT Training Data State Flushing Utility
Completely purges generated model weights, parameters, metadata states, and configurations
from deployment tracking directories, establishing fresh placeholders for clean repository pushes.
"""

import os
import shutil

def flush_training_artifacts():
    print("🧹 Initializing MINDSIGHT Training State Flush...")

    # Define the exact core operational resource directories to wipe clean
    target_dirs = [
        "models/saved_states",
        "models/weights"
    ]

    for d in target_dirs:
        if os.path.exists(d):
            print(f"\n⚠️ Purging contents of tracking folder: {d}")
            
            # Safely iterate and destroy all nested elements inside the folder
            for filename in os.listdir(d):
                file_path = os.path.join(d, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                        print(f"   🗑️ Removed file: {file_path}")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        print(f"   🗑️ Removed subdirectory: {file_path}")
                except Exception as e:
                    print(f"   ❌ Failed to delete {file_path}. Reason: {e}")
        else:
            # Ensure folder structure exists even if not previously built
            os.makedirs(d, exist_ok=True)

        # Establish a fresh, clean placeholder .gitkeep to preserve directory context in Git
        gitkeep_path = os.path.join(d, ".gitkeep")
        with open(gitkeep_path, "w") as f:
            pass
        print(f"   📝 Placeholder established: {gitkeep_path} (Tracked on Git)")

    print("\n✨ Success! All transient model states and weights flushed cleanly.")

if __name__ == "__main__":
    flush_training_artifacts()