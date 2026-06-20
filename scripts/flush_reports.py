#!/usr/bin/env python3
"""
flush_reports.py

Deletes all contents of the `individual_eval/` directory (files and subfolders)
without confirmation, then places a .gitkeep file inside so the empty folder
remains tracked by Git. Use with caution.
"""

import os
import shutil
import sys


def flush_individual_eval():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    eval_dir = os.path.join(project_root, "individual_eval")

    if not os.path.isdir(eval_dir):
        print(f"Error: Directory not found: {eval_dir}")
        sys.exit(1)

    contents = os.listdir(eval_dir)
    if contents:
        print(f"Deleting all contents of {eval_dir} ...")
        for item in contents:
            item_path = os.path.join(eval_dir, item)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"  Removed directory: {item}")
                else:
                    os.remove(item_path)
                    print(f"  Removed file: {item}")
            except Exception as e:
                print(f"Error removing {item_path}: {e}")
                sys.exit(1)
    else:
        print("individual_eval is already empty.")

    # Write a fresh .gitkeep file
    gitkeep_path = os.path.join(eval_dir, ".gitkeep")
    with open(gitkeep_path, "w") as f:
        pass
    print(f"Created {gitkeep_path}")

    print("Flush complete.")


if __name__ == "__main__":
    flush_individual_eval()