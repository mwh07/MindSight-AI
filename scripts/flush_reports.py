#!/usr/bin/env python3
"""
flush_reports.py

Deletes all contents of the `reports/` directory (files and subfolders)
without confirmation, then places a .gitkeep file inside so the empty folder
remains tracked by Git. Use with caution.
"""

import os
import shutil
import sys


def flush_reports():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    reports_dir = os.path.join(project_root, "reports")

    if not os.path.isdir(reports_dir):
        print(f"Error: Directory not found: {reports_dir}")
        sys.exit(1)

    contents = os.listdir(reports_dir)
    if contents:
        print(f"Deleting all contents of {reports_dir} ...")
        for item in contents:
            item_path = os.path.join(reports_dir, item)
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
        print("reports/ is already empty.")

    # Write a fresh .gitkeep file
    gitkeep_path = os.path.join(reports_dir, ".gitkeep")
    with open(gitkeep_path, "w") as f:
        pass
    print(f"Created {gitkeep_path}")

    print("Flush complete.")


if __name__ == "__main__":
    flush_reports()