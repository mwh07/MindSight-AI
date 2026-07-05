#!/usr/bin/env python3
"""
flush.py - Delete all contents of given folders, but keep/create a .gitkeep.

Usage:
    py flush.py folder1 folder2 folder3 ...

Examples:
    py flush.py reports/ individual_eval/
    py flush.py results/ logs/ temp/
"""

import sys
import shutil
from pathlib import Path


def flush_folder(folder_path: str) -> None:
    """Delete all contents inside `folder_path`, preserving/creating .gitkeep."""
    p = Path(folder_path).resolve()

    # If folder doesn't exist, create it and add .gitkeep
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
        (p / '.gitkeep').touch()
        print(f"✅ Created: {p} (with .gitkeep)")
        return

    # Folder exists – delete everything except .gitkeep
    for item in p.iterdir():
        if item.name == '.gitkeep':
            continue
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        except Exception as e:
            print(f"⚠️  Could not delete {item}: {e}")

    # Ensure .gitkeep exists (touch it to update timestamp if needed)
    (p / '.gitkeep').touch()
    print(f"✅ Flushed: {p} (kept .gitkeep)")


def main():
    if len(sys.argv) < 2:
        print("❌ Please specify at least one folder.")
        print(f"Usage: {sys.argv[0]} folder1 folder2 ...")
        sys.exit(1)

    for folder in sys.argv[1:]:
        flush_folder(folder)


if __name__ == "__main__":
    main()