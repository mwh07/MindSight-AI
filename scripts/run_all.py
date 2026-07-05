#!/usr/bin/env python3
"""
run_all.py - Executes all Python scripts in one or more specified folders.

Usage:
    python run_all.py [folder1] [folder2] ... [--recursive] [--stop-on-error]

Examples:
    python run_all.py scripts/validations/
    python run_all.py folder1 folder2 folder3
    python run_all.py . --recursive
    python run_all.py src/tests --stop-on-error
    python run_all.py --recursive              # Scans current dir recursively
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def get_py_files(folder, recursive=False):
    """Get all .py files in a folder (optionally recursive)."""
    path = Path(folder)
    if not path.exists():
        print(f"❌ Folder not found: {folder}")
        return []
    
    if recursive:
        files = list(path.rglob("*.py"))
    else:
        files = list(path.glob("*.py"))
    
    return sorted(files)  # Sort for predictable order

def run_script(filepath):
    """Execute a single Python script and return success status."""
    rel_path = filepath.relative_to(Path.cwd()) if filepath.is_relative_to(Path.cwd()) else filepath
    
    print(f"\n{'='*70}")
    print(f"▶️  Running: {rel_path}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(filepath)],
            capture_output=True,
            text=True,
            cwd=filepath.parent,
            timeout=300  # 5-minute timeout (adjust as needed)
        )
        
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: {rel_path} (exit code: {result.returncode})")
            return True
        else:
            print(f"❌ FAILED: {rel_path} (exit code: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT: {rel_path} exceeded 5 minutes")
        return False
    except Exception as e:
        print(f"💥 ERROR: {rel_path} crashed - {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Execute all Python scripts in one or more folders.",
        epilog="Tip: Name your scripts with numeric prefixes (e.g., 01_setup.py, 02_process.py) to control order."
    )
    parser.add_argument(
        "folders",
        nargs="*",  # Accept zero or more folder arguments
        default=["."],
        help="One or more folders containing the scripts (default: current directory)"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recursively scan subdirectories for .py files"
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop execution immediately if any script fails"
    )
    parser.add_argument(
        "--skip-self",
        action="store_true",
        default=True,
        help="Skip the script itself if it's in the target folder (default: True)"
    )
    
    args = parser.parse_args()
    
    # Collect all .py files from all specified folders
    all_files = []
    this_script = Path(__file__).resolve()
    
    for folder in args.folders:
        folder_path = Path(folder).resolve()
        files = get_py_files(folder_path, args.recursive)
        
        # Filter out this script itself if --skip-self is True
        if args.skip_self:
            files = [f for f in files if f != this_script]
        
        all_files.extend(files)
    
    if not all_files:
        print(f"⚠️  No .py files found in: {', '.join(args.folders)}")
        return
    
    # Remove duplicates (if same file appears from multiple paths)
    all_files = sorted(set(all_files))
    
    print(f"📂 Found {len(all_files)} Python script(s) across {len(args.folders)} folder(s)")
    print(f"{'='*70}\n")
    
    # Run each script
    success_count = 0
    for idx, filepath in enumerate(all_files, 1):
        print(f"  [{idx}/{len(all_files)}]")
        success = run_script(filepath)
        
        if success:
            success_count += 1
        elif args.stop_on_error:
            print(f"\n🛑 Stopping due to error as requested.")
            sys.exit(1)
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"📊 Summary: {success_count}/{len(all_files)} scripts succeeded.")
    if success_count == len(all_files):
        print("✅ All scripts executed successfully!")
        sys.exit(0)
    else:
        print(f"❌ {len(all_files) - success_count} script(s) failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()