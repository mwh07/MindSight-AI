import os
from pathlib import Path

def print_tree(directory, prefix="", exclude_dirs=None, exclude_files=None):
    if exclude_dirs is None:
        exclude_dirs = {'.git', '__pycache__', '.venv', 'venv', '.pytest_cache', '.vscode', '.idea'}
    if exclude_files is None:
        exclude_files = {'.DS_Store', 'desktop.ini', 'scan_project.py'}

    try:
        # Fetch and sort directory contents
        entries = sorted(list(Path(directory).iterdir()), key=lambda s: (s.is_file(), s.name.lower()))
        # Filter out excluded items
        entries = [e for e in entries if e.name not in exclude_dirs and e.name not in exclude_files]
    except PermissionError:
        return

    for i, entry in enumerate(entries):
        is_last = (i == len(entries) - 1)
        connector = "└── " if is_last else "├── "
        
        if entry.is_dir():
            print(f"{prefix}{connector}{entry.name}/")
            # Recurse into subdirectory with updated indentation spacing
            new_prefix = prefix + ("    " if is_last else "│   ")
            print_tree(entry, new_prefix, exclude_dirs, exclude_files)
        else:
            print(f"{prefix}{connector}{entry.name}")

if __name__ == "__main__":
    root_dir = Path(".")
    print(f"Project Root: {root_dir.resolve().name}\n")
    print_tree(root_dir)