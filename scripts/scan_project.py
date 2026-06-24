import argparse
from pathlib import Path


def print_tree(directory, prefix="", exclude_dirs=None, exclude_files=None):
    """
    Recursively print directory tree.
    Directories in exclude_dirs are shown as 'name/...' and not traversed.
    Files in exclude_files are omitted entirely.
    """
    if exclude_dirs is None:
        exclude_dirs = {'.git', '__pycache__', '.venv', 'venv',
                        '.pytest_cache', '.vscode', '.idea'}
    if exclude_files is None:
        exclude_files = {'.DS_Store', 'desktop.ini', 'scan_project.py'}

    try:
        entries = sorted(
            list(Path(directory).iterdir()),
            key=lambda s: (s.is_file(), s.name.lower())
        )
        # Filter out *files* that are in exclude_files, but keep all directories
        entries = [e for e in entries
                   if not (e.is_file() and e.name in exclude_files)]
    except PermissionError:
        return

    for i, entry in enumerate(entries):
        is_last = (i == len(entries) - 1)
        connector = "└── " if is_last else "├── "

        if entry.is_dir():
            if entry.name in exclude_dirs:
                # Show directory with an ellipsis and do not recurse
                print(f"{prefix}{connector}{entry.name}/...")
            else:
                print(f"{prefix}{connector}{entry.name}/")
                new_prefix = prefix + ("    " if is_last else "│   ")
                print_tree(entry, new_prefix, exclude_dirs, exclude_files)
        else:
            print(f"{prefix}{connector}{entry.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Print directory tree with exclusions."
    )
    parser.add_argument(
        '--ignore',
        nargs='*',
        default=[],
        help="Additional subfolder names to ignore (space-separated)."
    )
    args = parser.parse_args()

    # Clean user-provided ignores: strip trailing slashes (and leading './')
    extra_ignores = {name.strip('/').lstrip('./') for name in args.ignore}

    default_exclude_dirs = {
        '.git', '__pycache__', '.venv', 'venv',
        '.pytest_cache', '.vscode', '.idea'
    }
    exclude_dirs = default_exclude_dirs.union(extra_ignores)

    root_dir = Path(".")
    print(f"Project Root: {root_dir.resolve().name}\n")
    print_tree(root_dir, exclude_dirs=exclude_dirs)


if __name__ == "__main__":
    main()