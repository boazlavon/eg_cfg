#!/usr/bin/env python3

"""
redirect_env_to_submodules.py

Redirects selected Python packages in the current conda environment
to local submodule sources by replacing them with symlinks.

Assumes:
- The conda environment is active.
- Python version is hardcoded (e.g., python3.9).
- You provide the path to the submodules directory.
"""

import os
import sys
import argparse
from pathlib import Path

# === Constants ===

SITE_PACKAGES_SUFFIX = "lib/python3.9/site-packages"

# Mapping: site-packages module name → submodules dir name
MODULE_REDIRECTS = {
    "trepan": "trepan",
    "xpython": "xpython",
    "trepanxpy": "trepan-xpy",
    "transformers": "transformers",
}

# === Core logic ===


def redirect_modules(submodules_dir: Path, site_packages_dir: Path, dry_run: bool):
    for module_name, submodule_dir in MODULE_REDIRECTS.items():
        module_path = site_packages_dir / module_name
        target_path = submodules_dir / submodule_dir

        print(f"Redirecting {module_name} → {target_path}")

        if module_path.exists() and not module_path.is_symlink():
            backup_path = site_packages_dir / f".{module_name}"
            print(f"  - Would move original to {backup_path}")
            if not dry_run:
                module_path.rename(backup_path)

        elif module_path.is_symlink():
            print(f"  - Would remove existing symlink")
            if not dry_run:
                module_path.unlink()

        print(f"  - Would create symlink: {module_path} -> {target_path}")
        if not dry_run:
            module_path.symlink_to(target_path)
    if not dry_run:
        print("\n🔗 Current symlinks in site-packages:")
        os.system(f"ls -la {site_packages_dir} | grep '\->'")


# === CLI entry ===


def main():
    parser = argparse.ArgumentParser(
        description="Redirect conda site-packages to local submodules via symlinks."
    )
    parser.add_argument(
        "submodules_dir", type=Path, help="Path to submodules directory"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print actions without applying changes"
    )
    args = parser.parse_args()

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        sys.exit("Error: You must activate a conda environment first.")

    site_packages_dir = Path(conda_prefix) / SITE_PACKAGES_SUFFIX
    if not site_packages_dir.exists():
        sys.exit(f"Error: site-packages directory not found: {site_packages_dir}")
    if not args.submodules_dir.exists():
        sys.exit(f"Error: submodules directory not found: {args.submodules_dir}")

    redirect_modules(
        args.submodules_dir.resolve(), site_packages_dir.resolve(), args.dry_run
    )


if __name__ == "__main__":
    main()
