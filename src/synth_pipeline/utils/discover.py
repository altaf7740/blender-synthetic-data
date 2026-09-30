"""
Input-folder discovery. The pipeline's only contract for an input folder is:

    <input_dir>/<class_name>/

one subfolder per part class, each containing either a .step/.stp file
directly, or a single .zip that contains one when extracted. Nothing else
about a project's parts needs to be declared anywhere else in this repo -
classes are discovered from the folder names at run time.
"""

import zipfile
from pathlib import Path

STEP_SUFFIXES = {".step", ".stp"}


def _find_step_file(search_dir: Path) -> Path | None:
    matches = [p for p in search_dir.rglob("*") if p.suffix.lower() in STEP_SUFFIXES]
    if len(matches) > 1:
        raise ValueError(f"Multiple STEP files found under {search_dir}, expected exactly one: {matches}")
    return matches[0] if matches else None


def resolve_step_file(class_dir: Path) -> Path:
    """Resolve the STEP file for one class's input subfolder.

    If a .step/.stp is already present, use it directly. Otherwise, if the
    subfolder contains exactly one .zip, extract it in place first.

    Raises:
        FileNotFoundError: If no STEP file is found (directly, or after
            extracting a zip, if one is present).
        ValueError: If the input is ambiguous (multiple STEP files, or
            multiple zips with no direct STEP file already present).
    """
    step_file = _find_step_file(class_dir)
    if step_file:
        return step_file

    zips = list(class_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No .step/.stp or .zip found under {class_dir}")
    if len(zips) > 1:
        raise ValueError(f"Multiple .zip files found under {class_dir}, expected exactly one: {zips}")

    with zipfile.ZipFile(zips[0]) as zf:
        zf.extractall(class_dir)

    step_file = _find_step_file(class_dir)
    if not step_file:
        raise FileNotFoundError(f"{zips[0].name} did not contain a .step/.stp file")
    return step_file


def discover_classes(input_dir: Path) -> list:
    """List class names from an input folder's immediate subdirectories."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    classes = sorted(p.name for p in input_dir.iterdir() if p.is_dir())
    if not classes:
        raise ValueError(f"No class subfolders found under {input_dir} - expected one subfolder per part.")
    return classes
