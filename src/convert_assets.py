"""
Stage 1: for each class subfolder in --input-dir, resolve its STEP file
(extracting a zip first if that's what's there) and convert it to GLB.
Writes <output-dir>/meshes/<class_name>.glb and
<output-dir>/parts_manifest.json for stage 2 to read.

Fails (nonzero exit, no manifest written) if ANY class can't be converted -
silently dropping a class would produce a dataset/model missing that part.

Run from the repo root:
    uv run python src/convert_assets.py --input-dir <folder> --output-dir <folder>
"""

import argparse
import json
from pathlib import Path

from synth_pipeline.utils.discover import discover_classes, resolve_step_file
from synth_pipeline.utils.step_to_glb import convert_step_to_glb


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, type=Path, help="Folder with one subfolder per part class.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Where the GLB files/manifest are written.")
    args = parser.parse_args()
    # Resolve to absolute paths immediately - relative paths are relative to
    # wherever the caller ran this command from.
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.input_dir.is_dir():
        parser.error(f"--input-dir not found: {args.input_dir}")
    return args


def convert_all(input_dir: Path, output_dir: Path) -> dict:
    manifest = {}
    failures = {}
    for class_name in discover_classes(input_dir):
        try:
            step_path = resolve_step_file(input_dir / class_name)
            print(f"[{class_name}] STEP file: {step_path}")
            glb_path = convert_step_to_glb(step_path, output_dir / "meshes" / f"{class_name}.glb")
            print(f"[{class_name}] converted -> {glb_path}")
            manifest[class_name] = str(glb_path)
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"[{class_name}] FAILED: {e}")
            failures[class_name] = str(e)

    # Try every class before failing, so one run reports every problem at once.
    if failures:
        raise SystemExit(f"{len(failures)} class(es) failed to convert: {sorted(failures)}")
    return manifest


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = convert_all(args.input_dir, args.output_dir)

    manifest_path = args.output_dir / "parts_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote manifest for {len(manifest)} classes to {manifest_path}")


if __name__ == "__main__":
    main()
