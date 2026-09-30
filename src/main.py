"""
Pipeline entry point. Point it at a folder of CAD parts and it runs the
whole thing: STEP -> GLB -> synthetic renders (with instance-segmentation
masks) -> YOLO-seg dataset -> trained model.

Input folder contract: one subfolder per class, each containing either a
.step/.stp file directly, or a single .zip that contains one
(see utils/discover.py). Nothing about the parts needs to be declared
anywhere else - classes are discovered from the folder names.

Every stage runs in the uv venv (Blender is the `bpy` package there), each
in its own process, so a failing stage stops the pipeline with a nonzero exit:
  1. convert_assets.py      - STEP -> GLB (OpenCascade)
  2. generate_dataset.py    - GLB -> rendered frames + masks (Blender)
  3. build_yolo_dataset.py  - masks -> YOLO-seg labels
  4. train.py               - YOLO26-seg training

Usage (from the repo root):
    uv run python src/main.py --input-dir examples/fasteners --output-dir workspace
    uv run python src/main.py --input-dir examples/fasteners --output-dir workspace --stage convert
    uv run python src/main.py --input-dir examples/fasteners --output-dir workspace --from generate
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

STAGES = [
    ("convert", HERE / "convert_assets.py"),
    ("generate", HERE / "generate_dataset.py"),
    ("dataset", HERE / "build_yolo_dataset.py"),
    ("train", HERE / "train.py"),
]
STAGE_NAMES = [s[0] for s in STAGES]


def stage_args(name: str, args) -> list:
    """Arguments beyond --output-dir that a given stage accepts."""
    if name == "convert":
        return ["--input-dir", str(args.input_dir)]
    if name == "generate":
        extra = []
        if args.num_frames is not None:
            extra += ["--num-frames", str(args.num_frames)]
        if args.textures_dir is not None:
            extra += ["--textures-dir", str(args.textures_dir)]
        return extra
    return []


def run_stage(name: str, script: Path, args) -> None:
    # sys.executable: the uv venv's Python, as long as main.py itself was
    # launched with `uv run python`.
    cmd = [sys.executable, str(script), "--output-dir", str(args.output_dir)] + stage_args(name, args)

    print(f"\n=== stage '{name}': {' '.join(cmd)} ===\n", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"Stage '{name}' failed (exit {result.returncode}) - stopping pipeline.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, type=Path, help="Folder with one subfolder per part class.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Where generated meshes/dataset/runs go.")
    parser.add_argument("--stage", choices=STAGE_NAMES, help="Run only this stage.")
    parser.add_argument("--from", dest="from_stage", choices=STAGE_NAMES, help="Run this stage and all after it.")
    parser.add_argument("--num-frames", type=int, help="Frames to render (default: pyproject.toml's num_frames).")
    parser.add_argument("--textures-dir", type=Path, help="Folder of photos to mix into ground/backdrop textures.")
    args = parser.parse_args()

    # Resolve to absolute paths immediately - user-typed relative paths are
    # relative to wherever they ran this command from.
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.textures_dir is not None:
        args.textures_dir = args.textures_dir.resolve()

    if args.stage:
        stages_to_run = [s for s in STAGES if s[0] == args.stage]
    elif args.from_stage:
        start = STAGE_NAMES.index(args.from_stage)
        stages_to_run = STAGES[start:]
    else:
        stages_to_run = STAGES

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, script in stages_to_run:
        run_stage(name, script, args)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
