"""
Stage 4: train YOLO26-seg (instance segmentation) on the dataset built in
stage 3.

Run from the repo root:
    uv run python src/train.py --output-dir <folder>
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO

# Ultralytics downloads pretrained weights into the working directory (plus a
# separate yolo26n.pt it uses only to self-test mixed precision). Training
# runs from this cache instead, so those are downloaded once and never land
# in the repo.
WEIGHTS_CACHE = Path.home() / ".cache" / "synth-pipeline"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="Same one passed to earlier stages.")
    parser.add_argument(
        "--model",
        default="yolo26n-seg.pt",
        help="Pretrained weights: yolo26n-seg.pt (fastest) up to yolo26s/m/l/x-seg.pt (more accurate, slower).",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", help="Training device: e.g. 0 (first CUDA GPU), mps (Apple GPU), cpu. Default: auto.")
    parser.add_argument("--name", default="seg", help="Run name under <output-dir>/runs/ (auto-suffixed if taken).")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    return args


# Must be guarded by __name__ == "__main__" on Windows: Ultralytics' DataLoader
# spawns worker processes, and Windows' multiprocessing "spawn" start method
# re-imports this module in each worker - without the guard that re-triggers
# model.train() itself and crashes with a "freeze_support()" RuntimeError.
if __name__ == "__main__":
    args = parse_args()
    WEIGHTS_CACHE.mkdir(parents=True, exist_ok=True)
    os.chdir(WEIGHTS_CACHE)  # safe: every path passed to Ultralytics below is absolute
    model = YOLO(args.model)  # COCO-pretrained, segment task

    model.train(
        data=str(args.output_dir / "yolo_seg_dataset" / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.output_dir / "runs"),
        name=args.name,
        device=args.device,
    )
