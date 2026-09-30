"""
Stage 4: train YOLO26 on the dataset built in stage 3 - segmentation,
detection or classification, whichever the dataset was built for.

Run from the repo root:
    uv run python src/train.py --output-dir <folder>
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO

from synth_pipeline.utils.yolo_dataset import DATASET_DIR, TASKS, read_task

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
        help="Pretrained weights. Default: yolo26n-seg.pt for a segmentation dataset, yolo26n.pt for detection, "
        "yolo26n-cls.pt for classification. Larger ones (yolo26s/m/l/x) are more accurate and slower.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, help="Training image size. Default: 224 for classification, else 640.")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", help="Training device: e.g. 0 (first CUDA GPU), mps (Apple GPU), cpu. Default: auto.")
    parser.add_argument(
        "--name", help="Run name under <output-dir>/runs/ (auto-suffixed if taken). Default: the task name."
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    return args


# Must be guarded by __name__ == "__main__" on Windows: Ultralytics' DataLoader
# spawns worker processes, and Windows' multiprocessing "spawn" start method
# re-imports this module in each worker - without the guard that re-triggers
# model.train() itself and crashes with a "freeze_support()" RuntimeError.
if __name__ == "__main__":
    args = parse_args()
    dataset_dir = args.output_dir / DATASET_DIR
    task = read_task(dataset_dir)
    WEIGHTS_CACHE.mkdir(parents=True, exist_ok=True)
    os.chdir(WEIGHTS_CACHE)  # safe: every path passed to Ultralytics below is absolute
    model = YOLO(args.model or TASKS[task])  # COCO-pretrained

    model.train(
        # Classification reads the class folders directly; the others read data.yaml.
        data=str(dataset_dir if task == "classify" else dataset_dir / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz or (224 if task == "classify" else 640),
        batch=args.batch,
        project=str(args.output_dir / "runs"),
        name=args.name or task,
        device=args.device,
    )
