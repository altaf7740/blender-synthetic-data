"""
Shared helpers for building an Ultralytics-style YOLO dataset folder:
images/ + labels/ (segment, detect) or one folder of crops per class
(classify), split into train/val, plus data.yaml.
"""

import json
from pathlib import Path

import yaml

DATASET_DIR = "yolo_dataset"  # under --output-dir
# Label type -> the pretrained weights train.py starts from by default.
TASKS = {"segment": "yolo26n-seg.pt", "detect": "yolo26n.pt", "classify": "yolo26n-cls.pt"}


def make_dataset_dirs(dst_dir: Path, task: str, class_names: list) -> None:
    for split in ("train", "val"):
        if task == "classify":
            # Every class in both splits, even if empty: Ultralytics numbers
            # classes by folder, so both splits must list the same ones.
            for name in class_names:
                (dst_dir / split / name).mkdir(parents=True, exist_ok=True)
        else:
            (dst_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (dst_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def num_val(total: int, val_split: float) -> int:
    """How many of `total` frames go to the val split.

    At least one whenever there are 2+ frames - a plain int(total * val_split)
    rounds to 0 on small runs (e.g. a 5-frame smoke test), and Ultralytics
    fails to train with an empty val set.
    """
    if total < 2:
        return 0
    return max(1, int(total * val_split))


def split_for_index(i: int, total: int, val_split: float) -> str:
    return "val" if i < num_val(total, val_split) else "train"


def write_data_yaml(dst_dir: Path, class_names: list, task: str) -> Path:
    # Quoted (a JSON string is valid YAML), so names like "yes", "null" or "10"
    # stay strings instead of turning into booleans/numbers.
    names_yaml = "\n".join(f"  {idx}: {json.dumps(name)}" for idx, name in enumerate(class_names))
    # No `path:` key - Ultralytics then resolves train/val against this file's
    # own folder, so the dataset still works after it's moved or downloaded.
    # `task` isn't read by Ultralytics; train.py and preview_dataset.py use it
    # to pick the model and how to draw the labels.
    if task == "classify":
        # Classification reads the class folders directly (data=<this folder>);
        # this file only records the task and class names.
        content = f"""task: {task}
names:
{names_yaml}
"""
    else:
        content = f"""task: {task}
train: images/train
val: images/val
names:
{names_yaml}
"""
    path = dst_dir / "data.yaml"
    path.write_text(content)
    return path


def read_task(dataset_dir: Path) -> str:
    """The label type a dataset was built with ("segment", "detect" or "classify")."""
    return yaml.safe_load((dataset_dir / "data.yaml").read_text()).get("task", "segment")
