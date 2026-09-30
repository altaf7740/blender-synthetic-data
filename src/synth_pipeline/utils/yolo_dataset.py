"""
Shared helpers for building an Ultralytics-style YOLO dataset folder
(images/labels train+val split, data.yaml).
"""

from pathlib import Path


def make_dataset_dirs(dst_dir: Path) -> None:
    for split in ("train", "val"):
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


def write_data_yaml(dst_dir: Path, class_names: list) -> Path:
    names_yaml = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(class_names))
    content = f"""path: {dst_dir}
train: images/train
val: images/val
names:
{names_yaml}
"""
    path = dst_dir / "data.yaml"
    path.write_text(content)
    return path
