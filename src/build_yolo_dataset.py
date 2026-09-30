"""
Stage 3: convert the rendered frames and instance masks (stage 2) into an
Ultralytics YOLO dataset, with either outline labels (--task segment) or
box labels (--task detect). Classes are read from parts_manifest.json
(written in stage 1) rather than declared again here.

Run from the repo root:
    uv run python src/build_yolo_dataset.py --output-dir <folder> [--task segment|detect]
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from synth_pipeline import config
from synth_pipeline.utils.labels import (
    frame_to_yolo_box_lines,
    frame_to_yolo_seg_lines,
    load_instance_class_map,
    load_instance_ids,
)
from synth_pipeline.utils.yolo_dataset import (
    DATASET_DIR,
    TASKS,
    make_dataset_dirs,
    num_val,
    split_for_index,
    write_data_yaml,
)


def camera_effects(img: Image.Image, rng) -> Image.Image:
    """Degrade a clean render the way a real camera would: defocus/motion
    blur, sensor noise, JPEG compression. Each independently, so some frames
    stay clean. Geometry is untouched, so labels stay valid."""
    arr = np.asarray(img.convert("RGB"))
    if rng.random() < 0.35:
        k = int(rng.choice([3, 5]))
        if rng.random() < 0.5:
            arr = cv2.GaussianBlur(arr, (k, k), 0)
        else:  # short motion streak at a random angle
            kernel = np.zeros((k, k), np.float32)
            kernel[k // 2, :] = 1.0 / k
            rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), float(rng.uniform(0, 180)), 1.0)
            kernel = cv2.warpAffine(kernel, rot, (k, k))
            arr = cv2.filter2D(arr, -1, kernel / kernel.sum())
    if rng.random() < 0.4:
        noise = rng.normal(0.0, rng.uniform(2, 8), arr.shape)
        arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        ok, buf = cv2.imencode(".jpg", arr[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 95))])
        arr = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]
    return Image.fromarray(np.ascontiguousarray(arr))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path, help="Same one passed to earlier stages.")
    parser.add_argument(
        "--task", choices=TASKS, default="segment", help="Label type: segment (outlines) or detect (boxes)."
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    return args


def main():
    args = parse_args()
    manifest_path = args.output_dir / "parts_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    class_names = sorted(manifest.keys())
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    src = args.output_dir / "synthetic_dataset"
    dst = args.output_dir / DATASET_DIR
    to_lines = frame_to_yolo_seg_lines if args.task == "segment" else frame_to_yolo_box_lines
    # Start clean - images/labels left over from a previous, larger run would
    # otherwise stay in train/val alongside this run's.
    shutil.rmtree(dst, ignore_errors=True)
    make_dataset_dirs(dst)

    # Stage 2 writes flat files directly in src, one set per frame:
    # rgb_NNNN.png, mask_NNNN.png and mask_NNNN.json. Each frame's mask and
    # mapping are looked up by its frame number, rather than zipping three
    # separately sorted file lists, which would silently misalign frames if
    # any single file were missing.
    rgb_files = sorted(src.glob("rgb_*.png"))
    if not rgb_files:
        raise FileNotFoundError(f"No rgb_*.png frames in {src} - run generate_dataset.py first (stage 2).")

    total = len(rgb_files)
    seen_classes = set()
    rng = np.random.default_rng(config.SEED)
    stats = {name: [] for name in class_names}  # per class: pixel area of each labelled instance

    for i, img_path in enumerate(rgb_files):
        frame = img_path.stem.removeprefix("rgb_")
        mask_path = src / f"mask_{frame}.png"
        mapping_path = src / f"mask_{frame}.json"
        missing = [p.name for p in (mask_path, mapping_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Frame {frame} is missing {missing} in {src}")

        split = split_for_index(i, total, config.VAL_SPLIT)

        img = Image.open(img_path)
        w, h = img.size
        # Validation images stay clean, so val metrics stay comparable between runs.
        out = camera_effects(img, rng) if split == "train" else img.convert("RGB")
        out.save(dst / "images" / split / img_path.name)

        instance_ids = load_instance_ids(mask_path)
        instance_class_map = load_instance_class_map(mapping_path)
        seen_classes.update(instance_class_map.values())

        lines = to_lines(instance_ids, instance_class_map, class_to_idx, w, h)
        for instance_id, name in instance_class_map.items():
            area = int((instance_ids == instance_id).sum())
            if area and name in stats:
                stats[name].append(area)

        label_txt = dst / "labels" / split / (img_path.stem + ".txt")
        label_txt.write_text("\n".join(lines))

    unknown = seen_classes - set(class_to_idx)
    if unknown:
        raise ValueError(f"Found class(es) {unknown} not in {manifest_path} - re-run stage 1/2 if parts changed.")

    write_data_yaml(dst, class_names, args.task)

    n_val = num_val(total, config.VAL_SPLIT)
    print(f"Done. {total - n_val} train / {n_val} val images written to {dst}")
    print(f"Classes: {class_names}, labels: {'outlines' if args.task == 'segment' else 'boxes'}")
    # A quick health check: a class that is rare or mostly tiny will train poorly.
    print(f"\n{'class':<20}{'instances':>10}{'median px':>11}{'< 20x20':>9}")
    warnings = []
    for name, areas in stats.items():
        a = np.array(areas or [0])
        tiny = np.mean(a < 400)
        print(f"{name:<20}{len(areas):>10}{np.median(a):>11.0f}{tiny:>9.0%}")
        if len(areas) < 0.1 * total:
            warnings.append(f"{name}: only {len(areas)} instances - raise num_frames or instances_per_class.")
        if tiny > 0.25:
            warnings.append(f"{name}: {tiny:.0%} of instances are under 20x20 px - raise camera_target_pixels.")
    for w in warnings:
        print(f"WARNING: {w}")


if __name__ == "__main__":
    main()
