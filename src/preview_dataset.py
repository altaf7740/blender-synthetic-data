"""
Visual review of a built dataset: for each class, picks N images containing
it and draws the YOLO labels (outlines or boxes) over them exactly as
training will read them. For a classification dataset it shows N of each
class's crops instead. The class being reviewed is filled and outlined thickly; other labelled
parts are outlined thinly.

Writes to <output-dir>/preview/:
  <class>/<frame>.png   full-size annotated images
  <class>.png           contact sheet of that class's images
  all.png               every class sheet stacked, one row per class

Run from the repo root (after build_yolo_dataset.py):
    uv run python src/preview_dataset.py --output-dir <folder> [--per-class N]
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from synth_pipeline.utils.yolo_dataset import DATASET_DIR, read_task

_TILE = 384  # px, contact-sheet tile size
_PALETTE = [(60, 60, 255), (60, 220, 60), (255, 140, 40), (40, 220, 255), (220, 60, 220), (255, 255, 60)]  # BGR


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, type=Path, help="Same one passed to earlier stages.")
    parser.add_argument("--per-class", type=int, default=5, help="Images to show per class (default: 5).")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    return args


def load_labels(dataset_dir: Path) -> list:
    """[(image_path, [(class_idx, polygon_px), ...]), ...] over train and val."""
    samples = []
    for split in ("train", "val"):
        for label_path in sorted((dataset_dir / "labels" / split).glob("*.txt")):
            image_path = dataset_dir / "images" / split / f"{label_path.stem}.png"
            polys = []
            for line in label_path.read_text().splitlines():
                values = line.split()
                if len(values) == 5:  # detection: class cx cy w h -> the box's 4 corners
                    cx, cy, w, h = map(float, values[1:])
                    corners = [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)]
                    polys.append((int(values[0]), np.array(corners, dtype=np.float32)))
                elif len(values) >= 7:  # segmentation: class + at least 3 points
                    polys.append((int(values[0]), np.array(values[1:], dtype=np.float32).reshape(-1, 2)))
            samples.append((image_path, polys))
    return samples


def annotate(image: np.ndarray, polys: list, focus_class: int, class_names: list) -> np.ndarray:
    h, w = image.shape[:2]
    overlay = image.copy()
    for cls, poly in polys:
        pts = (poly * [w, h]).astype(np.int32)
        if cls == focus_class:
            cv2.fillPoly(overlay, [pts], _PALETTE[cls % len(_PALETTE)])
    out = cv2.addWeighted(overlay, 0.35, image, 0.65, 0)
    for cls, poly in polys:
        pts = (poly * [w, h]).astype(np.int32)
        color = _PALETTE[cls % len(_PALETTE)]
        cv2.polylines(out, [pts], True, color, 2 if cls == focus_class else 1, cv2.LINE_AA)
        if cls == focus_class:
            x, y = pts.min(axis=0)
            cv2.putText(out, class_names[cls], (int(x), max(14, int(y) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def contact_sheet(images: list, title: str, count: int) -> np.ndarray:
    tiles = [cv2.resize(img, (_TILE, _TILE), interpolation=cv2.INTER_AREA) for img in images]
    tiles += [np.full((_TILE, _TILE, 3), 40, np.uint8)] * (count - len(tiles))  # pad short rows
    sheet = np.hstack(tiles)
    header = np.full((32, sheet.shape[1], 3), 25, np.uint8)
    cv2.putText(header, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([header, sheet])


def main():
    args = parse_args()
    class_names = sorted(json.loads((args.output_dir / "parts_manifest.json").read_text()))
    dataset_dir = args.output_dir / DATASET_DIR
    if not (dataset_dir / "data.yaml").exists():
        raise FileNotFoundError(f"No dataset in {dataset_dir} - run build_yolo_dataset.py first (stage 3).")
    classify = read_task(dataset_dir) == "classify"
    samples = [] if classify else load_labels(dataset_dir)

    preview_dir = args.output_dir / "preview"
    shutil.rmtree(preview_dir, ignore_errors=True)
    preview_dir.mkdir(parents=True)

    sheets = []
    for idx, name in enumerate(class_names):
        if classify:  # the crops themselves are what training sees
            crops = sorted((dataset_dir / "train" / name).glob("*.jpg")) + sorted((dataset_dir / "val" / name).glob("*.jpg"))
            chosen = [(p, None) for p in crops[: args.per_class]]
        else:
            chosen = [(p, polys) for p, polys in samples if any(c == idx for c, _ in polys)][: args.per_class]
        (preview_dir / name).mkdir()
        annotated = []
        for image_path, polys in chosen:
            img = cv2.imread(str(image_path))
            if not classify:
                img = annotate(img, polys, idx, class_names)
            cv2.imwrite(str(preview_dir / name / image_path.name), img)
            annotated.append(img)
        title = f"{name}: {len(chosen)} of {args.per_class} requested"
        sheet = contact_sheet(annotated, title, args.per_class)
        cv2.imwrite(str(preview_dir / f"{name}.png"), sheet)
        sheets.append(sheet)
        note = "" if len(chosen) == args.per_class else "  <- not enough images containing this class; render more frames"
        print(f"{name}: {len(chosen)} images{note}")

    cv2.imwrite(str(preview_dir / "all.png"), np.vstack(sheets))
    print(f"\nPreview written to {preview_dir}")


if __name__ == "__main__":
    main()
