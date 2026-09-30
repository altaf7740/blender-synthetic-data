"""
Convert stage 2's instance masks into YOLO labels: outline polygons
(segmentation), bounding boxes (detection), or per-part image crops
(classification).

Per frame, generate_dataset.py writes:
  mask_NNNN.png
      RGB image; each pixel's instance ID is R + 256 * G, 0 = not a part
      (ground, backdrop, clutter).
  mask_NNNN.json
      {"<id>": "<class name>"} for every part instance in the frame.
Every instance has its own ID even when two share a class, so each still
becomes its own polygon or box.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics.data.converter import merge_multi_segment


def load_instance_ids(mask_path: Path) -> np.ndarray:
    """mask_NNNN.png -> (H, W) array of instance IDs."""
    rgb = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)[..., ::-1].astype(np.int32)
    return rgb[..., 0] + 256 * rgb[..., 1]


def load_instance_class_map(mapping_path: Path) -> dict:
    """mask_NNNN.json -> {instance id (int): class name}."""
    return {int(k): v for k, v in json.loads(mapping_path.read_text()).items()}


def instance_mask_to_polygon(mask: np.ndarray, min_area: float = 4.0):
    """Trace one binary instance mask into a single polygon.

    YOLO-seg has one polygon per instance and no holes. Tracing only outer
    contours would fill holes (a nut's bore) and emit each piece of an
    occluded part as a separate instance. Instead, every contour - outer
    pieces and holes - is joined into one polygon by zero-width cuts
    between nearest points. Rasterized with the even-odd rule (OpenCV's
    fillPoly, as Ultralytics does), holes stay empty.

    Args:
        mask: 2D boolean/uint8 array, non-zero where the instance is present.
        min_area: Discard contours smaller than this many pixels (noise).

    Returns:
        (N, 2) array of (x, y) pixel coordinates, or None if nothing is left.
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    parts = [c.reshape(-1, 2) for c in contours if len(c) >= 3 and cv2.contourArea(c) >= min_area]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(merge_multi_segment(parts), axis=0)


def frame_to_yolo_seg_lines(
    instance_ids: np.ndarray,
    instance_class_map: dict,
    class_to_idx: dict,
    img_w: int,
    img_h: int,
) -> list:
    """Build YOLO-seg label lines ("class x1 y1 x2 y2 ...", normalized) for one frame.

    Args:
        instance_ids: (H, W) instance-ID image, from load_instance_ids().
        instance_class_map: instance id -> class name, from this same frame's mask_NNNN.json.
        class_to_idx: class name -> fixed YOLO class index.
        img_w, img_h: image dimensions, for normalizing coordinates.

    Returns:
        List of label lines, one per instance.
    """
    lines = []
    for instance_id, class_name in instance_class_map.items():
        if class_name not in class_to_idx:
            continue
        mask = instance_ids == instance_id
        if not mask.any():  # fully hidden behind something else
            continue
        poly = instance_mask_to_polygon(mask)
        if poly is None:
            continue
        norm = poly.astype(np.float64)
        norm[:, 0] /= img_w
        norm[:, 1] /= img_h
        coords = " ".join(f"{v:.6f}" for v in norm.flatten())
        lines.append(f"{class_to_idx[class_name]} {coords}")
    return lines


def frame_to_yolo_box_lines(
    instance_ids: np.ndarray,
    instance_class_map: dict,
    class_to_idx: dict,
    img_w: int,
    img_h: int,
    min_area: int = 4,
) -> list:
    """Build YOLO detection label lines ("class cx cy w h", normalized) for one frame.

    Boxes are tight around each instance's visible pixels, so a partly hidden
    part gets a box around what the camera sees - the same region its
    segmentation outline covers. Arguments as in frame_to_yolo_seg_lines().
    """
    lines = []
    for instance_id, class_name in instance_class_map.items():
        if class_name not in class_to_idx:
            continue
        ys, xs = np.nonzero(instance_ids == instance_id)
        if len(xs) < min_area:  # fully hidden, or a few stray pixels
            continue
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        box = ((x0 + x1) / 2 / img_w, (y0 + y1) / 2 / img_h, (x1 - x0) / img_w, (y1 - y0) / img_h)
        lines.append(f"{class_to_idx[class_name]} " + " ".join(f"{v:.6f}" for v in box))
    return lines


def instance_crops(
    image: np.ndarray,
    instance_ids: np.ndarray,
    instance_class_map: dict,
    min_side: int = 32,
    margin: float = 0.3,
    min_purity: float = 0.6,
) -> list:
    """Cut a square crop around each visible part, for classification.

    Each crop is centered on the part's visible pixels, with `margin` extra
    around them for context, and shifted to stay inside the image. Crops are
    skipped when the part is smaller than `min_side` px, or when other parts
    make up so much of the crop that the part is no longer what it shows
    (under `min_purity` of the part pixels in it are this part's).

    Returns:
        [(class name, (S, S, 3) crop of image), ...]
    """
    img_h, img_w = instance_ids.shape
    crops = []
    for instance_id, class_name in instance_class_map.items():
        ys, xs = np.nonzero(instance_ids == instance_id)
        if len(xs) == 0:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        if max(x1 - x0, y1 - y0) < min_side:
            continue
        side = min(int(max(x1 - x0, y1 - y0) * (1 + margin)), img_w, img_h)
        left = int(np.clip(round((x0 + x1 - side) / 2), 0, img_w - side))
        top = int(np.clip(round((y0 + y1 - side) / 2), 0, img_h - side))
        window = instance_ids[top : top + side, left : left + side]
        if (window == instance_id).sum() < min_purity * (window > 0).sum():
            continue
        crops.append((class_name, image[top : top + side, left : left + side]))
    return crops
