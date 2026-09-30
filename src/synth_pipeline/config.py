"""
Loads tool-level settings from pyproject.toml's [tool.synth-pipeline] table
and exposes them as typed Python objects (tuples) for every stage script to
import.

Only tool-level settings live here - per-run paths (input CAD folder, output
workspace) are passed explicitly via --input-dir/--output-dir on each stage
script's command line instead, since this is a general-purpose tool meant to
run against any folder of parts, not one fixed project.
"""

import tomllib
from pathlib import Path

# config.py -> synth_pipeline/ -> src/ -> repo root
_PYPROJECT_PATH = Path(__file__).resolve().parents[2] / "pyproject.toml"
with open(_PYPROJECT_PATH, "rb") as f:
    _raw = tomllib.load(f)["tool"]["synth-pipeline"]


def _range(key: str) -> tuple:
    """[min, max] -> (min, max); nested [[x,y,z], [x,y,z]] -> ((x,y,z), (x,y,z))."""
    lo, hi = _r[key]
    return (tuple(lo), tuple(hi)) if isinstance(lo, list) else (lo, hi)


CONVERSION_TOLERANCE = _raw["conversion"]["tolerance"]

NUM_FRAMES = _raw["rendering"]["num_frames"]
RESOLUTION = tuple(_raw["rendering"]["resolution"])
SEED = _raw["rendering"]["seed"]
SAMPLES = _raw["rendering"]["samples"]
DEVICE = _raw["rendering"]["device"]

SETTLE_SECONDS = _raw["physics"]["settle_seconds"]
PHYSICS_STEPS_PER_SECOND = _raw["physics"]["steps_per_second"]
PART_COLLISION_SHAPE = _raw["physics"]["part_collision_shape"]

VAL_SPLIT = _raw["dataset"]["val_split"]

_r = _raw["randomization"]
INSTANCES_PER_CLASS = _range("instances_per_class")
PART_SPREAD = _range("part_spread")
EMPTY_FRAME_PROBABILITY = _r["empty_frame_probability"]

CAMERA_TARGET_PIXELS = _range("camera_target_pixels")
CAMERA_FOCAL_LENGTH = _range("camera_focal_length")
CAMERA_ELEVATION_DEG = _range("camera_elevation_deg")
CAMERA_FRAME_OFFSET = _r["camera_frame_offset"]

DISTRACTORS_PER_SHAPE = _r["distractors_per_shape"]
DISTRACTOR_SIZE = _range("distractor_size")
DISTRACTOR_SPREAD = _r["distractor_spread"]
DISTRACTOR_VISIBLE_PROBABILITY = _r["distractor_visible_probability"]

GROUND_SIZE = _r["ground_size"]
GROUND_TEXTURE_TILE = _range("ground_texture_tile")

WORLD_STRENGTH_RANGE = _range("world_strength")
POINT_LIGHT_POSITION_RANGE = _range("point_light_position")
POINT_LIGHT_RADIUS = _r["point_light_radius"]
POINT_LIGHT_POWER_RANGE = _range("point_light_power")
SUN_PROBABILITY = _r["sun_probability"]
SUN_STRENGTH_RANGE = _range("sun_strength")
SUN_ELEVATION_DEG = _range("sun_elevation_deg")
