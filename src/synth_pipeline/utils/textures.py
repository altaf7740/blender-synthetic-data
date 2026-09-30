"""
Surface and background textures for domain randomization.

Built-in textures are generated procedurally (numpy + PIL), so rendering
works offline with no asset downloads. Real photos (workbenches, floors, tables...) can be mixed in via
find_images() - they narrow the sim-to-real gap more than anything
procedural can.

Ground textures are tileable (periodic noise), so no seams show where the
renderer repeats them across the ground plane.
"""

from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".hdr", ".exr"}

_GROUND_SIZE = 512  # px; must be divisible by every noise cell count used below
_DOME_SIZE = (1024, 512)  # px, lat-long (equirectangular) layout


def find_images(directory: Path) -> list:
    """All image files under directory (recursive), as sorted string paths."""
    return sorted(str(p) for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


# ---- noise primitives -------------------------------------------------------


def _noise(rng, size: int, cells_x: int, cells_y: int = None) -> np.ndarray:
    """Smooth, tileable value noise in [0, 1], shape (size, size).

    A random grid is wrap-padded before bicubic upsampling, so opposite edges
    match and the texture repeats seamlessly.
    """
    cells_y = cells_y or cells_x
    grid = rng.random((cells_y, cells_x)).astype(np.float32)
    pad = 2
    padded = np.pad(grid, pad, mode="wrap")
    sx, sy = size // cells_x, size // cells_y
    big = Image.fromarray(padded).resize((padded.shape[1] * sx, padded.shape[0] * sy), Image.BICUBIC)
    out = np.asarray(big)[pad * sy : pad * sy + size, pad * sx : pad * sx + size]
    return np.clip(out, 0.0, 1.0)


def _fractal(rng, size: int, base_cells: int = 4, octaves: int = 5) -> np.ndarray:
    total = np.zeros((size, size), np.float32)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        total += amp * _noise(rng, size, base_cells * 2**o)
        norm += amp
        amp *= 0.5
    return total / norm


def _lerp(field: np.ndarray, c0, c1) -> np.ndarray:
    c0, c1 = np.asarray(c0, np.float32), np.asarray(c1, np.float32)
    return c0 + (c1 - c0) * field[..., None]


def _jitter(rng, color, amount: float = 0.08) -> np.ndarray:
    return np.clip(np.asarray(color, np.float32) + rng.uniform(-amount, amount, 3), 0.0, 1.0)


def _grid(size: int):
    t = np.arange(size, dtype=np.float32) / size
    return np.meshgrid(t, t)  # x, y in [0, 1)


# ---- ground surfaces (tileable) --------------------------------------------


def _wood(rng, n):
    x, y = _grid(n)
    warp = _fractal(rng, n, base_cells=2, octaves=3)
    rings = int(rng.integers(16, 48))
    grain = 0.5 + 0.5 * np.sin(2 * np.pi * (rings * x + 0.8 * warp))
    field = 0.4 * grain**3 + 0.35 * _noise(rng, n, 128, 4) + 0.25 * _fractal(rng, n, 2, 3)
    dark = _jitter(rng, (0.33, 0.2, 0.1))
    light = _jitter(rng, (0.62, 0.45, 0.28))
    return _lerp(field, dark, light)


def _brushed_metal(rng, n):
    streaks = _noise(rng, n, 4, 128)  # stretched along x
    field = 0.7 * streaks + 0.3 * _noise(rng, n, 64)
    base = rng.uniform(0.35, 0.75)
    return _lerp(field, _jitter(rng, (base - 0.15,) * 3, 0.03), _jitter(rng, (base + 0.15,) * 3, 0.03))


def _concrete(rng, n):
    field = _fractal(rng, n, base_cells=4, octaves=6)
    pits = (_noise(rng, n, 128) > 0.82).astype(np.float32) * 0.25
    base = rng.uniform(0.35, 0.7)
    return _lerp(np.clip(field - pits, 0, 1), _jitter(rng, (base - 0.2,) * 3, 0.04), _jitter(rng, (base + 0.1,) * 3, 0.04))


def _muted(rng, sat=(0.0, 0.35), val=(0.15, 0.85)) -> np.ndarray:
    """A random color at low saturation - what real benches, mats and floors look like."""
    h, s, v = rng.random(), rng.uniform(*sat), rng.uniform(*val)
    i, f = int(h * 6) % 6, h * 6 - int(h * 6)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    return np.array([(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i], np.float32)


def _fabric(rng, n):
    x, y = _grid(n)
    f = int(rng.choice([32, 64, 128]))
    weave = 0.5 + 0.25 * (np.sin(2 * np.pi * f * x) + np.sin(2 * np.pi * f * y))
    field = 0.7 * weave + 0.3 * _noise(rng, n, 32)
    color = _muted(rng)
    return _lerp(field, color * 0.7, color)


def _tiles(rng, n):
    x, y = _grid(n)
    count = int(rng.choice([2, 4]))
    fx, fy = (x * count) % 1.0, (y * count) % 1.0
    grout_w = rng.uniform(0.01, 0.04)
    grout = (fx < grout_w) | (fy < grout_w)
    color = _muted(rng, sat=(0.0, 0.2), val=(0.4, 0.9))
    per_tile = _noise(rng, n, count) * 0.12  # slight tile-to-tile shade difference, not a checkerboard
    img = color * (0.88 + per_tile[..., None] + 0.08 * _noise(rng, n, 64)[..., None])
    img[grout] = color * rng.uniform(0.4, 0.7)
    return img


def _cardboard(rng, n):
    x, _ = _grid(n)
    corrugation = 0.5 + 0.5 * np.sin(2 * np.pi * int(rng.integers(8, 24)) * x)
    fibers = _noise(rng, n, 128, 16)
    field = 0.15 * corrugation + 0.55 * fibers + 0.3 * _fractal(rng, n, 4, 3)
    return _lerp(field, _jitter(rng, (0.45, 0.33, 0.2)), _jitter(rng, (0.72, 0.58, 0.4)))


def _speckle(rng, n):
    """Speckled laminate / epoxy floor."""
    base = _muted(rng, val=(0.3, 0.85))
    img = _lerp(_noise(rng, n, 16), base * 0.9, base)
    for _ in range(3):
        dots = _noise(rng, n, 128) > rng.uniform(0.8, 0.9)
        img[dots] = base * rng.uniform(0.4, 1.3)
    return img


def _rubber_mat(rng, n):
    x, y = _grid(n)
    count = int(rng.choice([8, 16, 32]))
    fx, fy = (x * count) % 1.0 - 0.5, (y * count) % 1.0 - 0.5
    bumps = (np.hypot(fx, fy) < rng.uniform(0.18, 0.32)).astype(np.float32)
    base = rng.uniform(0.03, 0.2)
    field = 0.6 * bumps + 0.4 * _noise(rng, n, 64)
    return _lerp(field, (base,) * 3, (base + 0.12,) * 3)


def _esd_mat(rng, n):
    """Anti-static work mat: matte green, blue or grey with a faint mottle."""
    color = _jitter(rng, [(0.18, 0.38, 0.28), (0.2, 0.3, 0.45), (0.35, 0.36, 0.38)][int(rng.integers(3))], 0.05)
    return _lerp(0.6 * _fractal(rng, n, 8, 4) + 0.4 * _noise(rng, n, 128), color * 0.85, color * 1.1)


def _paper(rng, n):
    """Paper, whiteboard or light laminate."""
    color = _muted(rng, sat=(0.0, 0.08), val=(0.78, 0.97))
    fibers = 0.5 * _noise(rng, n, 128, 32) + 0.5 * _noise(rng, n, 32, 128)
    return _lerp(0.7 * fibers + 0.3 * _fractal(rng, n, 4, 3), color * 0.93, color)


def _plain(rng, n):
    return _lerp(_fractal(rng, n, 4, 4), (c := _muted(rng)) * 0.85, c)


def _wild(rng, n):
    """Loud, saturated color. A small share keeps the model from keying on "surfaces are muted"."""
    color = rng.uniform(0.05, 0.95, 3)
    return _lerp(_fractal(rng, n, 4, 4), color * 0.85, color)


# One entry per generator; each makes `variants` textures. Realistic surfaces
# dominate; _wild is 1 in 12.
_GROUND_GENERATORS = {
    "wood": _wood,
    "brushed_metal": _brushed_metal,
    "concrete": _concrete,
    "fabric": _fabric,
    "tiles": _tiles,
    "cardboard": _cardboard,
    "speckle": _speckle,
    "rubber_mat": _rubber_mat,
    "esd_mat": _esd_mat,
    "paper": _paper,
    "plain": _plain,
    "wild": _wild,
}


# ---- dome environments (lat-long) -------------------------------------------


def _environment(rng, width: int, height: int) -> np.ndarray:
    """Vertical sky/studio gradient with soft blotches, lat-long layout."""
    v = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]  # 0 = top (zenith)
    if rng.random() < 0.5:  # neutral studio
        top, horizon, bottom = (rng.uniform(0.6, 1.0),) * 3, (rng.uniform(0.4, 0.8),) * 3, (rng.uniform(0.1, 0.4),) * 3
    else:  # room / sky with a mild tint
        top = _muted(rng, sat=(0.0, 0.3), val=(0.5, 1.0))
        horizon = _muted(rng, sat=(0.0, 0.2), val=(0.5, 0.95))
        bottom = _muted(rng, sat=(0.0, 0.3), val=(0.05, 0.5))
    top, horizon, bottom = (np.asarray(c, np.float32) for c in (top, horizon, bottom))
    upper = top + (horizon - top) * np.clip(v / 0.5, 0, 1)[..., None]
    lower = horizon + (bottom - horizon) * np.clip((v - 0.5) / 0.5, 0, 1)[..., None]
    img = np.where((v < 0.5)[..., None], upper, lower) * np.ones((1, width, 1), np.float32)
    blotches = np.asarray(Image.fromarray(_fractal(rng, height, 4, 4)).resize((width, height), Image.BICUBIC))
    return np.clip(img * (0.8 + 0.4 * blotches[..., None]), 0.0, 1.0)


# ---- public API ---------------------------------------------------------------


def _save(img: np.ndarray, path: Path) -> str:
    Image.fromarray((np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)).save(path)
    return str(path)


def generate_procedural_textures(out_dir: Path, seed: int, variants: int = 4, environments: int = 12):
    """Generate built-in ground and environment textures.

    Returns:
        (ground_paths, environment_paths) - lists of written PNG paths.
    """
    rng = np.random.default_rng(seed)
    ground_dir, env_dir = out_dir / "ground", out_dir / "environment"
    ground_dir.mkdir(parents=True, exist_ok=True)
    env_dir.mkdir(parents=True, exist_ok=True)

    ground = [
        _save(gen(rng, _GROUND_SIZE), ground_dir / f"{name}_{i}.png")
        for name, gen in _GROUND_GENERATORS.items()
        for i in range(variants)
    ]
    env = [_save(_environment(rng, *_DOME_SIZE), env_dir / f"environment_{i}.png") for i in range(environments)]
    return ground, env
