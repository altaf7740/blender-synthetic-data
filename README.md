# Synthetic Data Pipeline (Blender)

Generate an instance-segmentation, object-detection or classification
training dataset —
and train a model on it — from CAD parts alone. No real photos required.
Point it at a folder of STEP files; it converts each to a mesh, renders
thousands of domain-randomized frames in Blender with per-pixel instance
masks, builds a YOLO dataset (outline labels, box labels or part crops), and trains a
YOLO26 model.

This is the Blender counterpart of `omniverse-synthetic-data`: the same
stages, settings and outputs, with no NVIDIA Omniverse/Isaac Sim. Blender
comes in as the `bpy` package from PyPI, so **everything runs in one uv
venv** — no separate Blender install, no second Python interpreter — on
macOS, Windows and Linux.

## How it works

```
<input folder>/<class_name>/*.step
        │
        ▼
 1. src/convert_assets.py     STEP → GLB (OpenCascade, via cascadio)
        │
        ▼
 2. src/generate_dataset.py   GLB → 2000 rendered frames + instance masks
        │                     (Blender: Cycles + Bullet physics, domain randomization)
        ▼
 3. src/build_yolo_dataset.py masks → YOLO labels (outlines, boxes or crops), train/val split
        │
        ▼
 4. src/train.py              YOLO26n-seg / YOLO26n / YOLO26n-cls training (Ultralytics)
```

### What each rendered frame varies

| | |
|---|---|
| **Placement** | Parts and clutter are dropped onto the surface and settled with rigid-body physics, so they rest in natural poses (on their side, flat, leaning on each other) |
| **Composition** | 0–3 copies of each part per frame, dropped close together (piles, overlaps) or far apart (scattered); 3% of frames have no parts at all |
| **Surface** | 48 built-in procedural textures, mostly realistic (wood, brushed metal, concrete, fabric, tiles, cardboard, speckled laminate, rubber mat, anti-static mat, paper, plain) plus a small share of loud colors; random tiling, rotation and brightness; plus your own photos via `--textures-dir` |
| **Backdrop** | Textured environment (studio gradients, mildly tinted rooms and skies), rotated per frame; visible in low-angle shots, and reflected by metal parts |
| **Part finishes** | Steel/zinc, black oxide, brass/yellow zinc, white and black nylon, plus a share of random colors |
| **Clutter** | Unlabelled cubes, spheres, cylinders and cones with random finishes, including metallic greys |
| **Camera** | Frames one random part at a random apparent size (50–400 px), so small parts are seen up close as often as large ones; random lens (18–60 mm), elevation (20–88°), azimuth and off-center aim |
| **Lights** | Environment strength; point light position, power and warm/neutral tint; a sun with crisp shadows in half the frames |
| **Camera effects** | Stage 3 adds blur, motion blur, sensor noise and JPEG compression to a share of the training images (validation images stay clean) |

Every range is in `pyproject.toml`'s `[tool.synth-pipeline.*]` tables, and a
fixed `seed` makes runs reproducible.

### How the masks are made

Each frame is rendered twice from the same settled scene:

1. **Cycles** (path traced, denoised) → `rgb_NNNN.png`
2. **Workbench** with flat, unlit object colors, no anti-aliasing, no
   dithering and the `Raw` view transform → `mask_NNNN.png`. Every part
   instance is painted with its own ID color (`id = R + 256 * G`); ground,
   backdrop and clutter are black. `mask_NNNN.json` maps each ID to its
   class.

So the mask is pixel-exact, not an estimate: occlusion by other parts or
clutter falls out of the render for free.

### Works for any part set without retuning

Nothing is tuned to the example fasteners. Stage 2 measures your parts and
scales the scene to them:

- **Size:** the scene is built in units of R, the parts' median bounding
  radius, and every length setting (spacing, clutter size, ground, texture
  scale, light placement) is a multiple of it. So 3 mm components and
  500 mm brackets get the same kind of scene, and the same light powers give
  the same exposure. The camera frames each part by its own size.
- **Units:** the GLB files are always in meters, whatever unit the CAD was
  authored in (mm, inch, m).
- **Class count:** spacing grows with the number of parts dropped, so 2
  classes or 20 give similar crowding. Classes come from the input folders.

Stage 2 prints the scale it measured, e.g.
`Scene scale: median part radius 8.64 mm = 1 scene unit`.

Physics runs in those R units with standard gravity. At true scale (a
5 mm nut under 9.81 m/s²), Bullet lets small parts sink into the ground;
resting poses don't depend on scale, so nothing is lost by scaling up.

Parts are merged into one mesh each and centered on their bounding box
when loaded. CAD exports often put geometry far from the file's origin,
which would otherwise make parts land out of frame or rotate into the
ground.

If any stage fails, the pipeline stops there with a nonzero exit code, so
`make` and CI can see the failure. Stage 1 also fails if even one class
can't be converted, rather than quietly leaving that class out of the
dataset.

## Requirements

- macOS (Apple Silicon), Windows or Linux (x86-64)
- A GPU is recommended for rendering: Cycles uses the first one it finds
  (OptiX/CUDA on NVIDIA, HIP on AMD, Metal on Apple, oneAPI on Intel) and
  falls back to the CPU
- [uv](https://docs.astral.sh/uv/) — `brew install uv`, `pip install uv`, or
  `winget install astral-sh.uv`

Blender itself does not need to be installed. uv fetches Python 3.13 if you
don't have it; that's the version the `bpy` wheel is built for.

## Setup

```
uv sync
```

(or `make setup`)

This creates `.venv/` and installs everything from `pyproject.toml`,
including Blender (`bpy`, ~400 MB). On Windows and Linux it installs a
**CUDA build of torch**, not the default CPU-only PyPI wheel. That routing
is declared in `pyproject.toml`'s `[tool.uv.sources]` / `[[tool.uv.index]]`
and applies only off macOS, where the regular wheel already supports Apple
GPUs (MPS).

## Usage

Prepare an input folder — one subfolder per class, each holding a `.step`
file (or a `.zip` that contains one):

```
my_parts/
├── bolt/
│   └── bolt.step
└── nut/
    └── nut.zip        # gets extracted in place automatically
```

Run everything:

```
uv run python src/main.py --input-dir my_parts --output-dir workspace
```

Or run one stage at a time (useful while iterating):

```
uv run python src/main.py --input-dir my_parts --output-dir workspace --stage convert
uv run python src/main.py --input-dir my_parts --output-dir workspace --stage generate
uv run python src/main.py --input-dir my_parts --output-dir workspace --from dataset
```

### Segmentation, detection or classification

Labels are outlines for instance segmentation by default. `--task` picks
another kind of dataset:

```
uv run python src/main.py --input-dir my_parts --output-dir workspace --task detect
make all TASK=detect
make dataset TASK=classify      # re-label frames you already rendered
```

| `--task` | Dataset | Training starts from |
|---|---|---|
| `segment` (default) | Frames + one outline polygon per part | `yolo26n-seg.pt` |
| `detect` | Frames + one box per part, tight around its visible pixels | `yolo26n.pt` |
| `classify` | One folder of cropped parts per class: `train/<class>/`, `val/<class>/` | `yolo26n-cls.pt` (224 px) |

Rendering is the same for all three, so you can re-run just the dataset
stage to switch. For classification, each visible part is cut out as a
square crop with some surroundings for context. Parts under 32 px, and
crops where other parts make up most of the part pixels, are skipped;
stage 3 prints how many crops each class got. Crops from validation frames
go to `val/`, so no frame feeds both splits.

The task is recorded in `data.yaml`, and `train.py` and `make preview`
follow it (override the model with `--model`).

Launch `main.py` via `uv run python`, not a bare `python`: each stage runs
with the interpreter that ran `main.py` (`sys.executable`), so that's what
puts them in the venv where `bpy`, opencv and ultralytics are installed.

`--input-dir`/`--output-dir` can be relative (to wherever you ran the
command from) or absolute.

`workspace/` (or whatever `--output-dir` you choose) ends up with:

```
workspace/
├── meshes/<class>.glb
├── parts_manifest.json      # class -> GLB path, written by stage 1
├── textures/                # built-in textures generated by stage 2
├── synthetic_dataset/       # rgb_NNNN.png, mask_NNNN.png, mask_NNNN.json
├── yolo_dataset/            # images/ + labels/ (or train/<class>/ + val/<class>/), data.yaml
├── preview/                 # review sheets (make preview)
└── runs/<task>/             # Ultralytics training run, weights/best.pt
```

Pretrained weights are downloaded once into `~/.cache/synth-pipeline/`,
outside the repo.

### Web UI

For a drag-and-drop version that anyone can use without the command line:

```
make ui                     # or: uv run python src/server.py --port 8000
```

Then open http://127.0.0.1:8000. Drop one STEP file per part and type each
one's **class name**, e.g. `hex nut M5` (letters, digits, spaces, `-` and
`_`). Pick the task (**Segmentation**, **Detection** or **Classification**)
and how many images to render, then **Generate dataset**.

The right panel shows the images as they render, then the finished dataset
with its labels drawn on (or, for classification, each crop with its class).
Click an image to see it full size; the arrow keys step through them. The
grid only builds the tiles in view, so it stays fast with thousands of
images. **Download** gives the `yolo_dataset/` folder as a zip; its
`data.yaml` uses relative paths, so it trains from wherever it's unzipped.

The UI keeps **one dataset at a time**. Generating a new one asks first,
then cancels anything still running and deletes the previous dataset. It's
kept in `jobs/current/` (with its state in `state.json`), so it's still
there after the server restarts. The server has no login: it listens on
localhost by default, and should only be opened to a trusted network
(`--host 0.0.0.0`).

### Training on a GPU

Training picks the fastest device it finds: an NVIDIA GPU (CUDA), else an
Apple GPU (MPS), else the CPU. It prints which one at the start. To choose
yourself, pass `--device`:

```
uv run python src/train.py --output-dir workspace --device cpu   # or 0, 1, ... (CUDA GPU), mps
```

Rendering does the same: Cycles uses the first GPU it finds (OptiX/CUDA on
NVIDIA, HIP on AMD, Metal on Apple, oneAPI on Intel), else the CPU.

### Using your own surface photos

Photos of real workbenches, floors, tables or mats narrow the gap to real
images more than anything procedural. Put them in a folder and pass it in;
they're mixed into both the ground and backdrop textures:

```
make generate TEXTURES_DIR=my_photos
```

### Reviewing a dataset

After stage 3, render review sheets with the labels drawn on exactly as
training will read them:

```
make preview                # 5 images per class
make preview PER_CLASS=10
```

This writes `preview/<class>.png` (one sheet per class),
`preview/<class>/` (full-size images) and `preview/all.png`. The sheet takes
the first N frames containing each class, so one frame can appear in more
than one class's row.

Stages 2 and 3 clear their own output folder before writing, so re-running
them never mixes in frames left over from an earlier run.

A ready-to-use example input folder (4 fastener parts) is included at
`examples/fasteners/` — `uv run python src/main.py --input-dir
examples/fasteners --output-dir workspace` runs end to end against it.

### Smoke-testing a new part set

Render a handful of frames and look at them before committing to a full run:

```
make convert INPUT_DIR=my_parts
make generate NUM_FRAMES=20
make dataset
make preview
```

## Known limitations

- **Sim-to-real gap is unmeasured.** Training only validates against
  synthetic data from the same distribution. Collect real photos of your
  parts and run `model.predict()` against them before trusting deployment
  accuracy; widen `pyproject.toml`'s randomization ranges if it's weak.
- **Part finishes assume mechanical parts** (metals, black oxide, plastics).
  For other kinds of objects, edit `PART_FINISHES` in `generate_dataset.py`.
- **Parts collide as their convex hull** by default, which is right for
  fasteners but lets a concave part (an L-bracket, a channel) rest as if it
  were filled in. Set `part_collision_shape = "MESH"` in `pyproject.toml`
  for those; it is slower.
- **Very mixed sizes in one set** (say 3 mm and 500 mm parts together) work,
  but clutter and spacing follow the median part, so the extremes look less
  natural. Check with `make preview`.
- **Procedural textures are a stand-in for real ones.** They vary the
  surface a lot but don't look like real materials up close; photos passed
  via `--textures-dir` are the better source when you have them.
- **Rendering is ~2 s per frame** at 640×640 and 64 samples on an M2 Pro
  GPU, so a 2000-frame dataset takes a bit over an hour. Lower `samples` for
  speed, raise it for less noise. The first frame on a new machine takes a
  minute or more longer while Cycles compiles its GPU kernels (cached
  afterwards).
- **Weight downloads from GitHub can be flaky.** Ultralytics retries on its
  own; once downloaded, weights are reused from `~/.cache/synth-pipeline/`.
