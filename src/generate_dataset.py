"""
Stage 2: render the synthetic dataset with instance segmentation masks for
every part converted in stage 1 (see convert_assets.py / parts_manifest.json
under --output-dir), using Blender (the `bpy` package).

Each frame:
  1. randomizes appearance - ground texture/tiling/tint, environment backdrop,
     lights, part and clutter materials, how many of each part (0-N) and
     which clutter are present;
  2. drops the parts and clutter onto the ground and lets rigid-body physics
     settle them, so they rest in natural poses (on their side, flat, leaning
     on each other, in piles) instead of arbitrary, half-buried angles;
  3. frames one random part at a random apparent size, lens and viewing
     angle, and renders twice from the same poses: a photoreal Cycles image,
     then a flat Workbench image where each part instance is painted with its
     own ID color - an exact instance mask.
Ranges live in pyproject.toml's [tool.synth-pipeline.*] tables.

Per frame, writes to <output-dir>/synthetic_dataset/:
  rgb_NNNN.png    the rendered image
  mask_NNNN.png   instance IDs: id = R + 256 * G, 0 = not a part
  mask_NNNN.json  {"<id>": "<class name>"} for each instance in the frame

Clears <output-dir>/synthetic_dataset first, so leftovers from a previous,
longer run never mix into this one.

Run from the repo root:
    uv run python src/generate_dataset.py --output-dir <folder>
        [--num-frames N] [--textures-dir <folder of photos>] [--seed S]
"""

import argparse
import contextlib
import ctypes
import ctypes.util
import json
import math
import os
import shutil
import sys
from pathlib import Path

import bpy  # before bmesh/mathutils: importing bpy is what makes them available
import bmesh
import numpy as np
from mathutils import Matrix, Vector
from tqdm import tqdm

from synth_pipeline import config
from synth_pipeline.utils.textures import find_images, generate_procedural_textures


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, type=Path, help="Same one passed to convert_assets.py.")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=config.NUM_FRAMES,
        help=f"Frames to render (default from pyproject.toml: {config.NUM_FRAMES}). Use a small value to smoke-test.",
    )
    parser.add_argument(
        "--textures-dir",
        type=Path,
        help="Optional folder of photos (workbenches, floors, tables...) mixed into the ground and backdrop "
        "textures alongside the built-in procedural set.",
    )
    parser.add_argument("--seed", type=int, default=config.SEED, help=f"Random seed (default: {config.SEED}).")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()

    manifest_path = args.output_dir / "parts_manifest.json"
    if not manifest_path.exists():
        parser.error(f"{manifest_path} not found - run convert_assets.py first (stage 1).")
    args.manifest = json.loads(manifest_path.read_text())
    if not args.manifest:
        parser.error(f"{manifest_path} is empty - no parts were converted in stage 1.")

    args.user_textures = []
    if args.textures_dir:
        args.textures_dir = args.textures_dir.resolve()
        if not args.textures_dir.is_dir():
            parser.error(f"--textures-dir not found: {args.textures_dir}")
        args.user_textures = find_images(args.textures_dir)
        if not args.user_textures:
            parser.error(f"--textures-dir has no images: {args.textures_dir}")
    return args


# Material families as (count, base color min/max RGB, metallic range,
# roughness range). The count sets each family's share of the pool. Channels
# are sampled independently, so every family keeps its color range narrow; one
# wide range produces saturated pastels, not finishes a fastener has.
PART_FINISHES = [
    (10, ((0.60, 0.60, 0.62), (0.75, 0.75, 0.78)), (0.85, 1.0), (0.15, 0.5)),  # steel / zinc plated
    (6, ((0.02, 0.02, 0.02), (0.10, 0.10, 0.10)), (0.4, 0.9), (0.3, 0.7)),  # black oxide
    (5, ((0.70, 0.50, 0.15), (0.90, 0.75, 0.40)), (0.8, 1.0), (0.2, 0.5)),  # brass / yellow zinc
    (3, ((0.80, 0.80, 0.78), (0.95, 0.95, 0.92)), (0.0, 0.1), (0.3, 0.7)),  # white nylon / plastic
    (3, ((0.02, 0.02, 0.02), (0.08, 0.08, 0.08)), (0.0, 0.1), (0.3, 0.8)),  # black nylon / plastic
    (4, ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), (0.0, 1.0), (0.1, 0.9)),  # anything - keeps some wildness
]
# Clutter includes metallic greys too, so "shiny grey" alone never means "part".
DISTRACTOR_FINISHES = [
    (20, ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), (0.0, 0.3), (0.2, 1.0)),
    (8, ((0.45, 0.45, 0.45), (0.8, 0.8, 0.8)), (0.8, 1.0), (0.15, 0.6)),
]
# (mesh operator, rigid-body shape). Every primitive fits a unit cube, so its
# bounding sphere has radius sqrt(3)/2 at scale 1.
DISTRACTOR_SHAPES = [
    (lambda: bpy.ops.mesh.primitive_cube_add(size=1), "BOX"),
    (lambda: bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5), "SPHERE"),
    (lambda: bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=1), "CYLINDER"),
    (lambda: bpy.ops.mesh.primitive_cone_add(radius1=0.5, depth=1), "CONE"),
]
FPS = 24
# C runtime, to flush Blender's own (C-level) stdout buffer - see quiet_stdout().
_LIBC = ctypes.cdll.ucrtbase if sys.platform == "win32" else ctypes.CDLL(ctypes.util.find_library("c"))


@contextlib.contextmanager
def quiet_stdout():
    """Silence Blender's per-frame C-level chatter ("Fra:...", "bake: frame...")
    so the progress bar (on stderr) stays readable. Errors still go to stderr."""
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    try:
        yield
    finally:
        _LIBC.fflush(None)  # or Blender's buffered output surfaces later, after stdout is restored
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


def _load_merged_mesh(class_name: str, glb_path: str) -> bpy.types.Mesh:
    """Import a GLB and merge all its meshes into one, in world space (meters).

    Vertices are welded so smooth shading works across the per-face splits the
    tessellator leaves, and materials are dropped (each frame assigns its own).
    """
    before = set(bpy.data.objects)
    with quiet_stdout():
        bpy.ops.import_scene.gltf(filepath=glb_path)
    imported = [o for o in bpy.data.objects if o not in before]
    bpy.context.view_layer.update()

    bm = bmesh.new()
    for obj in imported:
        if obj.type == "MESH":
            tmp = obj.data.copy()
            tmp.transform(obj.matrix_world)
            bm.from_mesh(tmp)  # appends
            bpy.data.meshes.remove(tmp)
    for obj in imported:
        bpy.data.objects.remove(obj)
    if not bm.verts:
        raise ValueError(f"{glb_path} contains no mesh geometry")
    size = max(Vector(np.ptp([v.co for v in bm.verts], axis=0)))
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=size * 1e-5)

    mesh = bpy.data.meshes.new(f"Mesh_{class_name}")
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.clear()
    mesh.materials.append(None)  # one slot; each object fills it (link="OBJECT")
    mesh.polygons.foreach_set("material_index", [0] * len(mesh.polygons))
    mesh.shade_smooth()
    mesh.set_sharp_from_angle(angle=math.radians(30))
    return mesh


def _center_radius(mesh: bpy.types.Mesh) -> tuple:
    co = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    lo, hi = co.min(axis=0), co.max(axis=0)
    return (lo + hi) / 2, float(np.linalg.norm(hi - lo) / 2)


def _new_object(name: str, data=None):
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _add_rigid_body(obj, shape: str, passive: bool = False) -> None:
    with bpy.context.temp_override(object=obj, active_object=obj, selected_objects=[obj]):
        bpy.ops.rigidbody.object_add(type="PASSIVE" if passive else "ACTIVE")
    rb = obj.rigid_body
    rb.collision_shape = shape
    rb.friction = 0.6
    rb.restitution = 0.0
    rb.use_margin = True
    rb.collision_margin = 0.005
    rb.linear_damping = 0.1
    rb.angular_damping = 0.4


def _use_gpu(scene) -> str:
    """Render Cycles on the first GPU backend available, else the CPU."""
    if config.DEVICE.lower() != "cpu":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for backend in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
            try:
                prefs.compute_device_type = backend
            except TypeError:  # backend not built into this platform's bpy
                continue
            prefs.refresh_devices()
            gpus = [d for d in prefs.devices if d.type == backend]
            if gpus:
                for d in prefs.devices:
                    d.use = d.type == backend
                scene.cycles.device = "GPU"
                return f"{backend} ({', '.join(d.name for d in gpus)})"
    scene.cycles.device = "CPU"
    return "CPU"


class Scene:
    def __init__(self, manifest: dict, ground_textures: list, env_textures: list, rng):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        self.scene = scene = bpy.context.scene
        self.ground_textures = [bpy.data.images.load(p, check_existing=True) for p in ground_textures]
        self.env_textures = [bpy.data.images.load(p, check_existing=True) for p in env_textures]

        # Parts are merged, centered on their bounding box (CAD exports often put
        # geometry far from the file's origin, which would make parts rotate into
        # the ground), then scaled by 1/R.
        meshes = {name: _load_merged_mesh(name, path) for name, path in manifest.items()}
        radii_m = {}
        for name, mesh in meshes.items():
            center, radii_m[name] = _center_radius(mesh)
            mesh.transform(Matrix.Translation(-Vector(center)))

        # Scene scale R: every length setting is a multiple of it, so the same
        # settings fit 5 mm screws and 500 mm brackets. The scene is built in
        # units of R (the median part is ~1 unit), where Bullet physics is most
        # stable; standard gravity is fine since resting poses don't depend on scale.
        R_m = float(np.median(list(radii_m.values())))
        print(f"Scene scale: median part radius {R_m * 1000:.3g} mm = 1 scene unit")
        for mesh in meshes.values():
            mesh.transform(Matrix.Scale(1 / R_m, 4))

        # Enough copies of every class for the most crowded frame; each frame
        # uses a random number of them and parks the rest. Copies share mesh data.
        copies = config.INSTANCES_PER_CLASS[1]
        self.parts, self.part_class, self.part_radii = [], [], []
        for name, mesh in meshes.items():
            for _ in range(copies):
                obj = _new_object(f"Part_{len(self.parts)}_{name}", mesh)
                obj.material_slots[0].link = "OBJECT"
                self.parts.append(obj)
                self.part_class.append(name)
                self.part_radii.append(radii_m[name] / R_m)
        self.classes = list(manifest)
        self.spawn_gap = 0.12  # clearance between spawned bodies, and above the ground
        self.park_x = 1000.0  # absent bodies wait here, far out of view

        self._build_ground()
        self._build_world()
        self._build_lights()
        self.camera = _new_object("Camera", bpy.data.cameras.new("Camera"))
        self.camera.data.sensor_fit = "HORIZONTAL"
        self.camera.data.clip_start, self.camera.data.clip_end = 0.01, 1e4
        scene.camera = self.camera

        self.distractors, self.distractor_shapes = [], []
        for add, shape in DISTRACTOR_SHAPES:
            for i in range(config.DISTRACTORS_PER_SHAPE):
                add()
                obj = bpy.context.object
                obj.name = f"Distractor_{shape.lower()}_{i}"
                obj.data.materials.append(None)
                obj.material_slots[0].link = "OBJECT"
                self.distractors.append(obj)
                self.distractor_shapes.append(shape)

        bpy.ops.rigidbody.world_add()
        rbw = scene.rigidbody_world
        rbw.substeps_per_frame = max(1, round(config.PHYSICS_STEPS_PER_SECOND / FPS))
        rbw.solver_iterations = 20
        scene.render.fps = FPS
        self.settle_frames = max(2, round(config.SETTLE_SECONDS * FPS))
        rbw.point_cache.frame_start, rbw.point_cache.frame_end = 1, self.settle_frames
        _add_rigid_body(self.collider, "BOX", passive=True)
        for obj in self.parts:
            _add_rigid_body(obj, config.PART_COLLISION_SHAPE)
        for obj, shape in zip(self.distractors, self.distractor_shapes):
            _add_rigid_body(obj, shape)

        self.part_materials = self._material_pool(PART_FINISHES, rng, "PartMaterial")
        self.distractor_materials = self._material_pool(DISTRACTOR_FINISHES, rng, "DistractorMaterial")
        self._setup_render()

    def _build_ground(self) -> None:
        """Visible ground plane, plus a thick, invisible collision slab: a
        zero-thickness plane lets thin, fast parts tunnel through it."""
        bpy.ops.mesh.primitive_plane_add(size=config.GROUND_SIZE)
        self.ground = bpy.context.object
        self.ground.name = "Ground"
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, -3))
        self.collider = bpy.context.object
        self.collider.name = "GroundCollider"
        self.collider.scale = (4 * self.park_x, 4 * self.park_x, 6)
        self.collider.hide_render = True

        mat = bpy.data.materials.new("GroundMaterial")
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        bsdf = nodes["Principled BSDF"]
        coords = nodes.new("ShaderNodeTexCoord")
        self.ground_mapping = nodes.new("ShaderNodeMapping")
        self.ground_image = nodes.new("ShaderNodeTexImage")
        self.ground_tint = nodes.new("ShaderNodeMix")
        self.ground_tint.data_type = "RGBA"
        self.ground_tint.blend_type = "MULTIPLY"
        self.ground_tint.inputs["Factor"].default_value = 1.0
        links.new(coords.outputs["UV"], self.ground_mapping.inputs["Vector"])
        links.new(self.ground_mapping.outputs["Vector"], self.ground_image.inputs["Vector"])
        links.new(self.ground_image.outputs["Color"], self.ground_tint.inputs["A"])
        links.new(self.ground_tint.outputs["Result"], bsdf.inputs["Base Color"])
        self.ground_bsdf = bsdf
        self.ground.data.materials.append(mat)

    def _build_world(self) -> None:
        """Textured environment: the backdrop in low-angle shots, and what metal parts reflect."""
        world = bpy.data.worlds.new("World")
        self.scene.world = world
        world.color = (0, 0, 0)  # what the Workbench mask pass sees as background
        nodes, links = world.node_tree.nodes, world.node_tree.links
        coords = nodes.new("ShaderNodeTexCoord")
        self.world_mapping = nodes.new("ShaderNodeMapping")
        self.world_image = nodes.new("ShaderNodeTexEnvironment")
        self.world_background = nodes["Background"]
        links.new(coords.outputs["Generated"], self.world_mapping.inputs["Vector"])
        links.new(self.world_mapping.outputs["Vector"], self.world_image.inputs["Vector"])
        links.new(self.world_image.outputs["Color"], self.world_background.inputs["Color"])

    def _build_lights(self) -> None:
        self.point_light = _new_object("PointLight", bpy.data.lights.new("PointLight", "POINT"))
        self.point_light.data.shadow_soft_size = config.POINT_LIGHT_RADIUS
        self.sun = _new_object("Sun", bpy.data.lights.new("Sun", "SUN"))
        self.sun.data.angle = math.radians(1.0)  # crisp shadows

    @staticmethod
    def _material_pool(finishes, rng, name: str) -> list:
        pool = []
        for count, color, metallic, roughness in finishes:
            for _ in range(count):
                mat = bpy.data.materials.new(f"{name}_{len(pool)}")
                bsdf = mat.node_tree.nodes["Principled BSDF"]
                bsdf.inputs["Base Color"].default_value = (*rng.uniform(*color), 1.0)
                bsdf.inputs["Metallic"].default_value = float(rng.uniform(*metallic))
                bsdf.inputs["Roughness"].default_value = float(rng.uniform(*roughness))
                pool.append(mat)
        return pool

    def _setup_render(self) -> None:
        scene = self.scene
        scene.render.resolution_x, scene.render.resolution_y = config.RESOLUTION
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.image_settings.color_depth = "8"
        scene.cycles.samples = config.SAMPLES
        scene.cycles.use_denoising = True
        self.device = _use_gpu(scene)

        # Mask pass: flat, unlit object colors, no anti-aliasing, dithering or
        # color transform - so every pixel is exactly one instance's ID color.
        shading = scene.display.shading
        shading.light = "FLAT"
        shading.color_type = "OBJECT"
        shading.show_shadows = shading.show_cavity = shading.show_specular_highlight = False
        shading.show_object_outline = False
        scene.display.render_aa = "OFF"
        scene.render.dither_intensity = 0.0

    # ---- per frame ---------------------------------------------------------

    def randomize_appearance(self, rng) -> tuple:
        """Everything except body placement. Returns (visible parts, visible distractors)."""
        repeats = config.GROUND_SIZE / rng.uniform(*config.GROUND_TEXTURE_TILE)  # both in units of R
        self.ground_image.image = self.ground_textures[rng.integers(len(self.ground_textures))]
        self.ground_mapping.inputs["Scale"].default_value = (repeats, repeats, 1.0)
        self.ground_mapping.inputs["Rotation"].default_value = (0.0, 0.0, float(rng.uniform(0, 2 * math.pi)))
        # Brightness plus a slight warm/cool cast - not a per-channel tint, which turns grey surfaces pink or green.
        tint = rng.uniform(0.7, 1.0) * (1.0 + rng.uniform(-0.05, 0.05, 3))
        self.ground_tint.inputs["B"].default_value = (*tint, 1.0)
        self.ground_bsdf.inputs["Roughness"].default_value = float(rng.uniform(0.3, 1.0))

        self.world_image.image = self.env_textures[rng.integers(len(self.env_textures))]
        self.world_background.inputs["Strength"].default_value = float(rng.uniform(*config.WORLD_STRENGTH_RANGE))
        self.world_mapping.inputs["Rotation"].default_value = (0.0, 0.0, float(rng.uniform(0, 2 * math.pi)))

        lo, hi = config.POINT_LIGHT_POSITION_RANGE
        self.point_light.location = rng.uniform(lo, hi)
        self.point_light.data.energy = float(rng.uniform(*config.POINT_LIGHT_POWER_RANGE))
        self.point_light.data.color = rng.uniform((0.8, 0.75, 0.65), (1.0, 1.0, 1.0))

        sun_on = rng.random() < config.SUN_PROBABILITY
        self.sun.data.energy = float(rng.uniform(*config.SUN_STRENGTH_RANGE)) if sun_on else 0.0
        tilt = math.radians(90.0 - rng.uniform(*config.SUN_ELEVATION_DEG))  # the sun shines along its -Z
        self.sun.rotation_euler = (tilt, 0.0, float(rng.uniform(0, 2 * math.pi)))

        for bodies, pool in ((self.parts, self.part_materials), (self.distractors, self.distractor_materials)):
            for obj, i in zip(bodies, rng.integers(0, len(pool), len(bodies))):
                obj.material_slots[0].material = pool[i]

        counts = {name: int(rng.integers(config.INSTANCES_PER_CLASS[0], config.INSTANCES_PER_CLASS[1] + 1)) for name in self.classes}
        if rng.random() < config.EMPTY_FRAME_PROBABILITY:
            counts = dict.fromkeys(counts, 0)
        part_on = np.zeros(len(self.parts), bool)
        for i, name in enumerate(self.part_class):
            if counts[name] > 0:
                part_on[i], counts[name] = True, counts[name] - 1
        dist_on = rng.random(len(self.distractors)) < config.DISTRACTOR_VISIBLE_PROBABILITY
        return part_on, dist_on

    def drop_and_settle(self, rng, part_on, dist_on) -> int:
        """Drop visible bodies onto the ground, let physics settle them, apply the result.

        Returns how many visible parts fell through the ground (should be 0).
        """
        sizes = rng.uniform(*config.DISTRACTOR_SIZE, len(self.distractors))
        dist_radii = list(sizes * 0.87)  # a unit primitive fits in a sphere of radius sqrt(3)/2

        bodies = self.parts + self.distractors
        radii = self.part_radii + dist_radii
        visible = list(part_on) + list(dist_on)
        scales = [1.0] * len(self.parts) + list(sizes)
        # Spread grows with the number of parts dropped, so crowding (and how
        # often parts pile up) doesn't depend on how many classes there are.
        part_spread = rng.uniform(*config.PART_SPREAD) * np.sqrt(max(int(np.sum(part_on)), 1))
        spreads = [part_spread] * len(self.parts) + [config.DISTRACTOR_SPREAD] * len(self.distractors)
        gap = self.spawn_gap

        self.scene.frame_set(1)
        placed, stack = [], 0.0
        for i, (obj, r, on, spread, s) in enumerate(zip(bodies, radii, visible, spreads, scales)):
            obj.scale = (s, s, s)
            # Absent bodies are parked far away, static and hidden from both renders.
            obj.hide_render = not on
            obj.rigid_body.enabled = bool(on)
            if not on:
                obj.location = (self.park_x + 20 * i, 0.0, r + gap)
                continue
            for _ in range(100):  # non-overlapping spot; bodies that start interpenetrating get launched
                xy = rng.uniform(-spread, spread, 2)
                if all(np.hypot(*(xy - q)) > r + rq + gap for q, rq in placed):
                    z = r + gap + rng.uniform(0, r)
                    break
            else:  # too crowded: drop it from above the others instead
                stack += 2 * r + gap
                z = r + gap + stack
            placed.append((xy, r))
            obj.location = (float(xy[0]), float(xy[1]), float(z))
            obj.rotation_euler = rng.uniform(0, 2 * math.pi, 3)
        bpy.context.view_layer.update()

        # Bake the drop, read the settled poses at the last frame, then free the
        # bake and write those poses back at frame 1, where rigid bodies simply
        # take their object transforms - so both renders see the settled scene.
        cache = self.scene.rigidbody_world.point_cache
        with quiet_stdout(), bpy.context.temp_override(scene=self.scene, point_cache=cache):
            bpy.ops.ptcache.free_bake()
            bpy.ops.ptcache.bake(bake=True)
        self.scene.frame_set(self.settle_frames)
        final = [obj.matrix_world.copy() for obj in bodies]
        with bpy.context.temp_override(scene=self.scene, point_cache=cache):
            bpy.ops.ptcache.free_bake()
        self.scene.frame_set(1)
        for obj, m in zip(bodies, final):
            obj.matrix_world = m
        bpy.context.view_layer.update()

        # What the camera can frame: visible parts, else visible clutter, with their bounding radii.
        centers = [tuple(m.translation) for m in final]
        on_parts = [(c, r) for c, r, on in zip(centers, self.part_radii, part_on) if on]
        on_clutter = [(c, r) for c, r, on in zip(centers[len(self.parts) :], dist_radii, dist_on) if on]
        self.framing_candidates = on_parts or on_clutter or [((0.0, 0.0, 0.0), 1.0)]
        return sum(1 for (c, r) in on_parts if c[2] < -r)

    def place_camera(self, rng) -> None:
        """Frame one random visible part at a random apparent size.

        Picking the distance from a target size in pixels, instead of a fixed
        distance range, keeps small parts (a 5 mm nut) from being only a few
        pixels in most frames while large parts fill the view.
        """
        cam = self.camera.data
        focal = float(rng.uniform(*config.CAMERA_FOCAL_LENGTH))
        focal_px = config.RESOLUTION[0] * focal / cam.sensor_width
        center, radius = self.framing_candidates[rng.integers(len(self.framing_candidates))]
        size_px = rng.uniform(*config.CAMERA_TARGET_PIXELS)
        dist = max(focal_px * 2 * radius / size_px, 3 * radius)

        # Shift the aim point so the framed part lands off-center, but stays in frame.
        half_view = dist * cam.sensor_width / (2 * focal)
        shift = rng.uniform(-1, 1, 2) * config.CAMERA_FRAME_OFFSET * half_view
        target = Vector(center) + Vector((shift[0], shift[1], 0.0))

        elev = math.radians(rng.uniform(*config.CAMERA_ELEVATION_DEG))
        azim = rng.uniform(0.0, 2 * math.pi)
        offset = Vector((math.cos(elev) * math.cos(azim), math.cos(elev) * math.sin(azim), math.sin(elev)))
        cam.lens = focal
        self.camera.location = target + offset * dist
        self.camera.rotation_euler = (-offset).to_track_quat("-Z", "Y").to_euler()

    def render(self, rgb_path: Path, mask_path: Path) -> dict:
        """Render the RGB image and the instance mask. Returns {id: class} for the mask."""
        scene = self.scene
        scene.render.engine = "CYCLES"
        scene.view_settings.view_transform = "AgX"
        scene.render.filepath = str(rgb_path)
        with quiet_stdout():
            bpy.ops.render.render(write_still=True)

        # Part instance k gets ID color (k % 256, k // 256, 0); everything else is black.
        instances = {}
        for obj in self.ground, *self.distractors:
            obj.color = (0.0, 0.0, 0.0, 1.0)
        for obj, name in zip(self.parts, self.part_class):
            if not obj.hide_render:
                k = len(instances) + 1
                instances[k] = name
                obj.color = ((k % 256) / 255, (k // 256) / 255, 0.0, 1.0)
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.view_settings.view_transform = "Raw"
        scene.render.filepath = str(mask_path)
        with quiet_stdout():
            bpy.ops.render.render(write_still=True)
        return instances


def render(manifest: dict, dataset_dir: Path, ground_textures: list, env_textures: list, num_frames: int, seed: int):
    rng = np.random.default_rng(seed)
    dataset_dir.mkdir(parents=True)
    scene = Scene(manifest, ground_textures, env_textures, rng)
    print(f"Rendering on {scene.device}")

    fell_through = 0
    with tqdm(total=num_frames, desc="Rendering", unit="frame", dynamic_ncols=True) as bar:
        for i in range(num_frames):
            part_on, dist_on = scene.randomize_appearance(rng)
            fell_through += scene.drop_and_settle(rng, part_on, dist_on)
            scene.place_camera(rng)
            instances = scene.render(dataset_dir / f"rgb_{i:04d}.png", dataset_dir / f"mask_{i:04d}.png")
            (dataset_dir / f"mask_{i:04d}.json").write_text(json.dumps(instances))
            bar.update()
            if fell_through:
                bar.set_postfix(fell_through=fell_through)

    if fell_through:
        print(f"WARNING: {fell_through} part placement(s) fell through the ground and are missing from their frames.")


def main() -> None:
    args = parse_args()
    dataset_dir = args.output_dir / "synthetic_dataset"
    shutil.rmtree(dataset_dir, ignore_errors=True)

    builtin_ground, builtin_env = generate_procedural_textures(args.output_dir / "textures", args.seed)
    ground_textures = builtin_ground + args.user_textures
    env_textures = builtin_env + args.user_textures
    print(
        f"Textures: {len(builtin_ground)} built-in ground, {len(builtin_env)} built-in environment, "
        f"{len(args.user_textures)} from --textures-dir"
    )

    render(args.manifest, dataset_dir, ground_textures, env_textures, args.num_frames, args.seed)
    print(f"\nRendered {args.num_frames} frames for {sorted(args.manifest)} to {dataset_dir}")


if __name__ == "__main__":
    main()
