"""
Web UI for the pipeline: drop STEP files in the browser, name a class for
each, choose segmentation, detection or classification, then browse the
generated images and download the dataset.

The UI keeps exactly one dataset. Generating a new one cancels any running
job and deletes the previous dataset first. It lives in <jobs-dir>/current/
(the usual input layout, a workspace, the zip and state.json), so it is still
there after a server restart. Stages 1-3 run through main.py - the same code
path as the command line.

Run from the repo root:
    uv run python src/server.py [--host 127.0.0.1] [--port 8000] [--jobs-dir jobs]

It has no authentication - keep it on localhost or a trusted network.
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from synth_pipeline.utils.yolo_dataset import DATASET_DIR, TASKS

HERE = Path(__file__).resolve().parent
WEB_DIR = HERE / "web"
MAIN = HERE / "main.py"

CAD_SUFFIXES = {".step", ".stp", ".zip"}
CLASS_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,63}$")
MAX_FRAMES = 20000
THUMB_PX = 320
PALETTE = [(60, 60, 255), (60, 220, 60), (255, 140, 40), (40, 220, 255), (220, 60, 220), (255, 255, 60)]  # BGR, per class


def rendered_frames(frames_dir: Path) -> list:
    """Stage 2's finished frames, in order. A frame's mask_NNNN.json is written
    last, so an rgb image without one may still be being written."""
    return [p for p in sorted(frames_dir.glob("rgb_*.png")) if (frames_dir / f"mask_{p.stem[4:]}.json").exists()]


@dataclass
class Job:
    classes: list
    num_frames: int
    task: str
    created: float = field(default_factory=time.time)
    status: str = "running"  # running | done | failed | cancelled
    stage: str = ""
    frames_done: int = 0
    render_started: float = 0.0
    error: str = ""

    def public(self) -> dict:
        data = asdict(self)
        data["eta_seconds"] = self.eta()
        data["zip_bytes"] = zip_path().stat().st_size if self.status == "done" and zip_path().exists() else None
        return data

    def eta(self):
        if self.status != "running" or self.stage != "generate" or self.frames_done < 2:
            return None
        per_frame = (time.time() - self.render_started) / self.frames_done
        return round(per_frame * (self.num_frames - self.frames_done))

    def save(self) -> None:
        (current_dir() / "state.json").write_text(json.dumps(asdict(self)))


JOBS_DIR: Path = Path("jobs")  # set from --jobs-dir


def current_dir() -> Path:
    return JOBS_DIR / "current"


def workspace_dir() -> Path:
    return current_dir() / "workspace"


def zip_path() -> Path:
    return current_dir() / "dataset.zip"


job: Job | None = None
proc: subprocess.Popen | None = None  # the step running now, for cancelling
worker: threading.Thread | None = None
lock = threading.Lock()  # one replace at a time

app = FastAPI(title="Blender Synthetic Data")


def _log_tail(log_path: Path, lines: int = 25) -> str:
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    # tqdm redraws its bar with \r; keep only the last state of each line.
    cleaned = [line.split("\r")[-1] for line in text.splitlines()]
    return "\n".join(line for line in cleaned[-lines:] if line.strip())


def _kill_tree(p: subprocess.Popen) -> None:
    """Stop a step and every process it started (main.py runs each stage as a child)."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
    else:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def _run(j: Job, cmd: list, log_path: Path) -> None:
    """Run one pipeline step, tracking render progress. Raises on failure or cancel."""
    global proc
    with open(log_path, "ab") as log:
        # Own process group, so cancelling also stops the stage main.py started.
        group = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32" else {"start_new_session": True}
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, **group)
        frames_dir = workspace_dir() / "synthetic_dataset"
        try:
            while proc.poll() is None:
                if j.stage == "generate" and frames_dir.is_dir():
                    j.frames_done = len(rendered_frames(frames_dir))
                    j.save()
                time.sleep(1.0)
        finally:
            returncode, proc = proc.returncode, None
    if j.status == "cancelled":
        raise InterruptedError
    if returncode != 0:
        raise RuntimeError(_log_tail(log_path) or f"exit code {returncode}")


def _process(j: Job) -> None:
    log_path = current_dir() / "log.txt"
    base = [sys.executable, str(MAIN), "--input-dir", str(current_dir() / "input"), "--output-dir", str(workspace_dir())]
    try:
        for stage, extra in (
            ("convert", []),
            ("generate", ["--num-frames", str(j.num_frames)]),
            ("dataset", ["--task", j.task]),
        ):
            j.stage = stage
            if stage == "generate":
                j.render_started = time.time()
            j.save()
            _run(j, base + ["--stage", stage] + extra, log_path)
        j.frames_done = j.num_frames
        j.stage = "package"
        j.save()
        archive = shutil.make_archive(str(zip_path().with_suffix("")), "zip", workspace_dir(), DATASET_DIR)
        Path(archive).replace(zip_path())
        j.status = "done"
    except InterruptedError:
        pass  # status already "cancelled"
    except Exception as e:
        j.status = "failed"
        j.error = str(e)
    j.save()


# ---- images --------------------------------------------------------------------
#
# The panel shows the renders while stage 2 runs ("renders"), then the finished
# dataset's images with their labels drawn on, or its crops ("dataset"). Images
# are addressed by position in a sorted listing, never by a client-given path.

_listing: dict = {}


def _images(kind: str) -> list:
    """[(image path, label path or class name or None), ...] for `kind`."""
    key = (kind, job.created, job.status, job.frames_done) if job else None
    if _listing.get("key") != key:
        items = []
        if job and kind == "renders":
            items = [(p, None) for p in rendered_frames(workspace_dir() / "synthetic_dataset")]
        elif job and kind == "dataset" and job.status == "done":
            root = workspace_dir() / DATASET_DIR
            if job.task == "classify":
                items = [(p, p.parent.name) for p in sorted(root.glob("*/*/*.jpg"), key=lambda p: (p.name, str(p)))]
            else:
                images = sorted(root.glob("images/*/*.png"), key=lambda p: p.name)
                items = [(p, root / "labels" / p.parent.name / f"{p.stem}.txt") for p in images]
        _listing.update(key=key, items=items)
    return _listing["items"]


def _draw_labels(img: np.ndarray, label_path: Path) -> np.ndarray:
    """Outlines or boxes, filled lightly in each class's color."""
    if not label_path.exists():
        return img
    h, w = img.shape[:2]
    shapes = []
    for line in label_path.read_text().splitlines():
        values = line.split()
        if len(values) == 5:  # class cx cy w h
            cx, cy, bw, bh = map(float, values[1:])
            pts = [(cx - bw / 2, cy - bh / 2), (cx + bw / 2, cy - bh / 2), (cx + bw / 2, cy + bh / 2), (cx - bw / 2, cy + bh / 2)]
        elif len(values) >= 7:  # class x1 y1 x2 y2 ...
            pts = np.array(values[1:], dtype=np.float32).reshape(-1, 2)
        else:
            continue
        shapes.append((int(values[0]), (np.array(pts, dtype=np.float32) * [w, h]).astype(np.int32)))
    overlay = img.copy()
    for cls, pts in shapes:
        cv2.fillPoly(overlay, [pts], PALETTE[cls % len(PALETTE)])
    img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
    for cls, pts in shapes:
        cv2.polylines(img, [pts], True, PALETTE[cls % len(PALETTE)], max(1, w // 320), cv2.LINE_AA)
    return img


def _image_response(kind: str, index: int, thumb: bool) -> Response:
    if kind not in ("renders", "dataset"):
        raise HTTPException(404, "No such image.")
    items = _images(kind)
    if not 0 <= index < len(items):
        raise HTTPException(404, "No such image.")
    cached = current_dir() / "thumbs" / f"{kind}-{index}.jpg"
    if thumb and cached.exists():
        return FileResponse(cached, media_type="image/jpeg")
    path, label = items[index]
    img = cv2.imread(str(path))
    if img is None:
        raise HTTPException(404, "That image isn't readable yet.")
    if isinstance(label, Path):
        img = _draw_labels(img, label)
    if thumb:
        scale = THUMB_PX / max(img.shape[:2])
        if scale < 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82 if thumb else 92])
    if thumb:
        cached.parent.mkdir(exist_ok=True)
        cached.write_bytes(buf.tobytes())
    return Response(buf.tobytes(), media_type="image/jpeg")


# ---- API ---------------------------------------------------------------------


@app.post("/api/dataset")
async def create_dataset(
    files: list[UploadFile] = File(...),
    classes: list[str] = Form(...),
    num_frames: int = Form(...),
    task: str = Form("segment"),
):
    global job, worker
    if task not in TASKS:
        raise HTTPException(400, f"Task must be one of: {', '.join(TASKS)}.")
    if len(files) != len(classes):
        raise HTTPException(400, "Each file needs exactly one class name.")
    if not 1 <= num_frames <= MAX_FRAMES:
        raise HTTPException(400, f"Images must be between 1 and {MAX_FRAMES}.")
    classes = [" ".join(name.split()) for name in classes]
    for name in classes:
        if not CLASS_NAME.match(name):
            raise HTTPException(
                400,
                f"Class name '{name}' can only use letters, digits, spaces, - and _, and must start with a letter or digit.",
            )
    if len(set(classes)) != len(classes):
        raise HTTPException(400, "Two files have the same class name. Give each part its own name.")
    for f in files:
        if Path(f.filename or "").suffix.lower() not in CAD_SUFFIXES:
            raise HTTPException(400, f"{f.filename} isn't a STEP file. Use .step, .stp, or a .zip containing one.")

    with lock:
        # Replace: stop the running job, wait for it to wind down, delete the old dataset.
        if job and job.status == "running":
            job.status = "cancelled"
            if proc:
                _kill_tree(proc)
        if worker:
            worker.join(timeout=30)
            if worker.is_alive():
                raise HTTPException(409, "The previous dataset is still stopping. Try again in a moment.")
        shutil.rmtree(current_dir(), ignore_errors=True)

        new = Job(classes=list(classes), num_frames=num_frames, task=task)
        for f, name in zip(files, classes):
            class_dir = current_dir() / "input" / name
            class_dir.mkdir(parents=True)
            # Keep only the extension from the client's file name.
            with open(class_dir / f"{name}{Path(f.filename).suffix.lower()}", "wb") as out:
                shutil.copyfileobj(f.file, out)
        new.save()
        job = new
        worker = threading.Thread(target=_process, args=(new,), daemon=True)
        worker.start()
    return job.public()


@app.get("/api/dataset")
def get_dataset():
    return job.public() if job else {"status": "none"}


@app.post("/api/dataset/cancel")
def cancel_dataset():
    if job and job.status == "running":
        job.status = "cancelled"
        if proc:
            _kill_tree(proc)
    return get_dataset()


@app.get("/api/dataset/images")
def list_images():
    """What the panel shows: the dataset once finished, else the renders so far."""
    if not job:
        return {"kind": "renders", "count": 0, "labels": None}
    kind = "dataset" if job.status == "done" else "renders"
    items = _images(kind)
    labels = [label for _, label in items] if job.task == "classify" and kind == "dataset" else None
    return {"kind": kind, "count": len(items), "labels": labels}


@app.get("/api/dataset/thumb/{kind}/{index}")
def thumbnail(kind: str, index: int):
    return _image_response(kind, index, thumb=True)


@app.get("/api/dataset/image/{kind}/{index}")
def full_image(kind: str, index: int):
    return _image_response(kind, index, thumb=False)


@app.get("/api/dataset/download")
def download():
    if not job or job.status != "done" or not zip_path().exists():
        raise HTTPException(404, "There's no finished dataset to download.")
    return FileResponse(zip_path(), media_type="application/zip", filename="dataset.zip")


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


def main():
    global JOBS_DIR, job
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--jobs-dir", type=Path, default=Path("jobs"), help="Where the current dataset is kept.")
    args = parser.parse_args()
    JOBS_DIR = args.jobs_dir.resolve()
    JOBS_DIR.mkdir(parents=True, exist_ok=True)

    # Pick the current dataset back up; one that was mid-way when the server stopped can't resume.
    state = current_dir() / "state.json"
    if state.exists():
        job = Job(**json.loads(state.read_text()))
        if job.status == "running":
            job.status, job.error = "failed", "The server stopped before this dataset finished."
            job.save()

    print(f"Open http://{args.host}:{args.port} in your browser.")
    # Jobs run in their own process group (so Cancel can stop them), which
    # also means they'd outlive the server - stop them with it. Ctrl+C lands
    # in `finally`; SIGTERM needs a handler, since uvicorn re-raises it after
    # shutting down, which would otherwise end the process on the spot.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        if proc:
            _kill_tree(proc)


if __name__ == "__main__":
    main()
