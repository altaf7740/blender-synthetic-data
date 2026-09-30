"""
Web UI for the pipeline: drop STEP files in the browser, name a class for
each, choose segmentation or detection labels, and get the YOLO dataset
back as a zip.

Each submitted job gets its own folder under --jobs-dir with the usual input
layout (<class>/<file>.step) and workspace, and runs stages 1-3 plus the
preview through main.py - the same code path as the command line. Jobs run
one at a time, in submission order: rendering already uses the whole GPU.

Job state lives in memory, so restarting the server forgets the job list
(finished zips stay on disk under --jobs-dir).

Run from the repo root:
    uv run python src/server.py [--host 127.0.0.1] [--port 8000] [--jobs-dir jobs]

It has no authentication - keep it on localhost or a trusted network.
"""

import argparse
import queue
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from synth_pipeline.utils.yolo_dataset import DATASET_DIR, TASKS

HERE = Path(__file__).resolve().parent
WEB_DIR = HERE / "web"
MAIN = HERE / "main.py"
PREVIEW = HERE / "preview_dataset.py"

CAD_SUFFIXES = {".step", ".stp", ".zip"}
CLASS_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,63}$")
MAX_FRAMES = 20000

# (stage key, what the UI calls it) - in pipeline order.
STAGES = [("convert", "Convert"), ("generate", "Render"), ("dataset", "Label"), ("package", "Package")]


@dataclass
class Job:
    id: str
    classes: list
    num_frames: int
    task: str
    created: float = field(default_factory=time.time)
    status: str = "queued"  # queued | running | done | failed | cancelled
    stage: str = ""
    frames_done: int = 0
    render_started: float = 0.0
    finished: float = 0.0
    error: str = ""

    def public(self) -> dict:
        data = asdict(self)
        data["eta_seconds"] = self.eta()
        data["has_preview"] = (self.dir / "workspace" / "preview" / "all.png").exists()
        return data

    def eta(self):
        if self.stage != "generate" or self.frames_done < 2:
            return None
        per_frame = (time.time() - self.render_started) / self.frames_done
        return round(per_frame * (self.num_frames - self.frames_done))

    @property
    def dir(self) -> Path:
        return JOBS_DIR / self.id

    @property
    def zip_path(self) -> Path:
        return self.dir / "dataset.zip"


JOBS_DIR: Path = Path("jobs")
jobs: dict = {}
pending: queue.Queue = queue.Queue()
current_process: dict = {}  # job id -> running subprocess, for cancelling

app = FastAPI(title="Synthetic dataset generator")


def _log_tail(log_path: Path, lines: int = 25) -> str:
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    # tqdm redraws its bar with \r; keep only the last state of each line.
    cleaned = [line.split("\r")[-1] for line in text.splitlines()]
    return "\n".join(line for line in cleaned[-lines:] if line.strip())


def _kill_tree(proc: subprocess.Popen) -> None:
    """Stop a step and every process it started (main.py runs each stage as a child)."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        os.killpg(proc.pid, signal.SIGTERM)


def _run(job: Job, cmd: list, log_path: Path) -> None:
    """Run one pipeline step, tracking render progress. Raises on failure or cancel."""
    with open(log_path, "ab") as log:
        # Own process group, so cancelling also stops the stage main.py started.
        group = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32" else {"start_new_session": True}
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, **group)
        current_process[job.id] = proc
        frames_dir = job.dir / "workspace" / "synthetic_dataset"
        try:
            while proc.poll() is None:
                if job.stage == "generate" and frames_dir.is_dir():
                    job.frames_done = sum(1 for _ in frames_dir.glob("mask_*.json"))
                time.sleep(1.0)
        finally:
            current_process.pop(job.id, None)
    if job.status == "cancelled":
        raise InterruptedError
    if proc.returncode != 0:
        raise RuntimeError(_log_tail(log_path) or f"exit code {proc.returncode}")


def _process(job: Job) -> None:
    workspace = job.dir / "workspace"
    log_path = job.dir / "log.txt"
    base = [sys.executable, str(MAIN), "--input-dir", str(job.dir / "input"), "--output-dir", str(workspace)]
    job.status = "running"
    try:
        job.stage = "convert"
        _run(job, base + ["--stage", "convert"], log_path)
        job.stage = "generate"
        job.render_started = time.time()
        _run(job, base + ["--stage", "generate", "--num-frames", str(job.num_frames)], log_path)
        job.frames_done = job.num_frames
        job.stage = "dataset"
        _run(job, base + ["--stage", "dataset", "--task", job.task], log_path)
        job.stage = "package"
        _run(job, [sys.executable, str(PREVIEW), "--output-dir", str(workspace)], log_path)
        archive = shutil.make_archive(str(job.zip_path.with_suffix("")), "zip", workspace, DATASET_DIR)
        Path(archive).replace(job.zip_path)
        job.status = "done"
    except InterruptedError:
        pass  # status already "cancelled"
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
    job.finished = time.time()


def _worker() -> None:
    while True:
        job = pending.get()
        if job.status == "queued":  # skip jobs cancelled while waiting
            _process(job)


# ---- API ---------------------------------------------------------------------


@app.post("/api/jobs")
async def create_job(
    files: list[UploadFile] = File(...),
    classes: list[str] = Form(...),
    num_frames: int = Form(...),
    task: str = Form("segment"),
):
    if task not in TASKS:
        raise HTTPException(400, f"Task must be one of: {', '.join(TASKS)}.")
    if len(files) != len(classes):
        raise HTTPException(400, "Each file needs exactly one class name.")
    if not 1 <= num_frames <= MAX_FRAMES:
        raise HTTPException(400, f"Frames must be between 1 and {MAX_FRAMES}.")
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

    job = Job(id=uuid.uuid4().hex[:10], classes=list(classes), num_frames=num_frames, task=task)
    for f, name in zip(files, classes):
        class_dir = job.dir / "input" / name
        class_dir.mkdir(parents=True)
        # Keep only the extension from the client's file name.
        with open(class_dir / f"{name}{Path(f.filename).suffix.lower()}", "wb") as out:
            shutil.copyfileobj(f.file, out)
    jobs[job.id] = job
    pending.put(job)
    return job.public()


@app.get("/api/jobs")
def list_jobs():
    return [job.public() for job in sorted(jobs.values(), key=lambda j: j.created, reverse=True)]


def _get(job_id: str) -> Job:
    if job_id not in jobs:
        raise HTTPException(404, "No job with that ID. It may be from before the server restarted.")
    return jobs[job_id]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return _get(job_id).public()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = _get(job_id)
    if job.status in ("queued", "running"):
        job.status = "cancelled"
        if proc := current_process.get(job_id):
            _kill_tree(proc)
    return job.public()


@app.get("/api/jobs/{job_id}/preview")
def job_preview(job_id: str):
    path = _get(job_id).dir / "workspace" / "preview" / "all.png"
    if not path.exists():
        raise HTTPException(404, "The preview is made after labelling finishes.")
    return FileResponse(path, media_type="image/png")


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str):
    job = _get(job_id)
    if job.status != "done":
        raise HTTPException(409, "The dataset isn't ready yet.")
    return FileResponse(job.zip_path, media_type="application/zip", filename=f"dataset-{job.id}.zip")


@app.get("/api/stages")
def stages():
    return STAGES


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


def main():
    global JOBS_DIR
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--jobs-dir", type=Path, default=Path("jobs"), help="Where uploads and results are kept.")
    args = parser.parse_args()
    JOBS_DIR = args.jobs_dir.resolve()
    JOBS_DIR.mkdir(parents=True, exist_ok=True)

    threading.Thread(target=_worker, daemon=True).start()
    print(f"Open http://{args.host}:{args.port} in your browser.")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
