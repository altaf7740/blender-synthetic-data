"""
Convert a STEP/IGES CAD file to a GLB mesh via OpenCascade (cascadio), so
Blender can import it. GLB is always in meters, whatever unit the CAD file
was authored in.
"""

from pathlib import Path

import cascadio

from synth_pipeline import config


def convert_step_to_glb(step_path: Path, glb_path: Path) -> Path:
    """Convert one STEP/IGES file to GLB.

    Raises:
        RuntimeError: If the conversion fails.
    """
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    glb_path.unlink(missing_ok=True)  # so a stale file can't pass for a successful conversion
    status = cascadio.step_to_glb(
        str(step_path), str(glb_path), tol_linear=config.CONVERSION_TOLERANCE, tol_relative=True
    )
    if status != 0 or not glb_path.exists() or glb_path.stat().st_size == 0:
        raise RuntimeError(f"STEP->GLB conversion failed for {step_path} (status {status})")
    return glb_path
