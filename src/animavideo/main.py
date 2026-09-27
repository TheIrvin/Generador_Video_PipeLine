from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from fastapi.middleware.cors import CORSMiddleware

from .render import RenderOptions, render_video


ROOT = Path(os.getenv("VIDEO_PIPELINE_DATA_DIR", "data/jobs")).resolve()
ROOT.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = 30 * 1024 * 1024
MAX_MOTION_REFERENCE_BYTES = 100 * 1024 * 1024
_job_lock = threading.Lock()
_render_lock = threading.Lock()
_active_jobs: set[str] = set()

app = FastAPI(
    title="Generador Video Pipeline API",
    version="0.1.0",
    description="Submit an image and animation instructions; poll a job and download its MP4.",
)
cors_origins = [item.strip() for item in os.getenv("VIDEO_PIPELINE_CORS_ORIGINS", "").split(",") if item.strip()]
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key"],
    )


def _authorize(x_api_key: str | None = Header(default=None)) -> None:
    configured_key = os.getenv("VIDEO_PIPELINE_API_KEY")
    if configured_key and not secrets.compare_digest(x_api_key or "", configured_key):
        raise HTTPException(status_code=401, detail="valid X-API-Key header required")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_dir(job_id: str) -> Path:
    if not job_id.isalnum() or len(job_id) != 32:
        raise HTTPException(status_code=404, detail="job not found")
    return ROOT / job_id


def _read_job(job_id: str) -> dict[str, Any]:
    path = _job_dir(job_id) / "job.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="job not found")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_job(job: dict[str, Any]) -> None:
    directory = _job_dir(job["job_id"])
    directory.mkdir(parents=True, exist_ok=True)
    temp_path = directory / "job.json.tmp"
    temp_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(directory / "job.json")


def _adjust_prompt(prompt: str) -> str:
    # The prompt remains part of the trace; a supplied guide video drives visible movement.
    return (
        "Use the supplied still as the only visual source and as the opening composition of one continuous shot. "
        "Preserve the character's identity, face, armor, hands, pose, colors, lighting style, and existing city background. "
        "When a motion reference starts from this same still, preserve its frame-by-frame animation and adapt it "
        "to the requested output settings. For a different guide, transfer its camera and visible-object movement "
        "onto the still using optical flow. "
        "Do not invent effects, particles, spirals, objects, poses, or scene changes. "
        "Use a single continuous shot and finish with a steady frame; no text, logos, dialogue, or music."
    )


def _run_job(job_id: str) -> None:
    directory = _job_dir(job_id)
    with _job_lock:
        if job_id in _active_jobs:
            return
        _active_jobs.add(job_id)
    try:
        # Serialize renders to keep memory and integrated-GPU use predictable.
        with _render_lock:
            job = _read_job(job_id)
            try:
                job["status"] = "processing"
                job["started_at"] = _utc_now()
                _write_job(job)
                options = RenderOptions.from_json(json.dumps(job["options"]))
                reference_path = directory / "motion_reference.mp4"
                metrics = render_video(
                    directory / "input_image", directory / "result.mp4", options,
                    reference_path if reference_path.exists() else None,
                )
                job.update(status="completed", completed_at=_utc_now(), result=metrics)
            except Exception as exc:
                job.update(status="failed", completed_at=_utc_now(), error=str(exc), traceback=traceback.format_exc()[-4000:])
            finally:
                _write_job(job)
    finally:
        with _job_lock:
            _active_jobs.discard(job_id)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "generador-video-pipeline"}


@app.post("/v1/jobs", status_code=202)
async def create_job(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="Source still: JPEG, PNG, or WebP"),
    prompt: str = Form(..., description="Original animation prompt; retained for traceability"),
    options: str | None = Form(None, description="Optional RenderOptions JSON object"),
    motion_reference: UploadFile | None = File(
        None, description="Optional MP4 guide clip; only its motion is transferred to the source still"
    ),
    _auth: None = Depends(_authorize),
) -> dict[str, Any]:
    if image.filename is None or Path(image.filename).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(status_code=415, detail="image must be JPEG, PNG, or WebP")
    image_bytes = await image.read(MAX_UPLOAD_BYTES + 1)
    if not image_bytes or len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="image is empty or exceeds 30 MB")
    try:
        with Image.open(image_bytes_to_file(image_bytes)) as decoded:
            decoded.verify()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=415, detail="uploaded file is not a valid image")
    try:
        render_options = RenderOptions.from_json(options)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    reference_bytes = None
    reference_name = None
    if motion_reference is not None:
        if motion_reference.filename is None or Path(motion_reference.filename).suffix.lower() != ".mp4":
            raise HTTPException(status_code=415, detail="motion_reference must be an MP4 file")
        reference_bytes = await motion_reference.read(MAX_MOTION_REFERENCE_BYTES + 1)
        if not reference_bytes or len(reference_bytes) > MAX_MOTION_REFERENCE_BYTES:
            raise HTTPException(status_code=413, detail="motion_reference is empty or exceeds 100 MB")
        reference_name = Path(motion_reference.filename).name

    job_id = uuid.uuid4().hex
    directory = _job_dir(job_id)
    directory.mkdir(parents=True, exist_ok=False)
    extension = Path(image.filename).suffix.lower()
    source_path = directory / f"input{extension}"
    source_path.write_bytes(image_bytes)
    # Renderer uses a stable internal name independent of the client's original filename.
    source_path.rename(directory / "input_image")
    reference_metadata = None
    if reference_bytes is not None:
        (directory / "motion_reference.mp4").write_bytes(reference_bytes)
        reference_metadata = {
            "filename": reference_name,
            "sha256": hashlib.sha256(reference_bytes).hexdigest(),
            "bytes": len(reference_bytes),
        }
    job = {
        "job_id": job_id,
        "status": "queued",
        "created_at": _utc_now(),
        "original_filename": Path(image.filename).name,
        "source_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "motion_reference": reference_metadata,
        "prompt": prompt,
        "adjusted_animation_prompt": _adjust_prompt(prompt),
        "options": render_options.__dict__,
        "result": None,
        "error": None,
    }
    _write_job(job)
    background_tasks.add_task(_run_job, job_id)
    return {"job_id": job_id, "status": "queued", "status_url": f"/v1/jobs/{job_id}", "video_url": f"/v1/jobs/{job_id}/video"}


def image_bytes_to_file(data: bytes):
    from io import BytesIO
    return BytesIO(data)


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str, _auth: None = Depends(_authorize)) -> dict[str, Any]:
    return _read_job(job_id)


@app.get("/v1/jobs/{job_id}/video")
def get_video(job_id: str, _auth: None = Depends(_authorize)) -> FileResponse:
    job = _read_job(job_id)
    if job["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"job is {job['status']}")
    path = _job_dir(job_id) / "result.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="video result not found")
    return FileResponse(path, media_type="video/mp4", filename=f"{job_id}.mp4")
