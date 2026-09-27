from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageOps


@dataclass
class RenderOptions:
    duration_seconds: float = 10.0
    fps: int = 24
    width: int = 720
    height: int = 1280
    preset: str = "motion_transfer"
    zoom_start: float = 1.0
    zoom_end: float = 2.35
    focal_x: float = 0.47
    focal_y: float = 0.25
    hold_seconds: float = 0.5

    @classmethod
    def from_json(cls, value: str | None) -> "RenderOptions":
        if not value:
            return cls()
        try:
            raw = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"options must be valid JSON: {exc.msg}") from exc
        if not isinstance(raw, dict):
            raise ValueError("options must be a JSON object")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"unknown render option(s): {', '.join(unknown)}")
        try:
            result = cls(**raw)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc
        result.validate()
        return result

    def validate(self) -> None:
        if not 1 <= self.duration_seconds <= 30:
            raise ValueError("duration_seconds must be between 1 and 30")
        if not 1 <= self.fps <= 60:
            raise ValueError("fps must be between 1 and 60")
        if not (64 <= self.width <= 1920 and 64 <= self.height <= 1920):
            raise ValueError("width and height must be between 64 and 1920")
        if self.width % 2 or self.height % 2:
            raise ValueError("width and height must be even for H.264 output")
        if not (1 <= self.zoom_start <= 4 and 1 <= self.zoom_end <= 4):
            raise ValueError("zoom values must be between 1 and 4")
        if not (0 <= self.focal_x <= 1 and 0 <= self.focal_y <= 1):
            raise ValueError("focal_x and focal_y must be normalized from 0 to 1")
        if not (0 <= self.hold_seconds < self.duration_seconds):
            raise ValueError("hold_seconds must be >= 0 and less than duration_seconds")


def _ease_in_out(value: float) -> float:
    return value * value * (3 - 2 * value)


def _camera_frame(
    image: Image.Image,
    progress: float,
    options: RenderOptions,
) -> Image.Image:
    width, height = image.size
    zoom = options.zoom_start + (options.zoom_end - options.zoom_start) * _ease_in_out(progress)
    # Move the framing toward the requested focal point while retaining a full-frame opening.
    cx = 0.5 + (options.focal_x - 0.5) * _ease_in_out(progress)
    cy = 0.5 + (options.focal_y - 0.5) * _ease_in_out(progress)
    crop_w, crop_h = width / zoom, height / zoom
    left = min(max(cx * width - crop_w / 2, 0), width - crop_w)
    top = min(max(cy * height - crop_h / 2, 0), height - crop_h)
    frame = image.crop((left, top, left + crop_w, top + crop_h))
    return frame.resize((options.width, options.height), Image.Resampling.LANCZOS)


def _frame_at(image: Image.Image, index: int, total_frames: int, options: RenderOptions) -> Image.Image:
    moving_frames = max(1, total_frames - round(options.hold_seconds * options.fps))
    progress = min(1.0, index / moving_frames)
    return _camera_frame(image, progress, options)


def _iter_video_frames(path: Path, options: RenderOptions, total_frames: int) -> Iterator[np.ndarray]:
    """Stream a motion guide at output size/FPS using the bundled FFmpeg."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    vf = (
        f"scale={options.width}:{options.height}:force_original_aspect_ratio=increase,"
        f"crop={options.width}:{options.height},fps={options.fps}"
    )
    command = [ffmpeg, "-v", "error", "-i", str(path), "-vf", vf, "-frames:v", str(total_frames),
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_bytes = options.width * options.height * 3
    frame_count = 0
    last_frame = None
    try:
        assert process.stdout is not None
        while frame_count < total_frames:
            raw = process.stdout.read(frame_bytes)
            if len(raw) != frame_bytes:
                break
            last_frame = np.frombuffer(raw, dtype=np.uint8).reshape(options.height, options.width, 3).copy()
            frame_count += 1
            yield last_frame
        stderr = process.stderr.read() if process.stderr else b""
        code = process.wait()
    except Exception:
        process.kill()
        process.wait()
        raise
    if code != 0:
        raise RuntimeError(f"Could not decode motion reference: {stderr.decode('utf-8', errors='replace')[-1500:]}")
    if last_frame is None:
        raise ValueError("motion_reference contains no decodable video frames")
    # Hold the last decoded frame when the guide is shorter than the requested output.
    while frame_count < total_frames:
        frame_count += 1
        yield last_frame


def _iter_motion_warp_frames(source: Image.Image, guide_frames: Iterator[np.ndarray],
                             options: RenderOptions) -> Iterator[np.ndarray]:
    """Transfer guide motion as dense optical flow while retaining source-image pixels."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for motion-reference rendering; install project dependencies") from exc

    width, height = options.width, options.height
    original = np.asarray(source, dtype=np.uint8)
    scale = 0.5
    small_size = (max(64, round(width * scale)), max(64, round(height * scale)))
    try:
        first_guide = next(guide_frames)
    except StopIteration as exc:
        raise ValueError("motion_reference contains no decodable video frames") from exc
    base_guide = cv2.resize(first_guide, small_size, interpolation=cv2.INTER_AREA)
    base_gray = cv2.cvtColor(base_guide, cv2.COLOR_RGB2GRAY)
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    yield original
    for guide in guide_frames:
        small = cv2.resize(guide, small_size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        # Backward flow maps each target pixel to its matching coordinate in the first guide frame.
        flow = cv2.calcOpticalFlowFarneback(
            gray, base_gray, None, pyr_scale=0.5, levels=3, winsize=21,
            iterations=3, poly_n=7, poly_sigma=1.5, flags=0,
        )
        flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
        flow[..., 0] *= width / small_size[0]
        flow[..., 1] *= height / small_size[1]
        # Damp noisy local vectors slightly; preserve camera and large object movement.
        flow = cv2.GaussianBlur(flow, (0, 0), sigmaX=1.2, sigmaY=1.2)
        map_x = grid_x + flow[..., 0]
        map_y = grid_y + flow[..., 1]
        frame = cv2.remap(original, map_x, map_y, interpolation=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)
        yield frame


def render_video(image_path: Path, output_path: Path, options: RenderOptions,
                 motion_reference_path: Path | None = None) -> dict[str, Any]:
    options.validate()
    total_frames = round(options.duration_seconds * options.fps)
    if not 1 <= total_frames <= 1800:
        raise ValueError("requested frame count is outside the supported range")

    with Image.open(image_path) as source:
        source = ImageOps.exif_transpose(source).convert("RGB")
        source = ImageOps.fit(source, (options.width, options.height), method=Image.Resampling.LANCZOS)
    frames: Iterator[np.ndarray]
    if motion_reference_path:
        guide = _iter_video_frames(motion_reference_path, options, total_frames)
        frames = _iter_motion_warp_frames(source, guide, options)
    else:
        frames = (np.asarray(_frame_at(source, index, total_frames, options), dtype=np.uint8)
                  for index in range(total_frames))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{options.width}x{options.height}", "-r", str(options.fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
    ]
    started = time.perf_counter()
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        assert process.stdin is not None
        for pixels in frames:
            process.stdin.write(pixels.tobytes())
        process.stdin.close()
        stderr = process.stderr.read() if process.stderr else b""
        code = process.wait()
    except Exception:
        process.kill()
        process.wait()
        raise
    if code != 0:
        raise RuntimeError(f"FFmpeg failed: {stderr.decode('utf-8', errors='replace')[-2000:]}")
    return {
        "duration_seconds": options.duration_seconds,
        "fps": options.fps,
        "frame_count": total_frames,
        "width": options.width,
        "height": options.height,
        "render_seconds": round(time.perf_counter() - started, 2),
        "motion_reference_used": motion_reference_path is not None,
        "options": asdict(options),
    }
