from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


@dataclass
class RenderOptions:
    duration_seconds: float = 10.0
    fps: int = 24
    width: int = 720
    height: int = 1280
    preset: str = "energy_push_in"
    zoom_start: float = 1.0
    zoom_end: float = 2.35
    focal_x: float = 0.47
    focal_y: float = 0.25
    hold_seconds: float = 0.5
    animate_energy: bool = True
    energy_x: float = 0.30
    energy_y: float = 0.69
    energy_radius: float = 0.235
    energy_turns: float = 1.15
    energy_growth: float = 0.75
    energy_strength: float = 0.95

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
        for name in ("energy_x", "energy_y", "energy_radius", "energy_strength"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if not 0 <= self.energy_growth <= 3 or not 0 <= self.energy_turns <= 5:
            raise ValueError("energy_growth must be 0..3 and energy_turns 0..5")


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


def _energy_overlay(image_size: tuple[int, int], phase: float, progress: float, options: RenderOptions) -> Image.Image:
    width, height = image_size
    cx, cy = round(options.energy_x * width), round(options.energy_y * height)
    base_radius = options.energy_radius * min(width, height)
    radius_progress = min(1.0, progress / 0.9)
    radius = base_radius * (1.0 + options.energy_growth * radius_progress)
    pad = round(radius * 1.3)
    box = (max(0, cx - pad), max(0, cy - pad), min(width, cx + pad), min(height, cy + pad))
    patch_w, patch_h = box[2] - box[0], box[3] - box[1]
    if patch_w < 2 or patch_h < 2:
        return Image.new("RGBA", image_size)

    # Draw the luminous spiral at half resolution; camera zoom makes it fill the frame later.
    scale = 0.5
    small = (max(2, round(patch_w * scale)), max(2, round(patch_h * scale)))
    overlay = Image.new("RGBA", small, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    ox, oy = (cx - box[0]) * scale, (cy - box[1]) * scale
    pulse = 0.72 + 0.28 * math.sin(phase * 2.0 * math.pi)
    fade = 1.0 if progress <= 0.9 else max(0.0, (1.0 - progress) / 0.1)
    rmax = radius * scale * (0.88 + 0.12 * math.sin(phase * 2 * math.pi))

    # Draw several bright, offset spiral ribbons; blur only these trails to create their glow.
    for ribbon, color, width_scale, direction in (
        (0, (255, 68, 245, 235), 0.095, 1),
        (1, (255, 218, 104, 215), 0.070, -1),
        (2, (174, 70, 255, 190), 0.050, 1),
    ):
        points: list[tuple[int, int]] = []
        steps = 420
        for index in range(steps + 1):
            t = index / steps
            angle = direction * (t * options.energy_turns * 2 * math.pi) + phase * 2 * math.pi + ribbon * math.pi
            r = max(1, rmax * (0.14 + 0.86 * t))
            points.append((round(ox + r * math.cos(angle)), round(oy + r * math.sin(angle))))
        draw.line(points, fill=(color[0], color[1], color[2], round(color[3] * pulse * fade * options.energy_strength)),
                  width=max(1, round(rmax * width_scale)), joint="curve")

    # A few light flecks make the existing vortex read as motion without changing the subject.
    for particle in range(18):
        angle = phase * 2 * math.pi + particle * (2 * math.pi / 18)
        orbit = rmax * (0.72 + 0.25 * math.sin(particle * 2.3 + phase * 5))
        px, py = ox + orbit * math.cos(angle), oy + orbit * math.sin(angle)
        dot = max(1, round(rmax * (0.006 + (particle % 3) * 0.002)))
        draw.ellipse((px - dot, py - dot, px + dot, py + dot),
                     fill=(255, 216, 255, round(165 * pulse * fade * options.energy_strength)))

    # A small bright center suggests the existing energy core without changing the character.
    core = max(2, round(rmax * 0.075))
    draw.ellipse((ox - core, oy - core, ox + core, oy + core),
                 fill=(255, 225, 255, round(215 * pulse * fade * options.energy_strength)))
    blur = overlay.filter(ImageFilter.GaussianBlur(max(1, round(7 * scale))))
    overlay.alpha_composite(blur)
    overlay = overlay.resize((patch_w, patch_h), Image.Resampling.BICUBIC)
    full = Image.new("RGBA", image_size, (0, 0, 0, 0))
    full.alpha_composite(overlay, (box[0], box[1]))
    return full


def _frame_at(image: Image.Image, index: int, total_frames: int, options: RenderOptions) -> Image.Image:
    time_seconds = index / options.fps
    moving_frames = max(1, total_frames - round(options.hold_seconds * options.fps))
    progress = min(1.0, index / moving_frames)
    frame = _camera_frame(image, progress, options)
    if options.animate_energy:
        overlay = _energy_overlay(
            frame.size,
            phase=(time_seconds / max(options.duration_seconds, 0.001) * 2.4 + progress * 0.14) % 1.0,
            progress=progress,
            options=options,
        )
        frame = Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB")
    return frame


def render_video(image_path: Path, output_path: Path, options: RenderOptions) -> dict[str, Any]:
    options.validate()
    total_frames = round(options.duration_seconds * options.fps)
    if not 1 <= total_frames <= 1800:
        raise ValueError("requested frame count is outside the supported range")

    with Image.open(image_path) as source:
        source = ImageOps.exif_transpose(source).convert("RGB")
        source = ImageOps.fit(source, (options.width, options.height), method=Image.Resampling.LANCZOS)

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
        for index in range(total_frames):
            frame = _frame_at(source, index, total_frames, options)
            process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
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
        "options": asdict(options),
    }
