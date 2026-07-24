"""Render video from per-frame high-resolution route contexts.

This is the replacement renderer after rebuilding the visual module.
Unlike 03_render_video.py, this script does not hand-author most roads and
buildings.  It uses the frame-by-frame crops produced by
06_export_frame_contexts.py as visual evidence:

  PRIMARY   : oblique/perspective view from the current high-res crop
  COMPANION : trailing view from a previous high-res crop, with main drone mark
  BEV       : original preview image with route, current positions and frustums

The goal of this module is to restore real campus visual content first.  It is
a pragmatic rendering pass, not a photogrammetric 3D reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = PROJECT_ROOT.parents[0] / "题目2-release-pdf"
INPUT_DIR = RELEASE_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FRAME_CONTEXT_DIR = OUTPUT_DIR / "frame_contexts"
PANEL = 800


@dataclass
class FrameContext:
    frame: int
    preview_x: float
    preview_y: float
    heading_dx: float
    heading_dy: float
    highres_crop: Path


def imread(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    return image


def imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"failed to encode image: {path}")
    encoded.tofile(str(path))


def load_contexts(csv_path: Path) -> list[FrameContext]:
    contexts: list[FrameContext] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            contexts.append(
                FrameContext(
                    frame=int(row["frame"]),
                    preview_x=float(row["preview_x"]),
                    preview_y=float(row["preview_y"]),
                    heading_dx=float(row["heading_dx"]),
                    heading_dy=float(row["heading_dy"]),
                    highres_crop=Path(row["highres_crop"]),
                )
            )
    return contexts


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return v
    return v / n


def rotate_to_heading_up(image: np.ndarray, heading_dx: float, heading_dy: float) -> np.ndarray:
    # In image coordinates y goes down.  We rotate so the route heading points
    # approximately upward in the panel, like a forward-looking drone camera.
    angle = np.degrees(np.arctan2(heading_dy, heading_dx)) + 90.0
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def oblique_view_from_overhead(crop: np.ndarray, heading_dx: float, heading_dy: float, companion: bool = False) -> np.ndarray:
    rotated = rotate_to_heading_up(crop, heading_dx, heading_dy)
    h, w = rotated.shape[:2]
    side = min(h, w)
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    square = rotated[y0:y0 + side, x0:x0 + side]
    square = cv2.resize(square, (PANEL, PANEL), interpolation=cv2.INTER_AREA)

    # A trapezoid warp creates an oblique low-altitude visual impression while
    # preserving real texture from the high-resolution JPG.
    if companion:
        src = np.float32([[110, 70], [690, 70], [790, 790], [10, 790]])
        dst = np.float32([[120, 250], [680, 250], [800, 800], [0, 800]])
    else:
        src = np.float32([[90, 30], [710, 30], [790, 790], [10, 790]])
        dst = np.float32([[160, 170], [640, 170], [800, 800], [0, 800]])
    mat = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(square, mat, (PANEL, PANEL), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    sky_h = 145 if not companion else 230
    sky = np.zeros_like(warped)
    sky[:] = (218, 228, 234)
    sky[sky_h:] = warped[sky_h:]
    cv2.line(sky, (0, sky_h), (PANEL, sky_h), (205, 215, 218), 2, cv2.LINE_AA)
    return sky


def draw_drone_icon(panel: np.ndarray, center: tuple[int, int], heading_up: bool = True, scale: int = 26) -> None:
    x, y = center
    pts = np.array([[x, y - scale], [x + scale, y + scale // 2], [x, y + scale // 4], [x - scale, y + scale // 2]], dtype=np.int32)
    cv2.fillConvexPoly(panel, pts, (30, 70, 215), cv2.LINE_AA)
    cv2.circle(panel, (x, y - scale), 7, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.line(panel, (x - scale, y + scale // 2), (x + scale, y + scale // 2), (0, 0, 0), 2, cv2.LINE_AA)


def add_label(panel: np.ndarray, label: str, idx: int, total: int) -> None:
    cv2.rectangle(panel, (0, 0), (430, 52), (0, 0, 0), -1)
    cv2.putText(panel, f"{label}  {idx:04d}/{total:04d}", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)


def make_primary(context: FrameContext, idx: int, total: int) -> np.ndarray:
    crop = imread(context.highres_crop)
    panel = oblique_view_from_overhead(crop, context.heading_dx, context.heading_dy, companion=False)
    add_label(panel, "PRIMARY", idx, total)
    return panel


def make_companion(context: FrameContext, idx: int, total: int) -> np.ndarray:
    crop = imread(context.highres_crop)
    panel = oblique_view_from_overhead(crop, context.heading_dx, context.heading_dy, companion=True)
    # Main drone visible in companion view.
    draw_drone_icon(panel, (PANEL // 2, 245), scale=28)
    cv2.line(panel, (PANEL // 2 - 125, 245), (PANEL // 2 + 125, 245), (0, 0, 0), 3, cv2.LINE_AA)
    add_label(panel, "COMPANION", idx, total)
    return panel


def make_bev(preview: np.ndarray, contexts: list[FrameContext], idx: int, companion_lag: int) -> np.ndarray:
    panel = np.zeros((PANEL, PANEL, 3), dtype=np.uint8)
    scaled_h = int(round(preview.shape[0] * PANEL / preview.shape[1]))
    resized = cv2.resize(preview, (PANEL, scaled_h), interpolation=cv2.INTER_AREA)
    y0 = (PANEL - scaled_h) // 2
    panel[y0:y0 + scaled_h] = resized
    scale = PANEL / preview.shape[1]

    pts = np.array([[c.preview_x * scale, c.preview_y * scale + y0] for c in contexts], dtype=np.float32)
    cv2.polylines(panel, [np.rint(pts).astype(np.int32).reshape(-1, 1, 2)], False, (0, 255, 255), 3, cv2.LINE_AA)
    main = pts[idx]
    comp = pts[max(0, idx - companion_lag)]
    h = normalize(np.array([contexts[idx].heading_dx, contexts[idx].heading_dy], dtype=np.float32))
    for pt, color, label in [(main, (0, 0, 255), "P"), (comp, (255, 80, 0), "C")]:
        p = tuple(np.rint(pt).astype(np.int32))
        cv2.circle(panel, p, 7, color, -1, cv2.LINE_AA)
        end = tuple(np.rint(pt + h * 42).astype(np.int32))
        left = tuple(np.rint(pt + h * 62 + np.array([-h[1], h[0]]) * 22).astype(np.int32))
        right = tuple(np.rint(pt + h * 62 - np.array([-h[1], h[0]]) * 22).astype(np.int32))
        cv2.line(panel, p, end, color, 2, cv2.LINE_AA)
        cv2.line(panel, p, left, color, 1, cv2.LINE_AA)
        cv2.line(panel, p, right, color, 1, cv2.LINE_AA)
        cv2.putText(panel, label, (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    add_label(panel, "BEV", idx, len(contexts))
    return panel


def convert_to_h264(input_path: Path, output_path: Path, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        input_path.replace(output_path)
        return
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-crf",
        "24",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    input_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render video from frame_context high-resolution crops.")
    parser.add_argument("--frames", type=int, default=2000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "video_context_render.mp4")
    parser.add_argument("--frame-dir", type=Path, default=OUTPUT_DIR / "frames_context_render")
    parser.add_argument("--save-all-frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--companion-lag", type=int, default=20)
    parser.add_argument("--context-dir", type=Path, default=FRAME_CONTEXT_DIR)
    args = parser.parse_args()

    contexts = load_contexts(args.context_dir / "frame_contexts.csv")
    if args.frames < len(contexts):
        indices = np.linspace(0, len(contexts) - 1, args.frames, dtype=int)
        contexts = [contexts[int(i)] for i in indices]
    else:
        contexts = contexts[: args.frames]

    preview = imread(INPUT_DIR / "tju_overview_60k_4x4_preview_3000x2000.png")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.save_all_frames:
        args.frame_dir.mkdir(parents=True, exist_ok=True)

    temp = args.output.with_name(args.output.stem + "_tmp.mp4")
    writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (2400, 800))
    if not writer.isOpened():
        raise RuntimeError(f"failed to create video: {temp}")

    total = len(contexts)
    for i, context in enumerate(contexts):
        primary = make_primary(context, i, total)
        companion_context = contexts[max(0, i - args.companion_lag)]
        companion = make_companion(companion_context, i, total)
        bev = make_bev(preview, contexts, i, args.companion_lag)
        combo = np.hstack([primary, companion, bev])
        writer.write(combo)
        if args.save_all_frames:
            imwrite(args.frame_dir / f"{i:04d}.png", combo)
        if i == 0 or (i + 1) % 100 == 0:
            print(f"rendered {i + 1}/{total}")
    writer.release()
    convert_to_h264(temp, args.output.resolve(), args.fps)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
