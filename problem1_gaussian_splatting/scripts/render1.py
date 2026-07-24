"""Render per-frame Gaussians into render1.mp4.

This is a compact exam-oriented renderer for question 1. It reads:
  - per_frame_gaussians.pt
  - COLMAP cameras.txt and images.txt

Then it projects Gaussian centers into the 448x448 COLMAP cameras and alpha
composites their DC colors into a video.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "input"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "render1.mp4"


@dataclass(frozen=True)
class Camera:
    camera_id: int
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class ImagePose:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    image_name: str


def read_colmap_cameras(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        camera_id = int(parts[0])
        model = parts[1]
        if model != "PINHOLE":
            raise ValueError(f"Only PINHOLE camera is supported, got {model}")
        width, height = int(parts[2]), int(parts[3])
        fx, fy, cx, cy = map(float, parts[4:8])
        cameras[camera_id] = Camera(camera_id, width, height, fx, fy, cx, cy)
    return cameras


def read_colmap_images(path: Path) -> list[ImagePose]:
    poses: list[ImagePose] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        parts = line.split()
        image_id = int(parts[0])
        qvec = np.array(list(map(float, parts[1:5])), dtype=np.float32)
        tvec = np.array(list(map(float, parts[5:8])), dtype=np.float32)
        camera_id = int(parts[8])
        image_name = parts[9]
        poses.append(ImagePose(image_id, qvec, tvec, camera_id, image_name))
        i += 2
    return poses


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qz * qx + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qz * qx - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float32,
    )


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + torch.exp(-x))


def colors_from_harmonics(harmonics: torch.Tensor) -> np.ndarray:
    # 3DGS stores DC SH coefficients. The common conversion is
    # rgb = SH_C0 * dc + 0.5, then clamp to [0, 1].
    sh_c0 = 0.28209479177387814
    rgb = harmonics.squeeze(0).squeeze(-1) * sh_c0 + 0.5
    return rgb.clamp(0.0, 1.0).cpu().numpy().astype(np.float32)


def render_frame(frame: dict, camera: Camera, pose: ImagePose, point_radius: int) -> np.ndarray:
    means = frame["means"].squeeze(0).cpu().numpy().astype(np.float32)
    colors = colors_from_harmonics(frame["harmonics"])
    opacities = sigmoid(frame["opacities"].squeeze(0)).cpu().numpy().astype(np.float32)

    rot = qvec_to_rotmat(pose.qvec)
    points_cam = means @ rot.T + pose.tvec
    z = points_cam[:, 2]

    valid = z > 1e-4
    x = camera.fx * (points_cam[:, 0] / z) + camera.cx
    y = camera.fy * (points_cam[:, 1] / z) + camera.cy
    valid &= (x >= 0) & (x < camera.width) & (y >= 0) & (y < camera.height)

    idx = np.where(valid)[0]
    if idx.size == 0:
        return np.full((camera.height, camera.width, 3), 255, dtype=np.uint8)

    order = idx[np.argsort(z[idx])[::-1]]
    canvas = np.ones((camera.height, camera.width, 3), dtype=np.float32)
    alpha_canvas = np.zeros((camera.height, camera.width), dtype=np.float32)

    for gaussian_id in order:
        px = int(round(float(x[gaussian_id])))
        py = int(round(float(y[gaussian_id])))
        alpha = float(opacities[gaussian_id])
        color = colors[gaussian_id]

        y0 = max(0, py - point_radius)
        y1 = min(camera.height, py + point_radius + 1)
        x0 = max(0, px - point_radius)
        x1 = min(camera.width, px + point_radius + 1)
        if y0 >= y1 or x0 >= x1:
            continue

        local_alpha = alpha * (1.0 - alpha_canvas[y0:y1, x0:x1])
        canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * (1.0 - local_alpha[..., None]) + color * local_alpha[..., None]
        alpha_canvas[y0:y1, x0:x1] += local_alpha

    return (canvas.clip(0.0, 1.0) * 255.0).astype(np.uint8)


def save_image(path: Path, bgr: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, bgr)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def convert_to_h264(input_path: Path, output_path: Path, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        input_path.replace(output_path)
        print("[warning] ffmpeg not found; kept OpenCV mp4v video instead of H.264.")
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
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    input_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render question-1 video render1.mp4.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--save-frames", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    gaussian_path = input_dir / "per_frame_gaussians.pt"
    camera_dir = input_dir / "camera" / "0"

    data = torch.load(gaussian_path, map_location="cpu")
    frames = data["frames"]
    cameras = read_colmap_cameras(camera_dir / "cameras.txt")
    poses = read_colmap_images(camera_dir / "images.txt")

    if len(frames) != len(poses):
        raise ValueError(f"Frame count mismatch: gaussians={len(frames)}, poses={len(poses)}")

    first_camera = cameras[poses[0].camera_id]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output.parent / "render1_frames"
    if args.save_frames:
        frame_dir.mkdir(parents=True, exist_ok=True)

    temp_output = args.output.with_name(args.output.stem + "_tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_output), fourcc, args.fps, (first_camera.width, first_camera.height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {temp_output}")

    for frame_idx, (frame, pose) in enumerate(zip(frames, poses)):
        camera = cameras[pose.camera_id]
        rgb = render_frame(frame, camera, pose, args.point_radius)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
        if args.save_frames:
            save_image(frame_dir / f"{frame_idx:03d}.png", bgr)
        print(f"rendered frame {frame_idx + 1:02d}/{len(frames)}: {frame.get('image_name', pose.image_name)}")

    writer.release()
    convert_to_h264(temp_output, args.output, args.fps)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
