"""Render the fused background with original-resolution cameras.

Question 3 asks for direct rendering at the original portrait resolution
without cropping or resizing the video afterwards.  This script therefore
uses:
  - outputs/fusion.ply
  - outputs/camera/0/{cameras.txt, images.txt}

and writes:
  - outputs/render2.mp4
  - outputs/render2_frames/*.png, if --save-frames is enabled

The renderer is a CPU-friendly center splat renderer.  It projects Gaussian
centers and keeps the closest splat per pixel.  It is intentionally simple so
it can finish during an exam environment without a custom CUDA rasterizer.
"""

from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FUSION = PROJECT_ROOT / "outputs" / "fusion.ply"
DEFAULT_CAMERA_DIR = PROJECT_ROOT / "outputs" / "camera" / "0"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "render2.mp4"
DEFAULT_CACHE = PROJECT_ROOT / "outputs" / "fusion_render2_cache.npz"


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
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[1] != "PINHOLE":
            raise ValueError(f"Only PINHOLE cameras are supported, got {parts[1]}")
        camera_id = int(parts[0])
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
        poses.append(
            ImagePose(
                image_id=int(parts[0]),
                qvec=np.array(list(map(float, parts[1:5])), dtype=np.float32),
                tvec=np.array(list(map(float, parts[5:8])), dtype=np.float32),
                camera_id=int(parts[8]),
                image_name=parts[9],
            )
        )
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


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def dc_to_rgb(dc: np.ndarray) -> np.ndarray:
    sh_c0 = 0.28209479177387814
    return np.clip(dc * sh_c0 + 0.5, 0.0, 1.0).astype(np.float32)


def parse_ply_header(path: Path) -> tuple[str, int, list[str], int]:
    properties: list[str] = []
    fmt = ""
    vertex_count = -1
    header_bytes = 0
    with path.open("rb") as f:
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"PLY header is incomplete: {path}")
            header_bytes += len(raw)
            line = raw.decode("ascii", errors="strict").strip()
            if line.startswith("format "):
                fmt = line.split()[1]
            elif line.startswith("element vertex "):
                vertex_count = int(line.split()[2])
            elif line.startswith("property "):
                properties.append(line.split()[-1])
            elif line == "end_header":
                break
    if vertex_count < 0 or not fmt:
        raise ValueError(f"PLY header misses format or vertex count: {path}")
    return fmt, vertex_count, properties, header_bytes


def load_fusion_points(path: Path, cache_path: Path, refresh_cache: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cache_path.exists() and not refresh_cache and cache_path.stat().st_mtime >= path.stat().st_mtime:
        cached = np.load(cache_path)
        return cached["xyz"], cached["rgb"], cached["opacity"]

    fmt, vertex_count, props, header_bytes = parse_ply_header(path)
    prop_index = {name: idx for idx, name in enumerate(props)}
    required = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
    missing = [name for name in required if name not in prop_index]
    if missing:
        raise ValueError(f"PLY misses required properties: {missing}")

    cols = [prop_index[name] for name in required]
    if fmt == "ascii":
        data = np.loadtxt(path, dtype=np.float32, skiprows=_header_line_count(path), usecols=cols)
    elif fmt == "binary_little_endian":
        dtype = np.dtype([(name, "<f4") for name in props])
        with path.open("rb") as f:
            f.seek(header_bytes)
            rows = np.fromfile(f, dtype=dtype, count=vertex_count)
        data = np.column_stack([rows[name] for name in required]).astype(np.float32)
    else:
        raise ValueError(f"Unsupported PLY format: {fmt}")

    if data.shape[0] != vertex_count:
        raise ValueError(f"Expected {vertex_count} vertices, loaded {data.shape[0]}")

    xyz = np.ascontiguousarray(data[:, 0:3], dtype=np.float32)
    rgb = dc_to_rgb(np.ascontiguousarray(data[:, 3:6], dtype=np.float32))
    opacity_raw = np.ascontiguousarray(data[:, 6], dtype=np.float32)
    if opacity_raw.min() < 0.0 or opacity_raw.max() > 1.0:
        opacity = sigmoid_np(opacity_raw).astype(np.float32)
    else:
        opacity = opacity_raw

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, xyz=xyz, rgb=rgb, opacity=opacity)
    return xyz, rgb, opacity


def _header_line_count(path: Path) -> int:
    count = 0
    with path.open("rb") as f:
        for raw in f:
            count += 1
            if raw.strip() == b"end_header":
                return count
    raise ValueError(f"PLY header is incomplete: {path}")


def render_frame(
    xyz: np.ndarray,
    rgb: np.ndarray,
    opacity: np.ndarray,
    camera: Camera,
    pose: ImagePose,
    point_radius: int,
    opacity_min: float,
) -> np.ndarray:
    rot = qvec_to_rotmat(pose.qvec)
    points_cam = xyz @ rot.T + pose.tvec
    z = points_cam[:, 2]

    valid = z > 1e-4
    if opacity_min > 0:
        valid &= opacity >= opacity_min

    x = camera.fx * (points_cam[:, 0] / z) + camera.cx
    y = camera.fy * (points_cam[:, 1] / z) + camera.cy
    valid &= (x >= 0) & (x < camera.width) & (y >= 0) & (y < camera.height)

    ids = np.where(valid)[0]
    canvas = np.full((camera.height, camera.width, 3), 255, dtype=np.uint8)
    if ids.size == 0:
        return canvas

    base_px = np.rint(x[ids]).astype(np.int32)
    base_py = np.rint(y[ids]).astype(np.int32)
    base_z = z[ids].astype(np.float32)
    base_rgb = np.rint(rgb[ids] * 255.0).clip(0, 255).astype(np.uint8)

    zbuf = np.full(camera.height * camera.width, np.inf, dtype=np.float32)
    owner = np.full(camera.height * camera.width, -1, dtype=np.int32)

    offsets = [(0, 0)]
    if point_radius >= 1:
        offsets += [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if point_radius >= 2:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    point_local_ids = np.arange(ids.size, dtype=np.int32)
    for dx, dy in offsets:
        px = base_px + dx
        py = base_py + dy
        inside = (px >= 0) & (px < camera.width) & (py >= 0) & (py < camera.height)
        pix = py[inside] * camera.width + px[inside]
        depth = base_z[inside]
        local = point_local_ids[inside]

        np.minimum.at(zbuf, pix, depth)
        closest = depth == zbuf[pix]
        owner[pix[closest]] = local[closest]

    filled = owner >= 0
    canvas.reshape(-1, 3)[filled] = base_rgb[owner[filled]]
    return canvas


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
    parser = argparse.ArgumentParser(description="Render question-3 original-resolution video render2.mp4.")
    parser.add_argument("--fusion", type=Path, default=DEFAULT_FUSION)
    parser.add_argument("--camera-dir", type=Path, default=DEFAULT_CAMERA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--opacity-min", type=float, default=0.0)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    fusion = args.fusion.resolve()
    camera_dir = args.camera_dir.resolve()
    output = args.output.resolve()
    cameras = read_colmap_cameras(camera_dir / "cameras.txt")
    poses = read_colmap_images(camera_dir / "images.txt")
    xyz, rgb, opacity = load_fusion_points(fusion, args.cache.resolve(), args.refresh_cache)

    if not poses:
        raise ValueError("No camera poses found.")
    first_camera = cameras[poses[0].camera_id]
    if (first_camera.width, first_camera.height) != (1080, 1920):
        raise ValueError(f"Expected 1080x1920 camera, got {first_camera.width}x{first_camera.height}")

    output.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = output.parent / "render2_frames"
    if args.save_frames:
        frame_dir.mkdir(parents=True, exist_ok=True)

    temp_output = output.with_name(output.stem + "_tmp.mp4")
    writer = cv2.VideoWriter(str(temp_output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (first_camera.width, first_camera.height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {temp_output}")

    print(f"loaded gaussians: {xyz.shape[0]}")
    print(f"output video: {first_camera.width}x{first_camera.height}, {args.fps} fps")
    for frame_idx, pose in enumerate(poses):
        camera = cameras[pose.camera_id]
        rgb_frame = render_frame(xyz, rgb, opacity, camera, pose, args.point_radius, args.opacity_min)
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
        if args.save_frames:
            save_image(frame_dir / f"{frame_idx:03d}.png", bgr)
        print(f"rendered frame {frame_idx + 1:02d}/{len(poses)}: {pose.image_name}")

    writer.release()
    convert_to_h264(temp_output, output, args.fps)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
