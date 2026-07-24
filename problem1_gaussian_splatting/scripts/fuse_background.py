"""Filter dynamic/white Gaussians and export a fused background PLY.

Question 2:
  - remove person Gaussians using the provided masks
  - remove Gaussians from the right-side white padding region
  - fuse all remaining per-frame Gaussians into fusion.ply
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "input"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "fusion.ply"


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
        if parts[1] != "PINHOLE":
            raise ValueError(f"Only PINHOLE camera is supported, got {parts[1]}")
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


def project_points(means: np.ndarray, camera: Camera, pose: ImagePose) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rot = qvec_to_rotmat(pose.qvec)
    points_cam = means @ rot.T + pose.tvec
    z = points_cam[:, 2]
    x = camera.fx * (points_cam[:, 0] / z) + camera.cx
    y = camera.fy * (points_cam[:, 1] / z) + camera.cy
    valid = (z > 1e-4) & (x >= 0) & (x < camera.width) & (y >= 0) & (y < camera.height)
    return x, y, valid


def build_processed_mask(mask_path: Path, target_size: int) -> np.ndarray:
    src = Image.open(mask_path).convert("L")
    src_w, src_h = src.size
    square = Image.new("L", (src_h, src_h), 0)
    square.paste(src, (0, 0))
    resized = square.resize((target_size, target_size), Image.Resampling.NEAREST)
    return np.array(resized) > 127


def colors_from_harmonics(harmonics: np.ndarray) -> np.ndarray:
    sh_c0 = 0.28209479177387814
    return np.clip(harmonics[:, :, 0] * sh_c0 + 0.5, 0.0, 1.0).astype(np.float32)


def write_3dgs_ply(
    path: Path,
    means: np.ndarray,
    harmonics: np.ndarray,
    opacities: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scales_log = np.log(np.maximum(scales, 1e-8)).astype(np.float32)

    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {means.shape[0]}\n")
        for prop in (
            "x",
            "y",
            "z",
            "nx",
            "ny",
            "nz",
            "f_dc_0",
            "f_dc_1",
            "f_dc_2",
            "opacity",
            "scale_0",
            "scale_1",
            "scale_2",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
        ):
            f.write(f"property float {prop}\n")
        f.write("end_header\n")

        zeros = np.zeros((means.shape[0], 3), dtype=np.float32)
        rows = np.concatenate([means, zeros, harmonics, opacities[:, None], scales_log, rotations], axis=1)
        for row in rows:
            f.write(" ".join(f"{float(v):.8g}" for v in row))
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fusion.ply for question 2.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--white-threshold", type=float, default=0.92)
    parser.add_argument("--max-points-per-frame", type=int, default=0)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    data = torch.load(input_dir / "per_frame_gaussians.pt", map_location="cpu")
    frames = data["frames"]
    cameras = read_colmap_cameras(input_dir / "camera" / "0" / "cameras.txt")
    poses = read_colmap_images(input_dir / "camera" / "0" / "images.txt")

    if len(frames) != len(poses):
        raise ValueError(f"Frame count mismatch: gaussians={len(frames)}, poses={len(poses)}")

    fused_means: list[np.ndarray] = []
    fused_harmonics: list[np.ndarray] = []
    fused_opacities: list[np.ndarray] = []
    fused_scales: list[np.ndarray] = []
    fused_rotations: list[np.ndarray] = []

    for frame_idx, (frame, pose) in enumerate(zip(frames, poses)):
        camera = cameras[pose.camera_id]
        means = frame["means"].squeeze(0).cpu().numpy().astype(np.float32)
        harmonics = frame["harmonics"].squeeze(0).cpu().numpy().astype(np.float32)
        opacities = frame["opacities"].squeeze(0).cpu().numpy().astype(np.float32)
        scales = frame["scales"].squeeze(0).cpu().numpy().astype(np.float32)
        rotations = frame["rotations"].squeeze(0).cpu().numpy().astype(np.float32)

        x, y, visible = project_points(means, camera, pose)
        px = np.clip(np.rint(x).astype(np.int32), 0, camera.width - 1)
        py = np.clip(np.rint(y).astype(np.int32), 0, camera.height - 1)

        person_mask = build_processed_mask(input_dir / "masks" / f"{frame_idx}.png", camera.width)
        original_w, original_h = Image.open(input_dir / "masks" / f"{frame_idx}.png").size
        padding_start_x = math.ceil(original_w / original_h * camera.width)

        in_person = np.zeros(means.shape[0], dtype=bool)
        in_person[visible] = person_mask[py[visible], px[visible]]
        in_padding = visible & (px >= padding_start_x)

        rgb = colors_from_harmonics(harmonics)
        near_white = (rgb.min(axis=1) > args.white_threshold) & ((rgb.max(axis=1) - rgb.min(axis=1)) < 0.08)

        keep = visible & (~in_person) & (~in_padding) & (~near_white)
        keep_ids = np.where(keep)[0]
        keep_before_sampling = keep_ids.size
        if args.max_points_per_frame > 0 and keep_ids.size > args.max_points_per_frame:
            keep_ids = keep_ids[np.linspace(0, keep_ids.size - 1, args.max_points_per_frame).astype(np.int64)]

        fused_means.append(means[keep_ids])
        fused_harmonics.append(harmonics[keep_ids, :, 0])
        fused_opacities.append(opacities[keep_ids])
        fused_scales.append(scales[keep_ids])
        fused_rotations.append(rotations[keep_ids])

        removed = means.shape[0] - keep_before_sampling
        print(
            f"frame {frame_idx:02d}: total={means.shape[0]}, keep={keep_before_sampling}, written={keep_ids.size}, "
            f"remove={removed}, person={int(in_person.sum())}, padding={int(in_padding.sum())}, white={int(near_white.sum())}"
        )

    means_all = np.concatenate(fused_means, axis=0)
    harmonics_all = np.concatenate(fused_harmonics, axis=0)
    opacities_all = np.concatenate(fused_opacities, axis=0)
    scales_all = np.concatenate(fused_scales, axis=0)
    rotations_all = np.concatenate(fused_rotations, axis=0)

    write_3dgs_ply(args.output.resolve(), means_all, harmonics_all, opacities_all, scales_all, rotations_all)
    print(f"saved: {args.output.resolve()}")
    print(f"fused vertices: {means_all.shape[0]}")


if __name__ == "__main__":
    main()
