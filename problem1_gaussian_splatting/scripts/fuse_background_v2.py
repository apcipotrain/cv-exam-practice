"""Create an experimental fused background model with conservative fusion.

This script is intentionally separate from fuse_background.py.  It keeps the
current deliverable untouched and writes a new model under outputs/innovation_v2.

Pipeline:
  1. Per-frame mask and padding filtering.
  2. Conservative near-white filtering to suppress padding residue.
  3. Optional multi-view mask voting for experiments; disabled by default.
  4. Small-voxel fusion for deduplication without erasing sparse background.
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
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "innovation_v2" / "fusion.ply"


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
            raise ValueError(f"Only PINHOLE is supported, got {parts[1]}")
        cameras[int(parts[0])] = Camera(
            int(parts[0]), int(parts[2]), int(parts[3]), *map(float, parts[4:8])
        )
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
                int(parts[0]),
                np.array(list(map(float, parts[1:5])), dtype=np.float32),
                np.array(list(map(float, parts[5:8])), dtype=np.float32),
                int(parts[8]),
                parts[9],
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


def project_points(points: np.ndarray, camera: Camera, pose: ImagePose) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rot = qvec_to_rotmat(pose.qvec)
    cam = points @ rot.T + pose.tvec
    z = cam[:, 2]
    x = camera.fx * (cam[:, 0] / z) + camera.cx
    y = camera.fy * (cam[:, 1] / z) + camera.cy
    valid = (z > 1e-4) & (x >= 0) & (x < camera.width) & (y >= 0) & (y < camera.height)
    return x, y, valid


def build_processed_mask(mask_path: Path, target_size: int, dilate: int) -> np.ndarray:
    src = Image.open(mask_path).convert("L")
    src_w, src_h = src.size
    square = Image.new("L", (src_h, src_h), 0)
    square.paste(src, (0, 0))
    resized = square.resize((target_size, target_size), Image.Resampling.NEAREST)
    mask = np.array(resized) > 127
    if dilate > 0:
        kernel = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return mask


def colors_from_harmonics(harmonics: np.ndarray) -> np.ndarray:
    sh_c0 = 0.28209479177387814
    return np.clip(harmonics[:, :, 0] * sh_c0 + 0.5, 0.0, 1.0).astype(np.float32)


def mask_vote_keep(
    points: np.ndarray,
    masks: list[np.ndarray],
    cameras: dict[int, Camera],
    poses: list[ImagePose],
    max_hits: int,
    batch_size: int,
) -> np.ndarray:
    if max_hits < 0:
        return np.ones(points.shape[0], dtype=bool)

    keep = np.ones(points.shape[0], dtype=bool)
    hits = np.zeros(points.shape[0], dtype=np.uint8)
    visible_count = np.zeros(points.shape[0], dtype=np.uint8)

    for pose_idx, pose in enumerate(poses):
        camera = cameras[pose.camera_id]
        for start in range(0, points.shape[0], batch_size):
            end = min(points.shape[0], start + batch_size)
            x, y, visible = project_points(points[start:end], camera, pose)
            if not np.any(visible):
                continue
            px = np.clip(np.rint(x[visible]).astype(np.int32), 0, camera.width - 1)
            py = np.clip(np.rint(y[visible]).astype(np.int32), 0, camera.height - 1)
            local_ids = np.where(visible)[0]
            mask_hit = masks[pose_idx][py, px]
            hits[start + local_ids[mask_hit]] += 1
            visible_count[start + local_ids] += 1

    keep &= hits <= max_hits
    keep &= visible_count > 0
    return keep


def voxel_fuse(
    means: np.ndarray,
    harmonics: np.ndarray,
    opacities: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    voxel_size: float,
    min_support: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keys = np.floor(means / voxel_size).astype(np.int32)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    voxel_count = counts.shape[0]
    weights = np.clip(opacities, 0.05, 1.0).astype(np.float32)

    sums_w = np.bincount(inverse, weights=weights, minlength=voxel_count).astype(np.float32)
    out_means = np.zeros((voxel_count, 3), dtype=np.float32)
    out_harmonics = np.zeros((voxel_count, 3), dtype=np.float32)
    out_scales = np.zeros((voxel_count, 3), dtype=np.float32)
    out_rotations = np.zeros((voxel_count, 4), dtype=np.float32)
    out_opacities = np.zeros(voxel_count, dtype=np.float32)

    for col in range(3):
        out_means[:, col] = np.bincount(inverse, weights=means[:, col] * weights, minlength=voxel_count) / sums_w
        out_harmonics[:, col] = np.bincount(inverse, weights=harmonics[:, col] * weights, minlength=voxel_count) / sums_w
        out_scales[:, col] = np.bincount(inverse, weights=scales[:, col] * weights, minlength=voxel_count) / sums_w
    for col in range(4):
        out_rotations[:, col] = np.bincount(inverse, weights=rotations[:, col] * weights, minlength=voxel_count) / sums_w
    out_opacities[:] = np.bincount(inverse, weights=opacities, minlength=voxel_count) / counts
    out_rotations /= np.maximum(np.linalg.norm(out_rotations, axis=1, keepdims=True), 1e-8)

    support_keep = counts >= min_support
    if support_keep.sum() < 50000:
        support_keep = counts >= 1
    return (
        out_means[support_keep],
        out_harmonics[support_keep],
        out_opacities[support_keep],
        out_scales[support_keep],
        out_rotations[support_keep],
        counts[support_keep],
    )


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
    opacity = np.clip(opacities, 1e-6, 1.0 - 1e-6)
    opacities_logit = np.log(opacity / (1.0 - opacity)).astype(np.float32)
    zeros = np.zeros((means.shape[0], 3), dtype=np.float32)
    rows = np.concatenate(
        [means, zeros, harmonics, opacities_logit[:, None], scales_log, rotations],
        axis=1,
    ).astype(np.float32)

    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {rows.shape[0]}",
            "property float x",
            "property float y",
            "property float z",
            "property float nx",
            "property float ny",
            "property float nz",
            "property float f_dc_0",
            "property float f_dc_1",
            "property float f_dc_2",
            "property float opacity",
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
            "end_header",
        ]
    )
    with path.open("wb") as f:
        f.write((header + "\n").encode("ascii"))
        rows.tofile(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create experimental conservative fusion.ply.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--voxel-size", type=float, default=0.004)
    parser.add_argument("--min-support", type=int, default=1)
    parser.add_argument("--mask-dilate", type=int, default=0)
    parser.add_argument(
        "--vote-max-hits",
        type=int,
        default=-1,
        help="Disable voting with -1. Set a non-negative value only for stricter experiments.",
    )
    parser.add_argument("--batch-size", type=int, default=120000)
    parser.add_argument("--white-threshold", type=float, default=0.97)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    data = torch.load(input_dir / "per_frame_gaussians.pt", map_location="cpu")
    frames = data["frames"]
    cameras = read_colmap_cameras(input_dir / "camera" / "0" / "cameras.txt")
    poses = read_colmap_images(input_dir / "camera" / "0" / "images.txt")
    masks = [build_processed_mask(input_dir / "masks" / f"{i}.png", cameras[poses[i].camera_id].width, args.mask_dilate) for i in range(len(poses))]

    all_means: list[np.ndarray] = []
    all_harmonics: list[np.ndarray] = []
    all_opacities: list[np.ndarray] = []
    all_scales: list[np.ndarray] = []
    all_rotations: list[np.ndarray] = []

    for frame_idx, (frame, pose) in enumerate(zip(frames, poses)):
        camera = cameras[pose.camera_id]
        means = frame["means"].squeeze(0).cpu().numpy().astype(np.float32)
        harmonics_full = frame["harmonics"].squeeze(0).cpu().numpy().astype(np.float32)
        harmonics = harmonics_full[:, :, 0]
        opacities = frame["opacities"].squeeze(0).cpu().numpy().astype(np.float32)
        scales = frame["scales"].squeeze(0).cpu().numpy().astype(np.float32)
        rotations = frame["rotations"].squeeze(0).cpu().numpy().astype(np.float32)

        x, y, visible = project_points(means, camera, pose)
        px = np.clip(np.rint(x).astype(np.int32), 0, camera.width - 1)
        py = np.clip(np.rint(y).astype(np.int32), 0, camera.height - 1)

        original_w, original_h = Image.open(input_dir / "masks" / f"{frame_idx}.png").size
        padding_start_x = math.ceil(original_w / original_h * camera.width)
        in_current_mask = np.zeros(means.shape[0], dtype=bool)
        in_current_mask[visible] = masks[frame_idx][py[visible], px[visible]]
        in_padding = visible & (px >= padding_start_x)
        rgb = colors_from_harmonics(harmonics_full)
        near_white = (rgb.min(axis=1) > args.white_threshold) & ((rgb.max(axis=1) - rgb.min(axis=1)) < 0.06)

        pre_keep = visible & (~in_current_mask) & (~in_padding) & (~near_white)
        ids = np.where(pre_keep)[0]
        if ids.size and args.vote_max_hits >= 0:
            vote_keep = mask_vote_keep(means[ids], masks, cameras, poses, args.vote_max_hits, args.batch_size)
            ids = ids[vote_keep]

        all_means.append(means[ids])
        all_harmonics.append(harmonics[ids])
        all_opacities.append(opacities[ids])
        all_scales.append(scales[ids])
        all_rotations.append(rotations[ids])
        print(
            f"frame {frame_idx:02d}: total={means.shape[0]}, keep={ids.size}, "
            f"mask={int(in_current_mask.sum())}, padding={int(in_padding.sum())}, white={int(near_white.sum())}"
        )

    means_all = np.concatenate(all_means, axis=0)
    harmonics_all = np.concatenate(all_harmonics, axis=0)
    opacities_all = np.concatenate(all_opacities, axis=0)
    scales_all = np.concatenate(all_scales, axis=0)
    rotations_all = np.concatenate(all_rotations, axis=0)
    print(f"before voxel fusion: {means_all.shape[0]}")

    means_v, harmonics_v, opacities_v, scales_v, rotations_v, support = voxel_fuse(
        means_all,
        harmonics_all,
        opacities_all,
        scales_all,
        rotations_all,
        args.voxel_size,
        args.min_support,
    )
    print(f"after voxel fusion: {means_v.shape[0]}")
    print(f"support min/mean/max: {support.min()} / {support.mean():.2f} / {support.max()}")
    write_3dgs_ply(args.output.resolve(), means_v, harmonics_v, opacities_v, scales_v, rotations_v)
    print(f"saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
