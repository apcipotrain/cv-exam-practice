"""Export per-frame route context crops and a table for manual scene modeling.

This script is intentionally different from 05_extract_scene_from_jpg.py:
it does not try to solve the whole segmentation problem at once.  Instead, it
creates a frame-by-frame evidence folder so the 3D scene can be modeled by
looking at the exact surroundings of every flight frame.

For each sampled frame it writes:
  - a local preview crop with route direction / left / right arrows
  - optionally a high-resolution JPG crop around the same route point
  - a row in frame_contexts.csv with center/left/right color-texture hints

The output is meant to be inspected first, then used to rebuild the visual
module and the 3D scene.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


Image.MAX_IMAGE_PIXELS = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = PROJECT_ROOT.parents[0] / "题目2-release-pdf"
INPUT_DIR = RELEASE_ROOT / "input"
ANNOTATION_DIR = PROJECT_ROOT / "data" / "annotations"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "frame_contexts"

PREVIEW_NAME = "tju_overview_60k_4x4_preview_3000x2000.png"
HIGHRES_NAME = "tju_overview_60000x40000_4x4_q92.jpg"
PREVIEW_TO_HIGHRES = 20


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


def load_route(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return np.array(payload["points"], dtype=np.float32)


def sample_polyline(points: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    delta = np.diff(points, axis=0)
    seg_len = np.sqrt((delta * delta).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    targets = np.linspace(0.0, float(cum[-1]), count, dtype=np.float32)
    out = np.empty((count, 2), dtype=np.float32)
    tangent = np.empty((count, 2), dtype=np.float32)
    for k, target in enumerate(targets):
        i = int(np.searchsorted(cum, target, side="right") - 1)
        i = max(0, min(i, len(points) - 2))
        alpha = (target - cum[i]) / max(1e-6, cum[i + 1] - cum[i])
        out[k] = points[i] * (1 - alpha) + points[i + 1] * alpha
        d = points[i + 1] - points[i]
        n = float(np.linalg.norm(d))
        tangent[k] = d / n if n > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)
    return out, tangent


def safe_crop_bgr(image: np.ndarray, center: np.ndarray, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    x, y = [int(round(v)) for v in center]
    half = size // 2
    crop = np.zeros((size, size, 3), dtype=np.uint8)
    x0, x1 = max(0, x - half), min(w, x + half)
    y0, y1 = max(0, y - half), min(h, y + half)
    dx0 = x0 - (x - half)
    dy0 = y0 - (y - half)
    crop[dy0:dy0 + (y1 - y0), dx0:dx0 + (x1 - x0)] = image[y0:y1, x0:x1]
    return crop


def classify_patch(patch: np.ndarray) -> tuple[str, float, float, float]:
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(patch)
    green_index = float(np.mean(g.astype(np.float32) - 0.5 * r.astype(np.float32) - 0.5 * b.astype(np.float32)))
    sat = float(np.mean(hsv[:, :, 1]))
    val = float(np.mean(hsv[:, :, 2]))
    tex = float(np.std(gray))

    if val < 30:
        return "outside/black", green_index, sat, tex
    if green_index > 14 and tex > 34:
        return "tree_cluster_candidate", green_index, sat, tex
    if green_index > 12 and tex <= 34:
        return "grass_or_water_candidate", green_index, sat, tex
    if sat < 55 and val > 105 and tex < 38:
        return "road_candidate", green_index, sat, tex
    if sat < 90 and val > 105 and tex >= 38:
        return "building_candidate", green_index, sat, tex
    return "mixed_uncertain", green_index, sat, tex


def draw_context_overlay(crop: np.ndarray, tangent: np.ndarray, title: str) -> np.ndarray:
    out = crop.copy()
    h, w = out.shape[:2]
    c = np.array([w / 2, h / 2], dtype=np.float32)
    t = tangent / max(1e-6, float(np.linalg.norm(tangent)))
    left = np.array([-t[1], t[0]], dtype=np.float32)
    scale = min(w, h) * 0.28

    cv2.arrowedLine(out, tuple(np.rint(c - t * scale * 0.5).astype(int)), tuple(np.rint(c + t * scale).astype(int)), (0, 0, 255), 4, cv2.LINE_AA)
    cv2.arrowedLine(out, tuple(np.rint(c).astype(int)), tuple(np.rint(c + left * scale * 0.75).astype(int)), (0, 255, 0), 3, cv2.LINE_AA)
    cv2.arrowedLine(out, tuple(np.rint(c).astype(int)), tuple(np.rint(c - left * scale * 0.75).astype(int)), (255, 80, 0), 3, cv2.LINE_AA)
    cv2.circle(out, tuple(np.rint(c).astype(int)), 7, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (min(w, 520), 38), (0, 0, 0), -1)
    cv2.putText(out, title, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def export_highres_crop(img: Image.Image, center_preview: np.ndarray, output_path: Path, radius_preview: int, max_size: int) -> None:
    cx = int(round(float(center_preview[0]) * PREVIEW_TO_HIGHRES))
    cy = int(round(float(center_preview[1]) * PREVIEW_TO_HIGHRES))
    hr = radius_preview * PREVIEW_TO_HIGHRES
    box = (
        max(0, cx - hr),
        max(0, cy - hr),
        min(img.width, cx + hr),
        min(img.height, cy + hr),
    )
    crop = img.crop(box)
    crop.thumbnail((max_size, max_size))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-frame route context crops and CSV table.")
    parser.add_argument("--frames", type=int, default=2000)
    parser.add_argument("--preview-size", type=int, default=420)
    parser.add_argument("--highres-radius", type=int, default=180)
    parser.add_argument("--highres-max-size", type=int, default=900)
    parser.add_argument("--highres-step", type=int, default=1, help="Export one high-res crop every N frames.")
    parser.add_argument("--no-highres", action="store_true")
    parser.add_argument("--route-json", type=Path, default=ANNOTATION_DIR / "route_points.json")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "preview_crops"
    highres_dir = output_dir / "highres_crops"
    preview_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_highres:
        highres_dir.mkdir(parents=True, exist_ok=True)

    preview = imread(INPUT_DIR / PREVIEW_NAME)
    route = load_route(args.route_json)
    samples, tangents = sample_polyline(route, args.frames)

    csv_path = output_dir / "frame_contexts.csv"
    highres_img = None if args.no_highres else Image.open(INPUT_DIR / HIGHRES_NAME)
    try:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "frame",
                    "preview_x",
                    "preview_y",
                    "heading_dx",
                    "heading_dy",
                    "center_label",
                    "left_label",
                    "right_label",
                    "center_green_index",
                    "center_saturation",
                    "center_texture_std",
                    "preview_crop",
                    "highres_crop",
                ]
            )
            for frame_idx, (center, tangent) in enumerate(zip(samples, tangents)):
                t = tangent / max(1e-6, float(np.linalg.norm(tangent)))
                left = np.array([-t[1], t[0]], dtype=np.float32)
                center_patch = safe_crop_bgr(preview, center, 55)
                left_patch = safe_crop_bgr(preview, center + left * 80, 55)
                right_patch = safe_crop_bgr(preview, center - left * 80, 55)
                center_label, gi, sat, tex = classify_patch(center_patch)
                left_label, _, _, _ = classify_patch(left_patch)
                right_label, _, _, _ = classify_patch(right_patch)

                preview_crop = safe_crop_bgr(preview, center, args.preview_size)
                preview_crop = draw_context_overlay(preview_crop, tangent, f"F{frame_idx:04d} x={center[0]:.1f} y={center[1]:.1f}")
                preview_path = preview_dir / f"frame_{frame_idx:04d}_preview.png"
                imwrite(preview_path, preview_crop)

                highres_path = ""
                if highres_img is not None and frame_idx % max(1, args.highres_step) == 0:
                    out = highres_dir / f"frame_{frame_idx:04d}_highres.jpg"
                    export_highres_crop(highres_img, center, out, args.highres_radius, args.highres_max_size)
                    highres_path = str(out)

                writer.writerow(
                    [
                        frame_idx,
                        f"{center[0]:.3f}",
                        f"{center[1]:.3f}",
                        f"{t[0]:.6f}",
                        f"{t[1]:.6f}",
                        center_label,
                        left_label,
                        right_label,
                        f"{gi:.3f}",
                        f"{sat:.3f}",
                        f"{tex:.3f}",
                        str(preview_path),
                        highres_path,
                    ]
                )
                if frame_idx == 0 or (frame_idx + 1) % 100 == 0:
                    print(f"exported {frame_idx + 1}/{args.frames}")
    finally:
        if highres_img is not None:
            highres_img.close()

    readme = f"""frame_contexts 输出说明

preview_crops/：
每一帧在 3000x2000 preview 上的局部裁片。红色箭头表示飞行方向，绿色表示左侧，蓝色表示右侧。

highres_crops/：
每一帧在 60000x40000 JPG 上对应位置的高清局部裁片。默认每帧都导出，可用 --highres-step 降低数量。

frame_contexts.csv：
每一帧的中心点坐标、方向、中心/左/右的粗分类标签和图块路径。

坐标关系：
preview 坐标乘以 {PREVIEW_TO_HIGHRES} 约等于高清 JPG 坐标。
"""
    (output_dir / "README.txt").write_text(readme, encoding="utf-8")
    print(output_dir)
    print(csv_path)


if __name__ == "__main__":
    main()
