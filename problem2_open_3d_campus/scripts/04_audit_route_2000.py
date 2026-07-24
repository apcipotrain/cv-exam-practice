"""Audit the flight route with 2000-frame sampling.

This script does not render the 3D video.  It inspects the BEV route itself:
  - samples the final route into 2000 frame positions
  - splits the route into readable segments
  - samples left/right image patches around each segment
  - writes visual debug images and CSV/JSON reports

The goal is to catch route problems before polishing the 3D scene, especially:
  - route points crossing buildings
  - route points drifting away from roads
  - unclear left/right surroundings
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = PROJECT_ROOT.parents[0] / "题目2-release-pdf"
INPUT_DIR = RELEASE_ROOT / "input"
ANNOTATION_DIR = PROJECT_ROOT / "data" / "annotations"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "route_audit_2000"

FRAME_COUNT = 2000
SEGMENT_COUNT = 20


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


def load_route() -> np.ndarray:
    payload = json.loads((ANNOTATION_DIR / "route_points.json").read_text(encoding="utf-8"))
    return np.array(payload["points"], dtype=np.float32)


def cumulative_lengths(points: np.ndarray) -> np.ndarray:
    delta = np.diff(points, axis=0)
    seg = np.sqrt((delta * delta).sum(axis=1))
    return np.concatenate([[0.0], np.cumsum(seg)])


def sample_polyline(points: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    cum = cumulative_lengths(points)
    total = float(cum[-1])
    targets = np.linspace(0.0, total, count, dtype=np.float32)
    out = np.empty((count, 2), dtype=np.float32)
    tangent = np.empty((count, 2), dtype=np.float32)
    for k, t in enumerate(targets):
        i = int(np.searchsorted(cum, t, side="right") - 1)
        i = max(0, min(i, len(points) - 2))
        alpha = (t - cum[i]) / max(1e-6, cum[i + 1] - cum[i])
        out[k] = points[i] * (1 - alpha) + points[i + 1] * alpha
        d = points[i + 1] - points[i]
        n = float(np.linalg.norm(d))
        tangent[k] = d / n if n > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)
    return out, tangent


def color_label(bgr: np.ndarray) -> str:
    b, g, r = [float(x) for x in bgr]
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = [int(x) for x in hsv]
    if v < 45:
        return "black/outside"
    if s < 35 and v > 170:
        return "light roof/road"
    if g > r * 1.08 and g > b * 1.05:
        return "vegetation"
    if b > r * 1.25 and b > g * 1.05:
        return "water"
    if abs(r - g) < 25 and abs(g - b) < 25:
        return "gray road/roof"
    if r > g and g > b:
        return "bare ground/building"
    return f"hsv({h},{s},{v})"


def safe_patch(image: np.ndarray, center: np.ndarray, size: int = 180) -> np.ndarray:
    h, w = image.shape[:2]
    x, y = [int(round(v)) for v in center]
    half = size // 2
    patch = np.zeros((size, size, 3), dtype=np.uint8)
    x0, x1 = max(0, x - half), min(w, x + half)
    y0, y1 = max(0, y - half), min(h, y + half)
    px0, py0 = x0 - (x - half), y0 - (y - half)
    patch[py0:py0 + (y1 - y0), px0:px0 + (x1 - x0)] = image[y0:y1, x0:x1]
    return patch


def draw_segment_patch(image: np.ndarray, center: np.ndarray, tangent: np.ndarray, title: str) -> np.ndarray:
    patch = safe_patch(image, center, 220)
    c = np.array([110, 110], dtype=np.float32)
    t = tangent / max(1e-6, float(np.linalg.norm(tangent)))
    left = np.array([-t[1], t[0]], dtype=np.float32)
    cv2.arrowedLine(patch, tuple(np.rint(c - t * 55).astype(int)), tuple(np.rint(c + t * 65).astype(int)), (0, 255, 255), 3, cv2.LINE_AA)
    cv2.arrowedLine(patch, tuple(np.rint(c).astype(int)), tuple(np.rint(c + left * 55).astype(int)), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.arrowedLine(patch, tuple(np.rint(c).astype(int)), tuple(np.rint(c - left * 55).astype(int)), (255, 80, 0), 2, cv2.LINE_AA)
    cv2.circle(patch, tuple(np.rint(c).astype(int)), 5, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.rectangle(patch, (0, 0), (220, 28), (0, 0, 0), -1)
    cv2.putText(patch, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return patch


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    preview = imread(INPUT_DIR / "tju_overview_60k_4x4_preview_3000x2000.png")
    route = load_route()
    samples, tangents = sample_polyline(route, FRAME_COUNT)

    overlay = preview.copy()
    cv2.polylines(overlay, [np.rint(route).astype(np.int32).reshape(-1, 1, 2)], False, (0, 255, 255), 7, cv2.LINE_AA)
    for k in np.linspace(0, FRAME_COUNT - 1, 41, dtype=int):
        p = samples[k]
        cv2.circle(overlay, tuple(np.rint(p).astype(int)), 8, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(overlay, str(k), tuple(np.rint(p + np.array([10, -10])).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
    imwrite(OUTPUT_DIR / "route_2000_sample_overlay.png", overlay)

    csv_path = OUTPUT_DIR / "frame_samples_2000.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y", "heading_dx", "heading_dy"])
        for i, (p, t) in enumerate(zip(samples, tangents)):
            writer.writerow([i, f"{p[0]:.3f}", f"{p[1]:.3f}", f"{t[0]:.5f}", f"{t[1]:.5f}"])

    rows = []
    patches = []
    for seg_id in range(SEGMENT_COUNT):
        a = int(round(seg_id * (FRAME_COUNT - 1) / SEGMENT_COUNT))
        b = int(round((seg_id + 1) * (FRAME_COUNT - 1) / SEGMENT_COUNT))
        mid = (a + b) // 2
        p = samples[mid]
        t = tangents[mid]
        left = np.array([-t[1], t[0]], dtype=np.float32)
        left_p = p + left * 70
        right_p = p - left * 70
        road_patch = safe_patch(preview, p, 45)
        left_patch = safe_patch(preview, left_p, 45)
        right_patch = safe_patch(preview, right_p, 45)
        road_color = road_patch.reshape(-1, 3).mean(axis=0).astype(np.uint8)
        left_color = left_patch.reshape(-1, 3).mean(axis=0).astype(np.uint8)
        right_color = right_patch.reshape(-1, 3).mean(axis=0).astype(np.uint8)
        rows.append(
            {
                "segment": seg_id + 1,
                "frames": f"{a}-{b}",
                "center_xy": [round(float(p[0]), 1), round(float(p[1]), 1)],
                "center_label": color_label(road_color),
                "left_label": color_label(left_color),
                "right_label": color_label(right_color),
            }
        )
        patches.append(draw_segment_patch(preview, p, t, f"S{seg_id + 1:02d} F{a}-{b}"))

    sheet = np.zeros((4 * 220, 5 * 220, 3), dtype=np.uint8)
    for i, patch in enumerate(patches):
        y = (i // 5) * 220
        x = (i % 5) * 220
        sheet[y:y + 220, x:x + 220] = patch
    imwrite(OUTPUT_DIR / "segment_contact_sheet.png", sheet)

    md = ["# 2000帧路线分段体检", "", "说明：绿色箭头表示左侧，蓝色箭头表示右侧，黄色箭头表示飞行方向。", ""]
    md.append("| 段 | 帧范围 | 中心点(x,y) | 路线附近 | 左侧 | 右侧 |")
    md.append("|---:|---|---|---|---|---|")
    for row in rows:
        md.append(
            f"| {row['segment']} | {row['frames']} | {row['center_xy']} | {row['center_label']} | {row['left_label']} | {row['right_label']} |"
        )
    (OUTPUT_DIR / "route_segment_report.md").write_text("\n".join(md), encoding="utf-8")
    (OUTPUT_DIR / "route_segment_report.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"samples: {FRAME_COUNT}")
    print(OUTPUT_DIR / "route_2000_sample_overlay.png")
    print(OUTPUT_DIR / "segment_contact_sheet.png")
    print(OUTPUT_DIR / "frame_samples_2000.csv")
    print(OUTPUT_DIR / "route_segment_report.md")


if __name__ == "__main__":
    main()
