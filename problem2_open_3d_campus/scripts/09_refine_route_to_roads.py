"""Refine route toward road / bridge / paved-surface centerlines.

Input:
  outputs/route_refine/refined_route_points.json

Output:
  outputs/route_road_refine/
    road_refined_route_points.json
    road_score.png
    road_refine_compare.png

This is a conservative route attraction step.  It searches near each route
point for low-saturation, bright, locally smooth pixels that look like roads,
bridges, or paved plazas, while limiting movement to avoid destroying the
teacher-provided route topology.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = PROJECT_ROOT.parents[0] / "题目2-release-pdf"
INPUT_DIR = RELEASE_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "route_road_refine"
REFINED_ROUTE = PROJECT_ROOT / "outputs" / "route_refine" / "refined_route_points.json"


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


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def texture_std(gray: np.ndarray, ksize: int = 17) -> np.ndarray:
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (ksize, ksize))
    mean2 = cv2.blur(gray_f * gray_f, (ksize, ksize))
    var = np.maximum(mean2 - mean * mean, 0)
    return np.sqrt(var)


def make_valid_mask(preview: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    valid = (gray > 22).astype(np.uint8) * 255
    invalid = cv2.bitwise_not(valid)
    invalid = cv2.dilate(invalid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    return cv2.bitwise_not(invalid)


def make_road_score(preview: np.ndarray, valid: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(preview, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    std = texture_std(gray)

    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    # Roads/bridges/plazas are generally low-saturation and light, but not
    # completely textureless water/grass.  The score is intentionally soft.
    low_sat = np.clip((100.0 - sat) / 100.0, 0, 1)
    bright = np.clip((val - 70.0) / 120.0, 0, 1)
    smooth = np.clip((65.0 - std) / 65.0, 0, 1)
    not_water_smooth = np.clip(std / 28.0, 0, 1)
    score = 255.0 * (0.42 * low_sat + 0.36 * bright + 0.16 * smooth + 0.06 * not_water_smooth)
    score[valid == 0] = 0
    # Enhance thin continuous road-like structures.
    score_u8 = np.clip(score, 0, 255).astype(np.uint8)
    score_u8 = cv2.GaussianBlur(score_u8, (0, 0), 2.0)
    return score_u8.astype(np.float32)


def local_best(point: np.ndarray, tangent: np.ndarray, score: np.ndarray, valid: np.ndarray, radius: int = 90) -> np.ndarray:
    h, w = score.shape
    x, y = [int(round(v)) for v in point]
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    if x1 <= x0 or y1 <= y0:
        return point.copy()

    yy, xx = np.mgrid[y0:y1, x0:x1]
    patch_score = score[y0:y1, x0:x1]
    patch_valid = valid[y0:y1, x0:x1] > 0
    if not patch_valid.any():
        return point.copy()

    dx = xx.astype(np.float32) - point[0]
    dy = yy.astype(np.float32) - point[1]
    dist = np.sqrt(dx * dx + dy * dy)
    t = tangent / max(1e-6, float(np.linalg.norm(tangent)))
    normal = np.array([-t[1], t[0]], dtype=np.float32)
    # Prefer moving laterally to the road center; discourage large forward/back jumps.
    along = np.abs(dx * t[0] + dy * t[1])
    lateral = np.abs(dx * normal[0] + dy * normal[1])
    objective = patch_score - dist * 0.55 - along * 0.25 + lateral * 0.04
    objective[~patch_valid] = -1e9
    best = np.unravel_index(int(np.argmax(objective)), objective.shape)
    best_pt = np.array([float(xx[best]), float(yy[best])], dtype=np.float32)
    # Damp movement; route_ref remains the main authority.
    return point * 0.45 + best_pt * 0.55


def tangents(points: np.ndarray) -> np.ndarray:
    out = np.zeros_like(points)
    out[0] = points[1] - points[0]
    out[-1] = points[-1] - points[-2]
    out[1:-1] = points[2:] - points[:-2]
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(n, 1e-6)


def smooth(points: np.ndarray, keep_ends: bool = True, iterations: int = 2) -> np.ndarray:
    out = points.copy()
    for _ in range(iterations):
        prev = out.copy()
        out[1:-1] = prev[1:-1] * 0.62 + (prev[:-2] + prev[2:]) * 0.19
        if not keep_ends:
            out[0] = prev[0] * 0.75 + prev[1] * 0.25
            out[-1] = prev[-1] * 0.75 + prev[-2] * 0.25
    return out


def make_compare(preview: np.ndarray, old: np.ndarray, new: np.ndarray, score: np.ndarray) -> np.ndarray:
    heat = cv2.applyColorMap(np.clip(score, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    base = cv2.addWeighted(preview, 0.72, heat, 0.28, 0)
    cv2.polylines(base, [np.rint(old).astype(np.int32).reshape(-1, 1, 2)], False, (255, 0, 255), 7, cv2.LINE_AA)
    cv2.polylines(base, [np.rint(new).astype(np.int32).reshape(-1, 1, 2)], False, (0, 255, 255), 5, cv2.LINE_AA)
    for idx in np.linspace(0, len(old) - 1, 28, dtype=int):
        cv2.line(base, tuple(np.rint(old[idx]).astype(int)), tuple(np.rint(new[idx]).astype(int)), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.rectangle(base, (10, 10), (970, 95), (0, 0, 0), -1)
    cv2.putText(base, "magenta=input refined route, yellow=road-refined route, heat=road score", (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return base


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    preview = imread(INPUT_DIR / "tju_overview_60k_4x4_preview_3000x2000.png")
    payload = load_payload(REFINED_ROUTE)
    points = np.array(payload["points"], dtype=np.float32)
    valid = make_valid_mask(preview)
    score = make_road_score(preview, valid)
    ts = tangents(points)

    adjusted = points.copy()
    moved = []
    for i, (point, tangent) in enumerate(zip(points, ts)):
        candidate = local_best(point, tangent, score, valid, radius=95)
        move = float(np.linalg.norm(candidate - point))
        if move > 1.0:
            adjusted[i] = candidate
            moved.append({"index": i, "move_px": move, "old": point.tolist(), "new": candidate.tolist()})

    adjusted = smooth(adjusted, keep_ends=True, iterations=2)

    out_payload = dict(payload)
    out_payload["method"] = "road_refined_v1"
    out_payload["source_method"] = payload.get("method", "")
    out_payload["moved_points_before_smoothing"] = moved
    out_payload["points"] = [[float(x), float(y)] for x, y in adjusted]

    (OUTPUT_DIR / "road_refined_route_points.json").write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    imwrite(OUTPUT_DIR / "road_score.png", np.clip(score, 0, 255).astype(np.uint8))
    imwrite(OUTPUT_DIR / "road_refine_compare.png", make_compare(preview, points, adjusted, score))
    print(f"moved points: {len(moved)}")
    print(OUTPUT_DIR / "road_refined_route_points.json")
    print(OUTPUT_DIR / "road_refine_compare.png")


if __name__ == "__main__":
    main()
