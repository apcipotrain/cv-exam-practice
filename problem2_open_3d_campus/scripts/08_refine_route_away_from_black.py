"""Refine route points away from black/outside image regions.

The hand-drawn route in route_ref.png partly touches the black background near
the campus boundary.  This script does not overwrite the original route.  It
creates a corrected candidate route by moving points with too much black
neighborhood toward nearby valid campus pixels.

Outputs:
  outputs/route_refine/
    route_refine_overlay.png
    route_refine_compare.png
    refined_route_points.json
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = PROJECT_ROOT.parents[0] / "题目2-release-pdf"
INPUT_DIR = RELEASE_ROOT / "input"
ANNOTATION_DIR = PROJECT_ROOT / "data" / "annotations"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "route_refine"


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


def load_route_payload() -> dict:
    return json.loads((ANNOTATION_DIR / "route_points.json").read_text(encoding="utf-8"))


def valid_campus_mask(preview: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    # Black background is nearly zero.  We dilate invalid regions to keep the
    # route comfortably inside the orthophoto boundary, not merely on the edge.
    valid = (gray > 20).astype(np.uint8) * 255
    invalid = cv2.bitwise_not(valid)
    invalid = cv2.dilate(invalid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)), iterations=1)
    return cv2.bitwise_not(invalid)


def road_preference_score(preview: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(preview, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    # Road-like: low saturation, middle/high brightness, locally not too dark.
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    score = (255 - sat) * 0.55 + val * 0.45
    score[gray < 35] = 0
    return score.astype(np.float32)


def black_ratio(preview: np.ndarray, point: np.ndarray, radius: int = 70) -> float:
    h, w = preview.shape[:2]
    x, y = [int(round(v)) for v in point]
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    crop = preview[y0:y1, x0:x1]
    if crop.size == 0:
        return 1.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float((gray < 20).mean())


def find_better_point(point: np.ndarray, valid: np.ndarray, score: np.ndarray, max_radius: int = 170) -> np.ndarray:
    h, w = valid.shape[:2]
    x, y = [int(round(v)) for v in point]
    x0, x1 = max(0, x - max_radius), min(w, x + max_radius + 1)
    y0, y1 = max(0, y - max_radius), min(h, y + max_radius + 1)
    patch_valid = valid[y0:y1, x0:x1] > 0
    if not patch_valid.any():
        return point.copy()
    yy, xx = np.where(patch_valid)
    abs_x = xx + x0
    abs_y = yy + y0
    dist = np.sqrt((abs_x - point[0]) ** 2 + (abs_y - point[1]) ** 2)
    local_score = score[abs_y, abs_x]
    # Favor valid road-like pixels, but do not jump too far.
    objective = local_score - dist * 0.55
    best = int(np.argmax(objective))
    return np.array([float(abs_x[best]), float(abs_y[best])], dtype=np.float32)


def smooth_route(points: np.ndarray, iterations: int = 3) -> np.ndarray:
    out = points.copy()
    for _ in range(iterations):
        prev = out.copy()
        out[1:-1] = prev[1:-1] * 0.55 + (prev[:-2] + prev[2:]) * 0.225
    return out


def resample(points: np.ndarray, count: int) -> np.ndarray:
    delta = np.diff(points, axis=0)
    dist = np.sqrt((delta * delta).sum(axis=1))
    cum = np.concatenate([[0.0], np.cumsum(dist)])
    targets = np.linspace(0.0, float(cum[-1]), count, dtype=np.float32)
    out = np.empty((count, 2), dtype=np.float32)
    out[:, 0] = np.interp(targets, cum, points[:, 0])
    out[:, 1] = np.interp(targets, cum, points[:, 1])
    return out


def make_compare(preview: np.ndarray, old: np.ndarray, new: np.ndarray) -> np.ndarray:
    canvas = preview.copy()
    cv2.polylines(canvas, [np.rint(old).astype(np.int32).reshape(-1, 1, 2)], False, (255, 0, 255), 7, cv2.LINE_AA)
    cv2.polylines(canvas, [np.rint(new).astype(np.int32).reshape(-1, 1, 2)], False, (0, 255, 255), 5, cv2.LINE_AA)
    for idx in np.linspace(0, len(old) - 1, 20, dtype=int):
        cv2.line(canvas, tuple(np.rint(old[idx]).astype(int)), tuple(np.rint(new[idx]).astype(int)), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (10, 10), (760, 90), (0, 0, 0), -1)
    cv2.putText(canvas, "magenta=old route, yellow=refined route, red=movement", (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    preview = imread(INPUT_DIR / "tju_overview_60k_4x4_preview_3000x2000.png")
    payload = load_route_payload()
    points = np.array(payload["points"], dtype=np.float32)
    valid = valid_campus_mask(preview)
    score = road_preference_score(preview)

    refined = points.copy()
    moved = []
    for i, point in enumerate(points):
        ratio = black_ratio(preview, point, radius=85)
        if ratio > 0.04:
            refined[i] = find_better_point(point, valid, score, max_radius=190)
            moved.append({"index": i, "black_ratio": ratio, "old": point.tolist(), "new": refined[i].tolist()})

    refined = smooth_route(refined, iterations=2)
    refined = resample(refined, len(points))

    out_payload = dict(payload)
    out_payload["method"] = "refined_away_from_black_v1"
    out_payload["source_method"] = payload.get("method", "")
    out_payload["moved_points_before_smoothing"] = moved
    out_payload["points"] = [[float(x), float(y)] for x, y in refined]
    (OUTPUT_DIR / "refined_route_points.json").write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    imwrite(OUTPUT_DIR / "valid_campus_mask.png", valid)
    imwrite(OUTPUT_DIR / "route_refine_compare.png", make_compare(preview, points, refined))
    print(f"moved points: {len(moved)}")
    print(OUTPUT_DIR / "refined_route_points.json")
    print(OUTPUT_DIR / "route_refine_compare.png")


if __name__ == "__main__":
    main()
