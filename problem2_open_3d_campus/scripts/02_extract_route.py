"""Extract the hand-drawn route from route_ref.png.

Outputs:
  - data/annotations/route_points.json
  - outputs/route_debug/auto_color_mask_reference.png
  - outputs/route_debug/route_overlay.png
  - outputs/route_debug/bev_sample.png
  - outputs/route_debug/README.txt
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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "route_debug"


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


def extract_yellow_route_mask(route_ref_bgr: np.ndarray, preview_bgr: np.ndarray | None = None) -> np.ndarray:
    hsv = cv2.cvtColor(route_ref_bgr, cv2.COLOR_BGR2HSV)
    # Yellow hand-drawn route and arrow.  Keep this threshold broad enough for
    # anti-aliased route pixels while avoiding vegetation.
    lower = np.array([18, 60, 100], dtype=np.uint8)
    upper = np.array([45, 255, 255], dtype=np.uint8)
    yellow = cv2.inRange(hsv, lower, upper)

    if preview_bgr is not None:
        diff = cv2.absdiff(route_ref_bgr, preview_bgr)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, changed = cv2.threshold(diff_gray, 35, 255, cv2.THRESH_BINARY)
        # Keep yellow-ish changes or saturated bright changes.  This catches
        # semi-transparent hand annotations that do not survive a pure HSV rule.
        saturated = cv2.inRange(hsv[:, :, 1], 50, 255)
        bright = cv2.inRange(hsv[:, :, 2], 80, 255)
        mask = cv2.bitwise_or(yellow, cv2.bitwise_and(changed, cv2.bitwise_and(saturated, bright)))
    else:
        mask = yellow

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if count <= 1:
        raise RuntimeError("no route component found")
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == best).astype(np.uint8) * 255


def centerline_by_principal_axis(mask: np.ndarray, bins: int = 220) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if xs.size < 20:
        raise RuntimeError("route mask is too small")
    xy = np.column_stack([xs, ys]).astype(np.float32)
    mean = xy.mean(axis=0)
    centered = xy - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    normal = vh[1]
    t = centered @ axis
    n = centered @ normal
    edges = np.linspace(t.min(), t.max(), bins + 1)

    points = []
    for i in range(bins):
        inside = (t >= edges[i]) & (t < edges[i + 1])
        if inside.sum() < 5:
            continue
        center_t = 0.5 * (edges[i] + edges[i + 1])
        center_n = np.median(n[inside])
        pt = mean + axis * center_t + normal * center_n
        points.append(pt)
    if len(points) < 2:
        raise RuntimeError("failed to estimate route centerline")
    return np.array(points, dtype=np.float32)


def simplify_and_resample(points: np.ndarray, spacing: float = 8.0) -> np.ndarray:
    if points.shape[0] < 2:
        return points
    eps = 4.0
    simplified = cv2.approxPolyDP(points.reshape(-1, 1, 2), eps, False).reshape(-1, 2).astype(np.float32)
    deltas = np.diff(simplified, axis=0)
    dist = np.sqrt((deltas * deltas).sum(axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(dist)])
    total = cumulative[-1]
    if total <= spacing:
        return simplified

    samples = np.arange(0.0, total, spacing, dtype=np.float32)
    if samples[-1] != total:
        samples = np.append(samples, total)
    out = np.empty((samples.shape[0], 2), dtype=np.float32)
    out[:, 0] = np.interp(samples, cumulative, simplified[:, 0])
    out[:, 1] = np.interp(samples, cumulative, simplified[:, 1])
    return out


def manual_route_control_points() -> np.ndarray:
    """Hand-checked control points in preview pixel coordinates.

    The route_ref annotation is a thick hand-drawn line with an arrow head and
    local loops near the start.  A purely automatic centerline can easily take a
    shortcut through that thick blob.  The task allows human-in-the-loop route
    interpretation, so these sparse control points encode the intended route
    topology and are then resampled into a smooth polyline.
    """
    return np.array(
        [
            [2600, 1135],
            [2495, 1100],
            [2415, 1080],
            [2358, 1120],
            [2325, 1242],
            [2185, 1240],
            [2198, 1122],
            [2248, 1002],
            [2168, 910],
            [2100, 875],
            [1990, 860],
            [1880, 855],
            [1760, 850],
            [1698, 845],
            [1696, 760],
            [1702, 655],
            [1718, 575],
            [1725, 500],
            [1726, 438],
            [1715, 390],
            [1730, 330],
        ],
        dtype=np.float32,
    )


def smooth_manual_route(control: np.ndarray, spacing: float = 8.0) -> np.ndarray:
    # OpenCV's approxPolyDP is not used here because the control points already
    # represent intentional turns.  Resampling keeps camera speed consistent.
    deltas = np.diff(control, axis=0)
    dist = np.sqrt((deltas * deltas).sum(axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(dist)])
    samples = np.arange(0.0, cumulative[-1], spacing, dtype=np.float32)
    if samples.size == 0 or samples[-1] != cumulative[-1]:
        samples = np.append(samples, cumulative[-1])
    points = np.empty((samples.shape[0], 2), dtype=np.float32)
    points[:, 0] = np.interp(samples, cumulative, control[:, 0])
    points[:, 1] = np.interp(samples, cumulative, control[:, 1])
    return points


def choose_direction(points: np.ndarray, route_ref_bgr: np.ndarray, preview_bgr: np.ndarray) -> np.ndarray:
    # The task says to fly from the no-arrow end to the arrow end.  The arrow is
    # usually drawn as a denser yellow component near one end.  We use a small
    # endpoint-neighborhood vote; if uncertain, preserve the extracted order.
    mask = extract_yellow_route_mask(route_ref_bgr, preview_bgr)
    start = points[0]
    end = points[-1]
    radius = 45

    def local_count(pt: np.ndarray) -> int:
        x, y = map(int, np.round(pt))
        x0, x1 = max(0, x - radius), min(mask.shape[1], x + radius + 1)
        y0, y1 = max(0, y - radius), min(mask.shape[0], y + radius + 1)
        return int((mask[y0:y1, x0:x1] > 0).sum())

    # Arrow heads have more yellow pixels around the endpoint, so make that the
    # final point.
    if local_count(start) > local_count(end) * 1.15:
        return points[::-1].copy()
    return points


def make_route_overlay(preview: np.ndarray, points: np.ndarray) -> np.ndarray:
    overlay = preview.copy()
    pts = np.rint(points).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(overlay, [pts], False, (0, 255, 255), 8, cv2.LINE_AA)
    cv2.circle(overlay, tuple(pts[0, 0]), 18, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(overlay, tuple(pts[-1, 0]), 18, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(overlay, "START", tuple(pts[0, 0] + np.array([20, -20])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    cv2.putText(overlay, "END", tuple(pts[-1, 0] + np.array([20, -20])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    return overlay


def make_bev_sample(preview: np.ndarray, points: np.ndarray) -> np.ndarray:
    panel = np.zeros((800, 800, 3), dtype=np.uint8)
    scaled_h = int(round(preview.shape[0] * 800 / preview.shape[1]))
    resized = cv2.resize(preview, (800, scaled_h), interpolation=cv2.INTER_AREA)
    y0 = (800 - scaled_h) // 2
    panel[y0:y0 + scaled_h] = resized

    scale = 800 / preview.shape[1]
    pts = points.copy()
    pts[:, 0] *= scale
    pts[:, 1] = pts[:, 1] * scale + y0
    pts_i = np.rint(pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(panel, [pts_i], False, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(panel, tuple(pts_i[0, 0]), 6, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(panel, tuple(pts_i[-1, 0]), 6, (0, 0, 255), -1, cv2.LINE_AA)
    return panel


def write_debug_readme() -> None:
    text = """route_debug 文件说明

route_overlay.png：
最终采用的路线叠加图。黄线是飞行路线，绿色点是起点，红色点是箭头终点。

bev_sample.png：
题目要求的 BEV 视图雏形。它把最终路线缩放到 800x800 面板中，检查是否符合不裁剪、不拉伸的要求。

auto_color_mask_reference.png：
旧的自动颜色阈值检测结果，只作为反例/调试参考。白色表示“算法认为可能是手绘路线的像素”，但它会误把操场、亮色区域等识别进去，所以最终没有用它作为路线依据。

route_points.json：
最终路线点坐标，坐标系是 3000x2000 预览图像素坐标。
"""
    (OUTPUT_DIR / "README.txt").write_text(text, encoding="utf-8")


def main() -> None:
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    route_ref = imread(INPUT_DIR / "route_ref.png")
    preview = imread(INPUT_DIR / "tju_overview_60k_4x4_preview_3000x2000.png")
    if route_ref.shape[:2] != preview.shape[:2]:
        raise ValueError(f"route_ref and preview size mismatch: {route_ref.shape[:2]} vs {preview.shape[:2]}")

    mask = largest_component(extract_yellow_route_mask(route_ref, preview))
    control_points = manual_route_control_points()
    points = smooth_manual_route(control_points, spacing=8.0)

    imwrite(OUTPUT_DIR / "auto_color_mask_reference.png", mask)
    imwrite(OUTPUT_DIR / "route_overlay.png", make_route_overlay(preview, points))
    imwrite(OUTPUT_DIR / "bev_sample.png", make_bev_sample(preview, points))
    write_debug_readme()

    payload = {
        "source": "route_ref.png",
        "coordinate_system": "preview_pixel_xy",
        "image_width": int(preview.shape[1]),
        "image_height": int(preview.shape[0]),
        "method": "manual_control_points_resampled",
        "control_points": [[float(x), float(y)] for x, y in control_points],
        "point_count": int(points.shape[0]),
        "points": [[float(x), float(y)] for x, y in points],
    }
    (ANNOTATION_DIR / "route_points.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"route points: {points.shape[0]}")
    print(ANNOTATION_DIR / "route_points.json")
    print(OUTPUT_DIR / "route_overlay.png")
    print(OUTPUT_DIR / "bev_sample.png")
    print(OUTPUT_DIR / "README.txt")


if __name__ == "__main__":
    main()
