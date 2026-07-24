"""Render an abstract low-poly 3D campus flight video for problem 2.

This is the final abstract renderer.  It uses the overhead imagery only as a
reference for route correction and scene layout, then renders a simplified
campus model instead of pasting aerial photos into PRIMARY/COMPANION.

It builds an abstract 3D scene with roads, water, building blocks, trees and
grass-like ground, then renders:
  PRIMARY 800x800 | COMPANION 800x800 | BEV 800x800

The style intentionally follows the passing examples: readable low-poly,
not photorealistic satellite texture.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = PROJECT_ROOT.parents[0] / "题目2-release-pdf"
INPUT_DIR = RELEASE_ROOT / "input"
ANNOTATION_DIR = PROJECT_ROOT / "data" / "annotations"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REFINED_ROUTE_JSON = OUTPUT_DIR / "route_refine" / "refined_route_points.json"

PANEL = 800
PREVIEW_W = 3000
PREVIEW_H = 2000
PIXELS_PER_METER = 3.0


@dataclass(frozen=True)
class Camera:
    position: np.ndarray
    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray
    fov_deg: float


@dataclass(frozen=True)
class Poly3D:
    points: np.ndarray
    color: tuple[int, int, int]
    outline: tuple[int, int, int] | None = None


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


def pixel_to_world(points_xy: np.ndarray) -> np.ndarray:
    world = np.empty((points_xy.shape[0], 3), dtype=np.float32)
    world[:, 0] = (points_xy[:, 0] - PREVIEW_W / 2) / PIXELS_PER_METER
    world[:, 1] = (points_xy[:, 1] - PREVIEW_H / 2) / PIXELS_PER_METER
    world[:, 2] = 0.0
    return world


def world_to_pixel(points_xyz: np.ndarray) -> np.ndarray:
    xy = np.empty((points_xyz.shape[0], 2), dtype=np.float32)
    xy[:, 0] = points_xyz[:, 0] * PIXELS_PER_METER + PREVIEW_W / 2
    xy[:, 1] = points_xyz[:, 1] * PIXELS_PER_METER + PREVIEW_H / 2
    return xy


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return v
    return v / n


def make_camera(position: np.ndarray, forward: np.ndarray, fov_deg: float = 90.0) -> Camera:
    forward = normalize(forward.astype(np.float32))
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    # The route/world XY coordinates come from image pixels, whose y-axis points
    # downward.  Using cross(forward, up) makes PRIMARY/COMPANION appear
    # left-right mirrored relative to BEV.  Flip the camera handedness so the
    # viewer's left/right matches the map reference.
    right = -normalize(np.cross(forward, world_up))
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    up = normalize(np.cross(forward, right))
    return Camera(position.astype(np.float32), forward, right, up, fov_deg)


def project(points: np.ndarray, camera: Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rel = points - camera.position
    x = rel @ camera.right
    y = rel @ camera.up
    z = rel @ camera.forward
    focal = PANEL / (2.0 * math.tan(math.radians(camera.fov_deg) / 2.0))
    u = PANEL / 2 + focal * x / z
    v = PANEL / 2 - focal * y / z
    valid = z > 0.1
    return np.column_stack([u, v]).astype(np.float32), z, valid


def rect_world(cx_px: float, cy_px: float, w_px: float, h_px: float, angle_deg: float = 0.0) -> np.ndarray:
    w = w_px / PIXELS_PER_METER
    h = h_px / PIXELS_PER_METER
    local = np.array([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0]], dtype=np.float32)
    angle = math.radians(angle_deg)
    rot = np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]], dtype=np.float32)
    center = pixel_to_world(np.array([[cx_px, cy_px]], dtype=np.float32))[0]
    return local @ rot.T + center


def box_polys(cx: float, cy: float, w: float, h: float, height_m: float, angle: float, color: tuple[int, int, int]) -> list[Poly3D]:
    base = rect_world(cx, cy, w, h, angle)
    top = base.copy()
    top[:, 2] = height_m
    side_color = tuple(max(0, int(c * 0.82)) for c in color)
    dark_color = tuple(max(0, int(c * 0.68)) for c in color)
    polys = [Poly3D(top, color, (210, 210, 210))]
    for i in range(4):
        j = (i + 1) % 4
        polys.append(Poly3D(np.array([base[i], base[j], top[j], top[i]], dtype=np.float32), side_color if i % 2 else dark_color))
    return polys


def route_from_json() -> np.ndarray:
    route_path = REFINED_ROUTE_JSON if REFINED_ROUTE_JSON.exists() else ANNOTATION_DIR / "route_points.json"
    payload = json.loads(route_path.read_text(encoding="utf-8"))
    points = np.array(payload["points"], dtype=np.float32)
    return pixel_to_world(points)


def sample_route(route: np.ndarray, frame_idx: int, frame_count: int) -> tuple[np.ndarray, np.ndarray, float]:
    deltas = np.diff(route[:, :2], axis=0)
    seg_len = np.sqrt((deltas * deltas).sum(axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cumulative[-1])
    target = total * frame_idx / max(1, frame_count - 1)
    i = int(np.searchsorted(cumulative, target, side="right") - 1)
    i = max(0, min(i, len(seg_len) - 1))
    alpha = (target - cumulative[i]) / max(1e-6, seg_len[i])
    pos2 = route[i, :2] * (1 - alpha) + route[i + 1, :2] * alpha
    tangent2 = normalize(route[i + 1, :2] - route[i, :2])
    pos = np.array([pos2[0], pos2[1], 50.0], dtype=np.float32)
    tangent = np.array([tangent2[0], tangent2[1], 0.0], dtype=np.float32)
    return pos, tangent, target


def min_distance_to_route_xy(point_xy: np.ndarray, route_xy: np.ndarray) -> float:
    best = float("inf")
    for a, b in zip(route_xy[:-1], route_xy[1:]):
        ab = b - a
        denom = float(ab @ ab)
        if denom <= 1e-6:
            d = float(np.linalg.norm(point_xy - a))
        else:
            t = max(0.0, min(1.0, float((point_xy - a) @ ab / denom)))
            closest = a + ab * t
            d = float(np.linalg.norm(point_xy - closest))
        best = min(best, d)
    return best


def road_polys_from_pixel_polyline(points_px: list[list[float]], width_px: float, color: tuple[int, int, int]) -> list[Poly3D]:
    """Convert a 2D map polyline to separate low-poly road surface strips."""
    points = pixel_to_world(np.array(points_px, dtype=np.float32))
    half = width_px / PIXELS_PER_METER * 0.5
    out: list[Poly3D] = []
    for a, b in zip(points[:-1], points[1:]):
        d = b[:2] - a[:2]
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            continue
        side = np.array([-d[1], d[0]], dtype=np.float32) / n * half
        quad = np.array(
            [
                [a[0] + side[0], a[1] + side[1], 0.08],
                [b[0] + side[0], b[1] + side[1], 0.08],
                [b[0] - side[0], b[1] - side[1], 0.08],
                [a[0] - side[0], a[1] - side[1], 0.08],
            ],
            dtype=np.float32,
        )
        out.append(Poly3D(quad, color, (105, 105, 100)))
    return out


def road_polys_from_world_polyline(points: np.ndarray, width_m: float, color: tuple[int, int, int]) -> list[Poly3D]:
    """Build road surface strips directly from the flight route in world space."""
    half = width_m * 0.5
    out: list[Poly3D] = []
    for a, b in zip(points[:-1], points[1:]):
        d = b[:2] - a[:2]
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            continue
        side = np.array([-d[1], d[0]], dtype=np.float32) / n * half
        quad = np.array(
            [
                [a[0] + side[0], a[1] + side[1], 0.10],
                [b[0] + side[0], b[1] + side[1], 0.10],
                [b[0] - side[0], b[1] - side[1], 0.10],
                [a[0] - side[0], a[1] - side[1], 0.10],
            ],
            dtype=np.float32,
        )
        out.append(Poly3D(quad, color, None))
    return out


def add_tree_row(
    trees: list[tuple[np.ndarray, float]],
    start_px: tuple[float, float],
    end_px: tuple[float, float],
    count: int,
    offset_px: float = 0.0,
    height: float = 9.0,
) -> None:
    """Add an evenly spaced row of abstract trees along a map segment."""
    a = np.array(start_px, dtype=np.float32)
    b = np.array(end_px, dtype=np.float32)
    d = b - a
    n = float(np.linalg.norm(d))
    if n < 1e-6:
        return
    side = np.array([-d[1], d[0]], dtype=np.float32) / n
    for t in np.linspace(0.0, 1.0, count):
        p = a * (1.0 - t) + b * t + side * offset_px
        trees.append((pixel_to_world(p[None, :])[0], height))


def add_route_tree_rows(
    trees: list[tuple[np.ndarray, float]],
    route: np.ndarray,
    offset_m: float,
    spacing_m: float,
    height: float,
) -> None:
    """Place trees along both sides of the route-following road."""
    for a, b in zip(route[:-1], route[1:]):
        d = b[:2] - a[:2]
        length = float(np.linalg.norm(d))
        if length < 1e-6:
            continue
        tangent = d / length
        side = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        count = max(2, int(length / spacing_m))
        for t in np.linspace(0.08, 0.92, count):
            base_xy = a[:2] * (1.0 - t) + b[:2] * t
            for sign in (-1.0, 1.0):
                p = np.array([base_xy[0] + side[0] * offset_m * sign, base_xy[1] + side[1] * offset_m * sign, 0.0], dtype=np.float32)
                trees.append((p, height))


def make_scene(route: np.ndarray) -> tuple[list[Poly3D], list[tuple[np.ndarray, tuple[int, int, int], int]], list[tuple[np.ndarray, float]]]:
    polys: list[Poly3D] = []
    lines: list[tuple[np.ndarray, tuple[int, int, int], int]] = []
    trees: list[tuple[np.ndarray, float]] = []

    # Ground.
    ground = np.array(
        [
            [-520, -360, 0],
            [520, -360, 0],
            [520, 360, 0],
            [-520, 360, 0],
        ],
        dtype=np.float32,
    )
    polys.append(Poly3D(ground, (220, 218, 204)))

    # Water bodies.
    water1 = pixel_to_world(np.array([[1450, 900], [1640, 875], [1820, 900], [1800, 980], [1520, 1020], [1320, 990]], dtype=np.float32))
    water2 = pixel_to_world(np.array([[1770, 850], [2060, 845], [2150, 900], [2050, 950], [1800, 940]], dtype=np.float32))
    polys += [Poly3D(water1, (230, 145, 35)), Poly3D(water2, (230, 145, 35))]

    # Main road surface.  Previous versions used a hand-authored campus road
    # network, which could appear horizontal or unrelated to the drone motion.
    # For the final exam video, PRIMARY/COMPANION should privilege readability:
    # the road in 3D follows the given flight route itself.
    polys.extend(road_polys_from_world_polyline(route, 18.0, (96, 98, 94)))

    # Buildings near the route and campus core.  Coordinates are in preview
    # pixels and intentionally coarse to match the passing low-poly examples.
    buildings = [
        (1550, 360, 160, 90, 18, -12), (1900, 360, 160, 100, 22, 8), (2050, 350, 130, 95, 20, 6),
        (1580, 520, 140, 100, 20, 0), (1870, 520, 130, 110, 18, 0),
        (2050, 480, 150, 90, 26, 8), (2245, 465, 130, 110, 24, -8), (2380, 470, 140, 120, 22, -8),
        (1860, 650, 150, 100, 22, 0), (2030, 650, 170, 95, 24, 0), (2240, 650, 130, 95, 24, 0),
        (1760, 780, 130, 90, 20, 0), (1950, 760, 170, 90, 26, 0), (2140, 770, 150, 100, 26, 0),
        (1830, 980, 180, 140, 24, 0), (2045, 990, 170, 145, 28, 0), (2260, 1010, 150, 150, 24, 0),
        (2210, 1220, 120, 130, 22, -10), (2360, 1220, 130, 130, 20, 5), (2480, 1010, 130, 120, 18, 20),
        (1550, 1110, 190, 120, 26, -5), (1700, 1240, 210, 140, 20, 0), (1980, 1260, 190, 130, 24, 0),
        (2400, 1430, 180, 120, 18, 0), (1450, 1440, 250, 110, 18, 0),
    ]
    route_xy = route[:, :2]
    for cx, cy, w, h, height, angle in buildings:
        center_world = pixel_to_world(np.array([[cx, cy]], dtype=np.float32))[0, :2]
        approx_radius = max(w, h) / PIXELS_PER_METER * 0.5
        if min_distance_to_route_xy(center_world, route_xy) < approx_radius * 0.35:
            continue
        polys.extend(box_polys(cx, cy, w, h, height, angle, (196, 193, 181)))

    # Dense tree rows along the actual route-following road.
    add_route_tree_rows(trees, route, offset_m=16.0, spacing_m=11.0, height=8.5)

    # Small grove clusters around grass fields and water edges.
    grove_pixels = [
        (1830, 930), (1880, 930), (1930, 930), (1980, 930), (2030, 930),
        (1860, 1080), (1930, 1080), (2000, 1080), (2070, 1080),
        (2320, 940), (2380, 960), (2440, 980), (2500, 1010),
        (2350, 1320), (2420, 1325), (2490, 1330), (2555, 1340),
        (1580, 620), (1620, 690), (1660, 760), (1780, 700),
    ]
    for x, y in grove_pixels:
        trees.append((pixel_to_world(np.array([[x, y]], dtype=np.float32))[0], 8.5))

    return polys, lines, trees


def draw_projected_poly(frame: np.ndarray, poly: Poly3D, camera: Camera) -> float | None:
    pts2, z, valid = project(poly.points, camera)
    if not np.all(valid):
        return None
    pts = np.rint(pts2).astype(np.int32)
    if np.all((pts[:, 0] < -200) | (pts[:, 0] > PANEL + 200) | (pts[:, 1] < -200) | (pts[:, 1] > PANEL + 200)):
        return None
    cv2.fillConvexPoly(frame, pts, poly.color, cv2.LINE_AA)
    if poly.outline is not None:
        cv2.polylines(frame, [pts.reshape(-1, 1, 2)], True, poly.outline, 1, cv2.LINE_AA)
    return float(z.mean())


def draw_line3d(frame: np.ndarray, points: np.ndarray, color: tuple[int, int, int], thickness: int, camera: Camera) -> None:
    pts2, _, valid = project(points, camera)
    if valid.sum() < 2:
        return
    pts = np.rint(pts2[valid]).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], False, color, thickness, cv2.LINE_AA)


def draw_drone(frame: np.ndarray, pos: np.ndarray, heading: np.ndarray, camera: Camera, scale: float = 9.0) -> None:
    h = normalize(heading[:2])
    right2 = np.array([-h[1], h[0]], dtype=np.float32)
    center = pos.copy()
    nose = center + np.array([h[0], h[1], 0], dtype=np.float32) * scale
    tail = center - np.array([h[0], h[1], 0], dtype=np.float32) * scale * 0.7
    left = center - np.array([right2[0], right2[1], 0], dtype=np.float32) * scale
    right = center + np.array([right2[0], right2[1], 0], dtype=np.float32) * scale
    pts = np.array([nose, right, tail, left], dtype=np.float32)
    pts2, _, valid = project(pts, camera)
    if valid.sum() < 4:
        return
    q = np.rint(pts2).astype(np.int32)
    cv2.fillConvexPoly(frame, q, (30, 70, 210), cv2.LINE_AA)
    cv2.circle(frame, tuple(q[0]), 7, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.line(frame, tuple(q[1]), tuple(q[3]), (0, 0, 0), 2, cv2.LINE_AA)


def draw_ground_grid(frame: np.ndarray, camera: Camera) -> None:
    """Draw a lightweight ground reference grid so end-of-route views stay readable."""
    for x in np.linspace(-500, 500, 13):
        pts = np.array([[x, y, 0.05] for y in np.linspace(-340, 340, 22)], dtype=np.float32)
        draw_line3d(frame, pts, (198, 198, 188), 1, camera)
    for y in np.linspace(-340, 340, 11):
        pts = np.array([[x, y, 0.05] for x in np.linspace(-500, 500, 24)], dtype=np.float32)
        draw_line3d(frame, pts, (198, 198, 188), 1, camera)


def render_view(
    camera: Camera,
    polys: list[Poly3D],
    lines: list[tuple[np.ndarray, tuple[int, int, int], int]],
    trees: list[tuple[np.ndarray, float]],
    route: np.ndarray,
    drone_pos: np.ndarray,
    drone_heading: np.ndarray,
    label: str,
    frame_idx: int,
    frame_count: int,
    show_drone: bool,
    show_ground_route: bool = False,
) -> np.ndarray:
    frame = np.full((PANEL, PANEL, 3), (220, 218, 204), dtype=np.uint8)
    frame[:300, :] = (218, 228, 234)

    draw_items = []
    for poly in polys:
        center = poly.points.mean(axis=0)
        _, z, valid = project(center[None, :], camera)
        if valid[0]:
            draw_items.append((float(z[0]), poly))
    for _, poly in sorted(draw_items, key=lambda item: item[0], reverse=True):
        draw_projected_poly(frame, poly, camera)

    # Do not draw a synthetic grid in the final answer.  The passing examples
    # use clean low-poly ground; grid lines look like unexplained artifacts.
    for pts, color, thickness in lines:
        draw_line3d(frame, pts, color, thickness, camera)
    if show_ground_route:
        # Disabled by default: drawing the route after buildings makes the
        # ground path visually sit on roofs.  BEV is the authoritative route
        # display; PRIMARY/COMPANION should focus on the 3D scene.
        draw_line3d(frame, route, (0, 230, 230), 3, camera)

    for base, height in trees:
        trunk = np.array([base + [0, 0, 0], base + [0, 0, height]], dtype=np.float32)
        crown = np.array([base[0], base[1], height + 4], dtype=np.float32)
        draw_line3d(frame, trunk, (80, 90, 70), 3, camera)
        p2, _, valid = project(crown[None, :], camera)
        if valid[0]:
            radius = min(18, max(3, int(15 / max(0.35, np.linalg.norm(crown - camera.position) / 90))))
            cv2.circle(frame, tuple(np.rint(p2[0]).astype(np.int32)), radius, (110, 205, 170), -1, cv2.LINE_AA)

    if show_drone:
        draw_drone(frame, drone_pos, drone_heading, camera)

    cv2.rectangle(frame, (0, 0), (430, 52), (0, 0, 0), -1)
    cv2.putText(frame, f"{label}  {frame_idx:04d}/{frame_count:04d}", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def make_bev(preview: np.ndarray, route_px: np.ndarray, companion_px: np.ndarray, main_px: np.ndarray, heading_px: np.ndarray, frame_idx: int, frame_count: int) -> np.ndarray:
    panel = np.zeros((PANEL, PANEL, 3), dtype=np.uint8)
    scaled_h = int(round(preview.shape[0] * PANEL / preview.shape[1]))
    resized = cv2.resize(preview, (PANEL, scaled_h), interpolation=cv2.INTER_AREA)
    y0 = (PANEL - scaled_h) // 2
    panel[y0:y0 + scaled_h] = resized
    scale = PANEL / preview.shape[1]

    def to_bev(xy: np.ndarray) -> np.ndarray:
        out = xy.copy().astype(np.float32)
        out[:, 0] *= scale
        out[:, 1] = out[:, 1] * scale + y0
        return out

    route_i = np.rint(to_bev(route_px)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(panel, [route_i], False, (0, 255, 255), 3, cv2.LINE_AA)

    main = to_bev(main_px[None, :])[0]
    comp = to_bev(companion_px[None, :])[0]
    h = normalize(heading_px.astype(np.float32))
    for pt, color, name in [(main, (0, 0, 255), "P"), (comp, (255, 80, 0), "C")]:
        p = tuple(np.rint(pt).astype(np.int32))
        cv2.circle(panel, p, 7, color, -1, cv2.LINE_AA)
        end = tuple(np.rint(pt + h * 45).astype(np.int32))
        left = tuple(np.rint(pt + h * 60 + np.array([-h[1], h[0]]) * 22).astype(np.int32))
        right = tuple(np.rint(pt + h * 60 - np.array([-h[1], h[0]]) * 22).astype(np.int32))
        cv2.line(panel, p, end, color, 2, cv2.LINE_AA)
        cv2.line(panel, p, left, color, 1, cv2.LINE_AA)
        cv2.line(panel, p, right, color, 1, cv2.LINE_AA)
        cv2.putText(panel, name, (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    cv2.rectangle(panel, (0, 0), (330, 52), (0, 0, 0), -1)
    cv2.putText(panel, f"BEV  {frame_idx:04d}/{frame_count:04d}", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
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
    parser = argparse.ArgumentParser(description="Render problem2 low-poly 3-view video.")
    parser.add_argument("--frames", type=int, default=2000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "video.mp4")
    parser.add_argument(
        "--save-preview-frames",
        action="store_true",
        help="Save only three check frames: first, middle, and last.",
    )
    parser.add_argument(
        "--save-all-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save every composed 2400x800 frame. Enabled by default for visual algorithm debugging.",
    )
    parser.add_argument("--frame-dir", type=Path, default=OUTPUT_DIR / "frames_all")
    args = parser.parse_args()

    preview = imread(INPUT_DIR / "tju_overview_60k_4x4_preview_3000x2000.png")
    route = route_from_json()
    route_px = world_to_pixel(route)
    polys, lines, trees = make_scene(route)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = args.frame_dir
    if args.save_preview_frames or args.save_all_frames:
        frame_dir.mkdir(parents=True, exist_ok=True)

    temp = args.output.with_name(args.output.stem + "_tmp.mp4")
    writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (2400, 800))
    if not writer.isOpened():
        raise RuntimeError(f"failed to create video: {temp}")

    for i in range(args.frames):
        main_pos, tangent, _ = sample_route(route, i, args.frames)
        horizontal = normalize(tangent[:2])
        main_forward = normalize(np.array([horizontal[0], horizontal[1], -1.0], dtype=np.float32))
        primary_cam = make_camera(main_pos, main_forward, 90.0)

        companion_pos = main_pos - np.array([horizontal[0], horizontal[1], 0.0], dtype=np.float32) * 20.0
        companion_pos[2] = 50.0
        # Aim at the main drone to satisfy visibility while remaining aligned
        # with the current flight direction.
        companion_forward = normalize((main_pos - companion_pos) + np.array([0.0, 0.0, -5.0], dtype=np.float32))
        companion_cam = make_camera(companion_pos, companion_forward, 90.0)

        primary = render_view(primary_cam, polys, lines, trees, route, main_pos, tangent, "PRIMARY", i, args.frames, False)
        companion = render_view(companion_cam, polys, lines, trees, route, main_pos, tangent, "COMPANION", i, args.frames, True)
        main_px = world_to_pixel(main_pos[None, :])[0]
        comp_px = world_to_pixel(companion_pos[None, :])[0]
        bev = make_bev(preview, route_px, comp_px, main_px, horizontal * 80.0, i, args.frames)

        combo = np.hstack([primary, companion, bev])
        writer.write(combo)
        if args.save_all_frames or (args.save_preview_frames and i in {0, args.frames // 2, args.frames - 1}):
            imwrite(frame_dir / f"{i:04d}.png", combo)
        if (i + 1) % 30 == 0 or i == 0:
            print(f"rendered {i + 1}/{args.frames}")

    writer.release()
    convert_to_h264(temp, args.output.resolve(), args.fps)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
