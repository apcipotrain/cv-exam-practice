"""Extract route-local scene elements from the high-resolution campus JPG.

This is the replacement visual module for problem 2.  It deliberately does
not render video.  Its job is to convert the overhead image into scene hints:
roads, buildings, tree clusters, grass regions, and water regions.

The module uses a route-local strategy:
  1. Load the 3000x2000 preview for global masks/debug.
  2. Use route_points.json to define the area that matters.
  3. Optionally crop corresponding high-resolution JPG tiles by multiplying
     preview coordinates by 20.
  4. Write masks and scene_annotations.json for the renderer.

This first version is conservative and explainable: color + texture + shape,
not a learned model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


Image.MAX_IMAGE_PIXELS = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = PROJECT_ROOT.parents[0] / "题目2-release-pdf"
INPUT_DIR = RELEASE_ROOT / "input"
ANNOTATION_DIR = PROJECT_ROOT / "data" / "annotations"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "scene_extract"

PREVIEW_NAME = "tju_overview_60k_4x4_preview_3000x2000.png"
HIGHRES_NAME = "tju_overview_60000x40000_4x4_q92.jpg"
PREVIEW_TO_HIGHRES = 20


@dataclass
class SceneObject:
    kind: str
    points_preview_xy: list[list[float]]
    confidence: float
    note: str


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


def make_route_roi_mask(shape: tuple[int, int], route: np.ndarray, radius: int = 180) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.rint(route).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(mask, [pts], False, 255, radius * 2, cv2.LINE_AA)
    return mask


def texture_variance(gray: np.ndarray, ksize: int = 15) -> np.ndarray:
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (ksize, ksize))
    mean_sq = cv2.blur(gray_f * gray_f, (ksize, ksize))
    var = np.maximum(mean_sq - mean * mean, 0)
    return cv2.normalize(var, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def cleanup(mask: np.ndarray, open_size: int = 5, close_size: int = 9, min_area: int = 200) -> np.ndarray:
    if open_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if close_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    out = np.zeros_like(mask)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def split_scene_masks(preview: np.ndarray, roi: np.ndarray) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(preview, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(preview, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    variance = texture_variance(gray)

    b, g, r = cv2.split(preview)
    green_index = g.astype(np.int16) - ((r.astype(np.int16) + b.astype(np.int16)) // 2)

    # Green but textured: tree/shrub clusters.
    green = ((green_index > 12) & (hsv[:, :, 1] > 35) & (hsv[:, :, 2] > 45) & (roi > 0)).astype(np.uint8) * 255
    tree = ((green > 0) & (variance > 32)).astype(np.uint8) * 255
    grass = ((green > 0) & (variance <= 45)).astype(np.uint8) * 255

    # Water in this image is uniform green-gray, not bright blue.
    water = (
        (green_index > 0)
        & (hsv[:, :, 1] < 95)
        & (hsv[:, :, 2] > 55)
        & (variance < 22)
        & (roi > 0)
    ).astype(np.uint8) * 255

    # Roads are low-saturation elongated gray/light surfaces near the route.
    road = (
        (hsv[:, :, 1] < 55)
        & (hsv[:, :, 2] > 95)
        & (variance < 70)
        & (roi > 0)
    ).astype(np.uint8) * 255

    # Buildings: bright/gray roofs plus strong edges; remove likely road/water.
    edges = cv2.Canny(gray, 70, 150)
    edge_dense = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    roof_like = (
        (hsv[:, :, 1] < 85)
        & (hsv[:, :, 2] > 105)
        & (variance > 18)
        & (roi > 0)
    )
    building = ((roof_like | (edge_dense > 0)) & (road == 0) & (water == 0) & (green == 0)).astype(np.uint8) * 255

    return {
        "road": cleanup(road, 5, 13, 400),
        "building": cleanup(building, 3, 9, 500),
        "tree": cleanup(tree, 3, 7, 120),
        "grass": cleanup(grass, 7, 17, 800),
        "water": cleanup(water, 7, 21, 1000),
    }


def contours_to_objects(mask: np.ndarray, kind: str, max_objects: int, epsilon_ratio: float, note: str) -> list[SceneObject]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_objects]
    objects: list[SceneObject] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 0:
            continue
        eps = epsilon_ratio * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)
        if approx.shape[0] < 3:
            continue
        objects.append(
            SceneObject(
                kind=kind,
                points_preview_xy=approx.astype(float).round(2).tolist(),
                confidence=0.55,
                note=note,
            )
        )
    return objects


def add_manual_priors(objects: list[SceneObject]) -> None:
    # Strong prior from visual inspection: a circular building appears near the
    # early route turn.  We keep this separate so it is easy to edit later.
    center = np.array([2305.0, 1032.0], dtype=np.float32)
    radius = 70.0
    pts = []
    for t in np.linspace(0, 2 * np.pi, 28, endpoint=False):
        pts.append([float(center[0] + np.cos(t) * radius), float(center[1] + np.sin(t) * radius)])
    objects.append(SceneObject("building_round_prior", pts, 0.8, "manual prior: obvious circular building near route"))


def make_overlay(preview: np.ndarray, route: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    overlay = preview.copy()
    colors = {
        "water": (255, 100, 0),
        "road": (80, 80, 80),
        "building": (180, 180, 210),
        "tree": (60, 180, 60),
        "grass": (80, 130, 40),
    }
    alpha = 0.45
    for name, mask in masks.items():
        color_layer = np.zeros_like(overlay)
        color_layer[:] = colors[name]
        m = mask > 0
        overlay[m] = cv2.addWeighted(overlay[m], 1 - alpha, color_layer[m], alpha, 0)
    cv2.polylines(overlay, [np.rint(route).astype(np.int32).reshape(-1, 1, 2)], False, (0, 0, 255), 7, cv2.LINE_AA)
    return overlay


def write_highres_route_tiles(route: np.ndarray, tile_count: int = 30, preview_radius: int = 140) -> None:
    """Crop route-local high-res JPG tiles without decoding the full image at once."""
    highres_path = INPUT_DIR / HIGHRES_NAME
    if not highres_path.exists():
        return
    tile_dir = OUTPUT_DIR / "tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(highres_path) as img:
        for idx in np.linspace(0, len(route) - 1, tile_count, dtype=int):
            x, y = route[idx]
            cx = int(round(x * PREVIEW_TO_HIGHRES))
            cy = int(round(y * PREVIEW_TO_HIGHRES))
            hr = preview_radius * PREVIEW_TO_HIGHRES
            box = (
                max(0, cx - hr),
                max(0, cy - hr),
                min(img.width, cx + hr),
                min(img.height, cy + hr),
            )
            crop = img.crop(box)
            crop.thumbnail((900, 900))
            crop.save(tile_dir / f"tile_{idx:04d}_previewxy_{int(x)}_{int(y)}.jpg", quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract route-local scene masks from campus overhead imagery.")
    parser.add_argument("--no-highres-tiles", action="store_true", help="Skip high-resolution JPG tile crops.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "masks").mkdir(parents=True, exist_ok=True)

    preview = imread(INPUT_DIR / PREVIEW_NAME)
    route = load_route()
    roi = make_route_roi_mask(preview.shape[:2], route, radius=220)
    masks = split_scene_masks(preview, roi)

    for name, mask in masks.items():
        imwrite(OUTPUT_DIR / "masks" / f"{name}_mask.png", mask)
    imwrite(OUTPUT_DIR / "route_roi_mask.png", roi)
    imwrite(OUTPUT_DIR / "scene_overlay_preview.png", make_overlay(preview, route, masks))

    objects: list[SceneObject] = []
    objects += contours_to_objects(masks["water"], "water", 12, 0.01, "uniform green-gray low-texture region")
    objects += contours_to_objects(masks["road"], "road", 40, 0.006, "low-saturation elongated region near route")
    objects += contours_to_objects(masks["building"], "building", 80, 0.012, "roof/edge-like region near route")
    objects += contours_to_objects(masks["tree"], "tree_cluster", 120, 0.02, "green high-texture blob")
    objects += contours_to_objects(masks["grass"], "grass", 20, 0.015, "green continuous low-texture region")
    add_manual_priors(objects)

    payload = {
        "source_preview": str(INPUT_DIR / PREVIEW_NAME),
        "source_highres": str(INPUT_DIR / HIGHRES_NAME),
        "coordinate_system": "preview_pixel_xy",
        "preview_to_highres_scale": PREVIEW_TO_HIGHRES,
        "method": "route_local_color_texture_shape_v1",
        "objects": [asdict(obj) for obj in objects],
    }
    (OUTPUT_DIR / "scene_annotations.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_highres_tiles:
        write_highres_route_tiles(route)

    print(OUTPUT_DIR / "scene_overlay_preview.png")
    print(OUTPUT_DIR / "scene_annotations.json")


if __name__ == "__main__":
    main()
