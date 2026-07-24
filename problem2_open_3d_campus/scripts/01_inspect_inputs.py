"""Inspect input images for problem 2.

The full-resolution overview is extremely large, so this script reads image
metadata only and does not decode the 60000x40000 JPEG into memory.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = PROJECT_ROOT.parents[0] / "题目2-release-pdf"
INPUT_DIR = RELEASE_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "route_debug"


def inspect_image(path: Path) -> dict:
    with Image.open(path) as image:
        return {
            "name": path.name,
            "width": image.size[0],
            "height": image.size[1],
            "mode": image.mode,
            "format": image.format,
            "file_size_mb": path.stat().st_size / 1024 / 1024,
            "pixels_million": image.size[0] * image.size[1] / 1_000_000,
        }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_names = [
        "route_ref.png",
        "tju_overview_60k_4x4_preview_3000x2000.png",
        "tju_overview_60000x40000_4x4_q92.jpg",
    ]

    lines = ["# Problem 2 Input Inspection", ""]
    for name in image_names:
        info = inspect_image(INPUT_DIR / name)
        lines.append(f"## {info['name']}")
        lines.append(f"- size: {info['width']} x {info['height']}")
        lines.append(f"- mode: {info['mode']}")
        lines.append(f"- format: {info['format']}")
        lines.append(f"- file size: {info['file_size_mb']:.2f} MB")
        lines.append(f"- pixels: {info['pixels_million']:.2f} million")
        if info["pixels_million"] > 200:
            lines.append("- note: treat as huge image; avoid full decode in normal workflow")
        lines.append("")

    preview = inspect_image(INPUT_DIR / "tju_overview_60k_4x4_preview_3000x2000.png")
    route = inspect_image(INPUT_DIR / "route_ref.png")
    aligned = preview["width"] == route["width"] and preview["height"] == route["height"]
    lines.append(f"preview and route_ref pixel-aligned: {aligned}")

    output = OUTPUT_DIR / "input_inspection.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
