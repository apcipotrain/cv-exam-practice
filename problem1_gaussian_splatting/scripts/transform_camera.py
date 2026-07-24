"""Transform the supplied 448x448 COLMAP cameras back to 1080x1920.

The source images were padded only on the right:
    1080x1920 -> 1920x1920 -> 448x448

Therefore the camera intrinsics are scaled by 1920 / 448.  There is no crop
offset to subtract because the original image starts at the top-left corner
of the padded square.  Camera poses in images.txt remain unchanged.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CAMERA = PROJECT_ROOT / "data" / "input" / "camera"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "camera"


def find_input_camera_dir(project_root: Path) -> Path:
    preferred = DEFAULT_INPUT_CAMERA / "0" / "cameras.txt"
    if preferred.exists():
        return DEFAULT_INPUT_CAMERA

    matches = list(project_root.glob("**/camera/0/cameras.txt"))
    matches = [path for path in matches if "outputs" not in path.relative_to(project_root).parts]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one */camera/0/cameras.txt under {project_root}, "
            f"found {len(matches)}"
        )
    return matches[0].parent.parent


def transform_cameras(
    source: Path,
    destination: Path,
    original_width: int,
    original_height: int,
    processed_size: int,
) -> int:
    scale = original_height / processed_size
    output_lines: list[str] = []
    camera_count = 0

    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            output_lines.append(raw_line)
            continue

        parts = line.split()
        if len(parts) != 8 or parts[1] != "PINHOLE":
            raise ValueError(f"Unsupported cameras.txt line: {raw_line}")

        camera_id = int(parts[0])
        width, height = int(parts[2]), int(parts[3])
        fx, fy, cx, cy = map(float, parts[4:8])
        if width != processed_size or height != processed_size:
            raise ValueError(
                f"Camera {camera_id} is {width}x{height}, expected "
                f"{processed_size}x{processed_size}"
            )

        output_lines.append(
            f"{camera_id} PINHOLE {original_width} {original_height} "
            f"{fx * scale:.12g} {fy * scale:.12g} "
            f"{cx * scale:.12g} {cy * scale:.12g}"
        )
        camera_count += 1

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    return camera_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transform 448x448 COLMAP intrinsics to 1080x1920."
    )
    parser.add_argument("--input-camera", type=Path)
    parser.add_argument("--output-camera", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--original-width", type=int, default=1080)
    parser.add_argument("--original-height", type=int, default=1920)
    parser.add_argument("--processed-size", type=int, default=448)
    args = parser.parse_args()

    input_camera = (
        args.input_camera.resolve()
        if args.input_camera
        else find_input_camera_dir(PROJECT_ROOT)
    )
    output_camera = args.output_camera.resolve()
    source_model = input_camera / "0"
    destination_model = output_camera / "0"

    count = transform_cameras(
        source_model / "cameras.txt",
        destination_model / "cameras.txt",
        args.original_width,
        args.original_height,
        args.processed_size,
    )

    for name in ("images.txt", "points3D.txt"):
        source_file = source_model / name
        if source_file.exists():
            destination_model.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination_model / name)

    scale = args.original_height / args.processed_size
    print(f"transformed cameras: {count}")
    print(f"intrinsic scale: {scale:.12g}")
    print(f"output size: {args.original_width}x{args.original_height}")
    print(f"saved camera model: {output_camera}")


if __name__ == "__main__":
    main()
