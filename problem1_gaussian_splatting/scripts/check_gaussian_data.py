"""Inspect the structure of per-frame Gaussian data.

This script is intentionally read-only. Run it before implementing rendering
logic to confirm that the exam data matches the expected schema.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "input" / "per_frame_gaussians.pt"
EXPECTED_FIELDS = ("means", "scales", "rotations", "harmonics")


def describe_value(value: Any) -> str:
    if hasattr(value, "shape"):
        dtype = getattr(value, "dtype", None)
        device = getattr(value, "device", None)
        suffix = []
        if dtype is not None:
            suffix.append(f"dtype={dtype}")
        if device is not None:
            suffix.append(f"device={device}")
        meta = f" ({', '.join(suffix)})" if suffix else ""
        return f"shape={tuple(value.shape)}{meta}"
    return f"type={type(value).__name__}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check per_frame_gaussians.pt structure before rendering."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to per_frame_gaussians.pt. Default: {DEFAULT_INPUT}",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data = torch.load(input_path, map_location="cpu")

    print(f"file: {input_path}")
    print(f"top-level type: {type(data).__name__}")
    if isinstance(data, dict):
        print(f"top-level keys: {list(data.keys())}")

    frames = data["frames"]
    print(f'len(data["frames"]): {len(frames)}')

    if not frames:
        raise ValueError('data["frames"] is empty')

    first_frame = frames[0]
    print(f'frames[0] type: {type(first_frame).__name__}')
    print(f'frames[0].keys(): {list(first_frame.keys())}')

    for field in EXPECTED_FIELDS:
        if field not in first_frame:
            print(f"[MISSING] frames[0][{field!r}]")
            continue
        print(f'frames[0]["{field}"]: {describe_value(first_frame[field])}')

    extra_fields = [key for key in first_frame.keys() if key not in EXPECTED_FIELDS]
    if extra_fields:
        print(f"extra fields in frames[0]: {extra_fields}")


if __name__ == "__main__":
    main()
