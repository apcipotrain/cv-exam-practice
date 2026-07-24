# Problem 1: Gaussian Splatting Rendering Practice

This folder contains an educational implementation for working with per-frame 3D Gaussian data, foreground masks, COLMAP-style cameras, background fusion, and original-resolution video rendering.

The original task materials and datasets are not included.

## Data Layout

Place your own data under:

```text
data/input/
  per_frame_gaussians.pt
  camera/0/cameras.txt
  camera/0/images.txt
  camera/0/points3D.txt
  images/
  masks/
```

Expected Gaussian frame fields include:

```text
means
scales
rotations
harmonics
opacities
```

## Usage

Inspect the Gaussian data:

```bash
python scripts/check_gaussian_data.py
```

Render the processed 448x448 sequence:

```bash
python scripts/render1.py --save-frames
```

Fuse a background model:

```bash
python scripts/fuse_background.py
```

Transform COLMAP intrinsics from the processed square resolution back to the original portrait resolution:

```bash
python scripts/transform_camera.py
```

Render the original-resolution video:

```bash
python scripts/render2.py --save-frames
```

Experimental conservative fusion:

```bash
python scripts/fuse_background_v2.py
python scripts/render2.py --fusion outputs/innovation_v2/fusion.ply --output outputs/innovation_v2/render2.mp4 --cache outputs/innovation_v2/fusion_render2_cache.npz --save-frames
```

## Notes

The renderer is a lightweight CPU-friendly approximation. It projects Gaussian centers and uses DC color coefficients; it is not a full CUDA 3D Gaussian rasterizer.
