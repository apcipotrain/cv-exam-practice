# CV Exam Practice

This repository contains computer vision practice projects based on anonymized exam-style tasks from an unnamed university.

Original task statements, datasets, generated videos, and large Gaussian/PLY files are not included.  The repository focuses on reusable code structure, data inspection, camera conversion, background Gaussian filtering/fusion, and lightweight rendering workflows.

## Projects

- `problem1_gaussian_splatting`: per-frame 3D Gaussian data inspection, mask-guided background fusion, COLMAP camera conversion, and MP4 rendering.
- `problem2_placeholder`: reserved for the second task.

## Repository Policy

This public version intentionally excludes:

- original exam PDFs or course materials
- raw datasets
- generated `.pt`, `.ply`, `.mp4`, `.npz`, and frame images
- school, teacher, or course-identifying information

## Setup

```bash
pip install -r requirements.txt
```

See each problem folder for its own data layout and usage notes.
