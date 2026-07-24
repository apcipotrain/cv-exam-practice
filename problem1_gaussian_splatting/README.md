# 第一题：Gaussian Splatting 渲染练习

本目录是第一题的脱敏实现，主要处理逐帧 3D Gaussian 数据、前景人物 mask、COLMAP 格式相机参数、背景 Gaussian 融合，以及原始分辨率视频渲染。

原始题目材料和数据集不包含在仓库中。

## 数据目录

请把自己的输入数据放到：

```text
data/input/
  per_frame_gaussians.pt
  camera/0/cameras.txt
  camera/0/images.txt
  camera/0/points3D.txt
  images/
  masks/
```

每帧 Gaussian 数据预期包含以下字段：

```text
means
scales
rotations
harmonics
opacities
```

## 运行方式

检查 Gaussian 数据结构：

```bash
python scripts/check_gaussian_data.py
```

渲染处理后 `448x448` 序列：

```bash
python scripts/render1.py --save-frames
```

融合背景 Gaussian 模型：

```bash
python scripts/fuse_background.py
```

将处理后正方形分辨率下的 COLMAP 相机内参转换回原始竖屏分辨率：

```bash
python scripts/transform_camera.py
```

渲染原始分辨率视频：

```bash
python scripts/render2.py --save-frames
```

实验性的保守融合版本：

```bash
python scripts/fuse_background_v2.py
python scripts/render2.py --fusion outputs/innovation_v2/fusion.ply --output outputs/innovation_v2/render2.mp4 --cache outputs/innovation_v2/fusion_render2_cache.npz --save-frames
```

## 说明

当前渲染器是一个 CPU 友好的轻量近似实现。它主要投影 Gaussian 中心点，并使用 DC 颜色系数进行渲染；它不是完整的 CUDA 版 3D Gaussian 光栅化器。
