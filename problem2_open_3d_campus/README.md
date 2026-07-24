# 题目二：校园无人机三视图可视化

本目录是题目二的脱敏实现版本。目标是根据题目提供的校园俯视图和路线参考，生成包含 `PRIMARY`、`COMPANION`、`BEV` 三个视图的无人机飞行视频。

## 目录说明

- `scripts/`：主要 Python 脚本。
  - `10_render_abstract_scene.py`：最终使用的抽象场景渲染脚本。
  - `02_extract_route.py`、`08_refine_route_away_from_black.py`：路线提取与修正相关脚本。
  - `05_extract_scene_from_jpg.py`、`06_export_frame_contexts.py`、`07_render_from_frame_contexts.py`、`09_refine_route_to_roads.py`：视觉识别和调试尝试脚本，最终效果有限，仅作为过程记录。
- `docs/`：需求、调试记录和抽象建模说明。
- `data/annotations/refined_route_points.json`：最终使用的路线点。
- `deliverables/`：题目要求的交付材料。
  - `video.mp4`
  - `readme.txt`
  - `自评表.txt`

## 运行方式

在本目录下运行：

```bash
python scripts/10_render_abstract_scene.py --frames 2000 --fps 30 --output outputs/video.mp4 --frame-dir outputs/frames_all --save-all-frames
```

脚本会生成三视图视频，分辨率为 `2400x800`，每个子视图为 `800x800`。

## 完成度说明

本题结果明确标记为“未完成”。当前版本实现了基本的视频生成流程，但主视图和伴随视图中的校园内容主要依赖抽象建模，并没有可靠完成对真实 JPG 中建筑、道路、树木、水体等语义信息的精确识别。
