"""HMSG (Hierarchical Multimodal Scene Graph) — 严格复刻 HoloAgent/FSR-VLN 的
层级多模态场景图 (fsr_vln/memory/hmsg, fork 自 HOV-SG), 作为 LiveSemMap 的语义地图。

五层: building(根) -> Floor -> Room -> {Object, View}
- 层级边: building-floor / floor-room / room-object / room-view
- 拓扑边: view-object (可见性)
构建输入: MASt3R-SLAM 产物 (逐关键帧米制世界系点图 + VIO 位姿 + 关键帧 RGB),
等价替换原版的 posed RGB-D (深度仅用于反投影, 点图=现成的逐像素 3D)。

与原版的明确差异 (均有注释标注):
- 高度轴: 原版 Y-up, 本实现参数化为 Z-up (insight9 VIO 世界系);
- mask 2D->3D: 原版 深度反投影+KDTree 吸附全局点云, 本实现直接布尔索引帧点图
  (同源同坐标, 无需吸附);
- 特征主干: OpenCLIP ViT-L/14 laion2B (与开源代码一致; 论文叙述的 SAM2+SigLIP
  属未开源的 HoloAgent-0 本体, 三描述子权重未公布不可复现);
- LLM/VLM: Azure GPT 全部换 L40 vLLM Qwen (提示词照抄)。
"""
from .graph import HMSGGraph, Floor, Room, Object, View  # noqa: F401
