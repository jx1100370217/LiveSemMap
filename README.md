<p align="center">
  <h1 align="center">LiveSemMap</h1>
  <h3 align="center">Live Semantic Mapping &amp; Language Navigation</h3>
  <p align="center">建图不停，语义不停，随时可导航。</p>
</p>

**LiveSemMap** 是一个纯视觉的**实时增量语义建图与自然语言导航**系统，基于
[MASt3R-SLAM](README_MASt3R-SLAM.md) 深度扩展：

- **实时增量建图** — MASt3R-SLAM 稠密建图（可选 VIO 位姿先验治漂移），viewer 实时显示 3D 重建与 BEV 俯视图；
- **语义节点增量创建** — 每个新关键帧异步送 VLM（vLLM 服务）标注类别/中文名/描述，
  相邻同类自动聚合成语义节点（门口/路口/电梯间/茶水间/会议室/打印区…），BEV 上实时高亮；
- **随时中断，图都在** — 增量保存线程每 20s 把导航所需产物原子落盘
  （轨迹/占据栅格/BEV/语义 JSON/关键帧图像），关窗口、Ctrl-C 甚至 `kill -9`
  都能得到截止当时的完整可导航地图；
- **自然语言导航** — Web 端在增量建出的地图上导航：打点、自然语言（「去茶水间」）、
  上传图像（SelaVPR++ 重定位）三种方式指定起终点，占据栅格 A* 规划 +
  机器人动画 + 第一人称观察流。

## 快速开始

在 `nav_config.yaml` 里配置一次数据集（内置 insight9 / Mapping_C8 两组配置），然后：

```bash
python main.py                 # 1. 增量语义建图 (随时 Ctrl-C 中断, 产物完整)
python nav_web/export_web.py   # 2. 导出导航 Web 数据
python nav_web/server.py       # 3. 打开 http://localhost:8080 开始导航
```

语义标注依赖 L40 上的 vLLM 服务（`setup/l40_start_vlm.sh`，默认 Qwen3.5-35B-A3B）；
置空 `nav_config.yaml` 的 `semantic_api` 可关闭语义仅跑建图。

## 上游与致谢

SLAM 核心来自 [MASt3R-SLAM](https://edexheim.github.io/mast3r-slam/)
（安装与原版用法见 [README_MASt3R-SLAM.md](README_MASt3R-SLAM.md)）；
VPR 重定位使用 [SelaVPR++](https://github.com/Lu-Feng/SelaVPRplusplus)；
导航策略与 Web 可视化设计参考 VGP-Nav。
