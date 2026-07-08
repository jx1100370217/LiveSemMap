# 本机 (58 / RTX 5090) 搭建与 Mapping_C8 增量建图 说明

本目录是把 MASt3R-SLAM 在本机跑通、并用 VGP-Nav 的 **Mapping_C8** 做增量建图测试时的辅助脚本。
conda env: **`mast3r-slam`** (python 3.11)。

## 脚本
| 脚本 | 作用 |
|---|---|
| `install_all.sh` | 一键安装全部需编译依赖 (curope/asmk/pyimgui/lietorch/后端)，含 5090/gcc-14 适配 |
| `dl_ckpt.sh` | 下载 MASt3R metric + retrieval 权重到 `checkpoints/` |
| `prep_mapping_c8.py` | 用 VGP-Nav 把 Mapping_C8 的 camera_1(MEI鱼眼) 去畸变成 640×480 针孔 png 序列 → `datasets/Mapping_C8/`，并写 `config/intrinsics_c8.yaml` (在 internvla 环境跑) |
| `render_incremental.py` | 把增量快照 (`logs/<save_as>/snapshots/snap_*.ply` + `centers_*.npy`) 渲染成增量建图视频/GIF |

## 针对 RTX 5090 (sm_120) 的关键适配
README 推荐的 torch 2.5.1 不支持 sm_120，故：
- torch **2.8.0+cu128** + conda **cuda-toolkit 12.8** (系统无 nvcc)
- 系统 gcc 是 15，CUDA 12.8 要 gcc ≤14 → conda 装 gcc-14 + 根目录 `.gcc14/` shim + `NVCC_PREPEND_FLAGS=-ccbin`
- `CPATH` 指向 conda CUDA 头 (lietorch 找 `cuda.h`)；`TORCH_CUDA_ARCH_LIST=9.0;12.0`
- 全程 `pip install --no-build-isolation`

### torch 2.8 源码补丁 (已改，勿回退)
- `setup.py`: gencode 加 `sm_120`
- `mast3r_slam/backend/src/gn_kernels.cu`: `torch::linalg::linalg_norm` → `at::linalg_norm` (3处)
- `mast3r_slam/backend/src/matching_kernels.cu` 与 `thirdparty/mast3r/dust3r/croco/models/curope/kernels.cu`: `.type()` → `.scalar_type()`
- `thirdparty/mast3r/mast3r/model.py`、`retrieval/processor.py`、`retrieval/model.py`: `torch.load(...)` 加 `weights_only=False`

## 运行
`main.py` 已默认 Mapping_C8 数据+配置，直接：
```bash
conda run -n mast3r-slam python main.py            # 默认 datasets/Mapping_C8 + config/mapping_c8.yaml + intrinsics_c8.yaml
```
- 加 `--no-viz` 无头运行；加 `--snapshot-every 3` 存增量快照。
- 单线程无头评测：`--config config/eval_calib.yaml`。
- 复现增量视频：
  ```bash
  conda run -n mast3r-slam python main.py --no-viz --snapshot-every 3 --save-as c8_incremental
  conda run -n mast3r-slam python setup/render_incremental.py logs/c8_incremental/snapshots logs/c8_incremental/incremental_mapping
  ```

产物在 `logs/<save_as>/`：`Mapping_C8.txt`(轨迹) / `Mapping_C8.ply`(稠密重建) / `snapshots/` / `incremental_mapping.{gif,mp4}`。

## 实时 viewer 的 GL 依赖
带 viewer 运行(不加 `--no-viz`)需要 moderngl 加载 `libGL.so`/`libEGL.so`，但系统只装了 `.so.1`(缺无版本号软链, 无 -dev 包)。`install_all.sh` 已在 `$ENV/lib/` 建软链指向系统运行库(env python RPATH 可找到)——若报 `OSError: libEGL.so 无法打开`，重建这些软链即可。viewer 需桌面(DISPLAY=:0)；SSH 无显示时用 `--no-viz`。实测 GL_RENDERER=RTX 5090 / GL 4.6。

> 注：MASt3R-SLAM 是 Sim(3)，全局尺度为任意尺度；用于度量导航需外部先验(地面高/IMU)锚定。
