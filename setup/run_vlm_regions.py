#!/usr/bin/env python3
"""VLM 区域生长语义地图 — 数据集直驱测试 (吃 datasets/<seq> 原始数据,
不依赖建图产物; 足迹用相机轨迹圆盘, 在线接入 SLAM 后换点云足迹)。

用法: python setup/run_vlm_regions.py --dataset datasets/cfds_floor28
      python setup/run_vlm_regions.py --max-frames 30      # 冒烟
"""
import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from mast3r_slam.run_config import load_run_config  # noqa: E402


def main():
    rc = load_run_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/cfds_floor28")
    ap.add_argument("--api", default=rc.get("semantic_api",
                                            "http://192.168.50.72:8299/v1"))
    ap.add_argument("--model", default=rc.get("semantic_model",
                                              "qwen3.5-35b-a3b"))
    ap.add_argument("--thin", type=float, default=0.6, help="抽稀间距(米)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ds = pathlib.Path(args.dataset)
    seq = ds.name
    out = pathlib.Path(args.out) if args.out else \
        pathlib.Path("logs") / f"vlm_region_{seq}"
    out.mkdir(parents=True, exist_ok=True)

    # 位姿: vio + timestamps 就近配对 (全数据帧)
    from scipy.spatial.transform import Rotation
    ts = np.loadtxt(ds / "timestamps.txt")
    vio = np.loadtxt(ds / "vio.txt")
    n_frames = len(ts)
    poses = []
    for i in range(n_frames):
        t = ts[i, 1]
        j = int(np.clip(np.searchsorted(vio[:, 0], t), 1, len(vio) - 1))
        if abs(vio[j - 1, 0] - t) < abs(vio[j, 0] - t):
            j -= 1
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(vio[j, 4:8]).as_matrix()
        T[:3, 3] = vio[j, 1:4]
        poses.append(T)
    # 抽稀 (VIO 距离)
    sel, last = [], None
    for i in range(n_frames):
        p = poses[i][:2, 3]
        if last is None or np.linalg.norm(p - last) >= args.thin:
            sel.append(i)
            last = p
    if args.max_frames:
        sel = sel[:args.max_frames]
    print(f"[vlm-region] {seq}: {n_frames} 帧 -> 抽稀 {len(sel)} 帧 "
          f"(间距 {args.thin}m), 输出 {out}")

    from mast3r_slam.vlm_region import VLMRegionEngine
    eng = VLMRegionEngine(ds / "surround", ds, out / "web", seq,
                          args.api, args.model)
    t0 = time.time()
    for n, i in enumerate(sel):
        eng.process_frame(kf_idx=i, fid=i, pose=poses[i])
        if n % 20 == 19:
            el = time.time() - t0
            print(f"[vlm-region] {n+1}/{len(sel)} 帧, {el:.0f}s "
                  f"({el/(n+1):.1f}s/帧), 区域 {len(eng.regions)}", flush=True)
    eng.finalize(out)
    print(f"[vlm-region] 总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
