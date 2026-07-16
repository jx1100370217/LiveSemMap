#!/usr/bin/env python3
"""离线构建 HMSG 层级多模态场景图 (严格按 HoloAgent/FSR-VLN 规格)。

前置: 建图产物含 {seq}_kf_pointmaps.npz (main.py save step 6)。
用法: python setup/build_hmsg.py                     # 数据集取自 nav_config.yaml
      python setup/build_hmsg.py --max-frames 40     # 小样冒烟
产物: logs/<run>/hmsg/ (full_pcd.ply + graph/{floors,rooms,objects,views})
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from mast3r_slam.run_config import load_run_config, run_dir, seq_name  # noqa: E402


def main():
    rc = load_run_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(run_dir(rc)))
    ap.add_argument("--seq", default=seq_name(rc))
    ap.add_argument("--dataset", default=rc.get("dataset", "datasets/cfds_floor28"))
    ap.add_argument("--stride", type=int, default=1, help="关键帧抽样步长")
    ap.add_argument("--max-frames", type=int, default=0, help=">0 只取前 N 帧 (冒烟)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--zh-api", default=rc.get("semantic_api", ""),
                    help="Qwen 中文化 (物体翻译+区域命名); 空串跳过")
    ap.add_argument("--zh-model", default=rc.get("semantic_model", "qwen3.5-35b-a3b"))
    args = ap.parse_args()

    from mast3r_slam.hmsg.build import build_hmsg
    g, out = build_hmsg(args.run, args.seq, args.dataset, device=args.device,
                        stride=args.stride, max_frames=args.max_frames,
                        zh_api=args.zh_api, zh_model=args.zh_model)

    print("\n=== HMSG 摘要 ===")
    for f in g.floors:
        print(f"Floor {f.floor_id}: zero={f.floor_zero_level:.2f} "
              f"height={f.floor_height:.2f} rooms={len(f.rooms)}")
    for r in g.rooms:
        print(f"  Room {r.room_id} [{r.name}] views={len(r.views)} "
              f"objects={len(r.objects)} rep={len(r.embeddings)}")
    from collections import Counter
    cnt = Counter(o.name for o in g.objects)
    print(f"物体 top15: {cnt.most_common(15)}")
    print(f"视图: {len(g.views)}, 边(含拓扑): {g.graph.number_of_edges()}")


if __name__ == "__main__":
    main()
