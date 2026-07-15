#!/usr/bin/env python3
"""离线重放房间拓扑: 用已存 {seq}_semantic.json 的逐帧标注喂 RoomTopoBuilder,
重写 {seq}_rooms.json —— 调切分/合并参数无需重跑 SLAM 与逐帧 VLM 标注。

注意: 标注须含空间结构字段 (space_kind/at_transition/...), 旧版标注请先跑
setup/relabel_semantic.py 重标 (新 prompt 会自动产出这些字段)。

用法: python setup/rebuild_rooms.py               # 数据集取自 nav_config.yaml
      python setup/rebuild_rooms.py --vlm          # 房间定稿也过 VLM 仲裁
      python setup/rebuild_rooms.py --trans-on 0.9 --merge-dist 2.5
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from mast3r_slam.run_config import load_run_config, run_dir, seq_name  # noqa: E402
from mast3r_slam.room_topo import RoomTopoBuilder  # noqa: E402


def main():
    rc = load_run_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(run_dir(rc)))
    ap.add_argument("--seq", default=seq_name(rc))
    ap.add_argument("--dataset", default=rc.get("dataset", "datasets/cfds_floor28"))
    ap.add_argument("--api", default=rc.get("semantic_api", ""))
    ap.add_argument("--model", default=rc.get("semantic_model", "qwen3.5-35b-a3b"))
    ap.add_argument("--vlm", action="store_true",
                    help="房间 close 时调 VLM 定稿 (默认只用段内投票, 重放零 GPU)")
    ap.add_argument("--trans-on", type=float, default=0.75)
    ap.add_argument("--trans-off", type=float, default=0.3)
    ap.add_argument("--merge-dist", type=float, default=3.0)
    ap.add_argument("--confirm-hits", type=int, default=3)
    args = ap.parse_args()

    run = pathlib.Path(args.run)
    sem = json.loads((run / f"{args.seq}_semantic.json").read_text())
    ann = {int(k): v for k, v in sem["annotations"].items()}
    n_new = sum(1 for a in ann.values() if "space_kind" in a)
    print(f"[rooms] {len(ann)} 条标注 ({n_new} 条含空间结构字段)")
    if n_new == 0:
        sys.exit("[rooms] 标注全部是旧版(无空间结构字段), 无法切房间 —— "
                 "先跑 setup/relabel_semantic.py 重标")

    # frame_id -> VIO 位置 (semantic.json 按 kf 存位置与 frame_id, 转成 fid 索引)
    kf_pos = np.asarray(sem["kf_positions"], np.float64)
    fid2pos = {int(f): kf_pos[i] for i, f in enumerate(sem["frame_ids"])
               if np.isfinite(kf_pos[i]).all()}

    b = RoomTopoBuilder(
        pos_fn=lambda fid: fid2pos.get(int(fid)),
        api_url=args.api if args.vlm else "", model=args.model,
        surround_dir=pathlib.Path(args.dataset) / "surround",
        trans_on=args.trans_on, trans_off=args.trans_off,
        merge_dist=args.merge_dist, confirm_hits=args.confirm_hits)
    b.tick(ann, min_inflight=1 << 30)   # 一次按 kf 序重放全部
    b.finalize()
    b.save(run / f"{args.seq}_rooms.json")
    from mast3r_slam.room_topo import render_rooms_png
    render_rooms_png(run, args.seq)

    snap = b.snapshot()
    live = [r for r in snap["rooms"] if r["status"] != "merged"]
    print(f"\n[rooms] {len(live)} 房间 / {len(snap['edges'])} 边:")
    for r in live:
        print(f"  #{r['id']:3d} {r['room_type'] or '?':12s} {r['name']:16s} "
              f"kfs={len(r['kf_indices']):3d} 标识={','.join(r['signage'][:3])}")
    for e in snap["edges"]:
        print(f"  边 #{e['a']} <-> #{e['b']}  [{e['kind']}] "
              f"{','.join(e['signage'][:2])}")


if __name__ == "__main__":
    main()
