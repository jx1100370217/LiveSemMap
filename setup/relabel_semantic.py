#!/usr/bin/env python3
"""离线重标语义: 对已建图产物重新跑 4 环视 VLM 标注 + 节点聚合, 重写 semantic.json。

用于调整语义 prompt/聚合参数后快速迭代 —— 不重跑 SLAM 建图 (关键帧列表与位置
取自 <seq>_occupancy.npz), 只重刷 <seq>_semantic.json。之后重跑 nav_web/export_web.py
即可在导航 Web 中生效 (BEV png 上的旧节点圈不会更新, 仅重新建图时才刷新, 不影响导航)。

用法: python setup/relabel_semantic.py            # 数据集取自 nav_config.yaml
      python setup/relabel_semantic.py --run logs/cfds_floor28_run --seq cfds_floor28 \\
             --dataset datasets/cfds_floor28
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from mast3r_slam.run_config import load_run_config, run_dir, seq_name  # noqa: E402
from mast3r_slam.semantic import SemanticAnnotator, aggregate_nodes  # noqa: E402


def main():
    rc = load_run_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(run_dir(rc)))
    ap.add_argument("--seq", default=seq_name(rc))
    ap.add_argument("--dataset", default=rc.get("dataset", "datasets/cfds_floor28"))
    ap.add_argument("--api", default=rc.get("semantic_api", "http://192.168.50.72:8299/v1"))
    ap.add_argument("--model", default=rc.get("semantic_model", "qwen3.5-35b-a3b"))
    ap.add_argument("--reuse", action="store_true",
                    help="复用现有 semantic.json 的逐帧标注, 只重跑节点聚合 (调聚合参数用)")
    ap.add_argument("--thin-dist", type=float, default=0.4,
                    help="空间抽稀: 距上一提交帧位移<该值(米)的关键帧跳过标注; 0=全量")
    args = ap.parse_args()

    run = pathlib.Path(args.run)
    z = np.load(run / f"{args.seq}_occupancy.npz")
    frame_ids = z["frame_ids"].tolist()
    kf_pos = z["kf_pos"]
    coord = str(z["coordinate"])

    if args.reuse:
        old = json.loads((run / f"{args.seq}_semantic.json").read_text())
        ann = {int(k): v for k, v in old["annotations"].items()}
        print(f"[relabel] 复用已有标注 {len(ann)} 条, 仅重聚合")
    else:
        surround = pathlib.Path(args.dataset) / "surround"
        assert surround.is_dir(), f"缺环视图目录 {surround}"
        print(f"[relabel] {len(frame_ids)} 个关键帧, 环视图 {surround}, VLM {args.api}")
        ann = {}
        a = SemanticAnnotator(args.api, ann, surround, model=args.model,
                              min_dist=args.thin_dist)
        n_sub = 0
        for i, fid in enumerate(frame_ids):
            n_sub += a.submit_thinned(i, fid, kf_pos[i])
        print(f"[relabel] 空间抽稀(thin_dist={args.thin_dist}m): "
              f"提交 {n_sub}/{len(frame_ids)} 帧")
        a.drain()
        assert not a.disabled, "VLM 服务连续失败, 已中止 (检查 L40 vLLM)"
        print(f"[relabel] 标注完成 {len(ann)}/{n_sub}")

    pos_by_kf = {i: kf_pos[i] for i in range(len(frame_ids))
                 if np.isfinite(kf_pos[i]).all()}
    nodes = aggregate_nodes(dict(ann), pos_by_kf)

    out = run / f"{args.seq}_semantic.json"
    tmp = out.with_name("tmp_" + out.name)
    with open(tmp, "w") as f:
        json.dump({
            "coordinate": coord,
            "nodes": nodes,
            "annotations": {str(k): v for k, v in sorted(ann.items())},
            "kf_positions": kf_pos.tolist(),
            "frame_ids": frame_ids,
        }, f, ensure_ascii=False, indent=1)
    tmp.replace(out)

    print(f"\n[relabel] {len(nodes)} 个语义节点 -> {out}")
    for n in nodes:
        print(f"  {n['category']:12s} {n['name']:16s} conf={n['confidence']:.2f} "
              f"kfs={len(n['kf_indices'])}")
    print("\n下一步: python nav_web/export_web.py 重导出导航数据")


if __name__ == "__main__":
    main()
