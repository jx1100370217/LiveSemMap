#!/usr/bin/env python3
"""对已构建的 HMSG 做 Qwen 中文化后处理 (免重跑 SAM/CLIP 构建):
物体词表英->中 + 每个区域由 Qwen 总结最佳中文名/房型/摘要。

用法: python setup/localize_hmsg.py            # 数据集取自 nav_config.yaml
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
    ap.add_argument("--api", default=rc.get("semantic_api",
                                            "http://192.168.50.72:8299/v1"))
    ap.add_argument("--model", default=rc.get("semantic_model", "qwen3.5-35b-a3b"))
    args = ap.parse_args()

    from mast3r_slam.hmsg import HMSGGraph
    from mast3r_slam.hmsg.qwen_zh import localize_graph
    hdir = pathlib.Path(args.run) / "hmsg"
    g = HMSGGraph.load(hdir)
    print(f"[hmsg-zh] 已加载: {len(g.rooms)}房间 {len(g.objects)}物体")
    localize_graph(g, args.dataset, args.run, args.seq, args.api, args.model)
    g.save(hdir)
    print("[hmsg-zh] 中文化完成, 下一步: python nav_web/export_hmsg.py")


if __name__ == "__main__":
    main()
