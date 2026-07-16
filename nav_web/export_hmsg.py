#!/usr/bin/env python3
"""把 HMSG 序列化产物导出为网页数据 (离线壳; 在线建图时 OnlineHMSG 会自动
周期写 web/hmsg.js, 本脚本用于对已序列化的 hmsg/ 重新导出)。"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from mast3r_slam.run_config import load_run_config, run_dir, seq_name  # noqa: E402


def main():
    rc = load_run_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(run_dir(rc)))
    ap.add_argument("--seq", default=seq_name(rc))
    args = ap.parse_args()

    from mast3r_slam.hmsg import HMSGGraph
    from mast3r_slam.hmsg.webexport import export_web_data
    run = pathlib.Path(args.run)
    g = HMSGGraph.load(run / "hmsg")
    rooms = [{"id": r.room_id, "name": r.name, "name_zh": r.name_zh,
              "type_zh": r.type_zh, "summary_zh": r.summary_zh,
              "vertices": np.asarray(r.vertices),
              "n_views": sum(1 for v in g.views if v.room_id == r.room_id),
              "n_objects": sum(1 for o in g.objects if o.room_id == r.room_id),
              "rep_feats": [np.asarray(e) for e in r.embeddings]}
             for r in g.rooms if r.vertices is not None]
    objects = [{"id": o.object_id, "room": o.room_id, "name": o.name,
                "name_zh": getattr(o, "name_zh", "") or o.name,
                "pts": np.asarray(o.pcd.points),
                "embedding": (np.asarray(o.embedding, np.float32)
                              if o.embedding is not None else None),
                "best_view": o.best_view_id, "views": o.view_ids}
               for o in g.objects if o.pcd is not None]
    views = [{"id": v.view_id, "room": v.room_id, "img_id": v.img_id,
              "pose": (np.asarray(v.pose) if v.pose is not None else None),
              "desc": v.vlm_description, "objects": v.object_ids}
             for v in g.views]
    export_web_data(args.seq, run / "web", rooms, objects, views,
                    hmsg_dir=run / "hmsg")


if __name__ == "__main__":
    main()
