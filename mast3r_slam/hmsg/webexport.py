"""HMSG 网页数据导出 (在线快照与离线导出共用)。

输入为轻量结构 (numpy/原生类型), 不依赖 o3d:
  rooms:   [{id, name, name_zh, type_zh, summary_zh, vertices(N,2), n_views,
             n_objects}]
  objects: [{id, room, name, name_zh, pts(N,3), embedding(D,)|None,
             best_view, views[list]}]
  views:   [{id, room, img_id, pose(4,4)|None, desc, objects[list]}]
输出: web_dir/hmsg.js (+ 可选 hmsg_dir/query_pack.npz)。
含轴对齐: 视图轨迹走向直方图主轴旋转 + 起点在东 (180 度翻转保手性)。
"""
import json
import pathlib

import numpy as np

from .graph import ROOM_PALETTE


def _axis_rot(views):
    vs = sorted([v for v in views if v.get("pose") is not None],
                key=lambda v: v["img_id"])
    if len(vs) < 8:
        return np.eye(2), 1.0
    V = np.array([[np.asarray(v["pose"])[0, 3], np.asarray(v["pose"])[1, 3]]
                  for v in vs])
    d = np.diff(V, axis=0)
    m = np.linalg.norm(d, axis=1) > 0.05
    if m.sum() < 8:
        return np.eye(2), 1.0
    ang = np.degrees(np.arctan2(d[m, 1], d[m, 0])) % 180
    hist, edges = np.histogram(ang, bins=36, range=(0, 180))
    main = (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2
    th = np.radians(-main)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    flip = -1.0 if (V[0] @ R.T)[0] < 0 else 1.0
    return R, flip


def export_web_data(seq, web_dir, rooms, objects, views, hmsg_dir=None,
                    max_room_pts=4000, floor_meta=None, quiet=False):
    web = pathlib.Path(web_dir)
    web.mkdir(parents=True, exist_ok=True)
    R, flip = _axis_rot(views)

    def rot2(p):
        return np.asarray(p, np.float64) @ R.T * flip

    out_rooms = []
    allv = []
    for i, r in enumerate(rooms):
        v = rot2(np.asarray(r["vertices"], np.float32))
        allv.append(v)
        if len(v) > max_room_pts:
            v = v[np.linspace(0, len(v) - 1, max_room_pts).astype(int)]
        out_rooms.append({"id": r["id"], "name": r.get("name", ""),
                          "name_zh": r.get("name_zh", ""),
                          "type_zh": r.get("type_zh", ""),
                          "summary_zh": r.get("summary_zh", ""),
                          "color": ROOM_PALETTE[i % len(ROOM_PALETTE)],
                          "pts": np.round(v, 2).tolist(),
                          "n_views": r.get("n_views", 0),
                          "n_objects": r.get("n_objects", 0)})
    allv = np.concatenate(allv) if allv else np.zeros((1, 2))
    x0, y0 = allv.min(0)
    x1, y1 = allv.max(0)

    out_objects = []
    for o in objects:
        p = np.asarray(o["pts"])
        pr = rot2(p[:, :2])
        out_objects.append({
            "id": o["id"], "room": o["room"], "name": o["name"],
            "name_zh": o.get("name_zh") or o["name"],
            "c": [round(float(pr[:, 0].mean()), 2),
                  round(float(pr[:, 1].mean()), 2),
                  round(float(p[:, 2].mean()), 2)],
            "bb": [round(float(x), 2) for x in
                   (*pr.min(0), *pr.max(0))],
            "z": [round(float(p[:, 2].min()), 2),
                  round(float(p[:, 2].max()), 2)],
            "n_pts": len(p),
            "best_view": o.get("best_view"), "views": o.get("views", [])})

    out_views = []
    for v in views:
        if v.get("pose") is None:
            continue
        T = np.asarray(v["pose"])
        xy = rot2(T[:2, 3])
        dxy = rot2((T[:3, :3] @ np.array([0, 0, 1.0]))[:2])
        out_views.append({"id": v["id"], "room": v["room"],
                          "img_id": int(v["img_id"]),
                          "x": round(float(xy[0]), 2),
                          "y": round(float(xy[1]), 2),
                          "dx": round(float(dxy[0]), 3),
                          "dy": round(float(dxy[1]), 3),
                          "desc": v.get("desc", ""),
                          "objects": v.get("objects", [])})

    data = {"seq": seq,
            "bounds": [round(float(x0), 2), round(float(y0), 2),
                       round(float(x1), 2), round(float(y1), 2)],
            "floors": floor_meta or [{"id": "0", "zero": 0.0, "height": 0.0,
                                      "rooms": [r["id"] for r in out_rooms]}],
            "rooms": out_rooms, "objects": out_objects, "views": out_views}
    (web / "hmsg.js").write_text(
        "window.HMSG = " + json.dumps(data, ensure_ascii=False,
                                      separators=(",", ":")) + ";",
        encoding="utf-8")
    if not quiet:
        print(f"[hmsg-web] {len(out_rooms)}房间 {len(out_objects)}物体 "
              f"{len(out_views)}视图 -> {web/'hmsg.js'}")

    if hmsg_dir is not None:
        hd = pathlib.Path(hmsg_dir)
        hd.mkdir(parents=True, exist_ok=True)
        feats = [o.get("embedding") for o in objects]
        ok = [i for i, f in enumerate(feats) if f is not None]
        room_feats, room_ids = [], []
        for r in rooms:
            for e in r.get("rep_feats", []):
                room_ids.append(r["id"])
                room_feats.append(np.asarray(e, np.float32))
        np.savez_compressed(
            hd / "query_pack.npz",
            obj_feats=(np.stack([np.asarray(feats[i], np.float32)
                                 for i in ok])
                       if ok else np.zeros((0, 768), np.float32)),
            obj_ids=np.array([objects[i]["id"] for i in ok]),
            obj_names=np.array([objects[i]["name"] for i in ok]),
            obj_rooms=np.array([objects[i]["room"] for i in ok]),
            obj_best_views=np.array([str(objects[i].get("best_view"))
                                     for i in ok]),
            room_rep_feats=(np.stack(room_feats)
                            if room_feats else np.zeros((0, 768), np.float32)),
            room_rep_ids=np.array(room_ids))
