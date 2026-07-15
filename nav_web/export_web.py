#!/usr/bin/env python3
"""把 MASt3R-SLAM 语义建图产物导出为导航 Web 应用数据。

输入 (logs/<save_as>/ 下, 由 main.py 退出/中断时保存):
  {seq}_occupancy.npz   占据栅格 + 世界<->像素 meta + 关键帧位置/像素坐标 + frame_ids
  {seq}_semantic.json   语义节点 + 逐关键帧标注
  {seq}_vpr_desc.npy    SelaVPR 关键帧描述子 (可选, server 用)
输出 (logs/<save_as>/web/):
  data.js               window.NAVDATA 单文件几何数据 (参考 VGP-Nav 设计)
  thumbs/kf{i}.jpg      关键帧缩略图 (从数据集原图生成, 360px 宽)

用法: python nav_web/export_web.py --run logs/cfds_floor28_run --seq cfds_floor28 --dataset datasets/cfds_floor28
"""
import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from mast3r_slam.semantic import SEMANTIC_CATEGORIES  # noqa: E402
from mast3r_slam.run_config import load_run_config, run_dir, seq_name  # noqa: E402


def main():
    rc = load_run_config()  # 默认值取自 nav_config.yaml, CLI 参数优先
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(run_dir(rc)), help="运行产物目录, 如 logs/cfds_floor28_run")
    ap.add_argument("--seq", default=seq_name(rc), help="序列名, 如 cfds_floor28")
    ap.add_argument("--dataset", default=rc.get("dataset", "datasets/cfds_floor28"),
                    help="数据集目录(取原图做缩略图)")
    ap.add_argument("--thumb-width", type=int, default=360)
    args = ap.parse_args()
    print(f"[export_web] run={args.run} seq={args.seq} dataset={args.dataset}")

    run = pathlib.Path(args.run)
    ds = pathlib.Path(args.dataset)
    web = run / "web"
    (web / "thumbs").mkdir(parents=True, exist_ok=True)

    z = np.load(run / f"{args.seq}_occupancy.npz")
    grid = z["grid"]                      # (G,G) uint8 0未知/1free/2障碍
    meta = json.loads(str(z["meta"]))
    kf_px = z["kf_px"]                    # (N,2) 像素坐标
    kf_pos = z["kf_pos"]                  # (N,3) 世界
    frame_ids = z["frame_ids"].tolist()
    coord = str(z["coordinate"])
    sem = json.loads((run / f"{args.seq}_semantic.json").read_text())

    G = meta["G"]
    m_per_px = 2 * meta["half"] / G       # VIO 系=真实米; SLAM 系=SLAM 单位

    # 语义节点 -> 像素坐标 (占据 npz 的 meta 与 semantic.json 同一次保存, 直接映射)
    a, b, half = meta["a"], meta["b"], meta["half"]
    def to_px(p3):
        px = (p3[a] - (meta["ca"] - half)) / (2 * half) * G
        py = (p3[b] - (meta["cb"] - half)) / (2 * half) * G
        return [round(float(px), 1), round(float(G - 1 - py), 1)]

    nodes = []
    for i, n in enumerate(sem["nodes"]):
        cn, color, _ = SEMANTIC_CATEGORIES.get(n["category"], ("?", (.8, .8, .8), False))
        nodes.append({
            "id": i, "category": n["category"], "cat_zh": cn,
            "name": n["name"], "desc": n["description"],
            "conf": round(n["confidence"], 2),
            "px": to_px(np.asarray(n["position"])),
            "rep_kf": n["rep_kf"], "kfs": n["kf_indices"],
            "color": "#%02x%02x%02x" % tuple(int(c * 255) for c in color),
        })

    # 房间平面布局 (rooms.json 可选): 可行走区按房间测地划分的行字符串 +
    # 标签锚点 + 门位, 供前端"房间图"独立视图画 HOV-SG 式楼层平面布局
    rooms_js = None
    rp = run / f"{args.seq}_rooms.json"
    if rp.exists():
        from mast3r_slam.room_topo import (ROOM_PALETTE, assign_room_regions,
                                           room_label_cells)
        rj = json.loads(rp.read_text())
        live = [r for r in rj["rooms"] if r["status"] != "merged"]
        if live:
            px_per_m = G / (2 * meta["half"])
            region = assign_room_regions(
                grid, kf_px, live,
                corridor_px=max(2, int(round(0.5 * px_per_m))),
                jump_px=max(10, int(3.0 * px_per_m)))
            anchors = room_label_cells(region, len(live))
            digits = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            region_rows = ["".join("." if v < 0 or v >= len(digits) else digits[v]
                                   for v in row) for row in region]
            rooms = []
            for i, r in enumerate(live):
                zh = SEMANTIC_CATEGORIES.get(r["room_type"] or "other", ("?",))[0]
                apx = anchors.get(i)
                rooms.append({
                    "id": r["id"], "name": r["name"], "type": r["room_type"],
                    "zh": zh, "desc": r["description"],
                    "color": ROOM_PALETTE[i % len(ROOM_PALETTE)],
                    "apx": ([round(apx[0], 1), round(apx[1], 1)]
                            if apx else None),
                    "n_kfs": len(r["kf_indices"]), "sig": r["signage"][:4],
                })
            def _via_px(e):
                v = e.get("via_kf", -1)
                if 0 <= v < len(kf_px):
                    return [round(float(kf_px[v][0]), 1),
                            round(float(kf_px[v][1]), 1)]
                return None
            rooms_js = {"rooms": rooms, "region": region_rows,
                        "edges": [{"a": e["a"], "b": e["b"], "kind": e["kind"],
                                   "via_px": _via_px(e)}
                                  for e in rj["edges"]]}
            print(f"[export_web] 房间平面布局: {len(rooms)} 房间, {len(rj['edges'])} 边")

    # 每个关键帧的标注摘要 (FPV 面板显示当前位置语义)
    kf_ann = {}
    for k, v in sem.get("annotations", {}).items():
        kf_ann[int(k)] = {"cat": v["category"], "name": v.get("name", ""),
                          "conf": round(v.get("confidence", 0), 2)}

    # 关键帧采集朝向 (地图像素系单位向量): VIO 姿态的相机 z 轴(前向)投影到 BEV 平面。
    # 供 FPV 选帧做方向过滤 —— 来回走过的走廊上只选与行进方向同向拍摄的帧, 消除"倒走感"。
    # 纯 RGB 数据集(无 vio.txt)不输出 dir, 前端自动退回纯距离选帧。
    kf_dir = [None] * len(frame_ids)
    vio_path, ts_path = ds / "vio.txt", ds / "timestamps.txt"
    if vio_path.exists() and ts_path.exists():
        from scipy.spatial.transform import Rotation
        vio = np.loadtxt(vio_path)
        ts = np.loadtxt(ts_path)
        for i, fid in enumerate(frame_ids):
            t = ts[min(int(fid), len(ts) - 1), 1]
            j = int(np.clip(np.searchsorted(vio[:, 0], t), 1, len(vio) - 1))
            if abs(vio[j - 1, 0] - t) < abs(vio[j, 0] - t):
                j -= 1
            fwd = Rotation.from_quat(vio[j, 4:8]).as_matrix()[:, 2]  # 相机光学系 z=前
            v = np.array([fwd[a], -fwd[b]])  # 与 to_px 同一像素系 (屏幕 y 向下)
            nv = np.linalg.norm(v)
            if nv > 1e-6:
                kf_dir[i] = [round(float(v[0] / nv), 3), round(float(v[1] / nv), 3)]
        n_dir = sum(1 for d in kf_dir if d)
        print(f"[export_web] 关键帧朝向 {n_dir}/{len(frame_ids)} (FPV 方向过滤)")

    # 缩略图: 从数据集原图按 frame_id 取。frame_id 是数据集内索引(RGBFiles 按
    # natsorted 顺序编号), 直接对 natsorted 文件列表按索引取, 兼容任意命名格式
    # (cfds_floor28 的 000123.png / Mapping_C8 的 frame_00123.png)。
    from natsort import natsorted
    all_pngs = natsorted(ds.glob("*.png"))
    n_thumb = 0
    for i, fid in enumerate(frame_ids):
        src = all_pngs[fid] if fid < len(all_pngs) else None
        if src is None:
            continue
        img = cv2.imread(str(src))
        h, w = img.shape[:2]
        tw = args.thumb_width
        th = int(h * tw / w)
        cv2.imwrite(str(web / "thumbs" / f"kf{i}.jpg"),
                    cv2.resize(img, (tw, th)),
                    [cv2.IMWRITE_JPEG_QUALITY, 82])
        n_thumb += 1

    # VPR 描述子兜底: 建图被中断时(增量保存只写轻产物)描述子缺失,
    # 这里从数据集原图按 frame_ids 自动补提取, 保证图像定位起点功能可用
    desc_path = run / f"{args.seq}_vpr_desc.npy"
    if not desc_path.exists():
        print(f"[export_web] 缺 {desc_path.name}, 自动补提取 SelaVPR 描述子...")
        try:
            import torch
            from natsort import natsorted as _ns
            from mast3r_slam.selavpr import SelaVPRExtractor
            ex = SelaVPRExtractor(backbone="dinov2-large", use_hashing=False,
                                  use_rerank=False, device="cuda:0")
            if isinstance(ex.model, torch.nn.DataParallel):
                ex.model = ex.model.module.to("cuda:0")
            pngs = _ns(ds.glob("*.png"))
            bgrs = [cv2.imread(str(pngs[f])) for f in frame_ids if f < len(pngs)]
            out = []
            for i in range(0, len(bgrs), 12):
                out.append(ex.extract_batch(bgrs[i:i + 12]))
            D = np.concatenate(out, 0).astype(np.float32)
            D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-9
            np.save(desc_path, D)
            print(f"[export_web] 描述子 {D.shape} -> {desc_path.name}")
        except Exception as e:
            print(f"[export_web] 补提取失败(图像定位功能不可用, 其余不受影响): {e}")

    # grid 压成每行字符串 ('0'/'1'/'2'), gzip 由 HTTP 层做
    grid_rows = ["".join(map(str, row.tolist())) for row in grid]

    data = {
        "seq": args.seq,
        "coordinate": coord,             # vio(米制) / slam(相对尺度)
        "G": G,
        "m_per_px": round(m_per_px, 5),
        "grid": grid_rows,
        "kf": [{"px": [round(float(x), 1), round(float(y), 1)], "fid": int(f),
                **({"dir": d} if d else {})}
               for (x, y), f, d in zip(kf_px, frame_ids, kf_dir)],
        "kf_ann": kf_ann,
        "nodes": nodes,
        "rooms": rooms_js,               # 房间图层数据 (无 rooms.json 时为 null)
        "categories": {k: {"zh": v[0],
                           "color": "#%02x%02x%02x" % tuple(int(c * 255) for c in v[1]),
                           "landmark": v[2]}
                       for k, v in SEMANTIC_CATEGORIES.items()},
    }
    (web / "data.js").write_text(
        "window.NAVDATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";",
        encoding="utf-8")
    kb = (web / "data.js").stat().st_size / 1024
    print(f"[export_web] {len(nodes)} 语义节点, {len(frame_ids)} kf, {n_thumb} 缩略图, "
          f"data.js {kb:.0f}KB -> {web}")


if __name__ == "__main__":
    main()
