#!/usr/bin/env python3
"""把 MASt3R-SLAM 语义建图产物导出为导航 Web 应用数据。

输入 (logs/<save_as>/ 下, 由 main.py 退出/中断时保存):
  {seq}_occupancy.npz   占据栅格 + 世界<->像素 meta + 关键帧位置/像素坐标 + frame_ids
  {seq}_semantic.json   语义节点 + 逐关键帧标注
  {seq}_vpr_desc.npy    SelaVPR 关键帧描述子 (可选, server 用)
输出 (logs/<save_as>/web/):
  data.js               window.NAVDATA 单文件几何数据 (参考 VGP-Nav 设计)
  thumbs/kf{i}.jpg      关键帧缩略图 (从数据集原图生成, 360px 宽)

用法: python nav_web/export_web.py --run logs/semantic_v1 --seq insight9 --dataset datasets/insight9
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
    ap.add_argument("--run", default=str(run_dir(rc)), help="运行产物目录, 如 logs/insight9_run")
    ap.add_argument("--seq", default=seq_name(rc), help="序列名, 如 insight9")
    ap.add_argument("--dataset", default=rc.get("dataset", "datasets/insight9"),
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

    # 每个关键帧的标注摘要 (FPV 面板显示当前位置语义)
    kf_ann = {}
    for k, v in sem.get("annotations", {}).items():
        kf_ann[int(k)] = {"cat": v["category"], "name": v.get("name", ""),
                          "conf": round(v.get("confidence", 0), 2)}

    # 缩略图: 从数据集原图按 frame_id 取。frame_id 是数据集内索引(RGBFiles 按
    # natsorted 顺序编号), 直接对 natsorted 文件列表按索引取, 兼容任意命名格式
    # (insight9 的 000123.png / Mapping_C8 的 frame_00123.png)。
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

    # grid 压成每行字符串 ('0'/'1'/'2'), gzip 由 HTTP 层做
    grid_rows = ["".join(map(str, row.tolist())) for row in grid]

    data = {
        "seq": args.seq,
        "coordinate": coord,             # vio(米制) / slam(相对尺度)
        "G": G,
        "m_per_px": round(m_per_px, 5),
        "grid": grid_rows,
        "kf": [{"px": [round(float(x), 1), round(float(y), 1)], "fid": int(f)}
               for (x, y), f in zip(kf_px, frame_ids)],
        "kf_ann": kf_ann,
        "nodes": nodes,
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
