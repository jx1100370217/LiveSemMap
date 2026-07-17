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
        rep_ann = sem.get("annotations", {}).get(str(n["rep_kf"]), {})
        nodes.append({
            "id": i, "category": n["category"], "cat_zh": cn,
            "name": n["name"], "desc": n["description"],
            "conf": round(n["confidence"], 2),
            "px": to_px(np.asarray(n["position"])),
            "rep_kf": n["rep_kf"], "kfs": n["kf_indices"],
            "objects": (rep_ann.get("objects") or [])[:8],
            "signage": [s for s in (rep_ann.get("signage") or []) if s][:4],
            "color": "#%02x%02x%02x" % tuple(int(c * 255) for c in color),
        })

    # 语义区域底图层 (VLM 区域生长): 区域格子归属行字符串 + 区域表 ——
    # 导航页底图由几何占据图换成语义区域图
    regions_js, region_rows = None, None
    vr = run / f"{args.seq}_vlm_regions.json"
    hj = run / "web" / "hmsg.js"
    cells_by = {}      # 区域顺序键 -> {name, (N,2) 原始系米制点}
    if vr.exists():
        vj = json.loads(vr.read_text())
        for r in vj.get("regions", []):
            if r.get("cells"):
                nm = r.get("name") or r.get("kind") or ""
                cells_by[str(r["id"])] = (nm, np.asarray(r["cells"]))
    if not cells_by and hj.exists():
        # 兼容旧产物: hmsg.js 的区域点是轴对齐旋转系, 用 kf 轨迹 Procrustes
        # 恢复旋转(严格线性), 逆变换回原始 VIO 系
        hd = json.loads(hj.read_text()[len("window.HMSG = "):-1])
        fid2raw = {int(f): kf_pos[i][[a, b]]
                   for i, f in enumerate(frame_ids)}
        A, B = [], []
        for v in hd.get("views", []):
            if int(v["img_id"]) in fid2raw:
                A.append([v["x"], v["y"]])
                B.append(fid2raw[int(v["img_id"])])
        if len(A) >= 8:
            A, B = np.asarray(A, np.float64), np.asarray(B, np.float64)
            ca, cb = A.mean(0), B.mean(0)
            U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
            Rm = (U @ Vt).T
            if np.linalg.det(Rm) < 0:
                U[:, -1] *= -1
                Rm = (U @ Vt).T
            for r in hd.get("rooms", []):
                pts = np.asarray(r["pts"], np.float64)
                cells_by[r["id"]] = (r.get("name_zh") or r.get("name") or "",
                                     (pts - ca) @ Rm.T + cb)
    if cells_by:
        from mast3r_slam.hmsg.graph import ROOM_PALETTE
        digits = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        lab = np.full((G, G), -1, np.int16)
        regions = []
        for i, (rid, (nm, pts)) in enumerate(cells_by.items()):
            if i >= len(digits) or not len(pts):
                continue
            p3 = np.zeros((len(pts), 3))
            p3[:, a] = pts[:, 0]
            p3[:, b] = pts[:, 1]
            px = (p3[:, a] - (meta["ca"] - half)) / (2 * half) * G
            py = G - 1 - (p3[:, b] - (meta["cb"] - half)) / (2 * half) * G
            xi, yi = np.round(px).astype(int), np.round(py).astype(int)
            ok = (xi >= 0) & (xi < G) & (yi >= 0) & (yi < G)
            lab[yi[ok], xi[ok]] = i
            cx, cy = (float(np.median(px[ok])), float(np.median(py[ok]))) \
                if ok.any() else (0.0, 0.0)
            regions.append({"id": rid, "name": nm,
                            "color": ROOM_PALETTE[i % len(ROOM_PALETTE)],
                            "cpx": [round(cx, 1), round(cy, 1)]})
        import cv2 as _cv
        masks = []
        for i in range(len(regions)):      # 0.15m 网格点 -> 补洞成面 + 去噪碎片
            mk = (lab == i).astype(np.uint8)
            mk = _cv.morphologyEx(mk, _cv.MORPH_CLOSE,
                                  np.ones((5, 5), np.uint8))
            # 玻璃反射等噪声足迹呈孤立小块: 保最大连通块 + 面积>=1.5m^2 的块
            n, cc, st, _ = _cv.connectedComponentsWithStats(mk, 8)
            if n > 2:
                areas = st[1:, _cv.CC_STAT_AREA]
                keep = {int(np.argmax(areas)) + 1} | \
                    {j for j in range(1, n) if areas[j - 1] >= 150}
                mk = np.isin(cc, list(keep)).astype(np.uint8)
            masks.append(mk)
        lab[:] = -1                        # 大区域先画, 空格子归属不互抢
        for i in sorted(range(len(masks)), key=lambda k: -int(masks[k].sum())):
            lab[(masks[i] > 0) & (lab == -1)] = i
        region_rows = ["".join("." if v < 0 else digits[v] for v in row)
                       for row in lab]
        regions_js = {"regions": regions, "rows": region_rows}
        print(f"[export_web] 语义区域底图: {len(regions)} 区域")

    # 每个关键帧的 VLM 区域判定描述 (点击 kf 点显示 Qwen 空间描述)
    kf_vlm = {}
    rid2name = {}
    if vr.exists():
        vj2 = json.loads(vr.read_text())
        rid2name = {r["id"]: (r.get("name") or r.get("kind") or "")
                    for r in vj2.get("regions", [])}
        for k, m in vj2.get("frames", {}).items():
            kf_vlm[int(k)] = {"d": (m.get("desc") or "")[:90],
                              "rn": rid2name.get(m.get("rid"), "")}
    if not kf_vlm and hj.exists():
        # 兼容旧产物: hmsg.js views 携带帧级判定描述 (img_id=fid)
        hd2 = json.loads(hj.read_text()[len("window.HMSG = "):-1])
        rn2 = {r["id"]: (r.get("name_zh") or r.get("name") or "")
               for r in hd2.get("rooms", [])}
        fid2kf = {int(f): i for i, f in enumerate(frame_ids)}
        for v in hd2.get("views", []):
            k = fid2kf.get(int(v["img_id"]))
            if k is not None:
                kf_vlm[k] = {"d": (v.get("desc") or "")[:90],
                             "rn": rn2.get(v.get("room"), "")}

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
        "kf_vlm": kf_vlm,
        "nodes": nodes,
        "semregions": regions_js,        # 语义区域底图层 (null=无)
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
