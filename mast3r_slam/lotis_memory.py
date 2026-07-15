#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoTIS 记忆分段构建 —— LiveSemMap 建图产物 step 8(仅推理, 不训练)。

按**语义节点边**切段(M4c: 段帧=**原始帧** 6.35Hz 全存, 非关键帧):
  - 节点锚点仍是关键帧(min(kf_indices)/rep_kf), 经 frame_ids[kf] 映射为原始帧 id;
    相邻两节点 A→B 的原始帧 id 区间内**全部原始帧** = 一条"边段"(edge_{Aid}_{Bid})。
    >max_len(40) 帧按 ceil(L/max_len) 均分为多个连续子段 _p{k}(不下采样丢帧)。
  - category=="junction"(路口/拐弯)的节点, 额外存 rep_kf ±turn_radius(关键帧半径)
    映射的原始帧全序列作"拐弯段"(turn_{id}[_p{k}]), 供丝滑转弯。
  - 段 frame_indices = **原始帧 id**(对应 datasets/<seq>/<id:06d>.png)。

产物(logs/<run>/):
  {seq}_lotis_seg.json   分段定义(key/type/frame_indices/节点链接/方裁参数/frame_source=raw)
  {seq}_lotis_traj.pkl   各段 encode_trajectory 预编码 {key: TrajectoryEncoding.to_dict()}
  {seq}_lotis_feats.npz  (可选 --with-feats)逐帧 DINOv3 patch 特征 fp16。

建图内调用: evaluate step 8 -> save_lotis_memory(savedir, seq, dataset_dir)
离线构建/自测: python -m mast3r_slam.lotis_memory --run logs/cfds_floor28_run --seq cfds_floor28
              (原始帧图来源默认 datasets/<seq>/<id:06d>.png; kf->原始帧 id 读 occupancy.npz)
"""
import os
import sys
import json
import pickle
import logging
import pathlib
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)


# --------------------------------------------------------------------------- #
# 分段(纯数据, 不依赖模型, 可单测)
# --------------------------------------------------------------------------- #
def _even_chunks(items: List[int], max_len: int) -> List[List[int]]:
    """把连续帧列表切成若干近似等长的连续子段, 每段长度 <= max_len。
    段数 n = ceil(L/max_len)(如 L=90,max_len=40 -> n=3), 再把 L 帧近似平均分到 n 段
    (前 L%n 段各多 1 帧)。不下采样丢帧, 各子段物理连续、不重叠。"""
    import math
    L = len(items)
    if L <= max_len:
        return [items]
    n = math.ceil(L / max_len)
    base, extra = divmod(L, n)
    out, i = [], 0
    for k in range(n):
        size = base + (1 if k < extra else 0)
        out.append(items[i:i + size])
        i += size
    return out


def build_segments(nodes: List[dict], frame_ids, n_raw: int, turn_radius: int = 20,
                   min_seg: int = 3, max_len: int = 40) -> List[dict]:
    """语义节点 -> 分段定义列表(M4c: 段帧=**原始帧**空间, 非关键帧)。
    node_id = 节点在 nodes 中的下标(与 data.js 一致)。
    - 节点锚点仍是关键帧(rep_kf / min(kf_indices)), 经 `frame_ids[kf]` 映射为原始帧 id。
    - 相邻节点的原始帧 id 区间内的**全部原始帧**构成边段(6.35Hz 全存), >max_len 按
      ceil(L/max_len) 均分为多个连续子段(每段 <= max_len, 不下采样)。
    - 段的 `frame_indices` 存的是**原始帧 id**(对应 datasets/<seq>/<id:06d>.png)。
    frame_ids: 长度 n_kf 的数组, frame_ids[kf] = 该关键帧对应的原始帧 id。
    n_raw: 原始帧总数(用于区间边界裁剪)。"""
    if not nodes:
        return []
    frame_ids = np.asarray(frame_ids).ravel().astype(int)
    n_kf = len(frame_ids)
    anchor = [min(n["kf_indices"]) for n in nodes]           # 关键帧序号
    order = sorted(range(len(nodes)), key=lambda i: anchor[i])

    def raw_range(kf_lo: int, kf_hi: int) -> List[int]:
        """关键帧区间 -> 原始帧 id 区间 -> 该区间内全部原始帧 id。"""
        a_kf = max(0, min(int(kf_lo), n_kf - 1))
        b_kf = max(0, min(int(kf_hi), n_kf - 1))
        r_lo, r_hi = sorted((int(frame_ids[a_kf]), int(frame_ids[b_kf])))
        r_lo = max(0, r_lo)
        r_hi = min(n_raw - 1, r_hi)
        return list(range(r_lo, r_hi + 1))

    segs: List[dict] = []
    # 1) 边段: 相邻节点之间的原始帧全序列; 过长(>max_len)则均分为多个连续子段
    for a, b in zip(order[:-1], order[1:]):
        frames_all = raw_range(anchor[a], anchor[b])
        if len(frames_all) < min_seg:
            continue
        parts = _even_chunks(frames_all, max_len)
        n_parts = len(parts)
        for pk, frames in enumerate(parts):
            key = f"edge_{a}_{b}" if n_parts == 1 else f"edge_{a}_{b}_p{pk}"
            segs.append({
                "key": key, "type": "edge",
                "frame_indices": frames, "node_from": a, "node_to": b,
                "part": pk, "n_parts": n_parts,
                "from_name": nodes[a].get("name", ""), "to_name": nodes[b].get("name", ""),
                "from_cat": nodes[a].get("category", ""), "to_cat": nodes[b].get("category", ""),
            })
    # 2) 拐弯段: junction 节点 rep_kf ±turn_radius(关键帧半径)-> 原始帧全序列, 同样均分
    for idx, n in enumerate(nodes):
        if n.get("category") != "junction":
            continue
        c = int(n.get("rep_kf", anchor[idx]))
        frames_all = raw_range(c - turn_radius, c + turn_radius)
        if len(frames_all) < min_seg:
            continue
        parts = _even_chunks(frames_all, max_len)
        n_parts = len(parts)
        for pk, frames in enumerate(parts):
            key = f"turn_{idx}" if n_parts == 1 else f"turn_{idx}_p{pk}"
            segs.append({
                "key": key, "type": "turn",
                "frame_indices": frames, "node": idx,
                "part": pk, "n_parts": n_parts,
                "name": n.get("name", ""), "center_kf": c,
            })
    return segs


# --------------------------------------------------------------------------- #
# VIO 位姿 + mined 边(空间闭环: 物理相邻但时序远的节点 -> 复用真实连续帧补可打点段)
# --------------------------------------------------------------------------- #
def load_frame_positions(dataset_dir, n_raw: int):
    """读 timestamps.txt(帧->时刻)+ vio.txt(时刻->位姿), 插值出每个原始帧的 xy 米制位置。
    返回 (n_raw,2) float 数组(无位姿的帧置 nan); 缺文件返回 None。"""
    ds = pathlib.Path(dataset_dir)
    ts_f, vio_f = ds / "timestamps.txt", ds / "vio.txt"
    if not (ts_f.exists() and vio_f.exists()):
        return None
    ts = {}
    for ln in ts_f.read_text().splitlines():
        p = ln.split()
        if p and not p[0].startswith("#"):
            ts[int(p[0])] = float(p[1])
    vt, vx, vy = [], [], []
    for ln in vio_f.read_text().splitlines():
        p = ln.split()
        if p and not p[0].startswith("#"):
            vt.append(float(p[0])); vx.append(float(p[1])); vy.append(float(p[2]))
    if not ts or not vt:
        return None
    vt = np.asarray(vt); vx = np.asarray(vx); vy = np.asarray(vy)
    fpos = np.full((n_raw, 2), np.nan)
    for rid, t in ts.items():
        if 0 <= rid < n_raw:
            fpos[rid, 0] = float(np.interp(t, vt, vx))
            fpos[rid, 1] = float(np.interp(t, vt, vy))
    return fpos


def _longest_increasing_window(s):
    """一维序列里取最长'近单调上升'窗口(容忍 0.3m 小回退)。返回 (lo,hi) 闭区间。"""
    n = len(s); best = (0, 0); bl = 1; i = 0
    while i < n:
        j = i; peak = s[i]; back = 0
        while j + 1 < n:
            if s[j + 1] >= peak - 0.3:
                peak = max(peak, s[j + 1]); j += 1
            else:
                back += 1
                if back > 2:
                    break
                j += 1
        if j - i + 1 > bl:
            bl = j - i + 1; best = (i, j)
        i = j + 1 if j > i else i + 1
    return best


def _mine_one(A, B, fpos, raw_ids, tube_r, run_gap, end_m, max_len):
    """A->B 管道里挖一段单调连续真实帧: 管道取帧->断趟->选趟定向->裁单调子段->抽稀。
    返回 (frame_indices[原始帧id], dir) 或 None。"""
    AB = B - A; L = float(np.hypot(*AB))
    if L < 1e-6:
        return None
    u = AB / L; rel = fpos[raw_ids] - A
    proj = rel @ u; perp = np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0])
    idx = np.where((perp < tube_r) & (proj > -tube_r) & (proj < L + tube_r))[0]
    if idx.size == 0:
        return None
    rid = raw_ids[idx]
    runs = []; cur = [idx[0]]                                  # 按原始帧号断趟
    for k in range(1, len(idx)):
        if rid[k] - rid[k - 1] <= run_gap:
            cur.append(idx[k])
        else:
            runs.append(np.array(cur)); cur = [idx[k]]
    runs.append(np.array(cur))
    best = None
    for r in runs:
        r = r[np.argsort(raw_ids[r])]
        pr = proj[r]
        mono = float(np.corrcoef(np.arange(len(pr)), pr)[0, 1]) if len(pr) > 2 else 0.0
        rr = r[::-1] if mono < 0 else r                       # 统一成 proj 递增(A->B)
        rr = rr[(proj[rr] >= -end_m) & (proj[rr] <= L + end_m)]  # 夹到 chord 附近去过冲
        if len(rr) < 2:
            continue
        lo, hi = _longest_increasing_window(proj[rr])
        sub = rr[lo:hi + 1]; ps = proj[sub]
        score = (ps.max() - ps.min()) / L * len(sub)
        if best is None or score > best[0]:
            best = (score, sub, ("rev" if mono < 0 else "fwd"))
    if best is None:
        return None
    sub = best[1]
    if len(sub) > max_len:
        sub = sub[np.unique(np.linspace(0, len(sub) - 1, max_len).round().astype(int))]
    return [int(raw_ids[k]) for k in sub], best[2]


def mine_segments(nodes: List[dict], fpos, walk_pairs, phys_max: float = 4.0,
                  walk_gap: int = 2, tube_r: float = 1.5, run_gap: int = 6,
                  end_m: float = 0.5, factor: float = 2.0, max_len: int = 40) -> List[dict]:
    """物理相邻但时序远的节点对 -> 贪心去冗余保留闭环边 -> 各挖一段单调真实帧。
    walk_pairs: 已有 walk 边的 (a,b) 集合(有向, 两向都含)。返回 mined 段 dict 列表。"""
    import heapq
    from collections import defaultdict
    n = len(nodes)
    pos = np.array([[nd["position"][0], nd["position"][1]] for nd in nodes])
    anchor = [min(nd["kf_indices"]) for nd in nodes]
    order = sorted(range(n), key=lambda i: anchor[i])
    wo = [0] * n
    for w, i in enumerate(order):
        wo[i] = w
    raw_ids = np.where(np.all(np.isfinite(fpos), axis=1))[0]   # 有位姿的原始帧 id
    dd = lambda i, j: float(np.hypot(*(pos[i] - pos[j])))
    cands = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in walk_pairs:
                continue
            if dd(i, j) < phys_max and abs(wo[i] - wo[j]) >= walk_gap:
                cands.append((i, j, dd(i, j)))
    cands.sort(key=lambda x: x[2])
    # 贪心去冗余: 当前图(walk 边, 权=直线距)最短路 > factor*物理 才保留
    adj = defaultdict(list)
    for a, b in {tuple(sorted(p)) for p in walk_pairs}:
        adj[a].append((b, dd(a, b))); adj[b].append((a, dd(a, b)))

    def sp(s, t):
        D = {s: 0.0}; pq = [(0.0, s)]
        while pq:
            du, uu = heapq.heappop(pq)
            if uu == t:
                return du
            if du > D.get(uu, 1e9):
                continue
            for v, w in adj[uu]:
                nd = du + w
                if nd < D.get(v, 1e9):
                    D[v] = nd; heapq.heappush(pq, (nd, v))
        return float("inf")

    out = []
    for i, j, d0 in cands:
        if sp(i, j) <= factor * d0:
            continue                                          # 冗余平行边, 跳过
        res = _mine_one(pos[i], pos[j], fpos, raw_ids, tube_r, run_gap, end_m, max_len)
        if res is None:
            logger.warning(f"[lotis-mine] n{i}<->n{j} 管道无连续帧, 跳过")
            continue
        frames, dr = res
        adj[i].append((j, d0)); adj[j].append((i, d0))        # 并入图, 影响后续去冗余
        out.append({
            "key": f"mined_{i}_{j}", "type": "mined",
            "frame_indices": frames, "node_from": i, "node_to": j,
            "part": 0, "n_parts": 1, "mine_dir": dr,
            "from_name": nodes[i].get("name", ""), "to_name": nodes[j].get("name", ""),
            "from_cat": nodes[i].get("category", ""), "to_cat": nodes[j].get("category", ""),
        })
    logger.info(f"[lotis-mine] 候选 {len(cands)} -> 去冗余保留 mined 边 {len(out)}: "
                + ", ".join(s["key"] for s in out))
    return out


# --------------------------------------------------------------------------- #
# 构建 + 编码 + 落盘
# --------------------------------------------------------------------------- #
def build_lotis_memory(run_dir, seq_name: str, raw_image_fn: Callable[[int], Image.Image],
                       frame_ids, n_raw: int, device: str = "cuda", turn_radius: int = 20,
                       min_seg: int = 3, max_len: int = 40, with_feats: bool = False,
                       frame_pos=None) -> dict:
    """核心(M4c): 读 {seq}_semantic.json -> 在**原始帧**空间切段 -> 逐段编码 -> 落 seg.json + traj.pkl。
    raw_image_fn(rid) 返回原始帧 id=rid 的 RGB PIL 图(datasets/<seq>/<rid:06d>.png)。
    frame_ids[kf] = 关键帧 kf 对应的原始帧 id; n_raw = 原始帧总数。"""
    run_dir = pathlib.Path(run_dir)
    sem_path = run_dir / f"{seq_name}_semantic.json"
    if not sem_path.exists():
        logger.warning(f"[lotis-mem] 缺 {sem_path}, 跳过(先存语义地图)")
        return {}
    frame_ids = np.asarray(frame_ids).ravel().astype(int)
    n_kf = len(frame_ids)
    nodes = json.loads(sem_path.read_text()).get("nodes", [])
    segs = build_segments(nodes, frame_ids, n_raw, turn_radius, min_seg, max_len)
    # mined 边: 物理相邻但时序远的节点对, 用 VIO 位姿从整段录制挖真实连续帧补可打点段
    if frame_pos is not None:
        walk_pairs = set()
        for s in segs:
            if s["type"] == "edge":
                walk_pairs.add((s["node_from"], s["node_to"]))
                walk_pairs.add((s["node_to"], s["node_from"]))
        segs.extend(mine_segments(nodes, frame_pos, walk_pairs, max_len=max_len))
    n_edge = sum(1 for s in segs if s["type"] == "edge")
    n_turn = sum(1 for s in segs if s["type"] == "turn")
    n_mined = sum(1 for s in segs if s["type"] == "mined")
    logger.info(f"[lotis-mem] {len(nodes)} 语义节点 -> 分段 {len(segs)} "
                f"(边段 {n_edge} + 拐弯段 {n_turn} + mined {n_mined}), "
                f"n_kf={n_kf} n_raw={n_raw} [原始帧空间]")
    if not segs:
        logger.warning("[lotis-mem] 无可用分段(语义节点不足?), 跳过")
        return {}

    from nav_web.lotis_engine import LotisEngine, crop_params

    # 方裁参数(段帧=原始 png, 取第一帧算一次; 仅作元信息, 前端映射用 query 帧自身尺寸)
    probe = raw_image_fn(segs[0]["frame_indices"][0])
    W, H = probe.size
    left, top, s = crop_params(W, H)

    eng = LotisEngine(device=device)
    cache: Dict[str, dict] = {}
    for seg in segs:
        try:
            imgs = [raw_image_fn(i) for i in seg["frame_indices"]]
            enc = eng.encode_images(imgs)
            cache[seg["key"]] = enc.to_dict()
            seg["seq_len"] = int(enc.seq_len)
            logger.info(f"[lotis-mem] {seg['key']}({seg['type']}) "
                        f"{len(imgs)}帧 -> seq_len={enc.seq_len}")
        except Exception as e:
            logger.warning(f"[lotis-mem] 段 {seg['key']} 编码失败: {e}")

    # 落盘
    seg_meta = {
        "seq": seq_name, "n_kf": int(n_kf), "n_raw": int(n_raw), "frame_source": "raw",
        "frame_wh": [int(W), int(H)], "crop": [int(left), int(top), int(s)],
        "turn_radius": int(turn_radius), "max_len": int(max_len),
        "segments": [s for s in segs if s["key"] in cache],
    }
    (run_dir / f"{seq_name}_lotis_seg.json").write_text(
        json.dumps(seg_meta, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    with open(run_dir / f"{seq_name}_lotis_traj.pkl", "wb") as f:
        pickle.dump(cache, f)
    logger.info(f"[lotis-mem] 已存 {len(cache)} 段 -> {seq_name}_lotis_seg.json / _lotis_traj.pkl")

    # 可选: 逐原始帧 DINOv3 特征(记忆本体, 供将来任意子路径现编码)
    if with_feats:
        try:
            _save_kf_feats(run_dir, seq_name, raw_image_fn, n_raw, eng)
        except Exception as e:
            logger.warning(f"[lotis-mem] 逐帧特征存储失败(不影响分段): {e}")

    return {"n_seg": len(cache), "n_edge": n_edge, "n_turn": n_turn, "n_mined": n_mined}


def _save_kf_feats(run_dir, seq_name, image_fn, n_frames, eng, batch: int = 16):
    """逐帧 DINOv3 patch 特征 [N,14,14,768] fp16 -> {seq}_lotis_feats.npz。"""
    import torch
    from nav_web.lotis_engine import square_crop
    sys.path.insert(0, os.path.join(_PROJ, "third_party", "lotis"))
    from lotis.preprocessing import preprocess_image
    from lotis.feature_extraction import extract_features

    feats = np.zeros((n_frames, 14, 14, 768), dtype=np.float16)
    for i0 in range(0, n_frames, batch):
        ims = [square_crop(image_fn(i)) for i in range(i0, min(i0 + batch, n_frames))]
        proc = torch.stack([preprocess_image(im) for im in ims]).to(eng.device)
        with torch.no_grad():
            f = extract_features(proc, eng.localizer.feature_extractor, eng.device)
        feats[i0:i0 + len(ims)] = f.float().cpu().numpy().astype(np.float16)
    np.savez_compressed(run_dir / f"{seq_name}_lotis_feats.npz", feats=feats)
    logger.info(f"[lotis-mem] 逐帧特征 {feats.shape} fp16 -> {seq_name}_lotis_feats.npz")


# --------------------------------------------------------------------------- #
# 建图内入口(evaluate step 8 调用)
# --------------------------------------------------------------------------- #
def save_lotis_memory(savedir, seq_name: str, dataset_dir, device: str = "cuda",
                      turn_radius: int = 20, with_feats: bool = False):
    """建图退出时调用(M4c): 段帧=原始帧, 图取自 datasets/<seq>/<id:06d>.png。
    kf->原始帧 id 映射读刚存的 {seq}_occupancy.npz(step6 产物, frame_ids)。"""
    savedir = pathlib.Path(savedir)
    occ = savedir / f"{seq_name}_occupancy.npz"
    if not occ.exists():
        logger.warning(f"[lotis-mem] 缺 {occ}(应在 step6 后), 跳过")
        return
    frame_ids = np.asarray(np.load(occ)["frame_ids"]).ravel().astype(int)
    ds = pathlib.Path(dataset_dir)
    pngs = sorted(ds.glob("[0-9]*.png"))
    n_raw = len(pngs) if pngs else int(frame_ids.max()) + 1

    def raw_img(rid):
        return Image.open(ds / f"{int(rid):06d}.png").convert("RGB")

    frame_pos = load_frame_positions(ds, n_raw)
    if frame_pos is None:
        logger.warning(f"[lotis-mem] 缺 timestamps.txt/vio.txt, 不挖 mined 边")
    build_lotis_memory(savedir, seq_name, raw_img, frame_ids, n_raw, device=device,
                       turn_radius=turn_radius, with_feats=with_feats, frame_pos=frame_pos)


# --------------------------------------------------------------------------- #
# 离线 CLI(自测/补算): 关键帧图来源 web/thumbs/kf{i}.jpg
# --------------------------------------------------------------------------- #
def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="LoTIS 记忆分段离线构建(M4c: 原始帧空间)")
    ap.add_argument("--run", default="logs/cfds_floor28_run")
    ap.add_argument("--seq", default="cfds_floor28")
    ap.add_argument("--dataset", default=None,
                    help="原始帧目录(含 <id:06d>.png); 默认 datasets/<seq>")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--turn-radius", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=40)
    ap.add_argument("--with-feats", action="store_true", help="额外存逐帧 DINOv3 特征")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    run = pathlib.Path(args.run if os.path.isabs(args.run) else os.path.join(_PROJ, args.run))
    ds = pathlib.Path(args.dataset) if args.dataset else pathlib.Path(_PROJ) / "datasets" / args.seq
    # kf->原始帧 id 映射 + 原始帧总数
    frame_ids = np.asarray(np.load(run / f"{args.seq}_occupancy.npz")["frame_ids"]).ravel().astype(int)
    pngs = sorted(ds.glob("[0-9]*.png"))
    n_raw = len(pngs) if pngs else int(frame_ids.max()) + 1

    def raw_img(rid):
        return Image.open(ds / f"{int(rid):06d}.png").convert("RGB")

    frame_pos = load_frame_positions(ds, n_raw)
    if frame_pos is None:
        logging.warning("[lotis-mem] 缺 timestamps.txt/vio.txt, 不挖 mined 边")
    build_lotis_memory(run, args.seq, raw_img, frame_ids, n_raw, device=args.device,
                       turn_radius=args.turn_radius, max_len=args.max_len,
                       with_feats=args.with_feats, frame_pos=frame_pos)


if __name__ == "__main__":
    _cli()
