#!/usr/bin/env python3
"""语义导航 Web 服务 (Flask)。

- 静态: nav.html / data.js / thumbs/
- POST /api/nl_query    自然语言 -> 语义节点/关键帧 (调 L40 vLLM, 与建图共用同一服务)
- POST /api/locate_image 上传观察图像 -> SelaVPR 描述子 -> 最近关键帧 (VPR 重定位)
- GET  /api/lotis_segments 列出可打点的 LoTIS 记忆段
- POST /api/lotis_point  在某记忆段内用一张查询帧做 LoTIS 打点

用法: python nav_web/server.py --run logs/cfds_floor28_run --seq cfds_floor28 [--port 8080]
"""
import argparse
import base64
import json
import pathlib
import sys
import threading

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))          # 便于 `from lotis_engine import ...`

app = Flask(__name__)
CFG = {}
_VPR = {"ex": None, "lock": threading.Lock()}     # SelaVPR 懒加载
_LOTIS = {"eng": None, "seg": None, "lock": threading.Lock()}   # LoTIS 懒加载
_RAW = {"px": None, "seg_frames": None}           # 原始帧像素坐标 + 段→帧号 缓存


def _load_nodes():
    sem = json.loads((CFG["run"] / f"{CFG['seq']}_semantic.json").read_text())
    return sem["nodes"]


@app.route("/")
def index():
    return send_from_directory(ROOT / "static", "nav.html")


@app.route("/data.js")
def data_js():
    return send_from_directory(CFG["run"] / "web", "data.js")


@app.route("/thumbs/<path:p>")
def thumbs(p):
    return send_from_directory(CFG["run"] / "web" / "thumbs", p)


@app.route("/raw/<int:fid>")
def raw_frame(fid):
    """原始数据集 front_1 帧图: datasets/<seq>/{fid:06d}.png (LoTIS query 用原图, 非关键帧缩略图)。"""
    return send_from_directory(CFG["dataset"], f"{fid:06d}.png")


@app.route("/surround/<int:fid>/<int:cam>")
def surround_frame(fid, cam):
    """环视鱼眼缩略图 (cam 1前/2右/3后/4左), 首次访问生成 420px 宽缓存。"""
    cache = CFG["run"] / "web" / "sur"
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"{fid:06d}_{cam}.jpg"
    if not f.exists():
        src = CFG["dataset"] / "surround" / f"{fid:06d}_{cam}.jpg"
        if not src.exists():
            return "", 404
        from PIL import Image
        im = Image.open(src)
        im = im.resize((420, int(im.height * 420 / im.width)),
                       Image.BILINEAR)
        im.convert("RGB").save(f, "JPEG", quality=82)
    return send_from_directory(cache, f.name)


# ---------------- HMSG 层级多模态场景图 ----------------
_HMSG = {"pack": None, "clip": None, "lock": threading.Lock()}


@app.route("/hmsg")
def hmsg_page():
    return send_from_directory(ROOT / "static", "hmsg.html")


@app.route("/hmsg.js")
def hmsg_js():
    return send_from_directory(CFG["run"] / "web", "hmsg.js")


@app.route("/api/hmsg_query", methods=["POST"])
def hmsg_query():
    """fast 层级检索 (照抄 fsr_vln 三级 CLIP 匹配): text -> top房间 + top物体。
    惰性加载 query_pack.npz + OpenCLIP 文本塔。"""
    text = (request.get_json(force=True) or {}).get("text", "").strip()
    if not text:
        return jsonify({"ok": False, "msg": "空查询"})
    raw_text = text
    if not text.isascii():          # 中文查询: Qwen 翻成英文短语再走 CLIP
        from mast3r_slam.hmsg.qwen_zh import translate_query
        text = translate_query(text, CFG.get("api", ""),
                               CFG.get("model", "qwen3.5-35b-a3b"))
        print(f"[hmsg] 查询翻译: {raw_text} -> {text}")
    with _HMSG["lock"]:
        if _HMSG["pack"] is None:
            p = CFG["run"] / "hmsg" / "query_pack.npz"
            if not p.exists():
                return jsonify({"ok": False, "msg": "HMSG 未构建"})
            _HMSG["pack"] = dict(np.load(p, allow_pickle=False))
            from mast3r_slam.hmsg.features import SamClipExtractor  # noqa
            import open_clip
            import torch
            from mast3r_slam.hmsg.features import CLIP_CKPT
            m, _, _ = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained=str(CLIP_CKPT))
            _HMSG["clip"] = (m.to("cuda:0").eval(),
                             open_clip.get_tokenizer("ViT-L-14"))
            print("[hmsg] 查询包 + CLIP 文本塔已加载")
    import torch
    model, tok = _HMSG["clip"]
    pk = _HMSG["pack"]
    with torch.no_grad():
        feats = []
        for tpl in ("{}", "a photo of {} in the scene."):
            f = model.encode_text(tok([tpl.format(text), tpl.format("background")]
                                      ).to("cuda:0")).float()
            feats.append(torch.nn.functional.normalize(f, dim=-1))
        tf = torch.nn.functional.normalize(torch.stack(feats).mean(0),
                                           dim=-1).cpu().numpy()
    qf, bg = tf[0], tf[1]
    # 房间: 各房间代表特征取 max
    rs = {}
    for rid, f in zip(pk["room_rep_ids"], pk["room_rep_feats"]):
        s = float(f @ qf)
        rs[str(rid)] = max(s, rs.get(str(rid), -1))
    top_rooms = sorted(rs.items(), key=lambda x: -x[1])[:5]
    cand = {r for r, _ in top_rooms}
    objs = []
    for i in range(len(pk["obj_ids"])):
        if str(pk["obj_rooms"][i]) not in cand:
            continue
        sq, sb = float(pk["obj_feats"][i] @ qf), float(pk["obj_feats"][i] @ bg)
        if sq <= sb:                       # 负词表过滤 (icra 版 ["background"])
            continue
        objs.append({"id": str(pk["obj_ids"][i]),
                     "name": str(pk["obj_names"][i]),
                     "room": str(pk["obj_rooms"][i]),
                     "best_view": str(pk["obj_best_views"][i]),
                     "score": round(sq, 4)})
    objs.sort(key=lambda x: -x["score"])
    return jsonify({"ok": True, "query_en": text,
                    "rooms": [{"id": r, "score": round(s, 4)}
                              for r, s in top_rooms],
                    "objects": objs[:8]})


@app.route("/api/nl_query", methods=["POST"])
def nl_query():
    """自然语言 -> 最匹配的语义节点。body: {text: str}
    返回 {ok, node_id, name, reason} 或 {ok: false, reason}"""
    j = request.json or {}
    text = j.get("text", "").strip()
    if not text:
        return jsonify({"ok": False, "reason": "空查询"})
    nodes = _load_nodes()
    if not nodes:
        return jsonify({"ok": False, "reason": "地图还没有语义节点"})

    # 前端可附带各节点距参考点(通常=已设起点)的真实可走距离, 用于同类多节点时就近选择
    dists = j.get("node_dists")
    ref_label = j.get("ref_label", "")
    ref_kind = j.get("ref_kind", "起点")

    def _dist_str(i):
        if not dists or i >= len(dists):
            return ""
        return f", 距{ref_kind} {dists[i]} 单位" if dists[i] is not None else f", 从{ref_kind}不可达"

    listing = "\n".join(
        f"- id={i}: {n['name']} (类别: {n['category']}, 描述: {n['description']}{_dist_str(i)})"
        for i, n in enumerate(nodes))
    dist_rule = ""
    if dists:
        dist_rule = (f"\n- 用户已设{ref_kind}「{ref_label}」。若多个节点同样符合描述, "
                     f"必须选距{ref_kind}最近的那个; 标注\"不可达\"的节点不要选。")
    prompt = f"""你是室内导航助手。地图上有以下语义地标节点:
{listing}

用户想去的地点描述: "{text}"

规则:
- 选出最匹配的一个节点。如果没有任何节点合理匹配(比如用户描述的地点类型在列表中不存在), matched=false。{dist_rule}
只输出 JSON: {{"matched": true/false, "node_id": 数字, "reason": "一句话理由"}}"""

    schema = {"type": "object",
              "properties": {"matched": {"type": "boolean"},
                             "node_id": {"type": "integer"},
                             "reason": {"type": "string", "maxLength": 80}},
              "required": ["matched", "node_id", "reason"]}
    try:
        import requests as rq
        r = rq.post(f"{CFG['api']}/chat/completions", json={
            "model": CFG["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "match", "schema": schema}},
        }, timeout=30, proxies={"http": None, "https": None})  # 内网 vLLM 不走代理
        r.raise_for_status()
        out = json.loads(r.json()["choices"][0]["message"]["content"])
    except Exception as e:
        return jsonify({"ok": False, "reason": f"语言模型服务不可用: {e}"})

    if not out.get("matched") or not (0 <= out.get("node_id", -1) < len(nodes)):
        return jsonify({"ok": False, "reason": out.get("reason", "未匹配到节点")})
    n = nodes[out["node_id"]]
    return jsonify({"ok": True, "node_id": out["node_id"], "name": n["name"],
                    "reason": out.get("reason", "")})


@app.route("/api/locate_image", methods=["POST"])
def locate_image():
    """上传一张观察图像 -> SelaVPR 检索最近关键帧。body: {image: dataURL 或 base64}
    返回 {ok, kf_idx, score, second: {kf_idx, score}}"""
    b64 = (request.json or {}).get("image", "")
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    if not b64:
        return jsonify({"ok": False, "reason": "无图像"})
    desc_path = CFG["run"] / f"{CFG['seq']}_vpr_desc.npy"
    if not desc_path.exists():
        return jsonify({"ok": False, "reason": "该地图未提取 VPR 描述子"})
    try:
        import cv2
        img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8),
                           cv2.IMREAD_COLOR)
        with _VPR["lock"]:
            if _VPR["ex"] is None:
                print("[vpr] 首次调用, 加载 SelaVPR++ ...")
                import torch
                from mast3r_slam.selavpr import SelaVPRExtractor
                ex = SelaVPRExtractor(backbone="dinov2-large", use_hashing=False,
                                      use_rerank=False, device="cuda:0")
                if isinstance(ex.model, torch.nn.DataParallel):
                    ex.model = ex.model.module.to("cuda:0")
                _VPR["ex"] = ex
            q = _VPR["ex"].extract_batch([img]).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-9
        db = np.load(desc_path)
        sims = (q @ db.T)[0]
        order = np.argsort(-sims)
        return jsonify({"ok": True, "kf_idx": int(order[0]),
                        "score": float(sims[order[0]]),
                        "second": {"kf_idx": int(order[1]),
                                   "score": float(sims[order[1]])} if len(order) > 1 else None})
    except Exception as e:
        return jsonify({"ok": False, "reason": f"定位失败: {e}"})


# --------------------------------------------------------------------------- #
# LoTIS 打点 (M2)
# --------------------------------------------------------------------------- #
def _lotis_seg():
    """加载并缓存 *_lotis_seg.json (段清单 + crop/frame_wh)。缺文件返回 None。"""
    if _LOTIS["seg"] is None:
        p = CFG["run"] / f"{CFG['seq']}_lotis_seg.json"
        if not p.exists():
            return None
        _LOTIS["seg"] = json.loads(p.read_text())
    return _LOTIS["seg"]


def _lotis_engine():
    """懒加载 LoTIS 引擎 + 段编码缓存(线程安全)。
    成功返回 (engine, seg_meta); 失败返回 (None, 错误串)。"""
    seg = _lotis_seg()
    if seg is None:
        return None, "该地图未生成 LoTIS 记忆分段(缺 *_lotis_seg.json)"
    with _LOTIS["lock"]:
        if _LOTIS["eng"] is None:
            traj = CFG["run"] / f"{CFG['seq']}_lotis_traj.pkl"
            if not traj.exists():
                return None, "缺段编码缓存 *_lotis_traj.pkl"
            print("[lotis] 首次调用, 加载 LoTIS 引擎 + 段编码 ...")
            from lotis_engine import LotisEngine
            _LOTIS["eng"] = LotisEngine(traj_cache=str(traj))
    return _LOTIS["eng"], seg


@app.route("/api/lotis_segments", methods=["GET"])
def lotis_segments():
    """列出可打点的 LoTIS 段(供前端选目标段/查询帧)。
    返回 {ok, frame_wh, crop, frame_source, segments:[{key,type,frame_lo,frame_hi,n_frames,...}]}"""
    seg = _lotis_seg()
    if seg is None:
        return jsonify({"ok": False, "reason": "该地图无 LoTIS 分段"})

    def _fi(s):   # 段帧序列: M4c=frame_indices(原始帧), 旧版=kf_indices(关键帧)
        return s.get("frame_indices") or s.get("kf_indices") or []
    segs = [{"key": s["key"], "type": s["type"],
             "frame_lo": int(min(_fi(s))), "frame_hi": int(max(_fi(s))),
             "n_frames": len(_fi(s)),
             "from_name": s.get("from_name", ""), "to_name": s.get("to_name", ""),
             "node_from": s.get("node_from"), "node_to": s.get("node_to"),
             "seq_len": s.get("seq_len")}
            for s in seg["segments"]]
    return jsonify({"ok": True, "frame_wh": seg["frame_wh"], "crop": seg["crop"],
                    "frame_source": seg.get("frame_source", "keyframe"), "segments": segs})


def _raw_px():
    """原始每帧的 BEV 像素坐标 (n_raw,2), 缺 VIO 的帧为 nan。缓存。
    帧位姿由 datasets/<seq>/timestamps.txt + vio.txt 插值, 经 occupancy.meta 变换到像素。"""
    if _RAW["px"] is not None:
        return _RAW["px"]
    seg = _lotis_seg() or {}
    n_raw = int(seg.get("n_raw", 0))
    occ = CFG["run"] / f"{CFG['seq']}_occupancy.npz"
    ds = CFG["dataset"]
    try:
        m = json.loads(str(np.load(occ, allow_pickle=True)["meta"]))
    except Exception:
        _RAW["px"] = np.full((n_raw, 2), np.nan); return _RAW["px"]
    # timestamps.txt: "idx ts"; vio.txt: "ts tx ty tz ..."  (world 米制 = kf_pos 同系)
    ts = np.full(n_raw, np.nan)
    tf = ds / "timestamps.txt"
    if tf.exists():
        for ln in tf.read_text().splitlines():
            if ln.startswith("#") or not ln.strip():
                continue
            a = ln.split()
            i = int(a[0])
            if 0 <= i < n_raw:
                ts[i] = float(a[1])
    vt, vx, vy = [], [], []
    vf = ds / "vio.txt"
    if vf.exists():
        for ln in vf.read_text().splitlines():
            if ln.startswith("#") or not ln.strip():
                continue
            a = ln.split()
            vt.append(float(a[0])); vx.append(float(a[1])); vy.append(float(a[2]))
    px = np.full((n_raw, 2), np.nan)
    if vt:
        vt = np.asarray(vt); vx = np.asarray(vx); vy = np.asarray(vy)
        o = np.argsort(vt); vt, vx, vy = vt[o], vx[o], vy[o]
        half, G, ca, cb = m["half"], m["G"], m["ca"], m["cb"]
        for i in range(n_raw):
            if np.isnan(ts[i]) or ts[i] < vt[0] or ts[i] > vt[-1]:
                continue
            x = float(np.interp(ts[i], vt, vx)); y = float(np.interp(ts[i], vt, vy))
            px[i, 0] = (x - ca) / half * (G / 2) + G / 2          # col
            px[i, 1] = -(y - cb) / half * (G / 2) + G / 2         # row
    _RAW["px"] = px
    return px


def _seg_frames():
    """seg_key -> 该段原始帧号列表 (frame_indices)。缓存。"""
    if _RAW["seg_frames"] is None:
        seg = _lotis_seg() or {"segments": []}
        _RAW["seg_frames"] = {
            s["key"]: (s.get("frame_indices") or s.get("kf_indices") or [])
            for s in seg["segments"]}
    return _RAW["seg_frames"]


def _raw_ts():
    """每原始帧成像时间戳(秒), 缺失为 nan。缓存。用于视觉观察按数据集真实节奏逐帧播放。"""
    if _RAW.get("ts") is not None:
        return _RAW["ts"]
    seg = _lotis_seg() or {}
    n_raw = int(seg.get("n_raw", 0))
    ts = np.full(n_raw, np.nan)
    tf = CFG["dataset"] / "timestamps.txt"
    if tf.exists():
        for ln in tf.read_text().splitlines():
            if ln.startswith("#") or not ln.strip():
                continue
            a = ln.split()
            i = int(a[0])
            if 0 <= i < n_raw:
                ts[i] = float(a[1])
    _RAW["ts"] = ts
    return ts


def _step_frames(keys, backward):
    """一步(沿行进方向)的逐帧播放序列: [[frame_id, seg_key, ts_sec], ...]。
    part 是同一趟连续采集的分块 -> 按 keys 顺序拼接即得连续帧序; 反向则段序与段内帧序都倒排。"""
    sf = _seg_frames(); ts = _raw_ts()
    out = []
    for k in keys:
        fi = sf.get(k, [])
        if backward:
            fi = list(reversed(fi))
        for f in fi:
            f = int(f)
            tv = float(ts[f]) if (0 <= f < len(ts) and not np.isnan(ts[f])) else None
            out.append([f, k, tv])
    return out


def _select_query_frame(sel_px, seg_keys):
    """在候选段的原始帧里, 选 BEV 像素离 sel_px 最近的原始帧号 (VIO 位姿)。
    返回 fid 或 None。—— 取代旧的关键帧 nearestKF 选帧逻辑, 直接用原始数据集帧。"""
    px = _raw_px()
    sf = _seg_frames()
    cand = set()
    for k in seg_keys:
        cand.update(sf.get(k, []))
    if not cand:
        return None
    sx, sy = float(sel_px[0]), float(sel_px[1])
    best, bd = None, 1e18
    for fid in cand:
        if fid < 0 or fid >= len(px):
            continue
        p = px[fid]
        if np.isnan(p[0]):
            continue
        d = (p[0] - sx) ** 2 + (p[1] - sy) ** 2
        if d < bd:
            bd = d; best = int(fid)
    return best


def _load_query_img(j):
    """从请求取 query 图, 优先级: image(dataURL/base64) > query_frame(原始帧) > query_kf(关键帧缩略图, 兼容旧)。
    返回 (PIL.Image 或 None, 错误串)。"""
    from PIL import Image
    b64 = j.get("image", "")
    if b64:
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        import io
        try:
            return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"), None
        except Exception as e:
            return None, f"图像解码失败: {e}"
    qf = j.get("query_frame")
    if qf is not None:
        raw = CFG["dataset"] / f"{int(qf):06d}.png"
        if not raw.exists():
            return None, f"原始帧 {int(qf):06d}.png 不存在"
        return Image.open(raw).convert("RGB"), None
    qk = j.get("query_kf")
    if qk is None:
        return None, "缺 query_frame / query_kf / image"
    thumb = CFG["run"] / "web" / "thumbs" / f"kf{int(qk)}.jpg"
    if not thumb.exists():
        return None, f"关键帧缩略图 kf{qk} 不存在"
    return Image.open(thumb).convert("RGB"), None


@app.route("/api/lotis_point", methods=["POST"])
def lotis_point():
    """在某记忆段内, 用一张查询帧做 LoTIS 打点。
    body: {seg_key: str, query_kf: int}  或  {seg_key, image: dataURL/base64}
    返回 {ok, center_pct{x,y}, found, confidence, visible, n_frames,
          closest_frame, aim_frame, crop, frame_wh}"""
    j = request.json or {}
    seg_key = j.get("seg_key", "")
    if not seg_key:
        return jsonify({"ok": False, "reason": "缺 seg_key"})
    eng, seg = _lotis_engine()
    if eng is None:
        return jsonify({"ok": False, "reason": seg})
    if not eng.has_key(seg_key):
        return jsonify({"ok": False, "reason": f"段 {seg_key} 无编码"})
    img, err = _load_query_img(j)
    if img is None:
        return jsonify({"ok": False, "reason": err})
    backward = bool(j.get("backward", False))
    try:
        with _LOTIS["lock"]:            # 串行化模型推理
            res = eng.point(img, seg_key=seg_key, backward=backward)
    except Exception as e:
        return jsonify({"ok": False, "reason": f"打点失败: {e}"})
    res["ok"] = True
    res["crop"] = seg["crop"]
    res["frame_wh"] = seg["frame_wh"]
    return jsonify(res)


@app.route("/api/lotis_point_auto", methods=["POST"])
def lotis_point_auto():
    """在一组候选段里用同一 query 帧各打一次, 返回定位最好的那段(自动选段)。
    body: {seg_keys:[...], backward, 以及 query 帧来源之一:
           sel_px:[col,row] (BEV 像素, 后端在候选段原始帧里选最近的 query_frame) /
           query_frame(原始帧) / query_kf(旧关键帧) / image}
    评分: found 优先 -> confidence -> visible。返回最优段结果 + best_seg + query_frame + tried。"""
    j = request.json or {}
    keys = j.get("seg_keys") or []
    if not keys:
        return jsonify({"ok": False, "reason": "缺 seg_keys"})
    eng, seg = _lotis_engine()
    if eng is None:
        return jsonify({"ok": False, "reason": seg})
    # sel_px: 后端按机器人位置在候选段原始帧里选 query_frame (取代前端关键帧选帧)
    if j.get("query_frame") is None and j.get("image") is None and j.get("sel_px"):
        qf = _select_query_frame(j["sel_px"], keys)
        if qf is None:
            return jsonify({"ok": False, "reason": "候选段原始帧无 VIO 位姿, 无法按位置选帧"})
        j["query_frame"] = qf
    img, err = _load_query_img(j)
    if img is None:
        return jsonify({"ok": False, "reason": err})
    backward = bool(j.get("backward", False))
    best, tried = None, []
    with _LOTIS["lock"]:               # 串行化模型推理
        for k in keys:
            if not eng.has_key(k):
                continue
            try:
                r = eng.point(img, seg_key=k, backward=backward)
            except Exception:
                continue
            conf = float(r.get("confidence") or 0.0)
            score = (1 if r.get("found") else 0, conf, int(r.get("visible") or 0))
            tried.append({"seg": k, "found": bool(r.get("found")),
                          "conf": round(conf, 3), "visible": int(r.get("visible") or 0)})
            if best is None or score > best[0]:
                best = (score, k, r)
    if best is None:
        return jsonify({"ok": False, "reason": "候选段均无编码/打点失败"})
    _, bk, res = best
    res["ok"] = True
    res["best_seg"] = bk
    res["backward"] = backward
    res["crop"] = seg["crop"]
    res["frame_wh"] = seg["frame_wh"]
    res["query_frame"] = j.get("query_frame")   # 前端据此显示 /raw/{query_frame}
    res["tried"] = tried
    return jsonify(res)


# --------------------------------------------------------------------------- #
# LoTIS 拓扑图 + 路径规划 (第二部分: walk 边 + 回环捷径边 -> Dijkstra)
# --------------------------------------------------------------------------- #
_LGRAPH = {"g": None}
SHORTCUT_MAX_M = 2.5     # 物理 < 该值且 walk 非相邻 -> 回环捷径边
SHORTCUT_MIN_WALKGAP = 2  # walk 顺序间隔 >= 该值才算"回环"(相邻边已是 walk 边)


def _build_lotis_graph():
    """拓扑图: 节点=语义节点; 边=① walk 边(相邻节点, 有 LoTIS 段, 双向可打点)
    ② 回环捷径边(物理<2.5m 但 walk 不相邻, 无段, 物理近直接跨)。边权=米制距离。"""
    if _LGRAPH["g"] is not None:
        return _LGRAPH["g"]
    seg = _lotis_seg()
    if seg is None:
        return None
    import math
    from collections import defaultdict
    nodes = _load_nodes()
    pos = [list((n.get("position") or [0, 0, 0])[:2]) for n in nodes]
    anchor = [min(n["kf_indices"]) for n in nodes]
    order = sorted(range(len(nodes)), key=lambda i: anchor[i])
    posw = [0] * len(nodes)
    for w, i in enumerate(order):
        posw[i] = w

    def dist(i, j):
        return math.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1])

    # walk 边: 聚合 edge 段; mined 边: 聚合 mined 段(空间闭环, 有真实影像, 同样可打点)
    walk = defaultdict(list)
    mined = defaultdict(list)
    for s in seg["segments"]:
        t = s.get("type")
        if t == "edge":
            walk[(s["node_from"], s["node_to"])].append((s.get("part", 0), s["key"]))
        elif t == "mined":
            mined[(s["node_from"], s["node_to"])].append((s.get("part", 0), s["key"]))
    walk_edges = {ab: [k for _, k in sorted(lst)] for ab, lst in walk.items()}
    mined_edges = {ab: [k for _, k in sorted(lst)] for ab, lst in mined.items()}

    edges = []
    for (a, b), keys in walk_edges.items():
        edges.append({"u": a, "v": b, "kind": "walk", "seg_keys": keys, "weight": dist(a, b)})
    # mined 边取代旧盲捷径: 有 seg_keys -> 可沿真实影像打点(正/反向), 边权=米制直线距
    for (a, b), keys in mined_edges.items():
        edges.append({"u": a, "v": b, "kind": "mined", "seg_keys": keys, "weight": dist(a, b)})

    g = {"nodes": [{"id": i, "name": nodes[i].get("name", ""),
                    "category": nodes[i].get("category", ""), "pos": pos[i],
                    "walk_order": posw[i]} for i in range(len(nodes))],
         "edges": edges}
    _LGRAPH["g"] = g
    return g


def _dijkstra(g, s, t):
    """无向 Dijkstra(边权=米制距离)。返回 (path_nodes, [(u,v,edge)...]) 或 None。"""
    import heapq
    from collections import defaultdict
    if s == t:
        return [s], []
    adj = defaultdict(list)
    for e in g["edges"]:
        adj[e["u"]].append((e["v"], e))
        adj[e["v"]].append((e["u"], e))
    D = {s: 0.0}
    prev = {}
    pq = [(0.0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == t:
            break
        if d > D.get(u, 1e18):
            continue
        for v, e in adj[u]:
            nd = d + e["weight"]
            if nd < D.get(v, 1e18):
                D[v] = nd
                prev[v] = (u, e)
                heapq.heappush(pq, (nd, v))
    if t not in D:
        return None
    path = [t]
    steps = []
    cur = t
    while cur != s:
        u, e = prev[cur]
        steps.append((u, cur, e))
        cur = u
        path.append(cur)
    path.reverse()
    steps.reverse()
    return path, steps


@app.route("/api/lotis_graph", methods=["GET"])
def lotis_graph():
    """拓扑图(供前端画节点连线): {ok, nodes:[{id,name,pos,walk_order}], edges:[{u,v,kind}]}"""
    g = _build_lotis_graph()
    if g is None:
        return jsonify({"ok": False, "reason": "无 LoTIS 分段"})
    return jsonify({"ok": True, "nodes": g["nodes"],
                    "edges": [{"u": e["u"], "v": e["v"], "kind": e["kind"],
                               "weight": round(e["weight"], 2)} for e in g["edges"]]})


@app.route("/api/lotis_route", methods=["POST"])
def lotis_route():
    """起点->终点在拓扑图上 Dijkstra(抄近路闭合回环)。body {start_node, goal_node}
    返回 {ok, path_nodes, path_names, steps:[{from,to,kind,backward,seg_keys}]}。
    walk 步: seg_keys 按行进方向排(反向则 reversed + backward=True); 捷径步: 物理近直接跨。"""
    j = request.json or {}
    s, t = j.get("start_node"), j.get("goal_node")
    g = _build_lotis_graph()
    if g is None:
        return jsonify({"ok": False, "reason": "无 LoTIS 图"})
    if s is None or t is None:
        return jsonify({"ok": False, "reason": "缺 start_node/goal_node"})
    r = _dijkstra(g, int(s), int(t))
    if r is None:
        return jsonify({"ok": False, "reason": "拓扑图上无可达路径"})
    path, steps = r
    out = []
    for u, v, e in steps:
        if e["kind"] in ("walk", "mined"):            # 两者都有影像可打点(正/反向)
            forward = (e["u"] == u and e["v"] == v)   # 行进方向 vs 段方向(node_from->node_to)
            keys = e["seg_keys"] if forward else list(reversed(e["seg_keys"]))
            out.append({"from": u, "to": v, "kind": e["kind"],
                        "backward": (not forward), "seg_keys": keys,
                        "frames": _step_frames(keys, not forward)})
        else:
            out.append({"from": u, "to": v, "kind": "shortcut",
                        "backward": False, "seg_keys": [], "dist_m": round(e["weight"], 2)})
    return jsonify({"ok": True, "path_nodes": path,
                    "path_names": [g["nodes"][i]["name"] for i in path], "steps": out})


def main():
    from mast3r_slam.run_config import load_run_config, run_dir, seq_name
    rc = load_run_config()  # 默认值取自 nav_config.yaml, CLI 参数优先
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(run_dir(rc)))
    ap.add_argument("--seq", default=seq_name(rc))
    ap.add_argument("--port", type=int, default=int(rc.get("web_port", 8080)))
    ap.add_argument("--api", default=rc.get("semantic_api", "http://192.168.50.72:8299/v1"),
                    help="vLLM 服务(自然语言匹配用)")
    ap.add_argument("--model", default=rc.get("semantic_model", "qwen3.5-35b-a3b"))
    ap.add_argument("--dataset", default=None,
                    help="原始数据集目录(LoTIS query 原图), 默认 datasets/<seq>")
    args = ap.parse_args()
    dataset = pathlib.Path(args.dataset).resolve() if args.dataset \
        else (ROOT.parent / "datasets" / args.seq)
    CFG.update(run=pathlib.Path(args.run).resolve(), seq=args.seq,
               api=args.api, model=args.model, dataset=dataset)
    if not (CFG["run"] / "web" / "data.js").exists():
        print(f"[server] 缺 {CFG['run']}/web/data.js, 先跑 nav_web/export_web.py")
        sys.exit(1)
    print(f"[server] http://localhost:{args.port}  (run={args.run}, seq={args.seq})")
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
