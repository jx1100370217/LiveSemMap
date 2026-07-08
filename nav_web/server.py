#!/usr/bin/env python3
"""语义导航 Web 服务 (Flask)。

- 静态: nav.html / data.js / thumbs/
- POST /api/nl_query    自然语言 -> 语义节点/关键帧 (调 L40 vLLM, 与建图共用同一服务)
- POST /api/locate_image 上传观察图像 -> SelaVPR 描述子 -> 最近关键帧 (VPR 重定位)

用法: python nav_web/server.py --run logs/semantic_v1 --seq insight9 [--port 8080]
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

app = Flask(__name__)
CFG = {}
_VPR = {"ex": None, "lock": threading.Lock()}   # SelaVPR 懒加载


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
        }, timeout=30)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--api", default="http://192.168.50.72:8299/v1",
                    help="vLLM 服务(自然语言匹配用)")
    ap.add_argument("--model", default="qwen3.5-9b")
    args = ap.parse_args()
    CFG.update(run=pathlib.Path(args.run).resolve(), seq=args.seq,
               api=args.api, model=args.model)
    if not (CFG["run"] / "web" / "data.js").exists():
        print(f"[server] 缺 {CFG['run']}/web/data.js, 先跑 nav_web/export_web.py")
        sys.exit(1)
    print(f"[server] http://localhost:{args.port}  (run={args.run}, seq={args.seq})")
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
