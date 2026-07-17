"""VLM 区域生长引擎 (在线/离线共用帧流接口)。

每帧: Qwen 判定 same_space / returned_region -> 区域生长/切换/回访合并;
帧点云 2D 足迹进全局投票网格 (cell -> {region: count}), 平面图按多数归属着色。
判定链有上下文依赖 (活动区域描述), 故单线程顺序处理; 命名调用异步不阻塞。
"""
import json
import pathlib
import threading
import time
from collections import Counter

import numpy as np

from .prompts import JUDGE_PROMPT, JUDGE_SCHEMA, NAME_PROMPT, NAME_SCHEMA

VOTE_VOXEL = 0.15          # 足迹投票网格 (米)
RETURN_CONFIRM_M = 3.0     # 回访几何确认: 相机距目标区域足迹的最大距离


def _b64_img(p, half=False):
    import base64
    import io

    from PIL import Image
    im = Image.open(p)
    if half:
        im = im.resize((im.width // 2, im.height // 2), Image.BILINEAR)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


class VLMRegionEngine:
    def __init__(self, surround_dir, dataset_dir, web_dir, seq,
                 api_url, model, semantic_ann=None, live=None, timeout=90,
                 judge_thin=0.4):
        self.surround = pathlib.Path(surround_dir)
        self.ds = pathlib.Path(dataset_dir)
        self.web_dir = pathlib.Path(web_dir)
        self.seq = seq
        self.api, self.model = api_url.rstrip("/"), model
        self.sem_ann = semantic_ann if semantic_ann is not None else {}
        self.live = live
        self.judge_thin = float(judge_thin)
        self.timeout = timeout

        self.regions = {}          # rid -> dict
        self.cur = None            # 活动区域 rid
        self.edges = []            # {a,b,via_kf,via_fid,reason}
        self.votes = {}            # (ix,iy) -> Counter{rid: n}
        self.frames = {}           # kf -> {fid,xy,pose,desc,rid}
        self._rid_next = 0
        self._region_tree = {}     # rid -> (cKDTree of member cam_xy, 缓存)
        self._lock = threading.Lock()
        self._last_snap = 0.0
        self.n_llm = 0
        self.n_skip = 0            # 空间抽稀跳过的判定数
        self._last_judge_xy = None

    # ---------------- VLM ----------------
    def _chat(self, content, schema, max_tokens=400):
        import requests
        r = requests.post(
            f"{self.api}/chat/completions",
            json={"model": self.model,
                  "messages": [{"role": "user", "content": content}],
                  "max_tokens": max_tokens, "temperature": 0.0,
                  "chat_template_kwargs": {"enable_thinking": False},
                  "response_format": {"type": "json_schema",
                                      "json_schema": {"name": "out",
                                                      "schema": schema}}},
            timeout=self.timeout, proxies={"http": None, "https": None})
        r.raise_for_status()
        self.n_llm += 1
        return json.loads(r.json()["choices"][0]["message"]["content"])

    def _judge(self, fid, cam_xy=None):
        """当前帧: 前视高清 + 4环视拼图 + 区域上下文 -> 判定。"""
        from mast3r_slam.semantic import _surround_grid
        front = self.surround / f"{fid:06d}_0.jpg"
        fisheyes = {k: self.surround / f"{fid:06d}_{k}.jpg" for k in range(1, 5)}
        fisheyes = {k: p for k, p in fisheyes.items() if p.exists()}
        content = []
        if front.exists():
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{_b64_img(front, half=True)}"}})
        grid = _surround_grid(fisheyes)
        if grid is not None:
            import base64
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64,"
                       + base64.b64encode(grid).decode()}})
        if self.cur is not None:
            r = self.regions[self.cur]
            cur_ctx = (f"R{self.cur}: {r['summary']} (形状:{r['shape']}, "
                       f"类型:{r['kind']}); 你上次在其中的位置: {r['last_pos']}")
        else:
            cur_ctx = "(无 —— 这是建图的第一帧, same_space 填 false, "\
                      "returned_region 填 -1)"
        cands = []
        for rid, r in self.regions.items():
            if rid == self.cur or not r["frames"]:
                continue
            pts = np.array([self.frames[k]["xy"] for k in r["frames"][-120:]])
            d = float(np.linalg.norm(pts - cam_xy, axis=1).min()) \
                if cam_xy is not None else 0.0
            cands.append((d, rid, r))
        cands.sort(key=lambda x: x[0])
        hist = [(rid, r, d) for d, rid, r in cands if d < 10.0][:6]
        hist_ctx = "\n".join(
            f"- R{rid}: {r.get('name') or r['summary']} (距当前位置约{d:.0f}米)"
            for rid, r, d in hist) or "(暂无)"
        content.append({"type": "text", "text": JUDGE_PROMPT.format(
            cur_ctx=cur_ctx, hist_ctx=hist_ctx)})
        return self._chat(content, JUDGE_SCHEMA)

    # ---------------- 区域生命周期 ----------------
    def _new_region(self, out):
        rid = self._rid_next
        self._rid_next += 1
        self.regions[rid] = {
            "id": rid, "kind": out["space_kind"], "shape": out["shape"],
            "summary": out["space_summary"], "last_pos": out["my_position"],
            "kinds": Counter([out["space_kind"]]),
            "frames": [], "name": "", "type_zh": "", "name_summary": "",
            "named": False}
        return rid

    def _close_region(self, rid):
        r = self.regions[rid]
        if r["named"] or len(r["frames"]) < 2:
            return
        r["named"] = True
        threading.Thread(target=self._name_region, args=(rid,),
                         daemon=True).start()

    def _name_region(self, rid):
        r = self.regions[rid]
        try:
            descs = [self.frames[k]["desc"] for k in r["frames"]
                     if self.frames[k].get("desc")]
            descs = descs[:: max(1, len(descs) // 6)][:6]
            sigs = set()
            for k in r["frames"]:
                a = self.sem_ann.get(k)
                if not a:
                    continue
                sigs |= {s for s in (a.get("signage") or []) if s}
                if a.get("landmark") and a.get("name"):
                    sigs.add(a["name"])
            fids = sorted(self.frames[k]["fid"] for k in r["frames"])
            imgs = []
            for fr in (fids[len(fids) // 3], fids[2 * len(fids) // 3]):
                p = self.ds / f"{fr:06d}.png"
                if p.exists():
                    imgs.append({"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{_b64_img(p)}"}})
            content = imgs + [{"type": "text", "text": NAME_PROMPT.format(
                summaries="\n".join(f"  - {d}" for d in descs) or "  (无)",
                signage=", ".join(sorted(sigs)[:8]) or "(无)",
                n_img=len(imgs))}]
            out = self._chat(content, NAME_SCHEMA, max_tokens=260)
            r["name"], r["type_zh"] = out["name"], out["room_type"]
            r["name_summary"] = out["summary"]
            print(f"[vlm-region] 命名 R{rid}: {out['name']} "
                  f"[{out['room_type']}]", flush=True)
            if self.live is not None:      # 名字上图: 立即刷新 live 语义地图
                self._push_live()
        except Exception as e:
            r["named"] = False
            print(f"[vlm-region] R{rid} 命名失败: {e}", flush=True)

    # ---------------- 帧处理 (核心) ----------------
    def process_frame(self, kf_idx, fid, pose, foot2d=None):
        """顺序调用。pose: c2w(4,4); foot2d: 该帧点云 2D 足迹 (M,2) 或 None。"""
        cam_xy = np.asarray(pose[:2, 3], np.float64)
        # 空间抽稀: 活动区域内位移不足 judge_thin 的帧空间身份不会变,
        # 跳过 VLM 判定, 帧与足迹照常并入活动区域 (在线 kf 多为原地旋转/
        # 微动产生, floor1 实测 0.4m 可跳过 61%); judge_thin=0 关闭抽稀
        if self.judge_thin > 0 and self.cur is not None \
                and self._last_judge_xy is not None \
                and float(np.linalg.norm(cam_xy - self._last_judge_xy)) \
                < self.judge_thin:
            out = None
            self.n_skip += 1
        else:
            try:
                self._last_judge_xy = cam_xy
                out = self._judge(int(fid), cam_xy)
            except Exception as e:
                print(f"[vlm-region] kf{kf_idx} 判定失败(帧并入活动区域): "
                      f"{e}", flush=True)
                out = None
        prev = self.cur
        if out is None:
            if self.cur is None:
                return
        elif self.cur is None:
            self.cur = self._new_region(out)
            print(f"[vlm-region] kf{kf_idx} 起始区域 R{self.cur}: "
                  f"{out['space_summary']}", flush=True)
        elif out["same_space"]:
            r = self.regions[self.cur]
            r["summary"] = out["space_summary"]
            r["last_pos"] = out["my_position"]
            r["kinds"][out["space_kind"]] += 1
            r["kind"] = r["kinds"].most_common(1)[0][0]
        else:
            ret = int(out.get("returned_region", -1))
            target = None
            if ret in self.regions and ret != self.cur:
                # 回访几何确认: 相机须贴近该区域已有足迹
                pts = np.array([self.frames[k]["xy"]
                                for k in self.regions[ret]["frames"]])
                if len(pts) and np.linalg.norm(pts - cam_xy, axis=1).min() \
                        < RETURN_CONFIRM_M:
                    target = ret
            self._close_region(self.cur)
            if target is not None:
                self.cur = target
                r = self.regions[target]
                r["summary"] = out["space_summary"]
                r["last_pos"] = out["my_position"]
                print(f"[vlm-region] kf{kf_idx} 回访 R{target} "
                      f"({r.get('name') or r['kind']})", flush=True)
            else:
                self.cur = self._new_region(out)
                print(f"[vlm-region] kf{kf_idx} 新区域 R{self.cur} "
                      f"[{out['space_kind']}] {out['space_summary'][:30]} "
                      f"<- {out['exit_reason']}", flush=True)
            if prev is not None and prev != self.cur:
                self.edges.append({"a": prev, "b": self.cur,
                                   "via_kf": int(kf_idx), "via_fid": int(fid),
                                   "reason": out["exit_reason"]})
        # 帧与足迹并入活动区域
        rid = self.cur
        with self._lock:
            self.frames[kf_idx] = {
                "fid": int(fid), "xy": cam_xy, "pose": np.asarray(pose),
                "desc": (out or {}).get("space_summary", ""), "rid": rid}
            self.regions[rid]["frames"].append(kf_idx)
            # 相机圆盘 (r=0.4m): 无点云足迹时区域仍有带宽
            rr = int(np.ceil(0.4 / VOTE_VOXEL))
            disc = [(cam_xy / VOTE_VOXEL + [dx, dy]).astype(int)
                    for dx in range(-rr, rr + 1) for dy in range(-rr, rr + 1)
                    if dx * dx + dy * dy <= rr * rr]
            cells = {tuple(c) for c in disc}
            if foot2d is not None and len(foot2d):
                cells |= {tuple(c) for c in
                          np.floor(foot2d / VOTE_VOXEL).astype(int)}
            for c in cells:
                self.votes.setdefault(c, Counter())[rid] += 1
        if self.live is not None and kf_idx % 5 == 0:
            self._push_live()
        if time.time() - self._last_snap > 30:
            self._last_snap = time.time()
            try:
                self._export_web()
            except Exception as e:
                print(f"[vlm-region] web 快照失败: {e}", flush=True)

    # ---------------- 在线模式 (SLAM 挂载): 队列 + worker 自取点云足迹 ----------------
    def start_online(self, keyframes, vio_prior):
        import queue
        self.keyframes = keyframes
        self.vio = vio_prior
        self.q = queue.Queue()
        self.submitted = set()
        threading.Thread(target=self._worker_run, daemon=True).start()

    def catch_up(self):
        n = len(self.keyframes)
        for i in range(n):
            if i in self.submitted:
                continue
            self.submitted.add(i)
            with self.keyframes.lock:
                fid = int(self.keyframes.dataset_idx[i])
            self.q.put((i, fid))

    def pending(self):
        return self.q.qsize() if hasattr(self, "q") else 0

    def drain(self):
        while hasattr(self, "q") and self.q.unfinished_tasks > 0:
            print(f"[vlm-region] 等待剩余 ~{self.q.qsize()} 关键帧判定...",
                  flush=True)
            time.sleep(3.0)

    def _worker_run(self):
        while True:
            i, fid = self.q.get()
            try:
                s = self.vio.metric_scale() if self.vio is not None else 1.0
                if self.vio is not None and s is None:
                    self.q.put((i, fid))       # 尺度未标定, 稍后再处理
                    time.sleep(1.5)
                    continue
                p, Rm = self.vio._pose_at(fid)
                pose = np.eye(4)
                pose[:3, :3] = Rm.as_matrix()
                pose[:3, 3] = p
                foot2d = self._footprint(i, pose, float(s or 1.0))
                self.process_frame(i, fid, pose, foot2d)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[vlm-region] kf{i} 在线处理失败: {e}", flush=True)
            finally:
                self.q.task_done()

    def _footprint(self, kf_idx, pose, s):
        """该关键帧点图 -> 地面带 2D 足迹 (在线才有; 失败返回 None)。"""
        try:
            with self.keyframes.lock:
                kf = self.keyframes[kf_idx]
                h, w = [int(x) for x in kf.img_shape.flatten()[:2]]
                X = kf.X_canon.reshape(h, w, 3)[::3, ::3].cpu().numpy()
                conf = kf.get_average_conf().reshape(h, w)[::3, ::3]\
                    .cpu().numpy()
            Xw = (X.astype(np.float64) * s) @ pose[:3, :3].T + pose[:3, 3]
            cam = pose[:3, 3]
            m = (conf > 1.5) & np.isfinite(Xw).all(-1) \
                & (np.linalg.norm(Xw - cam, axis=-1) < 5.0) \
                & (Xw[..., 2] > cam[2] - 1.2) & (Xw[..., 2] < cam[2] + 0.4)
            pts = Xw[m][:, :2]
            if not len(pts):
                return None
            key = np.unique(np.floor(pts / VOTE_VOXEL).astype(np.int64),
                            axis=0)
            return (key + 0.5) * VOTE_VOXEL
        except Exception:
            return None

    # ---------------- 足迹归属 / 导出 ----------------
    def region_cells(self):
        """投票网格 -> {rid: (N,2) 米制 cell 中心}。"""
        by = {}
        with self._lock:
            for c, cnt in self.votes.items():
                rid = cnt.most_common(1)[0][0]
                by.setdefault(rid, []).append(c)
        return {rid: (np.asarray(cs, np.float64) + 0.5) * VOTE_VOXEL
                for rid, cs in by.items()}

    def _room_structs(self):
        from mast3r_slam.hmsg.graph import ROOM_PALETTE
        cells = self.region_cells()
        rooms = []
        order = sorted(self.regions)
        for i, rid in enumerate(order):
            r = self.regions[rid]
            v = cells.get(rid)
            if v is None or len(v) < 8:
                continue
            rooms.append({"id": f"R{rid}",
                          "name": r["kind"],
                          "name_zh": r.get("name") or "",
                          "type_zh": r.get("type_zh") or r["kind"],
                          "summary_zh": r.get("name_summary")
                          or r["summary"],
                          "color": ROOM_PALETTE[i % len(ROOM_PALETTE)],
                          "vertices": v,
                          "n_views": len(r["frames"]), "n_objects": 0,
                          "rep_feats": []})
        return rooms

    def _push_live(self):
        """渲染当前语义地图 (与导航 Web 同观感) 推给 viewer 直接贴图显示。"""
        try:
            img = self._render_live_map()
            if img is not None:
                self.live["map"] = img                      # uint8 (G,G,3)
                self.live["map_v"] = int(self.live.get("map_v", 0)) + 1
        except Exception as e:
            print(f"[vlm-region] live 推送失败: {e}", flush=True)

    def _render_live_map(self, px_max=520):
        """在线语义地图渲染: 暗底 + 区域实色块/深描边 + 关键帧白点 +
        区域中文名 + 语义 POI 节点。世界系俯视, 包围盒自适应增长。"""
        from PIL import Image as PImage
        from PIL import ImageDraw

        from mast3r_slam.hmsg.graph import ROOM_PALETTE
        from mast3r_slam.mapping2d import _get_font
        cells = self.region_cells()
        with self._lock:
            kf_xy = np.array([m["xy"] for m in self.frames.values()],
                             np.float64).reshape(-1, 2)
            order = sorted(self.regions)
            names = [self.regions[r].get("name")
                     or self.regions[r]["kind"] for r in order]
        pts_all = [v for v in cells.values() if len(v)]
        if len(kf_xy):
            pts_all.append(kf_xy)
        if not pts_all:
            return None
        P = np.concatenate(pts_all, 0)
        lo = P.min(0) - 1.5
        span = float(max((P.max(0) - lo).max(), 3.0)) + 1.5
        res = max(VOTE_VOXEL, span / px_max)    # 0.15m 投票格; 超大场景再放粗
        G0 = int(np.ceil(span / res)) + 1
        up = max(1, round(px_max / G0))         # 整数倍 NEAREST 放大 -> 字/点清晰
        G = G0 * up

        def to_px(p):
            q = (np.asarray(p, np.float64).reshape(-1, 2) - lo) / res
            return q[:, 0] * up, (G0 - 1 - q[:, 1]) * up   # y 翻转 (图像行向下)

        lab = np.full((G0, G0), -1, np.int16)
        colors = []
        for i, rid in enumerate(order):
            col = ROOM_PALETTE[i % len(ROOM_PALETTE)]
            colors.append([int(col[j:j + 2], 16) for j in (1, 3, 5)])
            v = cells.get(rid)
            if v is None or not len(v):
                continue
            q = (np.asarray(v, np.float64) - lo) / res
            xi = np.round(q[:, 0]).astype(int)
            yi = np.round(G0 - 1 - q[:, 1]).astype(int)
            ok = (xi >= 0) & (xi < G0) & (yi >= 0) & (yi < G0)
            lab[yi[ok], xi[ok]] = i
        bg = np.array([13, 17, 26], np.float32)
        img = np.tile(bg, (G0, G0, 1))
        inner = lab >= 0
        edge = np.zeros((G0, G0), bool)              # 区域边界 (4邻不同) 深描边
        edge[1:] |= inner[1:] & (lab[1:] != lab[:-1])
        edge[:-1] |= inner[:-1] & (lab[:-1] != lab[1:])
        edge[:, 1:] |= inner[:, 1:] & (lab[:, 1:] != lab[:, :-1])
        edge[:, :-1] |= inner[:, :-1] & (lab[:, :-1] != lab[:, 1:])
        carr = np.asarray(colors, np.float32).reshape(-1, 3)
        if inner.any():
            img[inner] = carr[lab[inner]] * 0.8 + bg * 0.2
            img[edge] = carr[lab[edge]] * 0.5
        u8 = img.astype(np.uint8).repeat(up, 0).repeat(up, 1)
        pil = PImage.fromarray(u8)
        d = ImageDraw.Draw(pil)
        if len(kf_xy):                               # 关键帧落点 (白点)
            xs, ys = to_px(kf_xy)
            kr = max(1.3, 0.85 * up)
            for x, y in zip(xs, ys):
                d.ellipse([x - kr, y - kr, x + kr, y + kr],
                          fill=(232, 238, 248), outline=(15, 20, 30))
        # 语义 POI 节点: 类别色圆点 + 名字
        try:
            from mast3r_slam.semantic import (SEMANTIC_CATEGORIES,
                                              aggregate_nodes)
            with self._lock:
                pos = {k: (float(m["xy"][0]), float(m["xy"][1]), 0.0)
                       for k, m in self.frames.items()}
            nodes = aggregate_nodes(dict(self.sem_ann), pos)
        except Exception:
            nodes = []
        sfont = _get_font(max(9, G // 52))
        for n in nodes:
            _, c, _ = SEMANTIC_CATEGORIES.get(n["category"],
                                              ("?", (.8, .8, .8), False))
            rgb = tuple(int(v * 255) for v in c)
            x, y = to_px(np.asarray(n["position"][:2]))
            x, y = float(x[0]), float(y[0])
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=rgb,
                      outline=(255, 255, 255))
            if sfont is not None and n.get("name"):
                d.text((x + 5, y - sfont.size / 2), n["name"], font=sfont,
                       fill=(232, 238, 248), stroke_width=2,
                       stroke_fill=(10, 14, 22))
        # 区域中文名 (同名只标一次, 已放置标签简单避让)
        font = _get_font(max(11, G // 36))
        if font is not None:
            placed, drawn = [], set()
            for i, rid in enumerate(order):
                nm = names[i]
                if not nm or nm in drawn:
                    continue
                ys_, xs_ = np.nonzero(lab == i)
                if not len(xs_):
                    continue
                cx = float(np.median(xs_)) * up
                cy = float(np.median(ys_)) * up
                tw = d.textlength(nm, font=font)
                box = (cx - tw / 2, cy - font.size / 2, tw, font.size * 1.2)
                if any(box[0] < p[0] + p[2] and box[0] + box[2] > p[0]
                       and box[1] < p[1] + p[3] and box[1] + box[3] > p[1]
                       for p in placed):
                    continue
                placed.append(box)
                drawn.add(nm)
                d.text((np.clip(box[0], 1, G - tw - 1), max(1, box[1])), nm,
                       font=font, fill=(255, 255, 255), stroke_width=2,
                       stroke_fill=(10, 12, 20))
        return np.asarray(pil)

    def _export_web(self):
        from mast3r_slam.hmsg.webexport import export_web_data
        rooms = self._room_structs()
        if not rooms:
            return
        with self._lock:
            frames = dict(self.frames)
        views = [{"id": f"R{m['rid']}_{k}", "room": f"R{m['rid']}",
                  "img_id": m["fid"], "pose": m["pose"],
                  "desc": m["desc"], "objects": []}
                 for k, m in sorted(frames.items())]
        export_web_data(self.seq, self.web_dir, rooms, [], views, quiet=True)

    def _absorb_fragments(self, min_frames=3):
        """帧数过少的碎片区域并入其时序前驱 (走廊转角抖动/误切的回收)。"""
        for rid in sorted(self.regions):
            r = self.regions[rid]
            if len(r["frames"]) >= min_frames or rid == self.cur:
                continue
            target = None
            for e in self.edges:          # 切出该碎片的来边 -> 前驱
                if e["b"] == rid and e["a"] in self.regions:
                    target = e["a"]
            if target is None:
                for e in self.edges:
                    if e["a"] == rid and e["b"] in self.regions:
                        target = e["b"]
            if target is None or target == rid:
                continue
            for k in r["frames"]:
                self.frames[k]["rid"] = target
                self.regions[target]["frames"].append(k)
            with self._lock:
                for cnt in self.votes.values():
                    if rid in cnt:
                        cnt[target] += cnt.pop(rid)
            for e in self.edges:
                if e["a"] == rid:
                    e["a"] = target
                if e["b"] == rid:
                    e["b"] = target
            del self.regions[rid]
        self.edges = [e for e in self.edges if e["a"] != e["b"]]
        seen, dedup = set(), []
        for e in self.edges:
            key = (min(e["a"], e["b"]), max(e["a"], e["b"]))
            if key in seen:
                continue
            seen.add(key)
            dedup.append(e)
        self.edges = dedup

    @staticmethod
    def _coarse_kind(k):
        """kind -> 粗类 (通道/房间): 跨粗类的区域从不合并。"""
        k = k or ""
        return "corridor" if any(w in k for w in (
            "走廊", "过道", "门厅", "前室", "电梯", "大堂", "大厅", "通道")) \
            else "room"

    def _merge_overlapping(self, share_th=0.40):
        """来回/误切换分裂区域的几何合并兜底 (v2)。

        floor28 实测教训: 命名后 type_zh 字符串比较过脆 (办公走廊 vs
        电梯前室), 同名又可能空间不相干 (两对同名区域重叠为 0)。规则:
        1. 时序三明治 A->B->A (B 是 A 内部的误切换碎段) + 同粗类 +
           膨胀重叠 >= 0.25;
        2. 生长期多数票 kind 完全相同 + 膨胀重叠 >= share_th (来/去两道)。
        合并组的名字/类型继承帧数最多的成员。"""
        cells = {}
        with self._lock:
            for c, cnt in self.votes.items():
                cells.setdefault(cnt.most_common(1)[0][0], set()).add(c)
        rids = sorted(self.regions)
        parent = {r: r for r in rids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def _neigh(cs):
            out = set()
            for (a, b) in cs:
                out |= {(a + i, b + j) for i in (-1, 0, 1) for j in (-1, 0, 1)}
            return out
        # 时序切换链 -> 三明治对 (A,B,A)
        chain = []
        with self._lock:
            for k in sorted(self.frames):
                rid = self.frames[k]["rid"]
                if not chain or chain[-1] != rid:
                    chain.append(rid)
        sandwich = {frozenset((chain[i - 1], chain[i]))
                    for i in range(1, len(chain) - 1)
                    if chain[i - 1] == chain[i + 1]}
        for i, ra in enumerate(rids):
            for rb in rids[i + 1:]:
                A, B = cells.get(ra, set()), cells.get(rb, set())
                if not A or not B:
                    continue
                ka, kb = self.regions[ra]["kind"], self.regions[rb]["kind"]
                if self._coarse_kind(ka) != self._coarse_kind(kb):
                    continue
                inter = len(_neigh(A) & B) / min(len(A), len(B))
                hit = (frozenset((ra, rb)) in sandwich and inter >= 0.25) \
                    or (ka == kb and inter >= share_th)
                if hit:
                    parent[find(rb)] = find(ra)
        groups = {}
        for r in rids:
            groups.setdefault(find(r), []).append(r)
        for root, members in groups.items():
            if len(members) > 1:      # 名字/类型继承帧数最多的成员
                best = max(members, key=lambda m: len(self.regions[m]["frames"]))
                for f in ("kind", "name", "type_zh", "summary", "name_summary"):
                    if self.regions[best].get(f):
                        self.regions[root][f] = self.regions[best][f]
            for m in members:
                if m == root:
                    continue
                r = self.regions[m]
                for k in r["frames"]:
                    self.frames[k]["rid"] = root
                    self.regions[root]["frames"].append(k)
                with self._lock:
                    for cnt in self.votes.values():
                        if m in cnt:
                            cnt[root] += cnt.pop(m)
                for e in self.edges:
                    if e["a"] == m:
                        e["a"] = root
                    if e["b"] == m:
                        e["b"] = root
                if not self.regions[root].get("name") and r.get("name"):
                    self.regions[root]["name"] = r["name"]
                    self.regions[root]["type_zh"] = r["type_zh"]
                if self.cur == m:
                    self.cur = root
                del self.regions[m]
                print(f"[vlm-region] 几何合并 R{m} -> R{root}", flush=True)
        self.edges = [e for e in self.edges if e["a"] != e["b"]]

    def finalize(self, save_dir, wait_naming=90.0):
        self._absorb_fragments()
        self._merge_overlapping()
        if self.cur is not None:
            self._close_region(self.cur)
        t0 = time.time()
        while time.time() - t0 < wait_naming and any(
                r["named"] and not r["name"] for r in self.regions.values()
                if len(r["frames"]) >= 2):
            time.sleep(2.0)
        # 未命名兜底 (同步)
        for rid, r in self.regions.items():
            if not r["name"] and len(r["frames"]) >= 2:
                self._name_region(rid)
        self._export_web()
        sd = pathlib.Path(save_dir)
        cells = self.region_cells()
        out = {
            "regions": [{**{k: v for k, v in r.items()
                            if k not in ("kinds",)},
                         "frames": [int(x) for x in r["frames"]],
                         "cells": np.round(cells.get(r["id"],
                                           np.zeros((0, 2))), 2).tolist()}
                        for r in self.regions.values()],
            "edges": self.edges,
            "frames": {str(k): {"fid": m["fid"], "rid": m["rid"],
                                "desc": m.get("desc", "")}
                       for k, m in sorted(self.frames.items())},
            "frame_rid": {int(k): m["rid"] for k, m in self.frames.items()},
        }
        (sd / f"{self.seq}_vlm_regions.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=1, default=str))
        png = self.render_layout_png(sd)
        n = sum(1 for r in self.regions.values() if len(r["frames"]) >= 2)
        print(f"[vlm-region] 收尾: {n} 区域 / {len(self.edges)} 边 / "
              f"LLM 调用 {self.n_llm} 次 (抽稀跳过 {self.n_skip}) "
              f"-> {sd}")
        return png

    def render_layout_png(self, save_dir, scale=None):
        """楼层平面布局图: 白底 + 区域实心色 + 描边 + 中文名 + 门位。"""
        import cv2

        from mast3r_slam.mapping2d import _get_font
        rooms = self._room_structs()
        if not rooms:
            return None
        allv = np.concatenate([r["vertices"] for r in rooms])
        # 轴对齐 (走向直方图), 与视觉版平面图同系
        cams = np.array([m["xy"] for m in self.frames.values()])
        d = np.diff(cams, axis=0)
        m = np.linalg.norm(d, axis=1) > 0.05
        if m.sum() > 8:
            ang = np.degrees(np.arctan2(d[m, 1], d[m, 0])) % 180
            hist, edges_ = np.histogram(ang, bins=36, range=(0, 180))
            main = (edges_[np.argmax(hist)] + edges_[np.argmax(hist) + 1]) / 2
            th = np.radians(-main)
            R = np.array([[np.cos(th), -np.sin(th)],
                          [np.sin(th), np.cos(th)]])
            flip = -1.0 if (cams[0] @ R.T)[0] < 0 else 1.0
        else:
            R, flip = np.eye(2), 1.0

        def rot(p):
            return np.asarray(p) @ R.T * flip
        allv = rot(allv)
        x0, y0 = allv.min(0) - 1
        x1, y1 = allv.max(0) + 1
        res = 0.075
        W = int((x1 - x0) / res) + 1
        H = int((y1 - y0) / res) + 1
        lab = np.full((H, W), -1, np.int16)
        for i, r in enumerate(rooms):
            v = rot(r["vertices"])
            xi = ((v[:, 0] - x0) / res).astype(int)
            yi = ((v[:, 1] - y0) / res).astype(int)
            ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            lab[H - 1 - yi[ok], xi[ok]] = i     # y 翻转 (图像行向下)
        # 形态学补洞 (0.15 cell -> 0.075 px 有缝隙)
        for i in range(len(rooms)):
            mk = (lab == i).astype(np.uint8)
            mk = cv2.morphologyEx(mk, cv2.MORPH_CLOSE,
                                  np.ones((3, 3), np.uint8))
            lab[(mk > 0) & (lab == -1)] = i
        img = np.full((H, W, 3), 255, np.uint8)
        cols = [tuple(int(r["color"][i:i + 2], 16) for i in (1, 3, 5))
                for r in rooms]
        fill = lab >= 0
        cvals = np.array(cols, np.uint8)
        img[fill] = cvals[lab[fill]]
        edge = np.zeros((H, W), bool)
        pad = np.full((H + 2, W + 2), -2, np.int16)
        pad[1:-1, 1:-1] = lab
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            edge |= fill & (pad[1 + di:H + 1 + di, 1 + dj:W + 1 + dj] != lab)
        img[edge] = (cvals[lab[edge]] * 0.45).astype(np.uint8)
        S = 2
        img = cv2.resize(img, (W * S, H * S), interpolation=cv2.INTER_NEAREST)
        # 门位 + 标签
        from PIL import Image as PImage
        from PIL import ImageDraw
        pil = PImage.fromarray(img)
        dr = ImageDraw.Draw(pil, "RGBA")

        def to_px(pt):
            q = rot(np.asarray(pt, np.float64))
            return (float((q[0] - x0) / res * S),
                    float((H - 1 - (q[1] - y0) / res) * S))
        for e in self.edges:
            k = e["via_kf"]
            if k in self.frames:
                x, y = to_px(self.frames[k]["xy"])
                dr.ellipse([x - 5, y - 5, x + 5, y + 5],
                           fill=(30, 34, 46, 255),
                           outline=(255, 255, 255, 255), width=2)
        font = _get_font(max(13, int(H * S) // 55))
        f_sub = _get_font(max(10, int(H * S) // 80))
        placed = []

        def hit(b):
            return any(b[0] < p[2] and b[2] > p[0] and b[1] < p[3]
                       and b[3] > p[1] for p in placed)
        for i, r in enumerate(rooms):
            v = rot(r["vertices"])
            c = v.mean(0)
            k = int(np.argmin(np.linalg.norm(v - c, axis=1)))
            x, y = to_px(r["vertices"][k])
            t = r["name_zh"] or r["type_zh"]
            if font is None or not t:
                continue
            tw = dr.textlength(t, font=font)
            bx, by = x - tw / 2, y - font.size / 2
            for _ in range(8):
                if not hit((bx, by, bx + tw, by + font.size)):
                    break
                by += font.size + 4
            placed.append((bx, by, bx + tw, by + font.size))
            dr.text((bx, by), t, font=font, fill=(20, 24, 34, 255),
                    stroke_width=3, stroke_fill=(255, 255, 255, 235))
        if f_sub is not None:
            dr.text((16, H * S - f_sub.size - 12),
                    f"{self.seq} VLM 区域生长平面布局 (区域=同一空间身份的帧足迹"
                    f"; ●=区域切换点)", font=f_sub, fill=(110, 120, 140, 255))
        out = pathlib.Path(save_dir) / f"{self.seq}_vlm_layout.png"
        pil.save(out)
        print(f"[vlm-region] 平面布局图 -> {out} ({len(rooms)} 区域)")
        return out
