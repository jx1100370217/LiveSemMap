"""房间级拓扑建图 (在线增量): 把逐关键帧的空间结构标注流切成"房间段",
维护 房间=节点 / 门口过渡=边 的语义拓扑图 (纯图像阶段, 不做几何锚定;
VIO 米制位置仅作合并判据与备用属性记录)。

切分: 外观驱动 (LEXI-SG 式) —— at_transition/doorway 信号做双阈值迟滞:
      累积越 trans_on 才切 (单帧误报不切), 切后武装解除, 信号跌破 trans_off
      才重新武装 (闸机阵/出入口连续区整段高信号只在进入时切一刀, 不切成碎段)。
合并: 确认再合并 —— 当前房间连续 confirm_hits 帧命中同一历史房间
      (signage 交集 + 主类别相同, 或 VIO 距离近 + 形态/类别兼容) 才合并,
      单帧误匹配不粘房; 纯 RGB 无位置时退化为仅 signage 判据。
定稿: 房间 close 时先用段内 conf 加权投票出房型, 再异步调 VLM 以代表帧+段内
      汇总做房间级仲裁 (HOV-SG 式"代表视图定房型"); VLM 失败保留投票结果。

消费顺序: 标注线程池完成顺序乱序, 而切分状态机必须按 kf 序喂 ——
tick() 只消费 kf_idx 小于 annotator 最小在途序号的标注。
"""
import base64
import json
import pathlib
import queue
import threading
from collections import Counter

import numpy as np

from mast3r_slam.semantic import SEMANTIC_CATEGORIES

# 房型词表 = POI 类别中的空间类 (doorway/junction 是过渡点, 不是房间类型)
ROOM_TYPES = tuple(k for k in SEMANTIC_CATEGORIES
                   if k not in ("doorway", "junction"))

# 平面布局图逐房间配色 (区分度优先, 深底可读; 超出循环复用)
ROOM_PALETTE = ("#4fc3f7", "#ff8a65", "#aed581", "#ba68c8", "#ffd54f", "#4db6ac",
                "#f06292", "#9575cd", "#81c784", "#ffb74d", "#64b5f6", "#e57373",
                "#dce775", "#7986cb", "#4dd0e1", "#ff8fa3", "#a1887f", "#90a4ae")


def _hex_rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))

ROOM_SCHEMA = {
    "type": "object",
    "properties": {
        "room_type": {"type": "string", "enum": list(ROOM_TYPES)},
        "name": {"type": "string", "maxLength": 24},
        "description": {"type": "string", "maxLength": 80},
    },
    "required": ["room_type", "name", "description"],
}

ROOM_PROMPT = """机器人刚穿行完室内某个连续空间段。给你该段按时间序抽取的 {n} 张前视画面, 以及该段逐帧标注的汇总:
- 逐帧类别投票: {cats}
- 常见物体: {objs}
- 读到的文字标识: {sigs}
请把这一段空间当作一个整体区域, 判定其类型、名称与描述。room_type 从以下选一:
{types}
只输出 JSON: {{"room_type": "类型", "name": "区域名称(优先用文字标识, 2-12字, 不带方位词)", "description": "一句话中文描述该区域"}}"""


def _medoid(positions):
    """离质心最近的成员位置 (必在轨迹上, 不会落进墙里)。"""
    P = np.asarray(positions, np.float64)
    return P[int(np.argmin(np.linalg.norm(P - P.mean(axis=0), axis=1)))]


class RoomTopoBuilder:
    """在线增量房间拓扑构建器: 主循环每迭代 tick() 一次, 退出时 finalize()。"""

    def __init__(self, pos_fn=None, api_url="", model="", surround_dir=None,
                 timeout=60,
                 trans_on=0.75, trans_off=0.3, decay=0.5, min_room_kfs=2,
                 merge_dist=3.0, confirm_hits=3):
        # trans_on/decay: 迟滞过渡累积 acc = acc*decay + s 的触发阈值/衰减
        #   (s=at_transition 的 conf, doorway 形态保底 0.6; 单帧 s=0.8 不够,
        #    需连续两帧强信号 ~0.8*0.5+0.8=1.2 或一强一中才越 0.75 —— 抗单帧误报)
        # trans_off: 切分后武装解除, 单帧信号 s < trans_off 才重新武装
        #   (闸机阵/出入口连续区每帧都贴着门, 无此门控会连环切出 2 帧碎房)
        # min_room_kfs: 房间段最少帧数 (防在门口连环切出碎段)
        # merge_dist/confirm_hits: 回环合并的 medoid 距离上限(米)/连续命中帧数
        self.pos_fn = pos_fn
        self.api_url = (api_url or "").rstrip("/")
        self.model = model
        self.surround_dir = pathlib.Path(surround_dir) if surround_dir else None
        self.timeout = timeout
        self.trans_on = trans_on
        self.trans_off = trans_off
        self.decay = decay
        self.min_room_kfs = min_room_kfs
        self.merge_dist = merge_dist
        self.confirm_hits = confirm_hits

        self.rooms = []            # 节点列表, 下标即 room id (合并后留 merged 墓碑)
        self.edges = []            # [{a, b, via_kf, via_frame, kind, signage}]
        self._cur = None           # 当前 open 房 id
        self._acc = 0.0            # 迟滞累积器
        self._armed = True         # 双阈值门控: 切分后解除, 信号释放才复位
        self._consumed = set()     # 已消费 kf_idx
        self._cand = (None, 0)     # 回环合并候选 (rid, 连续命中数)
        self._vlm_q = queue.Queue()   # VLM 定稿结果回填 (rid, dict)
        self._vlm_threads = []

    # ---------- 消费 ----------

    def tick(self, ann_by_kf, min_inflight):
        """主循环调用: 按 kf 序消费新完成的标注 (kf_idx < 最小在途序号才可消费)。"""
        self._apply_vlm()
        ready = sorted(k for k in ann_by_kf.keys()
                       if k not in self._consumed and k < min_inflight)
        for k in ready:
            self._step(k, ann_by_kf[k])
            self._consumed.add(k)

    def finalize(self, wait_vlm=90.0):
        """退出保存前调用: 关掉当前房间并等 VLM 定稿收尾。"""
        if self._cur is not None:
            self._close(self._cur)
            self._cur = None
        for t in self._vlm_threads:
            t.join(timeout=wait_vlm)
        self._apply_vlm()

    def reset(self):
        """重新建图: 清空全部状态 (kf_idx 从 0 复用)。"""
        self.rooms.clear()
        self.edges.clear()
        self._cur = None
        self._acc = 0.0
        self._armed = True
        self._consumed.clear()
        self._cand = (None, 0)

    # ---------- 状态机 ----------

    def _step(self, k, ann):
        fid = int(ann.get("frame_id", -1))
        pos = None
        if self.pos_fn is not None and fid >= 0:
            p = self.pos_fn(fid)
            if p is not None and np.isfinite(p).all():
                pos = np.asarray(p, np.float64)

        if self._cur is None:
            self._cur = self._new_room(k, ann, pos)
            return
        cur = self.rooms[self._cur]

        # 迟滞过渡累积 (双阈值门控)
        s = float(ann.get("transition_conf", 0.0)) if ann.get("at_transition") else 0.0
        if ann.get("space_kind") == "doorway":
            s = max(s, 0.6)
        self._acc = self._acc * self.decay + s
        if not self._armed and s < self.trans_off:
            self._armed = True     # 过渡信号已释放, 允许下一次切分

        if self._armed and self._acc >= self.trans_on \
                and len(cur["kf_indices"]) >= self.min_room_kfs:
            # 切分: 关旧房, 过渡帧归新房, 旧->新连边
            self._close(self._cur)
            prev = self._cur
            self._cur = self._new_room(k, ann, pos)
            kind = {"elevator": "elevator", "stairs": "stairs"}.get(
                ann.get("category"), "door")
            self._add_edge(prev, self._cur, k, fid, kind, ann.get("signage") or [])
            self._acc = 0.0
            self._armed = False
            self._cand = (None, 0)
            return

        # 3) 并入当前房 + 回环合并检查 (确认再合并)
        self._ingest(cur, k, ann, pos)
        hit = self._match_history(cur)
        rid, hits = self._cand
        self._cand = (hit, hits + 1 if (hit is not None and hit == rid) else
                      (1 if hit is not None else 0))
        if self._cand[0] is not None and self._cand[1] >= self.confirm_hits:
            self._merge(self._cur, self._cand[0])
            self._cand = (None, 0)

    # ---------- 房间节点 ----------

    def _new_room(self, k, ann, pos):
        rid = len(self.rooms)
        self.rooms.append({
            "id": rid, "status": "open",
            "kf_indices": [], "frame_ids": [],
            "cat_votes": Counter(), "space_votes": Counter(),
            "signage": set(), "objects": Counter(),
            "positions": [], "anns": [],
            "room_type": None, "name": "", "description": "",
            "finalized": None,   # None(open) / "vote" / "vlm"
        })
        self._ingest(self.rooms[rid], k, ann, pos)
        return rid

    def _ingest(self, room, k, ann, pos):
        room["kf_indices"].append(int(k))
        room["frame_ids"].append(int(ann.get("frame_id", -1)))
        room["cat_votes"][ann.get("category", "other")] += \
            float(ann.get("confidence", 0.0)) + 1e-3
        room["space_votes"][ann.get("space_kind", "corridor")] += 1
        room["signage"] |= {s for s in (ann.get("signage") or []) if s}
        for o in (ann.get("objects") or []):
            room["objects"][o] += 1
        if pos is not None:
            room["positions"].append(pos)
        room["anns"].append(ann)
        # 临时房型: 段内滚动投票 (open 期间导航即可用, close 时定稿覆盖)
        room["room_type"] = self._vote_type(room)

    def _vote_type(self, room):
        for c, _ in room["cat_votes"].most_common():
            if c in ROOM_TYPES:
                return c
        return "other"

    def _close(self, rid):
        room = self.rooms[rid]
        room["status"] = "closed"
        room["room_type"] = self._vote_type(room)
        if not room["name"]:
            best = max(room["anns"],
                       key=lambda a: (bool(a.get("signage")), a.get("confidence", 0)))
            room["name"] = best.get("name", "") or \
                SEMANTIC_CATEGORIES.get(room["room_type"], ("其他",))[0]
        room["finalized"] = "vote"
        print(f"[room] 房间 #{rid} 关闭: {room['name']} [{room['room_type']}] "
              f"{len(room['kf_indices'])} kf")
        if self.api_url and self.surround_dir is not None \
                and len(room["kf_indices"]) >= self.min_room_kfs:
            t = threading.Thread(target=self._vlm_finalize, args=(rid,), daemon=True)
            t.start()
            self._vlm_threads.append(t)

    # ---------- 回环合并 ----------

    def _match_history(self, cur):
        """当前帧后, 当前 open 房与哪个历史 closed 房疑似同一房间。"""
        cur_sig = cur["signage"]
        cur_type = self._vote_type(cur)
        cur_dom = cur["space_votes"].most_common(1)[0][0]
        cur_pos = cur["positions"][-1] if cur["positions"] else None
        best, best_d = None, np.inf
        for r in self.rooms:
            if r["status"] != "closed" or r["id"] == cur["id"]:
                continue
            # 强判据: 专属文字标识交集 + 主类别相同 (走廊路过门口也会读到门牌,
            # 故必须同类才认; A电梯/B电梯 等不同标识天然不交)
            if cur_sig & r["signage"] and self._vote_type(r) == cur_type:
                return r["id"]
            # 空间判据: 距历史房 medoid 近 + 形态与类别都兼容; 双方都有标识但
            # 不交 -> 是不同房间, 不并
            if cur_pos is None or not r["positions"]:
                continue
            if cur_sig and r["signage"] and not (cur_sig & r["signage"]):
                continue
            d = float(np.linalg.norm(cur_pos - _medoid(r["positions"])))
            if d < self.merge_dist and d < best_d \
                    and r["space_votes"].most_common(1)[0][0] == cur_dom \
                    and self._vote_type(r) == cur_type:
                best, best_d = r["id"], d
        return best

    def _merge(self, src, dst):
        """当前 open 房 src 并入历史房 dst, dst 重新 open 继续生长。"""
        s, d = self.rooms[src], self.rooms[dst]
        d["kf_indices"] += s["kf_indices"]
        d["frame_ids"] += s["frame_ids"]
        d["cat_votes"] += s["cat_votes"]
        d["space_votes"] += s["space_votes"]
        d["signage"] |= s["signage"]
        d["objects"] += s["objects"]
        d["positions"] += s["positions"]
        d["anns"] += s["anns"]
        d["status"] = "open"
        d["finalized"] = None
        s.update(status="merged", merged_into=dst)
        for e in self.edges:   # 改边指向, 去自环, 去重复对(汇标识)
            for key in ("a", "b"):
                if e[key] == src:
                    e[key] = dst
        self.edges = [e for e in self.edges if e["a"] != e["b"]]
        seen, dedup = {}, []
        for e in self.edges:
            pair = frozenset((e["a"], e["b"]))
            if pair in seen:
                seen[pair]["signage"] = sorted(
                    set(seen[pair]["signage"]) | set(e["signage"]))
            else:
                seen[pair] = e
                dedup.append(e)
        self.edges = dedup
        self._cur = dst
        print(f"[room] 回环合并: 房间 #{src} -> #{dst} ({d['name']}) 重新打开")

    def _add_edge(self, a, b, via_kf, via_frame, kind, signage):
        for e in self.edges:   # 同一对房间不重复建边 (重复穿同一门), 并标识
            if {e["a"], e["b"]} == {a, b}:
                e["signage"] = sorted(set(e["signage"]) | set(signage))
                return
        self.edges.append({"a": a, "b": b, "via_kf": int(via_kf),
                           "via_frame": int(via_frame), "kind": kind,
                           "signage": sorted(set(signage))})

    # ---------- VLM 房间级定稿 ----------

    def _rep_frames(self, room, n=4):
        """代表帧: 读到标识的高 conf 帧优先, 不足则沿段均匀补。"""
        idx = sorted(range(len(room["anns"])),
                     key=lambda i: (bool(room["anns"][i].get("signage")),
                                    room["anns"][i].get("confidence", 0)),
                     reverse=True)[:n]
        if len(room["anns"]) > n:
            step = len(room["anns"]) // n
            idx = sorted(set(idx) | set(range(0, len(room["anns"]), step)))[:n]
        return [room["frame_ids"][i] for i in sorted(idx)]

    def _vlm_finalize(self, rid):
        """(后台线程) 房间级 VLM 仲裁: 代表帧前视图 + 段内汇总 -> 房型/名称/描述。
        结果进 _vlm_q, 由主线程 tick() 应用; 任何失败静默保留投票结果。"""
        import io

        import requests
        from PIL import Image

        room = self.rooms[rid]
        try:
            content = []
            for fid in self._rep_frames(room):
                p = self.surround_dir / f"{int(fid):06d}_0.jpg"
                if not p.exists():
                    continue
                im = Image.open(p)
                im = im.resize((im.width // 2, im.height // 2), Image.BILINEAR)
                buf = io.BytesIO()
                im.convert("RGB").save(buf, "JPEG", quality=85)
                b64 = base64.b64encode(buf.getvalue()).decode()
                content.append({"type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            if not content:
                return
            cats = ", ".join(f"{SEMANTIC_CATEGORIES.get(c, (c,))[0]}x{n:.1f}"
                             for c, n in room["cat_votes"].most_common(3))
            objs = ", ".join(o for o, _ in room["objects"].most_common(8)) or "无"
            sigs = ", ".join(sorted(room["signage"])) or "无"
            types = ", ".join(f"{k}={SEMANTIC_CATEGORIES[k][0]}" for k in ROOM_TYPES)
            content.append({"type": "text", "text": ROOM_PROMPT.format(
                n=len(content), cats=cats, objs=objs, sigs=sigs, types=types)})
            r = requests.post(
                f"{self.api_url}/chat/completions",
                json={"model": self.model,
                      "messages": [{"role": "user", "content": content}],
                      "max_tokens": 200, "temperature": 0.0,
                      "chat_template_kwargs": {"enable_thinking": False},
                      "response_format": {"type": "json_schema",
                                          "json_schema": {"name": "room",
                                                          "schema": ROOM_SCHEMA}}},
                timeout=self.timeout, proxies={"http": None, "https": None})
            r.raise_for_status()
            res = json.loads(r.json()["choices"][0]["message"]["content"])
            if res.get("room_type") in ROOM_TYPES:
                self._vlm_q.put((rid, res))
        except Exception as e:
            print(f"[room] 房间 #{rid} VLM 定稿失败(保留投票结果): {e}")

    def _apply_vlm(self):
        try:
            while True:
                rid, res = self._vlm_q.get_nowait()
                room = self.rooms[rid]
                if room["status"] == "closed":   # 已被合并/重开则丢弃旧定稿
                    room.update(room_type=res["room_type"], name=res["name"],
                                description=res["description"], finalized="vlm")
                    print(f"[room] 房间 #{rid} VLM 定稿: {res['name']} "
                          f"[{res['room_type']}]")
        except queue.Empty:
            pass

    # ---------- 落盘 ----------

    def snapshot(self):
        """可序列化拓扑图 (rooms/edges; merged 墓碑保留 id 但只存指向)。"""
        rooms = []
        for r in self.rooms:
            if r["status"] == "merged":
                rooms.append({"id": r["id"], "status": "merged",
                              "merged_into": r["merged_into"]})
                continue
            rooms.append({
                "id": r["id"], "status": r["status"],
                "room_type": r["room_type"], "name": r["name"],
                "description": r["description"], "finalized": r["finalized"],
                "kf_indices": r["kf_indices"], "frame_ids": r["frame_ids"],
                "signage": sorted(r["signage"]),
                "objects": [o for o, _ in r["objects"].most_common(10)],
                "space_kind": (r["space_votes"].most_common(1)[0][0]
                               if r["space_votes"] else None),
                "position": ([float(x) for x in _medoid(r["positions"])]
                             if r["positions"] else None),
            })
        return {"rooms": rooms, "edges": self.edges, "current": self._cur}

    def save(self, path):
        path = pathlib.Path(path)
        tmp = path.with_name("tmp_" + path.name)
        with open(tmp, "w") as f:
            json.dump(self.snapshot(), f, ensure_ascii=False, indent=1)
        tmp.replace(path)
        n_rooms = sum(1 for r in self.rooms if r["status"] != "merged")
        print(f"[room] 拓扑图已保存: {n_rooms} 房间 / {len(self.edges)} 边 -> {path}")


# ---------- 平面布局图渲染 (落在几何建图 BEV 上) ----------

def _traj_cells(kf_px, G, jump_px):
    """关键帧轨迹线格 (相邻 kf 连线; 间距超 jump_px 视为回环跳变不连)。"""
    cells = np.zeros((G, G), bool)
    prev = None
    for x, y in np.asarray(kf_px, np.float64):
        if prev is not None:
            d = max(abs(x - prev[0]), abs(y - prev[1]))
            if d <= jump_px:
                n = int(d) + 1
                xs = np.clip(np.round(np.linspace(prev[0], x, n)).astype(int), 0, G - 1)
                ys = np.clip(np.round(np.linspace(prev[1], y, n)).astype(int), 0, G - 1)
                cells[ys, xs] = True
        i, j = int(round(y)), int(round(x))
        if 0 <= i < G and 0 <= j < G:
            cells[i, j] = True
        prev = (x, y)
    return cells


def assign_room_regions(grid, kf_px, rooms, corridor_px=3, jump_px=15):
    """把 BEV 可行走区按房间划分 (测地最近种子, 不穿墙/不越未观测区)。

    grid: (G,G) uint8 0未知/1可通行/2障碍; kf_px: (N,2) 关键帧像素 [x,y];
    rooms: 房间列表(需含 kf_indices), 其顺序即返回的标签值;
    corridor_px: 轨迹走廊带半径(格) —— 机器人走过=事实可走, 与 grid==1 一并
    作为可分配区 (占据栅格的 free 常稀疏, 只用它会把区域涂成碎屑)。
    返回 (G,G) int16: -1=未分配, 否则 rooms 列表下标。
    实现: 各房间成员 kf 格为种子, 在可分配掩码上逐圈 4 邻域标签传播
    (等价多源 BFS, 波前同速 -> 每格归测地最近种子的房间)。
    """
    from scipy import ndimage as ndi

    G = grid.shape[0]
    lab = np.zeros((G, G), np.int16)          # 0=未分配, 否则 下标+1
    r_ = int(corridor_px)
    yy, xx = np.ogrid[-r_:r_ + 1, -r_:r_ + 1]
    band = ndi.binary_dilation(_traj_cells(kf_px, G, jump_px),
                               (xx * xx + yy * yy) <= r_ * r_)
    mask = (grid == 1) | band
    for li, r in enumerate(rooms):
        for k in r["kf_indices"]:
            if 0 <= k < len(kf_px):
                j, i = int(round(kf_px[k][0])), int(round(kf_px[k][1]))
                if 0 <= i < G and 0 <= j < G:
                    lab[i, j] = li + 1
                    mask[i, j] = True
    for _ in range(2 * G):                    # 上限远大于最长测地距离, 必收敛
        grew = False
        empty = (lab == 0) & mask
        if not empty.any():
            break
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nb = np.zeros_like(lab)
            if di == -1:
                nb[1:, :] = lab[:-1, :]
            elif di == 1:
                nb[:-1, :] = lab[1:, :]
            elif dj == -1:
                nb[:, 1:] = lab[:, :-1]
            else:
                nb[:, :-1] = lab[:, 1:]
            take = empty & (lab == 0) & (nb > 0)
            if take.any():
                lab[take] = nb[take]
                grew = True
        if not grew:
            break
    return lab - 1


def render_rooms_png(run_dir, seq, scale=3, fill_alpha=110):
    """渲染"落在几何建图上的平面布局图": 每个房间一种颜色的半透明区域 + 同色
    描边 + 房心圆点/中文名标签(贪心避让), 门口边画白色细线。

    输入 logs 产物 {seq}_occupancy.npz + {seq}_rooms.json, 底图由占据栅格重建
    (干净版, 不带 POI 圆点), 输出 {seq}_rooms.png (G*scale 分辨率)。
    返回输出路径 (无房间返回 None)。
    """
    from PIL import Image, ImageDraw

    from mast3r_slam.mapping2d import _get_font, occupancy_vis, world_to_px

    run = pathlib.Path(run_dir)
    z = np.load(run / f"{seq}_occupancy.npz")
    grid, kf_px = z["grid"], z["kf_px"]
    meta = json.loads(str(z["meta"]))
    rj = json.loads((run / f"{seq}_rooms.json").read_text())
    live = [r for r in rj["rooms"] if r["status"] != "merged"]
    if not live:
        return None
    G = grid.shape[0]
    px_per_m = G / (2 * meta["half"])        # VIO 系为真实米
    traj = _traj_cells(kf_px, G, jump_px=max(10, int(3.0 * px_per_m)))
    region = assign_room_regions(grid, kf_px, live,
                                 corridor_px=max(2, int(round(0.5 * px_per_m))),
                                 jump_px=max(10, int(3.0 * px_per_m)))

    # 干净底图: 与导航 Web 同款视觉层次 (未知暗/观测微亮/走廊带亮/墙白)
    base = occupancy_vis(grid, traj).astype(np.float32) * 255.0

    # 区域半透明填色 + 房间间/外边界描实色 (格级, NEAREST 放大后即"框出来"的观感)
    cols = np.array([_hex_rgb(ROOM_PALETTE[i % len(ROOM_PALETTE)])
                     for i in range(len(live))], np.float32)
    fill = region >= 0
    a = fill_alpha / 255.0
    base[fill] = base[fill] * (1 - a) + cols[region[fill]] * a
    edge = np.zeros((G, G), bool)
    pad = np.full((G + 2, G + 2), -2, np.int16)
    pad[1:-1, 1:-1] = region
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nb = pad[1 + di:G + 1 + di, 1 + dj:G + 1 + dj]
        edge |= fill & (nb != region)
    base[edge] = cols[region[edge]]
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)) \
        .resize((G * scale, G * scale), Image.NEAREST)

    # 房心 (medoid 世界坐标投影; 无位置的房间取中段成员 kf 像素)
    def _rpx(r):
        if r.get("position"):
            px, py = world_to_px(np.asarray(r["position"])[None], meta)
            return float(px[0]) * scale, float(py[0]) * scale
        ks = [k for k in r["kf_indices"] if 0 <= k < len(kf_px)]
        x, y = kf_px[ks[len(ks) // 2]]
        return float(x) * scale, float(y) * scale

    d = ImageDraw.Draw(img, "RGBA")
    ctr = {r["id"]: _rpx(r) for r in live}
    for e in rj["edges"]:                    # 门口边: 折经真实过门点(via_kf 处),
        if e["a"] not in ctr or e["b"] not in ctr:   # 房心直连会斜穿无关空间
            continue
        pts = [ctr[e["a"]], ctr[e["b"]]]
        v = e.get("via_kf", -1)
        if 0 <= v < len(kf_px):
            vx, vy = float(kf_px[v][0]) * scale, float(kf_px[v][1]) * scale
            pts.insert(1, (vx, vy))
            d.ellipse([vx - 3, vy - 3, vx + 3, vy + 3],
                      fill=(255, 255, 255, 220))   # 门位标记
        d.line(pts, fill=(255, 255, 255, 120), width=2)
    font = _get_font(max(12, G * scale // 60))
    placed = []                              # 标签贪心避让: 相交则下移重试
    for i, r in enumerate(live):
        x, y = ctr[r["id"]]
        col = tuple(int(c) for c in cols[i])
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=col + (255,),
                  outline=(255, 255, 255, 230), width=2)
        text = r["name"] or (r["room_type"] or "?")
        if font is None:
            continue
        tw = d.textlength(text, font=font)
        W = G * scale
        bx = float(np.clip(x + 8, 2, W - tw - 4))
        by = y - font.size - 2
        for _ in range(8):                   # 逐次下移到不与已放标签相交
            box = (bx, by, bx + tw + 4, by + font.size + 4)
            if not any(box[0] < p[2] and box[2] > p[0]
                       and box[1] < p[3] and box[3] > p[1] for p in placed):
                break
            by += font.size + 6
        placed.append((bx, by, bx + tw + 4, by + font.size + 4))
        d.text((bx, by), text, font=font, fill=(255, 255, 255, 255),
               stroke_width=2, stroke_fill=(10, 12, 20, 220))
    out = run / f"{seq}_rooms.png"
    img.save(out)
    print(f"[room] 平面布局图 -> {out} ({len(live)} 房间)")
    return out
