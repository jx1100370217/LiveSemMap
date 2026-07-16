"""OnlineHMSG — HMSG 在线增量引擎 (建图过程中实时生长的层级多模态场景图)。

与离线 build_hmsg 的关系: 同一套算法族, 增量化调度 ——
- 逐关键帧 (GPU 工作线程, 与 SLAM 分时): SAM2 自动掩码 + ConceptFusion CLIP
  特征 + mask 升 3D (点图在线取自 SharedKeyframes: X_canon x VIO尺度 + VIO位姿);
- 在线实例关联 (论文 Fig5 的增量形态): 新帧 masks 与现有 3D 实例做
  AABB IoU 门 + 双向重叠率 (阈值 0.75 照抄 sequential), 匹配融合/不匹配新建;
- 周期房间重算 (每 room_period 个关键帧): watershed_rooms 秒级全量重算,
  房间 id 按区域重叠继承 (稳定不闪变);
- 房间稳定 (连续两轮边界基本不变) 即触发 Qwen 中文命名 (异步);
- 周期写 web 快照 (hmsg.js), 前端热更新可见地图生长;
- finalize(): 排空队列 -> 末轮房间/命名 -> 构造 HMSGGraph 全量序列化 + 查询包。

与离线版的显式差异 (在线性所需):
- 实例特征 = 成员 mask 特征累计均值 (离线为全局逐点特征 + cosine DBSCAN 精炼);
- 物体-视图拓扑 = 观测来源 (哪些帧的 mask 贡献了该实例) 而非重投影可见性;
- 房间代表特征 = 房内视图均匀抽样 (finalize 时改用 KMeans24 照抄离线)。
"""
import json
import pathlib
import queue
import threading
import time

import numpy as np

from .features import CLIP_DIM, load_vocab
from .instances import Inst, _voxel_down, overlap_ratio_faiss
from .segmentation import watershed_rooms

VOXEL = 0.05
DEPTH_CUT = 6.0
MIN_MASK_PTS = 25
OVERLAP_TH = 0.75          # 在线关联阈值 (照抄 sequential init_overlap_thresh)
ROOM_TYPES = ("office", "meeting room", "hallway", "pantry", "restroom",
              "lobby", "elevator hall", "print room", "lounge", "storage room")


class OnlineHMSG:
    def __init__(self, keyframes, vio_prior, dataset_dir, web_dir, seq,
                 semantic_ann=None, zh_api="", zh_model="", device="cuda:0",
                 room_period=40, snapshot_sec=30.0, live=None):
        # live: mp.Manager().dict() — 房间区域轻量快照跨进程共享给 viewer
        #       (viewer 的占据图面板换成实时区域生长图)
        self.live = live
        self.keyframes = keyframes
        self.vio = vio_prior
        self.ds = pathlib.Path(dataset_dir)
        self.web_dir = pathlib.Path(web_dir)
        self.seq = seq
        self.sem_ann = semantic_ann if semantic_ann is not None else {}
        self.zh_api, self.zh_model = zh_api, zh_model
        self.device = device
        self.room_period = room_period
        self.snapshot_sec = snapshot_sec

        self.insts = []            # 融合后实例 (Inst)
        self.frames = {}           # kf_idx -> {fid, pose(4,4), gfeat}
        self.rooms = []            # [{rid, vertices, n_prev, stable, named,
                                   #   name_zh, type_zh, summary_zh, views[kf]}]
        self._rid_next = 0
        self._wall_keys = set()    # 墙带 2D 点 voxel 去重
        self._wall_pts = []
        self._full_keys = set()
        self._full_pts = []
        self._obj_room = {}        # inst 下标 -> rid
        self._view_room = {}       # kf_idx -> rid
        self._since_room = 0
        self._last_snap = 0.0
        self._ext = None           # SAM2+CLIP 惰性加载
        self._vocab = None
        self._text_feats = None
        self._rtype_feats = None
        self._zh_map = {}
        self.q = queue.Queue()
        self.submitted = set()
        self.disabled = False
        self._lock = threading.Lock()      # rooms/insts 快照一致性
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # ---------- 提交 ----------
    def catch_up(self):
        n = len(self.keyframes)
        for i in range(n):
            if i in self.submitted:
                continue
            self.submitted.add(i)
            with self.keyframes.lock:
                fid = int(self.keyframes.dataset_idx[i])
            self.q.put((i, fid))

    def reset(self):
        try:
            while True:
                self.q.get_nowait()
                self.q.task_done()
        except queue.Empty:
            pass
        with self._lock:
            self.submitted.clear()
            self.insts.clear()
            self.frames.clear()
            self.rooms.clear()
            self._rid_next = 0
            self._wall_keys.clear()
            self._wall_pts.clear()
            self._full_keys.clear()
            self._full_pts.clear()
            self._obj_room.clear()
            self._view_room.clear()

    def pending(self):
        return self.q.qsize()

    # ---------- 工作线程 ----------
    def _load_models(self):
        from .features import SamClipExtractor
        print("[hmsg-live] 加载 SAM2 + CLIP ...", flush=True)
        self._ext = SamClipExtractor(device=self.device)
        self._vocab = load_vocab("scannet200")
        self._text_feats = self._ext.encode_text(self._vocab)
        self._rtype_feats = self._ext.encode_text(list(ROOM_TYPES))
        zp = pathlib.Path(__file__).parent / "vocab" / "scannet200_zh.json"
        if zp.exists():
            self._zh_map = json.loads(zp.read_text())
        print("[hmsg-live] 模型就绪", flush=True)

    def _run(self):
        import cv2
        while True:
            kf_idx, fid = self.q.get()
            try:
                if self.disabled:
                    continue
                s = self.vio.metric_scale() if self.vio is not None else 1.0
                if self.vio is not None and s is None:
                    self.q.put((kf_idx, fid))    # 尺度未标定, 稍后再处理
                    time.sleep(1.5)
                    continue
                if self._ext is None:
                    self._load_models()
                self._process(kf_idx, fid, float(s or 1.0), cv2)
                self._since_room += 1
                if self._since_room >= self.room_period:
                    self._room_pass()
                    self._since_room = 0
                if time.time() - self._last_snap > self.snapshot_sec:
                    self._snapshot()
                    self._last_snap = time.time()
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[hmsg-live] kf{kf_idx} 处理失败: {e}", flush=True)
            finally:
                self.q.task_done()

    def _process(self, kf_idx, fid, s, cv2):
        with self.keyframes.lock:
            kf = self.keyframes[kf_idx]
            h, w = [int(x) for x in kf.img_shape.flatten()[:2]]
            X = kf.X_canon.reshape(h, w, 3).cpu().numpy().astype(np.float64)
            conf = kf.get_average_conf().reshape(h, w).cpu().numpy()
        p, Rm = self.vio._pose_at(fid)
        pose = np.eye(4)
        pose[:3, :3] = Rm.as_matrix()
        pose[:3, 3] = p
        Xw = (X * s) @ pose[:3, :3].T + pose[:3, 3]
        rgb = cv2.cvtColor(cv2.imread(str(self.ds / f"{fid:06d}.png")),
                           cv2.COLOR_BGR2RGB)
        masks, mfeats, gfeat = self._ext.extract(rgb)
        cam = pose[:3, 3]
        valid = (conf > 1.5) & np.isfinite(Xw).all(-1) \
            & (np.linalg.norm(Xw - cam, axis=-1) < DEPTH_CUT)

        # 2D 缓存 (房间重算素材): 墙带 + 全高度, voxel 去重
        cz = cam[2]
        Xf = Xw[valid]
        band = Xf[(Xf[:, 2] >= cz + 0.5) & (Xf[:, 2] < cz + 1.3)][:, :2]
        fullb = Xf[Xf[:, 2] < cz + 1.6][:, :2]
        for pts, keys, store in ((band, self._wall_keys, self._wall_pts),
                                 (fullb, self._full_keys, self._full_pts)):
            kq = np.floor(pts / VOXEL).astype(np.int64)
            for k_, pt in zip(map(tuple, kq[::2]), pts[::2]):
                if k_ not in keys:
                    keys.add(k_)
                    store.append(pt)

        # masks -> 3D -> 在线实例关联
        new_insts = []
        for mk, mf in zip(masks, mfeats):
            seg = mk["segmentation"] & valid
            if seg.sum() < MIN_MASK_PTS:
                continue
            pts = _voxel_down(Xw[seg].reshape(-1, 3).astype(np.float32), VOXEL)
            if len(pts) < MIN_MASK_PTS:
                continue
            depth = float(np.linalg.norm(pts - cam, axis=1).mean())
            new_insts.append(Inst(pts, mf, kf_idx, depth))
        with self._lock:
            mins = [t.pts.min(0) for t in self.insts]
            maxs = [t.pts.max(0) for t in self.insts]
            for t in new_insts:
                nmin, nmax = t.pts.min(0), t.pts.max(0)
                best, best_r = -1, 0.0
                for j in range(len(self.insts)):
                    omin = np.maximum(mins[j], nmin)
                    omax = np.minimum(maxs[j], nmax)
                    inter = float(np.prod(np.maximum(omax - omin, 0)))
                    if inter <= 0:
                        continue
                    va = float(np.prod(maxs[j] - mins[j]))
                    vb = float(np.prod(nmax - nmin))
                    if inter / max(va + vb - inter, 1e-9) <= 0.05:
                        continue
                    r = overlap_ratio_faiss(self.insts[j].pts, t.pts,
                                            1.5 * VOXEL)
                    if r > best_r:
                        best_r, best = r, j
                if best >= 0 and best_r > OVERLAP_TH:
                    m = self.insts[best].merged_with([t], voxel=0.5 * VOXEL)
                    self.insts[best] = m
                    mins[best], maxs[best] = m.pts.min(0), m.pts.max(0)
                else:
                    self.insts.append(t)
                    mins.append(nmin)
                    maxs.append(nmax)
            self.frames[kf_idx] = {"fid": fid, "pose": pose, "gfeat": gfeat}
        if kf_idx % 25 == 0:
            print(f"[hmsg-live] kf{kf_idx}: 实例 {len(self.insts)}, "
                  f"房间 {len(self.rooms)}", flush=True)

    # ---------- 周期房间重算 ----------
    def _room_pass(self):
        if len(self._wall_pts) < 500 or len(self.frames) < 8:
            return
        cam_xy = np.array([f["pose"][:2, 3] for f in self.frames.values()])
        rooms_2d, rooms_mask, meta = watershed_rooms(
            np.asarray(self._wall_pts), np.asarray(self._full_pts),
            cam_xy=cam_xy)
        # 伪房过滤 (无相机经过) + id 继承 (与旧房间 2D 重叠 argmax)
        from scipy.spatial import cKDTree
        keep = []
        for r2d in rooms_2d:
            d = cKDTree(r2d).query(cam_xy, k=1, workers=-1)[0]
            if (d < 0.08).any():
                keep.append(r2d)
        if not keep:                 # 早期墙点稀疏, 分水岭无有效区域
            return
        old = [(r, cKDTree(r["vertices"])) for r in self.rooms]
        new_rooms, used = [], set()
        for r2d in keep:
            sub = r2d[::4]
            best, best_s = None, 0.3
            for r, tree in old:
                if r["rid"] in used:
                    continue
                share = float((tree.query(sub, k=1, workers=-1)[0] < 0.1).mean())
                if share > best_s:
                    best_s, best = share, r
            if best is not None:
                used.add(best["rid"])
                stable = abs(len(r2d) - best["n_prev"]) < 0.05 * best["n_prev"]
                nr = dict(best)
                nr.update(vertices=r2d, n_prev=len(r2d),
                          stable=best["stable"] + 1 if stable else 0)
            else:
                nr = {"rid": f"0_{self._rid_next}", "vertices": r2d,
                      "n_prev": len(r2d), "stable": 0, "named": False,
                      "name_zh": "", "type_zh": "", "summary_zh": ""}
                self._rid_next += 1
            new_rooms.append(nr)
        # 归属重指派
        view_room, obj_room = {}, {}
        trees = [cKDTree(r["vertices"]) for r in new_rooms]
        for k, f in self.frames.items():
            d = [t.query(f["pose"][:2, 3])[0] for t in trees]
            view_room[k] = new_rooms[int(np.argmin(d))]["rid"]
        with self._lock:
            insts = list(self.insts)
        for i, t in enumerate(insts):
            c2 = t.pts[:, :2].mean(0)
            shares = [float((tr.query(t.pts[::5, :2], k=1, workers=-1)[0]
                             < 0.2).mean()) for tr in trees]
            j = int(np.argmax(shares)) if max(shares) > 0 else \
                int(np.argmin([np.linalg.norm(r["vertices"].mean(0) - c2)
                               for r in new_rooms]))
            obj_room[i] = new_rooms[j]["rid"]
        with self._lock:
            self.rooms = new_rooms
            self._view_room = view_room
            self._obj_room = obj_room
        print(f"[hmsg-live] 房间重算: {len(new_rooms)} 房间", flush=True)
        self._push_live()
        # 稳定房间 -> Qwen 命名 (异步)
        if self.zh_api:
            for r in new_rooms:
                if not r["named"] and r["stable"] >= 2:
                    r["named"] = True     # 提交即置位, 防重复
                    threading.Thread(target=self._name_room, args=(r,),
                                     daemon=True).start()

    def _name_room(self, room):
        try:
            from .qwen_zh import name_room
            rid = room["rid"]
            vks = [k for k, v in self._view_room.items() if v == rid]
            if not vks:
                return
            gf = np.stack([self.frames[k]["gfeat"] for k in vks])
            rtype = ROOM_TYPES[int(np.argmax(gf.mean(0) @ self._rtype_feats.T))]
            objs = {}
            for i, r_ in self._obj_room.items():
                if r_ != rid or i >= len(self.insts):
                    continue
                en = self._vocab[int(np.argmax(
                    (self.insts[i].feat_sum / max(self.insts[i].n_obs, 1))
                    @ self._text_feats.T))]
                zh = self._zh_map.get(en, en)
                objs[zh] = objs.get(zh, 0) + 1
            objects_zh = ", ".join(f"{k}x{v}" for k, v in
                                   sorted(objs.items(), key=lambda x: -x[1])[:12])
            sigs, descs = set(), []
            for k in vks:
                a = self.sem_ann.get(k)
                if not a:
                    continue
                sigs |= {s for s in (a.get("signage") or []) if s}
                if a.get("landmark") and a.get("name"):
                    sigs.add(a["name"])
                if a.get("description"):
                    descs.append(a["description"])
            descs = descs[:: max(1, len(descs) // 6)]
            fids = sorted(self.frames[k]["fid"] for k in vks)
            imgs = [self.ds / f"{fids[len(fids)//3]:06d}.png",
                    self.ds / f"{fids[2*len(fids)//3]:06d}.png"]
            out = name_room(self.zh_api, self.zh_model, rtype, objects_zh,
                            sorted(sigs)[:8], descs, imgs)
            room["name_zh"] = out["name"]
            room["type_zh"] = out["room_type"]
            room["summary_zh"] = out["summary"]
            print(f"[hmsg-live] 命名 {rid}: {out['name']} [{out['room_type']}]",
                  flush=True)
        except Exception as e:
            room["named"] = False        # 失败允许下轮重试
            print(f"[hmsg-live] 房间命名失败 {room['rid']}: {e}", flush=True)

    # ---------- 快照 / 收尾 ----------
    def _export(self, quiet=True, query_pack_dir=None, kmeans_rep=False):
        from .webexport import export_web_data
        with self._lock:
            rooms = [dict(r) for r in self.rooms]
            insts = list(self.insts)
            frames = dict(self.frames)
            view_room = dict(self._view_room)
            obj_room = dict(self._obj_room)
        if not rooms or not frames:
            return
        # 视图/物体聚合
        views, view_id_of = [], {}
        per_room_cnt = {}
        for k in sorted(frames):
            rid = view_room.get(k)
            if rid is None:
                continue
            n = per_room_cnt.get(rid, 0)
            per_room_cnt[rid] = n + 1
            vid = f"{rid}_{n}"
            view_id_of[k] = vid
            a = self.sem_ann.get(k) or {}
            views.append({"id": vid, "room": rid, "img_id": frames[k]["fid"],
                          "pose": frames[k]["pose"],
                          "desc": a.get("description", ""), "objects": []})
        objects = []
        obj_cnt = {}
        for i, t in enumerate(insts):
            rid = obj_room.get(i)
            if rid is None:
                continue
            n = obj_cnt.get(rid, 0)
            obj_cnt[rid] = n + 1
            f = t.feat_sum / max(t.n_obs, 1)
            f = f / max(np.linalg.norm(f), 1e-9)
            en = self._vocab[int(np.argmax(f @ self._text_feats.T))] \
                if self._vocab else "object"
            vks = sorted(t.views, key=lambda k_: t.views[k_])
            objects.append({"id": f"{rid}_{n}", "room": rid, "name": en,
                            "name_zh": self._zh_map.get(en, en),
                            "pts": t.pts, "embedding": f.astype(np.float32),
                            "best_view": view_id_of.get(vks[0]) if vks else None,
                            "views": [view_id_of[k_] for k_ in vks
                                      if k_ in view_id_of]})
        vmap = {v["id"]: v for v in views}
        for o in objects:
            for vid in o["views"]:
                vmap[vid]["objects"].append(o["id"])
        # 房间导出结构 (+代表特征)
        out_rooms = []
        for r in rooms:
            vks = [k for k, v in view_room.items() if v == r["rid"]]
            gfs = [frames[k]["gfeat"] for k in vks if k in frames]
            if kmeans_rep and len(gfs) > 24:
                from sklearn.cluster import KMeans
                F = np.stack(gfs)
                km = KMeans(n_clusters=24, max_iter=100, n_init=5,
                            random_state=0).fit(F)
                reps = [F[np.argmax(F @ c)] for c in km.cluster_centers_]
            else:
                reps = gfs[:24]
            out_rooms.append({"id": r["rid"], "name": r.get("type_zh", ""),
                              "name_zh": r["name_zh"], "type_zh": r["type_zh"],
                              "summary_zh": r["summary_zh"],
                              "vertices": r["vertices"],
                              "n_views": len(vks),
                              "n_objects": sum(1 for x in obj_room.values()
                                               if x == r["rid"]),
                              "rep_feats": reps})
        export_web_data(self.seq, self.web_dir, out_rooms, objects, views,
                        hmsg_dir=query_pack_dir, quiet=quiet)
        return out_rooms, objects, views

    def _push_live(self):
        """房间区域轻量快照 -> 共享 dict (viewer 区域生长图)。"""
        if self.live is None:
            return
        try:
            from .graph import ROOM_PALETTE
            with self._lock:
                rooms = [dict(r) for r in self.rooms]
            out = []
            for i, r in enumerate(rooms):
                v = np.asarray(r["vertices"])
                step = max(1, len(v) // 600)
                out.append({"name": r.get("name_zh", ""),
                            "color": ROOM_PALETTE[i % len(ROOM_PALETTE)],
                            "pts": np.round(v[::step], 2).tolist()})
            self.live["rooms"] = out
        except Exception as e:
            print(f"[hmsg-live] live 推送失败: {e}", flush=True)

    def _snapshot(self):
        try:
            self._export(quiet=True)
        except Exception as e:
            print(f"[hmsg-live] 快照失败: {e}", flush=True)

    def drain(self):
        while self.q.unfinished_tasks > 0 and not self.disabled:
            print(f"[hmsg-live] 等待剩余 ~{self.q.qsize()} 关键帧处理...",
                  flush=True)
            time.sleep(3.0)

    def finalize(self, save_dir):
        """退出保存: 排空 -> 末轮房间 -> 等命名 -> 全量导出 + 序列化。"""
        self.drain()
        self._room_pass()
        if self.zh_api:      # 未命名房间最后一次机会 (同步等待)
            for r in self.rooms:
                if not r["named"]:
                    r["named"] = True
                    self._name_room(r)
        out = self._export(quiet=False,
                           query_pack_dir=pathlib.Path(save_dir) / "hmsg",
                           kmeans_rep=True)
        if out is None:
            return
        rooms, objects, views = out
        # HMSG 文件夹协议序列化 (ply+json)
        from .graph import Floor, HMSGGraph, Object, Room, View, _pcd
        g = HMSGGraph()
        f = Floor(0)
        zs = np.concatenate([o["pts"][:, 2] for o in objects]) \
            if objects else np.array([0.0])
        f.floor_zero_level = float(np.percentile(zs, 2))
        f.floor_height = float(np.percentile(zs, 98) - f.floor_zero_level)
        g.floors.append(f)
        for r in rooms:
            ro = Room(r["id"], 0)
            ro.name = r["type_zh"] or "room"
            ro.name_zh, ro.type_zh = r["name_zh"], r["type_zh"]
            ro.summary_zh = r["summary_zh"]
            ro.vertices = np.asarray(r["vertices"])
            ro.embeddings = [np.asarray(e) for e in r["rep_feats"]]
            g.rooms.append(ro)
            f.rooms.append(ro.room_id)
        for o in objects:
            ob = Object(o["id"], o["room"])
            ob.name, ob.name_zh = o["name"], o["name_zh"]
            ob.embedding = o["embedding"]
            ob.pcd = _pcd(o["pts"])
            ob.vertices = o["pts"][:, :2]
            ob.view_ids, ob.best_view_id = o["views"], o["best_view"]
            g.objects.append(ob)
        for v in views:
            vo = View(v["id"], v["room"], v["img_id"],
                      str((self.ds / f"{v['img_id']:06d}.png").resolve()))
            vo.pose = v["pose"]
            vo.vlm_description = v["desc"]
            vo.object_ids = v["objects"]
            g.views.append(vo)
        g.assemble()
        g.save(pathlib.Path(save_dir) / "hmsg")
