"""HMSG 构建主管线 (照抄 fsr_vln Graph.create_feature_map +
build_hier_multimodal_scene_graph 的阶段与超参; 输入改为 MASt3R-SLAM 产物)。

输入 (logs/<run>/):
  {seq}_kf_pointmaps.npz   逐关键帧米制世界系点图 (X/conf/frame_ids)
  {seq}_semantic.json      (可选) 在线 Qwen 逐帧标注 -> view.vlm_description
  datasets/<seq>/{fid:06d}.png   关键帧 RGB (与点图逐像素对齐)
  datasets/<seq>/vio.txt + timestamps.txt   c2w 位姿

阶段: A 全局点云 -> B 逐帧 SAM+CLIP+mask升3D -> C 层级实例融合 ->
D 楼层 -> E 房间+View -> F 物体(归属/词表标签/可见性) -> G 房间命名 -> 序列化。

与原版的显式偏差 (其余算法/超参照抄):
- mask 2D->3D 直接布尔索引帧点图 (同源同坐标, 免反投影+吸附);
- 全局点云去噪参数按 MASt3R 密度重调 (原版 eps=0.01/radius outlier 1000pts
  为超稠密 RGB-D 所调, 在 0.05 体素点云上退化);
- merge_type 默认 hierarchical (sequential 为原版发布值, 但 O(N^2 x 帧) 在
  559 关键帧不可行);
- 阶段 H Voronoi 导航图跳过 (LiveSemMap 已有 A*/LoTIS 导航栈)。
"""
import json
import pathlib

import numpy as np

from .features import CLIP_DIM, SamClipExtractor, load_vocab
from .graph import Floor, HMSGGraph, Object, Room, View, _pcd
from .instances import (Inst, _dbscan_largest, _voxel_down, hierarchical_merge,
                        instance_feature)
from .segmentation import (compute_room_embeddings, segment_floors,
                           segment_rooms)

VOXEL = 0.05            # voxel_size (config 默认)
DEPTH_CUT = 6.0         # 真机 depth_cut (米)
MIN_MASK_PTS = 25       # mask 升 3D 最少有效点
MIN_INST_PTS = 10       # 实例最少点数 (graph.py:447)
ROOM_TYPES = ("office", "meeting room", "hallway", "pantry", "restroom",
              "lobby", "elevator hall", "print room", "lounge", "storage room")


def _load_poses(dataset_dir, frame_ids):
    """帧号 -> c2w(4,4) (vio.txt TUM + timestamps 就近配对; 光学系 z前/x右/y下)。"""
    from scipy.spatial.transform import Rotation
    ds = pathlib.Path(dataset_dir)
    ts = np.loadtxt(ds / "timestamps.txt")
    vio = np.loadtxt(ds / "vio.txt")
    poses = []
    for fid in frame_ids:
        t = ts[min(int(fid), len(ts) - 1), 1]
        j = int(np.clip(np.searchsorted(vio[:, 0], t), 1, len(vio) - 1))
        if abs(vio[j - 1, 0] - t) < abs(vio[j, 0] - t):
            j -= 1
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(vio[j, 4:8]).as_matrix()
        T[:3, 3] = vio[j, 1:4]
        poses.append(T)
    return np.asarray(poses)


def _fit_pinhole(Xw, conf, c2w, conf_th):
    """从世界系点图拟合等效针孔 K (无标定模式 X_canon 为 ray-based, 拟合仅供
    可见性重投影用)。u = fx*(x/z)+cx 最小二乘。"""
    h, w = Xw.shape[:2]
    R, t = c2w[:3, :3], c2w[:3, 3]
    Xc = (Xw.reshape(-1, 3).astype(np.float64) - t) @ R
    v, u = np.mgrid[0:h, 0:w]
    m = (conf.reshape(-1) > conf_th) & (Xc[:, 2] > 0.2) & (Xc[:, 2] < DEPTH_CUT)
    x_z, y_z = Xc[m, 0] / Xc[m, 2], Xc[m, 1] / Xc[m, 2]
    A = np.stack([x_z, np.ones_like(x_z)], 1)
    fx, cx = np.linalg.lstsq(A, u.reshape(-1)[m], rcond=None)[0]
    A = np.stack([y_z, np.ones_like(y_z)], 1)
    fy, cy = np.linalg.lstsq(A, v.reshape(-1)[m], rcond=None)[0]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    return K


def check_object_in_view(obj_pts, c2w, K, hw, min_visible_ratio=0.5,
                         max_depth=10.0):
    """物体点云重投影可见性 (graph_utils.check_object_in_view 原样):
    可见点比例 >= 0.5 且平均深度 <= 10m。返回 (可见?, 平均深度)。"""
    R, t = c2w[:3, :3], c2w[:3, 3]
    Xc = (obj_pts.astype(np.float64) - t) @ R
    z = Xc[:, 2]
    front = z > 0.05
    if not front.any():
        return False, np.inf
    u = K[0, 0] * Xc[front, 0] / z[front] + K[0, 2]
    v = K[1, 1] * Xc[front, 1] / z[front] + K[1, 2]
    h, w = hw
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    ratio = inside.sum() / len(obj_pts)
    if not inside.any():
        return False, np.inf
    mean_d = float(z[front][inside].mean())
    return bool(ratio >= min_visible_ratio and mean_d <= max_depth), mean_d


def find_intersection_share(room_vertices, obj_2d, radius=0.2):
    """物体 2D 点落入房间 2D 点集 radius 邻域的占比 (graph_utils 原样)。"""
    from scipy.spatial import cKDTree
    d, _ = cKDTree(room_vertices).query(obj_2d, k=1, workers=-1)
    return float((d < radius).sum()) / max(len(obj_2d), 1)


def build_hmsg(run_dir, seq, dataset_dir, device="cuda:0", stride=1,
               max_frames=0, out_dir=None, zh_api="", zh_model=""):
    import cv2
    from scipy.spatial import cKDTree

    run = pathlib.Path(run_dir)
    ds = pathlib.Path(dataset_dir)
    out = pathlib.Path(out_dir) if out_dir else run / "hmsg"
    out.mkdir(parents=True, exist_ok=True)

    z = np.load(run / f"{seq}_kf_pointmaps.npz")
    X_all, conf_all = z["X"], z["conf"]
    frame_ids = z["frame_ids"].tolist()
    conf_th = float(z["conf_threshold"])
    sel = list(range(0, len(frame_ids), stride))
    if max_frames:
        sel = sel[:max_frames]
    poses = _load_poses(ds, [frame_ids[i] for i in sel])
    print(f"[hmsg] {len(sel)}/{len(frame_ids)} 关键帧, conf阈值 {conf_th}")

    # 逐帧 Qwen 标注 (view.vlm_description)
    ann_by_kf = {}
    semp = run / f"{seq}_semantic.json"
    if semp.exists():
        sem = json.loads(semp.read_text())
        ann_by_kf = {int(k): v for k, v in sem.get("annotations", {}).items()}

    K = _fit_pinhole(X_all[sel[0]].astype(np.float32), conf_all[sel[0]],
                     poses[0], conf_th)
    hw = X_all.shape[1:3]
    print(f"[hmsg] 拟合针孔 K: fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
          f"cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

    # ---------- 阶段 A: 全局点云 ----------
    pts_acc, col_acc = [], []
    for si, i in enumerate(sel):
        Xw = X_all[i].reshape(-1, 3).astype(np.float32)
        cf = conf_all[i].reshape(-1).astype(np.float32)
        cam = poses[si][:3, 3].astype(np.float32)
        dist = np.linalg.norm(Xw - cam, axis=1)
        m = (cf > conf_th) & (dist < DEPTH_CUT) & np.isfinite(Xw).all(1)
        img = cv2.imread(str(ds / f"{frame_ids[i]:06d}.png"))
        col = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1, 3)
        pts_acc.append(Xw[m][::4])          # 全局点云 1/4 抽样 (体素前减负)
        col_acc.append(col[m][::4])
    P = np.concatenate(pts_acc, 0)
    C = np.concatenate(col_acc, 0)
    key = np.floor(P / VOXEL).astype(np.int64)
    _, uidx = np.unique(key, axis=0, return_index=True)
    P, C = P[uidx], C[uidx]
    # 去噪 (参数按 MASt3R 密度重调, 见模块头注释)
    keep = _dbscan_largest(P, eps=3 * VOXEL, min_points=10)
    tree_tmp = cKDTree(keep)
    m = tree_tmp.query(P, k=1, workers=-1)[0] < 1e-6
    P, C = P[m], C[m]
    print(f"[hmsg] 阶段A 全局点云: {len(P)} 点 (voxel {VOXEL})")
    global_tree = cKDTree(P)
    full_feats = np.zeros((len(P), CLIP_DIM), np.float32)
    full_cnt = np.zeros(len(P), np.int32)

    # ---------- 阶段 B: SAM + CLIP + mask 升 3D ----------
    ext = SamClipExtractor(device=device)
    frames_insts, frame_gfeats = [], []
    for si, i in enumerate(sel):
        fid = frame_ids[i]
        rgb = cv2.cvtColor(cv2.imread(str(ds / f"{fid:06d}.png")),
                           cv2.COLOR_BGR2RGB)
        masks, mfeats, gfeat = ext.extract(rgb)
        frame_gfeats.append(gfeat)
        Xw = X_all[i].astype(np.float32)
        cf = conf_all[i].astype(np.float32)
        cam = poses[si][:3, 3].astype(np.float32)
        valid = (cf > conf_th) & np.isfinite(Xw).all(-1) \
            & (np.linalg.norm(Xw - cam, axis=-1) < DEPTH_CUT)
        insts = []
        for mk, mf in zip(masks, mfeats):
            seg = mk["segmentation"] & valid
            if seg.sum() < MIN_MASK_PTS:
                continue
            pts = _voxel_down(Xw[seg].reshape(-1, 3), VOXEL)
            if len(pts) < MIN_MASK_PTS:
                continue
            d, gi = global_tree.query(pts, k=1, workers=-1)
            hit = gi[d < 2 * VOXEL]
            full_feats[hit] += mf          # 逐点特征累积 (mask 特征铺到成员点)
            full_cnt[hit] += 1
            depth = float(np.linalg.norm(pts - cam, axis=1).mean())
            insts.append(Inst(pts, mf, i, depth))
        frames_insts.append(insts)
        if si % 50 == 0:
            print(f"[hmsg] 阶段B {si}/{len(sel)} 帧, 本帧 {len(insts)} masks",
                  flush=True)
    nz = full_cnt > 0
    full_feats[nz] /= full_cnt[nz, None]
    norm = np.linalg.norm(full_feats, axis=1, keepdims=True)
    full_feats = full_feats / np.clip(norm, 1e-9, None)

    # ---------- 阶段 C: 层级实例融合 ----------
    insts = hierarchical_merge([f for f in frames_insts if f],
                               init_th=0.75, th_factor=0.025, radius=VOXEL)
    insts = [t for t in insts if len(t.pts) >= MIN_INST_PTS]
    inst_feats = np.stack([instance_feature(t, global_tree, full_feats)
                           for t in insts])
    print(f"[hmsg] 阶段C 实例: {len(insts)} 个")

    # ---------- 阶段 D: 楼层 ----------
    g = HMSGGraph()
    g.full_pcd = _pcd(P, C / 255.0)
    g.full_feats = full_feats.astype(np.float16)
    g.mask_feats = list(inst_feats)
    floors_zh = segment_floors(P)
    print(f"[hmsg] 阶段D 楼层: {len(floors_zh)} 层 {floors_zh}")

    vocab = load_vocab("scannet200")
    text_feats = ext.encode_text(vocab)
    room_type_feats = ext.encode_text(list(ROOM_TYPES))
    frame_gfeats = np.stack(frame_gfeats)
    cam_xy = poses[:, :2, 3]

    view_of_kf = {}                     # 选中帧下标 si -> view_id
    for fi, (zero, height) in enumerate(floors_zh):
        f = Floor(fi)
        f.floor_zero_level, f.floor_height = zero, height
        fm = (P[:, 2] >= zero) & (P[:, 2] < zero + height)
        f.pcd = _pcd(P[fm], C[fm] / 255.0)
        bb = f.pcd.get_axis_aligned_bounding_box()
        f.vertices = np.asarray(bb.get_box_points())
        g.floors.append(f)

        # ---------- 阶段 E: 房间 + View ----------
        cam_z = float(np.median(poses[:, 2, 3]))
        rooms_2d, rooms_mask, meta = segment_rooms(
            P[fm], zero, height, debug_dir=out / f"tmp_floor{fi}",
            wall_band=(cam_z + 0.5, cam_z + 1.3),   # 单目适配: 相机锚定墙带
            cam_xy=cam_xy)
        # 伪房间合并 (单目适配): 相机轨迹未自然落入的分水岭区域 —— 玻璃反射
        # 把点打到墙外形成的"墙外伪房", 或无观测碎片 —— 并入最近真实房间,
        # 否则它们以 1-view 强制指派 + 噪声命名散布, 观感上"区域位置乱"。
        from scipy.spatial import cKDTree as _KD
        natural = [bool((_KD(r2d).query(cam_xy, k=1, workers=-1)[0] < 0.08).any())
                   for r2d in rooms_2d]
        if not all(natural) and any(natural):
            real = [i for i, n in enumerate(natural) if n]
            centers = {i: rooms_2d[i].mean(0) for i in range(len(rooms_2d))}
            merged_2d = {i: [rooms_2d[i]] for i in real}
            merged_mask = {i: rooms_mask[i].copy() for i in real}
            for i, n in enumerate(natural):
                if n:
                    continue
                j = min(real, key=lambda r: np.linalg.norm(centers[r] - centers[i]))
                merged_2d[j].append(rooms_2d[i])
                merged_mask[j] |= rooms_mask[i]
            rooms_2d = [np.concatenate(merged_2d[i]) for i in real]
            rooms_mask = [merged_mask[i] for i in real]
            print(f"[hmsg] 伪房间合并: {len(natural)} -> {len(rooms_2d)} "
                  f"(并入 {len(natural) - len(rooms_2d)} 个无相机经过的区域)")
        room_frames, rep_feats, rep_ids = compute_room_embeddings(
            rooms_2d, cam_xy, frame_gfeats, num_views=24)
        print(f"[hmsg] 阶段E 楼层{fi}: {len(rooms_2d)} 房间")
        Pf = P[fm]
        res, pad = meta["resolution"], meta["pad"]
        rows = ((Pf[:, 1] - meta["y0"]) / res + pad + 0.5).astype(int)
        cols = ((Pf[:, 0] - meta["x0"]) / res + pad + 0.5).astype(int)
        H_, W_ = meta["shape"]
        inb = (rows >= 0) & (rows < H_) & (cols >= 0) & (cols < W_)
        for ri, r2d in enumerate(rooms_2d):
            r = Room(f"{fi}_{ri}", fi)
            r.vertices = r2d
            r.room_zero_level, r.room_height = zero, height
            rm = np.zeros(len(Pf), bool)
            sub = rooms_mask[ri][rows[inb], cols[inb]]
            rm[np.where(inb)[0][sub]] = True
            r.pcd = _pcd(Pf[rm], C[fm][rm] / 255.0)
            r.embeddings = list(rep_feats[ri])
            r.represent_images = [int(frame_ids[sel[j]]) for j in rep_ids[ri]]
            r.sample_images = [int(frame_ids[sel[j]]) for j in room_frames[ri]]
            r.clip_embeddings = [frame_gfeats[j] for j in room_frames[ri]]
            # 房间命名 (view_embedding 法: 代表视图逐个 argmax 投票)
            if len(r.embeddings):
                votes = np.argmax(np.stack(r.embeddings) @ room_type_feats.T,
                                  axis=1)
                r.name = ROOM_TYPES[int(np.bincount(votes).argmax())]
            g.rooms.append(r)
            f.rooms.append(r.room_id)
            for vi, j in enumerate(room_frames[ri]):
                kf_i = sel[j]
                fid = frame_ids[kf_i]
                v = View(f"{fi}_{ri}_{vi}", r.room_id, fid,
                         str((ds / f"{fid:06d}.png").resolve()))
                v.pose = poses[j]
                ann = ann_by_kf.get(kf_i)
                if ann:
                    v.vlm_description = ann.get("description", "")
                g.views.append(v)
                r.views.append(v.view_id)
                view_of_kf[j] = v.view_id
    view_by_id = {v.view_id: v for v in g.views}

    # ---------- 阶段 F: 物体 ----------
    obj_cnt = {}
    for t, feat in zip(insts, inst_feats):
        zmin, zmax = float(t.pts[:, 2].min()), float(t.pts[:, 2].max())
        floor = next((f for f in g.floors
                      if zmin >= f.floor_zero_level - 0.2
                      and zmax <= f.floor_zero_level + f.floor_height + 0.2),
                     g.floors[0])
        cand = [r for r in g.rooms if r.floor_id == floor.floor_id]
        if not cand:
            continue
        obj2d = t.pts[:, :2]
        shares = [find_intersection_share(r.vertices, obj2d, 0.2)
                  for r in cand]
        if max(shares) > 0:
            room = cand[int(np.argmax(shares))]
        else:                            # 兜底: 房心-物心距离
            oc = obj2d.mean(0)
            room = cand[int(np.argmin(
                [np.linalg.norm(r.vertices.mean(0) - oc) for r in cand]))]
        oi = obj_cnt.get(room.room_id, 0)
        obj_cnt[room.room_id] = oi + 1
        o = Object(f"{room.room_id}_{oi}", room.room_id)
        o.pcd = _pcd(t.pts)
        o.vertices = obj2d
        o.embedding = feat
        o.name = vocab[int(np.argmax(feat @ text_feats.T))]
        # view 拓扑: 房间内每个 view 重投影判可见, best = 平均深度最小
        best_d = np.inf
        for vid in room.views:
            v = view_by_id[vid]
            si = next((j for j, vv in view_of_kf.items() if vv == vid), None)
            if si is None:
                continue
            vis, md = check_object_in_view(t.pts, poses[si], K, hw)
            if vis:
                o.view_ids.append(vid)
                v.object_ids.append(o.object_id)
                v.text_discription.append(o.name)
                if md < best_d:
                    best_d, o.best_view_id = md, vid
        room.objects.append(o.object_id)
        g.objects.append(o)
    print(f"[hmsg] 阶段F 物体: {len(g.objects)} 个已归房")

    # ---------- 阶段 G+: Qwen 中文化 (物体标签翻译 + 区域命名) ----------
    if zh_api:
        from .qwen_zh import localize_graph
        localize_graph(g, ds, run, seq, zh_api, zh_model)

    g.assemble()
    g.save(out)
    return g, out
