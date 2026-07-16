"""跨帧 3D 实例融合 (照抄 fsr_vln graph_utils 的 merge_3d_masks /
hierarchical_merge / seq_merge / find_overlapping_ratio_faiss / feats_denoise)。

判据: AABB 3D-IoU > iou_thresh(0.05) 才算 faiss 双向最近邻点重叠率
(radius=1.5*voxel), 重叠率 > overlap_threshold 连边 -> 连通分量合并
(点云相加 + DBSCAN(eps=0.1,min=10) 去噪 + voxel 降采样)。
调度默认 hierarchical (相邻帧两两归并, 每轮阈值递减, 末轮 0.75) ——
sequential 为原版发布配置但复杂度 O(全局masks^2 x 帧数), 559 关键帧不可行。
"""
import numpy as np


def _aabb_iou(amin, amax, bmin, bmax):
    omin, omax = np.maximum(amin, bmin), np.minimum(amax, bmax)
    inter = float(np.prod(np.maximum(omax - omin, 0.0)))
    va, vb = float(np.prod(amax - amin)), float(np.prod(bmax - bmin))
    return inter / max(va + vb - inter, 1e-9)


def overlap_ratio_faiss(p1, p2, radius):
    """双向最近邻重叠率, 取 max (find_overlapping_ratio_faiss 原样)。"""
    import faiss
    if len(p1) == 0 or len(p2) == 0:
        return 0.0
    i1, i2 = faiss.IndexFlatL2(3), faiss.IndexFlatL2(3)
    a, b = np.ascontiguousarray(p1, np.float32), np.ascontiguousarray(p2, np.float32)
    i1.add(a)
    i2.add(b)
    d1, _ = i2.search(a, 1)
    d2, _ = i1.search(b, 1)
    return max(float((d1 < radius ** 2).sum()) / len(a),
               float((d2 < radius ** 2).sum()) / len(b))


class Inst:
    """一个 3D mask/实例: 点 + 特征累积 + 观测来源 (kf, 平均深度)。"""

    __slots__ = ("pts", "feat_sum", "n_obs", "views")

    def __init__(self, pts, feat, kf_idx, mean_depth):
        self.pts = pts                       # (M,3) float32 世界系
        self.feat_sum = feat.astype(np.float64).copy()
        self.n_obs = 1
        self.views = {int(kf_idx): float(mean_depth)}   # kf -> 平均观测深度

    def merged_with(self, others, voxel):
        pts = np.concatenate([self.pts] + [o.pts for o in others], 0)
        pts = _voxel_down(pts, voxel)
        pts = _dbscan_largest(pts, eps=0.1, min_points=10)
        m = Inst.__new__(Inst)
        m.pts = pts
        m.feat_sum = self.feat_sum + sum(o.feat_sum for o in others)
        m.n_obs = self.n_obs + sum(o.n_obs for o in others)
        m.views = dict(self.views)
        for o in others:
            for k, d in o.views.items():
                m.views[k] = min(d, m.views.get(k, np.inf))
        return m


def _voxel_down(pts, voxel):
    if len(pts) == 0:
        return pts
    key = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return pts[np.sort(idx)]


def _dbscan_largest(pts, eps, min_points):
    """DBSCAN 去噪保留最大簇 (pcd_denoise_dbscan 等价, sklearn 实现)。"""
    from sklearn.cluster import DBSCAN
    if len(pts) < min_points:
        return pts
    lab = DBSCAN(eps=eps, min_samples=min_points).fit(
        pts.astype(np.float32)).labels_
    if (lab >= 0).any():
        vals, cnts = np.unique(lab[lab >= 0], return_counts=True)
        return pts[lab == vals[np.argmax(cnts)]]
    return pts


def merge_3d_masks(insts, overlap_threshold, radius, iou_thresh=0.05):
    """一轮融合 (merge_3d_masks 原样语义): AABB IoU 门 + 重叠率图 + 连通分量。
    IoU 门控向量化 (分块广播), 语义与逐对循环一致 —— 559 关键帧末轮 ~万级
    masks 时 O(n^2) Python 循环不可行。"""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(insts)
    if n <= 1:
        return list(insts)
    mins = np.stack([i.pts.min(0) for i in insts])
    maxs = np.stack([i.pts.max(0) for i in insts])
    vols = np.prod(maxs - mins, axis=1)
    rows, cols = [], []
    CH = 512
    for a in range(0, n, CH):
        b = min(a + CH, n)
        omin = np.maximum(mins[a:b, None], mins[None])     # (c,n,3)
        omax = np.minimum(maxs[a:b, None], maxs[None])
        inter = np.prod(np.maximum(omax - omin, 0.0), axis=2)
        iou = inter / np.maximum(vols[a:b, None] + vols[None] - inter, 1e-9)
        ii, jj = np.nonzero(iou > iou_thresh)
        for i_, j_ in zip(ii + a, jj):
            if i_ >= j_:                                   # 只取上三角
                continue
            if overlap_ratio_faiss(insts[i_].pts, insts[j_].pts,
                                   1.5 * radius) > overlap_threshold:
                rows.append(i_)
                cols.append(j_)
    g = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    n_comp, lab = connected_components(g, directed=False)
    out = []
    for c in range(n_comp):
        idx = np.where(lab == c)[0]
        if len(idx) == 1:
            out.append(insts[idx[0]])
        else:
            out.append(insts[idx[0]].merged_with([insts[k] for k in idx[1:]],
                                                 voxel=0.5 * radius))
    return out


def hierarchical_merge(frames, init_th=0.75, th_factor=0.025, radius=0.05,
                       iou_thresh=0.05, verbose=True):
    """层级归并 (hierarchical_merge 原样): 相邻帧两两归并至只剩一组, 末轮 0.75。"""
    th = init_th
    rounds = 0
    while len(frames) > 1:
        nxt = []
        for i in range(0, len(frames), 2):
            if i == len(frames) - 1:
                nxt.append(frames[i])
                break
            nxt.append(merge_3d_masks(frames[i] + frames[i + 1], th, radius,
                                      iou_thresh))
        frames = nxt
        if len(frames) > 1:
            th -= th_factor * (len(frames) - 2) / max(1, len(frames) - 1)
        rounds += 1
        if verbose:
            n = sum(len(f) for f in frames)
            print(f"[hmsg] 实例归并 第{rounds}轮: {len(frames)} 组 / {n} masks, "
                  f"th={th:.3f}", flush=True)
    return merge_3d_masks(frames[0], 0.75, radius, iou_thresh)


def instance_feature(inst, global_pts_tree, full_feats, max_dist=0.8):
    """实例级特征 (graph.py 451-488 原样): 实例点取 <=0.8m 内最近全局点的逐点
    特征, cosine DBSCAN(eps=0.01,min=100) 最大簇均值; 无有效簇退化为全体均值。"""
    from sklearn.cluster import DBSCAN
    d, idx = global_pts_tree.query(inst.pts, k=1, workers=-1)
    keep = d <= max_dist
    if not keep.any():
        f = inst.feat_sum / max(inst.n_obs, 1)
        return (f / max(np.linalg.norm(f), 1e-9)).astype(np.float32)
    F = full_feats[idx[keep]].astype(np.float32)
    F = F / np.clip(np.linalg.norm(F, axis=1, keepdims=True), 1e-9, None)
    if len(F) >= 100:
        lab = DBSCAN(eps=0.01, min_samples=100, metric="cosine").fit(F).labels_
        if (lab >= 0).any():
            vals, cnts = np.unique(lab[lab >= 0], return_counts=True)
            F = F[lab == vals[np.argmax(cnts)]]
    f = F.mean(0)
    return (f / max(np.linalg.norm(f), 1e-9)).astype(np.float32)
