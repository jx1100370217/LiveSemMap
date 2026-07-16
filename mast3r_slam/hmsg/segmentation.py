"""楼层/房间分割 (照抄 fsr_vln graph.py segment_floors_manually /
segment_hmsg_room + graph_utils.distance_transform 的算法与全部超参)。

坐标系: 本实现 Z-up (insight9 VIO 世界系), 高度=pts[:,2], 地面平面=pts[:,[0,1]];
原版 Y-up 的 [:,1]/[:,[0,2]] 已等价替换, 3D 挤出的 z*=-1 + RotX90 亦等价简化为
直接 (x,y,z) 堆叠。
"""
import numpy as np


def segment_floors(points, min_points=1000):
    """楼层分割 (segment_floors_manually): 高度直方图找地面/天花板峰 -> 分层。

    points: (N,3) 世界系点 (已降采样)。返回 [(zero_level, height)] 由低到高。
    参数照抄: bin 0.01m / gaussian sigma=2 / find_peaks(distance=0.2m, height=q90)
    / 峰位 DBSCAN(eps=1m,min_samples=1) / 首尾簇 top1 中间簇 top2 /
    相邻峰距>=2.5m 时在 下一峰-0.2m 插虚拟边界 / 无峰对兜底整段单层。
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    from sklearn.cluster import DBSCAN

    z = points[:, 2]
    z0, z1 = float(z.min()), float(z.max())
    bins = max(8, int((z1 - z0) / 0.01))
    hist, edges = np.histogram(z, bins=bins)
    hist = gaussian_filter1d(hist.astype(np.float64), sigma=2)
    centers = (edges[:-1] + edges[1:]) / 2
    peaks, props = find_peaks(hist, distance=max(1, int(0.2 / 0.01)),
                              height=np.percentile(hist, 90))
    peak_z = centers[peaks]
    peak_h = props["peak_heights"]
    if len(peak_z) >= 2:
        lab = DBSCAN(eps=1.0, min_samples=1).fit(peak_z.reshape(-1, 1)).labels_
        sel = []
        order = sorted(set(lab), key=lambda c: peak_z[lab == c].min())
        for k, c in enumerate(order):
            idx = np.where(lab == c)[0]
            top = 1 if (k == 0 or k == len(order) - 1) else 2
            best = idx[np.argsort(peak_h[idx])[::-1][:top]]
            sel += [float(peak_z[b]) for b in best]
        sel = sorted(sel)
        # 相邻峰距 >= 2.5m: 插虚拟边界 (处理只有地面峰无天花板峰的高挑层)
        bounds = [sel[0]]
        for a, b in zip(sel[:-1], sel[1:]):
            if b - a >= 2.5:
                bounds.append(b - 0.2)
            bounds.append(b)
        floors = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a < 1.0:      # 过薄的峰对不成层
                continue
            zero = (a + z0) / 2 if not floors else a   # 首层下沿取 (峰+min)/2
            floors.append((zero, b - zero))
        if floors:
            zero, h = floors[-1]
            floors[-1] = (zero, z1 - zero)             # 顶层上沿取点云 max
            return floors
    # 兜底: 整段单层。单目点云地面/天花板点稀 (俯角小/弱纹理), 直方图峰常是
    # 桌面/隔断而非楼板 —— 层界取稳健分位而非 min/max (排除地下反射噪声尾)。
    return [(float(np.percentile(z, 2)),
             float(np.percentile(z, 98) - np.percentile(z, 2)))]


def _hist2d_img(pts2d, ref2d, resolution):
    """2D 点 -> 直方图图像 (行=第二轴, 列=第一轴, 与原版 histogram2d 一致);
    bin 网格以 ref2d 的范围为准 (walls 与 outside 必须同网格)。"""
    x0, y0 = ref2d[:, 0].min(), ref2d[:, 1].min()
    x1, y1 = ref2d[:, 0].max(), ref2d[:, 1].max()
    nx = int((x1 - x0) / resolution) + 2
    ny = int((y1 - y0) / resolution) + 2
    hist, _, _ = np.histogram2d(pts2d[:, 1], pts2d[:, 0], bins=(ny, nx),
                                range=((y0, y1), (x0, x1)))
    return hist


def segment_rooms(floor_points, floor_zero, floor_height, resolution=0.05,
                  debug_dir=None, wall_band=None, cam_xy=None):
    """房间分割 (segment_hmsg_room + distance_transform 全参照抄)。

    floor_points: (N,3) 楼层点云。返回 (room_2d_points 列表[(Mi,2)米制],
    room_masks 列表[(H,W) bool], meta{x0,y0,resolution}) —— 调用方再做
    3D 挤出/归属。

    单目适配 (可选, 显式偏差):
    - wall_band=(lo,hi) 绝对高度: 墙骨架切片带锚定相机高度 (高于桌面/矮隔断,
      低于天花板), 替代原版 zero±0.3 比例带 —— 单目点云地面估计不可靠,
      且桌面/工位密度高会被 0.25 阈值误判成墙;
    - cam_xy=(N,2) 相机轨迹: 轨迹膨胀带 (0.3m) 从墙骨架强制擦除
      (机器人走过=必自由, 与导航栈同一原则), 保证走廊自由区连贯,
      距离变换/Otsu 种子在薄观测区不再全灭。
    """
    xyz = floor_points
    # 切片: 墙骨架带 (wall_band 绝对高度优先, 否则原版 zero±0.3); 全高度算外边界
    if wall_band is not None:
        band = xyz[(xyz[:, 2] >= wall_band[0]) & (xyz[:, 2] < wall_band[1])]
    else:
        band = xyz[(xyz[:, 2] >= floor_zero + 0.3)
                   & (xyz[:, 2] < floor_zero + floor_height - 0.3)]
    full = xyz[xyz[:, 2] < floor_zero + floor_height - 0.2]
    return watershed_rooms(band[:, [0, 1]], full[:, [0, 1]], resolution,
                           debug_dir=debug_dir, cam_xy=cam_xy)


def watershed_rooms(pcd_2d, full_2d, resolution=0.05, debug_dir=None,
                    cam_xy=None):
    """分水岭房间分割核心 (吃 2D 点, 在线/离线共用): 墙骨架 + 外边界 +
    轨迹擦障 + 距离变换局部极大种子 + watershed。返回同 segment_rooms。"""
    import cv2

    hist = _hist2d_img(pcd_2d, pcd_2d, resolution)
    hist = cv2.normalize(hist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hist = cv2.GaussianBlur(hist, (5, 5), 1)
    _, walls = cv2.threshold(hist, 0.25 * hist.max(), 255, cv2.THRESH_BINARY)
    walls = cv2.copyMakeBorder(walls, 10, 10, 10, 10,
                               cv2.BORDER_CONSTANT, value=0)
    walls = cv2.morphologyEx(
        walls, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=1)

    hist_f = _hist2d_img(full_2d, pcd_2d, resolution)
    hist_f = cv2.normalize(hist_f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hist_f = cv2.GaussianBlur(hist_f, (21, 21), 2)
    _, outside = cv2.threshold(hist_f, 0, 255, cv2.THRESH_BINARY)
    outside = cv2.copyMakeBorder(outside, 10, 10, 10, 10,
                                 cv2.BORDER_CONSTANT, value=0)
    outside = cv2.morphologyEx(
        outside, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=3)
    cont, _ = cv2.findContours(outside, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    outside = np.zeros_like(outside)
    cv2.drawContours(outside, cont, -1, 255, -1)

    full_map = cv2.bitwise_or(walls, cv2.bitwise_not(outside))
    full_map = cv2.morphologyEx(
        full_map, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)

    # 单目适配: 相机轨迹膨胀带从障碍中强制擦除 (走过=必自由)
    if cam_xy is not None and len(cam_xy):
        x0_, y0_ = pcd_2d[:, 0].min(), pcd_2d[:, 1].min()
        traj = np.zeros(full_map.shape, np.uint8)
        pts = np.asarray(cam_xy, np.float64)
        cells = np.column_stack(((pts[:, 0] - x0_) / resolution + 10,
                                 (pts[:, 1] - y0_) / resolution + 10))
        for a, b in zip(cells[:-1], cells[1:]):
            if np.hypot(*(b - a)) < 3.0 / resolution:   # 回环跳变不连线
                cv2.line(traj, tuple(np.round(a).astype(int)),
                         tuple(np.round(b).astype(int)), 255, 1)
        r_ = max(2, int(round(0.3 / resolution)))
        traj = cv2.dilate(traj, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * r_ + 1, 2 * r_ + 1)))
        full_map[traj > 0] = 0

    # ---- distance_transform + 分水岭 ----
    # 种子生成: 原版为 距离变换->Otsu 全局阈值 (graph_utils.distance_transform)。
    # 单目适配 (显式偏差): 开放办公楼层的大开间把 Otsu 阈值抬高, 中小空间腔体
    # 种子全灭 (实测晟和/未来基金等整片区域被吞进巨房) —— 改用 Bormann ICRA16
    # 距离变换法的本意: 距离变换**局部极大**做种子 (窗 2.4m, 距障碍 >=0.45m),
    # 每个空间腔体各自出种子; 代价是长走廊分段 (语义命名可辨, 可接受)。
    from scipy.ndimage import label as nd_label
    from scipy.ndimage import maximum_filter
    bw = cv2.bitwise_not(full_map).astype(np.uint8)
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    win = max(8, int(2.4 / resolution))
    mx = (dist == maximum_filter(dist, size=win)) & (dist >= 0.45 / resolution)
    lab, n_seed = nd_label(mx)
    markers = np.zeros(dist.shape, np.int32)
    ys, xs = np.nonzero(mx)
    for y, x in zip(ys, xs):
        cv2.circle(markers, (int(x), int(y)), 3, int(lab[y, x]), -1)
    cv2.circle(markers, (3, 3), 1, n_seed + 1, -1)      # 背景 marker
    cv2.watershed(cv2.cvtColor(full_map, cv2.COLOR_GRAY2BGR), markers)
    seeds = (mx * 255).astype(np.uint8)                  # 调试图用
    contours = list(range(n_seed))                       # 房间编号 1..n_seed

    if debug_dir is not None:
        import pathlib
        d = pathlib.Path(debug_dir)
        d.mkdir(parents=True, exist_ok=True)
        for name, img in (("walls_skeleton", walls), ("outside", outside),
                          ("full_map", full_map), ("dist", dist),
                          ("seeds", seeds),
                          ("markers", (markers.astype(np.float64)
                                       / max(1, markers.max()) * 255))):
            cv2.imwrite(str(d / f"{name}.png"), np.asarray(img, np.uint8))

    # 房间格子 -> 米制 2D 点 (map_grid_to_point_cloud: (cell-10.5)*res+min)
    x0, y0 = pcd_2d[:, 0].min(), pcd_2d[:, 1].min()
    min_cells = (0.5 / resolution) ** 2      # 原版 min_area (0.5m)^2
    rooms_2d, rooms_mask = [], []
    for i in range(len(contours)):
        rows, cols = np.where(markers == i + 1)
        if len(rows) < min_cells:
            continue
        pts = np.column_stack(((cols - 10.5) * resolution + x0,
                               (rows - 10.5) * resolution + y0))
        rooms_2d.append(pts)
        m = np.zeros(markers.shape, bool)
        m[rows, cols] = True
        rooms_mask.append(m)
    meta = {"x0": float(x0), "y0": float(y0), "resolution": resolution,
            "pad": 10, "shape": [int(s) for s in markers.shape]}
    return rooms_2d, rooms_mask, meta


def compute_room_embeddings(rooms_2d, cam_positions, frame_feats, num_views=24):
    """相机->房间指派 + KMeans 代表视图 (照抄 graph_utils.compute_room_embeddings)。

    rooms_2d: 房间 2D 点集列表; cam_positions: (N,2) 相机地面坐标 (与帧对应);
    frame_feats: (N,D) 每帧全图 CLIP。返回 (room_frame_ids 列表, room_rep_feats
    列表, room_rep_ids 列表)。无帧房间强制指派最近相机帧。
    """
    from scipy.spatial import cKDTree
    from sklearn.cluster import KMeans

    trees = [cKDTree(r) for r in rooms_2d]
    assign = []
    for p in cam_positions:
        d = [t.query(p)[0] for t in trees]
        assign.append(int(np.argmin(d)))
    assign = np.asarray(assign)
    room_frames = [list(np.where(assign == i)[0]) for i in range(len(rooms_2d))]
    for i, fr in enumerate(room_frames):        # 空房间兜底: 最近相机帧
        if not fr:
            d = trees[i].query(cam_positions)[0]
            room_frames[i] = [int(np.argmin(d))]
    rep_feats, rep_ids = [], []
    for fr in room_frames:
        F = frame_feats[fr]
        k = min(num_views, len(fr))
        if len(fr) <= num_views:
            rep_feats.append(F.copy())
            rep_ids.append(list(fr))
            continue
        km = KMeans(n_clusters=k, max_iter=100, n_init=5,
                    random_state=0).fit(F)
        ids, feats = [], []
        for c in range(k):
            m = np.where(km.labels_ == c)[0]
            best = m[np.argmax(F[m] @ km.cluster_centers_[c])]
            ids.append(fr[int(best)])
            feats.append(F[int(best)])
        rep_feats.append(np.asarray(feats))
        rep_ids.append(ids)
    return room_frames, rep_feats, rep_ids
