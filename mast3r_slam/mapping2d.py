"""2D 地图栅格化公共模块: BEV 彩色俯视 + 占据栅格 + 语义节点叠加。

从 visualization.py 的 _rasterize_map 抽出, 供两处共用:
- viewer 后台线程实时刷新 Map View 面板;
- 退出/中断时保存最终 2D 地图产物 (bev.png / occupancy.npz)。

坐标约定: 自动检测竖直轴(3 轴 extent 最小者), 其余两轴为地面 (a, b)。
返回的 meta 记录该映射, world_to_px() 可把世界系点投到输出图像素 (已含 flipud)。
"""
import numpy as np

_CJK_FONT = None


def _get_font(px):
    """加载 CJK 字体(缓存字体文件路径, 按字号实例化)。找不到返回 None(退化为无标签)。"""
    global _CJK_FONT
    from PIL import ImageFont

    if _CJK_FONT is None:
        import glob
        cands = sorted(glob.glob("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")) or \
            sorted(glob.glob("/usr/share/fonts/**/*CJK*", recursive=True)) or \
            sorted(glob.glob("/usr/share/fonts/**/wqy*", recursive=True))
        _CJK_FONT = cands[0] if cands else ""
    if not _CJK_FONT:
        return None
    try:
        return ImageFont.truetype(_CJK_FONT, px)
    except Exception:
        return None


def rasterize_map(P, C, centers, G=340):
    """点云 -> (bev, occ_vis, grid, meta)。

    P: (N,3) 世界系点; C: (N,3) RGB 0-1; centers: (M,3) 相机轨迹(含当前帧在末尾)。
    bev/occ_vis: (G,G,3) float 0-1, 已 flipud(y 向上为图像上方);
    grid: (G,G) uint8 原始占据栅格(0=未知 1=可通行 2=障碍), 与 bev 同像素系;
    meta: 世界<->像素映射参数 {v,a,b,ca,cb,half,G}。
    """
    lo, hi = np.percentile(P, 2, 0), np.percentile(P, 98, 0)
    ext = hi - lo
    v = int(np.argmin(ext))
    a, b = [k for k in range(3) if k != v]
    ca, cb = (lo[a] + hi[a]) / 2, (lo[b] + hi[b]) / 2
    half = max(ext[a], ext[b]) * 0.55 + 1e-6

    def to_grid(xa, xb):
        ga = np.clip((xa - (ca - half)) / (2 * half) * G, -1, G).astype(np.int32)
        gb = np.clip((xb - (cb - half)) / (2 * half) * G, -1, G).astype(np.int32)
        return ga, gb

    ga, gb = to_grid(P[:, a], P[:, b])
    inb = (ga >= 0) & (ga < G) & (gb >= 0) & (gb < G)
    ga, gb, yv, Ci = ga[inb], gb[inb], P[inb, v], C[inb]

    # bincount 累加 (比 np.add.at 快 ~10x)
    flat = gb.astype(np.int64) * G + ga.astype(np.int64)
    cnt = np.bincount(flat, minlength=G * G).astype(np.float32).reshape(G, G)
    acc = np.stack(
        [np.bincount(flat, weights=Ci[:, c], minlength=G * G) for c in range(3)], -1
    ).astype(np.float32).reshape(G, G, 3)
    nz = cnt > 0

    def _dilate(x, k=2):    # 最大值膨胀, 填补稀疏点云的散点空洞
        o = x.copy()
        for dy in range(-k, k + 1):
            for dx in range(-k, k + 1):
                o = np.maximum(o, np.roll(np.roll(x, dy, 0), dx, 1))
        return o

    # BEV 彩色俯瞰: 每格均值真彩 + 两轮 8 邻域填洞去散点
    bev = np.zeros((G, G, 3), np.float32)
    bev[nz] = acc[nz] / cnt[nz][:, None]
    filled = nz.astype(np.float32)
    for _ in range(2):
        sc = np.zeros_like(bev)
        sn = np.zeros((G, G), np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                sc += np.roll(np.roll(bev * filled[..., None], dy, 0), dx, 1)
                sn += np.roll(np.roll(filled, dy, 0), dx, 1)
        empty = (filled == 0) & (sn > 0)
        bev[empty] = sc[empty] / sn[empty][:, None]
        filled[empty] = 1.0
    bev[filled == 0] = 0.05     # 未观测=暗底

    # 占据栅格: 中段高度带有点=障碍, 观测到=空闲, 无观测=未知; 膨胀去空洞
    floor, ceil = np.percentile(yv, 8), np.percentile(yv, 92)
    rv = max(ceil - floor, 1e-3)
    obst = (yv > floor + 0.20 * rv) & (yv < floor + 0.65 * rv)
    ocnt = np.bincount(flat[obst], minlength=G * G).astype(np.float32).reshape(G, G)
    cnt_d, ocnt_d = _dilate(cnt), _dilate(ocnt)

    free = cnt_d > 0
    blocked = free & (ocnt_d >= 2)
    grid = np.zeros((G, G), np.uint8)
    grid[free] = 1
    grid[blocked] = 2

    occ = np.full((G, G, 3), 0.20, np.float32)              # 未知=灰
    occ[free] = np.array([0.82, 0.82, 0.80], np.float32)    # 空闲=亮
    occ[blocked] = np.array([0.10, 0.11, 0.14], np.float32)  # 障碍=暗

    # 叠相机轨迹(青) + 当前相机(品红)
    if len(centers) > 1:
        cga, cgb = to_grid(centers[:, a], centers[:, b])
        cin = (cga >= 0) & (cga < G) & (cgb >= 0) & (cgb < G)
        traj = np.array([0.18, 0.89, 0.90], np.float32)
        for im in (bev, occ):
            im[cgb[cin], cga[cin]] = traj
        if cin[-1]:
            yy, xx = int(cgb[-1]), int(cga[-1])
            mag = np.array([1.0, 0.16, 0.42], np.float32)
            for im in (bev, occ):
                im[max(0, yy - 2):yy + 3, max(0, xx - 2):xx + 3] = mag

    meta = {"v": v, "a": a, "b": b, "ca": float(ca), "cb": float(cb),
            "half": float(half), "G": G}
    return np.flipud(bev).copy(), np.flipud(occ).copy(), np.flipud(grid).copy(), meta


def world_to_px(pts, meta):
    """世界系点 (N,3) -> 输出图(已 flipud)像素坐标 (x, y) float 数组。"""
    pts = np.atleast_2d(np.asarray(pts, np.float64))
    G, half = meta["G"], meta["half"]
    xa = pts[:, meta["a"]]
    xb = pts[:, meta["b"]]
    px = (xa - (meta["ca"] - half)) / (2 * half) * G
    py = (xb - (meta["cb"] - half)) / (2 * half) * G
    return px, (G - 1) - py     # flipud: 行翻转


def draw_semantic_nodes(img, nodes, meta, label=True, dot_r=None):
    """在 rasterize_map 输出图上叠加语义节点: 类别色圆点(白描边) + 中文名标签。
    img: (G,G,3) float 0-1, 原地修改并返回。nodes: aggregate_nodes() 的输出。"""
    if not nodes:
        return img
    import cv2
    from mast3r_slam.semantic import SEMANTIC_CATEGORIES

    G = meta["G"]
    if dot_r is None:
        dot_r = max(3, G // 85)
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    px, py = world_to_px(np.array([n["position"] for n in nodes]), meta)
    for i, n in enumerate(nodes):
        x, y = int(round(px[i])), int(round(py[i]))
        if not (0 <= x < G and 0 <= y < G):
            continue
        color = SEMANTIC_CATEGORIES.get(n["category"], ("", (0.8, 0.8, 0.8), False))[1]
        rgb = tuple(int(c * 255) for c in color)  # u8 数组本身是 RGB 通道序
        cv2.circle(u8, (x, y), dot_r + 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(u8, (x, y), dot_r, rgb, -1, cv2.LINE_AA)

    if label:
        u8 = _draw_labels(u8, nodes, px, py, G)
    img[:] = u8.astype(np.float32) / 255.0
    return img


def _draw_labels(u8, nodes, px, py, G):
    """PIL 画中文标签(cv2.putText 不支持中文)。带黑色描边, 避让图外。"""
    from PIL import Image, ImageDraw

    font = _get_font(max(10, G // 28))
    if font is None:
        return u8
    pil = Image.fromarray(u8)
    d = ImageDraw.Draw(pil)
    for i, n in enumerate(nodes):
        x, y = int(round(px[i])), int(round(py[i]))
        if not (0 <= x < G and 0 <= y < G):
            continue
        text = n.get("name") or n["category"]
        tw = d.textlength(text, font=font)
        tx = int(np.clip(x - tw / 2, 1, G - tw - 1))
        ty = y - G // 24 - font.size
        if ty < 1:
            ty = y + G // 40
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            d.text((tx + dx, ty + dy), text, font=font, fill=(0, 0, 0))
        d.text((tx, ty), text, font=font, fill=(255, 255, 255))
    return np.asarray(pil)
