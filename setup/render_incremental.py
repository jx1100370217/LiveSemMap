"""把 MASt3R-SLAM 在 Mapping_C8 上的增量快照渲染成增量建图过程视频/GIF。
每帧: snap_%04d.ply (点云) + centers_%04d.npy (相机中心, 同基准)。
因 MASt3R-SLAM 在 Sim(3) 下优化, 快照间全局尺度/基准会漂移, 故用【每帧自适应视野】
(从该帧点云+相机中心的稳健分位算), 保证点云与轨迹始终同框可见。无头 (Agg)。
用法: python render_incremental.py <snap_dir> <out_prefix>
"""
import sys, glob, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plyfile import PlyData
import imageio.v2 as imageio

snap_dir = sys.argv[1] if len(sys.argv) > 1 else "logs/c8_incremental/snapshots"
out_prefix = sys.argv[2] if len(sys.argv) > 2 else "logs/c8_incremental/incremental_mapping"

ply_files = sorted(glob.glob(os.path.join(snap_dir, "snap_*.ply")),
                   key=lambda p: int(re.search(r"snap_(\d+)", p).group(1)))
assert ply_files, f"无快照: {snap_dir}/snap_*.ply"
print(f"{len(ply_files)} 个快照")


def load_ply(p):
    d = PlyData.read(p)["vertex"]
    pts = np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float32)
    col = np.stack([d["red"], d["green"], d["blue"]], 1).astype(np.float32) / 255.0
    m = np.isfinite(pts).all(1) & (np.abs(pts) < 1e4).all(1)  # 剔 NaN/Inf 与爆炸外点
    return pts[m], col[m]


def load_centers(pf):
    c = pf.replace("snap_", "centers_").replace(".ply", ".npy")
    if os.path.exists(c):
        a = np.load(c)
        return a[np.isfinite(a).all(1) & (np.abs(a) < 1e4).all(1)]
    return None


frames = []
for k, pf in enumerate(ply_files):
    n_kf = int(re.search(r"snap_(\d+)", pf).group(1))
    pts, col = load_ply(pf)
    ctrs = load_centers(pf)
    if len(pts) == 0:
        continue
    if len(pts) > 130000:
        idx = np.random.default_rng(0).choice(len(pts), 130000, replace=False)
        pts, col = pts[idx], col[idx]

    # 每帧自适应视野: 用点云(2/98分位)+相机中心的并集, 稳健且必同框
    ref = pts if ctrs is None or len(ctrs) == 0 else np.concatenate([pts, ctrs], 0)
    lo, hi = np.percentile(ref, 2, 0), np.percentile(ref, 98, 0)
    c0 = (lo + hi) / 2.0
    rng = float(np.clip((hi - lo).max() * 0.62, 0.5, 500.0))

    fig = plt.figure(figsize=(12, 6.2), facecolor="#0b1020")
    for sp, (i, j, ttl) in enumerate([(0, 2, "Top view (X-Z, bird's-eye)"), (0, 1, "Side view (X-Y)")]):
        ax = fig.add_subplot(1, 2, sp + 1, facecolor="#0b1020")
        ax.scatter(pts[:, i], pts[:, j], c=col, s=0.5, marker=".", linewidths=0)
        if ctrs is not None and len(ctrs) > 1:
            ax.plot(ctrs[:, i], ctrs[:, j], "-", color="#2de2e6", lw=1.6, alpha=0.95)
            ax.plot(ctrs[-1, i], ctrs[-1, j], "o", color="#ff2a6d", ms=7)  # 当前相机
        ax.set_xlim(c0[i] - rng, c0[i] + rng)
        ax.set_ylim(c0[j] - rng, c0[j] + rng)
        ax.set_aspect("equal")
        ax.set_title(ttl, color="#cfe9ee", fontsize=11)
        ax.tick_params(colors="#6f8b96", labelsize=8)
        for s in ax.spines.values():
            s.set_color("#26365a")
    fig.suptitle(f"MASt3R-SLAM   Mapping_C8   incremental mapping   |   keyframes {n_kf}   |   points {len(pts):,}",
                 color="#5cff9d", fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    frames.append(np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3].copy())
    plt.close(fig)
    if k % 10 == 0:
        print(f"  渲染 {k+1}/{len(ply_files)} (kf={n_kf}, pts={len(pts)})", flush=True)

frames += [frames[-1]] * 10
os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
imageio.mimsave(out_prefix + ".gif", frames, fps=6, loop=0)
print(f"GIF: {out_prefix}.gif ({len(frames)} 帧)")
try:
    imageio.mimsave(out_prefix + ".mp4", frames, fps=6, quality=8, macro_block_size=None)
    print(f"MP4: {out_prefix}.mp4")
except Exception as e:
    print(f"(mp4 失败, 仅 gif: {e})")
