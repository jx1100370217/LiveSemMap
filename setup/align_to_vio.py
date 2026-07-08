#!/usr/bin/env python
"""VIO 尺度 + 重力对齐 (后处理, 不改 SLAM 核心)。

把单目 MASt3R-SLAM 的 Sim(3) 重建(任意尺度、可能倾斜)鲁棒对齐到 VIO 度量轨迹:
  1) RANSAC Umeyama 求 相似变换(c,R,t) 使 MASt3R 相机中心 ≈ VIO 相机中心 -> 恢复真实米制尺度;
  2) 用 IMU 重力把地图校平到 Z-up(VIO world 已近似重力对齐, 这步主要定 Z 符号/微调)。

输入:
  --run  logs/<run>          含 <seq>_kf_poses.txt (frame_id t cx cy cz) 与 <seq>.ply
  --data datasets/insight9   含 timestamps.txt, vio.txt, imu.txt
输出到 logs/<run>/:
  <seq>_metric.ply           米制 + Z-up 点云 (可直接用于导航/占据栅格)
  <seq>_metric_traj.txt      对齐后关键帧轨迹 vs VIO 对照
  <seq>_metric_transform.txt 变换参数 (c, R, t, Rg)

用法:
  python setup/align_to_vio.py --run logs/insight9_rgb --data datasets/insight9
"""
import argparse
import pathlib

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation, Slerp


def umeyama(X, Y):
    """Umeyama(1991): 求 c,R,t 使 c*R@X.T + t ≈ Y。X,Y: (N,3)。"""
    n = len(X)
    mux, muy = X.mean(0), Y.mean(0)
    Xc, Yc = X - mux, Y - muy
    Sigma = (Yc.T @ Xc) / n
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    varx = (Xc ** 2).sum() / n
    c = float(np.trace(np.diag(D) @ S) / varx)
    t = muy - c * R @ mux
    return c, R, t


def ransac_umeyama(X, Y, iters=3000, thresh=0.30, seed=0):
    """3 点最小集 RANSAC + 内点重拟合, 抗 MASt3R 位姿外点。"""
    rng = np.random.default_rng(seed)
    N = len(X)
    best = None
    for _ in range(iters):
        idx = rng.choice(N, 3, replace=False)
        try:
            c, R, t = umeyama(X[idx], Y[idx])
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(c) or c <= 0:
            continue
        err = np.linalg.norm((c * (R @ X.T).T + t) - Y, axis=1)
        inl = err < thresh
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 3:
        raise RuntimeError("RANSAC 未找到足够内点; 检查 VIO/关键帧配对或放大 --inlier-thresh")
    c, R, t = umeyama(X[best], Y[best])
    err = np.linalg.norm((c * (R @ X.T).T + t) - Y, axis=1)
    return c, R, t, best, err


def load_vio(path):
    v = np.loadtxt(path)
    order = np.argsort(v[:, 0], kind="stable")
    v = v[order]
    _, uniq = np.unique(v[:, 0], return_index=True)  # 时间严格递增(Slerp 要求)
    return v[np.sort(uniq)]


def save_ply(path, points, colors):
    pts = np.array(
        [tuple(p) + tuple(c) for p, c in zip(points.astype(np.float32), colors.astype(np.uint8))],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    PlyData([PlyElement.describe(pts, "vertex")], text=False).write(str(path))


def gravity_to_zup(g_world):
    """求把 g_world 转到 -Z(重力朝下, Z朝上) 的旋转。"""
    z = np.array([0, 0, -1.0])
    v = np.cross(g_world, z)
    s = np.linalg.norm(v)
    c = g_world @ z
    if s < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="logs/<run> 目录")
    ap.add_argument("--data", default="datasets/insight9")
    ap.add_argument("--seq", default=None, help="序列名(默认取 *_kf_poses.txt 前缀)")
    ap.add_argument("--inlier-thresh", type=float, default=0.30, help="RANSAC 内点阈值(米)")
    args = ap.parse_args()

    run, data = pathlib.Path(args.run), pathlib.Path(args.data)
    seq = args.seq or sorted(run.glob("*_kf_poses.txt"))[0].name.replace("_kf_poses.txt", "")

    # 1. MASt3R 关键帧中心 (frame_id t cx cy cz)
    kf = np.loadtxt(run / f"{seq}_kf_poses.txt")
    if kf.ndim == 1:
        kf = kf[None]
    kf_t, kf_c = kf[:, 1], kf[:, 2:5]

    # 2. 数据集时间戳 t (=full_idx/30) -> 真实时间 (timestamps.txt: idx real_time)
    ts = np.loadtxt(data / "timestamps.txt")
    full_idx = np.clip(np.round(kf_t * 30).astype(int), 0, len(ts) - 1)
    kf_realt = ts[full_idx, 1]

    # 3. VIO 位置插值到关键帧真实时间
    vio = load_vio(data / "vio.txt")
    vt, vp = vio[:, 0], vio[:, 1:4]
    vio_c = np.stack([np.interp(kf_realt, vt, vp[:, k]) for k in range(3)], 1)

    valid = (
        (kf_realt >= vt[0]) & (kf_realt <= vt[-1])
        & np.isfinite(kf_c).all(1) & (np.abs(kf_c) < 1e4).all(1)
    )
    X, Y = kf_c[valid], vio_c[valid]
    print(f"关键帧 {len(kf)}, 时间落在VIO内且有效的配对 {valid.sum()}")
    if valid.sum() < 3:
        raise RuntimeError("有效配对 < 3, 无法对齐")

    # 4. 鲁棒 Sim3 对齐 MASt3R -> VIO(米)
    c, R, t, inl, err = ransac_umeyama(X, Y, thresh=args.inlier_thresh)
    rmse = np.sqrt((err[inl] ** 2).mean())
    print("\n== Sim3 对齐 (MASt3R -> VIO 度量) ==")
    print(f"  尺度 c = {c:.4f}   (1 MASt3R单位 = {c:.3f} m)")
    print(f"  内点 {int(inl.sum())}/{len(X)},  内点RMSE = {rmse*100:.1f} cm")

    # 5. 重力 -> Z-up (IMU 静止段的 accel 经 VIO 姿态转到 world)
    imu = np.loadtxt(data / "imu.txt")
    gyr = np.linalg.norm(imu[:, 4:7], axis=1)
    acc = np.linalg.norm(imu[:, 1:4], axis=1)
    stat = (gyr < 0.03) & (acc > 9.4) & (acc < 10.2)
    slerp = Slerp(vt, Rotation.from_quat(vio[:, 4:8]))
    its = imu[stat, 0]
    m = (its >= vt[0]) & (its <= vt[-1])
    its, g_body = its[m], imu[stat][m][:, 1:4]
    g_world = slerp(its).apply(g_body).mean(0)
    g_world /= np.linalg.norm(g_world)
    Rg = gravity_to_zup(g_world)
    print(f"\n== 重力(VIO world) = {np.round(g_world,3)} == (VIO world 重力对齐则≈±Z; Rg 转到 Z-up)")

    def transform(P):
        return (Rg @ (c * (R @ P.T).T + t).T).T

    # 6. 变换点云 -> 米制 Z-up
    d = PlyData.read(run / f"{seq}.ply")["vertex"]
    P = np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64)
    col = np.stack([d["red"], d["green"], d["blue"]], 1).astype(np.uint8)
    ok = np.isfinite(P).all(1) & (np.abs(P) < 1e4).all(1)
    Pm = transform(P[ok])
    save_ply(run / f"{seq}_metric.ply", Pm, col[ok])

    # 7. 轨迹对照 + 变换参数
    kf_out = transform(kf_c[valid])
    np.savetxt(run / f"{seq}_metric_traj.txt", np.c_[kf_realt[valid], kf_out, Y],
               header="t  mast3r_metric(x y z)  vio(x y z)  [米, 同系]")
    with open(run / f"{seq}_metric_transform.txt", "w") as f:
        f.write(f"# p_metric = Rg @ (c * R @ p + t)\nc {c:.9f}\n")
        f.write("R " + " ".join(f"{x:.9f}" for x in R.reshape(-1)) + "\n")
        f.write("t " + " ".join(f"{x:.9f}" for x in t) + "\n")
        f.write("Rg " + " ".join(f"{x:.9f}" for x in Rg.reshape(-1)) + "\n")

    zspan = Pm[:, 2].max() - Pm[:, 2].min()
    print("\n== 结果 ==")
    print(f"  米制点云 -> {run/(seq+'_metric.ply')}  ({len(Pm)} 点)")
    print(f"  地图尺寸(米): X {Pm[:,0].max()-Pm[:,0].min():.1f}  Y {Pm[:,1].max()-Pm[:,1].min():.1f}  Z(高) {zspan:.1f}")
    print(f"  VIO 轨迹总长(米): {np.linalg.norm(np.diff(Y,axis=0),axis=1).sum():.1f}")


if __name__ == "__main__":
    main()
