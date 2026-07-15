#!/usr/bin/env python
"""Offline VIO correction validation with MASt3R loop constraints.

This is an exploratory script meant to answer one question:
can visual loop closures pull a drifting VIO keyframe trajectory back into a
topologically cleaner bird's-eye map?

It does not modify the online pipeline. It:
1. loads saved keyframes from a run plus the raw VIO trajectory,
2. proposes long-range loop candidates from VPR descriptors,
3. verifies candidates with MASt3R pair tracking,
4. solves a 2D pose graph (x, y, yaw) with VIO odometry + visual loop edges,
5. saves a before/after BEV comparison and corrected keyframe poses.
"""
import argparse
import json
import math
import pathlib
import sys
from dataclasses import dataclass

import cv2
import matplotlib
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp
from scipy.sparse import lil_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from mast3r_slam.config import load_config  # noqa: E402
from mast3r_slam.dataloader import load_dataset  # noqa: E402
from mast3r_slam.frame import create_frame  # noqa: E402
from mast3r_slam.mast3r_utils import load_mast3r, mast3r_inference_mono  # noqa: E402
from mast3r_slam.run_config import load_run_config, run_dir, seq_name  # noqa: E402
from mast3r_slam.selavpr import SelaVPRExtractor  # noqa: E402
from mast3r_slam.tracker import FrameTracker  # noqa: E402

import lietorch  # noqa: E402


@dataclass
class LoopEdge:
    i: int
    j: int
    sim: float
    meas_xyyaw: np.ndarray
    raw_vio_dist: float


class SingleKeyframeStore:
    """Minimal adapter so FrameTracker.track() can be reused offline."""

    def __init__(self, keyframe):
        self.kf = keyframe

    def last_keyframe(self):
        return self.kf

    def __len__(self):
        return 1

    def __setitem__(self, idx, value):
        self.kf = value


def wrap_angle(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def load_txt(path):
    arr = np.loadtxt(path, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None]
    return arr


def load_vio(path):
    vio = load_txt(path)
    order = np.argsort(vio[:, 0], kind="stable")
    vio = vio[order]
    _, uniq = np.unique(vio[:, 0], return_index=True)
    return vio[np.sort(uniq)]


def gravity_to_zup(g_world):
    z = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    v = np.cross(g_world, z)
    s = np.linalg.norm(v)
    c = float(g_world @ z)
    if s < 1e-8:
        return np.eye(3, dtype=np.float64) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + vx + vx @ vx * ((1.0 - c) / (s * s))


def estimate_gravity_rotation(vio, imu_path):
    imu = load_txt(imu_path)
    gyr = np.linalg.norm(imu[:, 4:7], axis=1)
    acc = np.linalg.norm(imu[:, 1:4], axis=1)
    stat = (gyr < 0.03) & (acc > 9.4) & (acc < 10.2)
    if not np.any(stat):
        return np.eye(3, dtype=np.float64), None
    vt = vio[:, 0]
    slerp = Slerp(vt, Rotation.from_quat(vio[:, 4:8]))
    its = imu[stat, 0]
    inside = (its >= vt[0]) & (its <= vt[-1])
    its = its[inside]
    g_body = imu[stat][inside][:, 1:4]
    if len(its) == 0:
        return np.eye(3, dtype=np.float64), None
    g_world = slerp(its).apply(g_body).mean(0)
    g_world /= np.linalg.norm(g_world)
    return gravity_to_zup(g_world), g_world


def pose_at_time(vio, slerp, t):
    t = float(np.clip(t, vio[0, 0], vio[-1, 0]))
    p = np.array([np.interp(t, vio[:, 0], vio[:, 1 + k]) for k in range(3)], dtype=np.float64)
    R = slerp([t])[0]
    return p, R


def pose3d_to_2d(pos, rot):
    fwd = rot.as_matrix()[:, 2]
    yaw = math.atan2(float(fwd[1]), float(fwd[0]))
    return np.array([float(pos[0]), float(pos[1]), wrap_angle(yaw)], dtype=np.float64)


def build_sim3(pos, rot, device):
    data = torch.tensor([*pos, *rot.as_quat(), 1.0], dtype=torch.float32, device=device).reshape(1, 8)
    return lietorch.Sim3(data)


def rel_pose_2d(pi, pj):
    ci, si = math.cos(pi[2]), math.sin(pi[2])
    R_iT = np.array([[ci, si], [-si, ci]], dtype=np.float64)
    dt = R_iT @ (pj[:2] - pi[:2])
    return np.array([dt[0], dt[1], wrap_angle(pj[2] - pi[2])], dtype=np.float64)


def abs_from_rel(pi, z):
    ci, si = math.cos(pi[2]), math.sin(pi[2])
    R_i = np.array([[ci, -si], [si, ci]], dtype=np.float64)
    p = pi[:2] + R_i @ z[:2]
    return np.array([p[0], p[1], wrap_angle(pi[2] + z[2])], dtype=np.float64)


def sample_nodes(kf_rows, desc, step):
    idx = np.arange(0, len(kf_rows), max(step, 1), dtype=int)
    if idx[-1] != len(kf_rows) - 1:
        idx = np.unique(np.r_[idx, len(kf_rows) - 1])
    return kf_rows[idx], desc[idx], idx


def load_or_compute_descriptors(run_dir_path, seq, dataset, frame_ids, device, batch_size):
    desc_path = run_dir_path / f"{seq}_vpr_desc.npy"
    if desc_path.exists():
        D = np.load(desc_path).astype(np.float32)
        if len(D) >= len(frame_ids):
            return D[: len(frame_ids)]
        print(f"[warn] {desc_path.name} 长度 {len(D)} < 关键帧数 {len(frame_ids)}, 改为重算描述子")

    print("[vpr] 现成描述子不可用, 重新提取关键帧描述子...")
    ex = SelaVPRExtractor(backbone="dinov2-large", use_hashing=False, use_rerank=False, device=device)
    if isinstance(ex.model, torch.nn.DataParallel):
        ex.model = ex.model.module.to(device)
    imgs = [cv2.imread(str(dataset.dataset_path / f"{int(fid):06d}.png")) for fid in frame_ids]
    out = []
    for i in range(0, len(imgs), batch_size):
        out.append(ex.extract_batch(imgs[i : i + batch_size]))
    D = np.concatenate(out, axis=0).astype(np.float32)
    D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-9
    return D


def propose_loop_candidates(desc, poses2d, min_gap, max_vio_dist, min_score, topk_per_node, dedup_window):
    desc = desc.astype(np.float64)
    desc /= np.linalg.norm(desc, axis=1, keepdims=True) + 1e-9
    sims = desc @ desc.T
    N = len(desc)
    cand = {}
    for i in range(N):
        ranked = np.argsort(-sims[i])
        kept = 0
        for j in ranked:
            if j <= i or (j - i) < min_gap:
                continue
            sim = float(sims[i, j])
            if sim < min_score:
                break
            d = float(np.linalg.norm(poses2d[i, :2] - poses2d[j, :2]))
            if d > max_vio_dist:
                continue
            key = (i, int(j))
            prev = cand.get(key)
            if prev is None or sim > prev[0]:
                cand[key] = (sim, d)
            kept += 1
            if kept >= topk_per_node:
                break

    ordered = sorted(((s, d, i, j) for (i, j), (s, d) in cand.items()), reverse=True)
    accepted = []
    used = []
    for s, d, i, j in ordered:
        clash = False
        for ai, aj in used:
            if abs(i - ai) <= dedup_window and abs(j - aj) <= dedup_window:
                clash = True
                break
        if clash:
            continue
        accepted.append((s, d, i, j))
        used.append((i, j))
    return accepted


def estimate_loop_edge(
    model,
    tracker,
    dataset,
    vio,
    slerp,
    Rg,
    real_time_by_fid,
    frame_ids,
    i,
    j,
    device,
    scale_tol,
):
    fid_i = int(frame_ids[i])
    fid_j = int(frame_ids[j])
    _, img_i = dataset[fid_i]
    _, img_j = dataset[fid_j]
    t_i = float(real_time_by_fid[fid_i])
    t_j = float(real_time_by_fid[fid_j])
    p_i, r_i = pose_at_time(vio, slerp, t_i)
    p_j, r_j = pose_at_time(vio, slerp, t_j)

    frame_i = create_frame(fid_i, img_i, build_sim3(p_i, r_i, device), img_size=dataset.img_size, device=device)
    frame_j = create_frame(fid_j, img_j, build_sim3(p_j, r_j, device), img_size=dataset.img_size, device=device)

    with torch.inference_mode():
        Xj, Cj = mast3r_inference_mono(model, frame_j)
        frame_j.update_pointmap(Xj, Cj)
        tracker.keyframes = SingleKeyframeStore(frame_j)
        tracker.reset_idx_f2k()
        add_new_kf, _, try_reloc = tracker.track(frame_i)

    if try_reloc:
        return None

    M_rel = (frame_j.T_WC.inv() * frame_i.T_WC).matrix().reshape(4, 4).detach().cpu().numpy()
    scale = float(np.cbrt(np.linalg.det(M_rel[:3, :3])))
    if not np.isfinite(scale) or abs(scale - 1.0) > scale_tol:
        return None

    M_i = frame_i.T_WC.matrix().reshape(4, 4).detach().cpu().numpy()
    M_j = frame_j.T_WC.matrix().reshape(4, 4).detach().cpu().numpy()
    p_i_opt = Rg @ M_i[:3, 3]
    p_j_zup = Rg @ M_j[:3, 3]
    r_i_opt = Rotation.from_matrix(Rg @ M_i[:3, :3])
    r_j_zup = Rotation.from_matrix(Rg @ M_j[:3, :3])
    pose_i = pose3d_to_2d(p_i_opt, r_i_opt)
    pose_j = pose3d_to_2d(p_j_zup, r_j_zup)
    return rel_pose_2d(pose_j, pose_i)


def pack_vars(poses):
    return poses[1:].reshape(-1)


def unpack_vars(x, pose0):
    poses = np.empty((1 + len(x) // 3, 3), dtype=np.float64)
    poses[0] = pose0
    poses[1:] = x.reshape(-1, 3)
    poses[:, 2] = np.vectorize(wrap_angle)(poses[:, 2])
    return poses


def build_jac_sparsity(n_nodes, odom_edges, loop_edges):
    n_vars = (n_nodes - 1) * 3
    n_rows = 3 + 3 * len(odom_edges) + 3 * len(loop_edges)
    J = lil_matrix((n_rows, n_vars), dtype=np.int8)
    row = 0
    # prior on node 0 depends on no free vars
    row += 3
    for i, j, _ in odom_edges:
        if i > 0:
            J[row : row + 3, (i - 1) * 3 : i * 3] = 1
        if j > 0:
            J[row : row + 3, (j - 1) * 3 : j * 3] = 1
        row += 3
    for edge in loop_edges:
        if edge.i > 0:
            J[row : row + 3, (edge.i - 1) * 3 : edge.i * 3] = 1
        if edge.j > 0:
            J[row : row + 3, (edge.j - 1) * 3 : edge.j * 3] = 1
        row += 3
    return J.tocsr()


def make_residual_fn(pose0, odom_edges, loop_edges, prior_w, odom_w_xy, odom_w_yaw, loop_w_xy, loop_w_yaw):
    def residual(x):
        poses = unpack_vars(x, pose0)
        rows = []
        rows.append(prior_w * (poses[0] - pose0))
        rows[-1][2] = prior_w * wrap_angle(rows[-1][2])

        for i, j, meas in odom_edges:
            pred = rel_pose_2d(poses[i], poses[j])
            err = pred - meas
            err[2] = wrap_angle(err[2])
            rows.append(np.array([odom_w_xy * err[0], odom_w_xy * err[1], odom_w_yaw * err[2]], dtype=np.float64))

        for edge in loop_edges:
            pred = rel_pose_2d(poses[edge.i], poses[edge.j])
            err = pred - edge.meas_xyyaw
            err[2] = wrap_angle(err[2])
            w = max(edge.sim, 1e-3)
            rows.append(np.array([loop_w_xy * w * err[0], loop_w_xy * w * err[1], loop_w_yaw * w * err[2]], dtype=np.float64))

        return np.concatenate(rows, axis=0)

    return residual


def save_plot(out_path, raw_poses, corrected_poses, loop_edges, title):
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.plot(raw_poses[:, 0], raw_poses[:, 1], color="#d97706", lw=2.0, label="raw VIO keyframes")
    ax.plot(corrected_poses[:, 0], corrected_poses[:, 1], color="#2563eb", lw=2.0, label="corrected keyframes")
    if loop_edges:
        for edge in loop_edges:
            a = corrected_poses[edge.i, :2]
            b = corrected_poses[edge.j, :2]
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#94a3b8", lw=0.7, alpha=0.45)
    ax.scatter(raw_poses[0, 0], raw_poses[0, 1], c="#16a34a", s=70, label="start")
    ax.scatter(raw_poses[-1, 0], raw_poses[-1, 1], c="#dc2626", s=70, label="end")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True

    rc = load_run_config()
    default_dataset = rc.get("dataset", "datasets/cfds_floor28")
    default_run = str(run_dir(rc))
    default_seq = seq_name(rc)
    default_cfg = rc.get("config", f"config/{default_seq}.yaml")

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=default_run)
    ap.add_argument("--dataset", default=default_dataset)
    ap.add_argument("--seq", default=default_seq)
    ap.add_argument("--config", default=default_cfg)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--node-step", type=int, default=2, help="use every Nth keyframe node in the pose graph")
    ap.add_argument("--min-gap", type=int, default=40, help="minimum keyframe index gap for loop candidates (after node-step)")
    ap.add_argument("--max-vio-dist", type=float, default=4.0, help="candidate loop raw-VIO distance gate in meters")
    ap.add_argument("--min-vpr-score", type=float, default=0.62)
    ap.add_argument("--topk-per-node", type=int, default=2)
    ap.add_argument("--dedup-window", type=int, default=6)
    ap.add_argument("--max-candidates", type=int, default=36)
    ap.add_argument("--max-verified-loops", type=int, default=18)
    ap.add_argument("--scale-tol", type=float, default=0.15)
    ap.add_argument("--odom-sigma-xy", type=float, default=0.10)
    ap.add_argument("--odom-sigma-yaw-deg", type=float, default=3.0)
    ap.add_argument("--loop-sigma-xy", type=float, default=0.25)
    ap.add_argument("--loop-sigma-yaw-deg", type=float, default=10.0)
    ap.add_argument("--prior-sigma", type=float, default=1e-4)
    ap.add_argument("--vpr-batch", type=int, default=12)
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()

    run = pathlib.Path(args.run)
    dataset_path = pathlib.Path(args.dataset)
    out_prefix = run / (args.out_prefix or f"{args.seq}_vio_mast3r_loop2d")

    load_config(args.config)
    dataset = load_dataset(str(dataset_path))

    kf_rows = load_txt(run / f"{args.seq}_kf_poses.txt")
    frame_ids = kf_rows[:, 0].astype(int)
    desc = load_or_compute_descriptors(run, args.seq, dataset, frame_ids, args.device, args.vpr_batch)
    desc = desc[: len(frame_ids)]

    vio = load_vio(dataset_path / "vio.txt")
    timestamps = load_txt(dataset_path / "timestamps.txt")
    real_time_by_fid = timestamps[:, 1]
    real_times = timestamps[frame_ids, 1]
    slerp = Slerp(vio[:, 0], Rotation.from_quat(vio[:, 4:8]))
    Rg, g_world = estimate_gravity_rotation(vio, dataset_path / "imu.txt") if (dataset_path / "imu.txt").exists() else (np.eye(3), None)

    raw_poses_full = []
    for t in real_times:
        p, r = pose_at_time(vio, slerp, t)
        raw_poses_full.append(pose3d_to_2d(Rg @ p, Rotation.from_matrix(Rg @ r.as_matrix())))
    raw_poses_full = np.asarray(raw_poses_full, dtype=np.float64)

    kf_rows_s, desc_s, sample_idx = sample_nodes(kf_rows, desc, args.node_step)
    frame_ids_s = kf_rows_s[:, 0].astype(int)
    raw_poses = raw_poses_full[sample_idx]

    odom_edges = []
    for i in range(len(raw_poses) - 1):
        odom_edges.append((i, i + 1, rel_pose_2d(raw_poses[i], raw_poses[i + 1])))

    cand = propose_loop_candidates(
        desc_s,
        raw_poses,
        args.min_gap,
        args.max_vio_dist,
        args.min_vpr_score,
        args.topk_per_node,
        args.dedup_window,
    )
    cand = cand[: args.max_candidates]
    print(f"[loops] 候选 {len(cand)} 条 (node_step={args.node_step})")

    model = load_mast3r(device=args.device)
    tracker = FrameTracker(model, None, args.device)

    loop_edges = []
    for rank, (sim, vio_d, i, j) in enumerate(cand, start=1):
        print(f"[loops] verify {rank}/{len(cand)}  nodes {i}->{j}  sim={sim:.3f}  vio_d={vio_d:.2f}m")
        try:
            meas = estimate_loop_edge(
                model,
                tracker,
                dataset,
                vio,
                slerp,
                Rg,
                real_time_by_fid,
                frame_ids_s,
                i,
                j,
                args.device,
                args.scale_tol,
            )
        except Exception as exc:
            print(f"[loops]   reject: {exc}")
            meas = None
        if meas is not None:
            loop_edges.append(LoopEdge(i=i, j=j, sim=sim, meas_xyyaw=meas, raw_vio_dist=vio_d))
            print(f"[loops]   accept: rel=({meas[0]:.2f}, {meas[1]:.2f}, {np.degrees(meas[2]):.1f}deg)")
        if len(loop_edges) >= args.max_verified_loops:
            break
        torch.cuda.empty_cache()

    print(f"[loops] accepted {len(loop_edges)} / {len(cand)}")
    if len(loop_edges) == 0:
        raise RuntimeError("没有通过 MASt3R 验证的 loop edges，无法继续做离线纠偏")

    pose0 = raw_poses[0].copy()
    x0 = pack_vars(raw_poses)
    residual = make_residual_fn(
        pose0,
        odom_edges,
        loop_edges,
        prior_w=1.0 / args.prior_sigma,
        odom_w_xy=1.0 / args.odom_sigma_xy,
        odom_w_yaw=1.0 / np.deg2rad(args.odom_sigma_yaw_deg),
        loop_w_xy=1.0 / args.loop_sigma_xy,
        loop_w_yaw=1.0 / np.deg2rad(args.loop_sigma_yaw_deg),
    )
    jac_sparsity = build_jac_sparsity(len(raw_poses), odom_edges, loop_edges)
    res = least_squares(
        residual,
        x0,
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        jac_sparsity=jac_sparsity,
        max_nfev=120,
        verbose=2,
    )
    corrected = unpack_vars(res.x, pose0)

    np.savetxt(
        out_prefix.with_suffix(".txt"),
        np.c_[frame_ids_s, real_times[sample_idx], raw_poses, corrected],
        fmt="%.9f",
        header="frame_id real_time raw_x raw_y raw_yaw corrected_x corrected_y corrected_yaw",
    )
    meta = {
        "dataset": str(dataset_path),
        "run": str(run),
        "seq": args.seq,
        "gravity_world": None if g_world is None else [float(x) for x in g_world],
        "node_step": int(args.node_step),
        "candidate_count": int(len(cand)),
        "accepted_loop_count": int(len(loop_edges)),
        "optimization_cost": float(res.cost),
        "optimization_success": bool(res.success),
        "optimization_message": res.message,
        "loops": [
            {
                "i": int(e.i),
                "j": int(e.j),
                "frame_i": int(frame_ids_s[e.i]),
                "frame_j": int(frame_ids_s[e.j]),
                "sim": float(e.sim),
                "raw_vio_dist": float(e.raw_vio_dist),
                "meas_xyyaw": [float(v) for v in e.meas_xyyaw],
            }
            for e in loop_edges
        ],
    }
    out_prefix.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    save_plot(
        out_prefix.with_suffix(".png"),
        raw_poses,
        corrected,
        loop_edges,
        f"VIO Loop Correction Validation | {args.seq}",
    )

    raw_len = float(np.linalg.norm(np.diff(raw_poses[:, :2], axis=0), axis=1).sum())
    corr_len = float(np.linalg.norm(np.diff(corrected[:, :2], axis=0), axis=1).sum())
    print(f"[done] raw len={raw_len:.1f}m  corrected len={corr_len:.1f}m")
    print(f"[done] text -> {out_prefix.with_suffix('.txt')}")
    print(f"[done] json -> {out_prefix.with_suffix('.json')}")
    print(f"[done] plot -> {out_prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
