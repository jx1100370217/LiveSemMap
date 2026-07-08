import pathlib
from typing import Optional
import cv2
import numpy as np
import torch
from mast3r_slam.dataloader import Intrinsics
from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.config import config
from mast3r_slam.geometry import constrain_points_to_ray
from plyfile import PlyData, PlyElement


def prepare_savedir(args, dataset):
    save_dir = pathlib.Path("logs")
    if args.save_as != "default":
        save_dir = save_dir / args.save_as
    save_dir.mkdir(exist_ok=True, parents=True)
    seq_name = dataset.dataset_path.stem
    return save_dir, seq_name


def save_traj(
    logdir,
    logfile,
    timestamps,
    frames: SharedKeyframes,
    intrinsics: Optional[Intrinsics] = None,
):
    # log
    logdir = pathlib.Path(logdir)
    logdir.mkdir(exist_ok=True, parents=True)
    logfile = logdir / logfile
    with open(logfile, "w") as f:
        # for keyframe_id in frames.keyframe_ids:
        for i in range(len(frames)):
            keyframe = frames[i]
            t = timestamps[keyframe.frame_id]
            if intrinsics is None:
                T_WC = as_SE3(keyframe.T_WC)
            else:
                T_WC = intrinsics.refine_pose_with_calibration(keyframe)
            x, y, z, qx, qy, qz, qw = T_WC.data.numpy().reshape(-1)
            f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")


def save_keyframe_poses(logdir, logfile, timestamps, keyframes: SharedKeyframes):
    """存每关键帧 `frame_id  t  cx cy cz` —— t=数据集时间戳, (cx,cy,cz)=相机中心
    (与重建 .ply 同系, = Sim3 矩阵平移)。供 VIO 尺度+重力对齐(setup/align_to_vio.py),
    避免 save_traj 的 SE3/refine 坐标歧义。t 用于把关键帧映射回真实时间查 VIO。"""
    logdir = pathlib.Path(logdir)
    logdir.mkdir(exist_ok=True, parents=True)
    rows = []
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        c = keyframe.T_WC.matrix().reshape(-1, 4, 4)[0, :3, 3].cpu().numpy()
        t = timestamps[keyframe.frame_id]
        rows.append([int(keyframe.frame_id), float(t), float(c[0]), float(c[1]), float(c[2])])
    np.savetxt(
        pathlib.Path(logdir) / logfile,
        np.array(rows),
        fmt=["%d", "%.9f", "%.9f", "%.9f", "%.9f"],
        header="frame_id  t  cx cy cz  (t=数据集时间戳; 中心与重建同系)",
    )


def save_reconstruction(savedir, filename, keyframes, c_conf_threshold):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    pointclouds = []
    colors = []
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        if config["use_calib"]:
            X_canon = constrain_points_to_ray(
                keyframe.img_shape.flatten()[:2], keyframe.X_canon[None], keyframe.K
            )
            keyframe.X_canon = X_canon.squeeze(0)
        pW = keyframe.T_WC.act(keyframe.X_canon).cpu().numpy().reshape(-1, 3)
        color = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
        valid = (
            keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
            > c_conf_threshold
        )
        pointclouds.append(pW[valid])
        colors.append(color[valid])
    pointclouds = np.concatenate(pointclouds, axis=0)
    colors = np.concatenate(colors, axis=0)

    save_ply(savedir / filename, pointclouds, colors)


def save_reconstruction_vio(savedir, filename, keyframes, vio_prior, c_conf_threshold):
    """VIO 位姿重建 (治单目 Sim3 累积漂移): 每个关键帧的相机系点云用 **VIO 位姿** 摆放,
    而非漂移的 MASt3R 位姿 —— VIO 管全局轨迹(米制/无漂移), MASt3R 管局部几何。仅 --vio 时。"""
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    N = len(keyframes)
    mast_c, vio_p, vio_R = [], [], []
    for i in range(N):
        kf = keyframes[i]
        c = kf.T_WC.matrix().reshape(-1, 4, 4)[0, :3, 3].cpu().numpy()
        p, R = vio_prior._pose_at(int(kf.frame_id))
        mast_c.append(c)
        vio_p.append(p)
        vio_R.append(R.as_matrix())
    mast_c, vio_p, vio_R = np.array(mast_c), np.array(vio_p), np.array(vio_R)
    # 全局尺度 s (MASt3R单位->米): 相邻关键帧位移比的稳健中位数
    dm = np.linalg.norm(np.diff(mast_c, axis=0), axis=1)
    dv = np.linalg.norm(np.diff(vio_p, axis=0), axis=1)
    good = (dm > 1e-4) & (dv > 0.02) & np.isfinite(dm) & np.isfinite(dv)
    s = float(np.median(dv[good] / dm[good])) if good.any() else 1.0
    pts, cols = [], []
    for i in range(N):
        kf = keyframes[i]
        X = kf.X_canon
        if config["use_calib"]:
            X = constrain_points_to_ray(kf.img_shape.flatten()[:2], X[None], kf.K).squeeze(0)
        X = X.cpu().numpy().reshape(-1, 3).astype(np.float64) * s  # 米制相机系点
        world = (vio_R[i] @ X.T).T + vio_p[i]                       # 用 VIO 位姿摆放
        col = (kf.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
        valid = kf.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1) > c_conf_threshold
        pts.append(world[valid])
        cols.append(col[valid])
    P = np.concatenate(pts, 0)
    save_ply(savedir / filename, P, np.concatenate(cols, 0))
    print(f"[VIO重建] 尺度 s={s:.4f} m/单位, {len(P)} 点 -> {filename} (无漂移米制图)")


def save_keyframes(savedir, timestamps, keyframes: SharedKeyframes):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        t = timestamps[keyframe.frame_id]
        filename = savedir / f"{t}.png"
        cv2.imwrite(
            str(filename),
            cv2.cvtColor(
                (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
            ),
        )


def save_ply(filename, points, colors):
    colors = colors.astype(np.uint8)
    # Combine XYZ and RGB into a structured array
    pcd = np.empty(
        len(points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    pcd["x"], pcd["y"], pcd["z"] = points.T
    pcd["red"], pcd["green"], pcd["blue"] = colors.T
    vertex_element = PlyElement.describe(pcd, "vertex")
    ply_data = PlyData([vertex_element], text=False)
    ply_data.write(filename)
