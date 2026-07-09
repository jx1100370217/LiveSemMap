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


def save_keyframes(savedir, timestamps, keyframes: SharedKeyframes, start=0):
    """写关键帧图像; start>0 时只写新增部分(增量保存用, 图像入容器后不再变)。
    返回已写到的关键帧数, 供调用方作为下次 start。"""
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    n = len(keyframes)
    for i in range(start, n):
        keyframe = keyframes[i]
        t = timestamps[keyframe.frame_id]
        filename = savedir / f"{t}.png"
        cv2.imwrite(
            str(filename),
            cv2.cvtColor(
                (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
            ),
        )
    return n


def _kf_world_geometry(keyframes, vio_prior, c_conf_threshold, stride=6):
    """所有关键帧的 (世界系抽样点云, 颜色, 每帧相机位置)。

    VIO 可用: 相机系点 × 全局尺度 s 后用 VIO 位姿摆放 (米制/无漂移, 与 _vio.ply 同系);
    否则用 SLAM Sim3 位姿 (尺度未知)。返回 (P, C, kf_pos(N,3), coord_name)。"""
    N = len(keyframes)
    use_vio = vio_prior is not None
    s = 1.0
    vio_p, vio_R = None, None
    if use_vio:
        mast_c = []
        vio_p, vio_R = [], []
        for i in range(N):
            kf = keyframes[i]
            mast_c.append(kf.T_WC.matrix().reshape(-1, 4, 4)[0, :3, 3].cpu().numpy())
            p, R = vio_prior._pose_at(int(kf.frame_id))
            vio_p.append(p)
            vio_R.append(R.as_matrix())
        mast_c, vio_p, vio_R = np.array(mast_c), np.array(vio_p), np.array(vio_R)
        dm = np.linalg.norm(np.diff(mast_c, axis=0), axis=1)
        dv = np.linalg.norm(np.diff(vio_p, axis=0), axis=1)
        good = (dm > 1e-4) & (dv > 0.02) & np.isfinite(dm) & np.isfinite(dv)
        s = float(np.median(dv[good] / dm[good])) if good.any() else 1.0

    pts, cols, kf_pos = [], [], []
    for i in range(N):
        kf = keyframes[i]
        h, w = [int(x) for x in kf.img_shape.flatten()[:2]]
        X = kf.X_canon
        if config["use_calib"]:
            X = constrain_points_to_ray((h, w), X[None], kf.K).squeeze(0)
        X = X.reshape(h, w, 3)[::stride, ::stride].reshape(-1, 3).cpu().numpy()
        col = kf.uimg.reshape(h, w, 3)[::stride, ::stride].reshape(-1, 3).cpu().numpy()
        conf = kf.get_average_conf().reshape(h, w)[::stride, ::stride].reshape(-1)
        m = conf.cpu().numpy().astype(np.float32) > c_conf_threshold
        if use_vio:
            world = (vio_R[i] @ (X.astype(np.float64) * s).T).T + vio_p[i]
            kf_pos.append(vio_p[i])
        else:
            M = kf.T_WC.matrix().reshape(4, 4).cpu().numpy().astype(np.float64)
            world = X @ M[:3, :3].T + M[:3, 3]
            kf_pos.append(M[:3, 3])
        pts.append(world[m])
        cols.append(np.clip(col[m], 0, 1))
    P = np.concatenate(pts, 0)
    C = np.concatenate(cols, 0)
    ok = np.isfinite(P).all(1) & (np.abs(P) < 1e4).all(1)
    return P[ok], C[ok], np.asarray(kf_pos, np.float64), ("vio" if use_vio else "slam")


def save_semantic_map(savedir, seq_name, keyframes, semantic_ann, vio_prior,
                      c_conf_threshold, G=480, verbose=True):
    """保存 2D 语义地图产物 (中断/正常退出统一入口):
    - occupancy.npz: 原始占据栅格(0未知/1可通行/2障碍) + 世界<->像素 meta + 关键帧位置
    - bev.png / occupancy.png: 彩色俯视图与占据图 (含语义节点高亮+中文标签)
    - semantic.json: 聚合语义节点 + 逐关键帧原始标注
    返回聚合节点列表。"""
    import json
    from mast3r_slam.mapping2d import rasterize_map, draw_semantic_nodes, world_to_px
    from mast3r_slam.semantic import aggregate_nodes

    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    N = len(keyframes)
    if N == 0:
        print("[semantic_map] 无关键帧, 跳过")
        return []

    P, C, kf_pos, coord = _kf_world_geometry(keyframes, vio_prior, c_conf_threshold)
    if len(P) < 100:
        print("[semantic_map] 有效点过少, 跳过")
        return []
    bev, occ, grid, meta = rasterize_map(P, C, kf_pos, G=G)

    ann = dict(semantic_ann) if semantic_ann is not None else {}
    pos_by_kf = {i: kf_pos[i] for i in range(N) if np.isfinite(kf_pos[i]).all()}
    nodes = aggregate_nodes(ann, pos_by_kf)
    draw_semantic_nodes(bev, nodes, meta)
    draw_semantic_nodes(occ, nodes, meta, label=False)

    kf_px, kf_py = world_to_px(kf_pos, meta)
    frame_ids = [int(keyframes.dataset_idx[i]) for i in range(N)]

    # 原子写: 先写 tmp_ 前缀文件再 rename —— 增量保存周期重写这些文件,
    # kill -9 / 并发读(export_web) 不会撞上写了一半的产物。
    # 临时名保持真实扩展名 (np.savez 会给非 .npz 路径追加后缀, cv2.imwrite 按扩展名选编码器)
    def _atomic(path, write_fn):
        tmp = path.with_name("tmp_" + path.name)
        write_fn(tmp)
        tmp.replace(path)

    _atomic(savedir / f"{seq_name}_occupancy.npz", lambda p: np.savez_compressed(
        p, grid=grid, meta=json.dumps(meta), kf_pos=kf_pos,
        kf_px=np.stack([kf_px, kf_py], 1), frame_ids=np.array(frame_ids),
        coordinate=coord))
    _atomic(savedir / f"{seq_name}_bev.png", lambda p: cv2.imwrite(
        str(p), cv2.cvtColor((np.clip(bev, 0, 1) * 255).astype(np.uint8),
                             cv2.COLOR_RGB2BGR)))
    _atomic(savedir / f"{seq_name}_occupancy.png", lambda p: cv2.imwrite(
        str(p), cv2.cvtColor((np.clip(occ, 0, 1) * 255).astype(np.uint8),
                             cv2.COLOR_RGB2BGR)))

    def _write_json(p):
        with open(p, "w") as f:
            json.dump({
                "coordinate": coord,
                "nodes": nodes,
                "annotations": {str(k): v for k, v in sorted(ann.items())},
                "kf_positions": kf_pos.tolist(),
                "frame_ids": frame_ids,
            }, f, ensure_ascii=False, indent=1)
    _atomic(savedir / f"{seq_name}_semantic.json", _write_json)
    if verbose:
        print(f"[semantic_map] {len(nodes)} 个语义节点, 栅格 {G}x{G} ({coord} 系) -> "
              f"{seq_name}_semantic.json / _occupancy.npz / _bev.png")
    return nodes


def save_vpr_descriptors(savedir, seq_name, keyframes, batch=12):
    """对全部关键帧提取 SelaVPR++ 全局描述子 (4096 维, L2 归一化) 存 npy,
    供导航时 VPR 重定位/图像指定起点。模型仅在此处加载(退出时一次性), 不占建图显存。"""
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    N = len(keyframes)
    if N == 0:
        return
    print(f"[vpr] 加载 SelaVPR++ 并提取 {N} 个关键帧描述子...")
    try:
        import torch as _torch
        from mast3r_slam.selavpr import SelaVPRExtractor

        ex = SelaVPRExtractor(backbone="dinov2-large", use_hashing=False,
                              use_rerank=False, device="cuda:0")
        if isinstance(ex.model, _torch.nn.DataParallel):
            ex.model = ex.model.module.to("cuda:0")
        bgrs = []
        for i in range(N):
            rgb = (keyframes[i].uimg.cpu().numpy() * 255).astype(np.uint8)
            bgrs.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        out = []
        for i in range(0, N, batch):
            out.append(ex.extract_batch(bgrs[i:i + batch]))
            if i // batch % 5 == 0:
                print(f"[vpr] {min(i + batch, N)}/{N}")
        D = np.concatenate(out, 0).astype(np.float32)
        D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-9
        np.save(savedir / f"{seq_name}_vpr_desc.npy", D)
        print(f"[vpr] 描述子 {D.shape} -> {seq_name}_vpr_desc.npy")
    except Exception as e:
        print(f"[vpr] 描述子提取失败(不影响其他产物): {e}")


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
