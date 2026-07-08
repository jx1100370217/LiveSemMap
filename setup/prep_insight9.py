#!/usr/bin/env python
"""从 insight9 ROS2 mcap 抽出 MASt3R-SLAM 建图所需数据 (RealSense 类相机 wrist_right)。

产出到 --out (默认 datasets/insight9/):
  rgb/000000.png ...      已 rectified 的彩色帧(长边缩到 --img-size), 供纯 RGB 建图(同 Mapping_C8)
  timestamps.txt          `帧号 真实时间(秒)`  —— 供把 MASt3R 帧号映射回真实时间查 VIO
  vio.txt                 `t tx ty tz qx qy qz qw` (TUM, world 系度量轨迹) —— VIO 尺度+重力对齐用
  imu.txt                 `t ax ay az gx gy gz` (400Hz)
  depth/000000.png ...    (可选 --with-depth) mono16 毫米原始深度, 在左红外frame
并写 config/intrinsics_insight9.yaml (缩放后的针孔内参) 与 config/insight9.yaml。

用法:
  python setup/prep_insight9.py /home/jx/datas/0_insight9_raw_*.mcap
  python setup/prep_insight9.py <mcap> --limit 30      # 调试: 只抽前30帧RGB
"""
import argparse
import pathlib

import cv2
import numpy as np
import yaml
from mcap_ros2.reader import read_ros2_messages

BASE = "/camera/wrist_right"
T_RGB = f"{BASE}/color/image_rect_raw/compressed"
T_CI = f"{BASE}/color/camera_info"
T_VIO = f"{BASE}/vio_100hz"
T_IMU = f"{BASE}/imu"
T_DEP = f"{BASE}/depth/image_rect_raw"


def stamp(msg):
    s = msg.header.stamp
    return s.sec + s.nanosec * 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mcap")
    ap.add_argument("--out", default="datasets/insight9")
    ap.add_argument("--img-size", type=int, default=512, help="彩色长边缩放到(像素)")
    ap.add_argument("--rgb-stride", type=int, default=1, help="每N帧取1帧RGB(降采样)")
    ap.add_argument("--min-trans", type=float, default=0.01,
                    help="VIO运动过滤:相对上一保留帧平移<此(米)且旋转<--min-rot 则丢弃(去静止/冗余); 0=关闭")
    ap.add_argument("--min-rot", type=float, default=1.0, help="VIO运动过滤:旋转阈值(度)")
    ap.add_argument("--with-depth", action="store_true", help="同时导出原始深度(mono16毫米,左红外frame)")
    ap.add_argument("--limit", type=int, default=0, help="调试:只处理前N个RGB消息")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)  # RGB 扁平放(RGBFiles 非递归 glob *.png), sidecar txt 同目录
    for old in out.glob("*.png"):  # 清旧抽取结果, 避免残留高编号帧被 RGBFiles 读入
        old.unlink()
    if args.with_depth:
        (out / "depth").mkdir(parents=True, exist_ok=True)
        for old in (out / "depth").glob("*.png"):
            old.unlink()

    topics = [T_RGB, T_CI, T_VIO, T_IMU] + ([T_DEP] if args.with_depth else [])
    vio, imu, rgb_ts, depth_ts = [], [], [], []
    K0 = None
    W0 = H0 = None
    scale = None
    Wr = Hr = None
    n_rgb_seen = 0
    last_vio = None      # 最近 VIO 位姿 (px,py,pz, qx,qy,qz,qw), 用作 RGB 帧的运动判据
    last_kept = None     # 上一个保留 RGB 帧对应的 VIO 位姿
    n_dropped = 0

    print(f"读 mcap: {args.mcap}")
    for m in read_ros2_messages(args.mcap, topics=topics):
        topic = m.channel.topic
        msg = m.ros_msg

        if topic == T_CI:
            if K0 is None:
                K0 = np.array(msg.k, dtype=np.float64).reshape(3, 3)
                W0, H0 = int(msg.width), int(msg.height)

        elif topic == T_VIO:
            p, q = msg.pose.position, msg.pose.orientation
            vio.append([stamp(msg), p.x, p.y, p.z, q.x, q.y, q.z, q.w])
            last_vio = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)

        elif topic == T_IMU:
            a, g = msg.linear_acceleration, msg.angular_velocity
            imu.append([stamp(msg), a.x, a.y, a.z, g.x, g.y, g.z])

        elif topic == T_RGB:
            n_rgb_seen += 1
            if args.limit and n_rgb_seen > args.limit:
                break
            if (n_rgb_seen - 1) % args.rgb_stride != 0:
                continue
            # VIO 运动过滤: 丢弃相对上一保留帧几乎不动的帧(静止/冗余)
            if args.min_trans > 0 and last_vio is not None and last_kept is not None:
                dp = ((last_vio[0] - last_kept[0]) ** 2 + (last_vio[1] - last_kept[1]) ** 2
                      + (last_vio[2] - last_kept[2]) ** 2) ** 0.5
                dot = abs(sum(last_vio[3 + k] * last_kept[3 + k] for k in range(4)))
                drot = 2.0 * np.degrees(np.arccos(min(1.0, dot)))
                if dp < args.min_trans and drot < args.min_rot:
                    n_dropped += 1
                    continue
            buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR, (H0,W0,3)
            if scale is None:
                long_side = max(img.shape[:2])
                scale = args.img_size / long_side
                Wr = int(round(img.shape[1] * scale))
                Hr = int(round(img.shape[0] * scale))
                print(f"  彩色 {img.shape[1]}x{img.shape[0]} -> {Wr}x{Hr} (scale={scale:.4f})")
            imgr = cv2.resize(img, (Wr, Hr), interpolation=cv2.INTER_AREA)
            idx = len(rgb_ts)
            cv2.imwrite(str(out / f"{idx:06d}.png"), imgr)
            rgb_ts.append(stamp(msg))
            last_kept = last_vio  # 记录本保留帧的 VIO 位姿, 供下一帧比较
            if idx % 500 == 0:
                print(f"  RGB {idx} 帧 (已丢弃静止/冗余 {n_dropped})...")

        elif topic == T_DEP:
            d = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(msg.height, msg.width)
            idx = len(depth_ts)
            cv2.imwrite(str(out / "depth" / f"{idx:06d}.png"), d)
            depth_ts.append(stamp(msg))

    assert K0 is not None, "未读到 color/camera_info"
    assert len(rgb_ts) > 0, "未读到任何 RGB 帧"

    # 时间戳: 帧号 -> 真实时间
    np.savetxt(out / "timestamps.txt", np.c_[np.arange(len(rgb_ts)), rgb_ts],
               fmt=["%d", "%.9f"], header="frame_idx  timestamp_sec")
    # VIO / IMU
    vio = np.array(vio, dtype=np.float64)
    imu = np.array(imu, dtype=np.float64)
    np.savetxt(out / "vio.txt", vio, fmt="%.9f",
               header="timestamp tx ty tz qx qy qz qw  (world 系, 米制)")
    np.savetxt(out / "imu.txt", imu, fmt="%.9f",
               header="timestamp ax ay az gx gy gz  (imu_optical 系)")
    if args.with_depth:
        np.savetxt(out / "depth_timestamps.txt", np.c_[np.arange(len(depth_ts)), depth_ts],
                   fmt=["%d", "%.9f"], header="depth_idx  timestamp_sec  (mono16 毫米, 左红外frame)")

    # 缩放后针孔内参 (rectified, 无畸变)
    fx, fy = K0[0, 0] * scale, K0[1, 1] * scale
    cx, cy = K0[0, 2] * scale, K0[1, 2] * scale
    intr = {"width": Wr, "height": Hr,
            "calibration": [float(fx), float(fy), float(cx), float(cy)]}
    ipath = pathlib.Path("config/intrinsics_insight9.yaml")
    with open(ipath, "w") as f:
        yaml.safe_dump(intr, f, sort_keys=False)
    cpath = pathlib.Path("config/insight9.yaml")
    if not cpath.exists():  # 已存在就不覆盖(保留手工的 subsample 等设置)
        with open(cpath, "w") as f:
            f.write('inherit: "config/calib.yaml"\n'
                    "# insight9 (RealSense wrist_right, 已rectified针孔) 建图配置。内参 intrinsics_insight9.yaml。\n"
                    "# 数据已按 VIO 运动过滤; 快速行走帧间运动大, subsample>1 会跟丢, 故用全部帧。\n"
                    "#   纯RGB: python main.py   |   VIO增强: python main.py --vio datasets/insight9/vio.txt\n"
                    "dataset:\n"
                    "  subsample: 1\n")

    dt = rgb_ts[-1] - rgb_ts[0] if len(rgb_ts) > 1 else 0
    print("\n=== 完成 ===")
    print(f"  RGB 保留 {len(rgb_ts)} 帧 / 共 {n_rgb_seen} 帧 (VIO运动过滤丢弃静止/冗余 {n_dropped}), 时长 {dt:.1f}s -> {out}/*.png")
    print(f"  VIO {len(vio)} 条, IMU {len(imu)} 条" + (f", 深度 {len(depth_ts)} 帧" if args.with_depth else ""))
    print(f"  内参 {ipath} ({Wr}x{Hr}, fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f})")
    print(f"  配置 {cpath}")


if __name__ == "__main__":
    main()
