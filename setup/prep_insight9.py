#!/usr/bin/env python
"""从 insight9 机器人采集包 (0709 全量格式: all_images jpg + CSV) 提取建图数据。

原始目录结构 (rosbag 已由采集方全帧率解包, 无需读 mcap):
  <bag>/all_images/front_1/<stamp_ns>.jpg       前视 4K ~6.3Hz, 建图 + VPR 用
  <bag>/all_images/camera_{1..4}/<stamp_ns>.jpg 环视鱼眼 ~19Hz (1前/2右/3后/4左), 语义标注用
  <bag>/insight9_vio_odometry.csv               /insight/vio_100hz 位姿 (world 系米制,
                                                姿态已验证为相机 optical 约定 x右y下z前)
  <bag>/imu.csv                                 /robot/imu_raw
文件名即成像时刻 (ns), 无需另查时间表。

产出到 datasets/<name>/:
  000000.png ...          front_1 全量帧长边缩到 --img-size, 供 MASt3R 建图 (RGBFiles 扁平 glob)
  timestamps.txt          `帧号 真实秒` (front_1 成像时刻) —— 帧号映射回真实时间查 VIO
  vio.txt                 `t tx ty tz qx qy qz qw` (TUM, world 系度量轨迹)
  imu.txt                 `t ax ay az gx gy gz`
  surround/000000_0.jpg   front_1 高清版(长边 --front-sem-size), 供语义标注读文字标识
                          (门牌/公司名/电梯编号等 OCR 需要高分辨率)
  surround/000000_1.jpg ... 000000_4.jpg   每个建图帧就近配对的 4 环视图 (19Hz 下
                          时间偏差 <=26ms), 长边缩到 --surround-size, 供语义标注
并写 config/<name>.yaml (无相机内参 -> 无标定模式, inherit base.yaml)。

用法:
  python setup/prep_insight9.py /home/jx/datas/0709/20260709_085938 --name cfds_floor28
  python setup/prep_insight9.py /home/jx/datas/0709/20260709_090907 --name cfds_floor1
"""
import argparse
import bisect
import csv
import pathlib
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

SURROUND_CAMS = ("camera_1", "camera_2", "camera_3", "camera_4")


def find_bag_dir(path):
    """定位含 all_images/ 的采集包目录: 支持直接给包目录或其外层日期目录。"""
    p = pathlib.Path(path)
    if (p / "all_images").is_dir():
        return p
    subs = [d for d in sorted(p.iterdir()) if (d / "all_images").is_dir()]
    assert len(subs) == 1, f"{p} 下找到 {len(subs)} 个含 all_images 的子目录, 需恰好 1 个"
    return subs[0]


def load_cam_images(bag, cam):
    """某相机的全量图列表 [(成像秒, 路径)] 时间升序 (文件名即 ns 时间戳)。"""
    items = [(int(p.stem) * 1e-9, p) for p in (bag / "all_images" / cam).glob("*.jpg")]
    items.sort()
    assert items, f"all_images/{cam} 为空"
    return items


def load_vio(bag):
    """VIO 轨迹 (header 时刻, 与成像同域): 按时间排序去重。"""
    rows = []
    with open(bag / "insight9_vio_odometry.csv") as f:
        for row in csv.DictReader(f):
            rows.append([float(row["header_stamp_ns"]) * 1e-9,
                         float(row["pos_x"]), float(row["pos_y"]), float(row["pos_z"]),
                         float(row["ori_x"]), float(row["ori_y"]), float(row["ori_z"]),
                         float(row["ori_w"])])
    vio = np.array(rows, dtype=np.float64)
    vio = vio[np.argsort(vio[:, 0], kind="stable")]
    _, uq = np.unique(vio[:, 0], return_index=True)
    return vio[np.sort(uq)]


def resize_long_side(img, target):
    h, w = img.shape[:2]
    s = target / max(h, w)
    if s >= 1.0:
        return img
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                      interpolation=cv2.INTER_AREA)


def imread_reduced(path, target_long):
    """解码时按 2 的幂预降采样到不低于 target_long, 再精确缩放 (4K 图提速明显)。"""
    flags = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2,
             4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}
    img = None
    for r in (8, 4, 2, 1):
        img = cv2.imread(str(path), flags[r])
        if img is not None and max(img.shape[:2]) >= target_long:
            break
    return img


def nearest(items, t):
    """时间升序 [(t, path)] 中取与 t 最近的一项。"""
    i = bisect.bisect_left(items, (t,))
    cands = [j for j in (i - 1, i) if 0 <= j < len(items)]
    return min(cands, key=lambda j: abs(items[j][0] - t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", help="采集包目录 (含 all_images/) 或其外层日期目录")
    ap.add_argument("--name", required=True, help="数据集名, 产出到 datasets/<name>/")
    ap.add_argument("--img-size", type=int, default=512, help="建图帧长边(像素)")
    ap.add_argument("--surround-size", type=int, default=1024,
                    help="环视语义图长边(像素); 5 图一并送 VLM, 注意上下文预算")
    ap.add_argument("--front-sem-size", type=int, default=1920,
                    help="前视高清语义图长边(像素), 供 VLM 读文字标识")
    ap.add_argument("--jobs", type=int, default=8, help="图像缩放并行线程数")
    ap.add_argument("--limit", type=int, default=0, help="调试: 只处理前 N 个 front_1 帧")
    args = ap.parse_args()

    bag = find_bag_dir(args.bag)
    repo = pathlib.Path(__file__).resolve().parent.parent
    out = repo / "datasets" / args.name
    sur_dir = out / "surround"
    out.mkdir(parents=True, exist_ok=True)
    sur_dir.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("*.png")) + list(sur_dir.glob("*.jpg")):  # 清旧结果防残留
        old.unlink()

    fronts = load_cam_images(bag, "front_1")
    if args.limit:
        fronts = fronts[: args.limit]
    cams = {cam: load_cam_images(bag, cam) for cam in SURROUND_CAMS}
    vio = load_vio(bag)
    print(f"读采集包: {bag}\n  front_1 全量 {len(fronts)} 帧, 环视 "
          + "/".join(str(len(cams[c])) for c in SURROUND_CAMS) + " 张")
    kept = fronts

    # 并行缩放落盘: 每个保留帧 = 1 张建图 png + 1 张前视高清语义图 + 4 张环视 jpg
    def process(job):
        idx, (t, fpath) = job
        img = imread_reduced(fpath, args.front_sem_size)  # 一次解码, 建图/语义共用
        assert img is not None, f"读取失败: {fpath}"
        cv2.imwrite(str(out / f"{idx:06d}.png"), resize_long_side(img, args.img_size))
        n = 0
        cv2.imwrite(str(sur_dir / f"{idx:06d}_0.jpg"),
                    resize_long_side(img, args.front_sem_size),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        n += 1
        for k, cam in enumerate(SURROUND_CAMS, start=1):
            st, spath = cams[cam][nearest(cams[cam], t)]
            simg = imread_reduced(spath, args.surround_size)
            if simg is None:
                continue
            cv2.imwrite(str(sur_dir / f"{idx:06d}_{k}.jpg"),
                        resize_long_side(simg, args.surround_size),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            n += 1
        return n

    n_sur = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for i, n in enumerate(ex.map(process, enumerate(kept))):
            n_sur += n
            if i % 200 == 0:
                print(f"  帧 {i}/{len(kept)} ...")

    rgb_ts = [t for t, _ in kept]
    np.savetxt(out / "timestamps.txt", np.c_[np.arange(len(rgb_ts)), rgb_ts],
               fmt=["%d", "%.9f"], header="frame_idx  timestamp_sec (front_1 成像时刻)")
    np.savetxt(out / "vio.txt", vio, fmt="%.9f",
               header="timestamp tx ty tz qx qy qz qw  (world 系米制, 姿态=相机光学约定)")

    imu = []
    with open(bag / "imu.csv") as f:
        for row in csv.DictReader(f):
            imu.append([float(row["timestamp_sec"]),
                        float(row["lin_x"]), float(row["lin_y"]), float(row["lin_z"]),
                        float(row["ang_x"]), float(row["ang_y"]), float(row["ang_z"])])
    np.savetxt(out / "imu.txt", np.array(imu, dtype=np.float64), fmt="%.9f",
               header="timestamp ax ay az gx gy gz")

    cpath = repo / "config" / f"{args.name}.yaml"
    with open(cpath, "w") as f:
        f.write('inherit: "config/base.yaml"\n'
                f"# {args.name} (insight9 front_1 全量 ~6.3Hz, 无相机内参 -> 无标定模式)。\n"
                "# 全量帧建图(含静止段):\n"
                f"#   python main.py --vio datasets/{args.name}/vio.txt\n"
                "dataset:\n"
                "  subsample: 1\n")

    dt = rgb_ts[-1] - rgb_ts[0] if len(rgb_ts) > 1 else 0
    print("\n=== 完成 ===")
    print(f"  建图帧 {len(rgb_ts)}, 时长 {dt:.1f}s -> {out}/*.png")
    print(f"  环视语义图 {n_sur} 张 -> {sur_dir}/")
    print(f"  VIO {len(vio)} 条, IMU {len(imu)} 条")
    print(f"  配置 {cpath} (无标定)")
    print(f"\n下一步: 编辑 nav_config.yaml 指向 datasets/{args.name} 后运行 python main.py")


if __name__ == "__main__":
    main()
