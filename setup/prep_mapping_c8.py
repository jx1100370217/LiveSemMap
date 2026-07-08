"""把 VGP-Nav 的 Mapping_C8 camera_1 (MEI 鱼眼) 去畸变成针孔 .png 序列, 供 MASt3R-SLAM 读取。
输出: datasets/Mapping_C8/frame_%05d.png (natsorted) + 打印针孔内参写 intrinsics_c8.yaml。
在 internvla 环境跑 (有 vgpnav/cv2)。"""
import os, sys
import cv2
import numpy as np

sys.path.insert(0, "/home/jx/codes/VGP-Nav")
from vgpnav.config import Config
from vgpnav.database import list_frame_files
from vgpnav.undistort import PinholeUndistorter, load_camera_params

os.environ.setdefault("VGPNAV_DATASET", "Mapping_C8")
cfg = Config(dataset="Mapping_C8")
files = list_frame_files(cfg)                      # camera_1, 按时间戳排序
params = load_camera_params(cfg.cam_params, cfg.camera)
und = PinholeUndistorter(params, cfg.undist_w, cfg.undist_h,
                         cfg.undist_hfov, cfg.undist_pitch_down)

out_dir = "/home/jx/codes/MASt3R-SLAM/datasets/Mapping_C8"
os.makedirs(out_dir, exist_ok=True)
print(f"C8 camera_1 帧数: {len(files)}; 去畸变到 {cfg.undist_w}x{cfg.undist_h} 针孔 -> {out_dir}")
for i, f in enumerate(files):
    img = und.undistort(cv2.imread(f))
    cv2.imwrite(os.path.join(out_dir, f"frame_{i:05d}.png"), img)
    if i % 100 == 0:
        print(f"  {i}/{len(files)}", flush=True)

K = und.K
fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
yaml_path = "/home/jx/codes/MASt3R-SLAM/config/intrinsics_c8.yaml"
with open(yaml_path, "w") as fp:
    fp.write(f"width: {cfg.undist_w}\nheight: {cfg.undist_h}\n")
    fp.write("# 去畸变后针孔, 无畸变 (fx, fy, cx, cy)\n")
    fp.write(f"calibration: [{fx:.4f}, {fy:.4f}, {cx:.4f}, {cy:.4f}]\n")
print(f"针孔内参 fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f} -> {yaml_path}")
print(f"完成: {len(os.listdir(out_dir))} 张 png")
