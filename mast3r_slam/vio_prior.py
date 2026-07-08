"""方案B: 用 VIO 度量轨迹为 MASt3R-SLAM 跟踪提供运动补偿的位姿初始化 + 在线尺度。

仅在 `main.py --vio <vio.txt>` 时启用。纯 RGB 建图路径完全不经过本模块。

原理: MASt3R 单目对新帧用"常位姿"初始化(=上一帧位姿), 快速行走/转弯时离真值太远,
Gauss-Newton 跟丢。这里改成用 VIO 的帧间相对运动 (旋转 + 按在线尺度缩放的平移) 预测新帧位姿,
让 GN 从接近真值处起步 -> 快速运动不丢、抑制漂移。MASt3R 尺度(metric checkpoint, 但会漂)
与 VIO 米制的比例用最近若干帧的位移比在线中位数估计。
"""
import pathlib
from collections import deque

import lietorch
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp


class VIOPrior:
    def __init__(self, vio_path, subsample, device, min_scale_samples=8):
        vio_path = pathlib.Path(vio_path)
        vio = np.loadtxt(vio_path)  # t tx ty tz qx qy qz qw
        vio = vio[np.argsort(vio[:, 0], kind="stable")]
        _, uq = np.unique(vio[:, 0], return_index=True)  # 时间严格递增
        vio = vio[np.sort(uq)]
        self.vt = vio[:, 0]
        self.vp = vio[:, 1:4]
        self.slerp = Slerp(self.vt, Rotation.from_quat(vio[:, 4:8]))
        ts = np.loadtxt(vio_path.parent / "timestamps.txt")  # kept_idx real_time
        self.real = ts[:, 1]
        self.subsample = subsample
        self.device = device
        self.min_scale_samples = min_scale_samples
        self.scale = None            # MASt3R 单位/米; None=未估计出(此时只用旋转初始化)
        self.scale_ratios = deque(maxlen=60)
        self.window = 15             # 尺度用窗口基线(与~15帧前比), 比相邻帧稳
        self.hist = deque(maxlen=self.window)  # (mast3r中心, vio位置)
        self.kf_pos = None           # 上一个关键帧处的 VIO 位置/姿态 (强制建关键帧用)
        self.kf_rot = None

    def _real_t(self, fid):
        idx = int(min(max(fid, 0) * self.subsample, len(self.real) - 1))
        return float(self.real[idx])

    def _pos(self, t):
        return np.array([np.interp(t, self.vt, self.vp[:, k]) for k in range(3)])

    def _rel(self, fid_prev, fid_cur):
        """VIO 帧间相对运动 cam_prev<-cam_cur: 返回 (四元数xyzw, 平移米)。"""
        t0 = np.clip(self._real_t(fid_prev), self.vt[0], self.vt[-1])
        t1 = np.clip(self._real_t(fid_cur), self.vt[0], self.vt[-1])
        R = self.slerp([t0, t1])
        R0, R1 = R[0], R[1]
        R_rel = R0.inv() * R1
        t_rel = R0.inv().apply(self._pos(t1) - self._pos(t0))
        return R_rel.as_quat(), t_rel

    def _pose_at(self, fid):
        t = np.clip(self._real_t(fid), self.vt[0], self.vt[-1])
        return self._pos(t), self.slerp([t])[0]

    def note_keyframe(self, fid):
        """记录新关键帧处的 VIO 位姿, 作为后续"移动够了没"的基准。"""
        self.kf_pos, self.kf_rot = self._pose_at(fid)

    def moved_enough(self, cur_fid, trans=0.40, rot_deg=12.0):
        """相对上一关键帧, VIO 平移或旋转超阈值 -> 该建新关键帧了(趁重叠还够, 防跟丢)。"""
        if self.kf_pos is None:
            return False
        pos, rot = self._pose_at(cur_fid)
        dp = float(np.linalg.norm(pos - self.kf_pos))
        drot = float(np.degrees((self.kf_rot.inv() * rot).magnitude()))
        return dp > trans or drot > rot_deg

    def predict(self, cur_fid, prev_T_WC):
        """把 VIO 相对运动叠加到上一帧位姿, 得到新帧的位姿初始化 (lietorch.Sim3)。"""
        q, t_rel = self._rel(cur_fid - 1, cur_fid)
        s = self.scale if self.scale is not None else 0.0  # 尺度未知 -> 只补偿旋转
        t = s * t_rel
        data = torch.tensor(
            [t[0], t[1], t[2], q[0], q[1], q[2], q[3], 1.0],
            dtype=prev_T_WC.data.dtype, device=prev_T_WC.data.device,
        ).reshape(1, 8)
        return prev_T_WC * lietorch.Sim3(data)

    def update(self, cur_fid, T_WC):
        """用跟踪后的位姿更新 MASt3R<->VIO 局部尺度: 当前帧与~window帧前的位移比中位数。"""
        center = T_WC.matrix().reshape(-1, 4, 4)[0, :3, 3].detach().cpu().numpy()
        vpos = self._pos(np.clip(self._real_t(cur_fid), self.vt[0], self.vt[-1]))
        if len(self.hist) == self.window:
            c0, v0 = self.hist[0]  # ~window 帧前
            m_disp = float(np.linalg.norm(center - c0))
            v_disp = float(np.linalg.norm(vpos - v0))
            if v_disp > 0.05 and np.isfinite(m_disp) and m_disp > 1e-6:  # 基线>5cm 才算
                self.scale_ratios.append(m_disp / v_disp)
                if len(self.scale_ratios) >= self.min_scale_samples:
                    self.scale = float(np.median(self.scale_ratios))
        self.hist.append((center, vpos))
