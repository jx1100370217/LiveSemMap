"""增量产物保存线程: 建图运行中周期性把"导航所需全套产物"写盘。

体现增量建图的核心优势 —— 随时中断(关 viewer 窗口 / Ctrl-C / 甚至 kill -9),
logs/<save_as>/ 里都有截止当时的完整可导航地图:
  轨迹 txt / 关键帧位姿 txt / 占据栅格 npz / BEV png / semantic.json / keyframes 图像。
重产物(稠密 PLY ~860MB、SelaVPR 描述子需加载模型)仍在正常退出时保存;
描述子缺失时 nav_web/export_web.py 会自动补提取, 不影响导航。
"""
import threading
import time
import traceback


class IncrementalSaver:
    def __init__(self, save_dir, seq_name, keyframes, semantic_ann, vio_prior,
                 timestamps, get_conf_threshold, interval=20.0):
        self.save_dir = save_dir
        self.seq_name = seq_name
        self.keyframes = keyframes
        self.semantic_ann = semantic_ann
        self.vio_prior = vio_prior
        self.timestamps = timestamps
        self.get_conf = get_conf_threshold   # callable, 取 viewer 当前置信度阈值
        self.interval = interval
        self._saved_imgs = 0                 # 关键帧图像增量游标
        self._last_state = (0, 0)            # (kf数, 标注数) 无变化则跳过本轮
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()
        print(f"[增量保存] 已启用: 每 {self.interval:.0f}s 刷新导航产物到 {self.save_dir} "
              f"(随时中断都可用于导航)", flush=True)

    def _loop(self):
        while not self._stop.wait(self.interval):
            try:
                self.save_round(verbose=False)
            except Exception:
                traceback.print_exc()

    def save_round(self, verbose=True):
        """保存一轮 (轻产物全量 + 关键帧图像增量)。线程与退出流程都调用。"""
        import mast3r_slam.evaluate as eval

        n = len(self.keyframes)
        n_ann = len(self.semantic_ann) if self.semantic_ann is not None else 0
        if n == 0 or (n, n_ann) == self._last_state:
            return
        t0 = time.time()
        eval.save_traj(self.save_dir, f"{self.seq_name}.txt", self.timestamps, self.keyframes)
        eval.save_keyframe_poses(self.save_dir, f"{self.seq_name}_kf_poses.txt",
                                 self.timestamps, self.keyframes)
        self._saved_imgs = eval.save_keyframes(
            self.save_dir / "keyframes" / self.seq_name, self.timestamps,
            self.keyframes, start=self._saved_imgs)
        eval.save_semantic_map(self.save_dir, self.seq_name, self.keyframes,
                               self.semantic_ann, self.vio_prior,
                               self.get_conf(), verbose=verbose)
        self._last_state = (n, n_ann)
        if verbose:
            print(f"[增量保存] {n} kf 产物已刷新 ({time.time() - t0:.1f}s)", flush=True)

    def reset(self):
        """Restart mapping: 复位增量游标(kf_idx 从 0 复用, 图像需重写)。"""
        self._saved_imgs = 0
        self._last_state = (0, 0)

    def stop(self):
        """退出流程调用: 停周期线程(最终一致性由退出保存的全量流程负责)。"""
        self._stop.set()
        self._thread.join(timeout=30)
