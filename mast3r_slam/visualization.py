import dataclasses
import time as _time
import weakref
from pathlib import Path

import imgui
import lietorch
import torch
import moderngl
import moderngl_window as mglw
import numpy as np
from in3d.camera import Camera, ProjectionMatrix, lookat
from in3d.pose_utils import translation_matrix
from in3d.color import hex2rgba
from in3d.geometry import Axis
from in3d.viewport_window import ViewportWindow
from in3d.window import WindowEvents
from in3d.image import Image
from moderngl_window import resources
from moderngl_window.timers.clock import Timer

from mast3r_slam.frame import Mode
from mast3r_slam.geometry import get_pixel_coords
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.visualization_utils import (
    Frustums,
    Lines,
    depth2rgb,
    image_with_text,
)
from mast3r_slam.config import load_config, config, set_global_config


@dataclasses.dataclass
class WindowMsg:
    is_terminated: bool = False
    is_paused: bool = False
    next: bool = False
    C_conf_threshold: float = 1.5
    reset: bool = False  # 重新建图 (viewer 按钮触发, 一次性)


class Window(WindowEvents):
    title = "LiveSemMap"
    window_size = (3840, 2160)

    def __init__(self, states, keyframes, main2viz, viz2main, semantic_ann=None,
                 hmsg_live=None, **kwargs):
        super().__init__(**kwargs)
        # 语义关键帧标注 (主进程语义线程写入的 Manager.dict: kf_idx -> annotation)
        self.semantic_ann = semantic_ann
        self.hmsg_live = hmsg_live       # OnlineHMSG 房间区域快照 (Manager dict)
        self.ctx.gc_mode = "auto"
        # bit hacky, but detect whether user is using 4k monitor
        self.scale = 1.0
        if self.wnd.buffer_size[0] > 2560:
            self.set_font_scale(2.0)
            self.scale = 2
        self.clear = hex2rgba("#1E2326", alpha=1)
        resources.register_dir((Path(__file__).parent.parent / "resources").resolve())

        self.line_prog = self.load_program("programs/lines.glsl")
        self.surfelmap_prog = self.load_program("programs/surfelmap.glsl")
        self.trianglemap_prog = self.load_program("programs/trianglemap.glsl")
        self.pointmap_prog = self.surfelmap_prog

        width, height = self.wnd.size
        self.camera = Camera(
            ProjectionMatrix(width, height, 60, width // 2, height // 2, 0.05, 100),
            lookat(np.array([2, 2, 2]), np.array([0, 0, 0]), np.array([0, 1, 0])),
        )
        self.axis = Axis(self.line_prog, 0.1, 3 * self.scale)
        self.frustums = Frustums(self.line_prog)
        self.lines = Lines(self.line_prog)

        self.viewport = ViewportWindow("Scene", self.camera)
        self.state = WindowMsg()
        self.keyframes = keyframes
        self.states = states

        self.show_all = True
        self.show_keyframe_edges = True
        self.culling = True
        self.follow_cam = True

        self.depth_bias = 0.001
        self.frustum_scale = 0.05

        self.dP_dz = None

        self.line_thickness = 3
        self.show_keyframe = True
        self.show_curr_pointmap = True
        self.show_axis = True

        self.textures = dict()
        self.mtime = self.pointmap_prog.extra["meta"].resolved_path.stat().st_mtime
        self.curr_img, self.kf_img = Image(), Image()
        self.curr_img_np, self.kf_img_np = None, None

        # Map View 面板: 只显示 VLM 区域生长语义地图 (SLAM 进程 engine 渲染好
        # 的 RGB 图经 hmsg_live 推来, viewer 零计算零几何 —— BEV/占据图已退役)
        self.map_img = Image()
        self.map_img.write(np.zeros((8, 8, 3), np.float32))
        self._map_v = -1                 # live 地图版本号 (变了才重传纹理)
        self._last_reset = 0  # 重新建图: 检测到 states.reset_count 变化就清纹理/地图缓存(frame_id 会复用)
        # VIO 位姿渲染(仅 --vio): viewer 用 VIO 位姿画每个关键帧 -> 显示无漂移地图(单目位姿会漂)
        self.vio = None
        self._vio_cache = {}  # frame_id -> (pos, quat_xyzw)
        self._vio_scale = None  # MASt3R单位->米 全局尺度(用于当前帧)
        self._vio_render_cache = None  # (N, render_data): VIO模式老关键帧位姿固定, 按N缓存避免每帧重算
        _vio_path = config.get("vio_path")
        if _vio_path:
            from mast3r_slam.vio_prior import VIOPrior
            self.vio = VIOPrior(_vio_path, config["dataset"]["subsample"], "cpu")

        self.main2viz = main2viz
        self.viz2main = viz2main

    def _vio_pose_data(self, fids, mast_centers):
        """给关键帧 frame_id 返回 VIO 位姿的 Sim3 data (N,1,8): [VIO平移, VIO四元数, 尺度s]。
        s=MASt3R单位->米(相邻关键帧位移比中位数)。用它渲染=live无漂移。缓存每帧VIO位姿。"""
        for f in fids:
            f = int(f)
            if f not in self._vio_cache:
                p, R = self.vio._pose_at(f)
                self._vio_cache[f] = (p.astype(np.float32), R.as_quat().astype(np.float32))
        vp = np.stack([self._vio_cache[int(f)][0] for f in fids])
        vq = np.stack([self._vio_cache[int(f)][1] for f in fids])
        dm = np.linalg.norm(np.diff(mast_centers, axis=0), axis=1)
        dv = np.linalg.norm(np.diff(vp, axis=0), axis=1)
        good = (dm > 1e-4) & (dv > 0.02)
        if good.any():
            self._vio_scale = float(np.median(dv[good] / dm[good]))
        s = self._vio_scale if self._vio_scale is not None else 1.0
        data = np.concatenate([vp, vq, np.full((len(fids), 1), s, np.float32)], axis=1)
        return torch.from_numpy(data.reshape(len(fids), 1, 8))

    @torch.no_grad()  # viewer 不反传, 跳过 autograd 建图 —— lietorch 位姿运算大量, 建图会拖慢渲染/饿死主进程
    def render(self, t: float, frametime: float):
        # 渲染节流 + 无条件让出GPU给主进程tracking: VIO位姿把地图压到25×25m→点云重叠overdraw重,
        # 渲染吃满GPU会饿死同卡上的主进程SLAM推理(建图爬行)。故每帧强制让出 min_yield 秒,
        # 即使渲染帧本身很慢也让出 —— 看建图 ~8fps 足够, 主进程能拿到GPU时间稳定建图。
        min_dt = 1.0 / 6.0
        min_yield = 0.08
        if hasattr(self, "_last_render_t"):
            _dt = _time.time() - self._last_render_t
            _time.sleep(max(min_dt - _dt, min_yield))
        else:
            _time.sleep(min_yield)
        self._last_render_t = _time.time()
        r = self.states.get_reset()
        if r != self._last_reset:  # 重新建图: 清纹理/地图缓存(frame_id 会从0复用, 否则显示旧帧)
            self.textures.clear()
            self._vio_cache.clear()
            self._vio_render_cache = None
            self._map_v = -1
            self._last_reset = r
        self.viewport.use()
        self.ctx.enable(moderngl.DEPTH_TEST)
        if self.culling:
            self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.clear(*self.clear)

        self.ctx.point_size = 2
        if self.show_axis:
            self.axis.render(self.camera)

        curr_frame = self.states.get_frame()
        h, w = curr_frame.img_shape.flatten()
        self.frustums.make_frustum(h, w)

        self.curr_img_np = curr_frame.uimg.numpy()
        self.curr_img.write(self.curr_img_np)

        cam_T_WC = as_SE3(curr_frame.T_WC).cpu()
        if self.vio is not None and self._vio_scale is not None:
            try:  # 当前帧也用 VIO 位姿, 与 VIO 关键帧地图对齐(follow_cam/绿视锥不错位)
                p, R = self.vio._pose_at(int(curr_frame.frame_id))
                data = torch.tensor([*p, *R.as_quat(), self._vio_scale], dtype=torch.float32).reshape(1, 8)
                cam_T_WC = as_SE3(lietorch.Sim3(data)).cpu()
            except Exception:
                pass
        if self.follow_cam:
            T_WC = cam_T_WC.matrix().numpy().astype(
                dtype=np.float32
            ) @ translation_matrix(np.array([0, 0, -2], dtype=np.float32))
            self.camera.follow_cam(np.linalg.inv(T_WC))
        else:
            self.camera.unfollow_cam()
        self.frustums.add(
            cam_T_WC,
            scale=self.frustum_scale,
            color=[0, 1, 0, 1],
            thickness=self.line_thickness * self.scale,
        )

        with self.keyframes.lock:
            N_keyframes = len(self.keyframes)
            dirty_idx = self.keyframes.get_dirty_idx()

        for kf_idx in dirty_idx:
            keyframe = self.keyframes[kf_idx]
            h, w = keyframe.img_shape.flatten()
            X = self.frame_X(keyframe)
            C = keyframe.get_average_conf().cpu().numpy().astype(np.float32)

            if keyframe.frame_id not in self.textures:
                ptex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                ctex = self.ctx.texture((w, h), 1, dtype="f4", alignment=4)
                itex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                self.textures[keyframe.frame_id] = ptex, ctex, itex
                ptex, ctex, itex = self.textures[keyframe.frame_id]
                itex.write(keyframe.uimg.numpy().astype(np.float32).tobytes())

            ptex, ctex, itex = self.textures[keyframe.frame_id]
            ptex.write(X.tobytes())
            ctex.write(C.tobytes())

        # 批量取全部关键帧位姿/frame_id/shape 到 CPU (一次同步), 取代循环里逐帧 keyframe.T_WC.cpu()
        # —— 大量关键帧时逐帧 GPU->CPU 同步是卡顿/窗口"无响应"的主因
        if N_keyframes > 0:
            with self.keyframes.lock:
                # 锁内只做快速 GPU->GPU clone; .cpu()(GPU 同步)放锁外, 否则 GPU 忙时会长时间占锁、阻塞主循环写关键帧
                T_WC_gpu = self.keyframes.T_WC[:N_keyframes].clone()
                fids_gpu = self.keyframes.dataset_idx[:N_keyframes].clone()
                shapes_gpu = self.keyframes.img_shape[:N_keyframes].clone()
                kf_uimg = self.keyframes.uimg[N_keyframes - 1].clone()
            T_WC_cpu = T_WC_gpu.cpu()
            fids = fids_gpu.cpu().numpy().reshape(-1)
            shapes = shapes_gpu.cpu().numpy().reshape(N_keyframes, -1)
            self.kf_img_np = kf_uimg.numpy()
            self.kf_img.write(self.kf_img_np)

            # VIO 位姿渲染: 用 VIO 位姿替换漂移的 MASt3R 位姿(仅 --vio) -> live 显示无漂移地图
            # 按关键帧数 N 缓存: VIO模式下后端不优化, 老关键帧位姿固定, 只在新增关键帧时重算(否则每帧全量算会饿死主进程)
            render_data = T_WC_cpu
            if self.vio is not None:
                if self._vio_render_cache is None or self._vio_render_cache[0] != N_keyframes:
                    mast_c = lietorch.Sim3(T_WC_cpu.reshape(-1, 8)).matrix()[:, :3, 3].numpy()
                    self._vio_render_cache = (N_keyframes, self._vio_pose_data(fids, mast_c))
                render_data = self._vio_render_cache[1]

            # 高关键帧数时抽稀渲染: 每帧渲染全部关键帧的点云/视锥是 O(N), 关键帧多时吃满GPU/CPU→
            # 三方(主tracking/后端全局优化/viz)抢GPU→饿死主进程→建图卡住。抽稀到最多~120个,
            # 末尾3个必渲(当前建图区域)。只影响3D显示疏密, 不影响建图; BEV(后台线程)仍全量。
            step = max(1, (N_keyframes + 99) // 100)  # 向上取整到~100: 每帧最多渲~100个关键帧
            for kf_idx in range(N_keyframes):
                if step > 1 and (kf_idx % step) != 0 and kf_idx < N_keyframes - 3:
                    continue
                fid = int(fids[kf_idx])
                T = lietorch.Sim3(render_data[kf_idx])  # CPU 上构造, 不触发 GPU 同步
                if self.show_keyframe:
                    self.frustums.add(
                        as_SE3(T),
                        scale=self.frustum_scale,
                        color=[1, 0, 0, 1],
                        thickness=self.line_thickness * self.scale,
                    )
                if self.show_all and fid in self.textures:
                    h, w = int(shapes[kf_idx][0]), int(shapes[kf_idx][1])
                    ptex, ctex, itex = self.textures[fid]
                    self.render_pointmap(T, w, h, ptex, ctex, itex)

        if self.show_keyframe_edges:
            with self.states.lock:
                ii = torch.tensor(self.states.edges_ii, dtype=torch.long)
                jj = torch.tensor(self.states.edges_jj, dtype=torch.long)
                if ii.numel() > 0 and jj.numel() > 0:
                    T_WCi = lietorch.Sim3(self.keyframes.T_WC[ii, 0])
                    T_WCj = lietorch.Sim3(self.keyframes.T_WC[jj, 0])
            if ii.numel() > 0 and jj.numel() > 0:
                t_WCi = T_WCi.matrix()[:, :3, 3].cpu().numpy()
                t_WCj = T_WCj.matrix()[:, :3, 3].cpu().numpy()
                self.lines.add(
                    t_WCi,
                    t_WCj,
                    thickness=self.line_thickness * self.scale,
                    color=[0, 1, 0, 1],
                )
        if self.show_curr_pointmap and self.states.get_mode() != Mode.INIT:
            if config["use_calib"]:
                curr_frame.K = self.keyframes.get_intrinsics()
            h, w = curr_frame.img_shape.flatten()
            X = self.frame_X(curr_frame)
            C = curr_frame.C.cpu().numpy().astype(np.float32)
            if "curr" not in self.textures:
                ptex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                ctex = self.ctx.texture((w, h), 1, dtype="f4", alignment=4)
                itex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                self.textures["curr"] = ptex, ctex, itex
            ptex, ctex, itex = self.textures["curr"]
            ptex.write(X.tobytes())
            ctex.write(C.tobytes())
            itex.write(depth2rgb(X[..., -1], colormap="turbo"))
            self.render_pointmap(
                curr_frame.T_WC.cpu(),
                w,
                h,
                ptex,
                ctex,
                itex,
                use_img=True,
                depth_bias=self.depth_bias,
            )

        self.lines.render(self.camera)
        self.frustums.render(self.camera)
        self._upload_map()
        self.render_ui()

    def render_ui(self):
        self.wnd.use()
        imgui.new_frame()

        io = imgui.get_io()
        # get window size and full screen
        window_size = io.display_size
        imgui.set_next_window_size(window_size[0], window_size[1])
        imgui.set_next_window_position(0, 0)
        self.viewport.render()

        # 面板每帧按当前窗口尺寸重新布局(不用 FIRST_USE_EVER, 否则会锁定/恢复旧分辨率布局),
        # 禁滚动条; 任意分辨率都完整自适应、不溢出。
        margin = 12 * self.scale
        gui_w = window_size[0] * 0.24
        imgui.set_next_window_size(gui_w, window_size[1] - 2 * margin)
        imgui.set_next_window_position(margin, margin)
        imgui.set_next_window_focus()
        imgui.begin("GUI", flags=imgui.WINDOW_NO_SCROLLBAR)
        new_state = WindowMsg()
        _, new_state.is_paused = imgui.checkbox("pause", self.state.is_paused)
        imgui.same_line()
        if imgui.button("Restart mapping"):  # 重新建图: 从第0帧清空重来
            self.viz2main.put(WindowMsg(reset=True))

        imgui.spacing()
        _, new_state.C_conf_threshold = imgui.slider_float(
            "C_conf_threshold", self.state.C_conf_threshold, 0, 10
        )

        imgui.spacing()

        _, self.show_all = imgui.checkbox("show all", self.show_all)
        imgui.same_line()
        _, self.follow_cam = imgui.checkbox("follow cam", self.follow_cam)

        imgui.spacing()
        shader_options = [
            "surfelmap.glsl",
            "trianglemap.glsl",
        ]
        current_shader = shader_options.index(
            self.pointmap_prog.extra["meta"].resolved_path.name
        )

        for i, shader in enumerate(shader_options):
            if imgui.radio_button(shader, current_shader == i):
                current_shader = i

        selected_shader = shader_options[current_shader]
        if selected_shader != self.pointmap_prog.extra["meta"].resolved_path.name:
            self.pointmap_prog = self.load_program(f"programs/{selected_shader}")

        imgui.spacing()

        _, self.show_keyframe_edges = imgui.checkbox(
            "show_keyframe_edges", self.show_keyframe_edges
        )
        imgui.spacing()

        _, self.pointmap_prog["show_normal"].value = imgui.checkbox(
            "show_normal", self.pointmap_prog["show_normal"].value
        )
        imgui.same_line()
        _, self.culling = imgui.checkbox("culling", self.culling)
        if "radius" in self.pointmap_prog:
            _, self.pointmap_prog["radius"].value = imgui.drag_float(
                "radius",
                self.pointmap_prog["radius"].value,
                0.0001,
                min_value=0.0,
                max_value=0.1,
            )
        if "slant_threshold" in self.pointmap_prog:
            _, self.pointmap_prog["slant_threshold"].value = imgui.drag_float(
                "slant_threshold",
                self.pointmap_prog["slant_threshold"].value,
                0.1,
                min_value=0.0,
                max_value=1.0,
            )
        _, self.show_keyframe = imgui.checkbox("show_keyframe", self.show_keyframe)
        _, self.show_curr_pointmap = imgui.checkbox(
            "show_curr_pointmap", self.show_curr_pointmap
        )
        _, self.show_axis = imgui.checkbox("show_axis", self.show_axis)
        _, self.line_thickness = imgui.drag_float(
            "line_thickness", self.line_thickness, 0.1, 10, 0.5
        )

        _, self.frustum_scale = imgui.drag_float(
            "frustum_scale", self.frustum_scale, 0.001, 0, 0.1
        )

        imgui.spacing()

        # kf/curr 两张图: 保持比例塞进面板剩余空间(宽 与 剩余半高的较小约束), 不溢出面板 -> 无需滚动条
        gui_avail = imgui.get_content_region_available()
        tw, th = self.curr_img.texture.size
        gap = 30.0 * self.scale  # 两个标题文字高度余量
        img_scale = min(gui_avail[0] / tw, max(1.0, gui_avail[1] - gap) / 2.0 / th)
        size = (tw * img_scale, th * img_scale)
        image_with_text(self.kf_img, size, "kf", same_line=False)
        image_with_text(self.curr_img, size, "curr", same_line=False)

        imgui.end()

        # 语义地图面板 (右侧一整列): VLM 区域生长的在线语义地图, 与导航 Web 同观感
        panel_w = window_size[0] * 0.30
        imgui.set_next_window_size(panel_w, window_size[1] - 2 * margin)
        imgui.set_next_window_position(window_size[0] - panel_w - margin, margin)
        imgui.begin("Map View", flags=imgui.WINDOW_NO_SCROLLBAR)
        if self.semantic_ann is not None:
            imgui.text_ansi(f"semantic: {len(self.semantic_ann)} kf annotated")
        avail = imgui.get_content_region_available()
        side = max(16.0, min(avail[0], avail[1] - 30.0 * self.scale))
        image_with_text(self.map_img, (side, side),
                        "semantic regions (live)", same_line=False)
        imgui.end()

        if new_state != self.state:
            self.state = new_state
            self.send_msg()

        imgui.render()
        self.imgui.render(imgui.get_draw_data())

    def send_msg(self):
        self.viz2main.put(self.state)

    def render_pointmap(self, T_WC, w, h, ptex, ctex, itex, use_img=True, depth_bias=0):
        w, h = int(w), int(h)
        ptex.use(0)
        ctex.use(1)
        itex.use(2)
        model = T_WC.matrix().numpy().astype(np.float32).T

        # 复用 VAO, 避免每关键帧每帧 create/release (大量关键帧时是显著的驱动开销)
        if getattr(self, "_pm_vao", None) is None or self._pm_vao_prog is not self.pointmap_prog:
            self._pm_vao = self.ctx.vertex_array(self.pointmap_prog, [], skip_errors=True)
            self._pm_vao_prog = self.pointmap_prog
        vao = self._pm_vao
        vao.program["m_camera"].write(self.camera.gl_matrix())
        vao.program["m_model"].write(model)
        vao.program["m_proj"].write(self.camera.proj_mat.gl_matrix())

        vao.program["pointmap"].value = 0
        vao.program["confs"].value = 1
        vao.program["img"].value = 2
        vao.program["width"].value = w
        vao.program["height"].value = h
        vao.program["conf_threshold"] = self.state.C_conf_threshold
        vao.program["use_img"] = use_img
        if "depth_bias" in self.pointmap_prog:
            vao.program["depth_bias"] = depth_bias
        vao.render(mode=moderngl.POINTS, vertices=w * h)

    def frame_X(self, frame):
        if config["use_calib"]:
            Xs = frame.X_canon[None]
            if self.dP_dz is None:
                device = Xs.device
                dtype = Xs.dtype
                img_size = frame.img_shape.flatten()[:2]
                K = frame.K
                p = get_pixel_coords(
                    Xs.shape[0], img_size, device=device, dtype=dtype
                ).view(*Xs.shape[:-1], 2)
                tmp1 = (p[..., 0] - K[0, 2]) / K[0, 0]
                tmp2 = (p[..., 1] - K[1, 2]) / K[1, 1]
                self.dP_dz = torch.empty(
                    p.shape[:-1] + (3, 1), device=device, dtype=dtype
                )
                self.dP_dz[..., 0, 0] = tmp1
                self.dP_dz[..., 1, 0] = tmp2
                self.dP_dz[..., 2, 0] = 1.0
                self.dP_dz = self.dP_dz[..., 0].cpu().numpy().astype(np.float32)
            return (Xs[..., 2:3].cpu().numpy().astype(np.float32) * self.dP_dz)[0]

        return frame.X_canon.cpu().numpy().astype(np.float32)

    def _upload_map(self):
        """渲染线程: live 语义地图版本变化时上传纹理 (engine 已渲染好, 仅贴图)。"""
        if self.hmsg_live is None:
            return
        try:
            v = self.hmsg_live.get("map_v", 0)
            if v == self._map_v:
                return
            m = self.hmsg_live.get("map")
            if m is None:
                return
            self._map_v = v
            self.map_img.write(np.ascontiguousarray(m, np.float32) / 255.0)
        except Exception:
            pass


def run_visualization(cfg, states, keyframes, main2viz, viz2main, semantic_ann=None, hmsg_live=None) -> None:
    set_global_config(cfg)
    # Ctrl-C 属于主进程的"中断并保存"流程; viewer 忽略 SIGINT, 由窗口关闭/TERMINATED 退出
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    config_cls = Window
    backend = "glfw"
    window_cls = mglw.get_local_window_cls(backend)

    window = window_cls(
        title=config_cls.title,
        size=config_cls.window_size,
        fullscreen=False,
        resizable=True,
        visible=True,
        gl_version=(3, 3),
        aspect_ratio=None,
        vsync=True,
        samples=4,
        cursor=True,
        backend=backend,
    )
    window.print_context_info()
    mglw.activate_context(window=window)
    window.ctx.gc_mode = "auto"
    timer = Timer()
    window_config = config_cls(
        states=states,
        hmsg_live=hmsg_live,
        keyframes=keyframes,
        main2viz=main2viz,
        viz2main=viz2main,
        semantic_ann=semantic_ann,
        ctx=window.ctx,
        wnd=window,
        timer=timer,
    )
    # Avoid the event assigning in the property setter for now
    # We want the even assigning to happen in WindowConfig.__init__
    # so users are free to assign them in their own __init__.
    window._config = weakref.ref(window_config)

    # Swap buffers once before staring the main loop.
    # This can trigged additional resize events reporting
    # a more accurate buffer size
    window.swap_buffers()
    window.set_default_viewport()

    timer.start()

    while not window.is_closing:
        current_time, delta = timer.next_frame()

        if window_config.clear_color is not None:
            window.clear(*window_config.clear_color)

        # Always bind the window framebuffer before calling render
        window.use()

        window.render(current_time, delta)
        if not window.is_closing:
            window.swap_buffers()

    state = window_config.state
    window.destroy()
    state.is_terminated = True
    viz2main.put(state)
