import argparse
import datetime
import pathlib
import signal
import sys
import time
import cv2
import lietorch
import torch
import tqdm
import yaml
from mast3r_slam.global_opt import FactorGraph

from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.dataloader import Intrinsics, load_dataset
import mast3r_slam.evaluate as eval
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.visualization import WindowMsg, run_visualization
import torch.multiprocessing as mp


def solve_gn_safe(factor_graph, tag=""):
    """全局优化, OOM 时跳过而非让后端进程崩溃(崩了=后续无回环/全局优化)。
    注意: 不能用 torch.cuda.empty_cache() —— 后端与主进程共享 CUDA IPC 关键帧张量,
    在此进程 empty_cache 会干扰共享张量、破坏 tracking(实测会让跟踪很快丢失)。"""
    try:
        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()
    except torch.cuda.OutOfMemoryError:
        print(f"[backend] 显存不足, 跳过{tag}本次全局优化(后端存活; 最终漂移由 VIO 位姿重建 _vio.ply 修正)")


def relocalization(frame, keyframes, factor_graph, retrieval_database):
    # we are adding and then removing from the keyframe, so we need to be careful.
    # The lock slows viz down but safer this way...
    with keyframes.lock:
        kf_idx = []
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful_loop_closure = False
        if kf_idx:
            keyframes.append(frame)
            n_kf = len(keyframes)
            kf_idx = list(kf_idx)  # convert to list
            frame_idx = [n_kf - 1] * len(kf_idx)
            print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
            try:
                added = factor_graph.add_factors(
                    frame_idx,
                    kf_idx,
                    config["reloc"]["min_match_frac"],
                    is_reloc=config["reloc"]["strict"],
                )
            except torch.cuda.OutOfMemoryError:  # 显存不足时不崩后端, 当作重定位失败
                print("[backend] 显存不足, 跳过本次重定位")
                added = False
            if added:
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print("Success! Relocalized")
                successful_loop_closure = True
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            else:
                keyframes.pop_last()
                print("Failed to relocalize")

        if successful_loop_closure:
            solve_gn_safe(factor_graph, "reloc ")
        return successful_loop_closure


def run_backend(cfg, model, states, keyframes, K):
    set_global_config(cfg)
    # Ctrl-C 发给整个前台进程组: 子进程忽略 SIGINT, 统一由主进程置 TERMINATED 后自然退出
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    device = keyframes.device
    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model)

    last_reset = states.get_reset()
    mode = states.get_mode()
    while mode is not Mode.TERMINATED:
        mode = states.get_mode()
        r = states.get_reset()
        if r != last_reset:  # 重新建图: 重建因子图 + 清空检索库(丢弃旧关键帧)
            factor_graph = FactorGraph(model, keyframes, K, device)
            retrieval_database.reset()
            last_reset = r
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01)
            continue
        if mode == Mode.RELOC:
            frame = states.get_frame()
            success = relocalization(frame, keyframes, factor_graph, retrieval_database)
            if success:
                states.set_mode(Mode.TRACKING)
            states.dequeue_reloc()
            continue
        idx = -1
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01)
            continue

        # Graph Construction
        kf_idx = []
        # k to previous consecutive keyframes
        n_consec = 1
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)
        frame = keyframes[idx]
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=True,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds

        lc_inds = set(retrieval_inds)
        lc_inds.discard(idx - 1)
        if len(lc_inds) > 0:
            print("Database retrieval", idx, ": ", lc_inds)

        # 方案B: viewer 用 VIO 位姿显示无漂移地图, 后端全局优化(加边+GN求解)对显示地图无意义,
        # 跳过它省下后端约7GB显存 —— 否则 VIO+viewer 主/后端双模型推理会撑满显存导致卡顿。
        # 上面的 retrieval_database.update 保留(供重定位); 关键帧位姿由 tracker 维护(相对配准足够)。
        if config.get("vio_path"):
            with states.lock:
                if len(states.global_optimizer_tasks) > 0:
                    states.global_optimizer_tasks.pop(0)
            continue

        kf_idx = set(kf_idx)  # Remove duplicates by using set
        kf_idx.discard(idx)  # Remove current kf idx if included
        kf_idx = list(kf_idx)  # convert to list
        frame_idx = [idx] * len(kf_idx)
        # 加边+全局优化整块 OOM 保护: 任一处爆显存都 empty_cache 跳过, 而非让后端进程崩溃
        try:
            if kf_idx:
                factor_graph.add_factors(
                    kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
                )
            with states.lock:
                states.edges_ii[:] = factor_graph.ii.cpu().tolist()
                states.edges_jj[:] = factor_graph.jj.cpu().tolist()
            solve_gn_safe(factor_graph, f"kf{idx} ")
        except torch.cuda.OutOfMemoryError:
            print(f"[backend] 显存不足, 跳过 kf{idx} 图构建/优化(后端存活)")

        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks.pop(0)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0"
    save_frames = False
    datetime_now = str(datetime.datetime.now()).replace(" ", "_")

    # 默认值取自 nav_config.yaml (配置一次, 三个入口脚本零参数运行); CLI 参数优先
    from mast3r_slam.run_config import load_run_config
    rc = load_run_config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=rc.get("dataset", "datasets/insight9"))
    parser.add_argument("--config", default=rc.get("config", "config/insight9.yaml"))
    parser.add_argument("--save-as", default=rc.get("save_as", "default"))
    parser.add_argument("--no-viz", default=False)
    parser.add_argument("--calib", default=rc.get("calib", "config/intrinsics_insight9.yaml"))
    parser.add_argument("--vio", default=rc.get("vio", ""),
                        help="方案B: VIO 度量轨迹(vio.txt, 同目录需 timestamps.txt), 给跟踪做运动补偿位姿先验; "
                             "不给=纯RGB(行为不变)")
    parser.add_argument("--snapshot-every", type=int, default=0,
                        help="每 N 个关键帧存一次增量点云快照到 logs/<save_as>/snapshots/ (0=关)")
    parser.add_argument("--semantic-api",
                        default=rc.get("semantic_api", "http://192.168.50.72:8299/v1"),
                        help="语义标注 vLLM 服务地址; 空串=关闭语义标注")
    parser.add_argument("--semantic-model", default=rc.get("semantic_model", "qwen3.5-35b-a3b"))
    parser.add_argument("--no-vpr", action="store_true",
                        help="退出时跳过 SelaVPR 描述子提取(默认提取, 供导航重定位)")

    args = parser.parse_args()

    load_config(args.config)
    print(args.dataset)
    print(config)

    manager = mp.Manager()
    main2viz = new_queue(manager, args.no_viz)
    viz2main = new_queue(manager, args.no_viz)

    dataset = load_dataset(args.dataset)
    dataset.subsample(config["dataset"]["subsample"])
    h, w = dataset.get_img_shape()[0]

    if args.calib:
        with open(args.calib, "r") as f:
            intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
        config["use_calib"] = True
        dataset.use_calibration = True
        dataset.camera_intrinsics = Intrinsics.from_calib(
            dataset.img_size,
            intrinsics["width"],
            intrinsics["height"],
            intrinsics["calibration"],
        )

    # 方案B: VIO 位姿先验 (仅 --vio 时启用; 纯RGB路径下 vio_prior 恒为 None, 行为不变)
    vio_prior = None
    if args.vio:
        from mast3r_slam.vio_prior import VIOPrior
        vio_prior = VIOPrior(args.vio, config["dataset"]["subsample"], device)
        config["vio_path"] = args.vio  # 传给 viewer 进程, 让 viewer 用 VIO 位姿渲染(无漂移)
        print(f"[方案B] 已启用 VIO 运动补偿位姿先验: {args.vio}")

    # VIO 强制建关键帧会更密, 需更大缓冲; 纯RGB 保持默认 512 不变
    keyframes = SharedKeyframes(manager, h, w, buffer=(800 if args.vio else 512))
    states = SharedStates(manager, h, w)

    # 语义关键帧标注: 关键帧图像异步送 L40 vLLM 打语义标签, 结果进共享 dict
    # (viewer 读它画 BEV 语义节点; 退出时聚合保存 semantic.json)
    semantic_ann = manager.dict()
    annotator = None
    if args.semantic_api:
        from mast3r_slam.semantic import SemanticAnnotator
        annotator = SemanticAnnotator(args.semantic_api, semantic_ann,
                                      model=args.semantic_model)
        print(f"[semantic] 语义标注已启用: {args.semantic_api} ({args.semantic_model})")

    if not args.no_viz:
        viz = mp.Process(
            target=run_visualization,
            args=(config, states, keyframes, main2viz, viz2main, semantic_ann),
        )
        viz.start()

    model = load_mast3r(device=device)
    model.share_memory()

    has_calib = dataset.has_calib()
    use_calib = config["use_calib"]

    if use_calib and not has_calib:
        print("[Warning] No calibration provided for this dataset!")
        sys.exit(0)
    K = None
    if use_calib:
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
            device, dtype=torch.float32
        )
        keyframes.set_intrinsics(K)

    # remove the trajectory from the previous run
    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        traj_file = save_dir / f"{seq_name}.txt"
        recon_file = save_dir / f"{seq_name}.ply"
        if traj_file.exists():
            traj_file.unlink()
        if recon_file.exists():
            recon_file.unlink()

    tracker = FrameTracker(model, keyframes, device)
    last_msg = WindowMsg()

    backend = mp.Process(target=run_backend, args=(config, model, states, keyframes, K))
    backend.start()

    # 中断安全: Ctrl-C 不再直接抛 KeyboardInterrupt 跳过保存, 而是置 TERMINATED
    # 让主循环正常 break -> 走完整保存流程(点云/轨迹/语义地图/描述子)。
    # 保存进行中(已 TERMINATED)再按: 先警告一次, 又按才强退 —— 防止把
    # "保存中" 误当 "卡死" 一次 Ctrl-C 就丢产物。
    _sigint_n = {"n": 0}

    def _on_sigint(sig, frm):
        if states.get_mode() == Mode.TERMINATED:
            _sigint_n["n"] += 1
            if _sigint_n["n"] == 1:
                print("\n[中断] 正在保存地图产物(见 [save x/7] 进度), 请稍候; "
                      "再按一次 Ctrl-C 才会强制退出并丢失未保存产物", flush=True)
                return
            print("\n[中断] 再次 Ctrl-C, 强制退出(剩余产物不保存)", flush=True)
            sys.exit(1)
        print("\n[中断] 收到 Ctrl-C: 停止建图, 保存已建好的地图产物...", flush=True)
        states.set_mode(Mode.TERMINATED)

    signal.signal(signal.SIGINT, _on_sigint)

    # 增量产物保存: 建图过程中每 20s 把导航所需产物(轨迹/占据栅格/语义/关键帧图像)
    # 落盘 —— 随时中断(关窗口/Ctrl-C/kill -9)都有截止当时的完整可导航地图
    inc_saver = None
    if dataset.save_results:
        from mast3r_slam.incremental_saver import IncrementalSaver
        _sd, _sn = eval.prepare_savedir(args, dataset)
        inc_saver = IncrementalSaver(
            _sd, _sn, keyframes, semantic_ann, vio_prior, dataset.timestamps,
            get_conf_threshold=lambda: last_msg.C_conf_threshold)
        inc_saver.start()

    i = 0
    fps_timer = time.time()

    frames = []

    # 增量快照: 每 args.snapshot_every 个关键帧存一次当前点云 (体现增量建图过程)
    snap_dir = None
    next_snap = args.snapshot_every
    if args.snapshot_every and dataset.save_results:
        _sd, _sn = eval.prepare_savedir(args, dataset)
        snap_dir = _sd / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)

    while True:
        mode = states.get_mode()
        if mode == Mode.TERMINATED:  # SIGINT / viewer 终止 -> 跳出去走保存流程
            break
        msg = try_get_msg(viz2main)
        if msg is not None and getattr(msg, "reset", False):
            # 重新建图: 清空关键帧/状态/跟踪器, 从第 0 帧重来
            # (后端据 states.reset_count 变化重建因子图+检索库; viewer 清纹理/地图缓存)
            states.clear_for_reset()
            with keyframes.lock:
                keyframes.n_size.value = 0
            tracker = FrameTracker(model, keyframes, device)
            if annotator is not None:  # 语义标注同步清零(kf_idx 从 0 复用)
                annotator.reset()
            if inc_saver is not None:
                inc_saver.reset()
            i = 0
            fps_timer = time.time()
            next_snap = args.snapshot_every  # 快照序号也重新开始
            print("[重新建图] 已清空, 从第 0 帧重来")
            continue
        last_msg = msg if msg is not None else last_msg
        if last_msg.is_terminated:
            states.set_mode(Mode.TERMINATED)
            break

        if last_msg.is_paused and not last_msg.next:
            states.pause()
            time.sleep(0.01)
            continue

        if not last_msg.is_paused:
            states.unpause()

        if i == len(dataset):
            states.set_mode(Mode.TERMINATED)
            break

        timestamp, img = dataset[i]
        if save_frames:
            frames.append(img)

        # get frames last camera pose
        T_WC = (
            lietorch.Sim3.Identity(1, device=device)
            if i == 0
            else states.get_frame().T_WC
        )
        if vio_prior is not None and i > 0:  # 方案B: VIO 运动补偿位姿初始化(纯RGB时 vio_prior=None, 跳过)
            T_WC = vio_prior.predict(i, T_WC)
        frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)

        if mode == Mode.INIT:
            # Initialize via mono inference, and encoded features neeed for database
            X_init, C_init = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X_init, C_init)
            keyframes.append(frame)
            if vio_prior is not None:
                vio_prior.note_keyframe(i)
            states.queue_global_optimization(len(keyframes) - 1)
            states.set_mode(Mode.TRACKING)
            states.set_frame(frame)
            i += 1
            continue

        if mode == Mode.TRACKING:
            add_new_kf, match_info, try_reloc = tracker.track(frame)
            if vio_prior is not None:  # 方案B
                vio_prior.update(i, frame.T_WC)  # 更新 MASt3R<->VIO 尺度
                if not try_reloc and vio_prior.moved_enough(i):
                    add_new_kf = True  # VIO 运动够大 -> 趁重叠还够强制建关键帧, 防快速运动跟丢
            if try_reloc:
                states.set_mode(Mode.RELOC)
            states.set_frame(frame)

        elif mode == Mode.RELOC:
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            states.set_frame(frame)
            states.queue_reloc()
            # In single threaded mode, make sure relocalization happen for every frame
            while config["single_thread"]:
                with states.lock:
                    if states.reloc_sem.value == 0:
                        break
                time.sleep(0.01)

        else:
            raise Exception("Invalid mode")

        if add_new_kf:
            keyframes.append(frame)
            if vio_prior is not None:
                vio_prior.note_keyframe(i)
            states.queue_global_optimization(len(keyframes) - 1)
        # 语义标注追赶式提交: 覆盖 INIT/TRACKING/backend重定位 三种关键帧来源
        if annotator is not None:
            annotator.catch_up(keyframes)
            # In single threaded mode, wait for the backend to finish
            while config["single_thread"]:
                with states.lock:
                    if len(states.global_optimizer_tasks) == 0:
                        break
                time.sleep(0.01)
        # log time
        if i % 30 == 0:
            FPS = i / (time.time() - fps_timer)
            print(f"FPS: {FPS}")
        # 增量快照: 当前关键帧数达到阈值就存一次点云
        if snap_dir is not None and len(keyframes) >= next_snap:
            with keyframes.lock:
                nkf = len(keyframes)
                eval.save_reconstruction(
                    snap_dir, f"snap_{nkf:04d}.ply",
                    keyframes, last_msg.C_conf_threshold,
                )
                import numpy as _np
                _ctr = _np.array([keyframes[k].T_WC.data.cpu().numpy().reshape(-1)[:3]
                                  for k in range(nkf)])
                _np.save(snap_dir / f"centers_{nkf:04d}.npy", _ctr)
            print(f"[snapshot] kf={nkf} -> {snap_dir}")
            next_snap += args.snapshot_every
        i += 1

    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        if inc_saver is not None:
            inc_saver.stop()   # 停周期线程, 避免与下面的最终全量保存并发写

        # 每步独立容错: 单步失败打印堆栈后继续存后面的产物, 不再整链断掉
        def _save_step(idx, total, name, fn):
            print(f"[save {idx}/{total}] {name} ...", flush=True)
            try:
                t0 = time.time()
                fn()
                print(f"[save {idx}/{total}] {name} 完成 ({time.time() - t0:.1f}s)", flush=True)
            except Exception:
                import traceback
                traceback.print_exc()
                print(f"[save {idx}/{total}] {name} 失败, 跳过并继续保存其余产物", flush=True)

        n_steps = 7
        print(f"[save] 开始保存全部地图产物到 {save_dir} (共 {n_steps} 步, "
              f"约 1-3 分钟, 请勿关闭终端; 此时按 Ctrl-C 会丢失未保存产物)", flush=True)
        _save_step(1, n_steps, "轨迹", lambda: eval.save_traj(
            save_dir, f"{seq_name}.txt", dataset.timestamps, keyframes))
        _save_step(2, n_steps, "关键帧位姿", lambda: eval.save_keyframe_poses(
            save_dir, f"{seq_name}_kf_poses.txt", dataset.timestamps, keyframes))
        _save_step(3, n_steps, "稠密点云 PLY", lambda: eval.save_reconstruction(
            save_dir, f"{seq_name}.ply", keyframes, last_msg.C_conf_threshold))
        if vio_prior is not None:  # 方案B: 额外存一份 VIO 位姿重建(治漂移)
            _save_step(4, n_steps, "VIO 米制点云", lambda: eval.save_reconstruction_vio(
                save_dir, f"{seq_name}_vio.ply", keyframes, vio_prior,
                last_msg.C_conf_threshold))
        else:
            print(f"[save 4/{n_steps}] 无 VIO, 跳过米制点云", flush=True)
        _save_step(5, n_steps, "关键帧图像", lambda: eval.save_keyframes(
            save_dir / "keyframes" / seq_name, dataset.timestamps, keyframes))
        # 语义地图 + 占据栅格 + VPR 描述子 (中断/正常退出统一走到这里)
        if annotator is not None:
            annotator.drain()
        _save_step(6, n_steps, "语义地图/占据栅格", lambda: eval.save_semantic_map(
            save_dir, seq_name, keyframes, semantic_ann, vio_prior,
            last_msg.C_conf_threshold))
        if not args.no_vpr:
            _save_step(7, n_steps, "SelaVPR 描述子", lambda: eval.save_vpr_descriptors(
                save_dir, seq_name, keyframes))
        print(f"[save] 产物保存流程结束 -> {save_dir}", flush=True)
    if save_frames:
        savedir = pathlib.Path(f"logs/frames/{datetime_now}")
        savedir.mkdir(exist_ok=True, parents=True)
        for i, frame in tqdm.tqdm(enumerate(frames), total=len(frames)):
            frame = (frame * 255).clip(0, 255)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{savedir}/{i}.png", frame)

    print("done")
    backend.join()
    if not args.no_viz:
        viz.join()
