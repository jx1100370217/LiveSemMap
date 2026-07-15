#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoTIS 推理引擎 —— LiveSemMap 融合版(自包含, 仅推理, 不训练)。

- DinoV3TimmAdapter : timm + 本地 dinov3 safetensors, 绕开 Meta 门控权重。
- build_localizer   : 构建 TrajectoryLocalizer(checkpoint weights_only 安全加载)。
- LotisEngine       : 持有 localizer + 段编码缓存({seq}_lotis_traj.pkl),
                      提供 encode_images / point(query, seg_key)。

远程 LiveSemMap 的建图帧是**前向针孔图**(等价 front_1) -> 不做鱼眼/柱面去畸变,
仅 square_crop(中心方裁)后编码/查询, 与本地 POC 的 do_crop 路径一致。

坐标约定: LoTIS coords(row,col)∈[-1,1] 的参考系 = 喂进模型的 224 方裁帧。
打点结果回传 center_pct(裁剪帧, 0~1); 前端按 crop 参数(left,top,s)映射回全图/缩略图:
    x_full = left + center_pct.x * s ; y_full = top + center_pct.y * s
"""
import os
import sys
import pickle
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
_LOTIS_DIR = os.path.join(_PROJ, "third_party", "lotis")
if _LOTIS_DIR not in sys.path:
    sys.path.insert(0, _LOTIS_DIR)

DEFAULT_CKPT = os.path.join(_PROJ, "pretrained/lotis/final_model.pth")
DEFAULT_CONFIG = os.path.join(_PROJ, "pretrained/lotis/final_config.yaml")
DEFAULT_DINOV3 = os.path.join(_PROJ, "pretrained/dinov3_vitb16.safetensors")


# --------------------------------------------------------------------------- #
# DINOv3 特征提取 adapter(timm + safetensors)
# --------------------------------------------------------------------------- #
class DinoV3TimmAdapter(torch.nn.Module):
    """暴露 LoTIS 需要的 forward_features(x)["x_norm_patchtokens"] -> [B,196,768]。
    timm forward_features 返回 [B, num_prefix+196, C](DINOv3 ViT-B = 1 CLS+4 register=5),
    切掉 prefix 即 patch tokens。"""

    def __init__(self, weights_path: str):
        super().__init__()
        import timm
        from safetensors.torch import load_file

        model = timm.create_model("vit_base_patch16_dinov3", pretrained=False, num_classes=0)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"DINOv3 权重不存在: {weights_path}")
        sd = load_file(weights_path)
        sd = {k: v for k, v in sd.items() if not k.startswith("head.")}
        missing, _ = model.load_state_dict(sd, strict=False)
        bad = [k for k in missing if not k.startswith("head.")]
        if bad:
            logger.warning(f"[DINOv3] 缺失非 head 权重 {bad[:5]} ...")
        self.model = model.eval()
        self.num_prefix = int(getattr(model, "num_prefix_tokens", 5))

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> dict:
        feats = self.model.forward_features(x)
        patch = feats[:, self.num_prefix:, :].contiguous()
        return {"x_norm_patchtokens": patch}


# --------------------------------------------------------------------------- #
# 构建 localizer
# --------------------------------------------------------------------------- #
def build_localizer(checkpoint: str = DEFAULT_CKPT, config: str = DEFAULT_CONFIG,
                    dinov3_weights: str = DEFAULT_DINOV3, device: str = "cuda"):
    from lotis.config import load_config
    from lotis.model import TrajectoryLocalizationModel
    from lotis.localizer import TrajectoryLocalizer

    cfg = load_config(config)
    model = TrajectoryLocalizationModel(
        feature_dim=768,
        input_patches=(14, 14),
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_heads,
        num_blocks=cfg.num_blocks,
        head_depth=cfg.head_depth,
        dropout=0.0,
        attention_dropout=0.0,
        droppath=0.0,
        max_seq_len=cfg.max_seq_len,
        output_size=(32, 32),
        full_global_attention=True,
        rope_freq_seq=cfg.rope_freq_seq,
        rope_freq_spat=cfg.rope_freq_spat,
        heads=cfg.prediction_heads,
        use_nested_tensor=False,
        compile=False,
        mini_batch_size=cfg.mini_batch_size,
        layernorm_type=cfg.layernorm_type,
    )
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    state_dict = {k.replace("._orig_mod", ""): v for k, v in ckpt.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.info(f"[LoTIS] 缺失权重 {len(missing)} 个: {missing[:5]}")
    if unexpected:
        logger.info(f"[LoTIS] 多余权重 {len(unexpected)} 个: {unexpected[:5]}")

    return TrajectoryLocalizer(
        model=model,
        feature_extractor=DinoV3TimmAdapter(dinov3_weights),
        device=device,
        max_seq_len=cfg.max_seq_len,
    )


def square_crop(img: Image.Image) -> Image.Image:
    """中心正方形裁剪。编码(记忆段)与 query 必须同裁, 坐标系才一致。"""
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def crop_params(w: int, h: int):
    """中心方裁参数 (left, top, s), 供把 coords∈[-1,1] 映射回原全图像素。"""
    s = min(w, h)
    return (w - s) // 2, (h - s) // 2, s


# --------------------------------------------------------------------------- #
# 引擎
# --------------------------------------------------------------------------- #
class LotisEngine:
    def __init__(self, checkpoint: str = DEFAULT_CKPT, config: str = DEFAULT_CONFIG,
                 dinov3_weights: str = DEFAULT_DINOV3, traj_cache: Optional[str] = None,
                 device: Optional[str] = None, vis_threshold: float = 0.5, n_min: int = 3,
                 lookahead: bool = True, near_dist: float = 0.15, look_gap: int = 3):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vis_threshold = vis_threshold
        self.n_min = n_min
        # 近场退化处理(方案一, 单帧版): query≈段内某记忆帧时该处可定位性最低、目标点横甩,
        # 故驱动目标不取段终点, 而取"越过近场退化区(distance>near_dist)、离 closest≥look_gap 帧
        # 的第一个稳定可见帧"作前瞻 aim。lookahead=False 退回旧行为(取段终点/最远可见帧)。
        self.lookahead = lookahead
        self.near_dist = near_dist
        self.look_gap = look_gap
        logger.info(f"[LotisEngine] 构建 localizer device={self.device} ...")
        self.localizer = build_localizer(checkpoint, config, dinov3_weights, self.device)
        self._warmup()
        self.encodings: Dict[str, object] = {}
        if traj_cache and os.path.exists(traj_cache):
            self.load_cache(traj_cache)

    def _warmup(self):
        """model.spatial_pos 在 encode_trajectory 里副作用式创建(仅依赖 14x14 patch 网格)。
        从缓存 load_cache 后直接 localize 会因它未初始化而 AttributeError —— 初始化时
        编码一段 dummy 触发一次, 使"载入缓存即用"路径可靠。"""
        try:
            self.encode_images([Image.new("RGB", (224, 224)) for _ in range(2)])
        except Exception as e:
            logger.warning(f"[LotisEngine] 预热失败(载入缓存直接 localize 可能报错): {e}")

    def load_cache(self, path: str):
        from lotis.localizer import TrajectoryEncoding
        with open(path, "rb") as f:
            raw = pickle.load(f)
        self.encodings = {
            k: (v if not isinstance(v, dict) else TrajectoryEncoding.from_dict(v, self.device))
            for k, v in raw.items()
        }
        logger.info(f"[LotisEngine] 载入 {len(self.encodings)} 段编码 <- {path}")

    def has_key(self, key: str) -> bool:
        return bool(key) and key in self.encodings

    def encode_images(self, imgs: List[Image.Image]):
        """一段 RGB PIL 帧 -> TrajectoryEncoding(内部方裁; >max_seq_len 自动 linspace 下采样)。"""
        imgs = [square_crop(im) for im in imgs]
        return self.localizer.encode_trajectory(imgs, max_frames=self.localizer.max_seq_len)

    def point(self, query_img: Image.Image, seg_key: Optional[str] = None,
              encoding=None, vis_threshold: Optional[float] = None,
              backward: bool = False) -> dict:
        """query 帧 -> 打点结果。seg_key 命中缓存, 或直接传 encoding。
        目标 = 段终点(正向)/ 段起点(backward=True, 反向沿同段返回)。
        近场前瞻朝目标方向选稳定可见帧。"""
        thr = self.vis_threshold if vis_threshold is None else vis_threshold
        enc = encoding if encoding is not None else self.encodings.get(seg_key)
        if enc is None:
            return {"found": False, "reason": f"段 {seg_key} 无编码"}
        q = square_crop(query_img)
        result = self.localizer.localize(q, enc)
        vis = np.asarray(result.visibility)
        dist = np.asarray(result.distances) if result.distances is not None else None
        vis_idx = np.where(vis > thr)[0]
        closest = int(result.closest_frame())
        goal = 0 if backward else int(getattr(enc, "seq_len", len(vis))) - 1

        def _fallback_aim():
            if 0 <= goal < len(vis) and vis[goal] > thr:
                return goal                   # 目标端(段终/起点)可见 -> 直接作目标
            if vis_idx.size:                  # 否则取最靠目标端的可见帧作转向代理
                return int(vis_idx[0] if backward else vis_idx[-1])
            return closest

        method = "fallback"
        if self.lookahead and vis_idx.size:
            # 前瞻候选: 可见 且 朝目标方向离 closest≥look_gap 帧 且 越过近场退化区
            if backward:
                cand = vis_idx[vis_idx <= closest - self.look_gap]
            else:
                cand = vis_idx[vis_idx >= closest + self.look_gap]
            if dist is not None and cand.size:
                cand = cand[dist[cand] > self.near_dist]
            if cand.size:
                aim = int(cand[-1] if backward else cand[0])   # 最靠 closest 的稳定可见帧
                method = "lookahead"
            else:
                aim = _fallback_aim()
        else:
            aim = _fallback_aim()
        r, c = result.coords[aim]
        center_pct = {"x": float((c + 1.0) / 2.0), "y": float((r + 1.0) / 2.0)}
        # 全段每帧在当前查询视图里的投影点串(coords row/col∈[-1,1] -> center_pct 0~1),
        # 附各帧可见度; 前端按 vis 门控/淡化, 画成沿轨迹的一串点。
        coords = np.asarray(result.coords)
        track = [{"frame": int(i),
                  "x": float((coords[i, 1] + 1.0) / 2.0),
                  "y": float((coords[i, 0] + 1.0) / 2.0),
                  "vis": float(vis[i])} for i in range(len(coords))]
        return {
            "found": bool(vis_idx.size >= self.n_min),
            "center_pct": center_pct,
            "confidence": float(vis[aim]) if aim < len(vis) else 0.0,
            "visible": int(vis_idx.size),
            "n_frames": int(len(vis)),
            "closest_frame": closest,
            "aim_frame": int(aim),
            "aim_method": method,
            "backward": bool(backward),
            "closest_dist": float(dist[closest]) if dist is not None else None,
            "vis_threshold": float(thr),
            "track": track,
        }


# --------------------------------------------------------------------------- #
# M0 命令行验证: 从现有关键帧构一个"节点边段", 编码 + 单帧 localize 出坐标
# --------------------------------------------------------------------------- #
def _cli():
    import argparse
    import json
    ap = argparse.ArgumentParser(description="LoTIS 引擎 M0 验证")
    ap.add_argument("--run", default="logs/cfds_floor28_run")
    ap.add_argument("--seq", default="cfds_floor28")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run = os.path.join(_PROJ, args.run) if not os.path.isabs(args.run) else args.run
    thumbs = os.path.join(run, "web", "thumbs")
    sem = json.loads(open(os.path.join(run, f"{args.seq}_semantic.json")).read())
    nodes = sem["nodes"]
    # 节点按 walk 顺序(锚点 = min(kf_indices)) 排序, 取前两个节点构一条边段
    nodes = sorted(nodes, key=lambda n: min(n["kf_indices"]))
    A, B = nodes[0], nodes[1]
    a, b = min(A["kf_indices"]), min(B["kf_indices"])
    kf_lo, kf_hi = min(a, b), max(a, b)
    seg_kfs = list(range(kf_lo, kf_hi + 1))
    print(f"[M0] 边段: 节点[{A['name']}](kf{a}) -> [{B['name']}](kf{b}), "
          f"关键帧 {kf_lo}..{kf_hi} 共 {len(seg_kfs)} 帧")

    def load_kf(i):
        p = os.path.join(thumbs, f"kf{i}.jpg")
        return Image.open(p).convert("RGB")

    imgs = [load_kf(i) for i in seg_kfs]
    eng = LotisEngine(device=args.device)
    import time
    t0 = time.time()
    enc = eng.encode_images(imgs)
    print(f"[M0] 段编码完成 {time.time()-t0:.2f}s, seq_len={enc.seq_len} (原 {len(imgs)} 帧)")

    # query = 段中间一帧 (应 localize 到中部, closest≈中点, 可见帧一片)
    qi = seg_kfs[len(seg_kfs) // 2]
    t0 = time.time()
    res = eng.point(load_kf(qi), encoding=enc)
    print(f"[M0] query=kf{qi}(段中点) localize {time.time()-t0:.3f}s")
    print(f"[M0] 结果: {json.dumps(res, ensure_ascii=False)}")
    print(f"[M0] 解读: found={res['found']} 可见 {res['visible']}/{res['n_frames']} 帧, "
          f"closest_frame={res['closest_frame']}(段内序号, 中点≈{enc.seq_len//2}), "
          f"目标点 center_pct=({res['center_pct']['x']:.2f},{res['center_pct']['y']:.2f})")


if __name__ == "__main__":
    _cli()
