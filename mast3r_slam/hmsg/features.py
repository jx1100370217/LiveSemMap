"""SAM2 自动掩码 + ConceptFusion 式逐掩码 CLIP 特征
(掩码 CLIP 公式与超参照抄 fsr_vln perception/models/sam_clip_feats_extractor.py;
分割器按论文用 SAM2 — hiera-large 权重来自 L40, 输出 dict 与 SAM1 AMG 兼容,
自动掩码参数照抄原版: points_per_side=12, pred_iou=0.88, stability=0.95,
min_mask_region_area=100, crop_n_layers=0)。

- 每掩码特征: bbox 外扩 50px 裁两版 (保背景 crop / 遮背景 masked),
  f_local = 0.4418*f_masked + 0.5582*f_crop 后归一化;
  再与全图特征融合: w_i = softmax_i(cos(f_local_i, f_global)),
  f_i = w_i*f_global + (1-w_i)*f_local_i, 归一化。
- 文本特征: 双模板均值 ["{}", "a photo of {} in the scene."] (utils/clip_utils)。

权重路径统一 pretrained/。
"""
import pathlib

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
SAM2_DIR = _ROOT / "pretrained" / "sam2-hiera-large-hf"
CLIP_CKPT = _ROOT / "pretrained" / "openclip_vit_l14_laion2b.bin"
CLIP_DIM = 768

MASKED_WEIGHT = 0.4418      # clip_masked_weight (config 默认)
BBOX_MARGIN = 50            # clip_bbox_margin (px)


class SamClipExtractor:
    def __init__(self, device="cuda:0"):
        import open_clip
        from safetensors.torch import load_file
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2
        self.device = device
        sam = build_sam2("configs/sam2/sam2_hiera_l.yaml", ckpt_path=None,
                         device=device)
        missing, unexpected = sam.load_state_dict(
            load_file(SAM2_DIR / "model.safetensors"), strict=False)
        assert len(missing) <= 4, f"SAM2 权重键缺失过多: {missing}"
        self.mask_gen = SAM2AutomaticMaskGenerator(
            model=sam, points_per_side=12, pred_iou_thresh=0.88,
            stability_score_thresh=0.95, min_mask_region_area=100,
            crop_n_layers=0)
        self.clip, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained=str(CLIP_CKPT))
        self.clip = self.clip.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14")

    @torch.no_grad()
    def _encode_pil(self, pil_imgs):
        x = torch.stack([self.preprocess(im) for im in pil_imgs]).to(self.device)
        f = self.clip.encode_image(x).float()
        return torch.nn.functional.normalize(f, dim=-1)

    @torch.no_grad()
    def encode_text(self, labels):
        """双模板均值文本特征 (N_label, D), 归一化。"""
        feats = []
        for tpl in ("{}", "a photo of {} in the scene."):
            tok = self.tokenizer([tpl.format(x) for x in labels]).to(self.device)
            f = self.clip.encode_text(tok).float()
            feats.append(torch.nn.functional.normalize(f, dim=-1))
        f = torch.stack(feats).mean(0)
        return torch.nn.functional.normalize(f, dim=-1).cpu().numpy()

    @torch.no_grad()
    def extract(self, rgb):
        """rgb: (H,W,3) uint8 -> (masks[list of dict], mask_feats (M,D),
        global_feat (D,))。masks 按面积降序 (SAM 默认)。"""
        from PIL import Image
        masks = self.mask_gen.generate(rgb)
        pil_full = Image.fromarray(rgb)
        f_global = self._encode_pil([pil_full])[0]           # (D,)
        if not masks:
            return [], np.zeros((0, CLIP_DIM), np.float32), \
                f_global.cpu().numpy()
        H, W = rgb.shape[:2]
        crops, crops_masked = [], []
        for m in masks:
            x, y, w, h = [int(v) for v in m["bbox"]]
            x0, y0 = max(0, x - BBOX_MARGIN), max(0, y - BBOX_MARGIN)
            x1, y1 = min(W, x + w + BBOX_MARGIN), min(H, y + h + BBOX_MARGIN)
            crop = rgb[y0:y1, x0:x1]
            seg = m["segmentation"][y0:y1, x0:x1]
            masked = crop.copy()
            masked[~seg] = 0
            crops.append(Image.fromarray(crop))
            crops_masked.append(Image.fromarray(masked))
        f_crop = self._encode_pil(crops)                     # (M,D)
        f_masked = self._encode_pil(crops_masked)
        f_local = torch.nn.functional.normalize(
            MASKED_WEIGHT * f_masked + (1 - MASKED_WEIGHT) * f_crop, dim=-1)
        # 与全图特征加权融合 (softmax 权重跨掩码归一)
        w = torch.softmax(f_local @ f_global, dim=0).unsqueeze(-1)   # (M,1)
        f = torch.nn.functional.normalize(
            w * f_global.unsqueeze(0) + (1 - w) * f_local, dim=-1)
        return masks, f.cpu().numpy().astype(np.float32), \
            f_global.cpu().numpy().astype(np.float32)


def load_vocab(name="scannet200"):
    """物体词表 (原版 obj_labels csv, 每行一个类名)。"""
    p = pathlib.Path(__file__).parent / "vocab" / f"{name}.csv"
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
