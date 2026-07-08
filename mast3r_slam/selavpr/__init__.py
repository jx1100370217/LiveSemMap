"""SelaVPR++ 全局描述子提取 (自 memory-nav 内联, 使 VGP-Nav 自包含)。

原属 memory-nav 的 memory_nav 包; 迁入本项目后作为 vgpnav.selavpr 子包, 不再跨项目引用。
权重经 torch.hub 缓存离线加载 (~/.cache/torch/hub/checkpoints/SelaVPRplusplus_large.pth)。
"""
from .selavpr_extractor import SelaVPRExtractor

__all__ = ["SelaVPRExtractor"]
