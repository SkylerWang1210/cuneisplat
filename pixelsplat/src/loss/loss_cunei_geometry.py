"""CuneiSplat 几何监督损失：渲染深度 vs 网格真值深度（论文 §3.2 公式实现）。

与 pixelSplat 自带 loss_depth.py 的本质区别：
- loss_depth.py   = 无监督平滑正则（惩罚预测深度的空间梯度，不用真值）
- 本损失          = 有监督几何监督（对比渲染管线自带的 GT 深度，需可见性掩膜）
"""
from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossCuneiGeometryCfg:
    weight: float


@dataclass
class LossCuneiGeometryCfgWrapper:
    cunei_geometry: LossCuneiGeometryCfg


class LossCuneiGeometry(Loss[LossCuneiGeometryCfg, LossCuneiGeometryCfgWrapper]):
    """Masked L1 between rendered expected depth and ground-truth depth.

    batch["target"]["depth"]      : (b v h w) float, 真值深度（渲染管线产出）
    batch["target"]["depth_mask"] : (b v h w) bool,  可见性掩膜
    prediction.depth              : (b v h w) float, 光栅化期望深度（depth_mode=depth）
    """

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        depth_pred = prediction.depth
        depth_gt = batch["target"]["depth"].to(depth_pred.dtype)
        mask = batch["target"]["depth_mask"].to(depth_pred.dtype)
        diff = (depth_pred - depth_gt).abs() * mask
        return self.cfg.weight * diff.sum() / mask.sum().clamp(min=1.0)
