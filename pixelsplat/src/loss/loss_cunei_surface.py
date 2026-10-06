"""方案A（v2 方法）：逐高斯表面监督损失 LossCuneiSurface
对每个高斯中心，找训练板真值网格上最近表面点的距离，直接作为损失。
真值来源：训练板 PLY（KD 树，子采样 2 万顶点）；高斯中心在归一化世界系。
batch 需携带 context 的归一化参数 + ply 路径 —— 通过 dataset 注入 meta 信息。
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
class LossCuneiSurfaceCfg:
    weight: float
    max_points: int = 10000      # 参与损失的高斯上限（均匀子采样）
    gt_points: int = 20000        # 真值顶点子采样数


@dataclass
class LossCuneiSurfaceCfgWrapper:
    cunei_surface: LossCuneiSurfaceCfg


class LossCuneiSurface(Loss[LossCuneiSurfaceCfg, LossCuneiSurfaceCfgWrapper]):
    """Per-Gaussian nearest-surface supervision.

    batch["context"]["_surface_pts"]: (b, P, 3) float — 训练板的真值表面点
    （归一化世界系，由 dataset 预计算注入；评测板无需注入）。
    损失 = 每个高斯中心到最近真值点的平均距离（单向，opacity 加权）。
    """

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        pts = batch["context"].get("_surface_pts")
        if pts is None:
            return torch.zeros((), device=gaussians.means.device)
        means = gaussians.means  # (b, n, 3)
        opac = gaussians.opacities.clamp(0, 1)  # (b, n)
        b, n, _ = means.shape
        total = means.new_zeros(())
        for i in range(b):
            m = means[i]                        # (n, 3)
            o = opac[i]
            if n > self.cfg.max_points:
                sel = torch.randperm(n, device=m.device)[: self.cfg.max_points]
                m, o = m[sel], o[sel]
            # 分块 float32 cdist：1024×20000×4B = 82MB/块，可容纳多个块梯度
            nn_parts = []
            for j in range(0, m.shape[0], 1024):
                d = torch.cdist(m[j : j + 1024], pts[i])
                nn_parts.append(d.min(dim=1).values)
            nn = torch.cat(nn_parts)
            total = total + (nn * o).sum() / o.sum().clamp(min=1e-6)
        return self.cfg.weight * total / b
