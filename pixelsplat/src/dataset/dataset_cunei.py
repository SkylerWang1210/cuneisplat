"""CuneiSplat 数据集：把渲染管线输出适配为 pixelSplat 训练格式。

渲染输出约定（render_pipeline.py）：
  {stem}_{i:03d}.png / _depth.npy / _mask.npy / {stem}_000_meta.json
meta.json: {split, normalize_scale, views: [{id,K,R(w2c 3x3),t(3,)}]}

样本格式遵循 src/dataset/types.py：
  context/target: {extrinsics(c2w 4x4), intrinsics, image(3,H,W,[0,1]),
                   near, far, index, [depth, depth_mask](仅 target)}
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .dataset import DatasetCfgCommon
from .types import Stage
from .view_sampler import ViewSamplerCfg


@dataclass
class DatasetCuneiCfg(DatasetCfgCommon):
    name: Literal["cunei"]
    roots: list[Path]
    near: float
    far: float
    context_gap: int                    # 两个 context 视角的索引间隔
    num_target: int                     # 训练时随机目标视角数
    ply_dir: str = ""                   # 方案A真值表面点来源（空=不注入）
    surface_points: int = 20000         # 真值顶点子采样数


class DatasetCunei(Dataset):
    def __init__(
        self,
        cfg: DatasetCuneiCfg,
        stage: Stage,
        view_sampler,  # 兼容注册接口，不使用
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        assert self.cfg.image_shape[0] == self.cfg.image_shape[1]
        self.to_tensor = torch.get_default_dtype()

    def __len__(self) -> int:
        return len(self.index)

    @property
    def index(self) -> dict[str, Path]:
        merged = {}
        for root in map(Path, self.cfg.roots):
            for meta_path in root.glob("*_000_meta.json"):
                meta = json.loads(meta_path.read_text())
                if meta.get("split") != self.stage:
                    continue
                merged[meta["ply"].replace(".ply", "")] = root
        return merged

    def _load_view(self, root: Path, stem: str, i: int, meta: dict,
                   with_depth: bool) -> dict:
        v = meta["views"][i]
        img = np.asarray(Image.open(root / f"{stem}_{i:03d}.png"))[..., :3]
        img = torch.from_numpy(img).to(self.to_tensor) / 255.0
        img = rearrange(img, "h w c -> c h w")
        # w2c -> c2w（OpenCV 约定：+z 向前，+y 向下——与 pixelSplat 期望一致；
        # GL 翻转仅渲染数据时需要，喂模型时不能带）
        w2c = torch.eye(4, dtype=torch.float32)
        w2c[:3, :3] = torch.tensor(v["R"], dtype=torch.float32)
        w2c[:3, 3] = torch.tensor(v["t"], dtype=torch.float32)
        extr = w2c.inverse().to(self.to_tensor)
        # 内参归一化（pixelSplat 约定：fx/fy/cx/cy ∈ [0,1]，主点≈(0.5,0.5)）
        K = torch.tensor(v["K"], dtype=torch.float32)
        s = self.cfg.image_shape[0]
        intr = torch.tensor(
            [[K[0, 0] / s, 0.0, K[0, 2] / s],
             [0.0, K[1, 1] / s, K[1, 2] / s],
             [0.0, 0.0, 1.0]], dtype=torch.float32)
        out = {
            "extrinsics": extr, "intrinsics": intr, "image": img,
            "index": torch.tensor(i),
        }
        if with_depth:
            d = np.load(root / f"{stem}_{i:03d}_depth.npy")
            m = np.load(root / f"{stem}_{i:03d}_mask.npy")
            out["depth"] = torch.from_numpy(d).to(self.to_tensor)
            out["depth_mask"] = torch.from_numpy(m)
        return out

    def __getitem__(self, index: int) -> dict:
        scene, root = list(self.index.items())[index]
        meta = json.loads((root / f"{scene}_000_meta.json").read_text())
        n_views = len(meta["views"])
        gap = self.cfg.context_gap % n_views

        rng = np.random.default_rng()
        if self.stage == "train":
            i0 = int(rng.integers(0, n_views))
        else:  # val/test 固定起点，保证评测可复现
            i0 = 0
        i1 = (i0 + gap) % n_views
        ctx_idx = [i0, i1]

        if self.stage == "train":
            cands = [i for i in range(n_views) if i not in ctx_idx]
            tgt_idx = [int(x) for x in rng.choice(
                cands, size=self.cfg.num_target, replace=False)]
        else:
            tgt_idx = [(i0 + 3) % n_views, (i0 + 9) % n_views]

        def views(idxs, with_depth):
            items = [self._load_view(root, scene, i, meta, with_depth)
                     for i in idxs]
            return {
                k: torch.stack([it[k] for it in items])
                for k in items[0]
            }

        near = torch.full((len(ctx_idx),), self.cfg.near)
        far = torch.full((len(ctx_idx),), self.cfg.far)
        ctx = {**views(ctx_idx, False), "near": near, "far": far}
        # 训练时注入真值表面点（方案A逐高斯监督用；评测不注入）
        if self.stage == "train" and getattr(self.cfg, "ply_dir", ""):
            surf = self._surface_points(root, scene, meta)
            if surf is not None:
                ctx["_surface_pts"] = surf
        return {
            "context": ctx,
            "target": {**views(tgt_idx, True),
                       "near": near[: len(tgt_idx)], "far": far[: len(tgt_idx)]},
            "scene": scene,
        }

    _surf_cache: dict = {}

    def _surface_points(self, root, scene, meta):
        """板真值顶点（归一化世界系，固定子采样）——确定性缓存。"""
        import trimesh
        key = scene
        if key in self._surf_cache:
            return self._surf_cache[key]
        ply = os.path.join(self.cfg.ply_dir, meta["ply"])
        if not os.path.exists(ply):
            self._surf_cache[key] = None
            return None
        mesh = trimesh.load(ply, force="mesh", process=False)
        v = np.asarray(mesh.vertices, dtype=np.float32)
        v = (v - np.asarray(meta["normalize_center"])) / float(meta["normalize_scale"])
        n_pts = int(getattr(self.cfg, "surface_points", 20000))
        rng = np.random.default_rng(1234)
        if len(v) > n_pts:
            v = v[rng.choice(len(v), n_pts, replace=False)]
        elif len(v) < n_pts:
            v = v[rng.choice(len(v), n_pts, replace=True)]
        t = torch.from_numpy(v).to(self.to_tensor)
        self._surf_cache[key] = t
        return t
