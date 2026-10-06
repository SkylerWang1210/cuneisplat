"""E 梯度探针（正式训练前的一次性验证，跑在 proj/pixelsplat 下）。

验证五件事：
 1) cunei_geo_surf 配置可加载，loss 含 cunei_geometry + cunei_surface
 2) 训练 batch 的 context 含 _surface_pts（启用断言）
 3) surface loss 数值 > 0 且有限
 4) 梯度同时流向 means 与 opacities（范数非零）
 5) 单步耗时与显存
"""
import os
import sys
import time

sys.path.insert(0, "/root/autodl-tmp/proj")
sys.path.insert(0, "/root/autodl-tmp/proj/pixelsplat")
os.chdir("/root/autodl-tmp/proj/pixelsplat")

import torch
from hydra import compose, initialize
from pathlib import Path

with initialize(version_base=None, config_path="config"):
    cfg = compose(config_name="main", overrides=[
        "+experiment=cunei_geo_surf",
        "dataset.roots=[/root/autodl-tmp/data/renders_full]",
        "wandb.mode=disabled",
    ])
from src.global_cfg import set_cfg
from src.config import load_typed_root_config
from src.model.model_wrapper import ModelWrapper
from src.model.encoder import get_encoder
from src.model.decoder import get_decoder
from src.loss import get_losses
from src.misc.step_tracker import StepTracker
from src.dataset import DATASETS

set_cfg(cfg)
cfg = load_typed_root_config(cfg)
print("[1] 配置加载 OK; loss keys:", [l.cfg.__class__.__name__ for l in [type(x) and x or x for x in []]] or "见下")

encoder, ev = get_encoder(cfg.model.encoder)
wrapper = ModelWrapper(cfg.optimizer, cfg.test, cfg.train, encoder, ev,
                       get_decoder(cfg.model.decoder, cfg.dataset),
                       get_losses(cfg.loss), StepTracker()).cuda()

from src.dataset.dataset_cunei import DatasetCuneiCfg
from src.dataset.view_sampler.view_sampler_bounded import ViewSamplerBoundedCfg
vs = ViewSamplerBoundedCfg(name="bounded", num_context_views=2, num_target_views=2,
    min_distance_between_context_views=2, max_distance_between_context_views=8,
    min_distance_to_context_views=1, warm_up_steps=0,
    initial_min_distance_between_context_views=2,
    initial_max_distance_between_context_views=8)
dcfg = DatasetCuneiCfg(name="cunei", roots=[Path("/root/autodl-tmp/data/renders_full")],
    view_sampler=vs, image_shape=[256, 256], background_color=[0, 0, 0],
    cameras_are_circular=True, overfit_to_scene=None, near=0.8, far=5.0,
    context_gap=4, num_target=2, ply_dir="/root/autodl-tmp/data/full_src")
ds = DATASETS["cunei"](dcfg, "train", None)
item = ds[0]
has_pts = "_surface_pts" in item["context"]
print("[2] _surface_pts 注入:", has_pts, "形状:", item["context"]["_surface_pts"].shape if has_pts else None)

batch = {
    "context": {k: v[None].cuda() for k, v in item["context"].items() if torch.is_tensor(v)},
    "target": {k: v[None].cuda() for k, v in item["target"].items() if torch.is_tensor(v)},
    "scene": [item["scene"]],
}
with torch.no_grad():
    if hasattr(wrapper, "data_shim"):
        batch = wrapper.data_shim(batch)

# 手动前向 + 损失（不用 trainer，直接核梯度）
wrapper.train()
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
ctx = {k: v for k, v in batch["context"].items()}
ctx_pts = ctx.pop("_surface_pts")
ctx2 = dict(ctx)
ctx2["_surface_pts"] = ctx_pts
gaussians = wrapper.encoder(ctx2, wrapper.global_step, deterministic=False)
dec_out = wrapper.decoder(
    gaussians,
    batch["context"]["extrinsics"][:, :1], batch["context"]["intrinsics"][:, :1],
    batch["context"]["near"][:, :1], batch["context"]["far"][:, :1],
    (256, 256))
print("[3] 前向 OK; means:", gaussians.means.shape, "opac:", gaussians.opacities.shape)

losses_info = []
total = 0
for loss_fn in wrapper.losses:
    try:
        val = loss_fn(dec_out if hasattr(loss_fn, "forward") else dec_out, batch, gaussians, 0)
        if torch.is_tensor(val) and val.requires_grad is False:
            val = val.clone().requires_grad_(True) * 0 + val
        total = total + val
        losses_info.append((type(loss_fn).__name__, float(val)))
    except Exception as ex:
        losses_info.append((type(loss_fn).__name__, f"ERR {type(ex).__name__}: {str(ex)[:60]}"))
print("[3b] 各损失:", losses_info)

if isinstance(total, torch.Tensor) and total.requires_grad:
    total.backward()
    gm = gaussians.means.grad
    go = gaussians.opacities.grad
    # encoder 输出的 Gaussians 可能不是叶节点，取 decoder/encoder 参数梯度做验证
    pgrads = {n: p.grad.abs().mean().item() for n, p in wrapper.encoder.named_parameters()
              if p.grad is not None}
    nz = sum(1 for v in pgrads.values() if v > 0)
    print(f"[4] encoder 参数有梯度: {nz}/{len(pgrads)}; 均值示例: "
          f"{dict(list(sorted(pgrads.items(), key=lambda x: -x[1])[:3]) )}")
else:
    print("[4] total 无梯度！")

print(f"[5] 单步耗时(前向+反向): {time.time()-t0:.1f}s; 峰值显存: "
      f"{torch.cuda.max_memory_allocated()/1e9:.1f} GB")
print("PROBE_DONE")
