"""alpha 覆盖诊断（REV-009/010 配套）：转储逐像素累计透明度 + 原始深度。

对两个 seed1 检查点 × 9 测试板 × 2 目标视角：
  A(u)：render_cuda 用全 1 颜色、零背景 → 输出均值即累计 alpha
  D_raw：decoder.render_depth（论文口径的累积深度）
  GT 深度/掩膜：renders_full 的 npy
输出 /root/autodl-tmp/analysis2/out/alpha_dump/{model}/{scene}.npz
"""
import os
import sys

sys.path.insert(0, "/root/autodl-tmp/proj")
sys.path.insert(0, "/root/autodl-tmp/proj/pixelsplat")
os.chdir("/root/autodl-tmp/proj/pixelsplat")

import numpy as np
import torch
from einops import rearrange, repeat
from hydra import compose, initialize
from pathlib import Path

from src.model.decoder.cuda_splatting import render_cuda

BASE = "/root/autodl-tmp/analysis2"
CKPTS = {
    "cunei_base": "/root/autodl-tmp/proj/pixelsplat/outputs/2026-10-01/03-27-14/checkpoints/epoch=465-step=20000.ckpt",
    "cunei_geo": "/root/autodl-tmp/proj/pixelsplat/outputs/2026-10-01/11-50-30/checkpoints/epoch=465-step=20000.ckpt",
}


def build(experiment):
    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name="main", overrides=[
            f"+experiment={experiment}",
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
    set_cfg(cfg)
    cfg = load_typed_root_config(cfg)
    enc, ev = get_encoder(cfg.model.encoder)
    return ModelWrapper(cfg.optimizer, cfg.test, cfg.train, enc, ev,
                        get_decoder(cfg.model.decoder, cfg.dataset),
                        get_losses(cfg.loss), StepTracker())


def make_ds():
    from src.dataset.dataset_cunei import DatasetCuneiCfg
    from src.dataset.view_sampler.view_sampler_bounded import ViewSamplerBoundedCfg
    from src.dataset import DATASETS
    vs = ViewSamplerBoundedCfg(name="bounded", num_context_views=2, num_target_views=2,
        min_distance_between_context_views=2, max_distance_between_context_views=8,
        min_distance_to_context_views=1, warm_up_steps=0,
        initial_min_distance_between_context_views=2,
        initial_max_distance_between_context_views=8)
    dcfg = DatasetCuneiCfg(name="cunei", roots=[Path("/root/autodl-tmp/data/renders_full")],
        view_sampler=vs, image_shape=[256, 256], background_color=[0, 0, 0],
        cameras_are_circular=True, overfit_to_scene=None, near=0.8, far=5.0,
        context_gap=4, num_target=2)
    return DATASETS["cunei"](dcfg, "test", None)


def render_alpha(g, extr, intr, near, far):
    """累计 alpha 图 (v,h,w)：全1颜色零背景。"""
    b, v = extr.shape[:2]
    ones_sh = torch.ones(b, g.means.shape[1], 3, 1, device=g.means.device,
                         dtype=g.means.dtype)
    rgb = render_cuda(
        rearrange(extr, "b v i j -> (b v) i j"),
        rearrange(intr, "b v i j -> (b v) i j"),
        rearrange(near, "b v -> (b v)"), rearrange(far, "b v -> (b v)"),
        (256, 256),
        torch.zeros(b * v, 3, device=g.means.device),
        repeat(g.means, "b n xyz -> (b v) n xyz", v=v),
        repeat(g.covariances, "b n i j -> (b v) n i j", v=v),
        repeat(ones_sh, "b n s c -> (b v) n s c", v=v),
        repeat(g.opacities, "b n -> (b v) n", v=v),
        use_sh=False,
    )
    return rearrange(rgb.mean(dim=1), "(b v) h w -> b v h w", b=b, v=v)


def main():
    os.makedirs(f"{BASE}/out/alpha_dump", exist_ok=True)
    ds = make_ds()
    for model, ckpt in CKPTS.items():
        wrapper = build(model)
        sd = torch.load(ckpt, map_location="cuda", weights_only=False)
        wrapper.load_state_dict(sd.get("state_dict", sd), strict=True)
        wrapper = wrapper.cuda().eval()
        os.makedirs(f"{BASE}/out/alpha_dump/{model}", exist_ok=True)
        for k in range(len(ds)):
            item = ds[k]
            scene = item["scene"]
            batch = {
                "context": {kk: vv[None].cuda() for kk, vv in item["context"].items() if torch.is_tensor(vv)},
                "target": {kk: vv[None].cuda() for kk, vv in item["target"].items() if torch.is_tensor(vv)},
                "scene": [scene],
            }
            with torch.no_grad():
                if hasattr(wrapper, "data_shim"):
                    batch = wrapper.data_shim(batch)
                torch.manual_seed(0)
                g = wrapper.encoder(batch["context"], wrapper.global_step, deterministic=False)
                A = render_alpha(g, batch["target"]["extrinsics"], batch["target"]["intrinsics"],
                                 batch["target"]["near"], batch["target"]["far"])
                D = wrapper.decoder.render_depth(
                    g, batch["target"]["extrinsics"], batch["target"]["intrinsics"],
                    batch["target"]["near"], batch["target"]["far"], (256, 256), mode="depth")
            gt, msk = [], []
            for vi in batch["target"]["index"][0].tolist():
                gt.append(np.load(f"/root/autodl-tmp/data/renders_full/{scene}_{vi:03d}_depth.npy"))
                msk.append(np.load(f"/root/autodl-tmp/data/renders_full/{scene}_{vi:03d}_mask.npy"))
            np.savez_compressed(
                f"{BASE}/out/alpha_dump/{model}/{scene}.npz",
                alpha=A[0].cpu().numpy().astype(np.float16),
                depth=D[0].cpu().numpy(),
                gt=np.stack(gt), mask=np.stack(msk).astype(bool))
            print(f"[{model}] {scene} alpha均值={float(A.mean()):.3f}", flush=True)
    print("DUMP_DONE", flush=True)


if __name__ == "__main__":
    main()
