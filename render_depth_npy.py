"""render_depth_npy.py: 9 测试板 × 目标视角, 输出未剪枝期望深度 npy(与 prune_render_eval.py 完全同协议)。
用法: /root/miniconda3/envs/cunei/bin/python render_depth_npy.py --experiment cunei_geo --ckpt <path>
"""
import argparse
import dataclasses
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/proj")
sys.path.insert(0, "/root/autodl-tmp/proj/pixelsplat")


def build_model(experiment, renders):
    from hydra import compose, initialize
    with initialize(version_base=None, config_path="pixelsplat/config"):
        cfg = compose(config_name="main", overrides=[
            f"+experiment={experiment}",
            f"dataset.roots=[{renders}]",
            "wandb.mode=disabled",
        ])
    from src.global_cfg import set_cfg
    from src.config import load_typed_root_config
    from src.model.model_wrapper import ModelWrapper
    from src.model.encoder import get_encoder
    from src.model.decoder import get_decoder
    from src.model.decoder import get_decoder
    from src.loss import get_losses
    from src.misc.step_tracker import StepTracker
    set_cfg(cfg)
    cfg = load_typed_root_config(cfg)
    encoder, ev = get_encoder(cfg.model.encoder)
    return ModelWrapper(cfg.optimizer, cfg.test, cfg.train, encoder, ev,
                        get_decoder(cfg.model.decoder, cfg.dataset),
                        get_losses(cfg.loss), StepTracker())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--renders", default="/root/autodl-tmp/data/renders_full")
    ap.add_argument("--out", default="/root/depth_out")
    args = ap.parse_args()

    exp_dir = os.path.join(args.out, args.experiment)
    gt_dir = os.path.join(args.out, "gt")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    wrapper = build_model(args.experiment, args.renders)
    sd = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    state = sd.get("state_dict", sd)
    missing, unexpected = wrapper.load_state_dict(state, strict=False)
    print(f"ckpt loaded: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    wrapper = wrapper.cuda().eval()

    from src.dataset.dataset_cunei import DatasetCuneiCfg
    from src.dataset.view_sampler.view_sampler_bounded import ViewSamplerBoundedCfg
    from src.dataset import DATASETS
    from pathlib import Path
    vs = ViewSamplerBoundedCfg(
        name="bounded", num_context_views=2, num_target_views=2,
        min_distance_between_context_views=2, max_distance_between_context_views=8,
        min_distance_to_context_views=1, warm_up_steps=0,
        initial_min_distance_between_context_views=2,
        initial_max_distance_between_context_views=8)
    cfgd = DatasetCuneiCfg(
        name="cunei", roots=[Path(args.renders)], view_sampler=vs,
        image_shape=[256, 256], background_color=[0, 0, 0],
        cameras_are_circular=True, overfit_to_scene=None,
        near=0.8, far=5.0, context_gap=4, num_target=2)
    ds = DATASETS["cunei"](cfgd, "test", None)
    print("test boards:", len(ds), flush=True)

    def render_depth(gaussians, tgt):
        out = wrapper.decoder.forward(
            gaussians, tgt["extrinsics"], tgt["intrinsics"],
            tgt["near"], tgt["far"], (256, 256), depth_mode="depth")
        depth = getattr(out, "depth", None)
        if depth is None and isinstance(out, (tuple, list)):
            depth = out[1]
        return depth

    protocol = {}
    for k in range(len(ds)):
        item = ds[k]
        batch = {
            "target": {kk: vv[None].cuda() for kk, vv in item["target"].items()
                       if torch.is_tensor(vv)},
            "context": {kk: vv[None].cuda() for kk, vv in item["context"].items()
                        if torch.is_tensor(vv)},
            "scene": [item["scene"]],
        }
        with torch.no_grad():
            if hasattr(wrapper, "data_shim"):
                batch = wrapper.data_shim(batch)
            gaussians = wrapper.encoder(batch["context"], wrapper.global_step,
                                        deterministic=False)
            depth = render_depth(gaussians, batch["target"])
        d_pred = depth[0].float().cpu().numpy()
        tgt_idx = item["target"]["index"].tolist()
        ctx_idx = item["context"]["index"].tolist()
        protocol[item["scene"]] = {"context": ctx_idx, "target": tgt_idx}
        for bi, vi in enumerate(tgt_idx):
            np.save(os.path.join(exp_dir, f"{item['scene']}_{vi:03d}_depth.npy"),
                    d_pred[bi].astype(np.float32))
            np.save(os.path.join(gt_dir, f"{item['scene']}_{vi:03d}_depth.npy"),
                    np.load(os.path.join(args.renders,
                                         f"{item['scene']}_{vi:03d}_depth.npy")))
            np.save(os.path.join(gt_dir, f"{item['scene']}_{vi:03d}_mask.npy"),
                    np.load(os.path.join(args.renders,
                                         f"{item['scene']}_{vi:03d}_mask.npy")))
        print(f"[{k+1}/{len(ds)}] {item['scene']} ctx={ctx_idx} tgt={tgt_idx} "
              f"n_gauss={len(gaussians.means[0])}", flush=True)

    json.dump(protocol, open(os.path.join(args.out, f"protocol_{args.experiment}.json"), "w"),
              indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
