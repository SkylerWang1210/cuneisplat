"""深度口径评测：渲染期望深度 vs GT 深度（几何监督所监督的量）。
用法: python eval_depth.py --experiment cunei_base|cunei_geo --ckpt <path> ...
输出: 每板 masked MAE（归一化单位 + 毫米），均值汇总。
"""
import argparse, json, os, sys, time

import numpy as np
import torch

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
    from src.loss import get_losses
    from src.misc.step_tracker import StepTracker

    set_cfg(cfg)
    cfg = load_typed_root_config(cfg)
    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    wrapper = ModelWrapper(
        cfg.optimizer, cfg.test, cfg.train, encoder, encoder_visualizer,
        get_decoder(cfg.model.decoder, cfg.dataset), get_losses(cfg.loss),
        StepTracker(),
    )
    return wrapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--renders", default="/root/autodl-tmp/data/renders_full")
    args = ap.parse_args()

    wrapper = build_model(args.experiment, args.renders)
    sd = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    state = sd.get("state_dict", sd)
    wrapper.load_state_dict(state, strict=True)
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
    cfg = DatasetCuneiCfg(
        name="cunei", roots=[Path(args.renders)], view_sampler=vs,
        image_shape=[256, 256], background_color=[0, 0, 0],
        cameras_are_circular=True, overfit_to_scene=None,
        near=0.8, far=5.0, context_gap=4, num_target=2)
    ds = DATASETS["cunei"](cfg, "test", None)
    print("test boards:", len(ds), flush=True)

    import glob
    results = []
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
            gaussians = wrapper.encoder(
                batch["context"], wrapper.global_step, deterministic=False)
            out = wrapper.decoder.forward(
                gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (256, 256),
                depth_mode="depth",
            )
        d_pred = out.depth[0].cpu().numpy()  # (v, h, w)
        # GT 深度与掩膜
        meta = json.load(open(glob.glob(os.path.join(
            args.renders, item["scene"] + "*_000_meta.json"))[0]))
        scale_mm = float(meta["normalize_scale"])
        maes = []
        for vi, v in enumerate(item["target"]["index"].tolist()):
            gt = np.load(os.path.join(args.renders, f"{item['scene']}_{v:03d}_depth.npy"))
            m = np.load(os.path.join(args.renders, f"{item['scene']}_{v:03d}_mask.npy"))
            dp = d_pred[vi]
            valid = m & np.isfinite(dp) & (dp > 0)
            if valid.sum() < 100:
                continue
            mae_norm = np.abs(dp[valid] - gt[valid]).mean()
            maes.append(dict(view=v, mae_norm=float(mae_norm),
                             mae_mm=float(mae_norm * scale_mm)))
        board_mae = float(np.mean([x["mae_norm"] for x in maes]))
        board_mm = float(np.mean([x["mae_mm"] for x in maes]))
        results.append(board_mae)
        print(f"[{k+1}/{len(ds)}] {item['scene']} depthMAE_norm={board_mae:.4f} "
              f"= {board_mm:.2f}mm", flush=True)

    import statistics as st
    print(f"\nSUMMARY {args.experiment}: mean_norm={st.mean(results):.4f} "
          f"median_norm={st.median(results):.4f}", flush=True)


if __name__ == "__main__":
    main()
