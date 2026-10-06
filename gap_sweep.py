"""N1: 视角间隔鲁棒性扫描（推理期，无训练）。
测试时 context 固定为 [0, gap]，target 固定 [3,9]（dataset_cunei 确定性逻辑）。
对 gap in {1,2,4,8} 评测 T1/T2: PSNR + masked depth MAE + 统一协议 CD。
回答审稿人必问: "两张照片的拍摄角度差多少才行?"
用法: python gap_sweep.py --experiment cunei_geo --ckpt <path> [--gaps 1,2,4,8]
"""
import argparse, dataclasses, glob, json, os, sys

import numpy as np
import torch
from PIL import Image as PILImage

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
    from src.loss import get_losses
    from src.misc.step_tracker import StepTracker
    set_cfg(cfg)
    cfg = load_typed_root_config(cfg)
    encoder, ev = get_encoder(cfg.model.encoder)
    return ModelWrapper(cfg.optimizer, cfg.test, cfg.train, encoder, ev,
                        get_decoder(cfg.model.decoder, cfg.dataset),
                        get_losses(cfg.loss), StepTracker())


def unified_cd(means, gt, scale_cm):
    from scipy.spatial import cKDTree
    if len(means) > 50000:
        sel = np.random.default_rng(0).choice(len(means), 50000, replace=False)
        means = means[sel]
    r = np.linalg.norm(means, axis=1)
    means = means[r < 1.6]
    if len(means) < 100:
        return -1.0
    t1 = cKDTree(gt); d1, _ = t1.query(means, k=1)
    t2 = cKDTree(means); d2, _ = t2.query(gt, k=1)
    return float(0.5 * (d1.mean() + d2.mean()) * scale_cm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--renders", default="/root/autodl-tmp/data/renders_full")
    ap.add_argument("--ply_dir", default="/root/autodl-tmp/data/full_src")
    ap.add_argument("--gaps", default="1,2,4,8")
    ap.add_argument("--out_dir", default="/root/analysis_out")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    wrapper = build_model(args.experiment, args.renders)
    sd = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    state = sd.get("state_dict", sd)
    wrapper.load_state_dict(state, strict=False)
    wrapper = wrapper.cuda().eval()

    from src.dataset.dataset_cunei import DatasetCuneiCfg
    from src.dataset.view_sampler.view_sampler_bounded import ViewSamplerBoundedCfg
    from src.dataset import DATASETS
    from pathlib import Path
    import trimesh

    results = {}
    for gap in [int(x) for x in args.gaps.split(",")]:
        vs = ViewSamplerBoundedCfg(
            name="bounded", num_context_views=2, num_target_views=2,
            min_distance_between_context_views=2,
            max_distance_between_context_views=8,
            min_distance_to_context_views=1, warm_up_steps=0,
            initial_min_distance_between_context_views=2,
            initial_max_distance_between_context_views=8)
        cfg = DatasetCuneiCfg(
            name="cunei", roots=[Path(args.renders)], view_sampler=vs,
            image_shape=[256, 256], background_color=[0, 0, 0],
            cameras_are_circular=True, overfit_to_scene=None,
            near=0.8, far=5.0, context_gap=gap, num_target=2)
        ds = DATASETS["cunei"](cfg, "test", None)
        rows = []
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
                gaussians = wrapper.encoder(batch["context"],
                                            wrapper.global_step,
                                            deterministic=False)
                out = wrapper.decoder.forward(
                    gaussians, batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"], batch["target"]["near"],
                    batch["target"]["far"], (256, 256), depth_mode="depth")
            color = getattr(out, "color", None)
            if color is None and isinstance(out, (tuple, list)):
                color = out[0]
            d_pred = out.depth[0].cpu().numpy()
            means = gaussians.means.detach()[0].cpu().numpy()
            meta = json.load(open(glob.glob(os.path.join(
                args.renders, item["scene"] + "*_000_meta.json"))[0]))
            scale_cm = float(meta["normalize_scale"]) * 0.1
            mesh = trimesh.load(os.path.join(args.ply_dir, meta["ply"]),
                                force="mesh", process=False)
            v = (np.asarray(mesh.vertices, dtype=np.float32)
                 - np.asarray(meta["normalize_center"])) / float(meta["normalize_scale"])
            rng = np.random.default_rng(0)
            if len(v) > 20000:
                v = v[rng.choice(len(v), 20000, replace=False)]
            psnrs, maes = [], []
            for bi, vi in enumerate(item["target"]["index"].tolist()):
                pred = (color[0, bi].clamp(0, 1).cpu().numpy()
                        .transpose(1, 2, 0) * 255).astype(np.uint8)
                gt = np.asarray(PILImage.open(os.path.join(
                    args.renders, f"{item['scene']}_{vi:03d}.png")))[..., :3]
                mse = np.mean((pred.astype(np.float64) / 255
                               - gt.astype(np.float64) / 255) ** 2)
                psnrs.append(10 * np.log10(1.0 / max(mse, 1e-12)))
                gt_dep = np.load(os.path.join(
                    args.renders, f"{item['scene']}_{vi:03d}_depth.npy"))
                msk = np.load(os.path.join(
                    args.renders, f"{item['scene']}_{vi:03d}_mask.npy"))
                m = msk & np.isfinite(d_pred[bi]) & (d_pred[bi] > 0)
                if m.sum() >= 100:
                    maes.append(float(np.abs(d_pred[bi][m] - gt_dep[m]).mean()))
            rows.append(dict(scene=item["scene"],
                             psnr=round(float(np.mean(psnrs)), 3),
                             depth_mae=round(float(np.mean(maes)), 5),
                             cd_cm=round(unified_cd(means, v, scale_cm), 4)))
        ok_cd = [r["cd_cm"] for r in rows if r["cd_cm"] > 0]
        results[gap] = dict(
            rows=rows,
            psnr_mean=round(float(np.mean([r["psnr"] for r in rows])), 3),
            depth_mae_mean=round(float(np.mean(
                [r["depth_mae"] for r in rows if r["depth_mae"] > 0])), 5),
            cd_mean=round(float(np.mean(ok_cd)), 4) if ok_cd else -1)
        print(f"gap={gap}: {results[gap]['psnr_mean']} dB / "
              f"{results[gap]['depth_mae_mean']} / {results[gap]['cd_mean']} cm",
              flush=True)

    json.dump(results, open(os.path.join(
        args.out_dir, f"gap_sweep_{args.experiment}.json"), "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
