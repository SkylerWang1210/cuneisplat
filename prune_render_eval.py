"""A1+B9: 剪枝后实测渲染指标 + 高斯缓存 + 深度误差热图。
对每块测试板: encode -> (各 tau) 按 opacity 剪枝 -> 重渲染 color+depth
-> 实测 PSNR/SSIM/LPIPS + masked depth MAE + 统一协议 CD。
同时保存高斯缓存 npz 供离线分析(B8/A5)与 eval_protocol.json 供 3DGS 评测对齐。

用法:
  /root/miniconda3/envs/cunei/bin/python prune_render_eval.py \
      --experiment cunei_geo --ckpt <path> \
      [--renders /root/autodl-tmp/data/renders_full] [--limit 9]
"""
import argparse, dataclasses, glob, json, os, sys, time

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


def unified_cd(means, gt, scale_cm_factor):
    """与 eval_chamfer.py 完全一致的协议: pred cap 50k (rng0), GT 20k (rng0),
    r<1.6, 双向平均, x scale x 0.1 -> cm。"""
    from scipy.spatial import cKDTree
    if len(means) > 50000:
        sel = np.random.default_rng(0).choice(len(means), 50000, replace=False)
        means = means[sel]
    r = np.linalg.norm(means, axis=1)
    means = means[r < 1.6]
    if len(means) < 100:
        return -1.0
    t1 = cKDTree(gt)
    d1, _ = t1.query(means, k=1)
    t2 = cKDTree(means)
    d2, _ = t2.query(gt, k=1)
    return float(0.5 * (d1.mean() + d2.mean()) * scale_cm_factor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--renders", default="/root/autodl-tmp/data/renders_full")
    ap.add_argument("--ply_dir", default="/root/autodl-tmp/data/full_src")
    ap.add_argument("--taus", default="0.1,0.2,0.3")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--out_dir", default="/root/analysis_out")
    ap.add_argument("--heatmap_boards", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cache_dir = os.path.join(args.out_dir, f"gauss_cache_{args.experiment}")
    hm_dir = os.path.join(args.out_dir, "heatmaps", args.experiment)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(hm_dir, exist_ok=True)
    taus = [0.0] + [float(x) for x in args.taus.split(",")]

    wrapper = build_model(args.experiment, args.renders)
    sd = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    state = sd.get("state_dict", sd)
    missing, unexpected = wrapper.load_state_dict(state, strict=False)
    print(f"ckpt loaded: missing={len(missing)} unexpected={len(unexpected)}")
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

    from skimage.metrics import structural_similarity
    import lpips as lpips_mod
    lpips_fn = lpips_mod.LPIPS(net="vgg").cuda()
    import trimesh
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def render(gaussians, tgt):
        out = wrapper.decoder.forward(
            gaussians, tgt["extrinsics"], tgt["intrinsics"],
            tgt["near"], tgt["far"], (256, 256), depth_mode="depth")
        color = getattr(out, "color", None)
        depth = getattr(out, "depth", None)
        if color is None and isinstance(out, (tuple, list)):
            color, depth = out[0], out[1]
        return color, depth

    results = {"per_board": [], "taus": taus, "renders": args.renders}
    protocol = {}
    n_boards = min(args.limit, len(ds))
    for k in range(n_boards):
        item = ds[k]
        batch = {
            "target": {kk: vv[None].cuda() for kk, vv in item["target"].items()
                       if torch.is_tensor(vv)},
            "context": {kk: vv[None].cuda() for kk, vv in item["context"].items()
                        if torch.is_tensor(vv)},
            "scene": [item["scene"]],
        }
        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad():
            if hasattr(wrapper, "data_shim"):
                batch = wrapper.data_shim(batch)
            gaussians = wrapper.encoder(batch["context"], wrapper.global_step,
                                        deterministic=False)
        torch.cuda.synchronize(); enc_ms = (time.time() - t0) * 1000

        ctx_idx = item["context"]["index"].tolist()
        tgt_idx = item["target"]["index"].tolist()
        protocol[item["scene"]] = {"context": ctx_idx, "target": tgt_idx}

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

        # GT rgb/depth/mask for target views
        gt_rgb, gt_dep, gt_msk = {}, {}, {}
        for vi in tgt_idx:
            gt_rgb[vi] = np.asarray(PILImage.open(
                os.path.join(args.renders, f"{item['scene']}_{vi:03d}.png")))[..., :3]
            gt_dep[vi] = np.load(os.path.join(
                args.renders, f"{item['scene']}_{vi:03d}_depth.npy"))
            gt_msk[vi] = np.load(os.path.join(
                args.renders, f"{item['scene']}_{vi:03d}_mask.npy"))

        # 高斯缓存 (B8/A5 离线分析用)
        means_np = gaussians.means.detach()[0].cpu().numpy()
        opac_np = gaussians.opacities.detach()[0].cpu().numpy()
        cov = gaussians.covariances.detach()[0].cpu().numpy()  # (n,3,3)
        scale_proxy = np.cbrt(np.linalg.det(cov + 1e-12 * np.eye(3)[None]))
        ctx_ext = item["context"]["extrinsics"].cpu().numpy()  # (v,4,4) w2c
        np.savez_compressed(
            os.path.join(cache_dir, f"{item['scene']}.npz"),
            means=means_np.astype(np.float32), opacities=opac_np.astype(np.float32),
            scale_proxy=scale_proxy.astype(np.float32),
            gt20k=v.astype(np.float32), scale_cm=scale_cm,
            ctx_idx=np.array(ctx_idx), tgt_idx=np.array(tgt_idx),
            ctx_extrinsics=ctx_ext.astype(np.float32),
            ply=os.path.join(args.ply_dir, meta["ply"]),
            normalize_center=np.asarray(meta["normalize_center"], dtype=np.float32),
            normalize_scale=np.float32(meta["normalize_scale"]))

        board = {"scene": item["scene"], "n_gauss": int(len(means_np)),
                 "enc_ms": round(enc_ms, 1), "taus": {}}
        flds = [f.name for f in dataclasses.fields(gaussians)]
        opac_flat = gaussians.opacities[0]  # (n,)
        for tau in taus:
            if tau <= 0:
                keep = torch.ones_like(opac_flat, dtype=torch.bool)
            else:
                keep = opac_flat > tau
            n_keep = int(keep.sum().item())
            entry = {"n_kept": n_keep}
            if n_keep >= 100:
                g_tau = dataclasses.replace(
                    gaussians,
                    **{f: getattr(gaussians, f)[:, keep] for f in flds
                       if torch.is_tensor(getattr(gaussians, f))
                       and getattr(gaussians, f).shape[:1] == gaussians.means.shape[:1]})
                with torch.no_grad():
                    color, depth = render(g_tau, batch["target"])
                d_pred = depth[0].cpu().numpy()
                psnrs, ssims, lps, maes = [], [], [], []
                for bi, vi in enumerate(tgt_idx):
                    pred = (color[0, bi].clamp(0, 1).cpu().numpy()
                            .transpose(1, 2, 0) * 255).astype(np.uint8)
                    gt = gt_rgb[vi]
                    mse = np.mean((pred.astype(np.float64) / 255
                                   - gt.astype(np.float64) / 255) ** 2)
                    psnrs.append(10 * np.log10(1.0 / max(mse, 1e-12)))
                    ssims.append(structural_similarity(gt, pred, channel_axis=2))
                    tp = torch.from_numpy(pred).permute(2, 0, 1)[None].cuda() / 255 * 2 - 1
                    tg = torch.from_numpy(gt).permute(2, 0, 1)[None].cuda() / 255 * 2 - 1
                    with torch.no_grad():
                        lps.append(lpips_fn(tp, tg).item())
                    dp = d_pred[bi]
                    m = gt_msk[vi] & np.isfinite(dp) & (dp > 0)
                    if m.sum() >= 100:
                        maes.append(float(np.abs(dp[m] - gt_dep[vi][m]).mean()))
                r = np.linalg.norm(means_np, axis=1)
                m_np = (r < 1.6) & (opac_np > tau if tau > 0 else True)
                entry.update(
                    psnr=round(float(np.mean(psnrs)), 3),
                    ssim=round(float(np.mean(ssims)), 4),
                    lpips=round(float(np.mean(lps)), 4),
                    depth_mae=round(float(np.mean(maes)), 5) if maes else -1,
                    cd_cm=round(unified_cd(means_np[m_np], v, scale_cm), 4))
                # B9: 深度误差热图 (tau=0 与最大 tau, 前几块板)
                if k < args.heatmap_boards and (tau == 0.0 or tau == max(taus[1:])):
                    fig, axes = plt.subplots(1, len(tgt_idx) + 1,
                                             figsize=(3 * (len(tgt_idx) + 1), 3),
                                             dpi=150, squeeze=False)
                    for bi, vi in enumerate(tgt_idx):
                        dp = d_pred[bi]
                        m = gt_msk[vi] & np.isfinite(dp) & (dp > 0)
                        err = np.zeros_like(dp)
                        err[m] = np.abs(dp[m] - gt_dep[vi][m])
                        im = axes[0, bi].imshow(err, cmap="hot", vmin=0, vmax=0.2)
                        axes[0, bi].set_title(f"tau={tau} view={vi}", fontsize=8)
                        axes[0, bi].axis("off")
                        fig.colorbar(im, ax=axes[0, bi], fraction=0.046)
                    axes[0, -1].imshow(gt_rgb[tgt_idx[0]])
                    axes[0, -1].set_title("GT rgb", fontsize=8)
                    axes[0, -1].axis("off")
                    plt.tight_layout()
                    plt.savefig(os.path.join(
                        hm_dir, f"{item['scene'][:24]}_tau{tau}.png"),
                        bbox_inches="tight")
                    plt.close()
            else:
                entry.update(psnr=None, ssim=None, lpips=None,
                             depth_mae=None, cd_cm=-1)
            board["taus"][tau] = entry
        results["per_board"].append(board)
        e0 = board["taus"][0.0]
        print(f"[{k+1}/{n_boards}] {item['scene']} g={board['n_gauss']} "
              f"tau0: psnr={e0.get('psnr')} depth={e0.get('depth_mae')} "
              f"cd={e0.get('cd_cm')}", flush=True)

    json.dump(results, open(os.path.join(
        args.out_dir, f"prune_render_{args.experiment}.json"), "w"), indent=1)
    json.dump(protocol, open(os.path.join(
        args.out_dir, "eval_protocol.json"), "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
