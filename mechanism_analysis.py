"""离线机制分析脚本：散布可视化 + 全板统计 + 配对检验 + 剪枝对照
用途：生成 Fig 3（机制图）、Table 2 配对统计、MC1 剪枝对照
"""
import argparse, glob, json, os, sys

import numpy as np
sys.path.insert(0, "/root/autodl-tmp/proj")
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
    encoder, ev = get_encoder(cfg.model.encoder)
    return ModelWrapper(cfg.optimizer, cfg.test, cfg.train, encoder, ev,
                        get_decoder(cfg.model.decoder, cfg.dataset),
                        get_losses(cfg.loss), StepTracker())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--renders", default="/root/autodl-tmp/data/renders_full")
    ap.add_argument("--ply_dir", default="/root/autodl-tmp/data/full_src")
    ap.add_argument("--out_dir", default="/root/analysis_out")
    ap.add_argument("--prune_thresholds", default="0.1,0.2,0.3,0.5")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    wrapper = build_model(args.experiment, args.renders)
    sd = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    wrapper.load_state_dict(sd.get("state_dict", sd), strict=True)
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
    print(f"test boards: {len(ds)}", flush=True)

    import trimesh
    boards_data = []
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
        means = gaussians.means.detach()[0].cpu().numpy()  # (n, 3)
        opac = gaussians.opacities.detach()[0].cpu().numpy()  # (n,)
        # 真值网格
        meta = json.load(open(glob.glob(os.path.join(
            args.renders, item["scene"] + "*_000_meta.json"))[0]))
        mesh = trimesh.load(os.path.join(args.ply_dir, meta["ply"]),
                            force="mesh", process=False)
        v = np.asarray(mesh.vertices, dtype=np.float32)
        v = (v - np.asarray(meta["normalize_center"])) / float(meta["normalize_scale"])
        rng = np.random.default_rng(42)
        if len(v) > 20000:
            v = v[rng.choice(len(v), 20000, replace=False)]

        boards_data.append(dict(
            scene=item["scene"], means=means, opacities=opac,
            gt_vertices=v, scale=float(meta["normalize_scale"])))
        print(f"[{k+1}/{len(ds)}] {item['scene']} gauss={len(means)}", flush=True)

    # ====== 全板统计 ======
    print("\n===== Per-board statistics =====")
    for bd in boards_data:
        means, opac, gt = bd["means"], bd["opacities"], bd["gt_vertices"]
        r = np.linalg.norm(means, axis=1)
        inside = means[r < 1.6]
        inside_opac = opac[r < 1.6]
        # 沿射线散布度量：每板高斯标准差
        spread = inside.std(axis=0).mean() if len(inside) > 0 else 0
        # 最近表面距离
        if len(inside) > 0:
            from scipy.spatial import cKDTree
            tree = cKDTree(gt)
            dists, _ = tree.query(inside, k=1)
            mean_dist = dists.mean()
            median_dist = np.median(dists)
        else:
            mean_dist = median_dist = -1
        print(f"  {bd['scene'][:30]}: inside={len(inside)} "
              f"spread={spread:.4f} dist_mean={mean_dist:.4f} "
              f"dist_med={median_dist:.4f}")

    # ====== 剪枝对照实验（MC1）======
    print("\n===== Pruning control (opacity threshold sweep) =====")
    thresholds = [float(x) for x in args.prune_thresholds.split(",")] + [0.0]
    for tau in thresholds:
        cds = []
        for bd in boards_data:
            means, opac, gt = bd["means"], bd["opacities"], bd["gt_vertices"]
            r = np.linalg.norm(means, axis=1)
            if tau > 0:
                mask = (r < 1.6) & (opac > tau)
            else:
                mask = r < 1.6
            pts = means[mask]
            if len(pts) < 100:
                cds.append(-1); continue
            from scipy.spatial import cKDTree
            tree = cKDTree(gt)
            d1, _ = tree.query(pts, k=1)
            tree2 = cKDTree(pts)
            d2, _ = tree2.query(gt, k=1)
            cd = 0.5 * (d1.mean() + d2.mean()) * bd["scale"] * 0.1  # cm
            cds.append(cd)
        valid = [c for c in cds if c > 0]
        if valid:
            print(f"  tau={tau:.1f}: N={len(valid)} CD_mean={np.mean(valid):.4f}cm "
                  f"CD_med={np.median(valid):.4f}cm")
        else:
            print(f"  tau={tau:.1f}: no valid boards")

    # ====== 散布可视化（Fig 3 素材）======
    print("\n===== Generating scatter visualization =====")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n_boards = min(4, len(boards_data))
    fig, axes = plt.subplots(2, n_boards, figsize=(3.5 * n_boards, 6),
                             dpi=200, squeeze=False)
    for i in range(n_boards):
        bd = boards_data[i]
        means, gt = bd["means"], bd["gt_vertices"]
        r = np.linalg.norm(means, axis=1)
        inside = means[r < 1.6]
        if len(inside) > 8000:
            sel = np.random.default_rng(0).choice(len(inside), 8000, replace=False)
            inside = inside[sel]
        axes[0, i].scatter(gt[:, 2], gt[:, 0], s=0.1, c="gray", alpha=0.3)
        axes[0, i].scatter(inside[:, 2], inside[:, 0], s=0.1, c="blue", alpha=0.5)
        axes[0, i].set_title(bd["scene"][:20], fontsize=7)
        axes[0, i].set_xlabel("Z", fontsize=6); axes[0, i].set_ylabel("X", fontsize=6)
        axes[0, i].tick_params(labelsize=5)
        axes[1, i].hist(inside[:, 2], bins=50, alpha=0.5, color="blue",
                        label="Gaussians")
        axes[1, i].hist(gt[:, 2], bins=50, alpha=0.5, color="gray",
                        label="GT surface")
        axes[1, i].set_xlabel("Z coordinate", fontsize=6)
        axes[1, i].legend(fontsize=5)
        axes[1, i].tick_params(labelsize=5)
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, f"scatter_{args.experiment}.png"),
                dpi=200, bbox_inches="tight")
    print(f"  saved scatter_{args.experiment}.png")

    # ====== 配对统计（Wilcoxon / sign test）======
    print("\n===== Paired statistics (this model vs baseline placeholder) =====")
    # 输出全板 per-board 数值供后续配对
    per_board = []
    for bd in boards_data:
        means, gt = bd["means"], bd["gt_vertices"]
        r = np.linalg.norm(means, axis=1)
        inside = means[r < 1.6]
        if len(inside) < 100:
            per_board.append(-1); continue
        from scipy.spatial import cKDTree
        tree = cKDTree(gt)
        d1, _ = tree.query(inside, k=1)
        tree2 = cKDTree(inside)
        d2, _ = tree2.query(gt, k=1)
        cd = 0.5 * (d1.mean() + d2.mean()) * bd["scale"] * 0.1
        per_board.append(round(cd, 4))
    json.dump(per_board, open(os.path.join(args.out_dir,
             f"chamfer_per_board_{args.experiment}.json"), "w"))
    print(f"  per-board CD (cm): {per_board}")
    print("DONE")


if __name__ == "__main__":
    main()
