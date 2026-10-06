"""CuneiSplat 几何评测：Chamfer 距离（高斯中心 vs 网格真值）+ 推理计时。

用法:
  python eval_chamfer.py --experiment cunei_geo --ckpt <path.ckpt> \
      --renders /root/autodl-tmp/data/renders_full --ply_dir /root/autodl-tmp/data/full_src

Chamfer 协议（论文 §4.2）:
  预测点 = 高斯中心（世界系，单位包围球归一化）
  真值点 = PLY 顶点子采样（同一归一化）
  双向平均，除以归一化半径（=1），再按 meta.normalize_scale 换算厘米
"""
import argparse, glob, json, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/pixelsplat")


def chamfer(a, b):
    """对称平均最近邻距离（numpy, 暴力分块）。a:(N,3) b:(M,3)"""
    def nn_dist(x, y):
        dmin = np.full(len(x), np.inf)
        for i in range(0, len(y), 8192):
            chunk = y[i : i + 8192]
            dd = np.linalg.norm(x[:, None] - chunk[None], axis=-1).min(1)
            dmin = np.minimum(dmin, dd)
        return dmin
    return 0.5 * (nn_dist(a, b).mean() + nn_dist(b, a).mean())


def build_model(experiment, renders):
    from hydra import compose, initialize
    from omegaconf import OmegaConf

    with initialize(version_base=None, config_path="pixelsplat/config"):
        cfg = compose(config_name="main", overrides=[
            f"+experiment={experiment}",
            f"dataset.roots=[{renders}]",
            "wandb.mode=disabled",
        ])
    # 与 src/main.py 完全一致的构造方式
    from src.global_cfg import set_cfg
    from src.config import load_typed_root_config
    from src.model.model_wrapper import ModelWrapper
    from src.model.encoder import get_encoder
    from src.model.decoder import get_decoder
    from src.loss import get_losses
    from src.misc.step_tracker import StepTracker

    set_cfg(cfg)  # epipolar transformer 启动时读取全局 cfg（main.py 同款注册）
    cfg = load_typed_root_config(cfg)
    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        get_decoder(cfg.model.decoder, cfg.dataset),
        get_losses(cfg.loss),
        StepTracker(),
    )
    return wrapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--renders", required=True)
    ap.add_argument("--ply_dir", required=True)
    ap.add_argument("--max_boards", type=int, default=40)
    ap.add_argument("--n_gt", type=int, default=20000)
    args = ap.parse_args()

    wrapper = build_model(args.experiment, args.rendows if False else args.renders)
    sd = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    state = sd.get("state_dict", sd)
    missing, unexpected = wrapper.load_state_dict(state, strict=False)
    print("missing:", len(missing), "unexpected:", len(unexpected))
    wrapper = wrapper.cuda().eval()

    from src.dataset.dataset_cunei import DatasetCuneiCfg
    from src.dataset.view_sampler.view_sampler_bounded import ViewSamplerBoundedCfg
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
    from src.dataset import DATASETS
    ds = DATASETS["cunei"](cfg, "test", None)
    print("test boards:", len(ds))
    indices = list(range(min(args.max_boards, len(ds))))

    import trimesh
    cds, times = [], []
    for k, idx in enumerate(indices):
        item = ds[idx]
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
            gaussians = wrapper.encoder(
                batch["context"], wrapper.global_step, deterministic=False
            )
        torch.cuda.synchronize(); times.append(time.time() - t0)

        means = gaussians.means.detach()[0].cpu().numpy()
        # 公平子采样：两模型固定同点数（50k），同时大幅加速
        if len(means) > 50000:
            sel = np.random.default_rng(0).choice(
                len(means), 50000, replace=False)
            means = means[sel]
        # 健全性：单位包围球世界系内
        r = np.linalg.norm(means, axis=1)
        keep = r < 1.6
        means = means[keep]
        if len(means) < 100:
            print(f"[{k}] {item['scene']} gaussians too few: {len(means)}")
            continue

        meta = json.load(open(
            glob.glob(os.path.join(args.renders, item["scene"] + "*_000_meta.json"))[0]))
        ply = os.path.join(args.ply_dir, meta["ply"])
        mesh = trimesh.load(ply, force="mesh", process=False)
        mesh.apply_translation(-np.array(meta["normalize_center"]))
        mesh.apply_scale(1.0 / meta["normalize_scale"])
        sel = np.random.default_rng(0).choice(
            len(mesh.vertices), size=min(args.n_gt, len(mesh.vertices)),
            replace=False)
        gt = np.asarray(mesh.vertices)[sel]

        cd = chamfer(means.astype(np.float32), gt.astype(np.float32))
        cm = cd * float(meta["normalize_scale"]) * 0.1  # 原始坐标为毫米: mm->cm
        cds.append(cm)
        print(f"[{k+1}/{len(indices)}] {item['scene']} gauss={len(means)} "
              f"CD_norm={cd:.4f} CD_cm={cm:.3f}cm t={times[-1]*1000:.0f}ms",
              flush=True)

    print("\n===== SUMMARY =====")
    print(f"boards={len(cds)}")
    print(f"Chamfer: mean={np.mean(cds):.3f}cm std={np.std(cds):.3f}cm "
          f"median={np.median(cds):.3f}cm")
    print(f"Inference: mean={np.mean(times)*1000:.0f}ms")
    json.dump({"chamfer_cm": cds, "times_ms": [t * 1000 for t in times]},
              open(f"/root/eval_chamfer_{args.experiment}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
