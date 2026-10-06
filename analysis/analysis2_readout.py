"""实验 #7/#8/#9 三合一：读出方式敏感性 + 采样方差 + 置信可靠性。

对两个 seed1 检查点、9 块测试板：
  A) 确定性读出（top-k）：opacity 结构 + τ 网格 CD（与随机读出的缓存对照）
  B) K=8 次随机采样：每次 draw 的 τ=0/0.2 CD + 目标视角渲染深度；
     逐像素 std 与 |err| 的 Spearman 相关（可靠性）；
     K 次深度均值的 MAE vs 单次 MAE（平均是否变好）。
输出 out/readout.json
"""
import glob
import json
import os
import sys

sys.path.insert(0, "/root/autodl-tmp/proj")
sys.path.insert(0, "/root/autodl-tmp/proj/pixelsplat")
os.chdir("/root/autodl-tmp/proj")

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

BASE = "/root/autodl-tmp/analysis2"
OUT = f"{BASE}/out"
TAUS = [0.0, 0.05, 0.1, 0.2, 0.3]
K = 8
CKPTS = {
    "cunei_base": "/root/autodl-tmp/proj/pixelsplat/outputs/2026-10-01/03-27-14/checkpoints/epoch=465-step=20000.ckpt",
    "cunei_geo": "/root/autodl-tmp/proj/pixelsplat/outputs/2026-10-01/11-50-30/checkpoints/epoch=465-step=20000.ckpt",
}


def build(experiment):
    from hydra import compose, initialize
    with initialize(version_base=None, config_path="pixelsplat/config"):
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
    encoder, ev = get_encoder(cfg.model.encoder)
    return ModelWrapper(cfg.optimizer, cfg.test, cfg.train, encoder, ev,
                        get_decoder(cfg.model.decoder, cfg.dataset),
                        get_losses(cfg.loss), StepTracker())


def make_ds():
    from src.dataset.dataset_cunei import DatasetCuneiCfg
    from src.dataset.view_sampler.view_sampler_bounded import ViewSamplerBoundedCfg
    from src.dataset import DATASETS
    from pathlib import Path
    vs = ViewSamplerBoundedCfg(name="bounded", num_context_views=2, num_target_views=2,
        min_distance_between_context_views=2, max_distance_between_context_views=8,
        min_distance_to_context_views=1, warm_up_steps=0,
        initial_min_distance_between_context_views=2,
        initial_max_distance_between_context_views=8)
    cfg = DatasetCuneiCfg(name="cunei", roots=[Path("/root/autodl-tmp/data/renders_full")],
        view_sampler=vs, image_shape=[256, 256], background_color=[0, 0, 0],
        cameras_are_circular=True, overfit_to_scene=None, near=0.8, far=5.0,
        context_gap=4, num_target=2)
    return DATASETS["cunei"](cfg, "test", None)


def calc_cd(means, opac, gt20k, scale_cm, tau):
    keep = np.linalg.norm(means, axis=1) < 1.6
    if tau > 0:
        keep &= opac > tau
    pred = means[keep]
    if len(pred) < 100:
        return -1.0
    if len(pred) > 50000:
        pred = pred[np.random.default_rng(0).choice(len(pred), 50000, replace=False)]
    d1 = cKDTree(gt20k).query(pred, k=1)[0]
    d2 = cKDTree(pred).query(gt20k, k=1)[0]
    return float(0.5 * (d1.mean() + d2.mean()) * scale_cm)


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = make_ds()
    print("test boards:", len(ds), flush=True)
    out = {}
    for model, ckpt_path in CKPTS.items():
        wrapper = build(model)
        sd = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        wrapper.load_state_dict(sd.get("state_dict", sd), strict=True)
        wrapper = wrapper.cuda().eval()
        out[model] = {}
        for k in range(len(ds)):
            item = ds[k]
            scene = item["scene"]
            batch = {
                "target": {kk: vv[None].cuda() for kk, vv in item["target"].items() if torch.is_tensor(vv)},
                "context": {kk: vv[None].cuda() for kk, vv in item["context"].items() if torch.is_tensor(vv)},
                "scene": [scene],
            }
            with torch.no_grad():
                if hasattr(wrapper, "data_shim"):
                    batch = wrapper.data_shim(batch)
                # GT 参照（来自缓存 npz，协议一致）
                with np.load(f"{BASE}/caches/gauss_cache_{model}/{scene}.npz") as z:
                    gt20k, scale_cm = z["gt20k"], float(z["scale_cm"])
                # A) 确定性读出
                torch.manual_seed(0)
                g_det = wrapper.encoder(batch["context"], wrapper.global_step, deterministic=True)
                m_det, o_det = g_det.means[0].cpu().numpy(), g_det.opacities[0].cpu().numpy()
                det_stats = {"n": int(len(m_det)),
                             "p90_opac": float(np.quantile(o_det, 0.9)),
                             "frac_lt005": float((o_det < 0.05).mean()),
                             "cd": {str(t): calc_cd(m_det, o_det, gt20k, scale_cm, t) for t in TAUS}}
                # B) K 次随机 draw
                draws_cd02, draws_cd00, depth_stack = [], [], []
                tgt_ext = batch["target"]["extrinsics"]
                tgt_int = batch["target"]["intrinsics"]
                tgt_near = batch["target"].get("near")
                tgt_far = batch["target"].get("far")
                for d in range(K):
                    torch.manual_seed(1000 + d)
                    g = wrapper.encoder(batch["context"], wrapper.global_step, deterministic=False)
                    m, o = g.means[0].cpu().numpy(), g.opacities[0].cpu().numpy()
                    draws_cd00.append(calc_cd(m, o, gt20k, scale_cm, 0.0))
                    draws_cd02.append(calc_cd(m, o, gt20k, scale_cm, 0.2))
                    try:
                        dep = wrapper.decoder.render_depth(
                            g, tgt_ext, tgt_int, tgt_near, tgt_far, (256, 256), mode="depth")
                        depth_stack.append(dep[0].cpu().numpy())
                    except Exception as ex:
                        print(f"  render_depth 失败({type(ex).__name__})，跳过深度方差", flush=True)
                        depth_stack = []
                        break
                # 目标 GT 深度与掩膜
                vi0 = int(batch["target"]["index"][0][0].item())
                try:
                    gt_dep = np.load(f"/root/autodl-tmp/data/renders_full/{scene}_{vi0:03d}_depth.npy")
                    gt_msk = np.load(f"/root/autodl-tmp/data/renders_full/{scene}_{vi0:03d}_mask.npy").astype(bool)
                except FileNotFoundError:
                    gt_dep = gt_msk = None
                var_stats = {}
                if depth_stack and gt_dep is not None:
                    st = np.stack(depth_stack)[:, 0]        # (K, h, w) 第一目标视角
                    mean_d, std_d = st.mean(0), st.std(0)
                    err = np.abs(mean_d - gt_dep)
                    m = gt_msk & np.isfinite(err) & (gt_dep > 0)
                    if m.sum() > 100:
                        rho, pv = spearmanr(std_d[m], err[m])
                        var_stats = {
                            "k": K,
                            "single_mae": float(np.mean([np.abs(s - gt_dep)[m].mean() for s in st])),
                            "mean_mae": float(err[m].mean()),
                            "spearman_std_err": float(rho), "spearman_p": float(pv),
                            "n_px": int(m.sum()),
                        }
                out[model][scene] = {
                    "det": det_stats,
                    "stoch_cd00": draws_cd00, "stoch_cd02": draws_cd02,
                    "stoch_cd02_mean": float(np.mean(draws_cd02)),
                    "variance": var_stats,
                }
                print(f"[{model}] {scene}: det τ0.2 CD={det_stats['cd']['0.2']:.4f} "
                      f"stoch τ0.2 均值={np.mean(draws_cd02):.4f} "
                      f"方差可靠性 rho={var_stats.get('spearman_std_err', float('nan')):.3f}", flush=True)
        # 汇总
        agg = {
            "det_cd02_mean": float(np.mean([v["det"]["cd"]["0.2"] for v in out[model].values() if v["det"]["cd"]["0.2"] > 0])),
            "stoch_cd02_mean": float(np.mean([v["stoch_cd02_mean"] for v in out[model].values()])),
            "det_p90": float(np.mean([v["det"]["p90_opac"] for v in out[model].values()])),
            "mean_spearman": float(np.nanmean([v["variance"].get("spearman_std_err", np.nan) for v in out[model].values()])),
        }
        out[model]["_summary"] = agg
        print(f"[{model}] 汇总: det τ0.2={agg['det_cd02_mean']:.4f} stoch τ0.2={agg['stoch_cd02_mean']:.4f} "
              f"方差-误差相关均值={agg['mean_spearman']:.3f}", flush=True)
    json.dump(out, open(f"{OUT}/readout.json", "w"), indent=1)
    print("已写出", f"{OUT}/readout.json", flush=True)


if __name__ == "__main__":
    main()
