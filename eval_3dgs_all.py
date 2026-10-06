"""A4 + P0: 3DGS 基线全量指标 (在 cunei_gs 环境运行)。
关键坐标系事实: 3DGS 的 COLMAP scene 由 meta.json 的 w2c R|t 构建
(归一化渲染世界系), 因此 PLY 高斯坐标已在归一化系中, 与 GT 深度/掩膜同单位。
指标:
  - CD: PLY 高斯中心 vs 网格顶点, 与 eval_chamfer 相同协议 (cap 50k rng0, GT 20k rng0, r<1.6)
  - PSNR/SSIM: 在与 pixelSplat 评测相同的 target 视角渲染 (eval_protocol.json)
  - Depth MAE: override_color=视角空间z 的深度渲染, 与 GT 深度同掩膜 (归一化单位)
  - 每场景拟合 wall-clock (iteration PLY mtime 差分)
用法:
  /root/miniconda3/envs/cunei_gs/bin/python eval_3dgs_all.py \
    --bs_dir /root/autodl-tmp/data/bs_3dgs_2v --n_views 2
"""
import argparse, glob, json, os, sys

import numpy as np
import torch

GS_REPO = "/root/autodl-tmp/proj/gaussian-splatting"
sys.path.insert(0, GS_REPO)

FX = FY = 309.0193
CX = CY = 128.0
W = H = 256


def cd_cm(pred, gt, scale_cm):
    from scipy.spatial import cKDTree
    if len(pred) > 50000:
        sel = np.random.default_rng(0).choice(len(pred), 50000, replace=False)
        pred = pred[sel]
    r = np.linalg.norm(pred, axis=1)
    pred = pred[r < 1.6]
    if len(pred) < 100:
        return -1.0
    t1 = cKDTree(gt); d1, _ = t1.query(pred, k=1)
    t2 = cKDTree(pred); d2, _ = t2.query(gt, k=1)
    return float(0.5 * (d1.mean() + d2.mean()) * scale_cm)


def build_camera(R, t, uid):
    from scene.cameras import Camera
    fov = 2 * np.arctan(CX / FX)
    img = torch.zeros(H, W, 3)
    return Camera(colmap_id=uid, R=R.astype(np.float64), T=t.astype(np.float64),
                  FoVx=fov, FoVy=fov, image=img, gt_alpha_mask=None,
                  image_name=f"cam{uid}", uid=uid)


def render_safe(render_fn, cam, gm, pipe, bg, override=None):
    try:
        return render_fn(cam, gm, pipe, bg, override_color=override)
    except TypeError:
        return render_fn(cam, gm, pipe, bg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs_dir", required=True)
    ap.add_argument("--n_views", type=int, required=True)
    ap.add_argument("--renders", default="/root/autodl-tmp/data/renders_full")
    ap.add_argument("--ply_dir", default="/root/autodl-tmp/data/full_src")
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--protocol", default="/root/analysis_out/eval_protocol.json")
    ap.add_argument("--out_dir", default="/root/analysis_out")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    png_dir = os.path.join(args.out_dir, f"renders_3dgs_{args.n_views}v")
    os.makedirs(png_dir, exist_ok=True)

    from PIL import Image
    from argparse import Namespace
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel

    protocol = json.load(open(args.protocol)) if os.path.exists(args.protocol) else {}
    scene_dirs = sorted(glob.glob(os.path.join(args.bs_dir, f"*_{args.n_views}v")))
    print(f"3DGS scenes ({args.n_views}v): {len(scene_dirs)}", flush=True)

    mtimes = []
    for sd in scene_dirs:
        ply_p = os.path.join(sd, "model", "point_cloud",
                             f"iteration_{args.iters}", "point_cloud.ply")
        mtimes.append(os.path.getmtime(ply_p) if os.path.exists(ply_p) else -1)
    order = np.argsort(mtimes)
    durations = {}
    for oi in range(1, len(order)):
        prev_t = mtimes[order[oi - 1]]
        if prev_t > 0:
            durations[scene_dirs[order[oi]]] = mtimes[order[oi]] - prev_t
    dur_list = [v for v in durations.values() if 0 < v < 7200]

    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    bg = torch.zeros(3, device="cuda")
    results = []
    for k, sd in enumerate(scene_dirs):
        sc = os.path.basename(sd).replace(f"_{args.n_views}v", "")
        ply_p = os.path.join(sd, "model", "point_cloud",
                             f"iteration_{args.iters}", "point_cloud.ply")
        if not os.path.exists(ply_p):
            print(f"[{k+1}] {sc} MISSING ply", flush=True)
            continue
        meta = json.load(open(os.path.join(args.renders, f"{sc}_000_meta.json")))
        scale_cm = float(meta["normalize_scale"]) * 0.1

        gm = GaussianModel(sh_degree=3)
        gm.load_ply(ply_p)
        xyz = gm.get_xyz.detach().cpu().numpy().astype(np.float32)  # 已在归一化系

        gz = np.load(os.path.join(args.out_dir, "gt20k", f"{sc}.npz"))
        cd = cd_cm(xyz, gz["gt20k"], float(gz["scale_cm"]))

        tgt_idx = protocol.get(sc, {}).get("target", [12, 13])
        views = meta["views"]
        psnrs, dmaes = [], []
        for vi in tgt_idx:
            vw = next(x for x in views if x["id"] == vi)
            R = np.asarray(vw["R"]); t = np.asarray(vw["t"])
            cam = build_camera(R, t, vi)
            z_rgb = torch.zeros(len(xyz), 3, device="cuda")
            with torch.no_grad():
                # 深度渲染: override_color = 视角空间 z (归一化单位)
                z_view = (torch.from_numpy(xyz).cuda() @ torch.from_numpy(
                    R.astype(np.float32)).cuda().T
                    + torch.from_numpy(t.astype(np.float32)).cuda())[:, 2]
                z_rgb[:, :] = z_view[:, None]
                out = render_safe(render, cam, gm, pipe, bg, override=z_rgb)
                depth = out["render"].mean(0).cpu().numpy()
                # 彩色渲染
                out2 = render_safe(render, cam, gm, pipe, bg)
                rgb = (out2["render"].clamp(0, 1).cpu().numpy()
                       .transpose(1, 2, 0) * 255).astype(np.uint8)
            Image.fromarray(rgb).save(os.path.join(png_dir, f"{sc}_{vi:03d}.png"))

            gtp = np.asarray(Image.open(os.path.join(
                args.renders, f"{sc}_{vi:03d}.png")))[..., :3]
            mse = np.mean((rgb.astype(np.float64) / 255
                           - gtp.astype(np.float64) / 255) ** 2)
            psnrs.append(10 * np.log10(1.0 / max(mse, 1e-12)))
            gt_dep = np.load(os.path.join(args.renders, f"{sc}_{vi:03d}_depth.npy"))
            msk = np.load(os.path.join(args.renders, f"{sc}_{vi:03d}_mask.npy"))
            m = msk & np.isfinite(depth) & (depth > 0)
            if m.sum() >= 100:
                dmaes.append(float(np.abs(depth[m] - gt_dep[m]).mean()))
        results.append(dict(
            scene=sc, cd_cm=round(cd, 4),
            psnr=round(float(np.mean(psnrs)), 3),
            ssim=None,
            depth_mae=round(float(np.mean(dmaes)), 5) if dmaes else -1,
            n_points=len(xyz), fit_s=round(durations.get(sd, -1), 1)))
        print(f"[{k+1}/{len(scene_dirs)}] {sc} CD={cd:.3f} "
              f"PSNR={results[-1]['psnr']} depth={results[-1]['depth_mae']} "
              f"n={len(xyz)} fit={durations.get(sd, -1):.0f}s", flush=True)

    ok = [r for r in results if r["cd_cm"] > 0]
    dmae_ok = [r["depth_mae"] for r in results if r["depth_mae"] > 0]
    summary = dict(
        n_views=args.n_views, n=len(results),
        cd_mean=round(float(np.mean([r["cd_cm"] for r in ok])), 4),
        psnr_mean=round(float(np.mean([r["psnr"] for r in results])), 3),
        ssim_mean=None,
        depth_mae_mean=round(float(np.mean(dmae_ok)), 5) if dmae_ok else -1,
        fit_s_mean=round(float(np.mean(dur_list)), 1) if dur_list else -1,
        fit_s_std=round(float(np.std(dur_list)), 1) if dur_list else -1,
        rows=results)
    out = os.path.join(args.out_dir, f"eval_3dgs_{args.n_views}v.json")
    json.dump(summary, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
    print("DONE")


if __name__ == "__main__":
    main()
