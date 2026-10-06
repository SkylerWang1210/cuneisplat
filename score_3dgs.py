"""3DGS 数值评测 (cunei 环境; 依赖 scipy/skimage/lpips)。
输入: render_3dgs_views.py 的 exports_3dgs_{n}v/ 目录 + gt20k npz。
输出: eval_3dgs_{n}v.json (CD/PSNR/SSIM/LPIPS/depthMAE/耗时)。
用法: python score_3dgs.py --n_views 2
"""
import argparse, json, os

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity
from scipy.spatial import cKDTree


def cd_cm(pred, gt, scale_cm):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_views", type=int, required=True)
    ap.add_argument("--renders", default="/root/autodl-tmp/data/renders_full")
    ap.add_argument("--out_dir", default="/root/analysis_out")
    args = ap.parse_args()
    base = os.path.join(args.out_dir, f"exports_3dgs_{args.n_views}v")
    gt_dir = os.path.join(args.out_dir, "gt20k")

    import lpips as lpips_mod
    lpips_fn = lpips_mod.LPIPS(net="vgg").cuda()

    manifest = json.load(open(os.path.join(base, "manifest.json")))
    results = []
    for m in manifest:
        sc = m["scene"]
        gz = np.load(os.path.join(gt_dir, f"{sc}.npz"))
        xyz = np.load(os.path.join(base, f"{sc}_xyz.npy"))
        cd = cd_cm(xyz.astype(np.float32), gz["gt20k"], float(gz["scale_cm"]))
        psnrs, ssims, lps, dmaes = [], [], [], []
        for vi in m["targets"]:
            rgb = np.asarray(Image.open(os.path.join(base, f"{sc}_{vi:03d}.png")))[..., :3]
            gtp = np.asarray(Image.open(os.path.join(
                args.renders, f"{sc}_{vi:03d}.png")))[..., :3]
            mse = np.mean((rgb.astype(np.float64) / 255
                           - gtp.astype(np.float64) / 255) ** 2)
            psnrs.append(10 * np.log10(1.0 / max(mse, 1e-12)))
            ssims.append(structural_similarity(gtp, rgb, channel_axis=2))
            tp = torch.from_numpy(rgb).permute(2, 0, 1)[None].cuda() / 255 * 2 - 1
            tg = torch.from_numpy(gtp).permute(2, 0, 1)[None].cuda() / 255 * 2 - 1
            with torch.no_grad():
                lps.append(lpips_fn(tp, tg).item())
            dep = np.load(os.path.join(base, f"{sc}_{vi:03d}_depth.npy"))
            gt_dep = np.load(os.path.join(args.renders, f"{sc}_{vi:03d}_depth.npy"))
            msk = np.load(os.path.join(args.renders, f"{sc}_{vi:03d}_mask.npy"))
            mk = msk & np.isfinite(dep) & (dep > 0)
            if mk.sum() >= 100:
                dmaes.append(float(np.abs(dep[mk] - gt_dep[mk]).mean()))
        results.append(dict(
            scene=sc, cd_cm=round(cd, 4),
            psnr=round(float(np.mean(psnrs)), 3),
            ssim=round(float(np.mean(ssims)), 4),
            lpips=round(float(np.mean(lps)), 4),
            depth_mae=round(float(np.mean(dmaes)), 5) if dmaes else -1,
            n_points=m["n_points"], fit_s=m["fit_s"]))
        print(f"{sc} CD={cd:.3f} PSNR={results[-1]['psnr']} "
              f"LPIPS={results[-1]['lpips']} depth={results[-1]['depth_mae']}",
              flush=True)

    ok = [r for r in results if r["cd_cm"] > 0]
    fits = [r["fit_s"] for r in results if 0 < r["fit_s"] < 7200]
    summary = dict(
        n_views=args.n_views, n=len(results),
        cd_mean=round(float(np.mean([r["cd_cm"] for r in ok])), 4),
        psnr_mean=round(float(np.mean([r["psnr"] for r in results])), 3),
        ssim_mean=round(float(np.mean([r["ssim"] for r in results])), 4),
        lpips_mean=round(float(np.mean([r["lpips"] for r in results])), 4),
        depth_mae_mean=round(float(np.mean(
            [r["depth_mae"] for r in results if r["depth_mae"] > 0])), 5),
        fit_s_mean=round(float(np.mean(fits)), 1) if fits else -1,
        fit_s_std=round(float(np.std(fits)), 1) if fits else -1,
        rows=results)
    out = os.path.join(args.out_dir, f"eval_3dgs_{args.n_views}v.json")
    json.dump(summary, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
    print("DONE")


if __name__ == "__main__":
    main()
