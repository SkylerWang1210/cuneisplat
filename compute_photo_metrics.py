"""直接计算光度指标：outputs/test/<exp>/<scene>/color/*.png vs GT 渲染。
用法: python compute_photo_metrics.py <test_dir> <renders_dir> <out_json>
"""
import json, os, sys

import numpy as np
import torch
from PIL import Image

test_dir, renders, out_json = sys.argv[1], sys.argv[2], sys.argv[3]

# SSIM 用 skimage；LPIPS 用 env 内置
from skimage.metrics import structural_similarity
lpips_fn = None

def get_lpips():
    global lpips_fn
    if lpips_fn is None:
        import lpips as lpips_mod
        lpips_fn = lpips_mod.LPIPS(net="vgg").cuda()
    return lpips_fn

scenes = sorted(os.listdir(test_dir))
rows = []
for sc in scenes:
    color_dir = os.path.join(test_dir, sc, "color")
    if not os.path.isdir(color_dir):
        continue
    for f in sorted(os.listdir(color_dir)):
        idx = int(os.path.splitext(f)[0])
        gt_path = os.path.join(renders, f"{sc}_{idx:03d}.png")
        if not os.path.exists(gt_path):
            continue
        pred = np.asarray(Image.open(os.path.join(color_dir, f)))[..., :3]
        gt = np.asarray(Image.open(gt_path))[..., :3]
        if pred.shape != gt.shape:
            continue
        mse = np.mean((pred.astype(np.float64) / 255 - gt.astype(np.float64) / 255) ** 2)
        psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
        ssim = structural_similarity(gt, pred, channel_axis=2)
        t_pred = torch.from_numpy(pred).permute(2, 0, 1)[None].cuda() / 255 * 2 - 1
        t_gt = torch.from_numpy(gt).permute(2, 0, 1)[None].cuda() / 255 * 2 - 1
        with torch.no_grad():
            lp = get_lpips()(t_pred, t_gt).item()
        rows.append(dict(scene=sc, view=idx, psnr=round(psnr, 3),
                         ssim=round(ssim, 4), lpips=round(lp, 4)))

import statistics as st
summary = dict(
    n=len(rows),
    psnr_mean=round(st.mean(r["psnr"] for r in rows), 3),
    psnr_std=round(st.stdev(r["psnr"] for r in rows), 3),
    ssim_mean=round(st.mean(r["ssim"] for r in rows), 4),
    lpips_mean=round(st.mean(r["lpips"] for r in rows), 4),
    rows=rows)
json.dump(summary, open(out_json, "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
