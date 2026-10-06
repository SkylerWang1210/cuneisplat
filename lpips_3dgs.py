"""SSIM+LPIPS for saved 3DGS renders (在 cunei 环境运行; eval_3dgs_all 之后的补算)。
用法: python lpips_3dgs.py <renders_3dgs_dir> <renders_full_dir> <out_json>
"""
import json, os, sys

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

render_dir, renders, out_json = sys.argv[1], sys.argv[2], sys.argv[3]

import lpips as lpips_mod
lpips_fn = lpips_mod.LPIPS(net="vgg").cuda()

rows = []
for f in sorted(os.listdir(render_dir)):
    if not f.endswith(".png"):
        continue
    sc, idx = f[:-4].rsplit("_", 1)
    gt_path = os.path.join(renders, f)
    if not os.path.exists(gt_path):
        continue
    pred = np.asarray(Image.open(os.path.join(render_dir, f)))[..., :3]
    gt = np.asarray(Image.open(gt_path))[..., :3]
    tp = torch.from_numpy(pred).permute(2, 0, 1)[None].cuda() / 255 * 2 - 1
    tg = torch.from_numpy(gt).permute(2, 0, 1)[None].cuda() / 255 * 2 - 1
    with torch.no_grad():
        lp = lpips_fn(tp, tg).item()
    rows.append(dict(scene=sc, view=int(idx),
                     lpips=round(lp, 4),
                     ssim=round(structural_similarity(gt, pred, channel_axis=2), 4)))

if not rows:
    print("WARN: no render PNGs found in", render_dir)
    json.dump({"n": 0, "rows": []}, open(out_json, "w"))
    sys.exit(0)

import statistics as st
summary = dict(n=len(rows),
               lpips_mean=round(st.mean(r["lpips"] for r in rows), 4),
               ssim_mean=round(st.mean(r["ssim"] for r in rows), 4),
               rows=rows)
json.dump(summary, open(out_json, "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
