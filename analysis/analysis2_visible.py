"""实验 #3：可见面变体修复重评。

原 bug：offline_robustness.py 把缓存中的 c2w 直接当 w2c 投影。
修复：对 ctx_extrinsics 求逆得 w2c，再投影。相机内参按渲染协议重建
（45° vFOV @256px：f = 128/tan(22.5°) ≈ 309.0193，主点 128,128）。
可见判定：|z_vertex − GT深度图[u,v]| < tol 且 GT 掩膜为真。
CD：可见 GT 顶点（子采样20k）vs 球内预测点（τ=0 / τ=0.2 两档）。
输出 out/visible_fixed.json
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

BASE = "/root/autodl-tmp/analysis2"
OUT = f"{BASE}/out"
TOL = 0.03
F = 128.0 / np.tan(np.pi / 8)   # 309.0193
CX = CY = 128.0


def visible_gt(scene, bd, renders_dir):
    """返回在任一 context 视角可见的 GT 顶点掩码（用修复后的 w2c）。"""
    ply = bd["ply"]  # npz 里存的是服务器绝对路径
    mesh = trimesh.load(str(ply), force="mesh", process=False)
    v = (np.asarray(mesh.vertices, dtype=np.float32) - bd["normalize_center"]) / float(bd["normalize_scale"])
    visible = np.zeros(len(v), dtype=bool)
    for ci, vi in enumerate(bd["ctx_idx"].tolist()):
        c2w = bd["ctx_extrinsics"][ci]
        w2c = np.linalg.inv(c2w)
        R, t = w2c[:3, :3], w2c[:3, 3]
        p = v @ R.T + t
        z = p[:, 2]
        good = z > 0
        u = np.round(F * p[:, 0] / np.where(good, z, 1) + CX).astype(int)
        w = np.round(F * p[:, 1] / np.where(good, z, 1) + CY).astype(int)
        inb = good & (u >= 0) & (u < 256) & (w >= 0) & (w < 256)
        try:
            dep = np.load(os.path.join(renders_dir, f"{scene}_{vi:03d}_depth.npy"))
            msk = np.load(os.path.join(renders_dir, f"{scene}_{vi:03d}_mask.npy"))
            ui, wi = u[inb], w[inb]
            d_ok = (np.abs(z[inb] - dep[wi, ui]) < TOL) & msk[wi, ui]
            visible[inb] |= d_ok
        except FileNotFoundError:
            continue
    return v, visible


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {}
    for model in ["cunei_base", "cunei_geo"]:
        out[model] = {}
        for path in sorted(glob.glob(f"{BASE}/caches/gauss_cache_{model}/*.npz")):
            scene = Path(path).stem
            with np.load(path) as z:
                bd = {k: z[k] for k in z.files}
            v, visible = visible_gt(scene, bd, "/root/autodl-tmp/data/renders_full")
            rng = np.random.default_rng(42)
            gt_vis = v[visible]
            if len(gt_vis) > 20000:
                gt_vis = gt_vis[rng.choice(len(gt_vis), 20000, replace=False)]
            means, opac = bd["means"], bd["opacities"]
            scale_cm = float(bd["scale_cm"])
            inside = np.linalg.norm(means, axis=1) < 1.6
            row = {"n_gt_visible": int(visible.sum())}
            for tau in [0.0, 0.2]:
                keep = inside & (opac > tau if tau > 0 else True)
                pred = means[keep]
                if len(pred) < 100 or len(gt_vis) < 100:
                    row[f"tau{tau}"] = {"n_pred": int(len(pred)), "invalid": True}
                    continue
                if len(pred) > 50000:
                    pred = pred[np.random.default_rng(0).choice(len(pred), 50000, replace=False)]
                d1 = cKDTree(gt_vis).query(pred, k=1)[0]
                d2 = cKDTree(pred).query(gt_vis, k=1)[0]
                row[f"tau{tau}"] = {"n_pred": int(len(pred)),
                                    "cd_cm": float(0.5 * (d1.mean() + d2.mean()) * scale_cm)}
            out[model][scene] = row
            print(f"[{model}] {scene}: 可见GT={row['n_gt_visible']} "
                  f"τ0={out[model][scene].get('tau0.0', {}).get('cd_cm', 'invalid')} "
                  f"τ0.2={out[model][scene].get('tau0.2', {}).get('cd_cm', 'invalid')}", flush=True)
        valid = [r for r in out[model].values() if r.get("tau0.2", {}).get("cd_cm")]
        if valid:
            print(f"[{model}] 有效板 {len(valid)}/9, τ=0.2 可见面CD均值 "
                  f"{np.mean([r['tau0.2']['cd_cm'] for r in valid]):.4f} cm", flush=True)
    json.dump(out, open(f"{OUT}/visible_fixed.json", "w"), indent=1)
    print("已写出", f"{OUT}/visible_fixed.json", flush=True)


if __name__ == "__main__":
    main()
