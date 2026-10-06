"""实验 #4：opacity 分层重算（替换论文 2.94/0.24 一组数的可复现版本）。

协议显式：r<1.6 球内；分层带 opacity>0.3 / opacity<0.05 / 不透明度最高十分位；
逐板报告 n_gaussians；两种加权（对象等权[仅 n>=100 的板]与粒子等权）。
输出 out/stratify.json
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

BASE = "/root/autodl-tmp/analysis2"
OUT = f"{BASE}/out"


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {}
    for model in ["cunei_base", "cunei_geo"]:
        rows = []
        for path in sorted(glob.glob(f"{BASE}/caches/gauss_cache_{model}/*.npz")):
            scene = Path(path).stem
            with np.load(path) as z:
                means, opac, gt = z["means"], z["opacities"], z["gt20k"]
                scale_cm = float(z["scale_cm"])
            inside = np.linalg.norm(means, axis=1) < 1.6
            m, o = means[inside], opac[inside]
            tree = cKDTree(gt)
            dist = tree.query(m, k=1)[0] * scale_cm   # 每粒子到表面距离 (cm)
            bands = {
                "gt0.3": o > 0.3,
                "lt0.05": o < 0.05,
                "top_decile": o >= np.quantile(o, 0.9),
                "mid_0.05_0.3": (o >= 0.05) & (o <= 0.3),
            }
            row = {"scene": scene, "n_inside": int(len(m)), "q90_opacity": float(np.quantile(o, 0.9))}
            for bname, bmask in bands.items():
                dsel = dist[bmask]
                row[bname] = {"n": int(bmask.sum()),
                              "mean_cm": float(dsel.mean()) if len(dsel) else None,
                              "median_cm": float(np.median(dsel)) if len(dsel) else None}
            rows.append(row)
        # 两种加权汇总
        summary = {}
        for bname in ["gt0.3", "lt0.05", "top_decile", "mid_0.05_0.3"]:
            valid = [r for r in rows if r[bname]["n"] >= 100]
            summary[bname] = {
                "n_tablets_nonempty": sum(1 for r in rows if r[bname]["n"] > 0),
                "n_tablets_ge100": len(valid),
                "min_n": min((r[bname]["n"] for r in rows if r[bname]["n"] > 0), default=0),
                "object_equal_weight_mean_cm": float(np.mean([r[bname]["mean_cm"] for r in valid])) if valid else None,
                "particle_weighted_mean_cm": float(
                    sum(r[bname]["mean_cm"] * r[bname]["n"] for r in valid) /
                    sum(r[bname]["n"] for r in valid)) if valid else None,
            }
        out[model] = {"per_board": rows, "summary": summary}
        print(f"=== {model} ===")
        for bname, s in summary.items():
            print(f"  {bname:12s} 非空板 {s['n_tablets_nonempty']}/9 (>=100: {s['n_tablets_ge100']}) "
                  f"对象均值={s['object_equal_weight_mean_cm']} 粒子加权={s['particle_weighted_mean_cm']}")
    json.dump(out, open(f"{OUT}/stratify.json", "w"), indent=1)
    print("已写出", f"{OUT}/stratify.json")


if __name__ == "__main__":
    main()
