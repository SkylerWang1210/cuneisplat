"""λ 强度扫描汇总：E seed1 (w=0.5) / seed2 (w=0.5) / λ=1.0，
测试 9 板 + 验证 22 板，逐 τ 聚合 + 配对 Wilcoxon。
输出 data/analysis2_results/esurf_sweep_summary.json
"""
import json

import numpy as np
from scipy.stats import wilcoxon

BASE = "data/analysis2_results"
FILES = {
    "test": {
        "s1": f"{BASE}/prune_render_cunei_geo_surf.json",
        "s2": f"{BASE}/esurf_s2.json",
        "l10": f"{BASE}/esurf_l10.json",
    },
    "val": {
        "s1": f"{BASE}/esurf_val.json",
        "s2": f"{BASE}/esurf_s2_val.json",
        "l10": f"{BASE}/esurf_l10_val.json",
    },
}
TAUS = ["0.0", "0.05", "0.1", "0.2", "0.3"]


def load(path):
    d = json.load(open(path))
    pb = d["per_board"]
    return {b["scene"]: b["taus"] for b in pb}


def metric(taus_by_scene, tau, key):
    return np.array([t[tau][key] for t in taus_by_scene.values()])


out = {}
for split, models in FILES.items():
    data = {k: load(p) for k, p in models.items()}
    scenes = sorted(set.intersection(*(set(d) for d in data.values())))
    out[split] = {"n": len(scenes)}
    for tau in TAUS:
        row = {}
        for k in ["s1", "s2", "l10"]:
            row[k] = {
                m: float(np.mean(metric(data[k], tau, m)))
                for m in ["cd_cm", "depth_mae", "psnr", "lpips"]
            }
        # 配对：λ1.0 vs seed1
        a = metric(data["l10"], tau, "cd_cm")
        b = metric(data["s1"], tau, "cd_cm")
        if (a - b != 0).any():
            w = wilcoxon(a, b)
            row["l10_vs_s1_cd"] = {
                "mean_diff": float(np.mean(a - b)),
                "wins_l10": int((a < b).sum()),
                "p": float(w.pvalue),
            }
        out[split][tau] = row

json.dump(out, open(f"{BASE}/esurf_sweep_summary.json", "w"), indent=1)

for split in ["test", "val"]:
    print(f"===== {split} (n={out[split]['n']}) =====")
    print(f"{'τ':>5} | {'s1 CD':>7} {'s2 CD':>7} {'λ1.0 CD':>7} | {'s1 dMAE':>8} {'λ1.0 dMAE':>9} | {'λ1.0 vs s1':>22}")
    for tau in TAUS:
        r = out[split][tau]
        cmp_ = r.get("l10_vs_s1_cd")
        cmp_s = f"{cmp_['mean_diff']:+.3f} {cmp_['wins_l10']}/{out[split]['n']} p={cmp_['p']:.3f}" if cmp_ else ""
        print(f"{tau:>5} | {r['s1']['cd_cm']:7.4f} {r['s2']['cd_cm']:7.4f} {r['l10']['cd_cm']:7.4f} | {r['s1']['depth_mae']:8.4f} {r['l10']['depth_mae']:9.4f} | {cmp_s:>22}")
    r5 = out[split]["0.05"]
    print(f"      PSNR τ0.05: s1={r5['s1']['psnr']:.2f} s2={r5['s2']['psnr']:.2f} λ1.0={r5['l10']['psnr']:.2f}")
