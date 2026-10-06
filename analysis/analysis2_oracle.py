"""实验 #2：oracle 天花板（GT 辅助选点的可达参考）。

以每板 opacity>τ=0.2 的保留数为预算 k，比较五种选择器：
  opacity(τ0.2) / top-k opacity / oracle(真值表面距离最小 k) / random-k / scale-top-k。
指标：双向距离分别报告 + 对称 CD（cm）+ F-score@δ + 覆盖率（GT 侧）。
选点准则与评估指标分离：oracle 按"逐点到 GT 最近距离"选点（不看 CD 评估量）。
输出 out/oracle.json
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

BASE = "/root/autodl-tmp/analysis2"
OUT = f"{BASE}/out"
DELTAS = [0.25, 0.5, 1.0]  # cm


def evaluate(pred, gt20k, scale_cm):
    if len(pred) > 50000:
        pred = pred[np.random.default_rng(0).choice(len(pred), 50000, replace=False)]
    d1 = cKDTree(gt20k).query(pred, k=1)[0]      # pred -> GT
    d2 = cKDTree(pred).query(gt20k, k=1)[0]      # GT -> pred
    d1cm, d2cm = d1 * scale_cm, d2 * scale_cm
    res = dict(n_pred=int(len(pred)),
               d_pred2gt=float(d1cm.mean()), d_gt2pred=float(d2cm.mean()),
               cd_sym=float(0.5 * (d1cm.mean() + d2cm.mean())))
    for dl in DELTAS:
        res[f"f{dl}"] = float(2 * (d1cm < dl).mean() * (d2cm < dl).mean()
                              / max((d1cm < dl).mean() + (d2cm < dl).mean(), 1e-9))
        res[f"cov_gt{dl}"] = float((d2cm < dl).mean())
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {}
    for model in ["cunei_base", "cunei_geo"]:
        out[model] = {}
        for path in sorted(glob.glob(f"{BASE}/caches/gauss_cache_{model}/*.npz")):
            scene = Path(path).stem
            with np.load(path) as z:
                means, opac, gt = z["means"], z["opacities"], z["gt20k"]
                scale_cm = float(z["scale_cm"])
                scale_proxy = z["scale_proxy"]
            inside = np.linalg.norm(means, axis=1) < 1.6
            m, o, sp = means[inside], opac[inside], scale_proxy[inside]
            k = int((o > 0.2).sum())
            if k < 100:
                out[model][scene] = {"budget_k": k, "invalid": "opacity budget <100"}
                continue
            # 五种选择器
            sels = {
                "opacity_tau02": m[o > 0.2],
                "topk_opacity": m[np.argsort(-o)[:k]],
            }
            dist_all = cKDTree(gt).query(m, k=1)[0]      # oracle 选点准则：真值最近距离
            sels["oracle_topk"] = m[np.argsort(dist_all)[:k]]
            rng = np.random.default_rng(0)
            sels["random_k"] = m[rng.choice(len(m), k, replace=False)]
            sels["scale_topk"] = m[np.argsort(-sp)[:k]]
            out[model][scene] = {"budget_k": k,
                                 **{name: evaluate(p, gt, scale_cm)
                                    for name, p in sels.items()}}
        # 汇总
        agg = {}
        for name in ["opacity_tau02", "topk_opacity", "oracle_topk", "random_k", "scale_topk"]:
            boards = [v[name] for v in out[model].values() if name in v]
            if boards:
                agg[name] = {mkey: float(np.mean([b[mkey] for b in boards]))
                             for mkey in ["cd_sym", "d_pred2gt", "d_gt2pred",
                                          "f0.25", "f0.5", "f1.0", "cov_gt0.5"]}
                agg[name]["n_boards"] = len(boards)
        out[model]["_summary"] = agg
        print(f"=== {model} 汇总（{agg['opacity_tau02']['n_boards']} 板）===")
        for name, a in agg.items():
            print(f"  {name:14s} CD={a['cd_sym']:.4f} pred→GT={a['d_pred2gt']:.4f} "
                  f"GT→pred={a['d_gt2pred']:.4f} F@0.5={a['f0.5']:.3f} cov@0.5={a['cov_gt0.5']:.3f}")
    json.dump(out, open(f"{OUT}/oracle.json", "w"), indent=1)
    print("已写出", f"{OUT}/oracle.json")


if __name__ == "__main__":
    main()
