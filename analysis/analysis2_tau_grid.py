"""实验 #1：外部阈值完整网格（双模型 × τ∈{0,.05,.1,.2,.3}）。

协议与论文/评审一致：r<1.6、opacity>τ、预测点上限 50k（rng0）、GT 缓存 20k、
逐板 cm 缩放、双向最近邻非平方距离均值（×0.5）。使用上传的固定 draw 缓存。
输出 /root/autodl-tmp/analysis2/out/tau_grid.json
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon

BASE = "/root/autodl-tmp/analysis2"
OUT = f"{BASE}/out"
TAUS = [0.0, 0.05, 0.1, 0.2, 0.3]


def calc_cd(means, opacities, gt20k, scale_cm, tau):
    keep = np.linalg.norm(means, axis=1) < 1.6
    if tau > 0:
        keep &= opacities > tau
    pred = means[keep]
    if len(pred) < 100:
        return -1.0
    if len(pred) > 50000:
        sel = np.random.default_rng(0).choice(len(pred), 50000, replace=False)
        pred = pred[sel]
    d1 = cKDTree(gt20k).query(pred, k=1)[0]
    d2 = cKDTree(pred).query(gt20k, k=1)[0]
    return float(0.5 * (d1.mean() + d2.mean()) * scale_cm)


def load_model(name):
    per = {}
    for path in sorted(glob.glob(f"{BASE}/caches/ext_gauss_cache_{name}/*.npz")):
        with np.load(path) as z:
            per[Path(path).stem] = {k: z[k] for k in ("means", "opacities", "gt20k", "scale_cm")}
    return per


def main():
    os.makedirs(OUT, exist_ok=True)
    sweep = {}
    for model in ["cunei_base", "cunei_geo"]:
        per = load_model(model)
        sweep[model] = {}
        for scene in sorted(per):
            z = per[scene]
            sweep[model][scene] = {
                str(t): calc_cd(z["means"], z["opacities"], z["gt20k"],
                                float(z["scale_cm"]), t)
                for t in TAUS
            }
        # 交叉验证：与已发布 JSON 的 τ=0/.1/.2/.3 对账
        pub = json.load(open(f"{BASE}/results/ext_prune_render_{model}.json"))
        src = {b["scene"]: b["taus"] for b in pub["per_board"]}
        deltas = []
        for s, tv in sweep[model].items():
            for t in ["0.0", "0.1", "0.2", "0.3"]:
                pv = src[s][t]["cd_cm"]
                if tv[t] > 0 and pv > 0:
                    deltas.append(abs(tv[t] - pv))
        print(f"[对账] {model}: 与已发布值最大差 {max(deltas):.6f} cm, n={len(deltas)}")

    # 关键比较
    def paired(a_map, b_map, label):
        scenes = sorted(set(a_map) & set(b_map))
        x = np.array([a_map[s] for s in scenes])   # ID+E (geo)
        y = np.array([b_map[s] for s in scenes])   # ID (base)
        diff = x - y
        test = wilcoxon(x, y, alternative="two-sided", method="auto")
        boots = np.mean(np.random.default_rng(0).choice(diff, size=(10000, len(diff))), axis=1)
        res = dict(label=label, n=len(scenes), mean_geo=float(x.mean()), mean_base=float(y.mean()),
                   wins_geo=int((diff < 0).sum()), statistic=float(test.statistic),
                   p=float(test.pvalue),
                   boot_mean_diff_95ci=np.percentile(boots, [2.5, 97.5]).tolist(),
                   per_scene={s: dict(geo=a_map[s], base=b_map[s]) for s in scenes})
        print(f"[{label}] n={res['n']} geo均值={res['mean_geo']:.4f} base均值={res['mean_base']:.4f} "
              f"geo胜={res['wins_geo']} W={res['statistic']:.0f} p={res['p']:.4f} "
              f"CI={res['boot_mean_diff_95ci']}")
        return res

    results = {"tau_grid": sweep, "comparisons": {}}
    g, b = sweep["cunei_geo"], sweep["cunei_base"]
    results["comparisons"]["matched_tau02"] = paired(
        {s: v["0.2"] for s, v in g.items() if v["0.2"] > 0},
        {s: v["0.2"] for s, v in b.items() if v["0.2"] > 0},
        "匹配 τ=0.2（两侧同阈值）")
    # best-vs-best（探索性）：各自在有效对象上的均值最优 τ
    def best_tau(per):
        valid = {t: [v[str(t)] for v in per.values() if v[str(t)] > 0] for t in TAUS}
        means = {t: float(np.mean(vs)) for t, vs in valid.items()}
        bt = min(means, key=means.get)
        return bt, means
    bt_g, means_g = best_tau(g)
    bt_b, means_b = best_tau(b)
    print(f"[best-τ] geo 最优 τ={bt_g} (均值表 { {str(k): round(v,4) for k,v in means_g.items()} })")
    print(f"[best-τ] base 最优 τ={bt_b} (均值表 { {str(k): round(v,4) for k,v in means_b.items()} })")
    results["comparisons"]["best_vs_best"] = paired(
        {s: v[str(bt_g)] for s, v in g.items() if v[str(bt_g)] > 0},
        {s: v[str(bt_b)] for s, v in b.items() if v[str(bt_b)] > 0},
        f"best-vs-best geo@{bt_g} vs base@{bt_b}（探索性）")
    results["best_tau"] = {"geo": float(bt_g), "base": float(bt_b),
                           "means_geo": {str(k): v for k, v in means_g.items()},
                           "means_base": {str(k): v for k, v in means_b.items()}}

    json.dump(results, open(f"{OUT}/tau_grid.json", "w"), indent=1)
    print("已写出", f"{OUT}/tau_grid.json")


if __name__ == "__main__":
    main()
