"""实验 #5：关键配对比较的完整统计表（W/差值/CI/胜负/N）。

来源：主 split 用上传的 prune_render JSON（逐板 depth/psnr/ssim/cd）；
外部比较用 tau_grid.json 的重算值；种子对比用 seed2 JSON。
输出 out/stats.json 与可读表。
"""
import json
import os

import numpy as np
from scipy.stats import wilcoxon

BASE = "/root/autodl-tmp/analysis2"
OUT = f"{BASE}/out"


def paired_stats(x, y, label, unit="", extra=""):
    """x = ID+E (geo), y = ID (base)，逐板配对。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    diff = x - y
    test = wilcoxon(x, y, alternative="two-sided", method="auto")
    boots = np.mean(np.random.default_rng(0).choice(diff, size=(10000, len(diff))), axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    res = dict(label=label, unit=unit, note=extra, n=len(x),
               mean_geo=float(x.mean()), mean_base=float(y.mean()),
               paired_mean_diff=float(diff.mean()), wins_geo=int((diff < 0).sum()),
               wins_base=int((diff > 0).sum()), W=float(test.statistic), p=float(test.pvalue),
               boot95ci=[float(lo), float(hi)],
               bootstrap="percentile, 10000, seed=0, unit=tablet")
    print(f"[{label}] n={res['n']} geo={res['mean_geo']:.4f} base={res['mean_base']:.4f} "
          f"diff={res['paired_mean_diff']:+.4f} 胜geo/base={res['wins_geo']}/{res['wins_base']} "
          f"W={res['W']:.0f} p={res['p']:.4g} CI=[{lo:+.4f},{hi:+.4f}]")
    return res


def per_board_map(path, tau, key):
    d = json.load(open(path))
    return {b["scene"]: b["taus"][tau][key] for b in d["per_board"]}


def main():
    os.makedirs(OUT, exist_ok=True)
    res = []
    gb = json.load(open(f"{BASE}/results/prune_render_cunei_base.json"))["per_board"]
    gg = json.load(open(f"{BASE}/results/prune_render_cunei_geo.json"))["per_board"]
    scenes = [b["scene"] for b in gb]
    idx_b = {b["scene"]: b for b in gb}
    idx_g = {b["scene"]: b for b in gg}

    def pair(tau_b, tau_g, key):
        return ([idx_g[s]["taus"][tau_g][key] for s in scenes],
                [idx_b[s]["taus"][tau_b][key] for s in scenes])

    # 主 split 关键比较（已发表口径：cap50k 的 cd_cm）
    x, y = pair("0.0", "0.0", "depth_mae")
    res.append(paired_stats(x, y, "主split 深度MAE（τ=0，归一化单位）"))
    x, y = pair("0.0", "0.0", "cd_cm")
    res.append(paired_stats(x, y, "主split 原始CD（τ=0，cm）"))
    x, y = pair("0.2", "0.2", "cd_cm")
    res.append(paired_stats(x, y, "主split 匹配τ=0.2 CD（cm）"))
    x, y = pair("0.0", "0.0", "psnr")
    res.append(paired_stats(x, y, "主split PSNR（τ=0，dB）", extra="非显著≠等价"))
    x, y = pair("0.0", "0.0", "ssim")
    res.append(paired_stats(x, y, "主split SSIM（τ=0）", extra="非显著≠等价"))
    # best-vs-best：base@0.05 vs geo@0.2（τ=0.05 未在已发布 JSON，用缓存重算值见 tau_grid）
    # ——主 split 的 0.05 需要主缓存网格，此处先给已发布 JSON 内可算的部分；主缓存 0.05 网格由 tau_grid 补充。

    # 外部（来自实验 #1 输出）
    tg_path = f"{OUT}/tau_grid.json"
    if os.path.exists(tg_path):
        tg = json.load(open(tg_path))
        c = tg["comparisons"]
        for k in ["matched_tau02", "best_vs_best"]:
            if k in c:
                r = c[k]
                res.append({kk: r[kk] for kk in
                            ["label", "n", "mean_geo", "mean_base", "wins_geo",
                             "statistic", "p", "boot_mean_diff_95ci"]})
                print(f"[外部:{r['label']}] n={r['n']} geo={r['mean_geo']:.4f} base={r['mean_base']:.4f} "
                      f"胜geo={r['wins_geo']} W={r['statistic']:.0f} p={r['p']:.4g} CI={r['boot_mean_diff_95ci']}")

    # 种子稳定性（seed1 vs seed2，各自模型内部）
    for model in ["cunei_base", "cunei_geo"]:
        s1 = {b["scene"]: b for b in json.load(open(f"{BASE}/results/prune_render_{model}.json"))["per_board"]}
        s2 = {b["scene"]: b for b in json.load(open(f"{BASE}/results/seed2_prune_render_{model}.json"))["per_board"]}
        common = sorted(set(s1) & set(s2))
        res.append(paired_stats(
            [s2[s]["taus"]["0.0"]["depth_mae"] for s in common],
            [s1[s]["taus"]["0.0"]["depth_mae"] for s in common],
            f"{model} 深度：seed2 vs seed1（τ=0）", extra="训练方差参考，非模型比较"))
        res.append(paired_stats(
            [s2[s]["taus"]["0.2"]["cd_cm"] for s in common],
            [s1[s]["taus"]["0.2"]["cd_cm"] for s in common],
            f"{model} τ=0.2 CD：seed2 vs seed1（cm）", extra="训练方差参考，非模型比较"))

    json.dump(res, open(f"{OUT}/stats.json", "w"), indent=1)
    print("已写出", f"{OUT}/stats.json")


if __name__ == "__main__":
    main()
