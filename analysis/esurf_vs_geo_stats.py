"""E（cunei_geo_surf）vs ID+E（cunei_geo）vs ID（cunei_base）配对统计，
统一 prune_render 协议，测试 9 板。产出 C4 写入所需的全部数字。
"""
import json

import numpy as np
from scipy.stats import wilcoxon

BASE = "data/analysis2_results"
MODELS = {
    "ID": f"{BASE}/regen_main_cunei_base.json",
    "ID+E": f"{BASE}/regen_main_cunei_geo.json",
    "+surf s1": f"{BASE}/prune_render_cunei_geo_surf.json",
    "+surf s2": f"{BASE}/esurf_s2.json",
    "+surf L1.0": f"{BASE}/esurf_l10.json",
}
TAUS = ["0.0", "0.05", "0.1", "0.2", "0.3"]


def load(path):
    d = json.load(open(path))
    return {b["scene"]: b["taus"] for b in d["per_board"]}


data = {k: load(p) for k, p in MODELS.items()}
scenes = sorted(set.intersection(*(set(v) for v in data.values())))
print("配对板数:", len(scenes))


def arr(model, tau, key):
    return np.array([data[model][s][tau][key] for s in scenes])


print(f"{'模型':<10}", end="")
for tau in TAUS:
    print(f"  τ{tau:<5}", end="")
print("   raw depth  raw PSNR  raw LPIPS")

for m in MODELS:
    print(f"{m:<10}", end="")
    for tau in TAUS:
        v = np.mean(arr(m, tau, "cd_cm"))
        print(f"  {v:7.4f}", end="")
    print(f"   {np.mean(arr(m, '0.0', 'depth_mae')):8.4f}  {np.mean(arr(m, '0.0', 'psnr')):8.2f}  {np.mean(arr(m, '0.0', 'lpips')):9.4f}")

print()
for tau in TAUS:
    for a, b in [("+surf s1", "ID+E"), ("+surf s1", "ID"), ("+surf s2", "ID+E")]:
        x, y = arr(a, tau, "cd_cm"), arr(b, tau, "cd_cm")
        if (x - y != 0).any():
            w = wilcoxon(x, y)
            print(f"τ={tau}: {a} vs {b}: mean {np.mean(x):.4f} vs {np.mean(y):.4f} "
                  f"({(np.mean(x)/np.mean(y)-1)*100:+.1f}%), wins {int((x<y).sum())}/{len(x)}, p={w.pvalue:.4f}")

print()
for a, b in [("+surf s1", "ID+E")]:
    x, y = arr(a, "0.0", "depth_mae"), arr(b, "0.0", "depth_mae")
    w = wilcoxon(x, y)
    print(f"raw depth: {a} {np.mean(x):.4f} vs {b} {np.mean(y):.4f}, wins {int((x<y).sum())}/{len(x)}, p={w.pvalue:.4f}")
    for m in ["psnr", "lpips"]:
        x, y = arr(a, "0.0", m), arr(b, "0.0", m)
        w = wilcoxon(x, y)
        print(f"raw {m}: {np.mean(x):.4f} vs {np.mean(y):.4f}, p={w.pvalue:.4f}")

# 验证集操作点核对（τ=0.1 在 22 验证板上选定）
print()
val = load(f"{BASE}/esurf_val.json")
v05 = np.mean([b["0.05"]["cd_cm"] for b in val.values()])
v01 = np.mean([b["0.1"]["cd_cm"] for b in val.values()])
v02 = np.mean([b["0.2"]["cd_cm"] for b in val.values()])
print(f"验证集(22) CD: τ0.05={v05:.4f} τ0.1={v01:.4f} τ0.2={v02:.4f} → 选定操作点 τ=0.1")
