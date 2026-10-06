"""判官探针 P1/P2/P5/P3 四合一（14 号文档预注册，全部本地缓存 CPU）。

P1 安慰剂 oracle：球内均匀云 + 同预算 oracle 选择 → 判 0.064 是否密度假象
P2 采样地板：gt20k 自身最近邻间距均值
P5 opacity 强度：逐板 opacity vs 真值距离 Spearman
P3 可预测上限：全云特征 + 跨板留一岭回归；报告留一 R² 与"回归选点 CD"
    特征：opacity、log-opacity、scale、半径、局部密度（第10近邻距离）
输出 out/probes.json
"""
import glob
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

BASE = "/root/autodl-tmp/analysis2"
OUT = f"{BASE}/out"


def evaluate_cd(pred, gt20k, scale_cm):
    if len(pred) > 50000:
        pred = pred[np.random.default_rng(0).choice(len(pred), 50000, replace=False)]
    d1 = cKDTree(gt20k).query(pred, k=1)[0]
    d2 = cKDTree(pred).query(gt20k, k=1)[0]
    return float(0.5 * (d1.mean() + d2.mean()) * scale_cm)


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {}
    for model in ["cunei_base", "cunei_geo"]:
        boards = {}
        for path in sorted(glob.glob(f"{BASE}/caches/gauss_cache_{model}/*.npz")):
            with np.load(path) as z:
                boards[Path(path).stem] = {k: z[k] for k in
                                           ("means", "opacities", "gt20k", "scale_cm", "scale_proxy")}
        p1, p2, p5 = {}, {}, []
        feats_all, dists_all, coords_all, ks = {}, {}, {}, {}
        for scene, z in boards.items():
            means, opac, gt = z["means"], z["opacities"], z["gt20k"]
            sc = float(z["scale_cm"])
            sp = z["scale_proxy"]
            inside = np.linalg.norm(means, axis=1) < 1.6
            m, o, s = means[inside], opac[inside], sp[inside]
            k = int((o > 0.2).sum())
            ks[scene] = k
            t = cKDTree(gt)
            # P2
            p2[scene] = float(t.query(gt, k=2)[0][:, 1].mean() * sc)
            # 真值距离（全云）
            dist = t.query(m, k=1)[0] * sc
            # P5
            rho, _ = spearmanr(o, dist)
            p5.append(float(rho))
            # P1
            rng = np.random.default_rng(7)
            cloud = rng.uniform(-1.6, 1.6, size=(400000, 3))
            cloud = cloud[np.linalg.norm(cloud, axis=1) < 1.6]
            dcloud = t.query(cloud, k=1)[0]
            p1[scene] = evaluate_cd(cloud[np.argsort(dcloud)[:k]], gt, sc)
            # P3 特征（密度用自建树第10近邻；每板子采样 12 万点控内存）
            if len(m) > 120000:
                keep = np.random.default_rng(3).choice(len(m), 120000, replace=False)
                m2, o2, s2, d2 = m[keep], o[keep], s[keep], dist[keep]
            else:
                m2, o2, s2, d2 = m, o, s, dist
            dens = cKDTree(m2).query(m2, k=11)[0][:, 10]
            feats_all[scene] = np.stack([o2, np.log(o2 + 1e-6), s2,
                                         np.linalg.norm(m2, axis=1), dens], axis=1)
            dists_all[scene] = d2
            coords_all[scene] = m2
            ks[scene] = min(ks[scene], len(m2))
        # P3b 板内秩归一化（选点只需板内排序，消除跨板尺度漂移）
        scenes = list(boards.keys())
        ranks_all = {}
        for sc_ in scenes:
            X = feats_all[sc_]
            r = np.empty_like(X)
            for c in range(X.shape[1]):
                r[:, c] = np.argsort(np.argsort(X[:, c])) / len(X)
            ranks_all[sc_] = r
        r2b, selcdb = [], []
        for held in scenes:
            tr = [s_ for s_ in scenes if s_ != held]
            Xtr = np.concatenate([ranks_all[s_] for s_ in tr])
            ytr = np.concatenate([dists_all[s_] for s_ in tr])
            w = np.linalg.solve(Xtr.T @ Xtr + 1.0 * np.eye(5), Xtr.T @ ytr)
            pred = ranks_all[held] @ w
            yt = dists_all[held]
            r2b.append(1 - ((yt - pred) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())
            top = np.argsort(pred)[:ks[held]]
            selcdb.append(evaluate_cd(coords_all[held][top], boards[held]["gt20k"],
                                      float(boards[held]["scale_cm"])))
        # P3 跨板留一（原始特征）
        r2s, selcds = [], []
        for held in scenes:
            tr = [s for s in scenes if s != held]
            Xtr = np.concatenate([feats_all[s] for s in tr])
            ytr = np.concatenate([dists_all[s] for s in tr])
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            Xtr_n = (Xtr - mu) / sd
            w = np.linalg.solve(Xtr_n.T @ Xtr_n + 1.0 * np.eye(5), Xtr_n.T @ ytr)
            Xte = (feats_all[held] - mu) / sd
            pred = Xte @ w
            yt = dists_all[held]
            r2s.append(1 - ((yt - pred) ** 2).sum() / ((yt - yt.mean()) ** 2).sum())
            top = np.argsort(pred)[:ks[held]]
            selcds.append(evaluate_cd(coords_all[held][top], boards[held]["gt20k"],
                                      float(boards[held]["scale_cm"])))
        out[model] = {
            "P1_placebo_oracle_cd": p1, "P1_mean": float(np.mean(list(p1.values()))),
            "P2_gt_nn_floor_cm": p2, "P2_mean": float(np.mean(list(p2.values()))),
            "P5_opacity_spearman_mean": float(np.mean(p5)), "P5_per_board": p5,
            "P3_loo_r2_mean": float(np.mean(r2s)), "P3_loo_r2": r2s,
            "P3_regselect_cd_mean": float(np.mean(selcds)), "P3_regselect_cd": selcds,
            "P3b_rank_loo_r2_mean": float(np.mean(r2b)), "P3b_rank_loo_r2": r2b,
            "P3b_rank_regselect_cd_mean": float(np.mean(selcdb)),
        }
        print(f"=== {model} ===", flush=True)
        print(f"  P1 安慰剂oracle CD 均值: {out[model]['P1_mean']:.4f}（真云 oracle=0.064）")
        print(f"  P2 gt采样地板: {out[model]['P2_mean']:.4f} cm")
        print(f"  P5 opacity-距离 Spearman: {out[model]['P5_opacity_spearman_mean']:.3f}")
        print(f"  P3 留一R²: {out[model]['P3_loo_r2_mean']:.3f}, 回归选点CD: "
              f"{out[model]['P3_regselect_cd_mean']:.4f}（opacity 参照 "
              f"{0.4344 if model == 'cunei_base' else 0.2236}）", flush=True)


if __name__ == "__main__":
    main()
