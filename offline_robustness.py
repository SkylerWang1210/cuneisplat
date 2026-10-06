"""A5+B8+B10: 离线稳健性分析（CPU only，无需 GPU）。
输入: prune_render_eval.py 产生的 gauss_cache_{exp}/{scene}.npz
输出:
  1) 三臂剪枝对照 (opacity / random count-matched / scale count-matched)
  2) CD 协议稳健性 (GT 采样数/种子/半径/预测上限/可见面掩膜) 下关键排序是否稳定
  3) 配对统计 (Wilcoxon + 符号计数 + bootstrap CI)

用法: python offline_robustness.py --cache_t1 .../gauss_cache_cunei_base \
          --cache_t2 .../gauss_cache_cunei_geo [--out .../robustness.json]
"""
import argparse, glob, json, os

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon

FX = FY = 309.0193
CX = CY = 128.0


def load_cache(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(p, allow_pickle=True)
        sc = os.path.basename(p)[:-4]
        out[sc] = {k: z[k] for k in z.files}
    return out


def cd_cm(pred, gt, scale_cm):
    if len(pred) < 100 or len(gt) < 100:
        return -1.0
    t1 = cKDTree(gt); d1, _ = t1.query(pred, k=1)
    t2 = cKDTree(pred); d2, _ = t2.query(gt, k=1)
    return float(0.5 * (d1.mean() + d2.mean()) * scale_cm)


def cd_protocol(means, opac, gt_full, scale_cm, ply, ncenter, nscale,
                pred_cap=50000, pred_seed=0, gt_n=20000, gt_seed=0,
                radius=1.6, visible_mask=None):
    """统一 CD 协议; gt_full 为该板全部归一化顶点。visible_mask: (m,) bool
    作用在 gt_full 上(可见面变体)。"""
    rng = np.random.default_rng(gt_seed)
    if len(gt_full) > gt_n:
        gt = gt_full[rng.choice(len(gt_full), gt_n, replace=False)]
    else:
        gt = gt_full
    if visible_mask is not None:
        vm = visible_mask if visible_mask.shape == gt_full.shape[:1] else None
        if vm is None:
            return -1.0
        gt_v = gt_full[vm]
        if len(gt_v) > gt_n:
            gt = gt_v[rng.choice(len(gt_v), gt_n, replace=False)]
        else:
            gt = gt_v
    pred = means
    if pred_cap and len(pred) > pred_cap:
        sel = np.random.default_rng(pred_seed).choice(
            len(pred), pred_cap, replace=False)
        pred = pred[sel]
    r = np.linalg.norm(pred, axis=1)
    pred = pred[r < radius]
    if len(pred) < 100:
        return -1.0
    return cd_cm(pred, gt, scale_cm)


def gt_full_for(bd):
    """从缓存的 gt20k 恢复不了全量顶点 -> 直接读 PLY。"""
    import trimesh
    mesh = trimesh.load(str(bd["ply"]), force="mesh", process=False)
    v = (np.asarray(mesh.vertices, dtype=np.float32)
         - bd["normalize_center"]) / float(bd["normalize_scale"])
    return v


def visible_mask_for(scene, bd, gt_full, renders, tol=0.03):
    mask = np.zeros(len(gt_full), dtype=bool)
    for ci, vi in enumerate(bd["ctx_idx"].tolist()):
        E = bd["ctx_extrinsics"][ci]
        R, t = E[:3, :3], E[:3, 3]
        p = gt_full @ R.T + t
        z = p[:, 2]
        good = z > 0
        u = np.round(FX * p[:, 0] / np.where(good, z, 1) + CX).astype(int)
        w = np.round(FY * p[:, 1] / np.where(good, z, 1) + CY).astype(int)
        inb = good & (u >= 0) & (u < 256) & (w >= 0) & (w < 256)
        try:
            dep = np.load(os.path.join(renders, f"{scene}_{vi:03d}_depth.npy"))
            msk = np.load(os.path.join(renders, f"{scene}_{vi:03d}_mask.npy"))
            ui, wi = u[inb], w[inb]
            d_ok = (np.abs(z[inb] - dep[wi, ui]) < tol) & msk[wi, ui]
            mask[inb] |= d_ok
        except FileNotFoundError:
            return None
    return mask


def paired(a, b, n_boot=10000, seed=0):
    a, b = np.asarray(a), np.asarray(b)
    assert len(a) == len(b) and len(a) >= 5
    diff = a - b
    try:
        stat, p = wilcoxon(a, b)
    except ValueError:
        stat, p = float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = [np.mean(rng.choice(diff, len(diff))) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(n=len(a), mean_diff=float(diff.mean()),
                win=int((diff < 0).sum()), loss=int((diff > 0).sum()),
                tie=int((diff == 0).sum()),
                wilcoxon_p=float(p), boot_ci=[float(lo), float(hi)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_t1", required=True)
    ap.add_argument("--cache_t2", required=True)
    ap.add_argument("--renders", default="/root/autodl-tmp/data/renders_full")
    ap.add_argument("--out", default="/root/analysis_out/robustness.json")
    ap.add_argument("--taus", default="0.05,0.1,0.2,0.3")
    args = ap.parse_args()
    taus = [float(x) for x in args.taus.split(",")]

    c1, c2 = load_cache(args.cache_t1), load_cache(args.cache_t2)
    scenes = sorted(set(c1) & set(c2))
    print(f"common boards: {len(scenes)}")
    gt_full = {s: gt_full_for(c1[s]) for s in scenes}

    # ============ 1) 三臂剪枝对照 (统一协议) ============
    print("\n===== Three-arm pruning control (unified protocol) =====")
    arms = {"t1": {}, "t2": {}}
    for tau in [0.0] + taus:
        for tag, cache in (("t1", c1), ("t2", c2)):
            per = []
            for s in scenes:
                bd = cache[s]
                means, opac = bd["means"], bd["opacities"]
                r = np.linalg.norm(means, axis=1)
                m = (r < 1.6) & (opac > tau) if tau > 0 else (r < 1.6)
                per.append(cd_cm(means[m], bd["gt20k"], float(bd["scale_cm"])))
            arms[tag][tau] = [round(x, 4) for x in per]
            print(f"  opacity {tag} tau={tau}: mean={np.mean(per):.4f}cm")
    # count-matched random / scale arms (匹配 opacity 臂的保留数)
    rng_rep = 3
    for tau in taus:
        for tag, cache in (("t1", c1), ("t2", c2)):
            rand_all, scale_all = [], []
            for s in scenes:
                bd = cache[s]
                means, opac, sp = bd["means"], bd["opacities"], bd["scale_proxy"]
                r = np.linalg.norm(means, axis=1)
                inside = r < 1.6
                k = int(((opac > tau) & inside).sum())
                idx_in = np.where(inside)[0]
                if k < 100 or len(idx_in) <= k:
                    rand_all.append(-1); scale_all.append(-1); continue
                rs = []
                for rep in range(rng_rep):
                    sel = np.random.default_rng(100 + rep).choice(
                        idx_in, k, replace=False)
                    rs.append(cd_cm(means[sel], bd["gt20k"], float(bd["scale_cm"])))
                rand_all.append(float(np.mean(rs)))
                order = np.argsort(-sp[idx_in])[:k]  # 最大尺度优先
                sel = idx_in[order]
                scale_all.append(cd_cm(means[sel], bd["gt20k"], float(bd["scale_cm"])))
            arms[tag][f"random@{tau}"] = [round(x, 4) for x in rand_all]
            arms[tag][f"scale@{tau}"] = [round(x, 4) for x in scale_all]
            print(f"  random  {tag}@{tau}: mean={np.mean(rand_all):.4f}cm | "
                  f"scale {tag}@{tau}: mean={np.mean(scale_all):.4f}cm")

    # ============ 2) 协议稳健性 ============
    print("\n===== Protocol robustness =====")
    variants = {
        "default": dict(),
        "gt10k": dict(gt_n=10000),
        "gt50k": dict(gt_n=50000),
        "gt_seed1": dict(gt_seed=1),
        "gt_seed2": dict(gt_seed=2),
        "radius1.3": dict(radius=1.3),
        "radius2.0": dict(radius=2.0),
        "pred_cap20k": dict(pred_cap=20000),
        "pred_nocap": dict(pred_cap=None),
    }
    robust = {}
    key_cfgs = {
        "t1@0": lambda bd: (bd["means"], bd["opacities"], 0.0),
        "t2@0": lambda bd: (bd["means"], bd["opacities"], 0.0),
        "t1@0.1": lambda bd: (bd["means"], bd["opacities"], 0.1),
        "t2@0.2": lambda bd: (bd["means"], bd["opacities"], 0.2),
    }
    for vname, kw in variants.items():
        row = {}
        for cname, get in key_cfgs.items():
            tag = cname.split("@")[0]
            tau = float(cname.split("@")[1])
            cache = c1 if tag == "t1" else c2
            per = []
            for s in scenes:
                bd = cache[s]
                means, opac, _ = get(bd)
                r = np.linalg.norm(means, axis=1)
                m = (r < 1.6) & (opac > tau) if tau > 0 else (r < 1.6)
                per.append(cd_protocol(
                    means[m], None, gt_full[s], float(bd["scale_cm"]),
                    bd["ply"], bd["normalize_center"], bd["normalize_scale"],
                    **kw))
            row[cname] = round(float(np.mean(per)), 4)
        row["t2@0.2_better_than_t1@0.1"] = row["t2@0.2"] < row["t1@0.1"]
        row["t2_better_than_t1_raw"] = row["t2@0"] < row["t1@0"]
        robust[vname] = row
        print(f"  {vname}: " + " ".join(f"{k}={v}" for k, v in row.items()
                                        if isinstance(v, (int, float))))
    # 可见面变体
    vis_row = {}
    for cname, get in key_cfgs.items():
        tag = cname.split("@")[0]
        tau = float(cname.split("@")[1])
        cache = c1 if tag == "t1" else c2
        per = []
        for s in scenes:
            bd = cache[s]
            vm = visible_mask_for(s, bd, gt_full[s], args.renders)
            if vm is None:
                per.append(-1); continue
            means, opac, _ = get(bd)
            r = np.linalg.norm(means, axis=1)
            m = (r < 1.6) & (opac > tau) if tau > 0 else (r < 1.6)
            per.append(cd_protocol(
                means[m], None, gt_full[s], float(bd["scale_cm"]),
                bd["ply"], bd["normalize_center"], bd["normalize_scale"],
                visible_mask=vm))
        ok = [x for x in per if x > 0]
        vis_row[cname] = round(float(np.mean(ok)), 4) if ok else -1
        vis_row[f"{cname}_n_valid"] = len(ok)
    vis_row["t2@0.2_better_than_t1@0.1"] = vis_row["t2@0.2"] < vis_row["t1@0.1"]
    robust["visible_face"] = vis_row
    print(f"  visible_face: {vis_row}")

    # ============ 3) 配对统计 (CD) ============
    print("\n===== Paired statistics (CD) =====")
    stats = {}
    pairs = {
        "t2@0_vs_t1@0 (raw)": (arms["t2"][0.0], arms["t1"][0.0]),
        "t2@0.2_vs_t1@0.1 (best-vs-best)": (arms["t2"][0.2], arms["t1"][0.1]),
        "t2@0.2_vs_t1@0.2 (same-tau)": (arms["t2"][0.2], arms["t1"][0.2]),
        "t2@0.1_vs_t1@0.1 (same-tau)": (arms["t2"][0.1], arms["t1"][0.1]),
        "t2@0.2_vs_t2_random@0.2": (arms["t2"][0.2], arms["t2"]["random@0.2"]),
        "t2@0.2_vs_t2_scale@0.2": (arms["t2"][0.2], arms["t2"]["scale@0.2"]),
    }
    for name, (a, b) in pairs.items():
        a2 = [x for x in a if x > 0]; b2 = [x for x in b if x > 0]
        if len(a2) == len(b2) and len(a2) >= 5:
            stats[name] = paired(a2, b2)
            print(f"  {name}: {stats[name]}")

    json.dump({"arms": arms, "robustness": robust, "paired_cd": stats},
              open(args.out, "w"), indent=1)
    print(f"\nsaved {args.out}")
    print("DONE")


if __name__ == "__main__":
    main()
