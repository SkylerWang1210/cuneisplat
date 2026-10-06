"""alpha 覆盖诊断分析：覆盖率、低/高 alpha 分组误差、条件深度。

读 out/alpha_dump/{model}/{scene}.npz，输出 out/alpha_diag.json：
  每板：GT掩膜内 A>0.5 / A>0.9 覆盖率；A>0.5 子集 MAE；A 分层误差；
  条件深度 D/A（A>0.1 处）的 MAE vs 原始 MAE。
"""
import glob
import json
import os
from pathlib import Path

import numpy as np

BASE = "/root/autodl-tmp/analysis2"


def main():
    out = {}
    for model in ["cunei_base", "cunei_geo"]:
        rows = []
        for path in sorted(glob.glob(f"{BASE}/out/alpha_dump/{model}/*.npz")):
            with np.load(path) as z:
                A, D, gt, msk = (z["alpha"].astype(np.float32), z["depth"],
                                 z["gt"], z["mask"])
            for v in range(A.shape[0]):
                m = msk[v] & np.isfinite(D[v]) & (gt[v] > 0) & np.isfinite(A[v])
                if m.sum() < 100:
                    continue
                a, d, g = A[v][m], D[v][m], gt[v][m]
                hi = a > 0.5
                raw_mae = float(np.abs(d - g).mean())
                cond = d[hi] / np.clip(a[hi], 1e-3, None)
                rows.append(dict(
                    scene=Path(path).stem, view=v, n_px=int(m.sum()),
                    cov_05=float((a > 0.5).mean()), cov_09=float((a > 0.9).mean()),
                    raw_mae=raw_mae,
                    mae_highA=float(np.abs(d[hi] - g[hi]).mean()) if hi.sum() > 100 else None,
                    cond_mae=float(np.abs(cond - g[hi]).mean()) if hi.sum() > 100 else None,
                    mae_lowA=float(np.abs(d[~hi] - g[~hi]).mean()) if (~hi).sum() > 100 else None,
                    mean_A_hi=float(a[hi].mean()) if hi.sum() else None,
                ))
        agg = {k: float(np.nanmean([r[k] for r in rows if r.get(k) is not None]))
               for k in ["cov_05", "cov_09", "raw_mae", "mae_highA", "cond_mae", "mae_lowA"]}
        out[model] = {"per_view": rows, "summary": agg, "n_views": len(rows)}
        print(f"=== {model} ({len(rows)} 视角) ===")
        for k, v_ in agg.items():
            print(f"  {k}: {v_:.4f}")
    json.dump(out, open(f"{BASE}/out/alpha_diag.json", "w"), indent=1)
    print("已写出 alpha_diag.json")


if __name__ == "__main__":
    main()
