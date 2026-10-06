import glob, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import json
rows = {}
for path in sorted(glob.glob("/root/autodl-tmp/analysis2/eval_esurf/gauss_cache_cunei_geo_surf/*.npz")):
    with np.load(path) as z:
        m, o, gt = z["means"], z["opacities"], z["gt20k"]
        sc = float(z["scale_cm"])
    inside = np.linalg.norm(m, axis=1) < 1.6
    m, o = m[inside], o[inside]
    dist = cKDTree(gt).query(m, k=1)[0] * sc
    rows[Path(path).stem] = dict(n=int(len(o)), p90=float(np.quantile(o, .9)),
        frac_lt005=float((o < .05).mean()), frac_gt03=float((o > .3).mean()),
        band_lt005_cm=float(dist[o < .05].mean()) if (o < .05).sum() > 100 else None,
        band_gt03_cm=float(dist[o > .3].mean()) if (o > .3).sum() > 100 else None,
        band_mid_cm=float(dist[(o >= .05) & (o <= .3)].mean()) if ((o >= .05) & (o <= .3)).sum() > 100 else None)
agg = {k: float(np.mean([v[k] for v in rows.values() if v.get(k) is not None])) for k in ["p90","frac_lt005","frac_gt03","band_lt005_cm","band_gt03_cm","band_mid_cm"]}
print(json.dumps(agg, indent=1))
print("参照 ID+E: p90=0.287 frac<0.05见下 低带=1.09cm 高带=0.244cm 中带=0.618cm")
print("参照 ID:   p90=0.131 高带近乎空")
