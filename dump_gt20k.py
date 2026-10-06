"""预导出 GT 20k 点云 npz (cunei 环境, 供 eval_3dgs_all 等 cunei_gs 环境复用)。
用法: python dump_gt20k.py [--renders /root/autodl-tmp/data/renders_full]
输出: /root/analysis_out/gt20k/{scene}.npz  {gt20k, scale_cm, center, scale}
"""
import argparse, glob, json, os

import numpy as np
import trimesh

ap = argparse.ArgumentParser()
ap.add_argument("--renders", default="/root/autodl-tmp/data/renders_full")
ap.add_argument("--ply_dir", default="/root/autodl-tmp/data/full_src")
ap.add_argument("--out_dir", default="/root/analysis_out/gt20k")
ap.add_argument("--limit", type=int, default=100)
args = ap.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

metas_all = sorted(glob.glob(os.path.join(args.renders, "*_000_meta.json")))
metas = [m for m in metas_all
         if json.load(open(m)).get("split") == "test"][:args.limit]
print(f"test-split metas: {len(metas)}")
for k, mp in enumerate(metas):
    meta = json.load(open(mp))
    sc = os.path.basename(mp).replace("_000_meta.json", "")
    out_p = os.path.join(args.out_dir, f"{sc}.npz")
    ply = os.path.join(args.ply_dir, meta["ply"])
    if os.path.exists(out_p):
        continue
    if not os.path.exists(ply):
        print(f"[{k+1}] {sc} PLY MISSING: {ply}")
        continue
    mesh = trimesh.load(ply, force="mesh", process=False)
    v = np.asarray(mesh.vertices, dtype=np.float32)
    v = (v - np.asarray(meta["normalize_center"])) / float(meta["normalize_scale"])
    sel = np.random.default_rng(0).choice(len(v), min(20000, len(v)), replace=False)
    np.savez_compressed(out_p, gt20k=v[sel].astype(np.float32),
                        scale_cm=np.float32(meta["normalize_scale"] * 0.1),
                        normalize_center=np.asarray(meta["normalize_center"],
                                                    dtype=np.float32),
                        normalize_scale=np.float32(meta["normalize_scale"]))
    print(f"[{k+1}/{len(metas)}] {sc} ok ({len(v)} verts)")
print("DONE")
