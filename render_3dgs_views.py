"""3DGS 渲染导出 (仅 cunei_gs 环境依赖: torch/numpy/PIL + gaussian-splatting)。
对每块板: 加载 PLY -> 在 pixelSplat 相同 target 视角渲染彩色图 + 深度图
-> 保存 PNG / depth.npy / xyz.npy + 拟合耗时 -> manifest。
数值计算全部由 score_3dgs.py 在 cunei 环境完成。
用法:
  /root/miniconda3/envs/cunei_gs/bin/python render_3dgs_views.py \
    --bs_dir /root/autodl-tmp/data/bs_3dgs_2v --n_views 2
"""
import argparse, glob, json, os, sys

import numpy as np
import torch
from PIL import Image

GS_REPO = "/root/autodl-tmp/proj/gaussian-splatting"
sys.path.insert(0, GS_REPO)

FX = FY = 309.0193
CX = CY = 128.0
W = H = 256


def build_camera(R, t, uid):
    from scene.cameras import Camera
    fov = 2 * np.arctan(CX / FX)
    img = Image.new("RGB", (W, H))
    return Camera(resolution=(W, H), colmap_id=uid, R=R.astype(np.float64),
                  T=t.astype(np.float64), FoVx=fov, FoVy=fov,
                  depth_params=None, image=img, invdepthmap=None,
                  image_name=f"cam{uid}", uid=uid)


def render_safe(render_fn, cam, gm, pipe, bg, override=None):
    try:
        return render_fn(cam, gm, pipe, bg, override_color=override)
    except TypeError:
        return render_fn(cam, gm, pipe, bg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs_dir", required=True)
    ap.add_argument("--n_views", type=int, required=True)
    ap.add_argument("--renders", default="/root/autodl-tmp/data/renders_full")
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--protocol", default="/root/analysis_out/eval_protocol.json")
    ap.add_argument("--out_dir", default="/root/analysis_out")
    args = ap.parse_args()
    base = os.path.join(args.out_dir, f"exports_3dgs_{args.n_views}v")
    os.makedirs(base, exist_ok=True)

    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from argparse import Namespace

    protocol = json.load(open(args.protocol)) if os.path.exists(args.protocol) else {}
    scene_dirs = sorted(glob.glob(os.path.join(args.bs_dir, f"*_{args.n_views}v")))
    print(f"3DGS scenes ({args.n_views}v): {len(scene_dirs)}", flush=True)

    mtimes = []
    for sd in scene_dirs:
        ply_p = os.path.join(sd, "model", "point_cloud",
                             f"iteration_{args.iters}", "point_cloud.ply")
        mtimes.append(os.path.getmtime(ply_p) if os.path.exists(ply_p) else -1)
    order = np.argsort(mtimes)
    durations = {}
    for oi in range(1, len(order)):
        prev_t = mtimes[order[oi - 1]]
        if prev_t > 0:
            durations[scene_dirs[order[oi]]] = mtimes[order[oi]] - prev_t

    pipe = Namespace(convert_SHs_python=False, compute_cov3D_python=False,
                     debug=False, antialiasing=False)
    bg = torch.zeros(3, device="cuda")
    manifest = []
    for k, sd in enumerate(scene_dirs):
        sc = os.path.basename(sd).replace(f"_{args.n_views}v", "")
        ply_p = os.path.join(sd, "model", "point_cloud",
                             f"iteration_{args.iters}", "point_cloud.ply")
        if not os.path.exists(ply_p):
            print(f"[{k+1}] {sc} MISSING ply", flush=True)
            continue
        meta = json.load(open(os.path.join(args.renders, f"{sc}_000_meta.json")))
        gm = GaussianModel(sh_degree=3)
        gm.load_ply(ply_p)
        xyz = gm.get_xyz.detach().cpu().numpy().astype(np.float32)
        np.save(os.path.join(base, f"{sc}_xyz.npy"), xyz)

        tgt_idx = protocol.get(sc, {}).get("target", [12, 13])
        views = meta["views"]
        for vi in tgt_idx:
            vw = next(x for x in views if x["id"] == vi)
            R = np.asarray(vw["R"]); t = np.asarray(vw["t"])
            # 本 fork 的 COLMAP 读取器对旋转多一次转置: 相机需传 R.T 才与训练一致
            cam = build_camera(R.T.copy(), t, vi)
            with torch.no_grad():
                out2 = render_safe(render, cam, gm, pipe, bg)
                rgb = (out2["render"].clamp(0, 1).cpu().numpy()
                       .transpose(1, 2, 0) * 255).astype(np.uint8)
                # fork 原生输出为逆深度: z = 1/d; 有效值过滤长尾
                d_inv = out2["depth"].detach().cpu().numpy()
                if d_inv.ndim == 3:
                    d_inv = d_inv[0] if d_inv.shape[0] in (1, 3) else d_inv[-1]
                depth = np.where(d_inv > 0.2, 1.0 / np.maximum(d_inv, 1e-4), 0.0)
                depth = np.where((depth > 0.5) & (depth < 5.0), depth, 0.0)
            Image.fromarray(rgb).save(os.path.join(base, f"{sc}_{vi:03d}.png"))
            np.save(os.path.join(base, f"{sc}_{vi:03d}_depth.npy"), depth)
        manifest.append(dict(scene=sc, n_points=len(xyz), n_views=args.n_views,
                             fit_s=round(durations.get(sd, -1), 1),
                             targets=tgt_idx))
        print(f"[{k+1}/{len(scene_dirs)}] {sc} exported n={len(xyz)} "
              f"fit={durations.get(sd, -1):.0f}s", flush=True)

    json.dump(manifest, open(os.path.join(base, "manifest.json"), "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
