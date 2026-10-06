"""meta.json -> COLMAP 文本格式转换 + 3DGS 基线运行器。
为每块测试板生成 COLMAP scene（sparse/0/{cameras,images,points3d}.txt + images/），
然后按指定视角数跑原版 3DGS 训练。
用法: python run_3dgs_baseline.py --renders DIR --out DIR --n_views 2|16 [--limit N]
"""
import argparse, json, os, shutil, subprocess

import numpy as np


def rot_to_quat(R):
    """COLMAP 四元数 (w,x,y,z)。"""
    w = np.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w < 1e-7:  # 退化情形
        w = 0.0
        x = np.sqrt(max(0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
        y = np.sqrt(max(0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
        z = np.sqrt(max(0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    else:
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)
    return w, x, y, z


def build_colmap_scene(root, scene, out_dir, n_views):
    """从 meta 构造 COLMAP 目录。COLMAP 用 w2c；我们 meta 里就是 w2c R|t。"""
    meta = json.load(open(os.path.join(root, f"{scene}_000_meta.json")))
    views = meta["views"][:n_views]
    os.makedirs(os.path.join(out_dir, "sparse/0"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    res = 256
    with open(os.path.join(out_dir, "sparse/0/cameras.txt"), "w") as f:
        f.write("1 PINHOLE 256 256 309.0193 309.0193 128.0 128.0\n")
    with open(os.path.join(out_dir, "sparse/0/images.txt"), "w") as f:
        for i, v in enumerate(views):
            w, x, y, z = rot_to_quat(np.array(v["R"]))
            t = v["t"]
            f.write(f"{i+1} {w:.12f} {x:.12f} {y:.12f} {z:.12f} "
                    f"{t[0]:.12f} {t[1]:.12f} {t[2]:.12f} 1 "
                    f"{scene}_{v['id']:03d}.png\n\n")
    with open(os.path.join(out_dir, "sparse/0/points3D.txt"), "w") as f:
        f.write("# CuneiSplat random init\n")
        rng = np.random.default_rng(42)
        for pid in range(1000):
            x, y, z = rng.normal(0, 0.3, 3)
            r, g, b = rng.integers(100, 200, 3)
            f.write(f"{pid} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 1.0\n")
    for v in views:
        src = os.path.join(root, f"{scene}_{v['id']:03d}.png")
        shutil.copy(src, os.path.join(out_dir, "images"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_views", type=int, required=True)
    ap.add_argument("--limit", type=int, default=9)
    ap.add_argument("--iterations", type=int, default=30000)
    args = ap.parse_args()

    scenes = []
    for p in sorted(os.listdir(args.renders)):
        if p.endswith("_000_meta.json"):
            m = json.load(open(os.path.join(args.renders, p)))
            if m.get("split") == "test":
                scenes.append(p.replace("_000_meta.json", ""))
    scenes = scenes[: args.limit]
    print(f"test scenes ({args.n_views}-view): {len(scenes)}")

    gs = "/root/autodl-tmp/proj/gaussian-splatting/train.py"
    py = "/root/miniconda3/envs/cunei_gs/bin/python"
    os.makedirs(args.out, exist_ok=True)
    for k, sc in enumerate(scenes):
        scene_dir = os.path.join(args.out, f"{sc}_{args.n_views}v", "scene")
        model_dir = os.path.join(args.out, f"{sc}_{args.n_views}v", "model")
        if os.path.exists(os.path.join(model_dir, "cfg_args")):
            print(f"[{k+1}] {sc} done (skip)")
            continue
        build_colmap_scene(args.renders, sc, scene_dir, args.n_views)
        r = subprocess.run(
            [py, gs, "-s", scene_dir, "-m", model_dir,
             "--iterations", str(args.iterations),
             "--data_device", "cpu"],
            capture_output=True, text=True)
        ok = os.path.exists(os.path.join(model_dir, "point_cloud",
                                         f"iteration_{args.iterations}",
                                         "point_cloud.ply"))
        print(f"[{k+1}/{len(scenes)}] {sc} {'OK' if ok else 'FAIL'} "
              f"({r.returncode})", flush=True)
        if not ok:
            print(r.stderr[-500:], flush=True)


if __name__ == "__main__":
    main()
