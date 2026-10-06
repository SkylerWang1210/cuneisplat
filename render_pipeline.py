"""CuneiSplat 数据渲染管线 v2 (pyrender + EGL)
泥板 PLY -> 多视角 RGB + 深度 + 掩膜 + 相机/光照协议记录

v2 变更：Open3D(Filament/Vulkan) -> pyrender(OpenGL/EGL)。
- 显式灯光对象：key/fill 光方位、强度、颜色逐视角随机（论文 §3.3 协议）
- IntrinsicsCamera 精确使用渲染 K；深度为线性 z
- OpenCV(+z 前) <-> OpenGL(-z 前) 位姿转换
用法: PYOPENGL_PLATFORM=egl python render_pipeline.py --in_dir ... --out_dir ...
"""
import argparse, hashlib, json, os, time

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender
import trimesh
from PIL import Image

RES_DEFAULT = 256
VIEWS_DEFAULT = 16
CLAY_ALBEDO = [0.62, 0.55, 0.47]  # 烘干黏土反照率近似（灰黄）
FOV_Y_DEG = 45.0


def look_at_cv(eye, target, up=(0.0, 0.0, 1.0)):
    """OpenCV 约定 w2c [R|t]：+z 指向场景，+y 向下。"""
    eye = np.asarray(eye, float); target = np.asarray(target, float)
    up = np.asarray(up, float)
    zc = target - eye; zc /= np.linalg.norm(zc)
    xc = np.cross(up, zc)
    if np.linalg.norm(xc) < 1e-6:
        xc = np.cross((1.0, 0.0, 0.0), zc)
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    R = np.stack([xc, yc, zc], 0)
    return R, -R @ eye


def cv_w2c_to_gl_c2w(R, t):
    """OpenCV 外参 -> pyrender(OpenGL) 相机位姿 c2w。"""
    w2c = np.eye(4)
    w2c[:3, :3] = R; w2c[:3, 3] = t
    cv2gl = np.diag([1.0, -1.0, -1.0, 1.0])
    return np.linalg.inv(w2c) @ cv2gl


def ring_cameras(rng, radius, n):
    """环形（转台先验）：方位均匀+抖动，俯仰 25~65°。"""
    cams = []
    for i in range(n):
        az = 2 * np.pi * i / n + rng.uniform(-0.25, 0.25) * (2 * np.pi / n)
        el = np.deg2rad(rng.uniform(25.0, 65.0))
        eye = np.array([radius * np.cos(el) * np.cos(az),
                        radius * np.cos(el) * np.sin(az),
                        radius * np.sin(el)])
        R, t = look_at_cv(eye, (0.0, 0.0, 0.0))
        cams.append({"eye": eye.tolist(), "R": R.tolist(), "t": t.tolist()})
    return cams


def render_one(renderer, mesh_py, cams, res, rng, out_base, fixed=False):
    f = 0.5 * res / np.tan(np.deg2rad(FOV_Y_DEG / 2))
    cam = pyrender.IntrinsicsCamera(fx=f, fy=f, cx=res / 2, cy=res / 2,
                                    znear=0.05, zfar=20.0)
    K = [[f, 0, res / 2], [0, f, res / 2], [0, 0, 1]]
    views = []
    for i, c in enumerate(cams):
        amb = 0.15 if fixed else rng.uniform(0.08, 0.35)
        scene = pyrender.Scene(ambient_light=[amb] * 3,
                               bg_color=[0.0, 0.0, 0.0])
        scene.add(mesh_py)
        cam_node = scene.add(cam, pose=cv_w2c_to_gl_c2w(
            np.array(c["R"]), np.array(c["t"])))
        # --- 光照域随机化（论文 §3.3）---
        eye = np.array(c["eye"])
        key_dir = rng.uniform(0.7, 1.3)
        key_pos = eye * key_dir + (0 if fixed else rng.normal(0, 0.3, 3))
        key_color = np.array(CLAY_ALBEDO) + (0 if fixed else rng.normal(0, 0.04, 3))
        key = pyrender.PointLight(color=np.clip(key_color, 0, 1).tolist(),
                                  intensity=16.0 if fixed else rng.uniform(10.0, 22.0))
        scene.add(key, pose=_pose_at(key_pos))
        fill = pyrender.PointLight(color=[1.0, 1.0, 1.0],
                                   intensity=2.0 if fixed else rng.uniform(1.0, 4.0))
        scene.add(fill, pose=_pose_at(-eye * 0.8))
        del cam_node

        color, depth = renderer.render(scene)
        mask = np.isfinite(depth) & (depth > 0)
        tag = f"{i:03d}"
        Image.fromarray(color[..., :3]).save(f"{out_base}_{tag}.png")
        np.save(f"{out_base}_{tag}_depth.npy", depth.astype(np.float32))
        np.save(f"{out_base}_{tag}_mask.npy", mask)
        views.append({"id": i, "K": K, "R": c["R"], "t": c["t"],
                      "eye": c["eye"], "depth_far": 20.0})
    return views


def _pose_at(pos):
    T = np.eye(4); T[:3, 3] = pos
    return T


def assign_split(name, ratios=(0.8, 0.1, 0.1)):
    h = int(hashlib.md5(name.encode()).hexdigest(), 16) / 16 ** 32
    if h < ratios[0]:
        return "train"
    if h < ratios[0] + ratios[1]:
        return "val"
    return "test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--views", type=int, default=VIEWS_DEFAULT)
    ap.add_argument("--res", type=int, default=RES_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fixed_lights", action="store_true",
                    help="fixed lighting for ablation control")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    renderer = pyrender.OffscreenRenderer(args.res, args.res)

    plys = sorted(p for p in os.listdir(args.in_dir) if p.lower().endswith(".ply"))
    if args.limit:
        plys = plys[: args.limit]
    print(f"待渲染: {len(plys)} 块板 × {args.views} 视角 @{args.res}²", flush=True)

    ok, fail = 0, 0
    for n, fname in enumerate(plys):
        t0 = time.time()
        out_base = os.path.join(args.out_dir, os.path.splitext(fname)[0])
        if os.path.exists(out_base + "_000_meta.json"):
            ok += 1
            continue
        try:
            mesh = trimesh.load(os.path.join(args.in_dir, fname),
                                force="mesh", process=False)
            center = mesh.bounding_box.centroid
            R_bound = float(mesh.bounding_sphere.primitive.radius)
            mesh.apply_translation(-center)
            mesh.apply_scale(1.0 / R_bound)
            mesh.visual = trimesh.visual.TextureVisuals(
                material=trimesh.visual.material.PBRMaterial(
                    baseColorFactor=[int(CLAY_ALBEDO[0] * 255),
                                     int(CLAY_ALBEDO[1] * 255),
                                     int(CLAY_ALBEDO[2] * 255), 255],
                    metallicFactor=0.0,
                    roughnessFactor=float(0.7)))
            mesh_py = pyrender.Mesh.from_trimesh(mesh, smooth=True)

            rng = np.random.default_rng(args.seed + n)
            cams = ring_cameras(rng, 2.2, args.views)
            views = render_one(renderer, mesh_py, cams, args.res, rng, out_base,
                               fixed=args.fixed_lights)
            meta = {"ply": fname, "n_verts": int(len(mesh.vertices)),
                    "n_faces": int(len(mesh.faces)),
                    "normalize_center": center.tolist(),
                    "normalize_scale": R_bound,
                    "split": assign_split(os.path.splitext(fname)[0]),
                    "group": os.path.splitext(fname)[0],
                    "seed": args.seed + n, "views": views}
            json.dump(meta, open(out_base + "_000_meta.json", "w"), indent=1)
            ok += 1
            print(f"[{n+1}/{len(plys)}] {fname} ok "
                  f"({len(mesh.faces)} faces, {time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            fail += 1
            print(f"[{n+1}/{len(plys)}] {fname} FAIL: {e}", flush=True)

    json.dump({"res": args.res, "views": args.views, "seed": args.seed,
               "camera": "ring, az uniform+jitter, el 25-65deg, r=2.2R",
               "fov_y_deg": FOV_Y_DEG, "albedo": CLAY_ALBEDO,
               "lights": ("fixed key/fill/ambient" if args.fixed_lights else
                          "per-view random key/fill point lights + ambient"),
               "split_rule": "md5(name)->train/val/test 8/1/1",
               "ok": ok, "fail": fail},
              open(os.path.join(args.out_dir, "protocol.json"), "w"), indent=1)
    print(f"完成: ok={ok} fail={fail}", flush=True)


if __name__ == "__main__":
    main()
