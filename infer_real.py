"""Route A: real Met CC0 photo pair (view5 frontal + view6 oblique) -> CuneiSplat.

Pose is unknown; we approximate (el/az sweep + in-plane roll variants) and pick
the combo maximizing self-reconstruction PSNR at the two context poses. Then
render novel views + depth for the winner.

Usage:
  cd /root/autodl-tmp/proj && /root/miniconda3/envs/cunei/bin/python infer_real.py \
      --experiment cunei_geo --ckpt /root/ckpt_t2_cunei_geo.ckpt \
      --photos /root/real_photos --out /root/real_out
"""
import argparse, dataclasses, json, os, sys
import numpy as np
import torch
from PIL import Image as PILImage

sys.path.insert(0, "/root/autodl-tmp/proj")
sys.path.insert(0, "/root/autodl-tmp/proj/pixelsplat")

RADIUS = 2.2
FX_N = 309.0193 / 256.0
NEAR, FAR = 0.8, 5.0


def lookat(el_deg, az_deg):
    """c2w (OpenCV: +z forward toward origin, +y image-down).
    World y points DOWN; elevation>0 puts the camera above the tablet."""
    el, az = np.radians(el_deg), np.radians(az_deg)
    c = RADIUS * np.array([np.cos(el) * np.cos(az), -np.sin(el),
                           np.cos(el) * np.sin(az)])
    z = -c / np.linalg.norm(c)
    down = np.array([0.0, 1.0, 0.0])
    y = down - np.dot(down, z) * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    E = np.eye(4)
    E[:3, 0], E[:3, 1], E[:3, 2], E[:3, 3] = x, y, z, c
    return E


def intr():
    return np.array([[FX_N, 0, 0.5], [0, FX_N, 0.5], [0, 0, 1]], dtype=np.float64)


def build_model(experiment, renders):
    from hydra import compose, initialize
    with initialize(version_base=None, config_path="pixelsplat/config"):
        cfg = compose(config_name="main", overrides=[
            f"+experiment={experiment}",
            f"dataset.roots=[{renders}]",
            "wandb.mode=disabled",
        ])
    from src.global_cfg import set_cfg
    from src.config import load_typed_root_config
    from src.model.model_wrapper import ModelWrapper
    from src.model.encoder import get_encoder
    from src.model.decoder import get_decoder
    from src.loss import get_losses
    from src.misc.step_tracker import StepTracker
    set_cfg(cfg)
    cfg = load_typed_root_config(cfg)
    encoder, ev = get_encoder(cfg.model.encoder)
    return ModelWrapper(cfg.optimizer, cfg.test, cfg.train, encoder, ev,
                        get_decoder(cfg.model.decoder, cfg.dataset),
                        get_losses(cfg.loss), StepTracker())


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) / 255 - b.astype(np.float64) / 255) ** 2)
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def control(wrapper, args):
    """Manual-batch plumbing check on one synthetic test scene (known poses).

    Reproduces the test protocol (context [0,4], targets [3,9]) through the
    same manual-batch path used for real photos; also verifies lookat()
    against the scene's true camera poses."""
    import glob as _glob
    meta_path = None
    for p in sorted(_glob.glob(os.path.join(args.renders, "*_000_meta.json"))):
        m = json.load(open(p))
        if m.get("split") == "test" and len(m.get("views", [])) >= 10:
            meta_path = p
            meta = m
            break
    scene = os.path.basename(meta_path).replace("_000_meta.json", "")
    views = meta["views"]
    print(f"control scene={scene} split={meta.get('split')} "
          f"n_views={len(views)}", flush=True)

    # lookat convention check: positions from meta vs lookat(fitted el,az)
    pos_err = 0.0
    radii = []
    for v in views:
        w2c = np.eye(4)
        w2c[:3, :3] = np.asarray(v["R"], dtype=np.float64)
        w2c[:3, 3] = np.asarray(v["t"], dtype=np.float64)
        c2w = np.linalg.inv(w2c)
        c = c2w[:3, 3]
        r = np.linalg.norm(c)
        radii.append(r)
        el = np.degrees(np.arcsin(np.clip(-c[1] / r, -1, 1)))
        az = np.degrees(np.arctan2(c[2], c[0]))
        c_fit = lookat(el, az)[:3, 3]
        pos_err = max(pos_err, float(np.linalg.norm(c - c_fit)))
    print(f"training camera radii: min={min(radii):.3f} max={max(radii):.3f} "
          f"(lookat uses {RADIUS})", flush=True)
    print(f"lookat position check: max |c - c_fit| = {pos_err:.4f} "
          f"(radius {RADIUS})", flush=True)

    def load_view(i):
        v = views[i]
        w2c = np.eye(4)
        w2c[:3, :3] = np.asarray(v["R"], dtype=np.float32)
        w2c[:3, 3] = np.asarray(v["t"], dtype=np.float32)
        extr = np.linalg.inv(w2c).astype(np.float32)
        K = np.asarray(v["K"], dtype=np.float32)
        intr = np.array([[K[0, 0] / 256, 0, K[0, 2] / 256],
                         [0, K[1, 1] / 256, K[1, 2] / 256],
                         [0, 0, 1]], dtype=np.float32)
        return extr, intr, np.asarray(
            PILImage.open(os.path.join(
                args.renders, f"{scene}_{i:03d}.png")).convert("RGB"))[..., :3]

    ce, ci, cimg0 = load_view(0)
    _, _, cimg4 = load_view(4)
    ctx = {
        "extrinsics": torch.from_numpy(np.stack([ce, load_view(4)[0]]))[None].cuda(),
        "intrinsics": torch.from_numpy(np.stack([ci, load_view(4)[1]]))[None].cuda(),
        "image": torch.stack([
            torch.from_numpy(cimg0).to(torch.get_default_dtype()) / 255.0,
            torch.from_numpy(cimg4).to(torch.get_default_dtype()) / 255.0,
        ])[None].permute(0, 1, 4, 2, 3).contiguous().cuda(),
        "near": torch.full((1, 2), NEAR).cuda(),
        "far": torch.full((1, 2), FAR).cuda(),
    }
    batch = {"context": ctx,
             "target": {k: (v.clone() if torch.is_tensor(v) else v)
                        for k, v in ctx.items()},
             "scene": [scene]}
    if hasattr(wrapper, "data_shim"):
        batch = wrapper.data_shim(batch)
    with torch.no_grad():
        gaussians = wrapper.encoder(batch["context"], wrapper.global_step,
                                    deterministic=False)
    tE = np.stack([load_view(i)[0] for i in (3, 9)])
    tK = np.stack([load_view(i)[1] for i in (3, 9)])
    with torch.no_grad():
        out = wrapper.decoder.forward(
            gaussians,
            torch.from_numpy(tE)[None].cuda(), torch.from_numpy(tK)[None].cuda(),
            torch.full((1, 2), NEAR).cuda(), torch.full((1, 2), FAR).cuda(),
            (256, 256), depth_mode="depth")
    col = out.color if hasattr(out, "color") else out[0]
    cols = (col[0].clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1) * 255
            ).astype(np.uint8)
    ps = [psnr(cols[k], load_view(i)[2]) for k, i in enumerate((3, 9))]
    print(f"control PSNR targets[3,9]: {ps[0]:.2f} / {ps[1]:.2f} dB "
          f"(paper ID+E ~16.5)", flush=True)
    for k, i in enumerate((3, 9)):
        PILImage.fromarray(cols[k]).save(
            os.path.join(args.out, f"control_tgt{i}.png"))
    print("CONTROL_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="cunei_geo")
    ap.add_argument("--ckpt", default="/root/ckpt_t2_cunei_geo.ckpt")
    ap.add_argument("--renders", default="/root/autodl-tmp/data/renders_full")
    ap.add_argument("--photos", default="/root/real_photos")
    ap.add_argument("--out", default="/root/real_out")
    ap.add_argument("--control", action="store_true",
                    help="plumbing check on a synthetic scene (known poses)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    # 训练已结束, GPU 独占; 硬顶 0.11 已无必要且会误伤
    torch.cuda.set_per_process_memory_fraction(0.85, 0)

    wrapper = build_model(args.experiment, args.renders)
    sd = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    state = sd.get("state_dict", sd)
    missing, unexpected = wrapper.load_state_dict(state, strict=False)
    print(f"ckpt: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    wrapper = wrapper.cuda().eval()

    def load_img(p):
        a = np.asarray(PILImage.open(p).convert("RGB"))[..., :3]
        return torch.from_numpy(a).to(torch.get_default_dtype()) / 255.0

    if args.control:
        return control(wrapper, args)

    imgA = load_img(f"{args.photos}/realA.png")
    imgB = load_img(f"{args.photos}/realB.png")
    imgB180 = load_img(f"{args.photos}/realB180.png")

    def batch_ctx(img_b, el_b, az_b, ext_b=None):
        E1 = lookat(10, 0)
        E2 = ext_b if ext_b is not None else lookat(el_b, az_b)
        E = np.stack([E1, E2]).astype(np.float32)
        K = np.stack([intr(), intr()]).astype(np.float32)
        imgs = torch.stack([imgA, img_b])[None].permute(0, 1, 4, 2, 3).contiguous()
        ctx = {
            "extrinsics": torch.from_numpy(E)[None].cuda(),
            "intrinsics": torch.from_numpy(K)[None].cuda(),
            "image": imgs.cuda(),
            "near": torch.full((1, 2), NEAR).cuda(),
            "far": torch.full((1, 2), FAR).cuda(),
        }
        # data_shim 需要完整的 target 键（patch shim 会处理 target 视角）
        tgt = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in ctx.items()}
        batch = {"context": ctx, "target": tgt, "scene": ["met321699"]}
        if hasattr(wrapper, "data_shim"):
            batch = wrapper.data_shim(batch)
        return batch

    def encode(batch, det):
        with torch.no_grad():
            return wrapper.encoder(batch["context"], wrapper.global_step,
                                   deterministic=det)

    def render_poses(gaussians, poses):
        def to_ext(p):
            return p if isinstance(p, np.ndarray) else lookat(*p)
        E = np.stack([to_ext(p) for p in poses]).astype(np.float32)
        K = np.stack([intr() for _ in poses]).astype(np.float32)
        n = len(poses)
        with torch.no_grad():
            out = wrapper.decoder.forward(
                gaussians,
                torch.from_numpy(E)[None].cuda(),
                torch.from_numpy(K)[None].cuda(),
                torch.full((1, n), NEAR).cuda(),
                torch.full((1, n), FAR).cuda(),
                (256, 256), depth_mode="depth")
        col = out.color if hasattr(out, "color") else out[0]
        dep = out.depth if hasattr(out, "depth") else out[1]
        return (col[0].clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1) * 255
                ).astype(np.uint8), dep[0].cpu().numpy()

    combos = []
    for img_name, img in [("B", imgB), ("B180", imgB180)]:
        for el, az in [(15, 60), (15, -60), (35, 60), (35, -60),
                       (60, 0), (50, 25), (50, -25)]:
            combos.append((img_name, img, el, az, None))
    # 轮廓拟合出的测量位姿（若存在）作为第 15 组
    sil_path = os.path.join(args.photos, "pose_sil.json")
    if os.path.exists(sil_path):
        sil = json.load(open(sil_path))
        sil_ext = np.asarray(sil["c2w_B"], dtype=np.float32)
        combos.append(("SIL", imgB, sil.get("batch_el", 0),
                       sil.get("batch_az", 0), sil_ext))
        print(f"pose_sil loaded: IoU_B={sil.get('iou_B')} "
              f"el={sil.get('batch_el')} az={sil.get('batch_az')} "
              f"sep={sil.get('axis_sep_deg')}", flush=True)

    sweep = []
    best = None
    gt_a = np.asarray(PILImage.open(f"{args.photos}/realA.png").convert("RGB"))
    for name, img, el, az, ext in combos:
        batch = batch_ctx(img, el, az, ext_b=ext)
        gaussians = encode(batch, det=True)
        cols, _ = render_poses(
            gaussians, [(10, 0), ext if ext is not None else (el, az)])
        gt_b = np.asarray(PILImage.open(
            f"{args.photos}/realB.png" if name in ("B", "SIL")
            else f"{args.photos}/realB180.png").convert("RGB"))
        pA, pB = psnr(cols[0], gt_a), psnr(cols[1], gt_b)
        n_g = int(gaussians.means.shape[1])
        rec = {"variant": name, "el": el, "az": az,
               "self_psnr_A": round(float(pA), 2),
               "self_psnr_B": round(float(pB), 2),
               "self_mean": round(float((pA + pB) / 2), 2), "n_gauss": n_g}
        sweep.append(rec)
        print(json.dumps(rec), flush=True)
        if best is None or rec["self_mean"] > best["self_mean"]:
            best = rec
        del gaussians, batch
        torch.cuda.empty_cache()
    print("BEST:", json.dumps(best), flush=True)

    # 用最佳组合出 novel views（确定性模式 + 随机采样模式各一套）
    novel = [(25, -30), (25, 30), (12, -25), (12, 25),
             (45, -15), (45, 15), (55, 0), (2, 0)]
    ext_best = None
    if best["variant"] == "SIL":
        ext_best = sil_ext
    img_best = imgB if best["variant"] in ("B", "SIL") else imgB180
    for tag, det in [("det", True), ("stoch", False)]:
        batch = batch_ctx(img_best, best["el"], best["az"], ext_b=ext_best)
        gaussians = encode(batch, det)
        cols, deps = render_poses(gaussians, novel)
        d = os.path.join(args.out, tag)
        os.makedirs(d, exist_ok=True)
        for i, (e, a) in enumerate(novel):
            PILImage.fromarray(cols[i]).save(
                f"{d}/color_{i}_el{e}_az{a}.png")
            dp = deps[i]
            np.save(f"{d}/depthraw_{i}_el{e}_az{a}.npy", dp.astype(np.float32))
            m = np.isfinite(dp) & (dp > 0)
            vis = np.zeros((*dp.shape, 3), dtype=np.uint8)
            if m.sum() > 50:
                span = np.ptp(dp[m])
                v = (dp[m] - dp[m].min()) / max(span, 1e-6)
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.cm as cm
                vis[m] = (cm.turbo(v)[..., :3] * 255).astype(np.uint8)
            PILImage.fromarray(vis).save(f"{d}/depth_{i}_el{e}_az{a}.png")
        np.savez_compressed(
            f"{d}/gauss.npz",
            means=gaussians.means.detach()[0].cpu().numpy().astype(np.float32),
            opacities=gaussians.opacities.detach()[0].cpu().numpy().astype(np.float32))
        print(f"saved {tag} renders", flush=True)

    json.dump({"sweep": sweep, "best": best, "novel_poses": novel},
              open(os.path.join(args.out, "sweep.json"), "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
