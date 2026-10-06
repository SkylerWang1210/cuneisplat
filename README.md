# CuneiSplat

Code and benchmark generation pipeline for

> **CuneiSplat: Feed-Forward Sparse-View 3D Gaussian Reconstruction of Cuneiform Tablets with Geometry Supervision**
> Tian Wang, Jiale Li, Ziyi Huang (Hubei University)
> Submitted to ICVISP 2026, Track XI

CuneiSplat reconstructs a cuneiform tablet as a set of 3D Gaussians from two
photographs in a single feed-forward pass (0.15 s). It trains an unmodified
pixelSplat encoder with geometry supervision on rendered views, and adds an
optional opacity-weighted surface-distance term (ID+E+S). At inference,
opacity pruning removes low-opacity Gaussians that mask the geometric gain.

## Repository contents

| Path | Contents |
|---|---|
| `pixelsplat/` | Modified pixelSplat backbone. Adds two losses: `cunei_geometry` (rendered opacity-weighted depth accumulation regressed onto ground-truth depth, Eq. 3 of the paper) and `cunei_surface` (per-Gaussian opacity-weighted surface distance, Eq. 4). Experiment configs: `cunei_base` (ID), `cunei_geo` (ID+E), `cunei_geo_surf` (ID+E+S). |
| `render_pipeline.py` | Benchmark generation: renders multi-view RGB, linear depth, masks, and per-view lighting/camera protocol records from HeiCuBeDa mesh scans (pyrender/EGL). |
| `run_3dgs_baseline.py`, `render_3dgs_views.py`, `eval_3dgs_all.py`, `score_3dgs.py` | Per-scene vanilla 3DGS references at 2-view and 16-view budgets. |
| `prune_render_eval.py`, `prune_render_eval_val.py` | Main evaluation protocol: per-tablet encoding, opacity-pruning sweep over tau, measured PSNR/SSIM/LPIPS, masked depth MAE, and symmetric Chamfer distance under a fixed protocol; also dumps Gaussian caches for offline analysis. |
| `eval_chamfer.py`, `eval_depth.py`, `compute_photo_metrics.py` | Individual metric implementations. |
| `dump_gt20k.py`, `render_depth_npy.py` | Ground-truth preparation. |
| `gap_sweep.py`, `mechanism_analysis.py`, `offline_robustness.py`, `analysis2_readout.py` | Paper analyses: context-pair geometry sweep, opacity-stratified mechanism, protocol-variant robustness, readout sensitivity. |
| `infer_real.py` | Qualitative transfer to unposed real photographs (pose search over camera configurations). |
| `analysis/` | Statistics scripts reproducing the paper's numbers: threshold grid, oracle selection, opacity-band stratification, paired Wilcoxon statistics, weight sweep, seed comparison. |

## Setup

```bash
conda create -n cunei python=3.10 -y
conda activate cunei
pip install torch==2.3.* --index-url https://download.pytorch.org/whl/cu121
cd pixelsplat && pip install -e . && cd ..
pip install -r pixelsplat/requirements.txt
```

Training uses a single 32 GB GPU; batch size 4 at 256x256 input resolution for
20,000 steps (about 8.5 GPU-hours per run).

## Benchmark generation

Download the HeiCuBeDa corpus (CC BY-SA 4.0) and render the training views:

```bash
PYOPENGL_PLATFORM=egl python render_pipeline.py --in_dir <HeiCuBeDa_meshes> --out_dir <renders>
```

Each rendered view carries exact per-view ground-truth depth and a visibility
mask; lighting randomization follows the protocol described in the paper
(Sec. 3). Tablet-level train/validation/test splits (169/22/9) are produced by
the hash-based split recorded in the protocol manifest.

## Training

The three learned models of the paper:

```bash
cd pixelsplat
python -m src.main +experiment=cunei_base      # ID   (photometric only)
python -m src.main +experiment=cunei_geo       # ID+E (depth supervision, lambda = 0.5)
python -m src.main +experiment=cunei_geo_surf  # ID+E+S (adds surface term, gamma = 0.5)
```

Both loss weights retain their default value 0.5 without tuning; only the
inference-time pruning threshold is selected on the 22 validation tablets.

## Evaluation

```bash
python prune_render_eval.py --experiment cunei_geo_surf --ckpt <path/to/ckpt> \
    --renders <renders> --limit 9
```

This reproduces Table 2 of the paper for one model, including the tau grid,
depth MAE, and symmetric Chamfer distance. The `analysis/` scripts then
recompute paired statistics, oracle selection, and opacity-band
stratification from the dumped Gaussian caches.

## License

- Code: MIT (see `LICENSE`). The `pixelsplat/` directory is a fork of
  [pixelSplat](https://github.com/dcharatan/pixelsplat) and retains its
  upstream license.
- Benchmark generation and protocol files derived from the HeiCuBeDa corpus
  inherit its CC BY-SA 4.0 license.

## Citation

```bibtex
@misc{wang2026cuneisplat,
  title={CuneiSplat: Feed-Forward Sparse-View 3D Gaussian Reconstruction of
         Cuneiform Tablets with Geometry Supervision},
  author={Wang, Tian and Li, Jiale and Huang, Ziyi},
  year={2026},
  note={Submitted to ICVISP 2026, Track XI}
}
```
