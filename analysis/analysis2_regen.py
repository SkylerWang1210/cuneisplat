"""带固定种子的缓存再生成启动器：包装 prune_render_eval.py。

用法: python analysis2_regen.py <experiment> <ckpt_subdir> <renders_dir> <out_dir>
种子: torch.manual_seed(0) + torch.cuda.manual_seed_all(0)（推理 draw 可复现）
"""
import os
import sys

os.chdir("/root/autodl-tmp/proj")
sys.path.insert(0, "/root/autodl-tmp/proj")
sys.path.insert(0, "/root/autodl-tmp/proj/pixelsplat")

import torch

EXP, CKPT_SUB, RENDERS, OUT_DIR = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
CKPT = f"/root/autodl-tmp/proj/pixelsplat/outputs/{CKPT_SUB}/checkpoints/epoch=465-step=20000.ckpt"

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

sys.argv = ["prune_render_eval.py",
            "--experiment", EXP,
            "--ckpt", CKPT,
            "--renders", RENDERS,
            "--taus", "0.05,0.1,0.2,0.3",
            "--out_dir", OUT_DIR]
print(f"[regen] exp={EXP} ckpt={CKPT_SUB} renders={RENDERS} out={OUT_DIR}", flush=True)
src = open("/root/autodl-tmp/proj/prune_render_eval.py").read()
exec(src, {"__name__": "__main__"})
