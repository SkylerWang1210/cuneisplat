"""带固定种子的验证集评估启动器（stage=val 版）。用法同 regen。"""
import os
import sys

import torch

EXP, CKPT_SUB, RENDERS, OUT_DIR = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
CKPT = f"/root/autodl-tmp/proj/pixelsplat/outputs/{CKPT_SUB}/checkpoints/epoch=465-step=20000.ckpt"

os.chdir("/root/autodl-tmp/proj")
sys.path.insert(0, "/root/autodl-tmp/proj")
sys.path.insert(0, "/root/autodl-tmp/proj/pixelsplat")
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

sys.argv = ["prune_render_eval_val.py",
            "--experiment", EXP, "--ckpt", CKPT,
            "--renders", RENDERS, "--taus", "0.05,0.1,0.2,0.3",
            "--out_dir", OUT_DIR]
print(f"[regen-val] exp={EXP} ckpt={CKPT_SUB}", flush=True)
src = open("/root/autodl-tmp/proj/prune_render_eval_val.py").read()
exec(src, {"__name__": "__main__"})
