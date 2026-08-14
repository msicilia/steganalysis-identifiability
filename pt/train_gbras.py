"""
GBRAS-Net (PyTorch), faithful protocol:
  - binary COVER vs ONE algorithm,
  - 256x256 input: each 512 image -> 4 non-overlapping blocks,
  - 30 official SRM kernels + Tanh3.
Selection on FUSION; report (block level) on TEST.

Usage:  python train_gbras.py --alpha 0.4 --algo S-UNIWARD
"""
import sys, argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import config as C
from data import make_block_loader
from models import GBRASNet
from engine import train, set_seeds, DEVICE, _predict
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=C.ACTIVE_ALPHA)
    ap.add_argument("--algo", default="S-UNIWARD")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    set_seeds(args.seed)

    d = C.dirs_for(args.alpha)
    print(f"== GBRAS-Net | COVER vs {args.algo} | alpha={args.alpha} | dev={DEVICE}")
    bs = 64   # 256x256 fits easily on the GB10; fewer steps/epoch = faster
    tr = make_block_loader(d["train"],  C.COVER_NAME, args.algo, bs, shuffle=True,  num_workers=args.workers)
    va = make_block_loader(d["fusion"], C.COVER_NAME, args.algo, bs, shuffle=False, num_workers=args.workers)
    te = make_block_loader(d["test"],   C.COVER_NAME, args.algo, bs, shuffle=False, num_workers=args.workers)

    model = GBRASNet(n_classes=2)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    model, best = train(model, tr, va, task="multiclass", n_classes=2,
                        class_weights=False, epochs=(1 if args.smoke else args.epochs),
                        lr=1e-3, monitor="acc", smoke=args.smoke)

    probs, y = _predict(model, te, "multiclass")
    p1 = probs[:, 1]; pred = (p1 >= 0.5).astype(int)
    print(f"\n== TEST GBRAS COVER vs {args.algo} a={args.alpha} ==")
    print(classification_report(y, pred, target_names=["COVER", args.algo], digits=4, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    print(f"AUC={roc_auc_score(y, p1):.4f}  P_D={tp/(tp+fn):.4f}  P_FA={fp/(fp+tn):.4f}")
    if args.smoke:
        print("[smoke] OK"); return
    out = C.MODEL_DIR / f"pt_gbras_{args.algo}_{C.alpha_tag(args.alpha)}_s{args.seed}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    print("saved:", out)


if __name__ == "__main__":
    main()
