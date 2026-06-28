"""
Binary cover/stego detector (PyTorch).

  - SRM (high-pass) front-end ON by default; --no-srm for the ablation.
  - Selection/early-stopping on FUSION; final report ONCE on TEST.
  - Class weights applied (pos_weight) for the ~1:5 imbalance.

Usage:
    python train_binary.py --alpha 0.4
    python train_binary.py --alpha 0.4 --no-srm        # ablation
    python train_binary.py --alpha 0.4 --seed 1 --smoke # quick test
"""
import sys, argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))   # code/ -> only for config

import torch
import config as C
from data import make_loader, labels_of
from models import BinaryCNN
from engine import train, report_binary, set_seeds, DEVICE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=C.ACTIVE_ALPHA)
    ap.add_argument("--no-srm", dest="use_srm", action="store_false")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    set_seeds(args.seed)

    d = C.dirs_for(args.alpha)
    cover_idx = C.CLASS_FOLDERS.index(C.COVER_NAME)
    print(f"== Binary detector | alpha={args.alpha} | srm={args.use_srm} | seed={args.seed} | dev={DEVICE}")

    tr = make_loader(d["train"],  C.CLASS_FOLDERS, C.BATCH_SIZE, binary_cover_idx=cover_idx,
                     shuffle=True, num_workers=args.workers)
    va = make_loader(d["fusion"], C.CLASS_FOLDERS, C.BATCH_SIZE, binary_cover_idx=cover_idx,
                     shuffle=False, num_workers=args.workers)
    te = make_loader(d["test"],   C.CLASS_FOLDERS, C.BATCH_SIZE, binary_cover_idx=cover_idx,
                     shuffle=False, num_workers=args.workers)
    import numpy as np
    print("train label counts:", np.bincount(labels_of(tr)).tolist())

    model = BinaryCNN(in_ch=C.CHANNELS, use_srm=args.use_srm, dropout=C.DROPOUT)
    model, best = train(model, tr, va, task="binary", n_classes=2,
                        class_weights=C.USE_CLASS_WEIGHTS, epochs=C.EPOCHS_BIN,
                        lr=C.LR, monitor="auc", smoke=args.smoke)
    print(f"best epoch={best['epoch']} val_auc={best['score']:.4f}")

    tag = f"a={args.alpha} srm={args.use_srm} seed={args.seed}"
    report_binary(model, te, tag=tag)

    if args.smoke:
        print("[smoke] OK"); return
    name = C.alpha_tag(args.alpha) + ("" if args.use_srm else "_nosrm") + f"_s{args.seed}"
    out = C.MODEL_DIR / f"pt_binary_{name}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    print("saved:", out)


if __name__ == "__main__":
    main()
