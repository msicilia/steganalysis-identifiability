"""
Per-algorithm specialist (COVER vs ONE algorithm).

Each specialist solves a sub-problem (cover vs a single embedder). These are the
base learners of the modular ensemble. Same architecture as the generalist
detector (BinaryCNN + SRM).

Protocol: train on TRAIN, select on FUSION, report on TEST.

Usage:  python train_pair_specialist.py --algo HILL --alpha 0.4
"""
import sys, argparse, json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import config as C
from data import make_loader
from models import BinaryCNN
from engine import train, report_binary, set_seeds, DEVICE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, help="stego algorithm of the specialist")
    ap.add_argument("--alpha", type=float, default=C.FAMILY_ID_PAYLOAD)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    set_seeds(args.seed)

    d = C.dirs_for(args.alpha)
    classes = [C.COVER_NAME, args.algo]          # cover=0, algo=1  (label mapping)
    print(f"== Specialist COVER vs {args.algo} | alpha={args.alpha} | dev={DEVICE}")

    tr = make_loader(d["train"],  classes, C.BATCH_SIZE, binary_cover_idx=0, shuffle=True,  num_workers=args.workers)
    va = make_loader(d["fusion"], classes, C.BATCH_SIZE, binary_cover_idx=0, shuffle=False, num_workers=args.workers)
    te = make_loader(d["test"],   classes, C.BATCH_SIZE, binary_cover_idx=0, shuffle=False, num_workers=args.workers)

    model = BinaryCNN(in_ch=C.CHANNELS, use_srm=True, dropout=C.DROPOUT)
    model, best = train(model, tr, va, task="binary", class_weights=True,
                        epochs=(1 if args.smoke else C.EPOCHS_BIN), lr=C.LR,
                        monitor="auc", smoke=args.smoke)
    res = report_binary(model, te, tag=f"COVER vs {args.algo} a={args.alpha}")
    if args.smoke:
        print("[smoke] OK"); return
    out = C.MODEL_DIR / f"pt_spec_{args.algo}_{C.alpha_tag(args.alpha)}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    (C.MODEL_DIR / f"pt_spec_{args.algo}_{C.alpha_tag(args.alpha)}.json").write_text(json.dumps(res, indent=2))
    print("saved:", out)


if __name__ == "__main__":
    main()
