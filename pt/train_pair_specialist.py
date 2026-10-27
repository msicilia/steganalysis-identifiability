"""Lightweight-CNN specialist, cover vs one algorithm (BinaryCNN + SRM, the detector's
architecture). Train on TRAIN, select on FUSION, report on TEST.

Outputs: MODEL_DIR/pt_spec_<algo>_<aXXX>_s<seed>.pt and .json.

Usage:  python train_pair_specialist.py --algo HILL --alpha 0.4
"""
import sys, argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import config as C
from data import make_loader
from models import BinaryCNN
from engine import train, report_binary, set_seeds, write_metrics, DEVICE


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
    print(f"== Specialist COVER vs {args.algo} | alpha={args.alpha} | seed={args.seed} | dev={DEVICE}")

    tr = make_loader(d["train"],  classes, C.BATCH_SIZE, binary_cover_idx=0, shuffle=True,  num_workers=args.workers)
    va = make_loader(d["fusion"], classes, C.BATCH_SIZE, binary_cover_idx=0, shuffle=False, num_workers=args.workers,
                     subset="select")   # [V3] the meta half trains the ensemble

    model = BinaryCNN(in_ch=C.CHANNELS, use_srm=True, dropout=C.DROPOUT)
    model, best = train(model, tr, va, task="binary", class_weights=True,
                        epochs=(1 if args.smoke else C.EPOCHS_BIN), lr=C.LR,
                        monitor="auc", smoke=args.smoke)
    print(f"best epoch={best['epoch']} val_auc={best['score']:.4f}")

    # Smoke runs never touch TEST: it is evaluated once, by the real run.
    if args.smoke:
        print("[smoke] OK"); return
    te = make_loader(d["test"],   classes, C.BATCH_SIZE, binary_cover_idx=0, shuffle=False, num_workers=args.workers)
    res = report_binary(model, te, tag=f"COVER vs {args.algo} a={args.alpha} seed={args.seed}")
    out = C.MODEL_DIR / f"pt_spec_{args.algo}_{C.alpha_tag(args.alpha)}_s{args.seed}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    write_metrics(out.with_suffix(".json"), args, best, res)
    print("saved:", out)


if __name__ == "__main__":
    main()
