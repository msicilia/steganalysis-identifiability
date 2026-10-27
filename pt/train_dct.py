"""DCT-domain CNN (DCTNet) on the .npy cubes of gen_dct.py, the control for the JPEG
representation. Modes: --mode binary (cover vs any stego), --mode family (K classes),
--mode pair --algo X (cover vs X).

Usage:  STEG_DOMAIN=jpeg uv run python pt/train_dct.py --mode binary --alpha 0.4
"""

import sys, argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import config as C
from data import make_dct_loader
from models import DCTNet
from engine import train, report_binary, report_multiclass, set_seeds, write_metrics, DEVICE


def dct_dirs(alpha):
    root = C.DATASETS / f"{C.DATA_PREFIX}_{C.alpha_tag(alpha)}_DCT"
    return {s: root / s for s in ("train", "fusion", "test")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["binary", "family", "pair"], required=True)
    ap.add_argument("--algo", default=None, help="algorithm (pair mode)")
    ap.add_argument("--alpha", type=float, default=C.FAMILY_ID_PAYLOAD)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    assert C.DOMAIN == "jpeg", "Run with STEG_DOMAIN=jpeg"
    set_seeds(args.seed)
    d = dct_dirs(args.alpha)
    cover = C.COVER_NAME

    if args.mode == "family":
        classes, cidx, out_dim, task = C.CLASS_FOLDERS, None, len(C.CLASS_FOLDERS), "multiclass"
    elif args.mode == "pair":
        classes, cidx, out_dim, task = [cover, args.algo], 0, 1, "binary"
    else:  # binary: cover vs any stego
        classes, cidx, out_dim, task = C.CLASS_FOLDERS, C.CLASS_FOLDERS.index(cover), 1, "binary"

    print(f"== DCT {args.mode}{'/'+args.algo if args.algo else ''} | alpha={args.alpha} "
          f"| classes={classes} | dev={DEVICE}")
    tr = make_dct_loader(d["train"],  classes, C.BATCH_SIZE, binary_cover_idx=cidx, shuffle=True,  num_workers=args.workers)
    va = make_dct_loader(d["fusion"], classes, C.BATCH_SIZE, binary_cover_idx=cidx, shuffle=False, num_workers=args.workers)
    te = make_dct_loader(d["test"],   classes, C.BATCH_SIZE, binary_cover_idx=cidx, shuffle=False, num_workers=args.workers)

    model = DCTNet(out_dim=out_dim, in_ch=64, dropout=C.DROPOUT)
    epochs = 1 if args.smoke else (C.EPOCHS_SPEC if task == "multiclass" else C.EPOCHS_BIN)
    model, best = train(model, tr, va, task=task, n_classes=out_dim if task == "multiclass" else 2,
                        class_weights=True, label_smoothing=(C.LABEL_SMOOTHING if task == "multiclass" else 0.0),
                        epochs=epochs, lr=C.LR, monitor=("auc" if task == "binary" else "loss"),
                        smoke=args.smoke)

    # Smoke runs never touch TEST: it is evaluated once, by the real run.
    if args.smoke:
        print("[smoke] OK"); return
    tag = f"{args.mode}" + (f"_{args.algo}" if args.algo else "")
    if task == "binary":
        res = report_binary(model, te, tag=f"DCT {tag} a={args.alpha}")
    else:
        res = report_multiclass(model, te, classes, tag=f"DCT {tag} a={args.alpha}")
    out = C.MODEL_DIR / f"pt_dct_{tag}_{C.alpha_tag(args.alpha)}_s{args.seed}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    write_metrics(out.with_suffix(".json"), args, best, res)
    print("saved:", out)


if __name__ == "__main__":
    main()
