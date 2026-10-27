"""Six-class identification (COVER + HILL, HUGO, MiPOD, S-UNIWARD, WOW) with the lightweight
CNN at a fixed payload. Selection on FUSION, confusion matrix on TEST.

Outputs: MODEL_DIR/pt_family_<aXXX>_s<seed>.pt and .json.

Usage:  python train_family.py --alpha 0.4
"""
import sys, argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import config as C
from data import make_loader
from models import SpecialistCNN
from engine import train, report_multiclass, set_seeds, write_metrics, DEVICE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=C.FAMILY_ID_PAYLOAD)
    ap.add_argument("--no-srm", dest="use_srm", action="store_false")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    set_seeds(args.seed)

    d = C.dirs_for(args.alpha)
    names = C.CLASS_FOLDERS
    print(f"== Family-ID ({len(names)}c) | alpha={args.alpha} | srm={args.use_srm} | dev={DEVICE}")

    tr = make_loader(d["train"],  names, C.BATCH_SIZE, shuffle=True,  num_workers=args.workers)
    va = make_loader(d["fusion"], names, C.BATCH_SIZE, shuffle=False, num_workers=args.workers,
                     subset="select")   # [V3]

    model = SpecialistCNN(num_classes=len(names), in_ch=C.CHANNELS,
                          use_srm=args.use_srm, dropout=C.DROPOUT, img_size=C.IMG_SIZE[0])
    model, best = train(model, tr, va, task="multiclass", n_classes=len(names),
                        class_weights=C.USE_CLASS_WEIGHTS, label_smoothing=C.LABEL_SMOOTHING,
                        epochs=C.EPOCHS_SPEC, lr=C.LR, monitor="loss", smoke=args.smoke)
    print(f"best epoch={best['epoch']} val_loss={best['score']:.4f}")

    # Smoke runs never touch TEST: it is evaluated once, by the real run.
    if args.smoke:
        print("[smoke] OK"); return
    te = make_loader(d["test"],   names, C.BATCH_SIZE, shuffle=False, num_workers=args.workers)
    res = report_multiclass(model, te, names, tag=f"family a={args.alpha} seed={args.seed}")
    out = C.MODEL_DIR / f"pt_family_{C.alpha_tag(args.alpha)}_s{args.seed}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    write_metrics(out.with_suffix(".json"), args, best, res)
    print("saved:", out)


if __name__ == "__main__":
    main()
