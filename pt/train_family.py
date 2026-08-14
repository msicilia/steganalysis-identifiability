"""
Family identification (multiclass) at FIXED payload.

6 classes: COVER + {HILL, HUGO, MiPOD, S-UNIWARD, WOW}. Fixed payload (0.4 by
default) so the network learns WHICH algorithm, not HOW MUCH payload. SRM front-end ON.
Selection on FUSION; report (confusion matrix) ONCE on TEST.

Usage:  python train_family.py --alpha 0.4
"""
import sys, argparse, json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import config as C
from data import make_loader
from models import SpecialistCNN
from engine import train, report_multiclass, set_seeds, DEVICE


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
    va = make_loader(d["fusion"], names, C.BATCH_SIZE, shuffle=False, num_workers=args.workers)
    te = make_loader(d["test"],   names, C.BATCH_SIZE, shuffle=False, num_workers=args.workers)

    model = SpecialistCNN(num_classes=len(names), in_ch=C.CHANNELS,
                          use_srm=args.use_srm, dropout=C.DROPOUT, img_size=C.IMG_SIZE[0])
    model, best = train(model, tr, va, task="multiclass", n_classes=len(names),
                        class_weights=C.USE_CLASS_WEIGHTS, label_smoothing=C.LABEL_SMOOTHING,
                        epochs=C.EPOCHS_SPEC, lr=C.LR, monitor="loss", smoke=args.smoke)
    print(f"best epoch={best['epoch']} val_loss={best['score']:.4f}")

    res = report_multiclass(model, te, names, tag=f"family a={args.alpha} seed={args.seed}")
    if args.smoke:
        print("[smoke] OK"); return
    out = C.MODEL_DIR / f"pt_family_{C.alpha_tag(args.alpha)}_s{args.seed}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    (C.MODEL_DIR / f"pt_family_{C.alpha_tag(args.alpha)}_s{args.seed}.json").write_text(json.dumps(res, indent=2))
    print("saved:", out)


if __name__ == "__main__":
    main()
