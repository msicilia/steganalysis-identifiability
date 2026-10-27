"""Cover-vs-stego detector (lightweight CNN with SRM front end; --no-srm for the ablation).
Early stopping on FUSION, one evaluation on TEST, pos_weight for the class imbalance.

Outputs: MODEL_DIR/pt_binary_<aXXX>[_nosrm]_s<seed>.pt and .json.

Usage:
    python train_binary.py --alpha 0.4
    python train_binary.py --alpha 0.4 --no-srm
    python train_binary.py --alpha 0.4 --seed 1 --smoke
"""
import sys, argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))   # code/ -> only for config

import torch
import config as C
from data import make_loader, labels_of
from models import BinaryCNN
from engine import train, report_binary, set_seeds, write_metrics, DEVICE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=C.ACTIVE_ALPHA)
    ap.add_argument("--no-srm", dest="use_srm", action="store_false")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--init", default=None,
                    help="payload curriculum: start from this detector's weights (e.g. the "
                         "0.4 bpp model for the 0.2 bpp run) instead of from scratch")
    # The no-SRM ablation stopped at epoch 15 with its best epoch at 1, which records a
    # training failure rather than a detection limit. Passing the epoch cap here disables
    # early stopping, as the GBRAS trainers do, so the ablation is trained to the cap.
    ap.add_argument("--es-patience", type=int, default=None, dest="es_patience",
                    help="epochs without improvement before stopping "
                         "(default: config.ES_PATIENCE; pass the epoch cap to never stop)")
    args = ap.parse_args()
    set_seeds(args.seed)

    d = C.dirs_for(args.alpha)
    cover_idx = C.CLASS_FOLDERS.index(C.COVER_NAME)
    print(f"== Binary detector | alpha={args.alpha} | srm={args.use_srm} | seed={args.seed} | dev={DEVICE}")

    tr = make_loader(d["train"],  C.CLASS_FOLDERS, C.BATCH_SIZE, binary_cover_idx=cover_idx,
                     shuffle=True, num_workers=args.workers)
    va = make_loader(d["fusion"], C.CLASS_FOLDERS, C.BATCH_SIZE, binary_cover_idx=cover_idx,
                     shuffle=False, num_workers=args.workers, subset="select")   # [V3]
    import numpy as np
    print("train label counts:", np.bincount(labels_of(tr)).tolist())

    model = BinaryCNN(in_ch=C.CHANNELS, use_srm=args.use_srm, dropout=C.DROPOUT)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu"))
        print(f"   init from {args.init}")
    model, best = train(model, tr, va, task="binary", n_classes=2,
                        class_weights=C.USE_CLASS_WEIGHTS, epochs=C.EPOCHS_BIN,
                        lr=C.LR, monitor="auc", smoke=args.smoke,
                        es_patience=args.es_patience)
    print(f"best epoch={best['epoch']} val_auc={best['score']:.4f}")

    # Smoke runs never touch TEST: it is evaluated once, by the real run.
    if args.smoke:
        print("[smoke] OK"); return
    te = make_loader(d["test"],   C.CLASS_FOLDERS, C.BATCH_SIZE, binary_cover_idx=cover_idx,
                     shuffle=False, num_workers=args.workers)
    tag = f"a={args.alpha} srm={args.use_srm} seed={args.seed}"
    res = report_binary(model, te, tag=tag)

    name = C.alpha_tag(args.alpha) + ("" if args.use_srm else "_nosrm") + f"_s{args.seed}"
    out = C.MODEL_DIR / f"pt_binary_{name}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    write_metrics(out.with_suffix(".json"), args, best, res)
    print("saved:", out)


if __name__ == "__main__":
    main()
