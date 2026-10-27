"""GBRAS-Net cover-vs-one specialist: 256x256 blocks (four per image), 30 SRM kernels, Tanh3.
Selection on FUSION, block-level report on TEST.

Outputs: MODEL_DIR/pt_gbras_<algo>_<aXXX>_s<seed>.pt and .json.

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
from engine import train, set_seeds, write_metrics, DEVICE, _predict
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=C.ACTIVE_ALPHA)
    # Required: a silent S-UNIWARD default made a missing flag look like a valid run.
    ap.add_argument("--algo", required=True, help="stego algorithm (COVER vs this)")
    ap.add_argument("--seed", type=int, default=C.SEED)
    # 100 = the paper runs; 30 truncated HUGO/S-UNIWARD before they converged.
    ap.add_argument("--epochs", type=int, default=C.GBRAS_EPOCHS)
    ap.add_argument("--es-patience", type=int, default=C.GBRAS_ES_PATIENCE, dest="es_patience",
                    help="early-stopping patience; default config.GBRAS_ES_PATIENCE "
                         "(None = train all epochs, keep the best checkpoint)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    set_seeds(args.seed)

    d = C.dirs_for(args.alpha)
    print(f"== GBRAS-Net | COVER vs {args.algo} | alpha={args.alpha} | seed={args.seed} | dev={DEVICE}")
    bs = 64   # 256x256 fits easily on the GB10; fewer steps/epoch = faster
    tr = make_block_loader(d["train"],  C.COVER_NAME, args.algo, bs, shuffle=True,  num_workers=args.workers)
    va = make_block_loader(d["fusion"], C.COVER_NAME, args.algo, bs, shuffle=False, num_workers=args.workers,
                           subset="select")   # [V3] the meta half trains the ensemble

    model = GBRASNet(n_classes=2)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    # Optimizer: engine uses torch Adam(lr, betas=(0.9,0.999), eps=1e-8), which matches the
    # original Keras Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08).
    model, best = train(model, tr, va, task="multiclass", n_classes=2,
                        class_weights=False, epochs=(1 if args.smoke else args.epochs),
                        lr=1e-3, monitor="acc", smoke=args.smoke,
                        # original notebook: constant Adam LR, no scheduler callback
                        lr_schedule=False,
                        # patience = epochs never triggers: the first epoch always improves
                        es_patience=(args.epochs if args.es_patience is None else args.es_patience))
    print(f"best epoch={best['epoch']} val_acc={best['score']:.4f}")

    # Smoke runs never touch TEST: it is evaluated once, by the real run.
    if args.smoke:
        print("[smoke] OK"); return
    te = make_block_loader(d["test"],   C.COVER_NAME, args.algo, bs, shuffle=False, num_workers=args.workers)
    probs, y = _predict(model, te, "multiclass")
    p1 = probs[:, 1]; pred = (p1 >= 0.5).astype(int)
    print(f"\n== TEST GBRAS COVER vs {args.algo} a={args.alpha} ==")
    print(classification_report(y, pred, target_names=["COVER", args.algo], digits=4, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    # the unshuffled block loader yields 4 consecutive blocks per image: average them
    p_img, y_img = p1.reshape(-1, 4).mean(1), y.reshape(-1, 4)[:, 0]
    res = {"auc": float(roc_auc_score(y, p1)), "P_D": float(tp / (tp + fn)),
           "P_FA": float(fp / (fp + tn)), "level": "block",
           "auc_image": float(roc_auc_score(y_img, p_img))}
    print(f"AUC(block)={res['auc']:.4f}  AUC(image)={res['auc_image']:.4f}  "
          f"P_D={res['P_D']:.4f}  P_FA={res['P_FA']:.4f}")
    out = C.MODEL_DIR / f"pt_gbras_{args.algo}_{C.alpha_tag(args.alpha)}_s{args.seed}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    write_metrics(out.with_suffix(".json"), args, best, res)
    print("saved:", out)


if __name__ == "__main__":
    main()
