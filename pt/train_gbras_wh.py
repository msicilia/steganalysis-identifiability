"""
GBRAS-Net as the spatial WORKHORSE. Two modes:
  --mode detect : COVER vs ANY stego (GBRASNet 2 classes).
  --mode family : 6 classes (COVER + 5 adaptive), family identification.

Trains at the 256 block level (GBRAS protocol) and reports per image (averaging
the 4 blocks).

Usage:  python train_gbras_wh.py --mode detect   |   --mode family
"""
import sys, argparse, json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
import torch
import config as C
from data import make_block_loader_multi
from models import GBRASNet
from engine import train, set_seeds, DEVICE
from gbras_io import per_image_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["detect", "family"], required=True)
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    set_seeds(args.seed)
    names = C.CLASS_FOLDERS; cov = names.index(C.COVER_NAME)
    d = C.dirs_for(args.alpha)
    cidx = cov if args.mode == "detect" else None
    nC = 2 if args.mode == "detect" else len(names)
    print(f"== GBRAS-wh {args.mode} | {nC} classes | alpha={args.alpha} | dev={DEVICE}")

    mk = lambda split, sh, bal=False: make_block_loader_multi(
        d[split], names, args.bs, binary_cover_idx=cidx, balance_binary=bal,
        seed=args.seed, shuffle=sh, num_workers=args.workers)
    # detect: cover/stego balancing ONLY in train (1:5 imbalance)
    tr = mk("train", True, bal=(args.mode == "detect"))
    va, te = mk("fusion", False), mk("test", False)
    model = GBRASNet(n_classes=nC)
    # Recipe PER MODE:
    #  - detect: lr=1e-3, monitor=acc, class_weights=False, balanced data.
    #  - family: lower lr + loss-based monitor + more patience (6 classes).
    if args.mode == "detect":
        lr, mon, cw, esp = 1e-3, "acc", False, 8
    else:
        lr, mon, cw, esp = 3e-4, "loss", True, 12
    model, _ = train(model, tr, va, task="multiclass", n_classes=nC, class_weights=cw,
                     epochs=args.epochs, lr=lr, monitor=mon, es_patience=esp)

    # --- PER-IMAGE report ---
    P, y = per_image_probs(model, te, DEVICE)
    if args.mode == "detect":
        auc = float(roc_auc_score(y, P[:, 1]))
        pred = (P[:, 1] >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        res = {"auc": auc, "P_D": float(tp/(tp+fn)), "P_FA": float(fp/(fp+tn))}
        print(f"\n[GBRAS detect, per image] AUC={auc:.4f}  P_D={res['P_D']:.4f}  P_FA={res['P_FA']:.4f}")
    else:
        pred = P.argmax(1)
        res = {"acc": float(accuracy_score(y, pred)),
               "macro_f1": float(f1_score(y, pred, average="macro")),
               "cm": confusion_matrix(y, pred, labels=range(nC)).tolist()}
        print(f"\n[GBRAS family, per image] acc={res['acc']:.4f}  macro_F1={res['macro_f1']:.4f}")
        print("confusion:", res["cm"])

    out = C.MODEL_DIR / f"pt_gbraswh_{args.mode}_{C.alpha_tag(args.alpha)}_s{args.seed}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    (C.MODEL_DIR / f"pt_gbraswh_{args.mode}_{C.alpha_tag(args.alpha)}_s{args.seed}.json").write_text(json.dumps(res, indent=2))
    print("saved:", out)


if __name__ == "__main__":
    main()
