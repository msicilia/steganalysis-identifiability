"""GBRAS-Net as cover-vs-all detector (--mode detect) or six-class identifier (--mode family),
trained on 256x256 blocks and reported per image (mean of the four blocks). In detect mode
TRAIN draws a fresh seeded 1:1 stego subset every epoch.

Outputs: MODEL_DIR/pt_gbraswh_<mode>_<aXXX>_s<seed>.pt and .json.

Usage:  python train_gbras_wh.py --mode detect   |   --mode family
"""
import sys, argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
import torch
import config as C
from data import make_block_loader_multi
from models import GBRASNet
from engine import train, set_seeds, write_metrics, collapse_diagnosis, DEVICE
from gbras_io import per_image_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["detect", "family"], required=True)
    ap.add_argument("--alpha", type=float, default=C.ACTIVE_ALPHA)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--epochs", type=int, default=C.GBRAS_EPOCHS)
    ap.add_argument("--es-patience", type=int, default=C.GBRAS_ES_PATIENCE, dest="es_patience",
                    help="early-stopping patience; default config.GBRAS_ES_PATIENCE "
                         "(None = train all epochs, keep the best checkpoint)")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    set_seeds(args.seed)
    names = C.CLASS_FOLDERS; cov = names.index(C.COVER_NAME)
    d = C.dirs_for(args.alpha)
    cidx = cov if args.mode == "detect" else None
    nC = 2 if args.mode == "detect" else len(names)
    print(f"== GBRAS-wh {args.mode} | {nC} classes | alpha={args.alpha} | seed={args.seed} | dev={DEVICE}")

    mk = lambda split, sh, bal=False, res=False, sub=None: make_block_loader_multi(
        d[split], names, args.bs, binary_cover_idx=cidx, balance_binary=bal,
        resample_binary=res, seed=args.seed, shuffle=sh, num_workers=args.workers, subset=sub)
    # detect: cover/stego balancing in TRAIN and in the selection split (1:5 imbalance).
    # TRAIN draws a fresh 1:1 stego subset every epoch, so over the run the detector sees all
    # the stego images (a fixed 1/5 subset gave it 5x less stego than each cover-vs-one
    # specialist, which biased the modular-vs-monolithic comparison). The fusion split is
    # balanced with a FIXED subset, because selection watches accuracy: on the raw 1:5 mix a
    # model that answers "stego" every time already scores 0.83, so the best-epoch
    # checkpoint would track the prior rather than the detector. TEST stays untouched -- it
    # is the real distribution and is only ever evaluated once.
    balanced = (args.mode == "detect")
    tr = mk("train", True, res=balanced)
    # [V3] selection on the FUSION "select" half; the "meta" half trains the ensembles
    va, te = mk("fusion", False, bal=balanced, sub="select"), mk("test", False)
    model = GBRASNet(n_classes=nC)
    # Recipe PER MODE:
    #  - detect: lr=1e-3, monitor=acc, class_weights=False, balanced data.
    #  - family: lower lr + loss-based monitor (6 classes).
    # Early stopping follows config.GBRAS_ES_PATIENCE / --es-patience for both modes (None =
    # all epochs, best checkpoint kept). The learning rate is constant, as in the original
    # GBRAS-Net notebook (Adam, no scheduler callback).
    if args.mode == "detect":
        lr, mon, cw = 1e-3, "acc", False
    else:
        lr, mon, cw = 3e-4, "loss", True
    esp = args.epochs if args.es_patience is None else args.es_patience   # epochs = never stops
    model, best = train(model, tr, va, task="multiclass", n_classes=nC, class_weights=cw,
                        epochs=args.epochs, lr=lr, monitor=mon, es_patience=esp,
                        lr_schedule=False)

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
        cm = confusion_matrix(y, pred, labels=range(nC))
        # Per-scheme recall (rows = true class), reported in the paper.
        recall = cm.diagonal() / np.maximum(cm.sum(1), 1)
        res = {"acc": float(accuracy_score(y, pred)),
               "macro_f1": float(f1_score(y, pred, average="macro")),
               "cm": cm.tolist(),
               "recall_per_class": {n: float(r) for n, r in zip(names, recall)},
               **collapse_diagnosis(P, pred, nC)}
        print(f"\n[GBRAS family, per image] acc={res['acc']:.4f}  macro_F1={res['macro_f1']:.4f}")
        if res["collapsed"]:
            print(f"WARNING: COLLAPSED ({'; '.join(res['collapsed_reasons'])}): training failure, "
                  f"not an identification result")
        print("confusion:", res["cm"])
        print("recall:", {n: round(r, 4) for n, r in res["recall_per_class"].items()})

    out = C.MODEL_DIR / f"pt_gbraswh_{args.mode}_{C.alpha_tag(args.alpha)}_s{args.seed}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    write_metrics(out.with_suffix(".json"), args, best, res)
    print("saved:", out)


if __name__ == "__main__":
    main()
