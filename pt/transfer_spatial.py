"""
Spatial-domain detection transfer across algorithms.

Leave-one-algorithm-out: for each A, train an SRM detector COVER-vs-{4 algos minus A}
and evaluate COVER-vs-A detection (A unseen). In-dist reference: the already-trained
COVER-vs-all detector (pt_binary_a040_s42).

Usage (spatial domain = without STEG_DOMAIN):  python transfer_spatial.py --alpha 0.4
"""
import sys, argparse, json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import config as C
from data import make_loader
from models import BinaryCNN
from engine import train, set_seeds, DEVICE, _predict
from util import boot_ci, source_ids


def auc_cover_vs(model, alpha, held):
    """COVER vs `held` detection AUC on TEST. Returns (auc, y, scores)."""
    d = C.dirs_for(alpha)["test"]
    classes = [C.COVER_NAME, held]
    te = make_loader(d, classes, C.BATCH_SIZE, binary_cover_idx=0, shuffle=False, num_workers=8)
    p, y = _predict(model, te, "binary")     # y: 0=cover, 1=held
    return float(roc_auc_score(y, p)), y.ravel(), p.ravel(), source_ids(te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    assert C.DOMAIN == "spatial", "This script is for the spatial domain"
    algos = [c for c in C.CLASS_FOLDERS if c != C.COVER_NAME]
    d = C.dirs_for(args.alpha)

    # in-distribution reference: already-trained COVER-vs-all detector
    ref = BinaryCNN(in_ch=C.CHANNELS, use_srm=True, dropout=0.0)
    ref.load_state_dict(torch.load(C.MODEL_DIR / f"pt_binary_{C.alpha_tag(args.alpha)}_s42.pt",
                                   map_location=DEVICE))
    ref.eval().to(DEVICE)

    M = {}
    print(f"== Spatial transfer (leave-one-algo-out) alpha={args.alpha} | dev={DEVICE} ==")
    for held in algos:
        set_seeds(42)
        train_algos = [a for a in algos if a != held]
        classes = [C.COVER_NAME] + train_algos            # cover=0, rest stego=1
        tr = make_loader(d["train"],  classes, C.BATCH_SIZE, binary_cover_idx=0, shuffle=True,  num_workers=args.workers)
        va = make_loader(d["fusion"], classes, C.BATCH_SIZE, binary_cover_idx=0, shuffle=False, num_workers=args.workers)
        model = BinaryCNN(in_ch=C.CHANNELS, use_srm=True, dropout=C.DROPOUT)
        model, _ = train(model, tr, va, task="binary", class_weights=True,
                         epochs=C.EPOCHS_BIN, lr=C.LR, monitor="auc")
        auc_t, yt, pt, gt = auc_cover_vs(model, args.alpha, held)  # A unseen
        auc_in, _, _, _ = auc_cover_vs(ref, args.alpha, held)      # in-dist ref
        M[held] = {"transfer_auc": round(auc_t, 4), "transfer_ci95": boot_ci(yt, pt, groups=gt),
                   "indist_auc": round(auc_in, 4), "trained_on": train_algos}
        print(f"  held-out={held:10s} -> AUC(transfer)={auc_t:.4f} CI95={M[held]['transfer_ci95']} "
              f"| in-dist ref={auc_in:.4f}", flush=True)

    res = C.RESULTS_DIR
    (res / f"transfer_spatial_{C.alpha_tag(args.alpha)}.json").write_text(json.dumps(M, indent=2))
    aucs = [v["transfer_auc"] for v in M.values()]
    print(f"\nMean transfer AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print("saved:", res / f"transfer_spatial_{C.alpha_tag(args.alpha)}.json")


if __name__ == "__main__":
    main()
