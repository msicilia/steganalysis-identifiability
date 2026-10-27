"""Leave-one-algorithm-out detection transfer in the spatial domain: for each scheme A, a
cover-vs-rest detector trained on the other four and evaluated on cover-vs-A, against the
cover-vs-all detector pt_binary_<atag>_s<SEED>. Transfer models are saved as
pt_transfer_<A>_<atag>_s<SEED>.pt and reused unless --retrain.

Usage:  python transfer_spatial.py --alpha 0.4
"""
import sys, argparse, json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix

import config as C
from data import make_loader
from models import BinaryCNN
from engine import train, set_seeds, DEVICE, _predict, write_metrics
from util import boot_ci, group_boot, source_ids

INDIST_PROTOCOL = (
    "The in-distribution reference is the COVER-vs-all detector trained on all 5 schemes "
    "(including the held-out one), with BCE pos_weight from a 1:5 cover:stego ratio; each "
    "transfer detector saw 4 schemes (1:4). Both are scored on the identical COVER-vs-held "
    "test rows, so the paired difference isolates having seen the scheme, up to that "
    "class-weight difference.")


def scores_cover_vs(models, alpha, held, workers):
    """Scores of each model on the SAME COVER-vs-`held` TEST loader (rows aligned).
    Returns (y, [scores per model], cover ids)."""
    d = C.dirs_for(alpha)["test"]
    te = make_loader(d, [C.COVER_NAME, held], C.BATCH_SIZE, binary_cover_idx=0,
                     shuffle=False, num_workers=workers)
    ys, ss = [], []
    for m in models:
        p, y = _predict(m, te, "binary")     # y: 0=cover, 1=held
        ys.append(y.ravel()); ss.append(p.ravel())
    assert all(np.array_equal(ys[0], yy) for yy in ys), "label order differs between passes"
    g = source_ids(te)
    assert len(g) == len(ys[0]), "cover ids misaligned with scores"
    return ys[0], ss, g


def _binary_metrics(y, s):
    tn, fp, fn, tp = confusion_matrix(y, (s >= 0.5).astype(int), labels=[0, 1]).ravel()
    return {"auc": float(roc_auc_score(y, s)), "P_D": float(tp / (tp + fn)),
            "P_FA": float(fp / (fp + tn))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=C.ACTIVE_ALPHA)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--retrain", action="store_true",
                    help="retrain transfer models even if pt_transfer_* already exists")
    args = ap.parse_args()
    assert C.DOMAIN == "spatial", "This script is for the spatial domain"
    set_seeds(C.SEED)
    algos = [c for c in C.CLASS_FOLDERS if c != C.COVER_NAME]
    d = C.dirs_for(args.alpha); atag = C.alpha_tag(args.alpha)

    # in-distribution reference: already-trained COVER-vs-all detector
    ref = BinaryCNN(in_ch=C.CHANNELS, use_srm=True, dropout=0.0)
    ref.load_state_dict(torch.load(C.MODEL_DIR / f"pt_binary_{atag}_s{C.SEED}.pt",
                                   map_location=DEVICE))
    ref.eval().to(DEVICE)

    M = {}
    print(f"== Spatial transfer (leave-one-algo-out) alpha={args.alpha} | dev={DEVICE} ==")
    for held in algos:
        train_algos = [a for a in algos if a != held]
        mp = C.MODEL_DIR / f"pt_transfer_{held}_{atag}_s{C.SEED}.pt"
        # Seed BEFORE building the model: its initial weights must not depend on which
        # held-out schemes were trained or reused earlier in this process.
        set_seeds(C.SEED)
        model = BinaryCNN(in_ch=C.CHANNELS, use_srm=True, dropout=C.DROPOUT)
        if mp.exists() and not args.retrain:
            print(f"  [{held}] reusing {mp.name}", flush=True)
            model.load_state_dict(torch.load(mp, map_location=DEVICE))
            model.eval().to(DEVICE); best = None
        else:
            classes = [C.COVER_NAME] + train_algos            # cover=0, rest stego=1
            tr = make_loader(d["train"],  classes, C.BATCH_SIZE, binary_cover_idx=0, shuffle=True,  num_workers=args.workers)
            va = make_loader(d["fusion"], classes, C.BATCH_SIZE, binary_cover_idx=0, shuffle=False, num_workers=args.workers,
                             subset="select")   # [V3] same selection half as every spatial CNN
            model, best = train(model, tr, va, task="binary", class_weights=True,
                                epochs=C.EPOCHS_BIN, lr=C.LR, monitor="auc")
            mp.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), mp)

        # both models on one loader -> rows aligned for the paired difference
        yt, (pt, pin), gt = scores_cover_vs([model, ref], args.alpha, held, args.workers)
        if best is not None:
            write_metrics(mp.with_suffix(".json"), args, best,
                          dict(_binary_metrics(yt, pt), held_out=held, trained_on=train_algos))
        auc_t = float(roc_auc_score(yt, pt)); auc_in = float(roc_auc_score(yt, pin))
        delta = lambda i: roc_auc_score(yt[i], pt[i]) - roc_auc_score(yt[i], pin[i])
        M[held] = {"transfer_auc": round(auc_t, 4), "transfer_ci95": boot_ci(yt, pt, groups=gt),
                   "indist_auc": round(auc_in, 4), "indist_ci95": boot_ci(yt, pin, groups=gt),
                   "delta_transfer_minus_indist": round(auc_t - auc_in, 4),
                   "delta_ci95": group_boot(delta, gt),
                   "trained_on": train_algos, "model": mp.name}
        print(f"  held-out={held:10s} -> AUC(transfer)={auc_t:.4f} CI95={M[held]['transfer_ci95']} "
              f"| in-dist ref={auc_in:.4f} | Δ={auc_t - auc_in:+.4f} CI95={M[held]['delta_ci95']}",
              flush=True)

    aucs = [v["transfer_auc"] for v in M.values()]
    out = {"alpha": args.alpha, "seed": C.SEED, "per_scheme": M,
           "mean_transfer_auc": round(float(np.mean(aucs)), 4),
           "std_transfer_auc": round(float(np.std(aucs)), 4),
           "mean_indist_auc": round(float(np.mean([v["indist_auc"] for v in M.values()])), 4),
           "mean_delta_transfer_minus_indist": round(float(np.mean(
               [v["delta_transfer_minus_indist"] for v in M.values()])), 4),
           "indist_protocol": INDIST_PROTOCOL,
           "ci_protocol": "95% grouped bootstrap over test covers; Δ is paired on identical rows"}
    res = C.RESULTS_DIR
    res.mkdir(parents=True, exist_ok=True)
    (res / f"transfer_spatial_{atag}.json").write_text(json.dumps(out, indent=2))
    print(f"\nMean transfer AUC = {out['mean_transfer_auc']:.4f} ± {out['std_transfer_auc']:.4f}"
          f"  | mean Δ(transfer−indist) = {out['mean_delta_transfer_minus_indist']:+.4f}")
    print("saved:", res / f"transfer_spatial_{atag}.json")


if __name__ == "__main__":
    main()
