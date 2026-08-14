"""
Cross-algorithm detection transfer (JPEG/DCTR).

Leave-one-algorithm-out design: train COVER-vs-{all stego except A} and evaluate
COVER-vs-A detection (algorithm not seen in training).

Over already-extracted DCTR features. Usage:
  STEG_DOMAIN=jpeg python transfer_jpeg.py --alpha 0.4
"""
import sys, argparse, json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import config as C
from util import load_dctr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.4)
    args = ap.parse_args()
    names = C.CLASS_FOLDERS; cov = names.index(C.COVER_NAME)
    algos = [c for c in names if c != C.COVER_NAME]
    X, y = load_dctr(args.alpha)
    Xtr, ytr, Xte, yte = X["train"], y["train"], X["test"], y["test"]
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    print(f"== Detection transfer (JPEG/DCTR) alpha={args.alpha} ==")
    print("Rows: held-out algorithm. COVER-vs-that-algorithm AUC.\n")
    LR = dict(C=1.0, max_iter=500, class_weight="balanced", solver="lbfgs")
    # in-distribution reference: COVER-vs-all-stego detector (once)
    clf_full = LogisticRegression(**LR).fit(Xtr, (ytr != cov).astype(int))
    M = {}
    for held in algos:
        hi = names.index(held)
        train_algos = [a for a in algos if a != held]
        ti = [names.index(a) for a in train_algos]
        m = (ytr == cov) | np.isin(ytr, ti)        # COVER(0) vs the other stego(1)
        clf = LogisticRegression(**LR).fit(Xtr[m], (np.isin(ytr[m], ti)).astype(int))
        mt = (yte == cov) | (yte == hi)            # test: COVER vs the held-out algorithm
        ytrue = (yte[mt] == hi).astype(int)
        auc = float(roc_auc_score(ytrue, clf.predict_proba(Xte[mt])[:, 1]))
        auc_in = float(roc_auc_score(ytrue, clf_full.predict_proba(Xte[mt])[:, 1]))
        M[held] = {"transfer_auc": round(auc, 4), "indist_auc": round(auc_in, 4),
                   "trained_on": train_algos}
        print(f"  held-out={held:9s} | train on {train_algos} -> AUC(transfer)={auc:.4f} "
              f"| (in-dist ref={auc_in:.4f})")
    res = C.RESULTS_DIR
    (res/f"transfer_jpeg_{C.alpha_tag(args.alpha)}.json").write_text(json.dumps(M, indent=2))
    print("saved:", res/f"transfer_jpeg_{C.alpha_tag(args.alpha)}.json")


if __name__ == "__main__":
    main()
