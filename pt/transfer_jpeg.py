"""Leave-one-algorithm-out detection transfer on the DCTR features: a cover-vs-rest detector
trained without scheme A is evaluated on cover-vs-A. For a held-out adaptive scheme a second
detector is trained on the other adaptive schemes only (adaptive_only). run_transfer() takes
the in-distribution reference detector, so dctr_bootstrap.py can pass its fitted one.

Usage:  STEG_DOMAIN=jpeg python transfer_jpeg.py --alpha 0.4
"""
import sys, argparse, json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import config as C
from util import load_dctr, load_dctr_groups, fit_logreg, convergence_info, group_boot


def run_transfer(Xtr, ytr, Xte, yte, gte, names, clf_full):
    """Leave-one-out transfer on standardised features. `clf_full` is the in-distribution
    reference: the COVER-vs-ALL-stego detector, i.e. one that DID see the held-out scheme.
    Note this is a different protocol from the cover-vs-ONE specialists reported elsewhere;
    the two are not interchangeable."""
    cov = names.index(C.COVER_NAME)
    algos = [c for c in names if c != C.COVER_NAME]
    legacy = set(C.LEGACY_SCHEMES)
    assert len(gte) == len(yte), "cover ids and test features are misaligned"
    print("Rows: held-out algorithm. COVER-vs-that-algorithm AUC.\n")
    M = {}
    for held in algos:
        hi = names.index(held)
        train_algos = [a for a in algos if a != held]
        ti = [names.index(a) for a in train_algos]
        m = (ytr == cov) | np.isin(ytr, ti)        # COVER(0) vs the other stego(1)
        clf = fit_logreg(Xtr[m], (np.isin(ytr[m], ti)).astype(int), tag=f"transfer:{held}")
        mt = (yte == cov) | (yte == hi)            # test: COVER vs the held-out algorithm
        ytrue = (yte[mt] == hi).astype(int)
        s_tr = clf.predict_proba(Xte[mt])[:, 1]
        s_in = clf_full.predict_proba(Xte[mt])[:, 1]
        auc = float(roc_auc_score(ytrue, s_tr))
        auc_in = float(roc_auc_score(ytrue, s_in))
        g = gte[mt]
        ci = group_boot(lambda idx: roc_auc_score(ytrue[idx], s_tr[idx]), g, point=auc)
        ci_in = group_boot(lambda idx: roc_auc_score(ytrue[idx], s_in[idx]), g, point=auc_in)
        # Paired: both detectors score the SAME resampled covers, so the shared image
        # difficulty cancels and the CI is for the transfer gap itself.
        ci_gap = group_boot(lambda idx: roc_auc_score(ytrue[idx], s_tr[idx])
                                        - roc_auc_score(ytrue[idx], s_in[idx]),
                            g, point=auc - auc_in)
        M[held] = {"transfer_auc": round(auc, 4), "transfer_ci95": ci,
                   "indist_auc": round(auc_in, 4), "indist_ci95": ci_in,
                   "transfer_minus_indist": round(auc - auc_in, 4),
                   "transfer_minus_indist_ci95": ci_gap,
                   "ci_protocol": "grouped bootstrap by cover, n=2000; gap CI paired",
                   "indist_protocol": "cover-vs-all detector (saw the held-out scheme)",
                   "trained_on": train_algos,
                   "trained_on_includes_legacy": bool(legacy & set(train_algos))}
        print(f"  held-out={held:9s} | train on {train_algos} -> AUC(transfer)={auc:.4f} "
              f"CI95={ci} | (in-dist ref={auc_in:.4f}) | gap={auc - auc_in:+.4f} CI95={ci_gap}")
        if held in legacy or not (legacy & set(train_algos)):
            continue
        # adaptive-only training: same held-out test rows, detector never saw a legacy scheme
        ad = [a for a in train_algos if a not in legacy]
        adi = [names.index(a) for a in ad]
        m2 = (ytr == cov) | np.isin(ytr, adi)
        clf2 = fit_logreg(Xtr[m2], np.isin(ytr[m2], adi).astype(int),
                          tag=f"transfer-adaptive:{held}")
        s_ad = clf2.predict_proba(Xte[mt])[:, 1]
        auc_ad = float(roc_auc_score(ytrue, s_ad))
        M[held]["adaptive_only"] = {
            "transfer_auc": round(auc_ad, 4),
            "transfer_ci95": group_boot(lambda idx: roc_auc_score(ytrue[idx], s_ad[idx]),
                                        g, point=auc_ad),
            "transfer_minus_indist": round(auc_ad - auc_in, 4),
            "transfer_minus_indist_ci95": group_boot(
                lambda idx: roc_auc_score(ytrue[idx], s_ad[idx])
                            - roc_auc_score(ytrue[idx], s_in[idx]), g, point=auc_ad - auc_in),
            "trained_on": ad}
        print(f"  held-out={held:9s} | adaptive only {ad} -> AUC(transfer)={auc_ad:.4f} "
              f"CI95={M[held]['adaptive_only']['transfer_ci95']} | gap={auc_ad - auc_in:+.4f}")
    return M


def write_transfer(alpha, M):
    M = dict(M, provenance=convergence_info())
    res = C.RESULTS_DIR; res.mkdir(parents=True, exist_ok=True)
    dst = res / f"transfer_jpeg_{C.alpha_tag(alpha)}.json"
    dst.write_text(json.dumps(M, indent=2))
    print("saved:", dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.4)
    args = ap.parse_args()
    names = C.CLASS_FOLDERS; cov = names.index(C.COVER_NAME)
    X, y = load_dctr(args.alpha)
    Xtr, ytr, Xte, yte = X["train"], y["train"], X["test"], y["test"]
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    print(f"== Detection transfer (JPEG/DCTR) alpha={args.alpha} ==")
    gte = load_dctr_groups(args.alpha)["test"]     # cover id per test row, for the CI
    clf_full = fit_logreg(Xtr, (ytr != cov).astype(int), tag="indist-ref")
    write_transfer(args.alpha, run_transfer(Xtr, ytr, Xte, yte, gte, names, clf_full))


if __name__ == "__main__":
    main()
