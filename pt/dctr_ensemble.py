"""JPEG/DCTR single split: cover-vs-all detection, cover-vs-one specialists, the monolithic
six-class classifier and the modular stacking ensemble, all with logistic regression.
fit_base() and evaluate() are reused by dctr_bootstrap.py, which run_all.sh calls instead of
this script.

Usage:  STEG_DOMAIN=jpeg python dctr_ensemble.py --alpha 0.4
"""
import sys, argparse, json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, accuracy_score
import config as C
from util import load_dctr, fit_logreg, convergence_info

SPLITS = ("train", "fusion", "test")


def load_scaled(alpha):
    """DCTR features standardised with a scaler fitted on TRAIN, and their labels."""
    X, y = load_dctr(alpha)
    sc = StandardScaler().fit(X["train"])
    return {s: sc.transform(X[s]) for s in SPLITS}, y


def fit_base(X, y, names):
    """Every single-split LogReg fit, once. Every fit goes through util.fit_logreg, which
    escalates the iteration budget until lbfgs converges and raises rather than returning a
    truncated fit."""
    cov = names.index(C.COVER_NAME)
    algos = [c for c in names if c != C.COVER_NAME]
    Xtr, ytr = X["train"], y["train"]
    det = fit_logreg(Xtr, (ytr != cov).astype(int), tag="detect")
    spec = {}
    for a in algos:
        ai = names.index(a)
        m = (ytr == cov) | (ytr == ai)
        spec[a] = fit_logreg(Xtr[m], (ytr[m] == ai).astype(int), tag=f"spec:{a}")
    mono = fit_logreg(Xtr, ytr, tag="mono")
    # leak-free stacking: the meta-learner trains on the specialists' FUSION posteriors
    Pfu = np.column_stack([spec[a].predict_proba(X["fusion"])[:, 1] for a in algos])
    meta = fit_logreg(Pfu, y["fusion"], tag="meta")
    return {"det": det, "spec": spec, "mono": mono, "meta": meta, "Pfu": Pfu}


def evaluate(base, X, y, names):
    """TEST predictions of every base model, and the dctr_ensemble report."""
    cov = names.index(C.COVER_NAME)
    algos = [c for c in names if c != C.COVER_NAME]
    Xte, yte = X["test"], y["test"]

    # A) binary detection
    sdet = base["det"].predict_proba(Xte)[:, 1]
    ybt = (yte != cov).astype(int)
    tn, fp, fn, tp = confusion_matrix(ybt, (sdet >= 0.5).astype(int), labels=[0, 1]).ravel()
    detection = {"auc": float(roc_auc_score(ybt, sdet)),
                 "P_D": float(tp/(tp+fn)), "P_FA": float(fp/(fp+tn))}

    # B) specialists: posteriors on all test rows (modular arm) and cover-vs-own-scheme AUC
    Pte = np.column_stack([base["spec"][a].predict_proba(Xte)[:, 1] for a in algos])
    spec_auc = {}
    for k, a in enumerate(algos):
        ai = names.index(a)
        mt = (yte == cov) | (yte == ai)
        spec_auc[a] = float(roc_auc_score((yte[mt] == ai).astype(int), Pte[mt, k]))

    # C) monolithic multiclass, D) modular ensemble
    mono_pred = base["mono"].predict(Xte)
    mod_pred = base["meta"].predict(Pte)

    def mc(pred):
        return {"acc": float(accuracy_score(yte, pred)),
                "macro_f1": float(f1_score(yte, pred, average="macro")),
                "cm": confusion_matrix(yte, pred, labels=range(len(names))).tolist()}
    report = {"detection": detection, "specialist_auc": spec_auc,
              "monolithic": mc(mono_pred), "modular": mc(mod_pred)}
    preds = {"det": sdet, "Pte": Pte, "mono": mono_pred, "modular": mod_pred}
    return preds, report


def write_report(alpha, names, report):
    # Separability (Fisher J) is deliberately not recomputed here: jpeg_kfold_cv.py reports
    # the pairwise Fisher J on a PCA-50 (fitted on all rows, descriptive) and
    # dctr_bootstrap.py the multiclass/pairwise J on a train-fitted PCA-50.
    out = {"alpha": alpha, "classes": names, **report, "provenance": convergence_info()}
    res = C.RESULTS_DIR; res.mkdir(parents=True, exist_ok=True)
    dst = res / f"dctr_ensemble_{C.alpha_tag(alpha)}.json"
    dst.write_text(json.dumps(out, indent=2))

    det = report["detection"]
    print(f"\n### A) COVER/stego detection (DCTR): AUC={det['auc']:.4f} "
          f"P_D={det['P_D']:.4f} P_FA={det['P_FA']:.4f}")
    print("### B) COVER-vs-algo specialists (test AUC):")
    for a, v in report["specialist_auc"].items():
        print(f"   {a}: {v:.4f}")
    r_mono, r_ens = report["monolithic"], report["modular"]
    print(f"### C) Monolithic multiclass : acc={r_mono['acc']:.4f} macroF1={r_mono['macro_f1']:.4f}")
    print(f"### D) Modular   multiclass : acc={r_ens['acc']:.4f} macroF1={r_ens['macro_f1']:.4f}")
    print(f"\n>>> Δacc modular−monolithic = {r_ens['acc'] - r_mono['acc']:+.4f}")
    print("saved:", dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.4)
    args = ap.parse_args()
    names = C.CLASS_FOLDERS
    X, y = load_scaled(args.alpha)
    print(f"== DCTR ensemble | alpha={args.alpha} | feat={X['train'].shape[1]} | classes={names}")
    base = fit_base(X, y, names)
    _, report = evaluate(base, X, y, names)
    write_report(args.alpha, names, report)


if __name__ == "__main__":
    main()
