"""
JPEG (DCT domain, DCTR features): compare a modular ensemble against a monolithic
classifier.

On the DCTR features (extract_dctr.py), with linear classifiers (LogReg):

  A) Binary COVER vs any stego detection (AUC/P_D/P_FA).
  B) COVER-vs-each-algorithm specialists (standalone AUC).
  C) Monolithic multiclass classifier.
  D) Modular ensemble (meta-LogReg over specialist posteriors),
     leak-free: specialists on TRAIN, meta-features on FUSION, evaluation on TEST.

Reports Δacc(modular − monolithic). Usage:
  STEG_DOMAIN=jpeg python dctr_ensemble.py --alpha 0.4
"""
import sys, argparse, json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, accuracy_score
import config as C
from util import load_dctr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--Creg", type=float, default=1.0)
    args = ap.parse_args()
    names = C.CLASS_FOLDERS; cov = names.index(C.COVER_NAME)
    algos = [c for c in names if c != C.COVER_NAME]
    X, y = load_dctr(args.alpha)
    sc = StandardScaler().fit(X["train"])
    Xtr, Xfu, Xte = (sc.transform(X[s]) for s in ("train", "fusion", "test"))
    ytr, yfu, yte = y["train"], y["fusion"], y["test"]
    ybin = {s: (y[s] != cov).astype(int) for s in ("train", "fusion", "test")}
    print(f"== DCTR ensemble | alpha={args.alpha} | feat={Xtr.shape[1]} | classes={names}")

    def lr():
        return LogisticRegression(C=args.Creg, max_iter=3000, class_weight="balanced")

    # A) binary detection
    detb = lr().fit(Xtr, ybin["train"]); sdet = detb.predict_proba(Xte)[:, 1]
    pdet = (sdet >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(ybin["test"], pdet, labels=[0, 1]).ravel()
    detection = {"auc": float(roc_auc_score(ybin["test"], sdet)),
                 "P_D": float(tp/(tp+fn)), "P_FA": float(fp/(fp+tn))}

    # B) COVER-vs-algo specialists + posteriors (leak-free) for the ensemble
    spec, spec_auc = {}, {}
    Pfu = np.zeros((len(yfu), len(algos))); Pte = np.zeros((len(yte), len(algos)))
    for k, a in enumerate(algos):
        ai = names.index(a)
        m = (ytr == cov) | (ytr == ai)
        clf = lr().fit(Xtr[m], (ytr[m] == ai).astype(int))
        spec[a] = clf
        Pfu[:, k] = clf.predict_proba(Xfu)[:, 1]
        Pte[:, k] = clf.predict_proba(Xte)[:, 1]
        # specialist AUC on test cover-vs-its-algo
        mt = (yte == cov) | (yte == ai)
        spec_auc[a] = float(roc_auc_score((yte[mt] == ai).astype(int),
                                          clf.predict_proba(Xte[mt])[:, 1]))

    # C) monolithic multiclass
    mono = lr().fit(Xtr, ytr); pmono = mono.predict(Xte)
    r_mono = {"acc": float(accuracy_score(yte, pmono)),
              "macro_f1": float(f1_score(yte, pmono, average="macro")),
              "cm": confusion_matrix(yte, pmono, labels=range(len(names))).tolist()}

    # D) 4-class modular ensemble (meta over posteriors, leak-free: fit on FUSION)
    meta = lr().fit(Pfu, yfu); pens = meta.predict(Pte)
    r_ens = {"acc": float(accuracy_score(yte, pens)),
             "macro_f1": float(f1_score(yte, pens, average="macro")),
             "cm": confusion_matrix(yte, pens, labels=range(len(names))).tolist()}

    # Class separability (Fisher J) is reported by jpeg_kfold_cv.py on a numerically
    # stable PCA-50 of the DCTR features; it is deliberately not recomputed here.
    out = {"alpha": args.alpha, "classes": names, "detection": detection,
           "specialist_auc": spec_auc, "monolithic": r_mono, "modular": r_ens}
    res = C.RESULTS_DIR; res.mkdir(parents=True, exist_ok=True)
    (res / f"dctr_ensemble_{C.alpha_tag(args.alpha)}.json").write_text(json.dumps(out, indent=2))

    print(f"\n### A) COVER/stego detection (DCTR): AUC={detection['auc']:.4f} "
          f"P_D={detection['P_D']:.4f} P_FA={detection['P_FA']:.4f}")
    print("### B) COVER-vs-algo specialists (test AUC):")
    for a in algos: print(f"   {a}: {spec_auc[a]:.4f}")
    print(f"### C) Monolithic multiclass : acc={r_mono['acc']:.4f} macroF1={r_mono['macro_f1']:.4f}")
    print(f"### D) Modular   multiclass : acc={r_ens['acc']:.4f} macroF1={r_ens['macro_f1']:.4f}")
    dacc = r_ens["acc"] - r_mono["acc"]
    print(f"\n>>> Δacc modular−monolithic = {dacc:+.4f}")
    print("saved:", res / f"dctr_ensemble_{C.alpha_tag(args.alpha)}.json")


if __name__ == "__main__":
    main()
