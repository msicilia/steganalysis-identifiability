"""
JPEG/DCTR robustness checks that complement the cross-validated analysis in
jpeg_kfold_cv.py, computed on the already-extracted DCTR features (CPU):

  (1a) Single-split 95% bootstrap CIs for detection AUC and for monolithic,
       modular and delta accuracy.
  (1b) Class separability: Fisher J over a low-dimensional PCA of DCTR and over the
       specialist-posterior space.
  (1c) Monolithic vs modular accuracy across three classifier families
       (LogReg, LinearSVC, RandomForest), showing the ceiling is not an artifact of
       the classifier.

Usage:  STEG_DOMAIN=jpeg python dctr_bootstrap.py --alpha 0.4
"""
import sys, argparse, json, itertools
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import config as C
from util import load_dctr, fisher_J

RNG = np.random.default_rng(0)


def boot_ci(fn, n=2000):
    """Bootstrap 95% CI of a statistic fn(idx) over test indices."""
    vals = np.array([fn(RNG.integers(0, fn.N, fn.N)) for _ in range(n)])
    return round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--alpha", type=float, default=0.4)
    args = ap.parse_args()
    names = C.CLASS_FOLDERS; cov = names.index(C.COVER_NAME)
    algos = [c for c in names if c != C.COVER_NAME]
    X, y = load_dctr(args.alpha)
    sc = StandardScaler().fit(X["train"])
    Xtr, Xfu, Xte = sc.transform(X["train"]), sc.transform(X["fusion"]), sc.transform(X["test"])
    ytr, yfu, yte = y["train"], y["fusion"], y["test"]
    ybin_tr, ybin_te = (ytr != cov).astype(int), (yte != cov).astype(int)
    LR = lambda: LogisticRegression(C=1.0, max_iter=300, class_weight="balanced")
    out = {"alpha": args.alpha}

    # ---- base fits (once) ----
    det = LR().fit(Xtr, ybin_tr); det_p = det.predict_proba(Xte)[:, 1]
    mono = LR().fit(Xtr, ytr); mono_pred = mono.predict(Xte)
    # specialists -> posteriors (fusion trains meta, test evaluates) [leak-free]
    Pfu = np.zeros((len(yfu), len(algos))); Pte = np.zeros((len(yte), len(algos)))
    for k, a in enumerate(algos):
        ai = names.index(a); m = (ytr == cov) | (ytr == ai)
        s = LR().fit(Xtr[m], (ytr[m] == ai).astype(int))
        Pfu[:, k] = s.predict_proba(Xfu)[:, 1]; Pte[:, k] = s.predict_proba(Xte)[:, 1]
    meta = LR().fit(Pfu, yfu); mod_pred = meta.predict(Pte)

    # ---- (1a) bootstrap CIs ----
    def mk(fn):
        fn.N = len(yte); return fn
    ci_det = boot_ci(mk(lambda idx: roc_auc_score(ybin_te[idx], det_p[idx])))
    ci_mono = boot_ci(mk(lambda idx: accuracy_score(yte[idx], mono_pred[idx])))
    ci_mod = boot_ci(mk(lambda idx: accuracy_score(yte[idx], mod_pred[idx])))
    ci_delta = boot_ci(mk(lambda idx: accuracy_score(yte[idx], mod_pred[idx])
                                      - accuracy_score(yte[idx], mono_pred[idx])))
    out["bootstrap_CI95"] = {
        "detection_auc": [round(float(roc_auc_score(ybin_te, det_p)), 4), ci_det],
        "monolithic_acc": [round(float(accuracy_score(yte, mono_pred)), 4), ci_mono],
        "modular_acc": [round(float(accuracy_score(yte, mod_pred)), 4), ci_mod],
        "delta_modular_minus_mono": [round(float(accuracy_score(yte, mod_pred)
                                                  - accuracy_score(yte, mono_pred)), 4), ci_delta],
    }

    # ---- (1b) stable separability ----
    pca = PCA(n_components=50, random_state=0).fit(Xtr)
    Xte_p = pca.transform(Xte)
    J_pca = fisher_J(Xte_p, yte)
    pair_pca = {f"{names[a]}|{names[b]}": round(fisher_J(Xte_p[(yte==a)|(yte==b)],
                yte[(yte==a)|(yte==b)]), 3) for a, b in itertools.combinations(range(len(names)), 2)}
    J_post = fisher_J(Pte, yte)   # specialist-posteriors space (one dim per stego scheme)
    out["separability_stable"] = {"fisher_J_pca50_4class": round(J_pca, 3),
                                  "fisher_pairs_pca50": pair_pca,
                                  "fisher_J_specialist_posteriors": round(J_post, 3)}

    # ---- (1c) monolithic vs modular across classifiers ----
    # The tree ensemble does not scale to the full 20,480-dim DCTR at 10k covers
    # (~30k train rows); we cap its training set by a fixed random subsample. This is
    # the tree-baseline caveat itself (it needs far more data on this high-dim feature
    # than the linear classifiers), so the cap is documented rather than hidden. The
    # linear classifiers and every modular (low-dim posterior) fit use all rows.
    RF_TRAIN_CAP = 8000

    def fit_pred(mk2, Xf, yf, Xp, cap=None):
        if cap and len(yf) > cap:
            sel = RNG.choice(len(yf), cap, replace=False)
            Xf, yf = Xf[sel], yf[sel]
        return mk2().fit(Xf, yf).predict(Xp)

    clfs = {"LogReg": (LR, None),
            "LinearSVC": (lambda: LinearSVC(C=1.0, class_weight="balanced", max_iter=5000), None),
            "RandomForest": (lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                                            random_state=0), RF_TRAIN_CAP)}
    inv = {}
    for nm, (mk2, cap) in clfs.items():
        cm = fit_pred(mk2, Xtr, ytr, Xte, cap)              # monolithic
        cme = fit_pred(mk2, Pfu, yfu, Pte, cap)             # modular (meta over posteriors)
        inv[nm] = {"mono_acc": round(float(accuracy_score(yte, cm)), 4),
                   "modular_acc": round(float(accuracy_score(yte, cme)), 4),
                   "delta": round(float(accuracy_score(yte, cme) - accuracy_score(yte, cm)), 4)}
        if cap:
            inv[nm]["train_cap"] = cap
    out["ceiling_by_classifier"] = inv

    res = C.RESULTS_DIR
    (res/f"dctr_bootstrap_{C.alpha_tag(args.alpha)}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("saved:", res/f"dctr_bootstrap_{C.alpha_tag(args.alpha)}.json")


if __name__ == "__main__":
    main()
