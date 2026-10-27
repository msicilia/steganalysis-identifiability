"""Single-split JPEG/DCTR analysis on the extracted DCTR features (CPU). Fits every logistic
regression once (dctr_ensemble.fit_base) and writes three result files:

  dctr_ensemble_<a>.json  : detection, specialist AUCs, monolithic/modular accuracy and
                            confusion matrices
  dctr_bootstrap_<a>.json : 95% bootstrap CIs by cover; Fisher J on a PCA of DCTR and on the
                            specialist posteriors; monolithic vs modular accuracy for LogReg,
                            LinearSVC and RandomForest
  transfer_jpeg_<a>.json  : leave-one-algorithm-out transfer (transfer_jpeg.run_transfer)

Usage:  STEG_DOMAIN=jpeg python dctr_bootstrap.py --alpha 0.4
"""
import sys, argparse, json, itertools, warnings
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score, accuracy_score
import config as C
from util import load_dctr_groups, fisher_J, convergence_info, group_boot
from dctr_ensemble import load_scaled, fit_base, evaluate, write_report
from transfer_jpeg import run_transfer, write_transfer

SVC_MAX_ITER, SVC_MAX_ITER_CAP = 5_000, 80_000


def fit_linearsvc(X, y, tag=""):
    """LinearSVC with the same guarantee as util.fit_logreg: liblinear's ConvergenceWarning
    means the dual coordinate descent stopped on max_iter, so double it and refit (liblinear
    has no warm start), raising at the cap instead of reporting a truncated fit."""
    it = SVC_MAX_ITER
    while True:
        clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=it)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            clf.fit(X, y)
        if not any(issubclass(w.category, ConvergenceWarning) for w in caught):
            clf.max_iter_used_ = it
            return clf
        if it >= SVC_MAX_ITER_CAP:
            raise RuntimeError(f"LinearSVC ({tag}) did not converge at max_iter={it}")
        it *= 2
        print(f"    [svc {tag}] not converged, refitting at max_iter={it}", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert C.DOMAIN == "jpeg", "Run with STEG_DOMAIN=jpeg"
    names = C.CLASS_FOLDERS; cov = names.index(C.COVER_NAME)
    X, y = load_scaled(args.alpha)
    gte = load_dctr_groups(args.alpha)["test"]        # cover id per test row
    Xtr, Xte = X["train"], X["test"]
    ytr, yfu, yte = y["train"], y["fusion"], y["test"]
    ybin_te = (yte != cov).astype(int)
    assert len(gte) == len(yte), "cover ids and test features are misaligned"
    print(f"== DCTR single split | alpha={args.alpha} | feat={Xtr.shape[1]} | classes={names}")
    out = {"alpha": args.alpha}

    # ---- base fits (once) + the dctr_ensemble report ----
    base = fit_base(X, y, names)
    preds, report = evaluate(base, X, y, names)
    write_report(args.alpha, names, report)
    det_p, mono_pred, mod_pred = preds["det"], preds["mono"], preds["modular"]
    Pfu, Pte = base["Pfu"], preds["Pte"]

    # ---- (1a) bootstrap CIs, resampling covers ----
    def row(fn):
        pt = float(fn(np.arange(len(yte))))
        return [round(pt, 4), group_boot(fn, gte, n=2000, seed=args.seed, point=pt)]
    out["bootstrap_CI95"] = {
        "detection_auc": row(lambda i: roc_auc_score(ybin_te[i], det_p[i])),
        "monolithic_acc": row(lambda i: accuracy_score(yte[i], mono_pred[i])),
        "modular_acc": row(lambda i: accuracy_score(yte[i], mod_pred[i])),
        "delta_modular_minus_mono": row(lambda i: accuracy_score(yte[i], mod_pred[i])
                                                  - accuracy_score(yte[i], mono_pred[i])),
        "resampling_unit": "cover",
    }

    # ---- (1b) stable separability ----
    # PCA fitted on TRAIN only (jpeg_kfold_cv fits it on all rows for a descriptive
    # geometry measure; the two tables are therefore not identical by construction).
    pca = PCA(n_components=50, random_state=0).fit(Xtr)
    Xte_p = pca.transform(Xte)
    J_pca = fisher_J(Xte_p, yte)
    pair_pca = {f"{names[a]}|{names[b]}": round(fisher_J(Xte_p[(yte==a)|(yte==b)],
                yte[(yte==a)|(yte==b)]), 3) for a, b in itertools.combinations(range(len(names)), 2)}
    J_post = fisher_J(Pte, yte)   # specialist-posteriors space (one dim per stego scheme)
    # A PRIORI version for diagnostic_vs_confusion.py: PCA and J both from TRAIN rows only, so
    # the diagnostic never sees the TEST images whose confusion it is meant to predict.
    Xtr_p = pca.transform(Xtr)
    pair_pca_train = {f"{names[a]}|{names[b]}": round(fisher_J(Xtr_p[(ytr==a)|(ytr==b)],
                      ytr[(ytr==a)|(ytr==b)]), 3) for a, b in itertools.combinations(range(len(names)), 2)}
    out["separability_stable"] = {"fisher_J_pca50_multiclass": round(J_pca, 3),
                                  "n_classes": len(names),
                                  "fisher_pairs_pca50": pair_pca,
                                  "fisher_pairs_pca50_train_rows": pair_pca_train,
                                  "fisher_J_specialist_posteriors": round(J_post, 3),
                                  "pca_fitted_on": "train split"}

    # Headline results go to disk BEFORE the classifier-robustness section below: its
    # LinearSVC/RandomForest fits are the slowest and the likeliest to fail, and a failure
    # there must not take the CIs and the transfer table down with it.
    res = C.RESULTS_DIR; res.mkdir(parents=True, exist_ok=True)
    dst = res / f"dctr_bootstrap_{C.alpha_tag(args.alpha)}.json"

    def save():
        out["provenance"] = convergence_info()
        dst.write_text(json.dumps(out, indent=2))
    save()

    # ---- leave-one-algorithm-out transfer, reusing the cover-vs-all detector ----
    print(f"\n== Detection transfer (JPEG/DCTR) alpha={args.alpha} ==")
    write_transfer(args.alpha, run_transfer(Xtr, ytr, Xte, yte, gte, names, base["det"]))

    # ---- (1c) monolithic vs modular across classifiers ----
    # The tree ensemble does not scale to the full 20,480-dim DCTR at 10k covers
    # (~42k train rows); every RF fit ON DCTR (the monolith and each specialist) is capped by
    # a fixed random subsample. This is the tree-baseline caveat itself (it needs far more
    # data on this high-dim feature than the linear classifiers), so the cap is documented
    # rather than hidden. The linear classifiers and the low-dim meta-learners use all rows.
    RF_TRAIN_CAP = 8000
    algos = [c for c in names if c != C.COVER_NAME]
    rng = np.random.default_rng(args.seed)          # local: never shared with the bootstrap

    def cap_rows(Xf, yf, cap):
        if cap and len(yf) > cap:
            sel = rng.choice(len(yf), cap, replace=False)
            return Xf[sel], yf[sel]
        return Xf, yf

    def ceiling_row(mono_p, mod_p):
        a_mono, a_mod = accuracy_score(yte, mono_p), accuracy_score(yte, mod_p)
        return {"mono_acc": round(float(a_mono), 4), "modular_acc": round(float(a_mod), 4),
                "delta": round(float(a_mod - a_mono), 4)}

    # LogReg: identical to the base monolith and meta-learner, so reuse their predictions.
    inv = {"LogReg": ceiling_row(mono_pred, mod_pred)}
    rf = lambda Xf, yf: RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                               random_state=0).fit(Xf, yf)
    # (fitter, row cap on fits over DCTR, specialist score: P(stego) or its decision value)
    clfs = {"LinearSVC": (lambda Xf, yf, t: fit_linearsvc(Xf, yf, tag=t), None,
                          lambda mdl, Z: mdl.decision_function(Z)),
            "RandomForest": (lambda Xf, yf, t: rf(Xf, yf), RF_TRAIN_CAP,
                             lambda mdl, Z: mdl.predict_proba(Z)[:, 1])}
    failed = []
    for nm, (fitter, cap, score) in clfs.items():
        try:
            cm = fitter(*cap_rows(Xtr, ytr, cap), f"{nm}:mono").predict(Xte)   # monolithic
            # modular: this family's own cover-vs-one specialists on TRAIN, meta on FUSION scores
            Sfu, Ste = [], []
            for a in algos:
                ai = names.index(a); m = (ytr == cov) | (ytr == ai)
                sp = fitter(*cap_rows(Xtr[m], (ytr[m] == ai).astype(int), cap), f"{nm}:spec:{a}")
                Sfu.append(score(sp, X["fusion"])); Ste.append(score(sp, Xte))
            cme = fitter(np.column_stack(Sfu), yfu, f"{nm}:meta").predict(np.column_stack(Ste))
        except RuntimeError as e:          # non-convergence at the cap: record it, keep the rest
            failed.append(nm)
            inv[nm] = {"error": str(e)}
            print(f"WARNING: ceiling_by_classifier/{nm} FAILED: {e}", flush=True)
            continue
        inv[nm] = ceiling_row(cm, cme)
        inv[nm]["modular_base_learners"] = f"{nm} specialists (TRAIN) + {nm} meta (FUSION)"
        if cap:
            inv[nm]["dctr_fit_train_cap"] = cap
    out["ceiling_by_classifier"] = inv
    out["ceiling_by_classifier_failed"] = failed

    save()
    print(json.dumps(out, indent=2))
    print("saved:", dst)
    if failed:
        print(f"\nWARNING: the robustness rows {failed} did not converge and are recorded as "
              f"errors in {dst.name}; the headline results above are complete.", flush=True)


if __name__ == "__main__":
    main()
