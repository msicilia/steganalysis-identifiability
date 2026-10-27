"""Spearman correlation between pairwise Fisher J and the pairwise confusion rate
(cm[a,b] + cm[b,a]) / (n_a + n_b), with an exact Mantel permutation test over class labels.

JPEG: train-row J against the TEST confusion of the single split, and all-row J against the
out-of-fold confusion. Spatial: SRM statistics against their linear probe, and GBRAS detector
features against the GBRAS six-class network. Each over all classes, stego classes and
adaptive schemes.

Usage:  python diagnostic_vs_confusion.py --domain jpeg|spatial [--alpha 0.4]
"""
import sys, json, argparse, itertools
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
from scipy.stats import spearmanr
import config as C


def load(name):
    p = C.RESULTS_DIR / name
    if not p.exists():
        sys.exit(f"missing {p}: run the pipeline step that writes it first")
    return json.loads(p.read_text())


def j_matrix(pairs, names):
    """{'A|B': v or [v, ci]} -> symmetric K x K matrix (NaN diagonal)."""
    idx = {n: i for i, n in enumerate(names)}
    M = np.full((len(names), len(names)), np.nan)
    for k, v in pairs.items():
        a, b = k.split("|")
        M[idx[a], idx[b]] = M[idx[b], idx[a]] = v[0] if isinstance(v, (list, tuple)) else v
    return M


def confusion_rates(cm):
    """K x K pairwise confusion rate from a confusion matrix (rows = truth)."""
    cm = np.asarray(cm, float); n = cm.sum(1); K = len(cm)
    Q = np.full((K, K), np.nan)
    for a, b in itertools.combinations(range(K), 2):
        if n[a] + n[b] > 0:
            Q[a, b] = Q[b, a] = (cm[a, b] + cm[b, a]) / (n[a] + n[b])
    return Q


def mantel(J, Q, idx, names):
    """Spearman r between J and Q over the pairs of `idx`, with an exact permutation p-value
    for r <= observed (the predicted negative direction)."""
    idx = list(idx)
    pairs = list(itertools.combinations(range(len(idx)), 2))
    jv = np.array([J[idx[a], idx[b]] for a, b in pairs])

    def r(perm):
        qv = [Q[idx[perm[a]], idx[perm[b]]] for a, b in pairs]
        return spearmanr(jv, qv).correlation

    obs = r(tuple(range(len(idx))))
    null = np.array([r(p) for p in itertools.permutations(range(len(idx)))], float)
    null = null[~np.isnan(null)]
    ok = obs is not None and not np.isnan(obs) and len(null) > 0
    return {"classes": [names[i] for i in idx], "n_pairs": len(pairs),
            "spearman_r": round(float(obs), 4) if ok else None,
            "mantel_p_one_sided_negative": round(float(np.mean(null <= obs + 1e-12)), 4) if ok else None,
            "n_permutations": int(len(null)),
            "min_attainable_p": round(1 / len(null), 4) if len(null) else None,
            "pairs": [{"pair": f"{names[idx[a]]}|{names[idx[b]]}",
                       "fisher_J": float(J[idx[a], idx[b]]),
                       "confusion_rate": round(float(Q[idx[a], idx[b]]), 4)} for a, b in pairs]}


def analyse(J, cm, names, subsets):
    Q = confusion_rates(cm)
    return {label: mantel(J, Q, idx, names) for label, idx in subsets.items() if len(idx) >= 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=("jpeg", "spatial"), required=True)
    ap.add_argument("--alpha", type=float, default=C.FAMILY_ID_PAYLOAD)
    args = ap.parse_args()
    assert C.DOMAIN == args.domain, f"set STEG_DOMAIN={args.domain}"
    at = C.alpha_tag(args.alpha)
    names = C.CLASS_FOLDERS; cov = names.index(C.COVER_NAME)
    every = list(range(len(names)))
    stego = [i for i in every if i != cov]
    adaptive = [i for i in stego if names[i] not in C.LEGACY_SCHEMES]
    subsets = {"all_classes": every, "stego_only": stego}
    if adaptive != stego:
        subsets["adaptive_only"] = adaptive
    out = {"domain": args.domain, "alpha": args.alpha, "classes": names,
           "prediction": "negative Spearman r (higher Fisher J -> lower pairwise confusion)",
           "confusion_rate": "(cm[a,b] + cm[b,a]) / (n_a + n_b), rows = truth",
           "test": "exact Mantel permutation test over class labels, one-sided (r <= observed)",
           "sources": {}}

    if args.domain == "jpeg":
        bs, en, kf = (load(f"dctr_bootstrap_{at}.json"), load(f"dctr_ensemble_{at}.json"),
                      load(f"jpeg_kfold_{at}.json"))
        assert en["classes"] == names and kf["alpha_classes"] == names, "class order differs"
        out["sources"]["single_split_a_priori"] = {
            "fisher": "PCA-50 fitted on TRAIN, J on TRAIN rows (dctr_bootstrap)",
            "confusion": "monolithic LogReg on full DCTR, TEST split (dctr_ensemble)",
            **analyse(j_matrix(bs["separability_stable"]["fisher_pairs_pca50_train_rows"], names),
                      en["monolithic"]["cm"], names, subsets)}
        out["sources"]["cv_descriptive"] = {
            "fisher": "PCA-50 of all rows (jpeg_kfold_cv; not cross-validated)",
            "confusion": "out-of-fold monolithic LogReg (jpeg_kfold_cv)",
            **analyse(j_matrix(kf["fisher_pairs_pca50"], names), kf["monolithic_cm"], names, subsets)}
    else:
        sp = load(f"separability_srm_test_{at}.json")
        assert sp["classes"] == names, "class order differs"
        lp = sp["control_cover_vs_stego"]["linear_probe"]
        cm = np.zeros((len(names), len(names)))           # probe is stego-only: embed its 5x5
        lab = lp["stego_5class_labels"]
        for i, a in enumerate(lab):
            for j, b in enumerate(lab):
                cm[a, b] = lp["stego_5class_cm"][i][j]
        Jsrm = np.array([[np.nan if v is None else v for v in row] for row in sp["pairwise_fisher"]])
        out["sources"]["srm_statistics"] = {
            "fisher": "28-d SRM residual statistics, TEST split, all rows (separability_spatial)",
            "confusion": "grouped 5-fold linear probe, stego classes only, same features",
            **analyse(Jsrm, cm, names, {"stego_only": stego})}
        gb = load(f"gbras_ensemble_{at}.json")
        out["sources"]["gbras_detector_features"] = {
            "fisher": "GBRAS cover-vs-all detector features (no algorithm labels), TEST",
            "confusion": "GBRAS 6-class network, TEST (build_gbras_ensemble)",
            **analyse(j_matrix(gb["fisher_pairs_detect"], names),
                      gb["identification"]["monolithic_gbras_family"]["cm"], names, subsets)}

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dst = C.RESULTS_DIR / f"diagnostic_{args.domain}_{at}.json"
    dst.write_text(json.dumps(out, indent=2))
    for src, block in out["sources"].items():
        for label, res in block.items():
            if isinstance(res, dict):
                print(f"[{src} / {label}] pairs={res['n_pairs']} Spearman r={res['spearman_r']} "
                      f"Mantel p={res['mantel_p_one_sided_negative']} "
                      f"(min attainable {res['min_attainable_p']})")
    print("saved:", dst)


if __name__ == "__main__":
    main()
