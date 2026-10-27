"""Pairwise Fisher J on the leading k PCA components of the DCTR features for k in --dims, with
subsampling CIs and the correlation with the pairwise confusion at each k. --protocol all uses
the out-of-fold confusion; --protocol apriori fits PCA and J on the training rows of the
single split and uses its test confusion.

Usage:  STEG_DOMAIN=jpeg python pt/fisher_pca_sweep.py --alpha 0.4 [--protocol apriori]
Output: $STEG_RESULTS/fisher_pca_sweep_<alpha>[_apriori].json
"""
import sys, argparse, json, itertools, time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import config as C
from util import fisher_J
from jpeg_kfold_cv import load_or_extract
from diagnostic_vs_confusion import analyse, j_matrix

LEGACY = set(getattr(C, "LEGACY_SCHEMES", ["nsF5"]))


def pair_table(Z, y, groups, names, rng, n_sub, frac):
    """Pairwise J (point + subsampling CI) on an already-projected matrix Z."""
    ucov = np.unique(groups); by = {c: np.where(groups == c)[0] for c in ucov}
    m_sub = max(2, int(frac * len(ucov)))
    draws = [rng.choice(ucov, m_sub, replace=False) for _ in range(n_sub)]
    pairs = {}
    for a, b in itertools.combinations(range(len(names)), 2):
        m = (y == a) | (y == b)
        point = float(fisher_J(Z[m], y[m]))
        sub = []
        for samp in draws:
            idx = np.concatenate([by[c] for c in samp])
            idx = idx[(y[idx] == a) | (y[idx] == b)]
            sub.append(fisher_J(Z[idx], y[idx]))
        pairs[f"{names[a]}|{names[b]}"] = [round(point, 4),
                                           [round(float(np.percentile(sub, 2.5)), 4),
                                            round(float(np.percentile(sub, 97.5)), 4)]]
    return pairs


def summarise(pairs, names, cover):
    """Gap between the weakest legacy pair and the strongest adaptive-only pair, plus the
    cover-vs-stego values, so the reader can see what the projection keeps."""
    def kind(p):
        a, b = p.split("|")
        if cover in (a, b):
            return "cover"
        if a in LEGACY or b in LEGACY:
            return "legacy"
        return "adaptive"
    legacy = {p: v[0] for p, v in pairs.items() if kind(p) == "legacy"}
    adapt = {p: v[0] for p, v in pairs.items() if kind(p) == "adaptive"}
    coverp = {p: v[0] for p, v in pairs.items() if kind(p) == "cover"}
    return {"min_legacy_pair": min(legacy.items(), key=lambda kv: kv[1]),
            "max_adaptive_pair": max(adapt.items(), key=lambda kv: kv[1]),
            "legacy_above_all_adaptive": min(legacy.values()) > max(adapt.values()),
            "adaptive_min_pair": min(adapt.items(), key=lambda kv: kv[1]),
            "cover_pairs": coverp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=C.ID_ALPHA if hasattr(C, "ID_ALPHA") else 0.4)
    ap.add_argument("--dims", default="20,50,100,200,500")
    ap.add_argument("--n-sub", type=int, default=100, help="subsampling draws per pair")
    ap.add_argument("--frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--protocol", choices=("all", "apriori"), default="all",
                    help="all: PCA and J on all rows, confusion = out-of-fold k-fold matrix "
                         "(descriptive). apriori: PCA and J on the TRAIN rows of the single "
                         "70/15/15 split only, confusion = that split's TEST matrix "
                         "(dctr_ensemble), so J never sees the images whose confusion it "
                         "is meant to anticipate (as dctr_bootstrap's *_train_rows table).")
    args = ap.parse_args()
    dims = [int(d) for d in args.dims.split(",")]
    kmax = max(dims)
    at = C.alpha_tag(args.alpha)

    t0 = time.time()
    X, y, groups, names, meta = load_or_extract(args.alpha)
    names = [str(n) for n in names]
    cover = C.COVER_NAME
    print(f"features {X.shape} classes {names}  ({time.time()-t0:.0f}s)", flush=True)
    if args.protocol == "apriori":
        train_stems = {p.stem for p in (C.split_root(args.alpha) / "train" / cover).iterdir()}
        keep = np.isin(groups, list(train_stems))
        X, y, groups = X[keep], y[keep], groups[keep]
        en = json.loads((Path(C.RESULTS_DIR) / f"dctr_ensemble_{at}.json").read_text())
        assert list(en["classes"]) == names, "class order differs from the single-split result"
        cm_oof = en["monolithic"]["cm"]
        print(f"a priori protocol: {keep.sum()} train rows ({len(train_stems)} covers); "
              f"confusion = single-split TEST monolithic cm", flush=True)
    else:
        # Out-of-fold confusion matrix of the monolithic classifier (jpeg_kfold_cv), for the
        # diagnostic-vs-confusion test at each k (same subsets as diagnostic_vs_confusion.py).
        kf = json.loads((Path(C.RESULTS_DIR) / f"jpeg_kfold_{at}.json").read_text())
        assert list(kf["alpha_classes"]) == names, "class order differs from the k-fold result"
        cm_oof = kf["monolithic_cm"]
    ci = names.index(cover)
    stego = [i for i in range(len(names)) if i != ci]
    subsets = {"all_classes": list(range(len(names))), "stego_only": stego,
               "adaptive_only": [i for i in stego if names[i] not in LEGACY]}

    sc = StandardScaler().fit(X)
    Xs = sc.transform(X).astype(np.float32)
    del X
    # One PCA fit at kmax; the leading k components are the k-dimensional PCA (same scaler,
    # same solver seed as fisher_pairs_ci, which fits PCA(50, random_state=0)).
    pca = PCA(n_components=kmax, random_state=0).fit(Xs)
    Zall = pca.transform(Xs)
    evr = pca.explained_variance_ratio_
    print(f"PCA({kmax}) fitted ({time.time()-t0:.0f}s); cum. explained variance at "
          + ", ".join(f"{k}:{evr[:k].sum():.3f}" for k in dims), flush=True)

    rows = "all rows" if args.protocol == "all" else "TRAIN rows of the single split"
    cmsrc = ("out-of-fold monolithic cm (jpeg_kfold_cv)" if args.protocol == "all"
             else "single-split TEST monolithic cm (dctr_ensemble)")
    out = {"alpha": args.alpha, "classes": names, "dims": dims, "n_sub": args.n_sub,
           "frac": args.frac, "seed": args.seed, "rows": rows, "confusion_source": cmsrc,
           "protocol": ("pairwise Fisher J on the leading k PCA components of the standardised "
                        f"DCTR features ({rows}; PCA unsupervised); CI by subsampling "
                        f"{args.frac:.0%} of covers without replacement; diagnostic = Spearman r "
                        f"and exact Mantel test against {cmsrc}"),
           "feature_cache_meta": meta, "cum_explained_variance": {}, "by_dim": {}}
    ref = None
    for k in dims:
        rng = np.random.default_rng(args.seed)
        Z = Zall[:, :k]
        pairs = pair_table(Z, y, groups, names, rng, args.n_sub, args.frac)
        summ = summarise(pairs, names, cover)
        summ["multiclass_J"] = round(float(fisher_J(Z, y)), 4)
        order = [p for p, _ in sorted(pairs.items(), key=lambda kv: kv[1][0])]
        if ref is None:
            ref = order
        rho = spearmanr([ref.index(p) for p in order], list(range(len(order)))).correlation
        summ["spearman_vs_first_dim"] = round(float(rho), 3)
        stego_pairs = {p: v for p, v in pairs.items() if cover not in p}
        rho_s = spearmanr([ref.index(p) for p in sorted(stego_pairs, key=lambda p: stego_pairs[p][0])],
                          list(range(len(stego_pairs)))).correlation
        summ["spearman_vs_first_dim_stego_only"] = round(float(rho_s), 3)
        diag = analyse(j_matrix(pairs, list(names)), cm_oof, list(names), subsets)
        out["cum_explained_variance"][str(k)] = round(float(evr[:k].sum()), 4)
        out["by_dim"][str(k)] = {"pairs": pairs, "summary": summ, "ordering_low_to_high": order,
                                 "diagnostic_vs_oof_confusion": diag}
        dg = "  ".join(f"{lab}: r={v['spearman_r']} p={v['mantel_p_one_sided_negative']}"
                       for lab, v in diag.items())
        print(f"k={k:4d}  multiclass J={summ['multiclass_J']:.4f}  "
              f"legacy>adaptive: {summ['legacy_above_all_adaptive']}  "
              f"min legacy {summ['min_legacy_pair']}  max adaptive {summ['max_adaptive_pair']}  "
              f"rho vs k={dims[0]}: {summ['spearman_vs_first_dim']}  | {dg}  ({time.time()-t0:.0f}s)",
              flush=True)

    suffix = "" if args.protocol == "all" else "_apriori"
    dst = Path(C.RESULTS_DIR) / f"fisher_pca_sweep_{at}{suffix}.json"
    dst.write_text(json.dumps(out, indent=1))
    print("saved:", dst)


if __name__ == "__main__":
    main()
