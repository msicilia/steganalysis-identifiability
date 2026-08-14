"""
Cross-validated JPEG/DCTR analysis with confidence intervals: the headline detection,
identification, transfer and separability numbers, each with a 95% CI.

Instead of a single 70/15/15 split, this runs k-fold cross-validation DISJOINT BY COVER
(GroupKFold on cover id), so every cover is tested exactly once, out-of-fold. Every headline
number is then reported as a cross-validated estimate with a 95% bootstrap CI (resampling by
cover):

  - detection      : COVER vs any stego, AUC
  - monolithic     : 6-class accuracy
  - modular        : 6-class accuracy of the leak-free stacking ensemble
                     (per fold: specialists on 70% of train covers, meta-model on the other
                      30% -- the same train/fusion/test recipe as the paper, nested in CV)
  - delta          : modular - monolithic (paired)
  - specialists    : COVER-vs-one AUC per scheme
  - transfer       : leave-one-algorithm-out COVER-vs-held AUC per scheme
  - separability   : pairwise Fisher J on a PCA-50 of DCTR, with bootstrap CIs

DCTR is extracted once for all covers x classes and cached; all CV is then in-memory.

Usage:
  STEG_DOMAIN=jpeg uv run python pt/jpeg_kfold_cv.py --alpha 0.4 --folds 5
  STEG_DOMAIN=jpeg uv run python pt/jpeg_kfold_cv.py --selftest      # synthetic smoke test
"""
import sys, argparse, json, itertools
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score

import config as C
from util import fisher_J

LR = lambda: LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")


# --------------------------------------------------------------------------- #
# DCTR extraction (once), cached to <DATA>/DATA_JPEG_q75_aXXX_KFOLD.npz
# --------------------------------------------------------------------------- #
def extract_all(alpha):
    import torch
    from PIL import Image
    from extract_dctr import dct_basis_filters, dctr_features_batch
    from engine import DEVICE
    names = C.CLASS_FOLDERS
    root = C.data_root(alpha)
    stems = sorted(p.stem for p in (root / C.COVER_NAME).glob("*.jpg"))
    stems = [s for s in stems if all((root / nm / f"{s}.jpg").exists() for nm in names)]
    print(f"  covers with all {len(names)} classes present: {len(stems)}", flush=True)
    W = dct_basis_filters().to(DEVICE)

    def feats(paths):
        X = np.zeros((len(paths), 64 * 64 * 5), np.float32)
        for s in range(0, len(paths), 64):
            b = paths[s:s + 64]
            imgs = torch.stack([torch.from_numpy(np.asarray(Image.open(p).convert("L"), np.float32))
                                for p in b])[:, None].to(DEVICE)
            X[s:s + len(b)] = dctr_features_batch(imgs, W, 4.0, 4)
        return X

    Xs, ys, gs = [], [], []
    for ci, nm in enumerate(names):
        Xs.append(feats([root / nm / f"{st}.jpg" for st in stems]))
        ys.append(np.full(len(stems), ci, np.int64))
        gs.append(np.array(stems))
        print(f"  DCTR {nm}: {len(stems)} imgs", flush=True)
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(gs), names


def load_or_extract(alpha, rebuild=False):
    cache = C.DATASETS / f"DATA_JPEG_q75_{C.alpha_tag(alpha)}_KFOLD.npz"
    if cache.exists() and not rebuild:
        d = np.load(cache, allow_pickle=True)
        return d["X"], d["y"], d["groups"], list(d["names"])
    X, y, groups, names = extract_all(alpha)
    np.savez(cache, X=X, y=y, groups=groups, names=np.array(names))
    print("cached:", cache)
    return X, y, groups, names


# --------------------------------------------------------------------------- #
# bootstrap CI by cover
# --------------------------------------------------------------------------- #
def boot_ci(metric_fn, groups, mask=None, n=1000, seed=0):
    """95% CI of metric_fn(row_idx), resampling COVERS with replacement.
    `mask` restricts to a subset of rows (e.g. cover + one scheme)."""
    rng = np.random.default_rng(seed)
    rows = np.arange(len(groups)) if mask is None else np.where(mask)[0]
    g = groups[rows]
    by_cover = {c: rows[g == c] for c in np.unique(g)}
    covers = np.array(list(by_cover.keys()))
    vals = []
    for _ in range(n):
        samp = rng.choice(covers, len(covers), replace=True)
        idx = np.concatenate([by_cover[c] for c in samp])
        try:
            vals.append(metric_fn(idx))
        except ValueError:
            pass
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


# --------------------------------------------------------------------------- #
# k-fold CV (by cover) producing out-of-fold predictions
# --------------------------------------------------------------------------- #
def cv_predictions(X, y, groups, names, folds, seed=0):
    N = len(y); cov = 0
    algos = list(range(1, len(names)))
    det_p = np.full(N, np.nan)                       # P(stego)
    mono_pred = np.full(N, -1, np.int64)
    mod_pred = np.full(N, -1, np.int64)
    spec_p = {a: np.full(N, np.nan) for a in algos}  # cover-vs-one, filled on cover|a rows
    trans_p = {a: np.full(N, np.nan) for a in algos}  # leave-one-out, filled on cover|a rows

    gkf = GroupKFold(n_splits=folds)
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        ytr = y[tr]
        # detection
        det = LR().fit(Xtr, (ytr != cov).astype(int))
        det_p[te] = det.predict_proba(Xte)[:, 1]
        # monolithic
        mono_pred[te] = LR().fit(Xtr, ytr).predict(Xte)
        # specialists (cover vs one) + leave-one-out transfer
        for a in algos:
            m = (ytr == cov) | (ytr == a)
            s = LR().fit(Xtr[m], (ytr[m] == a).astype(int))
            te_a = te[(y[te] == cov) | (y[te] == a)]
            spec_p[a][te_a] = s.predict_proba(sc.transform(X[te_a]))[:, 1]
            # transfer: train cover vs the OTHER stego schemes, test on held a
            mt = (ytr == cov) | ((ytr != cov) & (ytr != a))
            t = LR().fit(Xtr[mt], (ytr[mt] != cov).astype(int))
            trans_p[a][te_a] = t.predict_proba(sc.transform(X[te_a]))[:, 1]
        # modular stacking (leak-free): inner 70/30 cover split of the train fold
        gtr = groups[tr]
        ucov = np.unique(gtr); rng = np.random.default_rng(seed + fold)
        rng.shuffle(ucov); cut = int(0.7 * len(ucov))
        spec_cov, meta_cov = set(ucov[:cut]), set(ucov[cut:])
        in_spec = np.array([g in spec_cov for g in gtr])
        in_meta = ~in_spec
        P_meta = np.zeros((in_meta.sum(), len(algos)))
        P_te = np.zeros((len(te), len(algos)))
        for k, a in enumerate(algos):
            m = in_spec & ((ytr == cov) | (ytr == a))
            s = LR().fit(Xtr[m], (ytr[m] == a).astype(int))
            P_meta[:, k] = s.predict_proba(Xtr[in_meta])[:, 1]
            P_te[:, k] = s.predict_proba(Xte)[:, 1]
        meta = LR().fit(P_meta, ytr[in_meta])
        mod_pred[te] = meta.predict(P_te)
        print(f"  fold {fold+1}/{folds} done", flush=True)
    return det_p, mono_pred, mod_pred, spec_p, trans_p


# --------------------------------------------------------------------------- #
def summarize(X, y, groups, names, folds, seed=0):
    cov = 0; algos = list(range(1, len(names)))
    det_p, mono_pred, mod_pred, spec_p, trans_p = cv_predictions(X, y, groups, names, folds, seed)
    ybin = (y != cov).astype(int)
    out = {"alpha_classes": names, "folds": folds, "n_covers": int(len(np.unique(groups)))}

    # detection
    out["detection_auc"] = [round(float(roc_auc_score(ybin, det_p)), 4),
                            boot_ci(lambda idx: roc_auc_score(ybin[idx], det_p[idx]), groups)]
    # monolithic / modular / delta
    acc_mono = lambda idx: accuracy_score(y[idx], mono_pred[idx])
    acc_mod = lambda idx: accuracy_score(y[idx], mod_pred[idx])
    out["monolithic_acc"] = [round(float(acc_mono(np.arange(len(y)))), 4), boot_ci(acc_mono, groups)]
    out["modular_acc"] = [round(float(acc_mod(np.arange(len(y)))), 4), boot_ci(acc_mod, groups)]
    dfun = lambda idx: acc_mod(idx) - acc_mono(idx)
    out["delta_modular_minus_mono"] = [round(float(dfun(np.arange(len(y)))), 4), boot_ci(dfun, groups)]
    # specialists + transfer, per scheme
    out["specialist_auc"], out["transfer_auc"] = {}, {}
    for a in algos:
        mask = (y == cov) | (y == a)
        ya = (y[mask] == a).astype(int)
        sp = spec_p[a][mask]
        out["specialist_auc"][names[a]] = [round(float(roc_auc_score(ya, sp)), 4),
            boot_ci(lambda idx: roc_auc_score((y[idx] == a).astype(int), spec_p[a][idx]), groups, mask)]
        tp = trans_p[a][mask]
        out["transfer_auc"][names[a]] = [round(float(roc_auc_score(ya, tp)), 4),
            boot_ci(lambda idx: roc_auc_score((y[idx] == a).astype(int), trans_p[a][idx]), groups, mask)]

    # separability: pairwise Fisher on PCA-50 with bootstrap CI (resample covers)
    out["fisher_pairs_pca50"] = fisher_pairs_ci(X, y, groups, names, seed)
    return out


def fisher_pairs_ci(X, y, groups, names, seed=0, n=300, frac=0.8):
    """Pairwise Fisher J on a PCA-50 of DCTR, with a SUBSAMPLING CI (repeatedly draw
    `frac` of the covers WITHOUT replacement). Subsampling avoids the duplicate-row
    inflation that bootstrap-with-replacement causes for the pinv-based Fisher ratio."""
    sc = StandardScaler().fit(X)
    Z = PCA(n_components=50, random_state=0).fit_transform(sc.transform(X))
    rng = np.random.default_rng(seed)
    ucov = np.unique(groups); by = {c: np.where(groups == c)[0] for c in ucov}
    m_sub = max(2, int(frac * len(ucov)))
    pairs = {}
    for a, b in itertools.combinations(range(len(names)), 2):
        m = (y == a) | (y == b)
        point = round(float(fisher_J(Z[m], y[m])), 3)
        sub = []
        for _ in range(n):
            samp = rng.choice(ucov, m_sub, replace=False)
            idx = np.concatenate([by[c] for c in samp])
            idx = idx[(y[idx] == a) | (y[idx] == b)]
            try:
                sub.append(fisher_J(Z[idx], y[idx]))
            except Exception:
                pass
        ci = [round(float(np.percentile(sub, 2.5)), 3), round(float(np.percentile(sub, 97.5)), 3)]
        pairs[f"{names[a]}|{names[b]}"] = [point, ci]
    return pairs


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rebuild", action="store_true", help="re-extract DCTR cache")
    ap.add_argument("--selftest", action="store_true", help="synthetic smoke test, no data")
    args = ap.parse_args()

    if args.selftest:
        rng = np.random.default_rng(0); ncov, d = 120, 60
        names = C.CLASS_FOLDERS
        stems = np.array([f"{i:05d}" for i in range(ncov)])
        X, y, groups = [], [], []
        for ci in range(len(names)):
            X.append(rng.normal(ci * 0.3, 1.0, (ncov, d)))   # separable-ish
            y.append(np.full(ncov, ci)); groups.append(stems)
        X = np.vstack(X).astype(np.float32); y = np.concatenate(y); groups = np.concatenate(groups)
        out = summarize(X, y, groups, names, folds=args.folds, seed=args.seed)
        print(json.dumps(out, indent=2)); return

    assert C.DOMAIN == "jpeg", "run with STEG_DOMAIN=jpeg"
    X, y, groups, names = load_or_extract(args.alpha, rebuild=args.rebuild)
    print(f"loaded X={X.shape} covers={len(np.unique(groups))} classes={names}", flush=True)
    out = summarize(X, y, groups, names, folds=args.folds, seed=args.seed)
    dst = C.RESULTS_DIR / f"jpeg_kfold_{C.alpha_tag(args.alpha)}.json"
    dst.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2)); print("saved:", dst)


if __name__ == "__main__":
    main()
