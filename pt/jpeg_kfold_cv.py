"""Cross-validated JPEG/DCTR analysis, k-fold disjoint by cover (GroupKFold), each estimate
with a 95% bootstrap CI by cover: detection AUC, monolithic and modular six-class accuracy
(level-1 posteriors cross-fitted inside the training fold), their paired difference,
specialist AUCs, leave-one-algorithm-out transfer, and pairwise Fisher J on a PCA-50 of all
rows. DCTR is extracted once for all covers and cached.

Usage:
  STEG_DOMAIN=jpeg uv run python pt/jpeg_kfold_cv.py --alpha 0.4 --folds 5
  STEG_DOMAIN=jpeg uv run python pt/jpeg_kfold_cv.py --selftest
"""
import sys, argparse, json, itertools
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score

import config as C
from util import fisher_J, fit_logreg, convergence_info, group_boot


# --------------------------------------------------------------------------- #
# DCTR extraction (once), cached to <DATA>/DATA_JPEG_q75_aXXX_KFOLD.npz
# --------------------------------------------------------------------------- #
def inventory(alpha):
    """Covers usable for the CV (all classes present) and the provenance of the feature
    cache they define: DCTR (q, T), class list, cover count and a hash of every source
    file's name and size. A regenerated dataset (new seeds / container) changes the sizes,
    so a stale cache can no longer be silently reused."""
    import hashlib
    from extract_dctr import DCTR_Q, DCTR_T, DECOMPRESSION
    names = C.CLASS_FOLDERS
    root = C.data_root(alpha)
    stems = sorted(p.stem for p in (root / C.COVER_NAME).glob("*.jpg"))
    stems = [s for s in stems if all((root / nm / f"{s}.jpg").exists() for nm in names)]
    h = hashlib.sha256()
    for nm in names:
        for s in stems:
            h.update(f"{nm}/{s}:{(root / nm / f'{s}.jpg').stat().st_size}\n".encode())
    meta = {"alpha": alpha, "q": DCTR_Q, "T": DCTR_T, "dctr_variant": "unmerged 64 phases",
            "decompression": DECOMPRESSION,
            "classes": list(names), "n_covers": len(stems), "files_sha256": h.hexdigest()}
    return stems, meta


def extract_all(alpha, stems):
    from extract_dctr import dct_basis_filters, dctr_features_batch, load_batch, DCTR_Q, DCTR_T
    from engine import DEVICE
    names = C.CLASS_FOLDERS
    root = C.data_root(alpha)
    print(f"  covers with all {len(names)} classes present: {len(stems)}", flush=True)
    W = dct_basis_filters().to(DEVICE)

    def feats(paths):
        X = np.zeros((len(paths), 64 * 64 * (DCTR_T + 1)), np.float32)
        for s in range(0, len(paths), 64):
            b = paths[s:s + 64]
            X[s:s + len(b)] = dctr_features_batch(load_batch(b, DEVICE), W, DCTR_Q, DCTR_T)
        return X

    Xs, ys, gs = [], [], []
    for ci, nm in enumerate(names):
        Xs.append(feats([root / nm / f"{st}.jpg" for st in stems]))
        ys.append(np.full(len(stems), ci, np.int64))
        gs.append(np.array(stems))
        print(f"  DCTR {nm}: {len(stems)} imgs", flush=True)
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(gs), names


def load_or_extract(alpha, rebuild=False):
    cache = C.DATASETS / f"{C.DATA_PREFIX}_{C.alpha_tag(alpha)}_KFOLD.npz"
    stems, meta = inventory(alpha)
    if not stems:
        sys.exit(f"no covers with all classes under {C.data_root(alpha)}")
    if cache.exists() and not rebuild:
        d = np.load(cache, allow_pickle=True)          # npz is lazy: meta read before X
        old = json.loads(str(d["meta"])) if "meta" in d.files else None
        if old != meta:
            diff = ("no provenance stored (cache predates provenance checks)" if old is None
                    else ", ".join(f"{k}: cache={old.get(k)!r} now={v!r}"
                                   for k, v in meta.items() if old.get(k) != v))
            sys.exit(f"refusing to reuse {cache.name}: {diff}. Re-run with --rebuild to "
                     f"re-extract DCTR from the current dataset.")
        return d["X"], d["y"], d["groups"], list(d["names"]), meta
    X, y, groups, names = extract_all(alpha, stems)
    np.savez(cache, X=X, y=y, groups=groups, names=np.array(names),
             meta=np.array(json.dumps(meta)))
    print("cached:", cache)
    return X, y, groups, names, meta


# --------------------------------------------------------------------------- #
# bootstrap CI by cover (shared implementation in util.group_boot)
# --------------------------------------------------------------------------- #
def boot_ci(metric_fn, groups, mask=None, n=1000, seed=0, point=None):
    """95% CI of metric_fn(row_idx), resampling COVERS with replacement.
    `mask` restricts to a subset of rows (e.g. cover + one scheme); `point` warns if the
    CI excludes the full-sample estimate."""
    return group_boot(metric_fn, groups, mask=mask, n=n, seed=seed, point=point)


# --------------------------------------------------------------------------- #
# k-fold CV (by cover) producing out-of-fold predictions
# --------------------------------------------------------------------------- #
PARTS = ("detection", "identification", "transfer", "separability")


class FoldCheckpoint:
    """Per-fold out-of-fold predictions on disk, so an interrupted CV resumes at the first
    unfinished fold instead of refitting days of 20,480-d logistic regressions. A fold file
    is reused only if its fingerprint (feature-cache provenance, folds, inner folds, fitted
    parts, row count) and its test rows match the current run; otherwise the run stops
    rather than mixing predictions from different configurations."""

    def __init__(self, directory, fingerprint):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fp = json.dumps(fingerprint, sort_keys=True)

    def _path(self, fold):
        return self.dir / f"fold{fold}.npz"

    def load(self, fold, te, arrays):
        p = self._path(fold)
        if not p.exists():
            return False
        d = np.load(p)
        if str(d["fingerprint"]) != self.fp or not np.array_equal(d["te"], te):
            sys.exit(f"refusing to resume from {p}: written by a different configuration or "
                     f"dataset. Delete {self.dir} to restart the CV from fold 1.")
        for k, arr in arrays.items():
            arr[te] = d[k]
        return True

    def save(self, fold, te, arrays):
        tmp = self.dir / f"fold{fold}.tmp.npz"            # atomic: no half-written fold file
        np.savez(tmp, fingerprint=np.array(self.fp), te=te,
                 **{k: arr[te] for k, arr in arrays.items()})
        tmp.replace(self._path(fold))


def cv_predictions(X, y, groups, names, folds, seed=0, inner_folds=4, parts=PARTS, ckpt=None):
    """Out-of-fold predictions for the requested arms, over a GroupKFold by cover.

    `parts` selects what is fitted: detection = 1 fit per fold; identification = monolith,
    specialists and cross-fitted stacking (1 + S + inner_folds*S + 1); transfer = S fits.
    Arms not requested stay NaN / -1 and are not reported.

    The modular arm uses CROSS-FITTED level-1 features: within each training fold an inner
    GroupKFold produces out-of-fold specialist posteriors covering *all* of that fold, so
    the meta-learner trains on every training cover while no specialist ever scores an
    image it was fitted on. The specialists that score the test fold are refit on the whole
    training fold. Both arms therefore see the same images, which an earlier 70/30 inner
    split did not: it gave the specialists only 70% of the training fold and so charged the
    modular arm a data-budget penalty on top of the posterior-compression effect the
    comparison is meant to isolate.
    """
    N = len(y); cov = 0
    algos = list(range(1, len(names)))
    det_p = np.full(N, np.nan)                       # P(stego)
    mono_pred = np.full(N, -1, np.int64)
    mod_pred = np.full(N, -1, np.int64)
    spec_p = {a: np.full(N, np.nan) for a in algos}  # cover-vs-one, filled on cover|a rows
    trans_p = {a: np.full(N, np.nan) for a in algos}  # leave-one-out, filled on cover|a rows
    # every per-row prediction array, by name (views: filling arrays[k] fills the originals)
    arrays = {"det_p": det_p, "mono_pred": mono_pred, "mod_pred": mod_pred,
              **{f"spec_{a}": spec_p[a] for a in algos},
              **{f"trans_{a}": trans_p[a] for a in algos}}

    gkf = GroupKFold(n_splits=folds)                 # deterministic (no shuffle): same folds on resume
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        if ckpt is not None and ckpt.load(fold, te, arrays):
            print(f"  fold {fold+1}/{folds} restored from checkpoint", flush=True)
            continue
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        ytr = y[tr]
        ident = "identification" in parts
        # detection
        if "detection" in parts:
            det = fit_logreg(Xtr, (ytr != cov).astype(int), tag="detect")
            det_p[te] = det.predict_proba(Xte)[:, 1]
        # monolithic: the whole training fold
        if ident:
            mono_pred[te] = fit_logreg(Xtr, ytr, tag="mono").predict(Xte)

        # specialists (cover vs one) on the whole training fold. The same fitted model
        # serves the cover-vs-one AUC and the modular arm's test-fold posteriors.
        P_te = np.zeros((len(te), len(algos)))
        for k, a in enumerate(algos):
            loc = (y[te] == cov) | (y[te] == a)      # cover|a rows, as positions within te
            te_a = te[loc]
            if ident:
                m = (ytr == cov) | (ytr == a)
                s = fit_logreg(Xtr[m], (ytr[m] == a).astype(int), tag=f"spec:{names[a]}")
                P_te[:, k] = s.predict_proba(Xte)[:, 1]
                spec_p[a][te_a] = P_te[loc, k]
            if "transfer" in parts:
                # transfer: train cover vs the OTHER stego schemes, test on held a
                mt = (ytr == cov) | ((ytr != cov) & (ytr != a))
                t = fit_logreg(Xtr[mt], (ytr[mt] != cov).astype(int), tag=f"transfer:{names[a]}")
                trans_p[a][te_a] = t.predict_proba(Xte[loc])[:, 1]

        if ident:
            # modular stacking: cross-fitted level-1 features over the whole training fold
            gtr = groups[tr]
            P_oof = np.zeros((len(tr), len(algos)))
            for inner, (itr, ite) in enumerate(GroupKFold(n_splits=inner_folds).split(Xtr, ytr, gtr)):
                for k, a in enumerate(algos):
                    m = itr[(ytr[itr] == cov) | (ytr[itr] == a)]
                    s = fit_logreg(Xtr[m], (ytr[m] == a).astype(int),
                                   tag=f"oof{inner}:{names[a]}")
                    P_oof[ite, k] = s.predict_proba(Xtr[ite])[:, 1]
            meta = fit_logreg(P_oof, ytr, tag="meta")
            mod_pred[te] = meta.predict(P_te)
        if ckpt is not None:
            ckpt.save(fold, te, arrays)
        print(f"  fold {fold+1}/{folds} done", flush=True)
    return det_p, mono_pred, mod_pred, spec_p, trans_p


# --------------------------------------------------------------------------- #
def summarize(X, y, groups, names, folds, seed=0, inner_folds=4, parts=PARTS,
              ckpt_dir=None, cache_meta=None, oof_path=None):
    """ckpt_dir: per-fold checkpoints (resume); oof_path: .npz with every out-of-fold
    prediction, so later analyses (confusion structure, diagnostic-vs-confusion) need no refit."""
    cov = 0; algos = list(range(1, len(names)))
    fit_parts = [p for p in parts if p != "separability"]
    out = {"alpha_classes": names, "folds": folds, "inner_folds": inner_folds,
           "n_covers": int(len(np.unique(groups))), "parts": list(parts)}
    if fit_parts:
        ckpt = None if ckpt_dir is None else FoldCheckpoint(ckpt_dir, {
            "feature_cache": cache_meta, "folds": folds, "inner_folds": inner_folds,
            "parts": sorted(fit_parts), "n_rows": int(len(y))})
        det_p, mono_pred, mod_pred, spec_p, trans_p = cv_predictions(
            X, y, groups, names, folds, seed, inner_folds, parts=fit_parts, ckpt=ckpt)
        if oof_path is not None:
            np.savez(oof_path, y=y, groups=groups, names=np.array(names),
                     parts=np.array(sorted(fit_parts)), det_p=det_p,
                     mono_pred=mono_pred, mod_pred=mod_pred,
                     spec_p=np.column_stack([spec_p[a] for a in algos]),
                     trans_p=np.column_stack([trans_p[a] for a in algos]),
                     note=np.array("out-of-fold predictions; NaN / -1 = arm not fitted; "
                                   "spec_p/trans_p columns follow names[1:] and are filled "
                                   "only on COVER + that scheme's rows"))
            out["oof_predictions"] = Path(oof_path).name
    ybin = (y != cov).astype(int)

    allrows = np.arange(len(y))

    def row(fn, mask=None):
        pt = float(fn(allrows if mask is None else np.where(mask)[0]))
        return [round(pt, 4), boot_ci(fn, groups, mask, point=pt)]
    if "detection" in parts:
        out["detection_auc"] = row(lambda idx: roc_auc_score(ybin[idx], det_p[idx]))
    if "identification" in parts:
        # monolithic / modular / delta
        acc_mono = lambda idx: accuracy_score(y[idx], mono_pred[idx])
        acc_mod = lambda idx: accuracy_score(y[idx], mod_pred[idx])
        out["monolithic_acc"] = row(acc_mono)
        out["modular_acc"] = row(acc_mod)
        out["delta_modular_minus_mono"] = row(lambda idx: acc_mod(idx) - acc_mono(idx))
        from sklearn.metrics import confusion_matrix
        K = range(len(names))
        out["monolithic_cm"] = confusion_matrix(y, mono_pred, labels=K).tolist()   # rows = truth
        out["modular_cm"] = confusion_matrix(y, mod_pred, labels=K).tolist()
        out["specialist_auc"] = {}
        for a in algos:
            mask = (y == cov) | (y == a)
            out["specialist_auc"][names[a]] = row(
                lambda idx: roc_auc_score((y[idx] == a).astype(int), spec_p[a][idx]), mask)
    if "transfer" in parts:
        out["transfer_auc"] = {}
        for a in algos:
            mask = (y == cov) | (y == a)
            out["transfer_auc"][names[a]] = row(
                lambda idx: roc_auc_score((y[idx] == a).astype(int), trans_p[a][idx]), mask)

    if "separability" in parts:
        # separability: pairwise Fisher on PCA-50 with a subsampling CI (resample covers).
        # NOTE: this is a descriptive geometry measure over the whole feature set, NOT a
        # cross-validated quantity -- it shares this file with the CV results but not their
        # protocol. Anything citing it (the heatmap figure and its caption) must say so.
        out["fisher_pairs_pca50"] = fisher_pairs_ci(X, y, groups, names, seed)
        out["fisher_protocol"] = ("pairwise Fisher J on a PCA-50 of the full DCTR set "
                                  "(PCA fitted on all rows); CI by subsampling 80% of covers "
                                  "without replacement; not cross-validated")
    out["provenance"] = convergence_info()
    return out


def fisher_pairs_ci(X, y, groups, names, seed=0, n=300, frac=0.8):
    """Pairwise Fisher J on a PCA-50 of DCTR, with a SUBSAMPLING CI (repeatedly draw
    `frac` of the covers WITHOUT replacement). Subsampling avoids the duplicate-row
    inflation that bootstrap-with-replacement causes for the pinv-based Fisher ratio.

    Computed once on all rows (no train/test split): this is a property of the feature
    geometry, not a generalisation estimate."""
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
    ap.add_argument("--inner-folds", type=int, default=4, dest="inner_folds",
                    help="inner GroupKFold used to cross-fit the stacking posteriors")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rebuild", action="store_true", help="re-extract DCTR cache")
    ap.add_argument("--parts", default=",".join(PARTS),
                    help=f"comma-separated subset of {','.join(PARTS)} (default: all). "
                         "The paper needs only detection at the low payloads.")
    ap.add_argument("--selftest", action="store_true", help="synthetic smoke test, no data")
    args = ap.parse_args()
    parts = tuple(p for p in args.parts.split(",") if p)
    bad = set(parts) - set(PARTS)
    if bad or not parts:
        ap.error(f"--parts must be a non-empty subset of {PARTS}; got {args.parts!r}")

    if args.selftest:
        rng = np.random.default_rng(0); ncov, d = 120, 60
        names = C.CLASS_FOLDERS
        stems = np.array([f"{i:05d}" for i in range(ncov)])
        X, y, groups = [], [], []
        for ci in range(len(names)):
            X.append(rng.normal(ci * 0.3, 1.0, (ncov, d)))   # separable-ish
            y.append(np.full(ncov, ci)); groups.append(stems)
        X = np.vstack(X).astype(np.float32); y = np.concatenate(y); groups = np.concatenate(groups)
        out = summarize(X, y, groups, names, folds=args.folds, seed=args.seed,
                        inner_folds=args.inner_folds, parts=parts)
        print(json.dumps(out, indent=2)); return

    assert C.DOMAIN == "jpeg", "run with STEG_DOMAIN=jpeg"
    X, y, groups, names, meta = load_or_extract(args.alpha, rebuild=args.rebuild)
    print(f"loaded X={X.shape} covers={len(np.unique(groups))} classes={names}", flush=True)
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = C.alpha_tag(args.alpha)
    out = summarize(X, y, groups, names, folds=args.folds, seed=args.seed,
                    inner_folds=args.inner_folds, parts=parts,
                    ckpt_dir=C.RESULTS_DIR / f"jpeg_kfold_{tag}_folds", cache_meta=meta,
                    oof_path=C.RESULTS_DIR / f"jpeg_kfold_{tag}_oof.npz")
    out["feature_cache"] = meta
    dst = C.RESULTS_DIR / f"jpeg_kfold_{tag}.json"
    dst.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2)); print("saved:", dst)


if __name__ == "__main__":
    main()
