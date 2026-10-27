"""Shared helpers: fit_logreg (logistic regression that raises the iteration budget until
sklearn reports convergence and fails otherwise), group_boot / boot_ci (bootstrap by cover),
Fisher J, and feature and cover-id loading.
"""
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))   # code/ (config)
import config as C


# --------------------------------------------------------------------------- #
# Classifier factory: one definition, convergence guaranteed
# --------------------------------------------------------------------------- #
LOGREG_C = 1.0
LOGREG_MAX_ITER = 5_000          # starting budget
# Per-call ceiling. sklearn forwards only `maxiter` to scipy's L-BFGS-B, whose `maxfun`
# stays at its 15,000-evaluation default, so a single call can never usefully run longer;
# past this we keep warm-starting further calls instead of raising max_iter.
LOGREG_CALL_MAX_ITER = 15_000
LOGREG_MAX_ITER_CAP = 160_000    # total iterations across calls; give up (and raise) beyond
# A failed call that advances fewer iterations than this, twice in a row, is a stalled line
# search (numerical precision floor), not a truncated fit: more budget will not fix it.
LOGREG_STALL_ITERS = 10

# Real convergence bookkeeping for this process. Scripts write convergence_info() into
# their JSON so the published numbers carry evidence that every fit converged.
MAX_ITER_USED = 0                # largest TOTAL iteration count any successful fit needed
N_FITS = 0                       # successful (converged) fits
N_FAILED = 0                     # fits that raised; nonzero only if a caller caught it


def make_logreg(**kw):
    """LogisticRegression with the project's fixed hyperparameters."""
    from sklearn.linear_model import LogisticRegression
    p = dict(C=LOGREG_C, max_iter=LOGREG_MAX_ITER, class_weight="balanced")
    p.update(kw)
    return LogisticRegression(**p)


def fit_logreg(X, y, *, tag="", verbose=False, **kw):
    """Fit a logistic regression and GUARANTEE it converged.

    Converged means lbfgs raised no ConvergenceWarning, i.e. scipy returned status 0. The
    old test (`n_iter < max_iter`) was wrong: a stop on the function-evaluation limit or a
    failed line search also leaves n_iter below max_iter.

    On failure the SAME estimator is refit with warm_start=True, so each call continues
    from the previous coefficients (sklearn's lbfgs seeds w0 from coef_/intercept_ for both
    the binary and the multinomial loss); the new call's budget equals the iterations
    spent so far (doubling the total), capped at LOGREG_CALL_MAX_ITER per call. Iterations
    are summed across calls. Raises RuntimeError at LOGREG_MAX_ITER_CAP total, or on a
    stalled line search, rather than returning a half-optimised model: such a model still
    predicts -- it just predicts something that depends on the cap rather than on the data.
    The returned model has warm_start reset to False and carries `n_iter_total_`.
    """
    global MAX_ITER_USED, N_FITS, N_FAILED
    from sklearn.exceptions import ConvergenceWarning
    start = kw.pop("max_iter", LOGREG_MAX_ITER)
    name = f"logistic regression{' (' + tag + ')' if tag else ''}"
    step = min(start, LOGREG_CALL_MAX_ITER)
    clf = make_logreg(max_iter=step, warm_start=True, **kw)
    total, stalls = 0, 0
    while True:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            clf.fit(X, y)
        for w in caught:                            # pass unrelated warnings through
            if not issubclass(w.category, ConvergenceWarning):
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        conv = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
        n_call = int(np.max(clf.n_iter_))
        total += n_call
        if not conv:
            MAX_ITER_USED = max(MAX_ITER_USED, total); N_FITS += 1
            clf.set_params(warm_start=False)        # a later refit by a caller starts fresh
            clf.n_iter_total_ = total
            if verbose:
                print(f"    [fit{' ' + tag if tag else ''}] converged in {total} iters",
                      flush=True)
            return clf
        # a stall stops short of its own budget (line search / maxfun), with little progress;
        # a call that simply used up a small budget is a truncation and just continues
        stalls = stalls + 1 if n_call < min(LOGREG_STALL_ITERS, step) else 0
        why = conv[0].split("\n")[1] if "\n" in conv[0] else conv[0]
        if stalls >= 2 or total >= LOGREG_MAX_ITER_CAP:
            N_FAILED += 1
            raise RuntimeError(
                f"{name} did not converge after {total} iterations (n={len(y)}, "
                f"d={np.shape(X)[1]}; last lbfgs stop: {why.strip()}). "
                + ("The line search stalled on consecutive warm restarts. "
                   if stalls >= 2 else "")
                + "Refusing to report a truncated fit; rescale the features, raise "
                  "LOGREG_MAX_ITER_CAP or reduce the feature dimension, then rerun.")
        step = min(max(total, start), LOGREG_CALL_MAX_ITER, LOGREG_MAX_ITER_CAP - total)
        clf.set_params(max_iter=step)
        print(f"    [fit{' ' + tag if tag else ''}] not converged after {total} iters "
              f"({why.strip()}); warm-starting {step} more", flush=True)


def convergence_info():
    """Provenance block for the result JSONs: what the fits actually did in this process."""
    return {"logreg_C": LOGREG_C, "logreg_max_iter_used": int(MAX_ITER_USED),
            "logreg_n_fits": int(N_FITS), "logreg_n_failed": int(N_FAILED),
            "logreg_all_converged": bool(N_FITS > 0 and N_FAILED == 0),
            "logreg_convergence_test": "no ConvergenceWarning (scipy L-BFGS-B status 0); "
                                       "warm-started continuation, total iters summed"}


# --------------------------------------------------------------------------- #
# Features and their cover ids
# --------------------------------------------------------------------------- #
def _dctr_dir(alpha):
    return C.DATASETS / f"{C.DATA_PREFIX}_{C.alpha_tag(alpha)}_DCTR"


def load_dctr(alpha):
    """Load cached DCTR features/labels for a payload.
    Returns (X, y), each a dict over the 'train'/'fusion'/'test' splits."""
    d = _dctr_dir(alpha)
    X = {s: np.load(d / f"X_{s}.npy") for s in ("train", "fusion", "test")}
    y = {s: np.load(d / f"y_{s}.npy") for s in ("train", "fusion", "test")}
    return X, y


def load_dctr_groups(alpha):
    """Cover id (file stem) per DCTR row, per split -- the unit for grouped bootstrap.

    Prefers the `g_<split>.npy` written by extract_dctr.py. For feature caches extracted
    before that was added, it reconstructs the ids by re-listing the split folders in the
    exact order extract_dctr.py used (class folders in CLASS_FOLDERS order, files sorted
    by name) and asserts the count matches, so a stale cache cannot silently misalign.
    """
    d = _dctr_dir(alpha)
    out = {}
    for split in ("train", "fusion", "test"):
        f = d / f"g_{split}.npy"
        if f.exists():
            out[split] = np.load(f, allow_pickle=True).astype(str)
            continue
        src = C.split_root(alpha) / split
        stems = []
        for cls in C.CLASS_FOLDERS:
            paths = []
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                paths += (src / cls).glob(ext)
            stems += [p.stem for p in sorted(paths, key=lambda p: p.name)]
        n = len(np.load(d / f"y_{split}.npy"))
        if len(stems) != n:
            raise RuntimeError(
                f"cannot recover cover ids for split '{split}': the feature cache has {n} "
                f"rows but {src} lists {len(stems)} images. Re-run extract_dctr.py so the "
                f"ids are written alongside the features.")
        out[split] = np.array(stems)
    return out


# --------------------------------------------------------------------------- #
# Separability
# --------------------------------------------------------------------------- #
def fisher_J(X, y):
    """Multiclass Fisher ratio J = tr(Sw^-1 Sb) on a feature matrix."""
    X = np.asarray(X, np.float64); mu = X.mean(0); d = X.shape[1]
    Sw = np.zeros((d, d)); Sb = np.zeros((d, d))
    for c in np.unique(y):
        Xc = X[y == c]; pi = len(Xc) / len(X); mc = Xc.mean(0)
        Sw += pi * np.cov(Xc, rowvar=False, bias=True)
        diff = (mc - mu).reshape(-1, 1); Sb += pi * (diff @ diff.T)
    return float(np.trace(np.linalg.pinv(Sw) @ Sb))


# --------------------------------------------------------------------------- #
# Grouped bootstrap
# --------------------------------------------------------------------------- #
def source_ids(loader):
    """Per-item source (cover) id, in loader order, for a grouped bootstrap.
    A cover and its stego variants share the same file stem, so grouping on the stem
    keeps correlated items together. Handles the image loaders (dataset.samples, one
    item per image) and the GBRAS block loader (dataset.index, 4 blocks per image)."""
    ds = loader.dataset
    if hasattr(ds, "samples"):
        return np.array([Path(p).stem for p, _ in ds.samples])
    idx = ds.index                                   # BlockDataset: 4 blocks/image
    return np.array([Path(idx[i][0]).stem for i in range(0, len(idx), 4)])


def group_boot(fn, groups, *, mask=None, n=2000, seed=0, digits=4, point=None):
    """95% CI of fn(row_indices), resampling COVERS with replacement.

    `fn` receives an index array into the full row set and returns a scalar; use it for
    any statistic (AUC, accuracy, a paired difference of two accuracies). `mask` restricts
    the resampling to a subset of rows (e.g. cover + one scheme). Resamples that make the
    statistic undefined (a single class present) are skipped.

    `point`, if given, is the full-sample estimate. The CI is returned unchanged, but a
    warning is emitted when it does not contain the point: a percentile interval that
    excludes its own estimate signals a skewed/biased statistic or a mask/fn mismatch.
    """
    groups = np.asarray(groups).ravel()
    rows = np.arange(len(groups)) if mask is None else np.where(mask)[0]
    g = groups[rows]
    by = {c: rows[g == c] for c in np.unique(g)}
    covers = np.array(list(by.keys()))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        samp = rng.choice(covers, len(covers), replace=True)
        idx = np.concatenate([by[c] for c in samp])
        try:
            vals.append(fn(idx))
        except ValueError:
            continue
    if not vals:
        raise RuntimeError("grouped bootstrap produced no valid resample")
    ci = [round(float(np.percentile(vals, 2.5)), digits),
          round(float(np.percentile(vals, 97.5)), digits)]
    if point is not None and not (ci[0] <= round(float(point), digits) <= ci[1]):
        warnings.warn(f"grouped bootstrap CI {ci} does not contain the point estimate "
                      f"{float(point):.{digits}f}", stacklevel=2)
    return ci


def boot_ci(y, scores, n=2000, seed=0, groups=None):
    """95% bootstrap CI of the ROC AUC, resampling COVERS when `groups` is given.

    Without `groups` it falls back to resampling items, which is only correct when the
    rows really are independent -- not the case for cover/stego pairs from one cover.
    """
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).ravel(); s = np.asarray(scores).ravel()
    auc = lambda idx: roc_auc_score(y[idx], s[idx])
    if groups is not None:
        return group_boot(auc, groups, n=n, seed=seed)
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        v.append(auc(idx))
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
