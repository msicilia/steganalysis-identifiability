"""Shared helpers for the analysis scripts."""
import sys
from pathlib import Path
import numpy as np
sys.path.append(str(Path(__file__).resolve().parent.parent))   # code/ (config)
import config as C


def load_dctr(alpha):
    """Load cached DCTR features/labels for a payload.
    Returns (X, y), each a dict over the 'train'/'fusion'/'test' splits."""
    d = C.DATASETS / f"{C.DATA_PREFIX}_{C.alpha_tag(alpha)}_DCTR"
    X = {s: np.load(d / f"X_{s}.npy") for s in ("train", "fusion", "test")}
    y = {s: np.load(d / f"y_{s}.npy") for s in ("train", "fusion", "test")}
    return X, y


def fisher_J(X, y):
    """Multiclass Fisher ratio J = tr(Sw^-1 Sb) on a feature matrix."""
    X = np.asarray(X, np.float64); mu = X.mean(0); d = X.shape[1]
    Sw = np.zeros((d, d)); Sb = np.zeros((d, d))
    for c in np.unique(y):
        Xc = X[y == c]; pi = len(Xc) / len(X); mc = Xc.mean(0)
        Sw += pi * np.cov(Xc, rowvar=False, bias=True)
        diff = (mc - mu).reshape(-1, 1); Sb += pi * (diff @ diff.T)
    return float(np.trace(np.linalg.pinv(Sw) @ Sb))


def source_ids(loader):
    """Per-item source (cover) id, in loader order, for a grouped bootstrap.
    A cover and its stego variants share the same file stem, so grouping on the stem
    keeps correlated items together. Handles the image loaders (dataset.samples, one
    item per image) and the GBRAS block loader (dataset.index, 4 blocks per image)."""
    from pathlib import Path
    ds = loader.dataset
    if hasattr(ds, "samples"):
        return np.array([Path(p).stem for p, _ in ds.samples])
    idx = ds.index                                   # BlockDataset: 4 blocks/image
    return np.array([Path(idx[i][0]).stem for i in range(0, len(idx), 4)])


def boot_ci(y, scores, n=2000, seed=0, groups=None):
    """95% bootstrap CI of the ROC AUC.
    If `groups` is given (one source/cover id per item), the bootstrap resamples GROUPS
    with replacement and keeps each group's items together -- the correct unit when a
    cover and its stego variants are correlated. A naive per-item bootstrap treats them
    as independent and understates the interval. Without `groups` it resamples items."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).ravel(); s = np.asarray(scores).ravel()
    rng = np.random.default_rng(seed); v = []
    if groups is not None:
        groups = np.asarray(groups).ravel()
        uniq = np.unique(groups)
        idx_by = {g: np.where(groups == g)[0] for g in uniq}
        for _ in range(n):
            gsel = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([idx_by[g] for g in gsel])
            if len(np.unique(y[idx])) < 2:
                continue
            v.append(roc_auc_score(y[idx], s[idx]))
    else:
        N = len(y)
        for _ in range(n):
            idx = rng.integers(0, N, N)
            if len(np.unique(y[idx])) < 2:
                continue
            v.append(roc_auc_score(y[idx], s[idx]))
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
