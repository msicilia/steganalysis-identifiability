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


def boot_ci(y, scores, n=2000, seed=0):
    """95% bootstrap CI of the ROC AUC, resampling test items with replacement."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).ravel(); s = np.asarray(scores).ravel()
    rng = np.random.default_rng(seed); N = len(y); v = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        if len(np.unique(y[idx])) < 2:
            continue
        v.append(roc_auc_score(y[idx], s[idx]))
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
