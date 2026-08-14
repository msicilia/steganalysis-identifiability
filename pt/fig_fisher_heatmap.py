"""Pairwise class-separability heatmap (Fisher ratio J).

Usage:
  python fig_fisher_heatmap.py --domain jpeg      # one panel: DCTR PCA-50 (from jpeg_kfold_cv)
  python fig_fisher_heatmap.py --domain spatial   # two panels: model-free SRM vs GBRAS learned

The spatial figure is deliberately two-panel: the same six classes are scored in a
model-free representation (SRM residual statistics) and in GBRAS-Net's learned features.
The adaptive block stays dark in the model-free space but separates in the learned space,
which is the paper's representation-dependence result made visual on a shared colour scale.
"""
import json, sys, argparse
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
# Embed text as scalable TrueType (Type 42), NOT matplotlib's default Type 3 bitmap
# fonts (which render fuzzy and clash with the paper's serif body text). Use a serif
# family and Computer-Modern-style math so the figures match the manuscript.
matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "mathtext.fontset": "cm",
})
import matplotlib.pyplot as plt
sys.path.append(str(Path(__file__).resolve().parent.parent))   # code/ (config)
import config as C

JPEG_NAMES = ["COVER", "JUNIWARD", "UERD", "nsF5", "UED", "JMIPOD"]
SPATIAL_NAMES = ["COVER", "HILL", "HUGO", "MiPOD", "S-UNIWARD", "WOW"]

# Each panel: (result file, extractor -> {"A|B": J}, class order, title).
CFG = {
    "jpeg": dict(out="fig_fisher_heatmap_jpeg", panels=[
        ("jpeg_kfold_a040.json", lambda J: _from_dict(J["fisher_pairs_pca50"]),
         JPEG_NAMES, "DCTR (PCA-50)"),
    ]),
    "spatial": dict(out="fig_fisher_heatmap", panels=[
        ("separability_srm_test_a040.json",
         lambda J: _from_matrix(J["pairwise_fisher"], J["classes"]),
         SPATIAL_NAMES, "Model-free SRM statistics"),
        ("gbras_ensemble_a040.json", lambda J: _from_dict(J["fisher_pairs"]),
         SPATIAL_NAMES, "GBRAS-Net learned features"),
    ]),
}


def _from_dict(d):
    """{'A|B': v}  or  {'A|B': [v, [lo, hi]]}  ->  {'A|B': v}."""
    return {k: (v[0] if isinstance(v, (list, tuple)) else v) for k, v in d.items()}


def _from_matrix(mat, classes):
    """n x n Fisher matrix (None on the diagonal) -> {'A|B': v} upper triangle."""
    out = {}
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            if mat[i][j] is not None:
                out[f"{classes[i]}|{classes[j]}"] = mat[i][j]
    return out


def _matrix(pairs, names):
    n = len(names); idx = {nm: i for i, nm in enumerate(names)}
    M = np.full((n, n), np.nan)
    for k, v in pairs.items():
        a, b = k.split("|"); M[idx[a], idx[b]] = v; M[idx[b], idx[a]] = v
    return M


def _draw(ax, M, names, title, vmax, ylabels=True):
    n = len(names)
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=vmax)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names if ylabels else [""] * n, fontsize=8)
    for i in range(n):
        for j in range(n):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] < 0.6 * vmax else "black")
    ax.set_title(title, fontsize=9)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=("jpeg", "spatial"), default="jpeg")
    args = ap.parse_args()
    cfg = CFG[args.domain]

    mats = []
    for src, extract, names, title in cfg["panels"]:
        J = json.load(open(C.RESULTS_DIR / src))
        mats.append((_matrix(extract(J), names), names, title))
    vmax = max(float(np.nanmax(M)) for M, _, _ in mats)   # shared scale across panels

    k = len(mats)
    fig, axes = plt.subplots(1, k, figsize=(5.4 * k, 4.6))
    axes = np.atleast_1d(axes)
    im = None
    for p, (ax, (M, names, title)) in enumerate(zip(axes, mats)):
        im = _draw(ax, M, names, title if k > 1 else None, vmax, ylabels=(p == 0))
    cb = fig.colorbar(im, ax=list(axes), fraction=0.046, pad=0.04)
    cb.set_label("Fisher ratio $J$ (higher = more separable)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    out = str(C.RESULTS_DIR / cfg["out"])
    for ext in ("pdf", "png"):
        plt.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
    print("saved:", out, "| Jmax=", round(vmax, 3))


if __name__ == "__main__":
    main()
