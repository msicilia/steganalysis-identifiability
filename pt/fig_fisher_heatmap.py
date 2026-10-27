"""Pairwise Fisher J heatmap on a logarithmic colour scale.

Usage:
  python fig_fisher_heatmap.py --domain jpeg --jpeg-k 200   # PCA-k table of fisher_pca_sweep.py
  python fig_fisher_heatmap.py --domain spatial             # GBRAS detector features
  python fig_fisher_heatmap.py --domain spatial_all         # adds SRM statistics and six-class features
"""
import json, sys, argparse
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
# Embed text as scalable TrueType (Type 42), NOT matplotlib's default Type 3 bitmap
# fonts. Serif family and Computer-Modern-style math to match the article.
matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "mathtext.fontset": "cm",
})
import matplotlib.pyplot as plt
sys.path.append(str(Path(__file__).resolve().parent.parent))   # code/ (config)
import config as C

# Class order of the JPEG panel: the three adaptive schemes that form the least separable
# group first, then UED, then nsF5, so the three levels of J appear as blocks. The keys are
# those of the result files; LABELS gives the display spelling.
JPEG_NAMES = ["COVER", "JUNIWARD", "UERD", "JMIPOD", "UED", "nsF5"]
SPATIAL_NAMES = ["COVER", "HILL", "HUGO", "MiPOD", "S-UNIWARD", "WOW"]
LABELS = {"JUNIWARD": "J-UNIWARD", "JMIPOD": "J-MiPOD"}
FS = 13                      # font size of labels and cell values (the figure is set at 0.7 of the text width)

# Each panel: (result file, extractor -> {"A|B": J}, class order, title, scale group).
# Panels in the same scale group share one colour scale and one colour bar.
# All panels use the family-identification payload (fixed-payload multiclass analysis).
_AT = C.alpha_tag(C.FAMILY_ID_PAYLOAD)
CFG = {
    # JPEG: pairwise Fisher J on a PCA of all DCTR rows, read from fisher_pca_sweep_*.json at
    # --jpeg-k components (default 200); --jpeg-k 0 reads the PCA-50 table in jpeg_kfold_*.json.
    "jpeg": dict(out="fig_fisher_heatmap_jpeg", panels=[
        (f"fisher_pca_sweep_{_AT}.json", None, JPEG_NAMES, "DCTR (PCA, all rows)", "dctr"),
    ]),
    "jpeg_pca50": dict(out="fig_fisher_heatmap_jpeg_pca50", panels=[
        (f"jpeg_kfold_{_AT}.json", lambda J: _from_dict(J["fisher_pairs_pca50"]),
         JPEG_NAMES, "DCTR (PCA-50, all rows; not cross-validated)", "dctr"),
    ]),
    # The SRM-statistics and the 6-class panels are only drawn by --domain spatial_all.
    "spatial": dict(out="fig_fisher_heatmap", panels=[
        (f"gbras_ensemble_{_AT}.json", lambda J: _from_dict(J["fisher_pairs_detect"]),
         SPATIAL_NAMES, "GBRAS-Net detector features (no algorithm labels)", "shared"),
    ]),
    "spatial_all": dict(out="fig_fisher_heatmap_all", panels=[
        (f"separability_srm_test_{_AT}.json",
         lambda J: _from_matrix(J["pairwise_fisher"], J["classes"]),
         SPATIAL_NAMES, "Model-free SRM statistics", "shared"),
        (f"gbras_ensemble_{_AT}.json", lambda J: _from_dict(J["fisher_pairs_detect"]),
         SPATIAL_NAMES, "GBRAS-Net detector features (no algorithm labels)", "shared"),
        (f"gbras_ensemble_{_AT}.json", lambda J: _from_dict(J["fisher_pairs"]),
         SPATIAL_NAMES, "GBRAS-Net 6-class features (trained on labels)", "circular"),
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


def _draw(ax, M, names, title, vmin, vmax, ylabels=True):
    """One heatmap panel on a logarithmic colour scale (J spans two orders of magnitude)."""
    from matplotlib.colors import LogNorm, ListedColormap
    n = len(names)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    # Start the ramp above white so the smallest J is still a tint, and grey the empty
    # diagonal so it cannot be read as a low value.
    # viridis (dark purple = low, yellow = high) on the log scale; values are set in white on
    # the dark end and in black on the light end.
    cmap = ListedColormap(plt.get_cmap("viridis")(np.linspace(0.0, 1.0, 256)))
    cmap.set_bad("#e6e6e6")
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=norm)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    labels = [LABELS.get(nm, nm) for nm in names]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=FS)
    ax.set_yticklabels(labels if ylabels else [""] * n, fontsize=FS)
    for i in range(n):
        for j in range(n):
            if not np.isnan(M[i, j]):
                dark = norm(M[i, j]) < 0.6          # white ink on the dark (low) cells
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=FS,
                        color="white" if dark else "black")
    ax.set_title(title, fontsize=FS)
    return im


def main():
    ap = argparse.ArgumentParser()
    # spatial_all adds the SRM-statistics and 6-class panels.
    ap.add_argument("--domain", choices=("jpeg", "jpeg_pca50", "spatial", "spatial_all"),
                    default="jpeg")
    ap.add_argument("--jpeg-k", type=int, default=200,
                    help="PCA components for the JPEG panel (a key of fisher_pca_sweep by_dim)")
    args = ap.parse_args()
    cfg = CFG[args.domain]
    if args.domain == "jpeg" and args.jpeg_k == 0:
        cfg = CFG["jpeg_pca50"]; cfg = dict(cfg, out="fig_fisher_heatmap_jpeg")

    mats = []
    for src, extract, names, title, group in cfg["panels"]:
        J = json.load(open(C.RESULTS_DIR / src))
        if extract is None:                       # fisher_pca_sweep file: pick the dimension
            k = str(args.jpeg_k)
            if k not in J["by_dim"]:
                sys.exit(f"{src} has no k={k}; available: {list(J['by_dim'])}")
            print(f"{src}: k={k} components, {J['protocol']}")
            extract = (lambda J, k=k: _from_dict(J["by_dim"][k]["pairs"]))
            title = f"DCTR (PCA-{k}, all rows; not cross-validated)"
        if "fisher_protocol" in J:
            print(f"{src}: {J['fisher_protocol']}")
        mats.append((_matrix(extract(J), names), names, title, group))
    # one colour scale per scale group (log scale: both ends are data-driven)
    vmax, vmin = {}, {}
    for M, _, _, group in mats:
        vmax[group] = max(vmax.get(group, 0.0), float(np.nanmax(M)))
        vmin[group] = min(vmin.get(group, np.inf), float(np.nanmin(M)))

    k = len(mats)
    fig, axes = plt.subplots(1, k, figsize=(5.4 * k, 4.6))
    axes = np.atleast_1d(axes)
    ims = {}
    for p, (ax, (M, names, title, group)) in enumerate(zip(axes, mats)):
        im = _draw(ax, M, names, title if k > 1 else None, vmin[group], vmax[group],
                   ylabels=(p == 0))
        ims.setdefault(group, (im, []))[1].append(ax)
    for group, (im, group_axes) in ims.items():
        cb = fig.colorbar(im, ax=group_axes, fraction=0.046, pad=0.04)
        cb.set_label("Fisher ratio $J$ (log scale)", fontsize=FS)
        ticks = [t for t in (0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5)
                 if vmin[group] <= t <= vmax[group]]
        cb.set_ticks(ticks); cb.set_ticklabels([f"{t:g}" for t in ticks])
        cb.ax.minorticks_off()
        cb.ax.tick_params(labelsize=FS)
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = str(C.RESULTS_DIR / cfg["out"])
    for ext in ("pdf", "png"):
        plt.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
    print("saved:", out, "| Jmax per scale group:", {g: round(v, 3) for g, v in vmax.items()})


if __name__ == "__main__":
    main()
