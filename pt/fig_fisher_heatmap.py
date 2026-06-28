"""Pairwise class-separability heatmap (Fisher ratio J).

Usage:
  python fig_fisher_heatmap.py --domain jpeg      # from dctr_revision (PCA-50 of DCTR)
  python fig_fisher_heatmap.py --domain spatial   # from gbras_ensemble (GBRAS features)
"""
import json, sys, argparse
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.append(str(Path(__file__).resolve().parent.parent))   # code/ (config)
import config as C

CFG = {
    "jpeg":    dict(src="dctr_revision_a040.json",
                    names=["COVER", "JUNIWARD", "UERD", "nsF5", "UED", "JMIPOD"],
                    out="fig_fisher_heatmap_jpeg"),
    "spatial": dict(src="gbras_ensemble_a040.json",
                    names=["COVER", "HILL", "HUGO", "MiPOD", "S-UNIWARD", "WOW"],
                    out="fig_fisher_heatmap"),
}


def get_pairs(domain, J):
    return J["separability_stable"]["fisher_pairs_pca50"] if domain == "jpeg" else J["fisher_pairs"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=("jpeg", "spatial"), default="jpeg")
    args = ap.parse_args()
    cfg = CFG[args.domain]
    J = json.load(open(C.RESULTS_DIR / cfg["src"]))
    pairs = get_pairs(args.domain, J)
    names = cfg["names"]; n = len(names); idx = {nm: i for i, nm in enumerate(names)}
    M = np.full((n, n), np.nan)
    for k, v in pairs.items():
        a, b = k.split("|"); M[idx[a], idx[b]] = v; M[idx[b], idx[a]] = v

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=np.nanmax(M))
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8); ax.set_yticklabels(names, fontsize=8)
    for i in range(n):
        for j in range(n):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] < 0.6 * np.nanmax(M) else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Fisher ratio $J$ (higher = more separable)", fontsize=8); cb.ax.tick_params(labelsize=7)
    plt.tight_layout()
    out = str(C.RESULTS_DIR / cfg["out"])
    for ext in ("pdf", "png"):
        plt.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
    print("saved:", out, "| Jmax=", round(float(np.nanmax(M)), 3))


if __name__ == "__main__":
    main()
