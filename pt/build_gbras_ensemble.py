"""
Spatial GBRAS ensemble + separability over GBRAS features.
The spatial ensemble is evaluated on detection: GBRAS specialists cover-vs-each-algo
+ meta (leakage-free) vs the monolithic GBRAS detector cover-vs-all. In addition,
separability (Fisher) over the penultimate features of the 6-class GBRAS.

Requires:
  pt_gbras_<algo>_a040_s42.pt  (cover-vs-algo specialists)
  pt_gbraswh_detect_a040_s42.pt (cover-vs-all generalist)
  pt_gbraswh_family_a040_s42.pt (6 classes, for separability features)

Usage:  python build_gbras_ensemble.py --alpha 0.4
"""
import sys, argparse, json, itertools
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
import config as C
from data import make_block_loader_multi
from gbras_io import load_gbras, per_image_probs, per_image_feats
from engine import DEVICE
from util import fisher_J

ALGOS = [c for c in C.CLASS_FOLDERS if c != C.COVER_NAME]


def _bl(split, alpha, w):
    cov = C.CLASS_FOLDERS.index(C.COVER_NAME)
    return make_block_loader_multi(C.dirs_for(alpha)[split], C.CLASS_FOLDERS, 64,
                                   binary_cover_idx=cov, shuffle=False, num_workers=w)


def _bl_multi(split, alpha, w):  # multiclass labels (for separability)
    return make_block_loader_multi(C.dirs_for(alpha)[split], C.CLASS_FOLDERS, 64,
                                   shuffle=False, num_workers=w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.4); ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args(); a = args.alpha; tag = C.alpha_tag(a)
    print(f"== GBRAS ensemble (detection) + separability | alpha={a} | dev={DEVICE}")

    def metrics(y, score):
        pred = (score >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        return {"auc": float(roc_auc_score(y, score)), "P_D": float(tp/(tp+fn)), "P_FA": float(fp/(fp+tn))}

    # specialist posteriors (cover-vs-each-algo) leakage-free: FUSION trains meta, TEST evaluates
    def spec_posteriors(split):
        cols, yref = [], None
        for alg in ALGOS:
            m = load_gbras(C.MODEL_DIR / f"pt_gbras_{alg}_{tag}_s{C.SEED}.pt", 2, DEVICE)
            P, y = per_image_probs(m, _bl(split, a, args.workers), DEVICE)
            cols.append(P[:, 1]); yref = y
        return np.stack(cols, 1), yref
    Xf, yf = spec_posteriors("fusion"); Xt, yt = spec_posteriors("test")

    meta = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xf, yf)
    r_ens = metrics(yt, meta.predict_proba(Xt)[:, 1])

    gen = load_gbras(C.MODEL_DIR / f"pt_gbraswh_detect_{tag}_s{C.SEED}.pt", 2, DEVICE)
    Pg, yg = per_image_probs(gen, _bl("test", a, args.workers), DEVICE)
    r_gen = metrics(yg, Pg[:, 1])

    # separability over the 6-class GBRAS features (test)
    fam = load_gbras(C.MODEL_DIR / f"pt_gbraswh_family_{tag}_s{C.SEED}.pt", len(C.CLASS_FOLDERS), DEVICE)
    Ff, yfam = per_image_feats(fam, _bl_multi("test", a, args.workers), DEVICE)
    J = fisher_J(Ff, yfam)
    pair = {f"{C.CLASS_FOLDERS[i]}|{C.CLASS_FOLDERS[j]}": round(fisher_J(Ff[(yfam==i)|(yfam==j)], yfam[(yfam==i)|(yfam==j)]), 3)
            for i, j in itertools.combinations(range(len(C.CLASS_FOLDERS)), 2)}

    out = {"alpha": a, "generalist_detect": r_gen, "ensemble_detect": r_ens,
           "delta_auc": round(r_ens["auc"] - r_gen["auc"], 4),
           "fisher_J_gbras_feats": round(J, 3), "fisher_pairs": pair}
    res = C.RESULTS_DIR; (res/f"gbras_ensemble_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"\n### GBRAS ensemble (detection, TEST)")
    print(f"  GBRAS generalist  : AUC={r_gen['auc']:.4f}")
    print(f"  Modular ensemble  : AUC={r_ens['auc']:.4f}")
    print(f"  ΔAUC = {out['delta_auc']:+.4f}")
    print(f"### Separability (GBRAS 6cl features): Fisher J={J:.3f}")
    print("  pairs:", pair)
    print("saved:", res/f"gbras_ensemble_{tag}.json")


if __name__ == "__main__":
    main()
