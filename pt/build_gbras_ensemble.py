"""GBRAS-Net modular ensemble (detection and six-class identification) against the monolithic
GBRAS-Net networks, and pairwise Fisher J in the penultimate features of the cover-vs-all
detector and of the six-class network.

Requires pt_gbras_<algo>_<atag>_s<seed>.pt, pt_gbraswh_detect_<atag>_s<seed>.pt and
pt_gbraswh_family_<atag>_s<seed>.pt. The meta-model takes the specialist posteriors only.

Usage:  python build_gbras_ensemble.py --alpha 0.4
"""
import sys, argparse, json, itertools
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score
import config as C
from data import make_block_loader_multi
from gbras_io import load_gbras, per_image_probs, per_image_feats
from engine import DEVICE
from util import fisher_J, fit_logreg, convergence_info, group_boot, source_ids

ALGOS = [c for c in C.CLASS_FOLDERS if c != C.COVER_NAME]


# [V3] FUSION rows always come from its "meta" half: the "select" half chose the checkpoints.
def _sub(split):
    return "meta" if split == "fusion" else None


def _bl(split, alpha, w):
    cov = C.CLASS_FOLDERS.index(C.COVER_NAME)
    return make_block_loader_multi(C.dirs_for(alpha)[split], C.CLASS_FOLDERS, 64,
                                   binary_cover_idx=cov, shuffle=False, num_workers=w,
                                   subset=_sub(split))


def _bl_multi(split, alpha, w):  # multiclass labels (for separability)
    return make_block_loader_multi(C.dirs_for(alpha)[split], C.CLASS_FOLDERS, 64,
                                   shuffle=False, num_workers=w, subset=_sub(split))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=C.FAMILY_ID_PAYLOAD); ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args(); a = args.alpha; tag = C.alpha_tag(a)
    print(f"== GBRAS ensemble (detection) + separability | alpha={a} | dev={DEVICE}")

    def metrics(y, score):
        pred = (score >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        return {"auc": float(roc_auc_score(y, score)), "P_D": float(tp/(tp+fn)), "P_FA": float(fp/(fp+tn))}

    cov = C.CLASS_FOLDERS.index(C.COVER_NAME)

    # specialist posteriors (cover-vs-each-algo) leakage-free: FUSION trains meta, TEST evaluates.
    # Read through the multiclass loader: a specialist's output does not depend on the labels,
    # and the 6-class labels serve both the detection meta (cover vs any) and the
    # identification meta. Row order is identical to the binary loader (same sample list).
    def spec_posteriors(split):
        cols, yref = [], None
        for alg in ALGOS:
            m = load_gbras(C.MODEL_DIR / f"pt_gbras_{alg}_{tag}_s{C.SEED}.pt", 2, DEVICE)
            P, y = per_image_probs(m, _bl_multi(split, a, args.workers), DEVICE)
            cols.append(P[:, 1]); yref = y
        return np.stack(cols, 1), yref
    Xf, yf_mc = spec_posteriors("fusion"); Xt, yt_mc = spec_posteriors("test")
    yf, yt = (yf_mc != cov).astype(int), (yt_mc != cov).astype(int)

    meta = fit_logreg(Xf, yf, tag="meta")
    ens_s = meta.predict_proba(Xt)[:, 1]
    r_ens = metrics(yt, ens_s)

    gen = load_gbras(C.MODEL_DIR / f"pt_gbraswh_detect_{tag}_s{C.SEED}.pt", 2, DEVICE)
    te_loader = _bl("test", a, args.workers)
    Pg, yg = per_image_probs(gen, te_loader, DEVICE)
    r_gen = metrics(yg, Pg[:, 1])
    # ΔAUC interval, resampling COVERS (a cover and its stego variants are one unit).
    # Unlike build_ensemble.py, this meta-model does NOT carry the generalist among its
    # features, so the margin here is not structurally guaranteed.
    gte = source_ids(te_loader)
    # the specialists' and the generalist's test loaders are built identically; the paired
    # difference is only valid if the rows really line up
    assert np.array_equal(yt, yg) and len(gte) == len(yt), "test rows misaligned"
    delta_ci = group_boot(lambda i: roc_auc_score(yt[i], ens_s[i])
                                    - roc_auc_score(yg[i], Pg[i, 1]), gte)

    # ---- identification: modular (meta over specialist posteriors) vs monolithic 6-class ----
    # Same design as the JPEG comparison: the specialists train on TRAIN, the multinomial
    # meta-learner on their FUSION posteriors, both arms are scored on the same TEST rows.
    K = len(C.CLASS_FOLDERS)
    meta_id = fit_logreg(Xf, yf_mc, tag="meta-identification")
    mod_pred = meta_id.predict(Xt)
    fam = load_gbras(C.MODEL_DIR / f"pt_gbraswh_family_{tag}_s{C.SEED}.pt", K, DEVICE)
    fam_loader = _bl_multi("test", a, args.workers)
    Pm, ym = per_image_probs(fam, fam_loader, DEVICE)
    mono_pred = Pm.argmax(1)
    assert np.array_equal(ym, yt_mc), "identification test rows misaligned"

    def id_metrics(pred):
        cm = confusion_matrix(yt_mc, pred, labels=range(K))
        return {"acc": float(accuracy_score(yt_mc, pred)),
                "macro_f1": float(f1_score(yt_mc, pred, average="macro")),
                "cm": cm.tolist(),
                "recall_per_class": {n: float(r) for n, r in
                                     zip(C.CLASS_FOLDERS, cm.diagonal() / np.maximum(cm.sum(1), 1))}}
    acc = lambda p: (lambda i: accuracy_score(yt_mc[i], p[i]))
    r_mono_id, r_mod_id = id_metrics(mono_pred), id_metrics(mod_pred)
    d_id = r_mod_id["acc"] - r_mono_id["acc"]
    identification = {
        "monolithic_gbras_family": r_mono_id, "modular_ensemble": r_mod_id,
        "delta_acc_modular_minus_mono": round(d_id, 4),
        "delta_acc_ci95": group_boot(lambda i: acc(mod_pred)(i) - acc(mono_pred)(i), gte,
                                     point=d_id),
        "mono_acc_ci95": group_boot(acc(mono_pred), gte, point=r_mono_id["acc"]),
        "modular_acc_ci95": group_boot(acc(mod_pred), gte, point=r_mod_id["acc"]),
        "protocol": ("modular = multinomial LogReg over the 5 GBRAS cover-vs-one specialist "
                     "posteriors, fitted on FUSION; monolithic = GBRAS 6-class (train_gbras_wh "
                     "--mode family); both on the same TEST images, CIs by cover")}

    def pairwise(F, yy):
        return {f"{C.CLASS_FOLDERS[i]}|{C.CLASS_FOLDERS[j]}":
                round(fisher_J(F[(yy == i) | (yy == j)], yy[(yy == i) | (yy == j)]), 3)
                for i, j in itertools.combinations(range(K), 2)}

    # separability over the 6-class GBRAS features (test) -- trained on these labels
    Ff, yfam = per_image_feats(fam, fam_loader, DEVICE)
    J = fisher_J(Ff, yfam)
    pair = pairwise(Ff, yfam)
    # the same measure over the cover-vs-all DETECTOR's features (test): never given the
    # algorithm labels, so this is the non-circular learned-representation comparison
    Fd, ydet = per_image_feats(gen, _bl_multi("test", a, args.workers), DEVICE)
    assert np.array_equal(ydet, yfam), "detector feature rows misaligned"
    J_det = fisher_J(Fd, ydet)
    pair_det = pairwise(Fd, ydet)

    out = {"alpha": a, "generalist_detect": r_gen, "ensemble_detect": r_ens,
           "delta_auc": round(r_ens["auc"] - r_gen["auc"], 4), "delta_auc_ci95": delta_ci,
           "identification": identification,
           "fisher_J_gbras_feats": round(J, 3), "fisher_pairs": pair,
           "fisher_J_gbras_detect_feats": round(J_det, 3), "fisher_pairs_detect": pair_det,
           "fisher_note": ("fisher_* (6-class network): features from a network TRAINED on "
                           "these six class labels, so a high J is close to guaranteed. "
                           "fisher_*_detect (cover-vs-all detector, 30-d bn11 like the 6-class "
                           "net): a learned representation that never saw algorithm labels, so "
                           "it is the non-circular comparison. J is not comparable across "
                           "feature spaces of different dimension or derivation; the held-out "
                           "accuracy is the evidence."),
           "meta_features": "specialist posteriors only (generalist not included; not nested)",
           "seed": C.SEED, "provenance": convergence_info()}
    res = C.RESULTS_DIR; res.mkdir(parents=True, exist_ok=True)
    (res/f"gbras_ensemble_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"\n### GBRAS ensemble (detection, TEST)")
    print(f"  GBRAS generalist  : AUC={r_gen['auc']:.4f}")
    print(f"  Modular ensemble  : AUC={r_ens['auc']:.4f}")
    print(f"  ΔAUC = {out['delta_auc']:+.4f}  CI95={delta_ci}")
    print(f"### GBRAS identification ({K} classes, TEST)")
    print(f"  monolithic 6-class : acc={r_mono_id['acc']:.4f} CI95={identification['mono_acc_ci95']}")
    print(f"  modular ensemble   : acc={r_mod_id['acc']:.4f} CI95={identification['modular_acc_ci95']}")
    print(f"  Δacc modular−mono  = {d_id:+.4f}  CI95={identification['delta_acc_ci95']}")
    print(f"### Separability (GBRAS 6cl features, trained on the labels): Fisher J={J:.3f}")
    print("  pairs:", pair)
    print(f"### Separability (GBRAS detector features, no algorithm labels): Fisher J={J_det:.3f}")
    print("  pairs:", pair_det)
    print("saved:", res/f"gbras_ensemble_{tag}.json")


if __name__ == "__main__":
    main()
