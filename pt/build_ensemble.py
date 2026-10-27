"""Modular detection ensemble of the lightweight CNN: a logistic-regression meta-model over the
five cover-vs-one specialist posteriors, compared with the cover-vs-all detector on TEST.

Specialists are trained on TRAIN with checkpoints selected on the FUSION "select" half; the
meta-model is fitted on their posteriors over the FUSION "meta" half. Reports the
specialists-only meta-model, the nested variant that also takes the generalist posterior, a
mean-fusion baseline and the generalist alone.

Usage:  python build_ensemble.py --alpha 0.4 [--device cuda]
"""
import sys, argparse, json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score

import config as C
from data import make_loader
from models import BinaryCNN, SpecialistCNN
from engine import set_seeds
from util import group_boot, source_ids, fit_logreg, convergence_info

# Stego algorithms = classes of the active domain minus COVER (spatial or JPEG).
ALGOS = [c for c in C.CLASS_FOLDERS if c != C.COVER_NAME]


def _load_specialist(algo, alpha, device):
    m = BinaryCNN(in_ch=C.CHANNELS, use_srm=True, dropout=0.0)
    m.load_state_dict(torch.load(C.MODEL_DIR / f"pt_spec_{algo}_{C.alpha_tag(alpha)}_s{C.SEED}.pt",
                                 map_location=device))
    return m.eval().to(device)


def _load_generalist(alpha, device):
    m = BinaryCNN(in_ch=C.CHANNELS, use_srm=True, dropout=0.0)
    m.load_state_dict(torch.load(C.MODEL_DIR / f"pt_binary_{C.alpha_tag(alpha)}_s{C.SEED}.pt",
                                 map_location=device))
    return m.eval().to(device)


def _monolith_preds(alpha, loader, device):
    """Multiclass predictions (argmax) of the monolithic multiclass CNN."""
    m = SpecialistCNN(num_classes=len(C.CLASS_FOLDERS), in_ch=C.CHANNELS,
                      use_srm=True, dropout=0.0, img_size=C.IMG_SIZE[0])
    m.load_state_dict(torch.load(C.MODEL_DIR / f"pt_family_{C.alpha_tag(alpha)}_s{C.SEED}.pt",
                                 map_location=device))
    m.eval().to(device)
    preds, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(F.softmax(m(x.to(device)), 1).argmax(1).cpu().numpy())
            ys.append(np.asarray(y))
    return np.concatenate(preds), np.concatenate(ys)


def _mc_metrics(y, pred):
    return {"acc": float((pred == y).mean()),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "cm": confusion_matrix(y, pred, labels=np.arange(len(C.CLASS_FOLDERS))).tolist()}


def _p_stego(model, loader, device):
    """P(stego) per image (sigmoid of the logit) + binary labels from the loader."""
    ps, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
            ys.append(np.asarray(y))
    return np.concatenate(ps), np.concatenate(ys)


def _metrics(y, score, thr=0.5):
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"auc": float(roc_auc_score(y, score)),
            "P_D": float(tp / (tp + fn)), "P_FA": float(fp / (fp + tn))}


META_COLS = ["GEN"] + ALGOS


def _meta_features(alpha, split, device, workers):
    """Meta-features (N, 1+len(ALGOS)) = [generalist, specialists] + MULTICLASS
    label (0..K-1) + the cover id of each row. Loader with real class labels."""
    d = C.dirs_for(alpha)[split]
    loader = make_loader(d, C.CLASS_FOLDERS, C.BATCH_SIZE,
                         shuffle=False, num_workers=workers,   # multiclass labels
                         subset=("meta" if split == "fusion" else None))   # [V3]
    pg, ymc = _p_stego(_load_generalist(alpha, device), loader, device)
    cols = [pg]
    for algo in ALGOS:
        p, _ = _p_stego(_load_specialist(algo, alpha, device), loader, device)
        cols.append(p)
    # cover ids travel with the features: the ΔAUC interval must resample covers, not
    # images, because a cover and its stego variants are the same content.
    return np.stack(cols, axis=1), ymc, source_ids(loader)


def _delta_vs_gen(y, s, gen, groups):
    """[ΔAUC point, grouped-bootstrap 95% CI] of AUC(s) - AUC(gen) on the same rows."""
    point = round(float(roc_auc_score(y, s) - roc_auc_score(y, gen)), 4)
    return [point, group_boot(lambda i: roc_auc_score(y[i], s[i])
                                        - roc_auc_score(y[i], gen[i]), groups)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=C.FAMILY_ID_PAYLOAD)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    set_seeds(C.SEED)
    dev = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print(f"== Modular ensemble | alpha={args.alpha} | dev={dev}")

    cover_idx = C.CLASS_FOLDERS.index(C.COVER_NAME)
    # leakage-free meta-features: FUSION trains the meta-model, TEST evaluates. y multiclass.
    Xf, yf_mc, _ = _meta_features(args.alpha, "fusion", dev, args.workers)
    Xt, yt_mc, gt = _meta_features(args.alpha, "test",   dev, args.workers)
    yf_bin = (yf_mc != cover_idx).astype(int)
    yt_bin = (yt_mc != cover_idx).astype(int)
    print(f"   fusion={Xf.shape} test={Xt.shape}  (cols={META_COLS})")

    # ===================== A) binary cover/stego DETECTION =====================
    gen_s = Xt[:, 0]                                  # col 0 = generalist
    # PRIMARY: specialists only (cols 1:), so the generalist is not among its inputs.
    meta_sp = fit_logreg(Xf[:, 1:], yf_bin, tag="meta-binary-specialists")
    sp_s = meta_sp.predict_proba(Xt[:, 1:])[:, 1]
    # SECONDARY (nested): [generalist, specialists]; nests its baseline, ΔAUC >= ~0 by construction.
    meta_nest = fit_logreg(Xf, yf_bin, tag="meta-binary-nested")
    nest_s = meta_nest.predict_proba(Xt)[:, 1]
    r_gen_bin = _metrics(yt_bin, gen_s)
    r_sp_bin = _metrics(yt_bin, sp_s)
    r_nest_bin = _metrics(yt_bin, nest_s)
    r_mean = _metrics(yt_bin, Xt[:, 1:].mean(1))      # non-learned fusion (specs)
    # ΔAUC intervals over the TEST scores, resampling COVERS.
    delta_sp = _delta_vs_gen(yt_bin, sp_s, gen_s, gt)
    delta_nest = _delta_vs_gen(yt_bin, nest_s, gen_s, gt)

    # ===================== B) MULTICLASS: modular vs monolithic ================
    # Modular: multiclass meta-model over [gen, specialists] (leakage-free fusion). The
    # baseline here is the multiclass CNN, not the generalist, so including GEN does not nest it.
    meta_mc = fit_logreg(Xf, yf_mc, tag="meta-multiclass")
    r_ens_mc = _mc_metrics(yt_mc, meta_mc.predict(Xt))
    # Monolithic: the multiclass CNN over the SAME test.
    d = C.dirs_for(args.alpha)["test"]
    te = make_loader(d, C.CLASS_FOLDERS, C.BATCH_SIZE, shuffle=False, num_workers=args.workers)
    mono_pred, y_mono = _monolith_preds(args.alpha, te, dev)
    r_mono_mc = _mc_metrics(y_mono, mono_pred)

    out = {"alpha": args.alpha, "seed": C.SEED, "classes": C.CLASS_FOLDERS, "meta_cols": META_COLS,
           "detection": {
               "generalist": r_gen_bin,
               "ensemble_specialists_only": r_sp_bin,
               "ensemble_nested_gen_plus_specialists": r_nest_bin,
               "fusion_mean": r_mean,
               "delta_specialists_only_minus_gen": delta_sp,
               "delta_nested_ens_minus_gen": delta_nest,
               "delta_note": ("[point, CI95] with the CI from a grouped bootstrap over test covers. "
                              "The primary comparison is specialists-only (columns 1:). The nested "
                              "meta-model has the generalist among its features, so its ΔAUC is "
                              ">= ~0 by construction and is not evidence for modularity.")},
           "multiclass": {"n_classes": len(C.CLASS_FOLDERS),
                          "monolithic_cnn": r_mono_mc, "modular_ensemble": r_ens_mc},
           "provenance": convergence_info()}
    res_dir = C.RESULTS_DIR
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / f"ensemble_{C.alpha_tag(args.alpha)}.json").write_text(json.dumps(out, indent=2))

    rb = lambda n, r: f"| {n} | {r['auc']:.4f} | {r['P_D']:.4f} | {r['P_FA']:.4f} |"
    print(f"\n### A) Cover/stego detection @{args.alpha} (TEST)")
    print("| model | AUC | P_D | P_FA |\n|---|---|---|---|")
    print(rb("Generalist (monolithic)", r_gen_bin))
    print(rb("Ensemble, specialists only (LogReg)", r_sp_bin))
    print(rb("Ensemble, nested GEN+specialists (LogReg)", r_nest_bin))
    print(rb("Mean fusion (non-learned)", r_mean))
    print(f"  ΔAUC specialists-only − gen = {delta_sp[0]:+.4f}  CI95={delta_sp[1]}  (primary)")
    print(f"  ΔAUC nested − gen           = {delta_nest[0]:+.4f}  CI95={delta_nest[1]}  (nested, >=0 by construction)")

    print(f"\n### B) {len(C.CLASS_FOLDERS)}-classes: MODULAR vs MONOLITHIC @{args.alpha} (TEST)")
    print("| model | accuracy | macro-F1 |\n|---|---|---|")
    print(f"| Monolithic multiclass CNN (train_family) | {r_mono_mc['acc']:.4f} | {r_mono_mc['macro_f1']:.4f} |")
    print(f"| Modular ensemble (meta-LogReg)           | {r_ens_mc['acc']:.4f} | {r_ens_mc['macro_f1']:.4f} |")
    dacc = r_ens_mc["acc"] - r_mono_mc["acc"]
    print(f"\nΔacc modular−monolithic = {dacc:+.4f}")
    print("Modular confusion matrix (rows=truth):", r_ens_mc["cm"])
    print("saved:", res_dir / f"ensemble_{C.alpha_tag(args.alpha)}.json")


if __name__ == "__main__":
    main()
