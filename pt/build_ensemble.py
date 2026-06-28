"""
Modular detection ensemble (leakage-free fusion).

Combines the posteriors of the 5 specialists (COVER vs each algorithm) through a
learned meta-model, and compares the cover-vs-stego detection of the ensemble against
the monolithic generalist detector (same architecture).

Leakage-free stacking:
  - specialists trained on TRAIN,
  - meta-features (their posteriors) computed on FUSION -> the meta-model is trained there,
  - final evaluation on TEST (disjoint by cover ID).

Meta-model: logistic regression (sklearn) over the 5 posteriors. A non-learned
mean-fusion baseline and the generalist alone are also reported, all over the
same cover-vs-any-stego test.

Usage:  python build_ensemble.py --alpha 0.4 [--device cuda]
"""
import sys, argparse, json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score

import config as C
from data import make_loader
from models import BinaryCNN, SpecialistCNN
from engine import set_seeds

# Stego algorithms = classes of the active domain minus COVER (spatial or JPEG).
ALGOS = [c for c in C.CLASS_FOLDERS if c != C.COVER_NAME]


def _load_specialist(algo, alpha, device):
    m = BinaryCNN(in_ch=C.CHANNELS, use_srm=True, dropout=0.0)
    m.load_state_dict(torch.load(C.MODEL_DIR / f"pt_spec_{algo}_{C.alpha_tag(alpha)}.pt",
                                 map_location=device))
    return m.eval().to(device)


def _load_generalist(alpha, device):
    m = BinaryCNN(in_ch=C.CHANNELS, use_srm=True, dropout=0.0)
    m.load_state_dict(torch.load(C.MODEL_DIR / f"pt_binary_{C.alpha_tag(alpha)}_s42.pt",
                                 map_location=device))
    return m.eval().to(device)


def _monolith_preds(alpha, loader, device):
    """4-class predictions (argmax) of the monolithic multiclass CNN."""
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
    label (0..K-1). Loader with real class labels (not binary)."""
    d = C.dirs_for(alpha)[split]
    loader = make_loader(d, C.CLASS_FOLDERS, C.BATCH_SIZE,
                         shuffle=False, num_workers=workers)   # multiclass labels
    pg, ymc = _p_stego(_load_generalist(alpha, device), loader, device)
    cols = [pg]
    for algo in ALGOS:
        p, _ = _p_stego(_load_specialist(algo, alpha, device), loader, device)
        cols.append(p)
    return np.stack(cols, axis=1), ymc       # (N, 1+len(ALGOS)), (N,) multiclass


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
    Xf, yf_mc = _meta_features(args.alpha, "fusion", dev, args.workers)
    Xt, yt_mc = _meta_features(args.alpha, "test",   dev, args.workers)
    yf_bin = (yf_mc != cover_idx).astype(int)
    yt_bin = (yt_mc != cover_idx).astype(int)
    print(f"   fusion={Xf.shape} test={Xt.shape}  (cols={META_COLS})")

    # ===================== A) binary cover/stego DETECTION =====================
    meta_b = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xf, yf_bin)
    ens_s = meta_b.predict_proba(Xt)[:, 1]
    r_ens_bin = _metrics(yt_bin, ens_s)
    r_gen_bin = _metrics(yt_bin, Xt[:, 0])           # col 0 = generalist
    r_mean = _metrics(yt_bin, Xt[:, 1:].mean(1))      # non-learned fusion (specs)
    # bootstrap 95% CI of the ΔAUC (ensemble - generalist), over the TEST scores
    _yb = np.asarray(yt_bin); _gs = Xt[:, 0]; _rng = np.random.default_rng(0); _v = []
    for _ in range(2000):
        _i = _rng.integers(0, len(_yb), len(_yb))
        if len(np.unique(_yb[_i])) < 2:
            continue
        _v.append(roc_auc_score(_yb[_i], ens_s[_i]) - roc_auc_score(_yb[_i], _gs[_i]))
    delta_det = [round(float(r_ens_bin["auc"] - r_gen_bin["auc"]), 4),
                 [round(float(np.percentile(_v, 2.5)), 4), round(float(np.percentile(_v, 97.5)), 4)]]

    # ===================== B) 4-CLASS: modular vs monolithic ====================
    # Modular: multiclass meta-model over [gen, specialists] (leakage-free fusion).
    meta_mc = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xf, yf_mc)
    r_ens_mc = _mc_metrics(yt_mc, meta_mc.predict(Xt))
    # Monolithic: the multiclass CNN over the SAME test.
    d = C.dirs_for(args.alpha)["test"]
    te = make_loader(d, C.CLASS_FOLDERS, C.BATCH_SIZE, shuffle=False, num_workers=args.workers)
    mono_pred, y_mono = _monolith_preds(args.alpha, te, dev)
    r_mono_mc = _mc_metrics(y_mono, mono_pred)

    out = {"alpha": args.alpha, "classes": C.CLASS_FOLDERS, "meta_cols": META_COLS,
           "detection": {"generalist": r_gen_bin, "ensemble": r_ens_bin, "fusion_mean": r_mean,
                         "delta_ens_minus_gen": delta_det},
           "fourclass": {"monolithic_cnn": r_mono_mc, "modular_ensemble": r_ens_mc}}
    res_dir = C.RESULTS_DIR
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / f"ensemble_{C.alpha_tag(args.alpha)}.json").write_text(json.dumps(out, indent=2))

    rb = lambda n, r: f"| {n} | {r['auc']:.4f} | {r['P_D']:.4f} | {r['P_FA']:.4f} |"
    print(f"\n### A) Cover/stego detection @{args.alpha} (TEST)")
    print("| model | AUC | P_D | P_FA |\n|---|---|---|---|")
    print(rb("Generalist (monolithic)", r_gen_bin))
    print(rb("Modular ensemble (LogReg)", r_ens_bin))
    print(rb("Mean fusion (non-learned)", r_mean))
    print(f"  ΔAUC = {r_ens_bin['auc']-r_gen_bin['auc']:+.4f}")

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
