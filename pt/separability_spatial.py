"""
Separability analysis (spatial domain).

Measures the overlap between the 6 classes (COVER + HILL, HUGO, MiPOD, S-UNIWARD,
WOW) in feature space.

What it does:
  1. Extracts a feature space for the 6 classes of the chosen split:
       --features detector : penultimate layer (128-d) of the trained binary detector.
       --features srm      : model-free — per-channel statistics of the 7 SRM residuals
                             (without bias from any training).
  2. Computes, over those features:
       - Multiclass Fisher ratio J = trace(Sw^-1 Sb)  (6 classes and stego-only 5).
       - Fisher per pair of classes (6x6 matrix).
       - 1-NN error (LOO) and neighborhood purity (kDN-like).
       - Global and per-class silhouette.
       - Neighborhood confusion matrix: which class the nearest neighbor of each
         class belongs to.
       - (Optional) pycol measures if the library is installed.
  3. Saves results/separability_<features>_<split>_a0XX.json and prints md tables.

By default it runs on CPU (--device cpu) so as not to disturb GPU training.

Usage:
  python separability_spatial.py --features detector --split test --alpha 0.4
  python separability_spatial.py --features srm --split fusion
"""
import sys, json, argparse, itertools
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, silhouette_samples

import config as C
from data import make_loader
from models import BinaryCNN, SRMFrontEnd


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
def _detector_features(model_path, loader, device, use_srm):
    """Penultimate layer (128-d) of the binary detector: output of head[4] (ReLU)."""
    model = BinaryCNN(in_ch=C.CHANNELS, use_srm=use_srm, dropout=0.0)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval().to(device)

    feats = {}
    h = model.head[4].register_forward_hook(           # ReLU after Linear(.,128)
        lambda m, i, o: feats.__setitem__("z", o.detach()))
    X, Y = [], []
    with torch.no_grad():
        for x, y in loader:
            model(x.to(device))
            X.append(feats["z"].cpu().numpy()); Y.append(np.asarray(y))
    h.remove()
    return np.concatenate(X), np.concatenate(Y)


def _srm_features(loader, device):
    """Model-free: per image, statistics of the 7 SRM residual channels.
    For each channel: mean, std, mean|.|, p90(|.|)  -> 7*4 = 28 dimensions."""
    srm = SRMFrontEnd().eval().to(device)
    X, Y = [], []
    with torch.no_grad():
        for x, y in loader:
            r = srm(x.to(device))                       # (B,7,H,W)
            a = r.abs()
            mean = r.mean(dim=(2, 3))
            std = r.std(dim=(2, 3))
            mabs = a.mean(dim=(2, 3))
            p90 = torch.quantile(a.flatten(2), 0.90, dim=2)
            feat = torch.cat([mean, std, mabs, p90], dim=1)   # (B,28)
            X.append(feat.cpu().numpy()); Y.append(np.asarray(y))
    return np.concatenate(X), np.concatenate(Y)


# --------------------------------------------------------------------------- #
# Separability metrics
# --------------------------------------------------------------------------- #
def fisher_multiclass(X, y):
    """J = trace(Sw^-1 Sb).  Higher = more separable classes."""
    X = np.asarray(X, np.float64); y = np.asarray(y)
    mu = X.mean(0); d = X.shape[1]
    Sw = np.zeros((d, d)); Sb = np.zeros((d, d))
    for c in np.unique(y):
        Xc = X[y == c]; pi = len(Xc) / len(X); mc = Xc.mean(0)
        Sw += pi * np.cov(Xc, rowvar=False, bias=True)
        diff = (mc - mu).reshape(-1, 1)
        Sb += pi * (diff @ diff.T)
    return float(np.trace(np.linalg.pinv(Sw) @ Sb))


def pairwise_fisher(X, y, names):
    """Fisher J restricted to each pair of classes -> symmetric matrix."""
    classes = list(range(len(names)))
    M = np.full((len(names), len(names)), np.nan)
    for a, b in itertools.combinations(classes, 2):
        m = (y == a) | (y == b)
        M[a, b] = M[b, a] = fisher_multiclass(X[m], y[m])
    return M


def knn_overlap(X, y, k=10):
    """1-NN error (LOO) and mean neighborhood purity (fraction of the k neighbors
    sharing the class). Over standardized features."""
    Xs = StandardScaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xs)
    _, idx = nn.kneighbors(Xs)
    idx = idx[:, 1:]                                    # removes the point itself
    nbr_lab = y[idx]
    err_1nn = float((nbr_lab[:, 0] != y).mean())
    purity = float((nbr_lab == y[:, None]).mean())
    return err_1nn, purity, Xs


def neighbor_confusion(Xs, y, names):
    """Matrix: for each class (row), distribution of nearest-neighbor classes
    (column)."""
    nn = NearestNeighbors(n_neighbors=2).fit(Xs)
    _, idx = nn.kneighbors(Xs)
    nn1 = y[idx[:, 1]]
    n = len(names); M = np.zeros((n, n))
    for true_c, nbr_c in zip(y, nn1):
        M[true_c, nbr_c] += 1
    M = M / M.sum(1, keepdims=True).clip(min=1)         # normalize by row
    return M


def _scalarize(v):
    """pycol sometimes returns scalars and sometimes arrays (per pair/class).
    Stores the raw value (list) and its mean for tables."""
    arr = np.asarray(v, dtype=float).ravel()
    if arr.size == 1:
        return {"mean": float(arr[0]), "raw": float(arr[0])}
    return {"mean": float(np.nanmean(arr)), "raw": arr.tolist()}


def try_pycol(X, y, names):
    """Complexity battery of Lorena et al. (pycol) if installed.
    Includes the measures that do NOT duplicate Fisher/1-NN/silhouette:
    F2 (overlap volume), F3/F4 (feature efficiency), N1 (MST boundary),
    N2 (intra/inter distance ratio), T1 (hyperspheres), LSC (local-set).
    """
    try:
        import arff as liacarff
        from pycol_complexity import complexity as pycol
    except Exception as e:
        return {"_available": False, "_reason": str(e)}
    p = Path("/tmp/sep_meta.arff")
    attrs = [(f"f{i}", "REAL") for i in range(X.shape[1])]
    attrs.append(("class", sorted(np.unique(y).astype(str).tolist())))
    data = [list(map(float, xi)) + [str(int(yi))] for xi, yi in zip(X, y)]
    liacarff.dump({"description": "", "relation": "sep", "attributes": attrs,
                   "data": data}, open(p, "w"))
    cx = pycol.Complexity(str(p), distance_func="default", file_type="arff")
    # name -> method. Some may not exist depending on the version: they are caught.
    wanted = {
        "F1_fisher": "F1", "F1v_dir_fisher": "F1v", "F2_overlap_volume": "F2",
        "F3_feat_efficiency": "F3", "F4_collective_feat_eff": "F4",
        "N1_borderline_mst": "N1", "N2_intra_inter_ratio": "N2",
        "N3_1nn_error": "N3", "T1_hyperspheres": "T1", "LSC_local_set": "LSC",
        "SI_separability": "SI", "kDN": "kDN", "purity": "purity",
    }
    out = {"_available": True}
    for label, meth in wanted.items():
        fn = getattr(cx, meth, None)
        if fn is None:
            continue
        try:
            out[label] = _scalarize(fn())
        except Exception as e:
            out[label] = {"error": str(e)[:80]}
    return out


# --------------------------------------------------------------------------- #
def md_matrix(M, names, fmt="{:.2f}"):
    head = "| | " + " | ".join(names) + " |"
    sep = "|" + "---|" * (len(names) + 1)
    rows = []
    for i, nm in enumerate(names):
        cells = [("—" if np.isnan(M[i, j]) else fmt.format(M[i, j])) for j in range(len(names))]
        rows.append(f"| **{nm}** | " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["detector", "srm"], default="detector")
    ap.add_argument("--split", choices=["train", "fusion", "test"], default="test")
    ap.add_argument("--alpha", type=float, default=C.FAMILY_ID_PAYLOAD)
    ap.add_argument("--model", default=None, help="path to the detector (def: pt_binary_a0XX_s42.pt)")
    ap.add_argument("--no-srm", dest="use_srm", action="store_false",
                    help="the loaded model is the ablation without SRM")
    ap.add_argument("--device", default="cpu", help="cpu (def, does not disturb the GPU) or cuda")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    names = C.CLASS_FOLDERS
    d = C.dirs_for(args.alpha)[args.split]
    loader = make_loader(d, names, C.BATCH_SIZE, shuffle=False, num_workers=args.workers)
    dev = args.device
    print(f"== Separability | features={args.features} | split={args.split} "
          f"| alpha={args.alpha} | dev={dev}")

    if args.features == "detector":
        mp = args.model or (C.MODEL_DIR / f"pt_binary_{C.alpha_tag(args.alpha)}_s42.pt")
        print(f"   model: {mp}")
        X, y = _detector_features(mp, loader, dev, args.use_srm)
    else:
        X, y = _srm_features(loader, dev)
    print(f"   X={X.shape}  classes={np.bincount(y).tolist()}")

    # --- metrics ---
    J6 = fisher_multiclass(X, y)
    stego_mask = y != names.index(C.COVER_NAME)
    # re-label stego 0..4 for the 5-class J
    ys = y[stego_mask]; _, ys = np.unique(ys, return_inverse=True)
    J5 = fisher_multiclass(X[stego_mask], ys)
    Mpair = pairwise_fisher(X, y, names)
    err1nn, purity, Xs = knn_overlap(X, y)
    sil = float(silhouette_score(Xs, y))
    sil_by = {names[c]: float(silhouette_samples(Xs, y)[y == c].mean())
              for c in range(len(names))}
    Mnbr = neighbor_confusion(Xs, y, names)
    pycol = try_pycol(X, y, names)

    # cover vs mean stego-stego (overlap summary)
    cov_i = names.index(C.COVER_NAME)
    cover_pairs = np.nanmean([Mpair[cov_i, j] for j in range(len(names)) if j != cov_i])
    stego_idx = [i for i in range(len(names)) if i != cov_i]
    stego_pairs = np.nanmean([Mpair[a, b] for a, b in itertools.combinations(stego_idx, 2)])

    # --- output ---
    out = {
        "features": args.features, "split": args.split, "alpha": args.alpha,
        "n": int(len(y)), "classes": names,
        "fisher_J_6class": J6, "fisher_J_5class_stego_only": J5,
        "fisher_cover_vs_stego_mean": float(cover_pairs),
        "fisher_stego_vs_stego_mean": float(stego_pairs),
        "knn_1nn_error": err1nn, "neighborhood_purity_k10": purity,
        "silhouette_global": sil, "silhouette_by_class": sil_by,
        "pairwise_fisher": np.where(np.isnan(Mpair), None, Mpair).tolist(),
        "neighbor_confusion": Mnbr.tolist(),
        "pycol": pycol,
    }
    res_dir = C.RESULTS_DIR
    res_dir.mkdir(parents=True, exist_ok=True)
    outp = res_dir / f"separability_{args.features}_{args.split}_{C.alpha_tag(args.alpha)}.json"
    outp.write_text(json.dumps(out, indent=2))

    print(f"\n### Multiclass Fisher J ({args.features}, {args.split})")
    print(f"- 6 classes (COVER+5 stego): **J = {J6:.3f}**")
    print(f"- stego only (5 classes):    **J = {J5:.3f}**")
    print(f"- Mean Fisher COVER↔stego: {cover_pairs:.3f}  |  stego↔stego: {stego_pairs:.3f}")
    print(f"\n### Global overlap")
    print(f"- 1-NN error (LOO): {err1nn:.3f}   neighborhood purity(k=10): {purity:.3f}")
    print(f"- global silhouette: {sil:.3f}")
    print("- silhouette by class: " + ", ".join(f"{k}={v:.2f}" for k, v in sil_by.items()))
    print(f"\n### Fisher per pair of classes (higher=more separable)\n{md_matrix(Mpair, names)}")
    print(f"\n### Neighborhood confusion (row=class, col=class of the nearest neighbor)\n{md_matrix(Mnbr, names)}")
    if pycol.get("_available"):
        print("\n### pycol (Lorena et al.) — measure = mean (↑/↓ = better separability)")
        direction = {  # Lorena/pycol convention: except SI and purity, HIGHER = more complex (worse)
            "F1_fisher": "↓", "F1v_dir_fisher": "↓", "F2_overlap_volume": "↓",
            "F3_feat_efficiency": "↓", "F4_collective_feat_eff": "↓",
            "N1_borderline_mst": "↓", "N2_intra_inter_ratio": "↓",
            "N3_1nn_error": "↓", "T1_hyperspheres": "↓", "LSC_local_set": "↓",
            "SI_separability": "↑", "kDN": "↓", "purity": "↑"}
        print("| measure | value | better |\n|---|---|---|")
        for k, v in pycol.items():
            if k.startswith("_"):
                continue
            val = v.get("mean") if isinstance(v, dict) else v
            val = f"{val:.4f}" if isinstance(val, (int, float)) else str(v)
            print(f"| {k} | {val} | {direction.get(k,'')} |")
    else:
        print(f"\n[pycol not installed: {pycol.get('_reason','')[:60]} — Fisher/kNN/silhouette metrics were computed]")
    print(f"\nsaved: {outp}")


if __name__ == "__main__":
    main()
