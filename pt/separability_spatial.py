"""Class separability of the six spatial classes in a feature space: --features detector
(penultimate layer of the binary detector) or --features srm (per-channel statistics of the
seven SRM residuals, unclamped).

Computes multiclass and pairwise Fisher J on all rows; 1-NN error, neighbourhood purity,
silhouette and pycol measures on one class variant per cover, so that no two rows share a
cover; and a cover-vs-stego control (Fisher J, neighbourhood measures, grouped 5-fold linear
probe). Writes results/separability_<features>_<split>_a0XX.json. Runs on CPU by default.

Usage:
  python separability_spatial.py --features detector --split test --alpha 0.4
  python separability_spatial.py --features srm --split fusion
  python separability_spatial.py --features srm --max-covers 100 --no-pycol   # quick check
"""
import sys, json, argparse, itertools
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (silhouette_score, silhouette_samples, roc_auc_score, accuracy_score,
                             confusion_matrix)

import config as C
from data import make_loader
from models import BinaryCNN, SRMFrontEnd
from util import source_ids, fit_logreg, group_boot, convergence_info

NEIGHBOURHOOD_PROTOCOL = (
    "kNN 1-NN error, neighbourhood purity, neighbour confusion, silhouette and all pycol "
    "measures are computed on a subsample with exactly one randomly chosen class variant "
    "per cover (seeded with config.SEED), so no two rows share a cover; on all rows the "
    "nearest neighbour is the same cover's sibling in another class and the measures "
    "collapse to ~1.0 error. Fisher J (means/covariances only) uses all rows.")


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
    w = SRMFrontEnd().weight.to(device)                 # the same 7 kernels, but NO TLU clamp
    X, Y = [], []
    with torch.no_grad():
        for x, y in loader:
            r = F.conv2d(x.to(device), w, padding=2)    # (B,7,H,W) unclamped residuals
            a = r.abs()
            mean = r.mean(dim=(2, 3))
            std = r.std(dim=(2, 3))
            mabs = a.mean(dim=(2, 3))
            # 90th percentile as the k-th order statistic (torch.quantile rejects inputs
            # above 16M elements, which a 512x512 batch of 16 exceeds)
            flat = a.flatten(2)
            p90 = flat.kthvalue(int(np.ceil(0.90 * flat.shape[2])), dim=2).values
            feat = torch.cat([mean, std, mabs, p90], dim=1)   # (B,28)
            X.append(feat.cpu().numpy()); Y.append(np.asarray(y))
    return np.concatenate(X), np.concatenate(Y)


def one_variant_per_cover(groups, seed):
    """Row indices keeping exactly one randomly chosen row per cover id (sorted)."""
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    order = np.argsort(groups, kind="stable")
    _, starts, counts = np.unique(groups[order], return_index=True, return_counts=True)
    pick = starts + (rng.random(len(starts)) * counts).astype(int)
    return np.sort(order[pick])


def one_variant_per_cover_binary(groups, y, cover_idx, seed):
    """Row indices keeping one row per cover for the cover-vs-stego control: the COVER row
    with probability 1/2, otherwise one of that cover's stego rows chosen uniformly. The
    binary classes are balanced in expectation, so 1-NN error at chance is 0.5."""
    rng = np.random.default_rng(seed)
    rows = {}
    for i, (g, yy) in enumerate(zip(np.asarray(groups), np.asarray(y))):
        rows.setdefault(g, ([], []))[0 if yy == cover_idx else 1].append(i)
    pick = []
    for g in sorted(rows):
        cov_rows, st_rows = rows[g]
        if cov_rows and (not st_rows or rng.random() < 0.5):
            pick.append(cov_rows[0])
        else:
            pick.append(st_rows[rng.integers(len(st_rows))])
    return np.sort(np.array(pick))


def linear_probe(X, y, groups, cover_idx, folds=5):
    """Grouped k-fold (by cover) linear probes on the features: cover-vs-any-stego AUC and
    stego-only 5-class accuracy, each out-of-fold with a grouped-bootstrap CI."""
    yb = (y != cover_idx).astype(int)
    s_bin = np.full(len(y), np.nan)
    p_st = np.full(len(y), -1)
    for tr, te in GroupKFold(n_splits=folds).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        m = fit_logreg(sc.transform(X[tr]), yb[tr], tag="probe:cover-vs-stego")
        s_bin[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
        trs, tes = tr[y[tr] != cover_idx], te[y[te] != cover_idx]
        m5 = fit_logreg(sc.transform(X[trs]), y[trs], tag="probe:stego-5class")
        p_st[tes] = m5.predict(sc.transform(X[tes]))
    stego = y != cover_idx
    auc = float(roc_auc_score(yb, s_bin))
    acc5 = float(accuracy_score(y[stego], p_st[stego]))
    st_labels = np.unique(y[stego])
    cm5 = confusion_matrix(y[stego], p_st[stego], labels=st_labels)   # rows = truth
    return {"cover_vs_stego_auc": [round(auc, 4), group_boot(
                lambda i: roc_auc_score(yb[i], s_bin[i]), groups, point=auc)],
            "stego_5class_acc": [round(acc5, 4), group_boot(
                lambda i: accuracy_score(y[i], p_st[i]), groups, mask=stego, point=acc5)],
            "stego_5class_chance": round(1 / int(len(np.unique(y[stego]))), 4),
            "stego_5class_cm": cm5.tolist(),
            "stego_5class_labels": st_labels.tolist(),
            "protocol": (f"logistic regression (util.fit_logreg) on standardised features, "
                         f"{folds}-fold GroupKFold by cover, out-of-fold predictions; CIs by "
                         f"grouped bootstrap over covers")}


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


def run_pycol(X, y, tag):
    """Complexity battery of Lorena et al. (pycol).
    Includes the measures that do NOT duplicate Fisher/1-NN/silhouette:
    F2 (overlap volume), F3/F4 (feature efficiency), N1 (MST boundary),
    N2 (intra/inter distance ratio), T1 (hyperspheres), LSC (local-set).
    Fails loudly: a missing library or a measure that raises must not make entries
    silently vanish from the table (skip the whole battery with --no-pycol instead).
    """
    import arff as liacarff
    from pycol_complexity import complexity as pycol
    # Scratch file next to the results, not in a shared /tmp path where two
    # concurrent runs would overwrite each other; removed afterwards.
    p = C.RESULTS_DIR / f"_sep_meta_{tag}.arff"
    attrs = [(f"f{i}", "REAL") for i in range(X.shape[1])]
    attrs.append(("class", sorted(np.unique(y).astype(str).tolist())))
    data = [list(map(float, xi)) + [str(int(yi))] for xi, yi in zip(X, y)]
    try:
        with open(p, "w") as fh:
            liacarff.dump({"description": "", "relation": "sep", "attributes": attrs,
                           "data": data}, fh)
        cx = pycol.Complexity(str(p), distance_func="default", file_type="arff")
    finally:
        p.unlink(missing_ok=True)
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
            raise RuntimeError(f"pycol has no measure '{meth}' (installed version differs); "
                               f"pin the version or rerun with --no-pycol")
        try:
            out[label] = _scalarize(fn())
        except Exception as e:
            raise RuntimeError(f"pycol measure {meth} failed: {e}") from e
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
    ap.add_argument("--model", default=None, help="path to the detector (def: pt_binary_a0XX_s<SEED>.pt)")
    ap.add_argument("--no-srm", dest="use_srm", action="store_false",
                    help="the loaded model is the ablation without SRM")
    ap.add_argument("--no-pycol", dest="pycol", action="store_false",
                    help="skip the pycol battery (otherwise a pycol failure is an error)")
    ap.add_argument("--max-covers", type=int, default=None,
                    help="quick check: keep only the first N covers (sorted by id); the "
                         "output name gets a _maxN suffix so it cannot replace a full result")
    ap.add_argument("--device", default="cpu", help="cpu (def, does not disturb the GPU) or cuda")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    names = C.CLASS_FOLDERS
    d = C.dirs_for(args.alpha)[args.split]
    loader = make_loader(d, names, C.BATCH_SIZE, shuffle=False, num_workers=args.workers)
    if args.max_covers:
        # Workers start on first iteration, so shrinking samples here is picked up.
        keep = set(sorted({Path(p).stem for p, _ in loader.dataset.samples})[:args.max_covers])
        loader.dataset.samples = [s for s in loader.dataset.samples if Path(s[0]).stem in keep]
    groups = source_ids(loader)
    dev = args.device
    print(f"== Separability | features={args.features} | split={args.split} "
          f"| alpha={args.alpha} | dev={dev}")

    if args.features == "detector":
        mp = args.model or (C.MODEL_DIR / f"pt_binary_{C.alpha_tag(args.alpha)}_s{C.SEED}.pt")
        print(f"   model: {mp}")
        X, y = _detector_features(mp, loader, dev, args.use_srm)
    else:
        X, y = _srm_features(loader, dev)
    assert len(groups) == len(y), "cover ids and features are misaligned"
    print(f"   X={X.shape}  classes={np.bincount(y).tolist()}  covers={len(np.unique(groups))}")

    # --- metrics on all rows: Fisher (class means/covariances only) ---
    J6 = fisher_multiclass(X, y)
    stego_mask = y != names.index(C.COVER_NAME)
    # re-label stego 0..4 for the 5-class J
    ys = y[stego_mask]; _, ys = np.unique(ys, return_inverse=True)
    J5 = fisher_multiclass(X[stego_mask], ys)
    Mpair = pairwise_fisher(X, y, names)

    # --- neighbourhood/topology metrics: one class variant per cover ---
    sub = one_variant_per_cover(groups, C.SEED)
    Xn, yn = X[sub], y[sub]
    print(f"   neighbourhood subsample: n={len(sub)}  classes="
          f"{np.bincount(yn, minlength=len(names)).tolist()}")
    err1nn, purity, Xs = knn_overlap(Xn, yn)
    sil = float(silhouette_score(Xs, yn))
    sil_s = silhouette_samples(Xs, yn)
    sil_by = {names[c]: (float(sil_s[yn == c].mean()) if (yn == c).any() else None)
              for c in range(len(names))}
    Mnbr = neighbor_confusion(Xs, yn, names)
    tag = f"{args.features}_{args.split}_{C.alpha_tag(args.alpha)}"
    if args.max_covers:
        tag += f"_max{args.max_covers}"
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pycol = run_pycol(Xn, yn, tag) if args.pycol else {"_available": False,
                                                        "_reason": "skipped (--no-pycol)"}

    # --- instrument control: can these features see the embedding at all? ---
    cov_idx = names.index(C.COVER_NAME)
    sub_b = one_variant_per_cover_binary(groups, y, cov_idx, C.SEED + 1)
    Xb, yb = X[sub_b], (y[sub_b] != cov_idx).astype(int)
    err_b, pur_b, Xs_b = knn_overlap(Xb, yb)
    control = {
        "protocol": ("cover vs ANY stego. Fisher J on all rows; kNN / silhouette / pycol on one "
                     "row per cover (cover with prob 1/2, else a random stego variant; seed "
                     "config.SEED+1); linear probe over all rows. Read stego-vs-stego overlap as "
                     "convergence only if cover-vs-stego separates in the same features."),
        "n_rows_neighbourhood": int(len(sub_b)),
        "neighbourhood_class_counts": np.bincount(yb, minlength=2).tolist(),
        "fisher_J_cover_vs_stego": fisher_multiclass(X, (y != cov_idx).astype(int)),
        "knn_1nn_error": err_b, "knn_1nn_error_chance": 0.5,
        "neighborhood_purity_k10": pur_b,
        "silhouette": float(silhouette_score(Xs_b, yb)),
        "pycol": (run_pycol(Xb, yb, tag + "_ctrl") if args.pycol
                  else {"_available": False, "_reason": "skipped (--no-pycol)"}),
        "linear_probe": linear_probe(X, y, groups, cov_idx),
    }

    # cover vs mean stego-stego (overlap summary)
    cov_i = names.index(C.COVER_NAME)
    cover_pairs = np.nanmean([Mpair[cov_i, j] for j in range(len(names)) if j != cov_i])
    stego_idx = [i for i in range(len(names)) if i != cov_i]
    stego_pairs = np.nanmean([Mpair[a, b] for a, b in itertools.combinations(stego_idx, 2)])
    K = len(names)

    # --- output ---
    out = {
        "features": args.features, "split": args.split, "alpha": args.alpha,
        "n": int(len(y)), "n_covers": int(len(np.unique(groups))), "classes": names,
        "max_covers": args.max_covers,
        "fisher_J_6class": J6, "fisher_J_5class_stego_only": J5,
        "fisher_cover_vs_stego_mean": float(cover_pairs),
        "fisher_stego_vs_stego_mean": float(stego_pairs),
        "neighbourhood_protocol": NEIGHBOURHOOD_PROTOCOL,
        "n_rows_neighbourhood": int(len(sub)),
        "neighbourhood_class_counts": np.bincount(yn, minlength=K).tolist(),
        "neighbourhood_seed": C.SEED,
        "knn_1nn_error": err1nn, "knn_1nn_error_chance": 1 - 1 / K,
        "neighborhood_purity_k10": purity, "neighborhood_purity_chance": 1 / K,
        "silhouette_global": sil, "silhouette_by_class": sil_by,
        "pairwise_fisher": np.where(np.isnan(Mpair), None, Mpair).tolist(),
        "neighbor_confusion": Mnbr.tolist(),
        "pycol": pycol,
        "srm_residuals": "unclamped (no TLU)" if args.features == "srm" else None,
        "control_cover_vs_stego": control,
        "provenance": convergence_info(),
    }
    outp = C.RESULTS_DIR / f"separability_{tag}.json"
    outp.write_text(json.dumps(out, indent=2))

    print(f"\n### Multiclass Fisher J ({args.features}, {args.split}, all {len(y)} rows)")
    print(f"- 6 classes (COVER+5 stego): **J = {J6:.3f}**")
    print(f"- stego only (5 classes):    **J = {J5:.3f}**")
    print(f"- Mean Fisher COVER↔stego: {cover_pairs:.3f}  |  stego↔stego: {stego_pairs:.3f}")
    print(f"\n### Global overlap (one variant per cover, n={len(sub)})")
    print(f"- 1-NN error (LOO): {err1nn:.3f} (chance {1 - 1/K:.3f})   "
          f"neighborhood purity(k=10): {purity:.3f} (chance {1/K:.3f})")
    print(f"- global silhouette: {sil:.3f}")
    print("- silhouette by class: " + ", ".join(
        f"{k}={'—' if v is None else f'{v:.2f}'}" for k, v in sil_by.items()))
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
            print(f"| {k} | {v['mean']:.4f} | {direction.get(k, '')} |")
    else:
        print(f"\n[pycol {pycol.get('_reason', '')}]")
    lp = control["linear_probe"]
    print(f"\n### Instrument control: cover vs ANY stego (can these features see the embedding?)")
    print(f"- Fisher J cover-vs-stego (all rows): {control['fisher_J_cover_vs_stego']:.3f}")
    print(f"- 1-NN error: {err_b:.3f} (chance 0.5), purity(k=10): {pur_b:.3f}, "
          f"silhouette: {control['silhouette']:.3f}, n={len(sub_b)}")
    print(f"- linear probe: cover-vs-stego AUC={lp['cover_vs_stego_auc'][0]:.3f} "
          f"CI95={lp['cover_vs_stego_auc'][1]} | stego 5-class acc={lp['stego_5class_acc'][0]:.3f} "
          f"CI95={lp['stego_5class_acc'][1]} (chance {lp['stego_5class_chance']:.3f})")
    print(f"\nsaved: {outp}")


if __name__ == "__main__":
    main()
