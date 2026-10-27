"""Fidelity check of the JPEG simulators on a sample of covers: change rate, cover-vs-one
detectability and adaptive-cluster membership of UED and J-MiPOD, and a cross-check of the
local framework against conseal for J-UNIWARD, UERD and nsF5. All classifiers are trained
and tested on disjoint covers.

Usage:
  uv run python pt/validate_embedders.py extract --n 400
  uv run python pt/validate_embedders.py analyze
"""
import sys, argparse, json, shutil
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))          # pt/ (jpeg_embedders)
sys.path.append(str(Path(__file__).resolve().parent.parent))   # repo root (config)
import config as C

ALPHA, QUALITY = 0.4, 75
HOME = ["nsF5", "UERD", "JUNIWARD", "UED", "JMIPOD"]   # via jpeg_embedders
REFERENCE = ["JUNIWARD", "UERD", "nsF5"]               # via conseal, for the cross-check
# Adaptive-cluster membership is tested against the schemes AS THEY APPEAR IN THE DATASET:
# J-UNIWARD and UERD come from conseal there (ref_*), UED and J-MiPOD from jpeg_embedders.
# The home J-UNIWARD/UERD are only for the home-vs-reference cross-check below.
ADAPTIVE = ["ref_UERD", "ref_JUNIWARD", "UED", "JMIPOD"]
DEFAULT_OUT = C.RESULTS_DIR / "embval"


def _label(scheme, source):
    return scheme if source == "home" else f"ref_{scheme}"


# --------------------------------------------------------------------------- #
def extract(n, out):
    import torch
    import jpeglib
    import conseal as cl
    from jpeg_embedders import EMBEDDERS, embed_seed, write_jpeg
    from gen_jpeg import make_cover
    from extract_dctr import dct_basis_filters, dctr_features_batch, load_batch, DCTR_Q, DCTR_T
    from engine import DEVICE

    _AC = np.ones((8, 8), bool); _AC[0, 0] = False
    shutil.rmtree(out, ignore_errors=True)
    (out / "COVER").mkdir(parents=True)

    covers = sorted((C.DATASETS / "covers" / "BOSSbase_1.01").glob("*.pgm"))[:n]
    print(f"[extract] covers={len(covers)} alpha={ALPHA} Q={QUALITY} dev={DEVICE}", flush=True)
    stems, rates = [], {}

    def ref_embed(scheme, Y0, qt, sp, seed):
        """conseal reference costs, restricted to AC so the bpnzAC convention matches."""
        if scheme == "nsF5":
            return cl.nsF5.simulate_single_channel(Y0, ALPHA, seed=seed).astype(Y0.dtype)
        if scheme == "UERD":
            rp, rm = cl.uerd.compute_cost_adjusted(Y0, qt)
        else:
            rp, rm = cl.juniward.compute_cost_adjusted(sp, Y0, qt)
        rp = rp.copy(); rm = rm.copy()
        rp[..., 0, 0] = 1e13; rm[..., 0, 0] = 1e13
        nz = int(((Y0 != 0) & _AC[None, None]).sum())
        return (Y0 + cl.simulate.ternary((rp, rm), ALPHA, nz, seed=seed)).astype(Y0.dtype)

    for pgm in covers:
        stem = pgm.stem
        cjpg = out / "COVER" / f"{stem}.jpg"
        make_cover(pgm, cjpg, QUALITY)                 # same container writer as the stego
        im = jpeglib.read_dct(str(cjpg))
        Y0 = im.Y.astype(np.int32); qt = im.qt[0].astype(np.float64)
        spatial = jpeglib.read_spatial(str(cjpg)).spatial[:, :, 0]   # same decoder as the generators
        n_nz = int(((Y0 != 0) & _AC[None, None]).sum())
        # Seeds match the dataset generators (distinct per scheme). Home and reference
        # versions of one scheme share a seed: common random numbers for that comparison.
        for scheme in HOME:
            Ys, nch = EMBEDDERS[scheme](Y0, qt, ALPHA, seed=embed_seed(stem, scheme, ALPHA),
                                        spatial=spatial)
            lab = _label(scheme, "home")
            (out / lab).mkdir(exist_ok=True)
            im.Y = Ys.astype(im.Y.dtype); write_jpeg(im, out / lab / f"{stem}.jpg")
            rates.setdefault(lab, []).append(nch / max(n_nz, 1))
        for scheme in REFERENCE:
            Ys = ref_embed(scheme, Y0, qt, spatial.astype(np.float64),
                           embed_seed(stem, scheme, ALPHA))
            lab = _label(scheme, "reference")
            (out / lab).mkdir(exist_ok=True)
            im.Y = Ys.astype(im.Y.dtype); write_jpeg(im, out / lab / f"{stem}.jpg")
            rates.setdefault(lab, []).append(
                float(((Ys != Y0) & _AC[None, None]).sum()) / max(n_nz, 1))
        stems.append(stem)

    W = dct_basis_filters().to(DEVICE)

    def dctr(folder):
        paths = [out / folder / f"{s}.jpg" for s in stems]
        X = np.zeros((len(paths), 64 * 64 * (DCTR_T + 1)), np.float32)
        for i in range(0, len(paths), 32):
            b = paths[i:i + 32]
            X[i:i + len(b)] = dctr_features_batch(load_batch(b, DEVICE), W, DCTR_Q, DCTR_T)
        return X

    folders = ["COVER"] + [_label(s, "home") for s in HOME] + \
                          [_label(s, "reference") for s in REFERENCE]
    for f in folders:
        np.save(out / f"X_{f}.npy", dctr(f))
        print(f"  DCTR {f}", flush=True)
    np.save(out / "stems.npy", np.array(stems))
    (out / "rates.json").write_text(json.dumps(
        {k: float(np.mean(v)) for k, v in rates.items()}, indent=2))
    print("[extract] done ->", out)


# --------------------------------------------------------------------------- #
def analyze(out):
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score
    from util import fit_logreg, convergence_info

    stems = np.load(out / "stems.npy", allow_pickle=True).astype(str)
    rates = json.loads((out / "rates.json").read_text())
    X = {p.stem[2:]: np.load(p) for p in out.glob("X_*.npy")}

    # Disjoint COVER halves: every image of a cover sits on one side only.
    rng = np.random.default_rng(0)
    order = rng.permutation(len(stems))
    tr_cov, te_cov = order[:len(order) // 2], order[len(order) // 2:]

    def fit_eval(a_key, b_key, metric):
        """Train A-vs-B on the training covers, score on the held-out covers."""
        A, B = X[a_key], X[b_key]
        Xtr = np.vstack([A[tr_cov], B[tr_cov]])
        ytr = np.r_[np.zeros(len(tr_cov)), np.ones(len(tr_cov))]
        Xte = np.vstack([A[te_cov], B[te_cov]])
        yte = np.r_[np.zeros(len(te_cov)), np.ones(len(te_cov))]
        sc = StandardScaler().fit(Xtr)
        # C=0.05: strong regularisation for n (~400) << d (20,480). Classes are balanced by
        # construction, so fit_logreg's class_weight="balanced" is a no-op here.
        clf = fit_logreg(sc.transform(Xtr), ytr, C=0.05, tag=f"{a_key}|{b_key}")
        Z = sc.transform(Xte)
        return (float(roc_auc_score(yte, clf.predict_proba(Z)[:, 1])) if metric == "auc"
                else float(accuracy_score(yte, clf.predict(Z))))

    res = {"alpha": ALPHA, "quality": QUALITY, "n_covers": len(stems),
           "protocol": "train/test split by COVER (disjoint halves); full DCTR, no PCA",
           "change_rate": rates, "detectability_auc": {}, "adaptive_pairwise_acc": {},
           "home_vs_reference": {}}

    print("\n-- change rate (fraction of nzAC changed) --")
    for k in sorted(rates):
        print(f"  {k:16s} {rates[k]:.4f}")

    print("\n-- cover-vs-one detectability (AUC, disjoint covers) --")
    for lab in sorted(X):
        if lab == "COVER":
            continue
        res["detectability_auc"][lab] = fit_eval("COVER", lab, "auc")
        print(f"  COVER vs {lab:16s} AUC={res['detectability_auc'][lab]:.4f}")

    # Do the adaptive schemes overlap each other? 0.5 = indistinguishable = they belong to
    # the same cluster; near 1.0 = the simulator carries its own signature.
    print("\n-- adaptive pairwise separability (accuracy, 0.5 = indistinguishable) --")
    for i in range(len(ADAPTIVE)):
        for j in range(i + 1, len(ADAPTIVE)):
            a, b = ADAPTIVE[i], ADAPTIVE[j]
            acc = fit_eval(a, b, "acc")
            res["adaptive_pairwise_acc"][f"{a}|{b}"] = acc
            print(f"  {a:10s} vs {b:10s} acc={acc:.3f}")

    # Framework cross-check: same cover, same payload, home cost vs conseal reference cost.
    print("\n-- home vs conseal reference (|ΔAUC| small = the framework agrees) --")
    for s in REFERENCE:
        h, r = res["detectability_auc"][s], res["detectability_auc"][f"ref_{s}"]
        res["home_vs_reference"][s] = {"home_auc": round(h, 4), "reference_auc": round(r, 4),
                                       "abs_delta": round(abs(h - r), 4),
                                       "home_rate": rates[s], "reference_rate": rates[f"ref_{s}"]}
        print(f"  {s:10s} home={h:.4f} reference={r:.4f} |Δ|={abs(h - r):.4f}")

    res["provenance"] = convergence_info()
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dst = C.RESULTS_DIR / "embedder_fidelity.json"
    dst.write_text(json.dumps(res, indent=2))
    print("\nsaved:", dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["extract", "analyze"])
    ap.add_argument("--n", type=int, default=400, help="number of covers (extract)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="scratch dir for the JPEGs")
    args = ap.parse_args()
    out = Path(args.out)
    extract(args.n, out) if args.mode == "extract" else analyze(out)


if __name__ == "__main__":
    main()
