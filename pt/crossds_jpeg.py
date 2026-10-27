"""Cross-dataset JPEG detection: the DCTR cover-vs-all detector trained on BOSSbase and
evaluated on the ALASKA#2 set written by gen_jpeg.py --name DATA_JPEG_ALASKA2. Writes
results/crossds_jpeg_<alpha>.json.

Usage:  STEG_DOMAIN=jpeg uv run python pt/crossds_jpeg.py --alpha 0.4
"""
import sys, argparse, json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))          # code/pt (extract_dctr, data)
sys.path.append(str(Path(__file__).resolve().parent.parent))   # code/ (config)
import numpy as np, torch
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
import config as C
from extract_dctr import (dct_basis_filters, dctr_features_batch, load_batch, DCTR_Q, DCTR_T,
                          DECOMPRESSION)
from data import list_images
from util import (load_dctr, load_dctr_groups, fit_logreg, convergence_info, group_boot,
                  _dctr_dir)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def feats(paths, W):
    X = np.zeros((len(paths), 64*64*(DCTR_T+1)), np.float32)
    for s in range(0, len(paths), 64):
        b = paths[s:s+64]
        X[s:s+len(b)] = dctr_features_batch(load_batch(b, DEVICE), W, DCTR_Q, DCTR_T)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=C.FAMILY_ID_PAYLOAD)
    ap.add_argument("--quality", type=int, default=75)
    ap.add_argument("--target", default="DATA_JPEG_ALASKA2",
                    help="dataset prefix of the cross-source set (gen_jpeg.py --name)")
    args = ap.parse_args()
    assert C.DOMAIN == "jpeg", "Run with STEG_DOMAIN=jpeg"
    names = C.CLASS_FOLDERS; cov = names.index(C.COVER_NAME)

    # The BOSS cache must have been extracted with the same (q, T) used on the target.
    pj = _dctr_dir(args.alpha) / "params.json"
    if pj.exists():
        p = json.loads(pj.read_text())
        assert (float(p["q"]), int(p["T"])) == (DCTR_Q, DCTR_T), \
            f"BOSS DCTR cache has q={p['q']} T={p['T']}, expected q={DCTR_Q} T={DCTR_T}"
        assert p.get("decompression") == DECOMPRESSION, \
            f"BOSS DCTR cache decompression {p.get('decompression')!r} != {DECOMPRESSION!r}; re-extract"

    # detector trained on BOSSbase (cover-vs-all)
    X, y = load_dctr(args.alpha); g = load_dctr_groups(args.alpha)
    sc = StandardScaler().fit(X["train"])
    det = fit_logreg(sc.transform(X["train"]), (y["train"] != cov).astype(int), tag="detect")
    yin = (y["test"] != cov).astype(int)
    sin = det.predict_proba(sc.transform(X["test"]))[:, 1]
    auc_in = float(roc_auc_score(yin, sin))
    ci_in = group_boot(lambda i: roc_auc_score(yin[i], sin[i]), g["test"], point=auc_in)
    print(f"[in-dist BOSS] detection AUC = {auc_in:.4f} CI95={ci_in}")
    del X

    # DCTR of the target set
    root = C.DATASETS / f"{args.target}_q{args.quality}_{C.alpha_tag(args.alpha)}"
    present = [nm for nm in names if (root / nm).is_dir() and list_images(root / nm)]
    if C.COVER_NAME not in present:
        sys.exit(f"no {C.COVER_NAME} images under {root}")
    W = dct_basis_filters().to(DEVICE)
    Xa, ya, ga = [], [], []
    for nm in present:
        ps = list_images(root / nm)
        Xa.append(feats(ps, W)); ya += [names.index(nm)] * len(ps); ga += [p.stem for p in ps]
        print(f"  {args.target} {nm}: {len(ps)} imgs")
    Xa = np.concatenate(Xa); ya = np.array(ya); ga = np.array(ga)

    # cross-eval
    sa = det.predict_proba(sc.transform(Xa))[:, 1]; ybin = (ya != cov).astype(int)
    pred = (sa >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(ybin, pred, labels=[0, 1]).ravel()
    auc_x = float(roc_auc_score(ybin, sa))
    ci_x = group_boot(lambda i: roc_auc_score(ybin[i], sa[i]), ga, point=auc_x)
    print(f"[CROSS {args.target}] detection AUC = {auc_x:.4f} CI95={ci_x}  "
          f"P_D={tp/(tp+fn):.4f}  P_FA={fp/(fp+tn):.4f}")
    per = {}
    for nm in present:
        if nm == C.COVER_NAME:
            continue
        m = (ya == cov) | (ya == names.index(nm)); yt = (ya == names.index(nm)).astype(int)
        a = float(roc_auc_score(yt[m], sa[m]))
        per[nm] = [round(a, 4), group_boot(lambda i: roc_auc_score(yt[i], sa[i]), ga,
                                           mask=m, point=a)]
        print(f"  cover-vs-{nm:9s}: AUC = {a:.4f} CI95={per[nm][1]}")

    out = {"alpha": args.alpha, "train_set": f"{C.DATA_PREFIX}_{C.alpha_tag(args.alpha)}",
           "target_set": root.name, "classes_evaluated": present,
           "dctr": {"q": DCTR_Q, "T": DCTR_T, "variant": "unmerged 64 phases"},
           "indist_auc": [round(auc_in, 4), ci_in],
           "cross_auc": [round(auc_x, 4), ci_x],
           "cross_P_D": round(float(tp/(tp+fn)), 4), "cross_P_FA": round(float(fp/(fp+tn)), 4),
           "cross_cover_vs_scheme_auc": per, "n_target_images": int(len(ya)),
           "ci_protocol": "grouped bootstrap by cover (file stem), n=2000",
           "provenance": convergence_info()}
    res = C.RESULTS_DIR; res.mkdir(parents=True, exist_ok=True)
    dst = res / f"crossds_jpeg_{C.alpha_tag(args.alpha)}.json"
    dst.write_text(json.dumps(out, indent=2))
    print("saved:", dst)


if __name__ == "__main__":
    main()
