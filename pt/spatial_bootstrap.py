"""Bootstrap 95% CIs for the spatial detection and GBRAS per-algorithm results.
Reuses already-trained models (pt_binary_*, pt_gbras_*): inference + grouped bootstrap only.
  - Detection sweep (CNN+SRM, seed 42) at 0.1/0.2/0.4 + no-SRM ablation @0.4.
  - GBRAS-Net per algorithm (cover-vs-one) @0.4.
"""
import sys, json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
from sklearn.metrics import roc_auc_score
import config as C
from data import make_loader, make_block_loader
from models import BinaryCNN
from engine import _predict, DEVICE
from gbras_io import load_gbras, per_image_probs
from util import boot_ci, source_ids

names = C.CLASS_FOLDERS; cov = names.index(C.COVER_NAME)


def det_auc(alpha, use_srm, tag):
    d = C.dirs_for(alpha)
    te = make_loader(d["test"], names, C.BATCH_SIZE, binary_cover_idx=cov, shuffle=False, num_workers=8)
    m = BinaryCNN(in_ch=C.CHANNELS, use_srm=use_srm, dropout=0.0)
    m.load_state_dict(torch.load(C.MODEL_DIR / f"pt_binary_{tag}_s42.pt", map_location=DEVICE))
    m.eval().to(DEVICE)
    s, y = _predict(m, te, "binary")
    return float(roc_auc_score(y.ravel(), s.ravel())), boot_ci(y, s, groups=source_ids(te))


def main():
    out = {"detection_sweep": {}, "gbras_per_algo": {}}
    for a in (0.1, 0.2, 0.4):
        auc, ci = det_auc(a, True, C.alpha_tag(a))
        out["detection_sweep"][f"{a}_bpp"] = {"auc": round(auc, 4), "ci95": ci}
        print(f"[det {a} bpp]  AUC={auc:.4f}  CI95={ci}")
    auc, ci = det_auc(0.4, False, "a040_nosrm")
    out["detection_sweep"]["0.4_bpp_noSRM"] = {"auc": round(auc, 4), "ci95": ci}
    print(f"[det 0.4 noSRM] AUC={auc:.4f}  CI95={ci}")

    d = C.dirs_for(0.4)
    for algo in [c for c in names if c != C.COVER_NAME]:
        m = load_gbras(C.MODEL_DIR / f"pt_gbras_{algo}_a040_s42.pt", 2, DEVICE)
        dl = make_block_loader(d["test"], C.COVER_NAME, algo, 64, shuffle=False, num_workers=8)
        P, y = per_image_probs(m, dl, DEVICE)
        auc = float(roc_auc_score(y, P[:, 1]))
        out["gbras_per_algo"][algo] = {"auc": round(auc, 4),
                                       "ci95": boot_ci(y, P[:, 1], groups=source_ids(dl))}
        print(f"[GBRAS {algo:9s}] AUC={auc:.4f}  CI95={out['gbras_per_algo'][algo]['ci95']}")

    res = C.RESULTS_DIR
    (res / "spatial_bootstrap.json").write_text(json.dumps(out, indent=2))
    print("\nsaved:", res / "spatial_bootstrap.json")


if __name__ == "__main__":
    main()
