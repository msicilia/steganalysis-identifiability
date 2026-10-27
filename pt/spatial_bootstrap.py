"""Bootstrap 95% CIs by cover for the spatial detection sweep (with the no-SRM ablation) and
the GBRAS-Net cover-vs-one specialists, from the trained models. Each entry carries a
collapse flag (near-constant scores), a flag when the CI does not contain the point estimate,
and the within-cover P(score_stego > score_cover).
"""
import sys, json, argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np, torch
from sklearn.metrics import roc_auc_score
import config as C
from data import make_loader, make_block_loader
from models import BinaryCNN
from engine import _predict, DEVICE
from gbras_io import load_gbras, per_image_probs
from util import boot_ci, source_ids, group_boot

names = C.CLASS_FOLDERS; cov = names.index(C.COVER_NAME)
COLLAPSE_EPS = 1e-3


def paired_win_rate(y, s, groups):
    """P(score_stego > score_cover) over (cover, stego) pairs sharing a cover id (ties = 1/2),
    with a grouped-bootstrap CI over covers."""
    y = np.asarray(y).ravel(); s = np.asarray(s).ravel(); g = np.asarray(groups).ravel()
    cover_s = {gi: si for gi, si, yi in zip(g, s, y) if yi == 0}
    assert len(cover_s) == int((y == 0).sum()), "a cover id has more than one cover row"
    st = np.where(y == 1)[0]
    st = st[np.isin(g[st], list(cover_s))]
    c = np.array([cover_s[gi] for gi in g[st]])
    win = (s[st] > c).astype(float) + 0.5 * (s[st] == c)
    if not len(win):
        return {"p_stego_gt_cover": None, "ci95": None, "n_pairs": 0}
    return {"p_stego_gt_cover": round(float(win.mean()), 4),
            "ci95": group_boot(lambda i: win[i].mean(), g[st]),
            "n_pairs": int(len(win))}


def auc_entry(y, s, groups):
    y = np.asarray(y).ravel(); s = np.asarray(s).ravel()
    auc = round(float(roc_auc_score(y, s)), 4)
    ci = boot_ci(y, s, groups=groups)
    sd = float(np.std(s))
    contains = bool(ci[0] <= auc <= ci[1])
    reasons = [r for r, bad in (("score_std<1e-3", sd < COLLAPSE_EPS),
                                ("ci_width<1e-3", ci[1] - ci[0] < COLLAPSE_EPS)) if bad]
    return {"auc": auc, "ci95": ci, "score_std": round(sd, 6), "ci_contains_point": contains,
            "collapsed": bool(reasons), "collapsed_reasons": reasons,
            "paired_within_cover": paired_win_rate(y, s, groups)}


def fmt(label, e):
    ci = "collapsed (constant scores)" if e["collapsed"] else f"CI95={e['ci95']}"
    if not e["ci_contains_point"]:
        ci += " (WARNING: CI excludes the point estimate)"
    pw = e["paired_within_cover"]
    return (f"[{label}] AUC={e['auc']:.4f}  {ci}  score_std={e['score_std']:.2e}  "
            f"P(stego>cover|same cover)={pw['p_stego_gt_cover']} CI95={pw['ci95']}")


def det_entry(alpha, use_srm, workers):
    d = C.dirs_for(alpha)
    te = make_loader(d["test"], names, C.BATCH_SIZE, binary_cover_idx=cov, shuffle=False,
                     num_workers=workers)
    name = C.alpha_tag(alpha) + ("" if use_srm else "_nosrm") + f"_s{C.SEED}"
    m = BinaryCNN(in_ch=C.CHANNELS, use_srm=use_srm, dropout=0.0)
    m.load_state_dict(torch.load(C.MODEL_DIR / f"pt_binary_{name}.pt", map_location=DEVICE))
    m.eval().to(DEVICE)
    s, y = _predict(m, te, "binary")
    return auc_entry(y, s, source_ids(te))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    a0 = C.ACTIVE_ALPHA
    out = {"seed": C.SEED, "collapse_rule": ("collapsed if score std < 1e-3 or CI width < 1e-3; "
                                             "a CI that excludes the point estimate is flagged "
                                             "separately (ci_contains_point=false)"),
           "detection_sweep": {}, "gbras_per_algo": {}}
    for a in C.PAYLOADS:
        e = det_entry(a, True, args.workers)
        out["detection_sweep"][f"{a}_bpp"] = e
        print(fmt(f"det {a} bpp", e))
    e = det_entry(a0, False, args.workers)
    out["detection_sweep"][f"{a0}_bpp_noSRM"] = e
    print(fmt(f"det {a0} noSRM", e))

    d = C.dirs_for(a0)
    for algo in [c for c in names if c != C.COVER_NAME]:
        m = load_gbras(C.MODEL_DIR / f"pt_gbras_{algo}_{C.alpha_tag(a0)}_s{C.SEED}.pt", 2, DEVICE)
        dl = make_block_loader(d["test"], C.COVER_NAME, algo, 64, shuffle=False,
                               num_workers=args.workers)
        P, y = per_image_probs(m, dl, DEVICE)
        e = auc_entry(y, P[:, 1], source_ids(dl))
        out["gbras_per_algo"][algo] = e
        print(fmt(f"GBRAS {algo:9s}", e))

    res = C.RESULTS_DIR
    res.mkdir(parents=True, exist_ok=True)
    (res / "spatial_bootstrap.json").write_text(json.dumps(out, indent=2))
    print("\nsaved:", res / "spatial_bootstrap.json")


if __name__ == "__main__":
    main()
