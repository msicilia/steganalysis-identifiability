"""Times one logistic-regression fit on the DCTR train split (the six-class fit and, unless
--mono-only, one binary specialist) and extrapolates the runtime of the JPEG analysis
scripts as seconds per row times rows.

Usage:
  STEG_DOMAIN=jpeg uv run python pt/time_logreg.py --alpha 0.4
  STEG_DOMAIN=jpeg uv run python pt/time_logreg.py --alpha 0.4 --subsample 300   # smoke
"""
import sys, argparse, time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
from sklearn.preprocessing import StandardScaler
import config as C
import util


def load_train(alpha, subsample, seed):
    d = util._dctr_dir(alpha)
    X = np.load(d / "X_train.npy", mmap_mode="r"); y = np.load(d / "y_train.npy")
    if subsample and subsample < len(y):
        # stratified, so every class survives a small smoke-test subsample
        rng = np.random.default_rng(seed); k = len(np.unique(y))
        sel = np.sort(np.concatenate([rng.choice(np.where(y == c)[0], subsample // k,
                                                 replace=False) for c in np.unique(y)]))
        return np.asarray(X[sel], np.float64), y[sel]
    return np.asarray(X, np.float64), y


def timed(X, y, tag):
    t = time.perf_counter()
    clf = util.fit_logreg(X, y, tag=tag, verbose=True)
    dt = time.perf_counter() - t
    print(f"  {tag}: n={len(y)} d={X.shape[1]} iters={clf.n_iter_total_} wall={dt:.1f}s "
          f"({dt / max(clf.n_iter_total_, 1) * 1e3:.2f} ms/iter)", flush=True)
    return dt / len(y)


def hms(s):
    return f"{s/3600:.1f} h" if s >= 3600 else f"{s/60:.1f} min"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--subsample", type=int, default=None,
                    help="use N stratified train rows (smoke test; extrapolation then "
                         "scales the per-row cost to the full sizes)")
    ap.add_argument("--mono-only", action="store_true", help="skip the binary timing")
    ap.add_argument("--spec-scheme", default="nsF5", dest="spec_scheme",
                    help="scheme of the timed COVER-vs-one specialist (default nsF5, the "
                         "slowest to converge)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--inner-folds", type=int, default=4, dest="inner_folds")
    ap.add_argument("--kfold-covers", type=int, default=10_000, dest="kfold_covers",
                    help="covers in the jpeg_kfold_cv cache")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert C.DOMAIN == "jpeg", "Run with STEG_DOMAIN=jpeg"

    names = C.CLASS_FOLDERS; K = len(names); S = K - 1; cov = names.index(C.COVER_NAME)
    X, y = load_train(args.alpha, args.subsample, args.seed)
    n_full = len(np.load(util._dctr_dir(args.alpha) / "y_train.npy", mmap_mode="r"))
    X = StandardScaler().fit_transform(X)
    print(f"== fit_logreg timing | alpha={args.alpha} | rows={len(y)} (train split has "
          f"{n_full}) | d={X.shape[1]} | classes={names}", flush=True)

    c_multi = timed(X, y, "mono (multinomial)")
    if args.mono_only:
        # a binary lbfgs step costs ~1/K of a K-class one (gradient over one logit)
        c_bin = c_multi / K
        print(f"  binary cost assumed = multinomial / {K} (not measured)")
    else:
        if args.spec_scheme not in names or args.spec_scheme == C.COVER_NAME:
            ap.error(f"--spec-scheme must be one of {[n for n in names if n != C.COVER_NAME]}")
        a = names.index(args.spec_scheme); m = (y == cov) | (y == a)
        c_bin = timed(X[m], (y[m] == a).astype(int), f"spec:{names[a]} (binary)")

    r = n_full / K                                  # train rows per class
    def cost(fits):
        return sum(n * rows * (c_multi if kind == "multi" else c_bin) for kind, n, rows in fits)

    # Fit counts read off each script (meta/posterior fits are 5-dim: negligible).
    # dctr_bootstrap.py as run by run_all.sh: base fits once (detect, S specialists, mono,
    # meta) + S leave-one-out transfer fits + one adaptive-only transfer fit per adaptive
    # scheme (cover + the other adaptive schemes); its LogReg ceiling row reuses the base fits.
    n_ad = S - len(C.LEGACY_SCHEMES)
    single = [("bin", 1, K*r), ("bin", S, 2*r), ("multi", 1, K*r), ("bin", S, (K-1)*r),
              ("bin", n_ad, n_ad*r)]
    rk = args.kfold_covers * (args.folds - 1) / args.folds          # rows/class, train fold
    rin = rk * (args.inner_folds - 1) / args.inner_folds            # rows/class, inner train
    # cv_predictions, per outer fold and per --parts entry:
    #   detection      : detect (bin, all)
    #   identification : mono (multi, all), S specialists (bin, cover+scheme), inner_folds x S
    #                    OOF specialists (bin, cover+scheme of the inner train), 1 meta
    #   transfer       : S transfer (bin, cover + S-1 schemes)
    per_part = {"detection": ([("bin", 1, K*rk)], 1),
                "identification": ([("multi", 1, K*rk), ("bin", S, 2*rk),
                                    ("bin", args.inner_folds * S, 2*rin)],
                                   1 + S + args.inner_folds * S + 1),
                "transfer": ([("bin", S, (K-1)*rk)], S)}

    def kfold(parts):
        fold = [f for p in parts for f in per_part[p][0]]
        n_fold = sum(per_part[p][1] for p in parts)
        return [(k, n * args.folds, rows) for k, n, rows in fold], n_fold * args.folds

    print("\n== extrapolated LogReg runtime (single process, full data) ==")
    print(f"  dctr_bootstrap.py : {1+S+1+1+S+n_ad:3d} fits  ~{hms(cost(single))}  "
          f"(+ LinearSVC and RandomForest, not estimated)")
    # what run_all.sh runs: detection only at the low payloads, no CV transfer at ID_ALPHA
    for label, parts in (("detection (0.1, 0.2 each)", ["detection"]),
                         ("detection+identification (0.4)", ["detection", "identification"]),
                         ("+ transfer, if enabled", ["transfer"])):
        fits, n = kfold(parts)
        print(f"  jpeg_kfold_cv.py {label:32s}: {n:3d} fits  ~{hms(cost(fits))}")
    print(f"    ({args.folds} folds, inner_folds={args.inner_folds}, {args.kfold_covers} covers)")
    print(f"\n  per-row cost: multinomial {c_multi*1e3:.3f} ms, binary {c_bin*1e3:.3f} ms")
    if args.subsample:
        # at d=20,480 a tiny subsample's time is dominated by the per-iteration d-sized
        # vector work, not by rows, so scaling it per row grossly OVERestimates
        print("  WARNING: measured on a subsample -- smoke test only. Per-row scaling is "
              "invalid here (fixed per-iteration cost dominates, iteration counts differ); "
              "rerun without --subsample for a usable estimate.")


if __name__ == "__main__":
    main()
