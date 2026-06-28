"""
[V3] Build DISJOINT train / fusion / test splits from a dataset folder.

KEY: the split is made by **cover ID**, not by individual image. Each cover produces
1 cover + N stego that SHARE content; if a cover went to train and its stego to test,
there would be content leakage. Here all variants of a cover go to the SAME split.

Uses symlinks (no copy) to avoid duplicating GB. Resulting layout:

    <DATASET>_SPLITS/
        train/  {COVER,HILL,MiPOD,S-UNIWARD,WOW}/...
        fusion/ {...}
        test/   {...}

Usage:  python make_splits.py            # 70/15/15 by default
"""
import argparse
import os
import random
from pathlib import Path

import config as C


def build_split(src, out, ratios, seed):
    src = Path(src)
    out = Path(out)
    classes = [d.name for d in sorted(src.iterdir()) if d.is_dir()]
    assert C.COVER_NAME in classes, f"Cover class {C.COVER_NAME} not found in {src}"

    # images: .png (spatial) or .jpg/.jpeg (JPEG)
    def _imgs(folder):
        ps = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            ps += folder.glob(ext)
        return ps
    # cover IDs = stems present in the COVER folder
    stems = sorted(p.stem for p in _imgs(src / C.COVER_NAME))
    random.Random(seed).shuffle(stems)
    r_tr, r_fu, r_te = (float(x) for x in ratios.split(","))
    n = len(stems)
    n_tr = int(n * r_tr)
    n_fu = int(n * r_fu)
    split_of = {}
    for i, s in enumerate(stems):
        split_of[s] = "train" if i < n_tr else ("fusion" if i < n_tr + n_fu else "test")

    counts = {"train": 0, "fusion": 0, "test": 0}
    for cls in classes:
        for img in _imgs(src / cls):
            sp = split_of.get(img.stem)
            if sp is None:
                continue  # stego without a matching cover (should not happen)
            dst_dir = out / sp / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            link = dst_dir / img.name
            if not link.exists():
                os.symlink(img.resolve(), link)   # absolute symlink
            counts[sp] += 1

    print(f"Covers: {n}  ->  train={n_tr}, fusion={n_fu}, test={n - n_tr - n_fu} (by ID)")
    print("Images linked per split:", counts)
    print("Splits in:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",    default=None, help="source dataset (default: config.DATA_ROOT)")
    ap.add_argument("--out",    default=None, help="splits destination (default: config.SPLIT_ROOT)")
    ap.add_argument("--alpha",  type=float, default=None, help="single payload")
    ap.add_argument("--alphas", default=None,
                    help="sweep: comma-separated payloads (e.g. 0.1,0.2,0.4)")
    ap.add_argument("--ratios", default="0.7,0.15,0.15", help="train,fusion,test")
    ap.add_argument("--seed",   type=int, default=C.SEED)
    args = ap.parse_args()

    if args.alphas:                                  # sweep: one split set per payload
        for a in (float(x) for x in args.alphas.split(",")):
            build_split(C.data_root(a), C.split_root(a), args.ratios, args.seed)
    elif args.src and args.out:                      # explicit paths
        build_split(args.src, args.out, args.ratios, args.seed)
    else:                                            # single payload
        a = args.alpha if args.alpha is not None else C.ACTIVE_ALPHA
        build_split(C.data_root(a), C.split_root(a), args.ratios, args.seed)


if __name__ == "__main__":
    main()
