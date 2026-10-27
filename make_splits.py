"""Train / fusion / test splits of a dataset folder, disjoint by cover id (a cover and all its
stego versions go to the same split), written as symlinks under
<DATASET>_SPLITS/{train,fusion,test}/<class>/.

Usage:  python make_splits.py            # 70/15/15 by default
"""
import argparse
import os
import random
import shutil
from pathlib import Path

import config as C


def build_split(src, out, ratios, seed):
    src = Path(src)
    out = Path(out)
    # Clear any previous split tree first. The links are created with
    # `if not link.exists()`, so stale symlinks from an earlier build (e.g. a
    # different cover count or seed) would otherwise survive and place the same
    # cover in more than one split -- silent train/test leakage. Removing the tree
    # first guarantees the partitions are disjoint by cover ID [V3].
    if out.exists():
        shutil.rmtree(out)
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
    # Every class must hold exactly the cover stem set: a partially generated class
    # (e.g. an interrupted embedding run) would otherwise silently produce class
    # imbalance and covers with missing stego, with no error downstream.
    cover_stems = set(stems)
    for cls in classes:
        cls_stems = {p.stem for p in _imgs(src / cls)}
        if cls_stems != cover_stems:
            raise SystemExit(
                f"[make_splits] class '{cls}' is not aligned with COVER: "
                f"{len(cover_stems - cls_stems)} covers without a stego, "
                f"{len(cls_stems - cover_stems)} stego without a cover. "
                f"Regenerate the dataset so every class matches COVER before splitting.")
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
