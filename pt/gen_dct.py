"""Quantised DCT coefficients of the JPEG splits as .npy cubes (64, Hb, Wb), int16, for the
DCT-domain CNN of train_dct.py. Reads with jpeglib.

Usage:  STEG_DOMAIN=jpeg uv run python pt/gen_dct.py --alpha 0.4
"""
import sys, argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import jpeglib
import config as C


def jpg_to_cube(path):
    """JPEG -> (64, Hb, Wb) int16: 64 channels (DCT modes) over the block grid."""
    y = jpeglib.read_dct(str(path)).Y                  # (Hb, Wb, 8, 8)
    cube = np.transpose(y, (2, 3, 0, 1)).reshape(64, y.shape[0], y.shape[1])
    return cube.astype(np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.4)
    args = ap.parse_args()
    assert C.DOMAIN == "jpeg", "Run with STEG_DOMAIN=jpeg"

    src_root = C.split_root(args.alpha)                       # ..._SPLITS
    dst_root = C.DATASETS / f"{C.DATA_PREFIX}_{C.alpha_tag(args.alpha)}_DCT"
    print(f"src={src_root}\ndst={dst_root}\nclasses={C.CLASS_FOLDERS}")

    total = done = 0
    for split in ("train", "fusion", "test"):
        for cls in C.CLASS_FOLDERS:
            sdir = src_root / split / cls
            ddir = dst_root / split / cls; ddir.mkdir(parents=True, exist_ok=True)
            jpgs = sorted(sdir.glob("*.jpg"))
            for p in jpgs:
                total += 1
                out = ddir / f"{p.stem}.npy"
                # resumable, but a cube older than its JPEG is stale (the JPEGs were
                # regenerated with new seeds/container) and must be rebuilt
                if out.exists() and out.stat().st_mtime >= p.resolve().stat().st_mtime:
                    continue
                np.save(out, jpg_to_cube(p)); done += 1
            print(f"  {split}/{cls}: {len(jpgs)} jpgs")
    print(f"done: {done} new cubes out of {total} images -> {dst_root}")


if __name__ == "__main__":
    main()
