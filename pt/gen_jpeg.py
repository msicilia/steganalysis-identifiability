"""JPEG dataset generation: BOSSbase covers compressed to quality Q (COVER) and stego from the
local simulators of jpeg_embedders.py (UED, J-MiPOD) at payloads in bpnzAC. Output:
DATA_JPEG_q75_a<alpha>/{COVER,UED,JMIPOD}/<id>.jpg.

Run before gen_jpeg_conseal.py (covers are created only here). Data from earlier versions
must be deleted first: seeds are now per (cover, scheme, payload) and covers and stego share
one container writer.

Usage:  python gen_jpeg.py --embedders UED,JMIPOD --alphas 0.1,0.2,0.4
"""
import sys, os, argparse, glob, tempfile
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
import jpeglib

import config as C
from jpeg_embedders import EMBEDDERS, embed_seed, write_jpeg, claim_dataset_root

COVERS_DIR = C.DATASETS / "covers" / "BOSSbase_1.01"


def atag(a): return f"a{int(round(a*100)):03d}"


def make_cover(pgm, cover_jpg, quality):
    """PIL compresses; jpeglib rewrites so the cover goes through the same writer as the
    stego. The rewrite is lossless (coefficients copied), which is asserted."""
    # temp file outside COVER/: an interrupted run must not leave a stray *.jpg there
    fd, tmp = tempfile.mkstemp(suffix=".jpg"); os.close(fd)
    try:
        Image.open(pgm).convert("L").save(tmp, quality=quality)
        ref = jpeglib.read_dct(tmp)
        Y, qt = ref.Y.copy(), ref.qt.copy()
        px = np.asarray(Image.open(tmp).convert("L"))
        part = cover_jpg.with_suffix(".part")         # atomic: no half-written cover
        write_jpeg(ref, part)
        chk = jpeglib.read_dct(str(part))
        assert np.array_equal(chk.Y, Y) and np.array_equal(chk.qt, qt), \
            "cover rewrite changed DCT"
        assert np.array_equal(np.asarray(Image.open(part).convert("L")), px), \
            "cover rewrite changed decoded pixels"
        part.replace(cover_jpg)
    finally:
        os.unlink(tmp)


def gen_one(pgm, out_root, quality, alpha, embedders):
    """Generate COVER + stego for one image. Returns a dict of change rates."""
    stem = Path(pgm).stem
    # COVER JPEG
    cover_dir = out_root / "COVER"; cover_dir.mkdir(parents=True, exist_ok=True)
    cover_jpg = cover_dir / f"{stem}.jpg"
    if not cover_jpg.exists():
        make_cover(pgm, cover_jpg, quality)

    # Resumable: only (re)generate missing stego.
    todo = {n: f for n, f in embedders.items()
            if not (out_root / n / f"{stem}.jpg").exists()}
    if not todo:
        return {}                                       # cover + stego already done

    im = jpeglib.read_dct(str(cover_jpg))
    Yn = im.Y.astype(np.int32); q = im.qt[0].astype(np.float64)
    n_nz = int(((Yn != 0) & (np.arange(64).reshape(8, 8) != 0)[None, None]).sum())
    # decoded cover (for J-MiPOD), through the same decoder gen_jpeg_conseal.py uses for J-UNIWARD
    spatial = jpeglib.read_spatial(str(cover_jpg)).spatial[:, :, 0]

    rates = {}
    for name, fn in todo.items():
        Ys, nch = fn(Yn, q, alpha, seed=embed_seed(stem, name, alpha), spatial=spatial)
        d = out_root / name; d.mkdir(parents=True, exist_ok=True)
        im.Y = Ys.astype(im.Y.dtype)
        write_jpeg(im, d / f"{stem}.jpg")
        rates[name] = nch / max(n_nz, 1)
    return rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", type=int, default=75)
    ap.add_argument("--alphas", default="0.4", help="bpnzAC payloads, comma-separated")
    ap.add_argument("--limit", type=int, default=None, help="number of covers (default: all)")
    ap.add_argument("--embedders", default="UED,JMIPOD")
    ap.add_argument("--covers", default=str(COVERS_DIR),
                    help="covers dir (default BOSSbase)")
    ap.add_argument("--ext", default="pgm",
                    help="covers extension (pgm/jpg/tif/...)")
    ap.add_argument("--name", default="DATA_JPEG",
                    help="output dataset prefix (e.g. DATA_JPEG_ALASKA2)")
    args = ap.parse_args()

    covers = sorted(glob.glob(str(Path(args.covers) / f"*.{args.ext}")))[:args.limit]
    if not covers:
        sys.exit(f"no *.{args.ext} covers found in {args.covers}")
    embedders = {k: EMBEDDERS[k] for k in args.embedders.split(",")}
    print(f"covers={len(covers)} ({args.covers})  Q={args.quality}  embedders={list(embedders)}")

    total_skip = 0
    for alpha in (float(x) for x in args.alphas.split(",")):
        out_root = C.DATASETS / f"{args.name}_q{args.quality}_{atag(alpha)}"
        # refuses data from earlier code; marks a fresh folder so an interrupted run resumes
        claim_dataset_root(out_root, create=True)
        print(f"\n== generating {out_root.name} (alpha={alpha} bpnzAC) ==")
        agg = {k: [] for k in embedders}
        n_skip = 0
        for i, pgm in enumerate(covers):
            try:
                r = gen_one(pgm, out_root, args.quality, alpha, embedders)
                for k, v in r.items():
                    agg[k].append(v)
            except Exception as e:
                n_skip += 1
                print(f"  [skip {Path(pgm).stem}] {str(e)[:60]}")
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(covers)}")
        for k, vs in agg.items():
            if vs:
                print(f"  {k}: mean change rate={np.mean(vs):.4f} (n={len(vs)})")
        print(f"  done: {out_root}" + (f"  ({n_skip} covers SKIPPED on error)" if n_skip else ""))
        total_skip += n_skip
    if total_skip:
        # a skipped cover leaves the classes misaligned; fail here, not hours later in make_splits
        sys.exit(f"{total_skip} cover(s) failed; rerun to retry them (existing files are kept)")


if __name__ == "__main__":
    main()
