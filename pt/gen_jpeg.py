"""
JPEG dataset generation (DCT domain) on this host (without conseal/MATLAB).

Fully local pipeline:
  BOSSbase PGM (gray) --PIL--> COVER JPEG (quality Q)   [same source]
  --jpeglib.read_dct--> quantised DCT
  --simulators (jpeg_embedders)--> stego coefficients
  --jpeglib.write_dct--> valid stego JPEG

Output layout (compatible with make_splits / the .jpg-aware loader):
  datasets/DATA_JPEG_q75_a040/{COVER,nsF5,UERD,...}/<id>.jpg

Used here for the local simulators (UED, J-MiPOD); J-UNIWARD/UERD/nsF5 use conseal
references (see gen_jpeg_conseal.py). Payload in bpnzAC. Usage:
  python gen_jpeg.py --quality 75 --alphas 0.1,0.2,0.4 --limit 2000 --embedders UED,JMIPOD
"""
import sys, argparse, glob, zlib
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
import jpeglib

import config as C
from jpeg_embedders import EMBEDDERS

COVERS_DIR = C.DATASETS / "covers" / "BOSSbase_1.01"


def atag(a): return f"a{int(round(a*100)):03d}"


def gen_one(pgm, out_root, quality, alpha, embedders):
    """Generate COVER + stego for one image. Returns a dict of change rates."""
    stem = Path(pgm).stem
    # COVER JPEG
    cover_dir = out_root / "COVER"; cover_dir.mkdir(parents=True, exist_ok=True)
    cover_jpg = cover_dir / f"{stem}.jpg"
    if not cover_jpg.exists():
        Image.open(pgm).convert("L").save(cover_jpg, quality=quality)

    # Resumable: only (re)generate missing stego.
    todo = {n: f for n, f in embedders.items()
            if not (out_root / n / f"{stem}.jpg").exists()}
    if not todo:
        return {}                                       # cover + stego already done

    im = jpeglib.read_dct(str(cover_jpg))
    Yn = im.Y.astype(np.int32); q = im.qt[0].astype(np.float64)
    n_nz = int(((Yn != 0) & (np.arange(64).reshape(8, 8) != 0)[None, None]).sum())
    spatial = np.asarray(Image.open(cover_jpg).convert("L"))   # decoded cover (for J-UNIWARD)

    rates = {}
    seed = int(stem) if stem.isdigit() else (zlib.crc32(stem.encode()) & 0x7fffffff)
    for name, fn in todo.items():
        Ys, nch = fn(Yn, q, alpha, seed=seed, spatial=spatial)
        d = out_root / name; d.mkdir(parents=True, exist_ok=True)
        im.Y = Ys.astype(im.Y.dtype)
        im.write_dct(str(d / f"{stem}.jpg"))
        rates[name] = nch / max(n_nz, 1)
    return rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", type=int, default=75)
    ap.add_argument("--alphas", default="0.4", help="bpnzAC payloads, comma-separated")
    ap.add_argument("--limit", type=int, default=2000, help="number of covers")
    ap.add_argument("--embedders", default="nsF5,UERD")
    ap.add_argument("--covers", default=str(COVERS_DIR),
                    help="covers dir (default BOSSbase)")
    ap.add_argument("--ext", default="pgm",
                    help="covers extension (pgm/jpg/tif/...)")
    ap.add_argument("--name", default="DATA_JPEG",
                    help="output dataset prefix (e.g. DATA_JPEG_ALASKA2)")
    args = ap.parse_args()

    covers = sorted(glob.glob(str(Path(args.covers) / f"*.{args.ext}")))[:args.limit]
    embedders = {k: EMBEDDERS[k] for k in args.embedders.split(",")}
    print(f"covers={len(covers)} ({args.covers})  Q={args.quality}  embedders={list(embedders)}")

    for alpha in (float(x) for x in args.alphas.split(",")):
        out_root = C.DATASETS / f"{args.name}_q{args.quality}_{atag(alpha)}"
        print(f"\n== generating {out_root.name} (alpha={alpha} bpnzAC) ==")
        agg = {k: [] for k in embedders}
        for i, pgm in enumerate(covers):
            try:
                r = gen_one(pgm, out_root, args.quality, alpha, embedders)
                for k, v in r.items():
                    agg[k].append(v)
            except Exception as e:
                print(f"  [skip {Path(pgm).stem}] {str(e)[:60]}")
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(covers)}")
        for k, vs in agg.items():
            if vs:
                print(f"  {k}: mean change rate={np.mean(vs):.4f} (n={len(vs)})")
        print(f"  done: {out_root}")


if __name__ == "__main__":
    main()
