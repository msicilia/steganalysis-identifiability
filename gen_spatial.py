"""
Spatial-domain stego generation with conseal (pure Python). Single source (BOSSbase),
same source for covers and stego [V2].

For each BOSSbase cover, generate its stego variant with HILL, WOW, S-UNIWARD, MiPOD
and HUGO at the same payload (bpp). Output layout (class folders):

    OUT_DIR/
        COVER/      <- original cover (copied as lossless PNG)
        HILL/
        WOW/
        S-UNIWARD/
        MiPOD/

CRITICAL:
  * Saved as PNG (LOSSLESS). Saving as JPEG would destroy the LSB embedding.
  * Same cover -> all its variants share content. To avoid train/test leakage by
    content, make_splits.py must split by cover ID (not by individual image).
  * Per-image seed for reproducibility [V4].

Usage:
    python gen_spatial.py --limit 2000 --alpha 0.4
    python gen_spatial.py                # all covers, default alpha
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import conseal as cl

# Spatial algorithms -> conseal simulator function
ALGORITHMS = {
    "HILL":      cl.hill.simulate_single_channel,
    "WOW":       cl.wow.simulate_single_channel,
    "S-UNIWARD": cl.suniward.simulate_single_channel,
    "MiPOD":     cl.mipod.simulate_single_channel,
    "HUGO":      cl.hugo.simulate_single_channel,
}
COVER_CLASS = "COVER"


def load_pgm(path):
    """Read a grayscale PGM as np.uint8 [H, W]."""
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def save_png(arr, path):
    """Save np.uint8 [H,W] as PNG (lossless)."""
    Image.fromarray(np.squeeze(arr).astype(np.uint8), mode="L").save(path)


def embed(fn, x0, alpha, seed):
    """Simulate the embedding. Some simulators accept seed; fall back otherwise."""
    try:
        x1 = fn(x0, alpha, seed=seed)
    except TypeError:
        x1 = fn(x0, alpha)   # in case a signature does not accept seed
    return np.squeeze(np.asarray(x1)).astype(np.uint8)


def generate(covers, out, alpha, algos, seed):
    """Generate COVER + stego of `algos` for a list of covers, at one payload."""
    algo_idx = {name: i for i, name in enumerate(ALGORITHMS)}  # fixed j per algorithm
    for cls in [COVER_CLASS, *algos]:
        (out / cls).mkdir(parents=True, exist_ok=True)
    print(f"[alpha={alpha}] covers={len(covers)} | algos={list(algos)} | -> {out}")
    for i, cover_path in enumerate(covers):
        stem = cover_path.stem
        x0 = load_pgm(cover_path)
        cover_out = out / COVER_CLASS / f"{stem}.png"      # lossless PNG
        if not cover_out.exists():
            save_png(x0, cover_out)
        for name, fn in algos.items():
            stego_out = out / name / f"{stem}.png"
            if stego_out.exists():
                continue                                    # resumable: keep existing stego
            s = seed + i * 10 + algo_idx[name]              # deterministic seed
            save_png(embed(fn, x0, alpha, s), stego_out)
        if (i + 1) % 100 == 0:
            print(f"  [alpha={alpha}] {i+1}/{len(covers)} covers")
    print(f"[alpha={alpha}] done in {out}")


def main():
    import config as C
    ap = argparse.ArgumentParser()
    ap.add_argument("--covers", default=str(C.DATASETS / "covers" / "BOSSbase_1.01"))
    ap.add_argument("--out",    default=None,
                    help="output folder; defaults to config.data_root(alpha)")
    ap.add_argument("--alpha",  type=float, default=C.ACTIVE_ALPHA, help="payload (bpp)")
    ap.add_argument("--alphas", default=None,
                    help="sweep: comma-separated payloads (e.g. 0.1,0.2,0.4). "
                         "Each goes to its own DATA_SPATIAL_aXXX folder.")
    ap.add_argument("--limit",  type=int, default=None, help="number of covers (None=all)")
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--algos",  default=None,
                    help="comma-separated list (e.g. HUGO). Default: all.")
    args = ap.parse_args()

    algos = ALGORITHMS if not args.algos else {
        k: ALGORITHMS[k] for k in args.algos.split(",")}

    covers = sorted(Path(args.covers).glob("*.pgm"))
    if args.limit:
        covers = covers[:args.limit]

    # Sweep (--alphas) or a single payload (--alpha).
    alphas = [float(a) for a in args.alphas.split(",")] if args.alphas else [args.alpha]
    for alpha in alphas:
        out = Path(args.out) if args.out else C.data_root(alpha)
        generate(covers, out, alpha, algos, args.seed)


if __name__ == "__main__":
    main()
