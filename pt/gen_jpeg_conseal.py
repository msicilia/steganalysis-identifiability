"""JPEG stego generation with conseal reference costs (J-UNIWARD, UERD, nsF5).
Reuses the SAME existing Q75 covers (DATA_JPEG_q75_aXXX/COVER/*.jpg) and writes the
stego into the JUNIWARD/UERD/nsF5 folders with jpeglib. UED and J-MiPOD are not in
conseal and are generated separately with jpeg_embedders.py.

Payload convention: bpnzAC (same as the rest). Seed = int(stem) (reproducible, same as
the local simulators). Resumable: skips stego already written.

Usage:  ../../.venv_conseal/bin/python gen_jpeg_conseal.py --alphas 0.1,0.2,0.4 --limit 2000
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import jpeglib
import conseal as cl
sys.path.append(str(Path(__file__).resolve().parent.parent))   # code/ (config.py)
import config as C

# folder name -> conseal reference simulator
SCHEMES = ("JUNIWARD", "UERD", "nsF5")
_WET = 1e13
_AC = np.ones((8, 8), bool); _AC[0, 0] = False


def _nz_ac(Y0):
    return int(((Y0 != 0) & _AC[None, None]).sum())


def embed_one(name, Y0, qt, sp, alpha, seed):
    """Stego with conseal REFERENCE costs, restricted to AC (DC=wet) so all schemes are
    consistent (bpnzAC convention). Returns int coefficients."""
    if name == "nsF5":                      # F5/nsF5 is AC-only by design
        ys = cl.nsF5.simulate_single_channel(Y0, alpha, seed=seed)
        return ys.astype(Y0.dtype)
    if name == "UERD":
        rp, rm = cl.uerd.compute_cost_adjusted(Y0, qt)
    elif name == "JUNIWARD":
        rp, rm = cl.juniward.compute_cost_adjusted(sp, Y0, qt)
    else:
        raise ValueError(name)
    rp = rp.copy(); rm = rm.copy()
    rp[..., 0, 0] = _WET; rm[..., 0, 0] = _WET          # exclude DC
    delta = cl.simulate.ternary((rp, rm), alpha, _nz_ac(Y0), seed=seed)
    return (Y0 + delta).astype(Y0.dtype)


def gen_for_alpha(alpha, limit, schemes):
    root = C.data_root(alpha)                       # DATA_JPEG_q75_aXXX
    covers = sorted((root / "COVER").glob("*.jpg"))[:limit]
    for s in schemes:
        (root / s).mkdir(parents=True, exist_ok=True)
    print(f"[alpha={alpha}] covers={len(covers)} -> {root.name} | conseal={list(schemes)}", flush=True)
    for i, cov in enumerate(covers):
        stem = cov.stem
        todo = [s for s in schemes if not (root / s / f"{stem}.jpg").exists()]
        if not todo:
            continue
        seed = int(stem) if stem.isdigit() else (abs(hash(stem)) % (2**31))
        im = jpeglib.read_dct(str(cov))
        Y0 = im.Y.copy(); qt = im.qt[0]
        sp = jpeglib.read_spatial(str(cov)).spatial[:, :, 0].astype(np.float64)
        for s in todo:
            ys = embed_one(s, Y0, qt, sp, alpha, seed)
            im.Y = ys
            im.write_dct(str(root / s / f"{stem}.jpg"))
        if (i + 1) % 200 == 0:
            print(f"  [alpha={alpha}] {i+1}/{len(covers)}", flush=True)
    print(f"[alpha={alpha}] done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--alphas", default=None, help="list e.g. 0.1,0.2,0.4")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--schemes", default=",".join(SCHEMES))
    args = ap.parse_args()
    assert C.DOMAIN == "jpeg", "Run with STEG_DOMAIN=jpeg"
    schemes = tuple(args.schemes.split(","))
    alphas = ([float(a) for a in args.alphas.split(",")] if args.alphas
              else [args.alpha if args.alpha is not None else C.ACTIVE_ALPHA])
    for a in alphas:
        gen_for_alpha(a, args.limit, schemes)


if __name__ == "__main__":
    main()
