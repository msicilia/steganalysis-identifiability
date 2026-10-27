"""DCTR features (Holub and Fridrich, 2015) for the JPEG splits: unrounded decompression, the
64 DCT basis filters, per-filter and per-phase histograms with step q and threshold T. Phases
are kept separate (64 x 64 x (T+1) = 20,480 features at T=4). Matrices are saved per split;
convolutions run on the GPU.

Usage:  STEG_DOMAIN=jpeg python extract_dctr.py --alpha 0.4 --q 4 --T 4
"""
import sys, argparse, json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
import jpeglib
from scipy.fft import idctn

import config as C
from data import list_images
from engine import DEVICE


# DCTR quantisation step and truncation threshold. Defined once here; every other
# script imports these rather than repeating the literals.
DCTR_Q = 4.0      # = DCTR.m's q = 8*(2 - QF/50) at QF 75
DCTR_T = 4
DECOMPRESSION = "unrounded: block IDCT of dequantised coefficients + 128 (as DCTR.m)"


def decompress_unrounded(path):
    """Spatial image of a grayscale JPEG WITHOUT rounding or clipping, float32 (H, W): the
    orthonormal block IDCT of the dequantised coefficients + 128, exactly what the reference
    DCTR.m filters. Decoding through a JPEG library instead returns uint8 pixels, rounded and
    clipped, which discards part of the coefficient change the features exist to see.
    (Rounded and clipped, this matches libjpeg's decode to within 1 grey level.)"""
    im = jpeglib.read_dct(str(path))
    Y = im.Y.astype(np.float64)
    q = im.qt[im.quant_tbl_no[0]].astype(np.float64)
    Hb, Wb = Y.shape[:2]
    blocks = idctn(Y * q[None, None], axes=(2, 3), norm="ortho") + 128.0
    return blocks.transpose(0, 2, 1, 3).reshape(Hb * 8, Wb * 8).astype(np.float32)


def load_batch(paths, device):
    """(B,1,H,W) float tensor of unrounded decompressed JPEGs, for dctr_features_batch."""
    return torch.stack([torch.from_numpy(decompress_unrounded(p)) for p in paths])[:, None].to(device)


def dct_basis_filters():
    """64 orthonormal 8x8 DCT-II base filters -> tensor (64,1,8,8)."""
    x = np.arange(8)
    def c(k): return np.sqrt(1/8) if k == 0 else np.sqrt(2/8)
    B = np.zeros((64, 8, 8), np.float32)
    for i in range(8):
        for j in range(8):
            bi = c(i) * np.cos((2*x+1)*i*np.pi/16)
            bj = c(j) * np.cos((2*x+1)*j*np.pi/16)
            B[i*8+j] = np.outer(bi, bj)
    return torch.from_numpy(B)[:, None]          # (64,1,8,8)


@torch.no_grad()
def dctr_features_batch(imgs, W, q, T):
    """imgs: (B,1,H,W) float. Returns (B, 64*64*(T+1)) DCTR features."""
    R = F.conv2d(imgs, W)                          # (B,64,H-7,W-7) undecimated residuals
    Q = torch.clamp(torch.round(R.abs() / q), 0, T).to(torch.int8)   # (B,64,h,w) in 0..T
    B, nf, h, w = Q.shape
    feats = []
    for pr in range(8):
        for pc in range(8):
            Qp = Q[:, :, pr::8, pc::8]             # (B,64,~h/8,~w/8) one phase
            # histogram per (image,filter): count of each bin 0..T
            oneh = F.one_hot(Qp.long(), num_classes=T+1).float()  # (B,64,hh,ww,T+1)
            feats.append(oneh.sum(dim=(2, 3)))     # (B,64,T+1)
    f = torch.stack(feats, dim=1)                  # (B,64phases,64filt,T+1)
    return f.reshape(B, -1).cpu().numpy()          # (B, 64*64*(T+1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--q", type=float, default=DCTR_Q)
    ap.add_argument("--T", type=int, default=DCTR_T)
    ap.add_argument("--bs", type=int, default=64)
    args = ap.parse_args()
    assert C.DOMAIN == "jpeg", "Run with STEG_DOMAIN=jpeg"

    W = dct_basis_filters().to(DEVICE)
    src = C.split_root(args.alpha)                 # ..._SPLITS (jpgs per symbol)
    dst = C.DATASETS / f"{C.DATA_PREFIX}_{C.alpha_tag(args.alpha)}_DCTR"
    dst.mkdir(parents=True, exist_ok=True)
    print(f"DCTR | q={args.q} T={args.T} | dev={DEVICE} | dst={dst.name}")

    for split in ("train", "fusion", "test"):
        paths, labels = [], []
        for idx, cls in enumerate(C.CLASS_FOLDERS):
            for p in list_images(src / split / cls):
                paths.append(p); labels.append(idx)
        X = np.zeros((len(paths), 64*64*(args.T+1)), np.float32)
        for s in range(0, len(paths), args.bs):
            batch = paths[s:s+args.bs]
            X[s:s+len(batch)] = dctr_features_batch(load_batch(batch, DEVICE), W, args.q, args.T)
        y = np.array(labels, np.int64)
        # Cover id per row: a cover and its stego variants share the file stem, and every
        # confidence interval downstream has to resample covers rather than images.
        g = np.array([p.stem for p in paths])
        np.save(dst / f"X_{split}.npy", X); np.save(dst / f"y_{split}.npy", y)
        np.save(dst / f"g_{split}.npy", g)
        print(f"  {split}: X={X.shape} classes={np.bincount(y).tolist()} "
              f"covers={len(np.unique(g))}")
    # The quantisation/truncation settings are part of the feature definition: record them
    # so a cache can never be silently paired with a different (q, T).
    (dst / "params.json").write_text(json.dumps({"q": args.q, "T": args.T,
                                                 "dim": 64*64*(args.T+1),
                                                 "variant": "unmerged 64 phases",
                                                 "decompression": DECOMPRESSION}, indent=2))
    print("done:", dst)


if __name__ == "__main__":
    main()
