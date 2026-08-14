"""
DCTR feature extraction (Holub & Fridrich, 2015) for JPEG steganalysis.
A content-invariant feature representer based on coefficient statistics.

Idea: decompress the JPEG to spatial (PIL, no torchjpeg) -> convolve with the
64 base DCT 8x8 filters (undecimated DCT) -> per filter and per PHASE (position mod 8),
histogram of |residual| quantized (step q) and truncated (threshold T). The phase is key:
the JPEG 8x8 grid creates phase-dependent statistics.

Dim per image: 64 filters x 64 phases x (T+1) bins. Matrices are saved per split.
Convolutions on GPU (torch) for speed; does NOT use torchjpeg.

Usage:  STEG_DOMAIN=jpeg python extract_dctr.py --alpha 0.4 --q 4 --T 4
"""
import sys, argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config as C
from data import list_images
from engine import DEVICE


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
    ap.add_argument("--q", type=float, default=4.0)
    ap.add_argument("--T", type=int, default=4)
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
            imgs = torch.stack([
                torch.from_numpy(np.asarray(Image.open(p).convert("L"), np.float32))
                for p in batch])[:, None].to(DEVICE)
            X[s:s+len(batch)] = dctr_features_batch(imgs, W, args.q, args.T)
        y = np.array(labels, np.int64)
        np.save(dst / f"X_{split}.npy", X); np.save(dst / f"y_{split}.npy", y)
        print(f"  {split}: X={X.shape} classes={np.bincount(y).tolist()}")
    print("done:", dst)


if __name__ == "__main__":
    main()
