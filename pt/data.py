"""
Data loading (PyTorch) for the spatial domain.

  - FULL 512x512 image in GRAYSCALE, WITHOUT rescaling or cropping (rescaling
    would alter the +-1 of the embedding).
  - The train/fusion/test partitions are already DISJOINT by cover ID
    (see make_splits.py); here they are only loaded.

PIL is used directly (without torchvision) to avoid pinning versions on aarch64.
"""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

IMG_EXTS = ("*.png", "*.jpg", "*.jpeg")


def list_images(folder):
    """Images from a folder (spatial png or JPEG jpg), stable order by name."""
    folder = Path(folder)
    paths = []
    for ext in IMG_EXTS:
        paths += folder.glob(ext)
    return sorted(paths, key=lambda p: p.name)


def list_samples(split_dir, class_names):
    """List (path, label_idx) with the class ORDER fixed by class_names."""
    split_dir = Path(split_dir)
    items = []
    for idx, cls in enumerate(class_names):
        for p in list_images(split_dir / cls):
            items.append((str(p), idx))
    return items


class StegoDataset(Dataset):
    """Grayscale 512x512 PNG images -> tensor (1,H,W) float32 in [0,255]."""
    def __init__(self, samples, binary_cover_idx=None):
        self.samples = samples
        self.binary_cover_idx = binary_cover_idx   # if not None -> label 0/1

    def __len__(self):
        return len(self.samples)

    def _label(self, idx):
        if self.binary_cover_idx is None:
            return idx
        return 0 if idx == self.binary_cover_idx else 1

    def __getitem__(self, i):
        path, idx = self.samples[i]
        img = Image.open(path).convert("L")               # grayscale
        x = torch.from_numpy(np.asarray(img, dtype=np.float32))[None]   # (1,H,W), 0..255
        return x, self._label(idx)


def make_loader(split_dir, class_names, batch_size, *, binary_cover_idx=None,
                shuffle=False, num_workers=8, limit_batches=None):
    samples = list_samples(split_dir, class_names)
    ds = StegoDataset(samples, binary_cover_idx=binary_cover_idx)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=True,
                        persistent_workers=(num_workers > 0), drop_last=False)
    return loader


def labels_of(loader):
    """Label vector of the loader (for class weights / counts)."""
    ds = loader.dataset
    return np.array([ds._label(idx) for _, idx in ds.samples], dtype=np.int64)


# --------------------------------------------------------------------------- #
# DCT domain (JPEG): .npy cubes (64 channels = DCT modes, 64x64 grid)
# --------------------------------------------------------------------------- #
class DCTDataset(Dataset):
    """DCT .npy cubes (64,Hb,Wb) int16 -> float32 tensor. Label like StegoDataset."""
    def __init__(self, samples, binary_cover_idx=None):
        self.samples = samples
        self.binary_cover_idx = binary_cover_idx

    def __len__(self):
        return len(self.samples)

    def _label(self, idx):
        if self.binary_cover_idx is None:
            return idx
        return 0 if idx == self.binary_cover_idx else 1

    def __getitem__(self, i):
        path, idx = self.samples[i]
        x = np.load(path).astype(np.float32)          # (64, Hb, Wb)
        return torch.from_numpy(x), self._label(idx)


def list_npy(folder):
    return sorted(Path(folder).glob("*.npy"), key=lambda p: p.name)


def make_dct_loader(split_dir, class_names, batch_size, *, binary_cover_idx=None,
                    shuffle=False, num_workers=8):
    """split_dir = .../DATA_JPEG_*_DCT/{train,fusion,test}. Reads .npy cubes."""
    split_dir = Path(split_dir)
    samples = []
    for idx, cls in enumerate(class_names):
        for p in list_npy(split_dir / cls):
            samples.append((str(p), idx))
    ds = DCTDataset(samples, binary_cover_idx=binary_cover_idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=(num_workers > 0), drop_last=False)


# ---------------------------------------------------------------------------
# Non-overlapping 256x256 blocks (GBRAS-Net protocol)
# ---------------------------------------------------------------------------
class BlockDataset(Dataset):
    """Each 512 image -> 4 blocks of 256 (deterministic crop, NO resize)."""
    def __init__(self, samples):
        # expands to (path, label, block_idx 0..3)
        self.index = [(p, y, b) for (p, y) in samples for b in range(4)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        path, y, b = self.index[i]
        img = np.asarray(Image.open(path).convert("L"), dtype=np.float32)   # (512,512)
        r, c = (b // 2) * 256, (b % 2) * 256
        block = img[r:r + 256, c:c + 256]
        return torch.from_numpy(block.copy())[None], y     # (1,256,256)


def make_block_loader(split_dir, cover_name, algo, batch_size, *, shuffle=False,
                      num_workers=8):
    """COVER(0) vs `algo`(1) in 256 blocks."""
    split_dir = Path(split_dir)
    cov = [(str(p), 0) for p in list_images(split_dir / cover_name)]
    stg = [(str(p), 1) for p in list_images(split_dir / algo)]
    ds = BlockDataset(cov + stg)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=(num_workers > 0))


def make_block_loader_multi(split_dir, class_names, batch_size, *, binary_cover_idx=None,
                            balance_binary=False, seed=0, shuffle=False, num_workers=8):
    """256 blocks with MULTICLASS label (index in class_names) or BINARY
    (binary_cover_idx → 0 cover / 1 stego). Stable order (4 consecutive blocks
    per image) if shuffle=False → allows aggregating per image afterwards.

    balance_binary: in binary mode (cover-vs-all), subsamples the stego (which
    has k× images, k=number of algorithms) to match the number of covers → 1:1.
    Use only in TRAIN."""
    split_dir = Path(split_dir)
    samples = []
    for idx, cls in enumerate(class_names):
        lbl = idx if binary_cover_idx is None else (0 if idx == binary_cover_idx else 1)
        for p in list_images(split_dir / cls):
            samples.append((str(p), lbl))
    if balance_binary and binary_cover_idx is not None:
        cov = [s for s in samples if s[1] == 0]
        stg = [s for s in samples if s[1] == 1]
        if len(stg) > len(cov):
            rng = np.random.default_rng(seed)
            keep = sorted(rng.choice(len(stg), size=len(cov), replace=False).tolist())
            stg = [stg[i] for i in keep]
        samples = cov + stg
    ds = BlockDataset(samples)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=(num_workers > 0))
