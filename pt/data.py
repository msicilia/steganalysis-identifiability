"""PyTorch data loading for the spatial domain: full 512x512 greyscale images without
rescaling, from the by-cover splits of make_splits.py. Uses PIL directly.
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))   # config lives in code/
import config as C

IMG_EXTS = ("*.png", "*.jpg", "*.jpeg")


def _worker_init(worker_id):
    """Give each DataLoader worker a seeded, distinct RNG stream. The datasets here read
    deterministically, so this guards future changes rather than current behaviour."""
    import random
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seed); random.seed(seed)


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


def fusion_subset_stems(split_dir, subset):
    """Cover stems of one half of the FUSION split [V3]: 'select' (checkpoint selection) or
    'meta' (stacking meta-learner training). Disjoint by cover, fixed by config.SEED, and
    derived from the COVER folder, so every script and class gets the same halves."""
    split_dir = Path(split_dir)
    if split_dir.name != "fusion":
        raise ValueError(f"fusion subsets apply only to the fusion split, got {split_dir}")
    if subset not in C.FUSION_SUBSETS:
        raise ValueError(f"subset must be one of {C.FUSION_SUBSETS}, got {subset!r}")
    stems = sorted(p.stem for p in list_images(split_dir / C.COVER_NAME))
    perm = np.random.default_rng(C.SEED).permutation(len(stems))
    half = len(stems) // 2
    return {stems[i] for i in (perm[:half] if subset == "select" else perm[half:])}


def _in_subset(samples, split_dir, subset):
    """Keep the (path, label) samples whose cover belongs to `subset` (None = all)."""
    if subset is None:
        return samples
    keep = fusion_subset_stems(split_dir, subset)
    return [s for s in samples if Path(s[0]).stem in keep]


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
                shuffle=False, num_workers=8, limit_batches=None, subset=None):
    samples = _in_subset(list_samples(split_dir, class_names), split_dir, subset)
    ds = StegoDataset(samples, binary_cover_idx=binary_cover_idx)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=True,
                        persistent_workers=(num_workers > 0), worker_init_fn=_worker_init, drop_last=False)
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
                      persistent_workers=(num_workers > 0), worker_init_fn=_worker_init, drop_last=False)


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
                      num_workers=8, subset=None):
    """COVER(0) vs `algo`(1) in 256 blocks."""
    split_dir = Path(split_dir)
    cov = [(str(p), 0) for p in list_images(split_dir / cover_name)]
    stg = [(str(p), 1) for p in list_images(split_dir / algo)]
    ds = BlockDataset(_in_subset(cov + stg, split_dir, subset))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=(num_workers > 0), worker_init_fn=_worker_init)


class BalancedEpochSampler(torch.utils.data.Sampler):
    """Cover-vs-all training sampler over a BlockDataset: every epoch yields the blocks of
    ALL cover images plus a FRESH random subset of stego images of the same size (1:1),
    shuffled at block level. Over the run the detector sees every stego image, while an
    epoch costs the same as a fixed 1:1 subsample. The RNG advances each epoch (seeded)."""
    def __init__(self, dataset, seed=0):
        labels = np.array([y for _, y, _ in dataset.index[::4]])   # one label per image
        self.cov = np.where(labels == 0)[0]
        self.stg = np.where(labels == 1)[0]
        self.n_stg = min(len(self.cov), len(self.stg))
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return 4 * (len(self.cov) + self.n_stg)

    def __iter__(self):
        imgs = np.concatenate([self.cov, self.rng.choice(self.stg, self.n_stg, replace=False)])
        blocks = (imgs[:, None] * 4 + np.arange(4)[None]).ravel()
        return iter(self.rng.permutation(blocks).tolist())


def make_block_loader_multi(split_dir, class_names, batch_size, *, binary_cover_idx=None,
                            balance_binary=False, resample_binary=False, seed=0,
                            shuffle=False, num_workers=8, subset=None):
    """256 blocks with MULTICLASS label (index in class_names) or BINARY
    (binary_cover_idx → 0 cover / 1 stego). Stable order (4 consecutive blocks
    per image) if shuffle=False → allows aggregating per image afterwards.

    balance_binary: in binary mode (cover-vs-all), subsamples the stego (which
    has k× images, k=number of algorithms) to match the number of covers → 1:1.
    The subset is drawn ONCE with default_rng(seed) and FIXED for the run; use it for the
    selection (fusion) split, where a fixed set keeps the monitored metric comparable
    across epochs. Never for TEST.

    resample_binary: for TRAIN in binary mode. Keeps all images and draws a fresh 1:1
    stego subset every epoch (BalancedEpochSampler), so no stego image is left unseen;
    implies shuffling."""
    split_dir = Path(split_dir)
    samples = []
    for idx, cls in enumerate(class_names):
        lbl = idx if binary_cover_idx is None else (0 if idx == binary_cover_idx else 1)
        for p in list_images(split_dir / cls):
            samples.append((str(p), lbl))
    samples = _in_subset(samples, split_dir, subset)      # before balancing
    if balance_binary and binary_cover_idx is not None:
        cov = [s for s in samples if s[1] == 0]
        stg = [s for s in samples if s[1] == 1]
        if len(stg) > len(cov):
            rng = np.random.default_rng(seed)
            keep = sorted(rng.choice(len(stg), size=len(cov), replace=False).tolist())
            stg = [stg[i] for i in keep]
        samples = cov + stg
    ds = BlockDataset(samples)
    if resample_binary:
        if binary_cover_idx is None or balance_binary:
            raise ValueError("resample_binary needs binary mode and excludes balance_binary")
        return DataLoader(ds, batch_size=batch_size, sampler=BalancedEpochSampler(ds, seed),
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=(num_workers > 0), worker_init_fn=_worker_init)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=(num_workers > 0), worker_init_fn=_worker_init)
