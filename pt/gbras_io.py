"""GBRAS-Net inference helpers: per-image scores and features from the four 256x256 blocks of
each image (loaders must not shuffle).
"""
import numpy as np
import torch
import torch.nn.functional as F
from models import GBRASNet


def load_gbras(path, n_classes, device):
    m = GBRASNet(n_classes=n_classes)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.eval().to(device)


@torch.no_grad()
def per_image_probs(model, block_loader, device):
    """(per-image probs (N,C), per-image labels (N,)). 4 blocks → mean."""
    P, Y = [], []
    for xb, yb in block_loader:
        P.append(F.softmax(model(xb.to(device)), 1).cpu().numpy())
        Y.append(np.asarray(yb))
    P = np.concatenate(P); Y = np.concatenate(Y)
    P = P.reshape(-1, 4, P.shape[1]).mean(1)     # averages the 4 blocks
    Y = Y.reshape(-1, 4)[:, 0]                    # same label in all 4
    return P, Y


@torch.no_grad()
def per_image_feats(model, block_loader, device):
    """Penultimate GBRAS features (output of bn11, 30-d, exactly what c8 receives) with
    GAP, per image."""
    box = {}
    h = model.bn11.register_forward_hook(lambda m, i, o: box.__setitem__("f", o))
    Fs, Y = [], []
    for xb, yb in block_loader:
        model(xb.to(device))
        g = F.adaptive_avg_pool2d(box["f"], 1).flatten(1).cpu().numpy()
        Fs.append(g); Y.append(np.asarray(yb))
    h.remove()
    Fs = np.concatenate(Fs); Y = np.concatenate(Y)
    Fs = Fs.reshape(-1, 4, Fs.shape[1]).mean(1)
    Y = Y.reshape(-1, 4)[:, 0]
    return Fs, Y
