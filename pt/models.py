"""PyTorch models: SRM residual front end (fixed 5x5 high-pass kernels + TLU), BinaryCNN
(cover/stego), SpecialistCNN (multiclass), GBRASNet (30 SRM kernels, Tanh3, separable
blocks) and MetaMLP. Inputs are full 512x512 greyscale images without rescaling or /255.
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Residual front-end (fixed high-pass, lightweight SRM style: 7 filters)
# ---------------------------------------------------------------------------
def _srm7_kernels() -> np.ndarray:
    """5x5 bank of 7 high-pass filters (zero sum). -> (7,1,5,5)."""
    K = []
    kv = np.array([[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]], np.float32) / 12.0
    K.append(kv)

    def pad3(k):
        z = np.zeros((5, 5), np.float32); z[1:4, 1:4] = k; return z
    K.append(pad3(np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], np.float32)))   # 1st order H
    K.append(pad3(np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], np.float32)))   # 1st order V
    K.append(pad3(np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]], np.float32)))   # 1st order D
    K.append(pad3(np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], np.float32)))   # 2nd order H
    K.append(pad3(np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], np.float32)))   # 2nd order V
    K.append(pad3(np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], np.float32)) / 4.0)
    return np.stack(K, axis=0)[:, None, :, :]    # (7,1,5,5)


class SRMFrontEnd(nn.Module):
    """FIXED (non-trainable) high-pass Conv2d + TLU clip to [-T, T]."""
    def __init__(self, T: float = 3.0):
        super().__init__()
        w = torch.from_numpy(_srm7_kernels())             # (7,1,5,5)
        self.register_buffer("weight", w)
        self.T = T

    def forward(self, x):
        x = F.conv2d(x, self.weight, padding=2)           # fixed, no bias
        return torch.clamp(x, -self.T, self.T)


class SeparableConv2d(nn.Module):
    """Depthwise (with depth_multiplier) + pointwise, like Keras SeparableConv2D."""
    def __init__(self, in_ch, out_ch, k=3, padding=1, depth_multiplier=1, bias=False):
        super().__init__()
        mid = in_ch * depth_multiplier
        self.dw = nn.Conv2d(in_ch, mid, k, padding=padding, groups=in_ch, bias=bias)
        self.pw = nn.Conv2d(mid, out_ch, 1, bias=bias)

    def forward(self, x):
        return self.pw(self.dw(x))


def _conv_bn_relu(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))


def _sepconv_bn_relu_pool(in_ch, out_ch):
    return nn.Sequential(
        SeparableConv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.MaxPool2d(2))


class _Backbone(nn.Module):
    """Common trunk: stride-2 conv + N separable blocks with maxpool."""
    def __init__(self, in_ch, channels=(16, 32, 64, 128)):
        super().__init__()
        layers = [_conv_bn_relu(in_ch, channels[0], stride=2)]
        prev = channels[0]
        for c in channels[1:]:
            layers.append(_sepconv_bn_relu_pool(prev, c)); prev = c
        self.body = nn.Sequential(*layers)
        self.out_ch = prev

    def forward(self, x):
        return self.body(x)


class BinaryCNN(nn.Module):
    """Expert A: binary cover/stego detector. Returns 1 logit (no sigmoid)."""
    def __init__(self, in_ch=1, use_srm=True, dropout=0.5):
        super().__init__()
        self.use_srm = use_srm
        self.srm = SRMFrontEnd() if use_srm else None
        feat_in = 7 if use_srm else in_ch
        self.backbone = _Backbone(feat_in, (16, 32, 64, 128))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(self.backbone.out_ch, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x):
        # Without SRM: normalize to [0,1] (ablation branch). With SRM: pixel values.
        x = self.srm(x) if self.use_srm else (x / 255.0)
        return self.head(self.backbone(x)).squeeze(1)     # (B,)


class SpecialistCNN(nn.Module):
    """Expert C / family classifier: backbone + GAP -> Dense, softmax logits.

    Head with Global Average Pooling (not Flatten). Same head as BinaryCNN.
    """
    def __init__(self, num_classes, in_ch=1, use_srm=True, dropout=0.5, img_size=512):
        super().__init__()
        self.use_srm = use_srm
        self.srm = SRMFrontEnd() if use_srm else None
        feat_in = 7 if use_srm else in_ch
        self.backbone = _Backbone(feat_in, (16, 32, 64, 128, 256))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(self.backbone.out_ch, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.srm(x) if self.use_srm else (x / 255.0)
        return self.head(self.backbone(x))                # logits (B,num_classes)


# ---------------------------------------------------------------------------
# GBRAS-Net (Tabares-Soto et al., 2021) — faithful port
# ---------------------------------------------------------------------------
_SRM30_PATH = Path(__file__).resolve().parent.parent / "SRM_Kernels1.npy"


def tanh3(x):
    return torch.tanh(x) * 3.0


def _bn(ch):
    # Original Keras: BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False).
    # Keras momentum weights the OLD running stat, torch momentum weights the NEW batch
    # stat, so Keras 0.2 == torch 0.8. "scale=False" -> affine weight fixed at 1.
    m = nn.BatchNorm2d(ch, momentum=0.8, eps=1e-3, affine=True)
    nn.init.ones_(m.weight); m.weight.requires_grad_(False)   # center=True, scale=False
    return m


class _SepELU(nn.Module):
    def __init__(self, in_ch, out_ch, dm=3):
        super().__init__()
        self.sep = SeparableConv2d(in_ch, out_ch, 3, padding=1, depth_multiplier=dm, bias=True)

    def forward(self, x):
        return F.elu(self.sep(x))


class GBRASNet(nn.Module):
    """Input (B,1,256,256). Output: logits (B,2)."""
    def __init__(self, n_classes=2):
        super().__init__()
        srm = np.load(_SRM30_PATH).astype("float32")          # (5,5,1,30)
        w = torch.from_numpy(np.transpose(srm, (3, 2, 0, 1)).copy())   # (30,1,5,5)
        self.srm = nn.Conv2d(1, 30, 5, padding=2, bias=True)
        with torch.no_grad():
            self.srm.weight.copy_(w); self.srm.bias.fill_(1.0)
        self.srm.weight.requires_grad_(False); self.srm.bias.requires_grad_(False)
        self.bn0 = _bn(30)
        # block 30 with skip
        self.dw1 = nn.Conv2d(30, 30, 1, groups=30)
        self.s1a = _SepELU(30, 30); self.bn1 = _bn(30)
        self.dw2 = nn.Conv2d(30, 30, 1, groups=30)
        self.s1b = _SepELU(30, 30); self.bn2 = _bn(30)
        self.c1 = nn.Conv2d(30, 30, 3, padding=1); self.bn3 = _bn(30)
        self.c2 = nn.Conv2d(30, 30, 3, padding=1); self.bn4 = _bn(30)
        # block 60 with skip
        self.c3 = nn.Conv2d(30, 60, 3, padding=1); self.bn5 = _bn(60)
        self.dw3 = nn.Conv2d(60, 60, 1, groups=60)
        self.s2a = _SepELU(60, 60); self.bn6 = _bn(60)
        self.dw4 = nn.Conv2d(60, 60, 1, groups=60)
        self.s2b = _SepELU(60, 60); self.bn7 = _bn(60)
        self.c4 = nn.Conv2d(60, 60, 3, padding=1); self.bn8 = _bn(60)
        self.c5 = nn.Conv2d(60, 60, 3, padding=1); self.bn9 = _bn(60)
        self.c6 = nn.Conv2d(60, 60, 3, padding=1); self.bn10 = _bn(60)
        self.c7 = nn.Conv2d(60, 30, 1); self.bn11 = _bn(30)
        self.c8 = nn.Conv2d(30, n_classes, 1); self.bn12 = _bn(n_classes)
        self.pool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        x = tanh3(self.srm(x)); l1 = self.bn0(x)
        x = self.dw1(l1); x = self.bn1(self.s1a(x))
        x = self.dw2(x);  l2 = self.bn2(self.s1b(x))
        s1 = l1 + l2
        x = F.elu(self.c1(s1)); x = self.bn3(x)
        x = F.elu(self.c2(x));  x = self.bn4(x)
        x = self.pool(x)
        x = F.elu(self.c3(x)); l3 = self.bn5(x)
        x = self.dw3(l3); x = self.bn6(self.s2a(x))
        x = self.dw4(x);  l4 = self.bn7(self.s2b(x))
        s2 = l3 + l4
        x = F.elu(self.c4(s2)); x = self.bn8(x); x = self.pool(x)
        x = F.elu(self.c5(x));  x = self.bn9(x); x = self.pool(x)
        x = F.elu(self.c6(x));  x = self.bn10(x); x = self.pool(x)
        x = F.elu(self.c7(x));  x = self.bn11(x)
        x = F.elu(self.c8(x));  x = self.bn12(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return x                                            # logits (B,n_classes)


class MetaMLP(nn.Module):
    """Meta-model (level 2): MLP that fuses the experts' posteriors."""
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(64, 64), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(64, num_classes))

    def forward(self, x):
        return self.net(x)


class DCTNet(nn.Module):
    """Detector in the DCT domain (JPEG). Input: cube (64, Hb, Wb) = 64 DCT modes.
    The ±1 changes of the stego are directly visible in the coefficients, so the
    SRM front-end is NOT used (that is spatial). The input is normalized per channel
    (BatchNorm) and conv blocks are applied over the grid of blocks.
    out_dim=1 -> binary (1 logit, BCEWithLogits); out_dim=K -> multiclass (CE)."""
    def __init__(self, out_dim, in_ch=64, dropout=0.5, trunc=8.0):
        super().__init__()
        self.trunc = trunc                                 # coefficient clipping
        self.in_bn = nn.BatchNorm2d(in_ch)
        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
                nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            block(64, 64), block(64, 128), block(128, 256))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(128, out_dim))
        self.out_dim = out_dim

    def forward(self, x):
        if self.trunc:                                     # TLU in DCT domain: focuses
            x = torch.clamp(x, -self.trunc, self.trunc)    # the range where the ±1 lives
        z = self.head(self.body(self.in_bn(x)))
        return z.squeeze(1) if self.out_dim == 1 else z
