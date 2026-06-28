"""
Home JPEG embedding simulators (DCT domain) over quantised coefficients.

They implement *rate-distortion bound simulation* (the same thing conseal / the DDE .m
files do internally): per-coefficient costs are computed, change probabilities are
obtained via a Boltzmann distribution with lambda tuned to the payload, and the +/-1
change is simulated. No real STC coding is used.

In this pipeline, J-UNIWARD, UERD and nsF5 are generated with conseal reference
implementations; the simulators used from here are UED and J-MiPOD (absent from conseal).
The others are kept for reference and cross-checking.

Schemes:
  - nsF5  : NON-adaptive (uniform cost), binary change (decrements |coef|),
            non-shrinkage (does not create zeros). Legacy.
  - UERD  : adaptive (block-energy cost), ternary +/-1 change.
  - J-UNIWARD : adaptive (wavelet UNIWARD distortion cost), ternary +/-1.
  - UED   : adaptive (coefficient-context energy cost), ternary +/-1.
  - J-MiPOD : adaptive, model-based (spatial-variance cost), ternary +/-1.

Payload in bpnzAC (bits per non-zero AC coefficient), the standard JPEG convention.
Works on Y: tensor (Hb,Wb,8,8) of quantised DCT coefficients (numpy int), and
quant: (8,8) luminance quantisation table.
"""
import numpy as np

_RNG_DEFAULT = 12345


def _binary_entropy(b):
    b = np.clip(b, 1e-12, 1 - 1e-12)
    return -b * np.log2(b) - (1 - b) * np.log2(1 - b)


def _inv_binary_entropy(target):
    """beta in [0,0.5] such that H_b(beta)=target (bisection)."""
    lo, hi = 0.0, 0.5
    for _ in range(50):
        mid = (lo + hi) / 2
        if _binary_entropy(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _ternary_entropy(p):  # p = prob of +1 (= prob of -1); total change 2p
    p = np.clip(p, 1e-12, 1 / 3)
    return -2 * p * np.log2(p) - (1 - 2 * p) * np.log2(1 - 2 * p)


def _ac_mask(shape):
    """(8,8) mask, True on AC (everything except DC[0,0])."""
    m = np.ones((8, 8), bool); m[0, 0] = False; return m


# --------------------------------------------------------------------------- #
def embed_nsf5(Y, quant, alpha, seed=_RNG_DEFAULT, spatial=None):
    """Simulated nsF5. Y:(Hb,Wb,8,8) int. Returns Y_stego and number of changes.
    (spatial is ignored; uniform signature for the generator.)"""
    rng = np.random.default_rng(seed)
    Ys = Y.copy()
    ac = _ac_mask(Y.shape)
    # non-zero AC coefficients (capacity) and the non-shrinkage ones (|c|>=2)
    nz = (Y != 0) & ac[None, None]
    n_nz = int(nz.sum())
    if n_nz == 0:
        return Ys, 0
    beta = _inv_binary_entropy(alpha)            # change rate (uniform cost)
    n_changes = int(round(beta * n_nz))
    # only change coefficients that do NOT create a zero (non-shrinkage)
    usable = (np.abs(Y) >= 2) & ac[None, None]
    idx = np.argwhere(usable)
    if len(idx) == 0:
        return Ys, 0
    n_changes = min(n_changes, len(idx))
    pick = idx[rng.choice(len(idx), size=n_changes, replace=False)]
    for bi, bj, u, v in pick:
        Ys[bi, bj, u, v] -= np.sign(Ys[bi, bj, u, v])   # decrement magnitude
    return Ys, n_changes


def _uerd_costs(Y, quant):
    """UERD per-coefficient cost (Hb,Wb,8,8). Lower cost = more texture."""
    ac = _ac_mask(Y.shape)
    deq = np.abs(Y) * quant[None, None]                 # dequantised magnitude
    e = (deq * ac[None, None]).sum(axis=(2, 3))         # AC energy per block (Hb,Wb)
    # energy of the 8 neighbours (sum) with edge padding
    pad = np.pad(e, 1, mode="edge")
    neigh = (pad[:-2, :-2] + pad[:-2, 1:-1] + pad[:-2, 2:] +
             pad[1:-1, :-2] +                  pad[1:-1, 2:] +
             pad[2:, :-2] + pad[2:, 1:-1] + pad[2:, 2:])
    blk = e + 0.25 * neigh                               # block + neighbour energy
    blk = np.where(blk <= 0, 1e-6, blk)                 # flat block -> high cost
    rho = quant[None, None] / blk[:, :, None, None]      # (Hb,Wb,8,8)
    rho = np.where(ac[None, None], rho, 1e10)            # DC not embeddable
    return rho


def embed_uerd(Y, quant, alpha, seed=_RNG_DEFAULT, spatial=None):
    """Simulated UERD (ternary +/-1, adaptive). (spatial is ignored.)"""
    rng = np.random.default_rng(seed)
    Ys = Y.copy()
    ac = _ac_mask(Y.shape)
    n_nz = int(((Y != 0) & ac[None, None]).sum())
    if n_nz == 0:
        return Ys, 0
    payload_bits = alpha * n_nz
    rho = _uerd_costs(Y, quant).ravel()
    # tune lambda by bisection to reach the payload (sum of ternary entropy = bits)
    def total_entropy(lmbda):
        p = np.exp(-lmbda * rho)
        p = p / (1 + 2 * p)                              # prob of +1 (=of -1)
        return _ternary_entropy(p).sum()
    lo, hi = 1e-6, 1e6
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        if total_entropy(mid) > payload_bits:
            lo = mid                                     # larger lambda -> fewer changes
        else:
            hi = mid
    lmbda = np.sqrt(lo * hi)
    p = np.exp(-lmbda * rho); p = p / (1 + 2 * p)
    # simulate: change with prob 2p; sign +/-1 equiprobable
    r = rng.random(p.shape)
    change = r < (2 * p)
    sign = np.where(rng.random(p.shape) < 0.5, 1, -1)
    delta = np.where(change, sign, 0).reshape(Y.shape).astype(Y.dtype)
    Ys = Y + delta
    return Ys, int(change.sum())


# --------------------------------------------------------------------------- #
# J-UNIWARD (adaptive, UNIWARD wavelet-distortion cost)
# --------------------------------------------------------------------------- #
from scipy.signal import fftconvolve
from scipy.fft import idctn

# Daubechies-8 (db8) low-pass decomposition filter (16 taps, PyWavelets).
_DB8_LO = np.array([
    -0.00011747678400228192, 0.0006754494059985568, -0.0003917403729959771,
    -0.00487035299301066, 0.008746094047015655, 0.013981027917015516,
    -0.04408825393106472, -0.01736930100202211, 0.128747426620186,
    0.00047248457399797254, -0.2840155429624281, -0.015829105256023893,
    0.5853546836548691, 0.6756307362980128, 0.3128715909144659,
    0.05441584224308161], np.float64)
_DB8_HI = _DB8_LO[::-1] * ((-1.0) ** np.arange(len(_DB8_LO)))   # high-pass QMF


def _uniward_filters():
    """3 directional 2D filters (LH, HL, HH) of the UNIWARD wavelet bank."""
    lo, hi = _DB8_LO, _DB8_HI
    return [np.outer(lo, hi), np.outer(hi, lo), np.outer(hi, hi)]


def _dct_basis():
    """64 spatial 8x8 patterns: IDCT of a unit coefficient at each mode (i,j)."""
    B = np.empty((8, 8, 8, 8))
    for i in range(8):
        for j in range(8):
            u = np.zeros((8, 8)); u[i, j] = 1.0
            B[i, j] = idctn(u, norm="ortho")
    return B


_UNI_F = _uniward_filters()
_UNI_B = _dct_basis()


def _juniward_costs(Y, quant, spatial):
    """J-UNIWARD per-coefficient cost (Hb,Wb,8,8). spatial = decoded image."""
    sigma = 2.0 ** -6
    Hb, Wb = Y.shape[:2]
    # cover wavelet band and its stabiliser 1/(sigma+|W_k|)
    xi = [1.0 / (sigma + np.abs(fftconvolve(spatial, F, mode="full"))) for F in _UNI_F]
    rho = np.full((Hb, Wb, 8, 8), 1e10)
    ac = _ac_mask(Y.shape)
    for i in range(8):
        for j in range(8):
            if not ac[i, j]:
                continue
            # wavelet residual of a magnitude change q_ij at mode (i,j)
            patt = quant[i, j] * _UNI_B[i, j]                 # spatial 8x8 change
            cost_map = np.zeros((Hb, Wb))
            for k, F in enumerate(_UNI_F):
                R = np.abs(fftconvolve(patt, F, mode="full"))  # (23,23)
                # block cost b = sum |R| * xi_k aligned -> correlation + subsampling
                corr = fftconvolve(xi[k], R[::-1, ::-1], mode="valid")  # ~ (Himg,Wimg)
                # subsample on the block grid (stride 8), cropping to HbxWb
                sub = corr[::8, ::8][:Hb, :Wb]
                cost_map[:sub.shape[0], :sub.shape[1]] += sub
            rho[:, :, i, j] = cost_map
    return rho


def embed_juniward(Y, quant, alpha, seed=_RNG_DEFAULT, spatial=None):
    """Simulated J-UNIWARD (ternary +/-1). Requires `spatial` (decoded image)."""
    if spatial is None:
        raise ValueError("J-UNIWARD needs the spatial image (decoded cover)")
    rng = np.random.default_rng(seed)
    ac = _ac_mask(Y.shape)
    n_nz = int(((Y != 0) & ac[None, None]).sum())
    if n_nz == 0:
        return Y.copy(), 0
    payload_bits = alpha * n_nz
    rho = _juniward_costs(Y, quant, spatial.astype(np.float64)).ravel()

    def total_entropy(lmbda):
        p = np.exp(-lmbda * rho); p = p / (1 + 2 * p)
        return _ternary_entropy(p).sum()
    lo, hi = 1e-6, 1e6
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        if total_entropy(mid) > payload_bits:
            lo = mid
        else:
            hi = mid
    lmbda = np.sqrt(lo * hi)
    p = np.exp(-lmbda * rho); p = p / (1 + 2 * p)
    r = rng.random(p.shape); change = r < (2 * p)
    sign = np.where(rng.random(p.shape) < 0.5, 1, -1)
    delta = np.where(change, sign, 0).reshape(Y.shape).astype(Y.dtype)
    return Y + delta, int(change.sum())


# --------------------------------------------------------------------------- #
# Shared ternary simulator (cost -> prob via Boltzmann, lambda tuned to payload)
# --------------------------------------------------------------------------- #
def _embed_ternary(Y, rho, alpha, seed):
    """Ternary +/-1 embedding at the rate-distortion bound, given a cost map rho."""
    rng = np.random.default_rng(seed)
    ac = _ac_mask(Y.shape)
    n_nz = int(((Y != 0) & ac[None, None]).sum())
    if n_nz == 0:
        return Y.copy(), 0
    payload_bits = alpha * n_nz
    rho = rho.ravel()

    def total_entropy(lmbda):
        p = np.exp(-lmbda * rho); p = p / (1 + 2 * p)
        return _ternary_entropy(p).sum()
    lo, hi = 1e-6, 1e6
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        if total_entropy(mid) > payload_bits:
            lo = mid
        else:
            hi = mid
    lmbda = np.sqrt(lo * hi)
    p = np.exp(-lmbda * rho); p = p / (1 + 2 * p)
    r = rng.random(p.shape); change = r < (2 * p)
    sign = np.where(rng.random(p.shape) < 0.5, 1, -1)
    delta = np.where(change, sign, 0).reshape(Y.shape).astype(Y.dtype)
    return Y + delta, int(change.sum())


# --------------------------------------------------------------------------- #
# UED (Uniform Embedding Distortion): COEFFICIENT-level cost from the context
# energy (other AC in the block + same mode in neighbouring blocks). A cost family
# distinct from UERD (which is block energy).
# --------------------------------------------------------------------------- #
def _ued_costs(Y, quant):
    # UED: cost from the CONTEXT energy (not the coefficient's own magnitude). The
    # context combines the OTHER AC of the block (intra) and the SAME mode in the 4
    # neighbouring blocks (inter). A different family from UERD (block energy).
    ac = _ac_mask(Y.shape)
    deq = np.abs(Y) * quant[None, None]                          # (Hb,Wb,8,8)
    block_e = (deq * ac[None, None]).sum(axis=(2, 3), keepdims=True)  # block AC energy
    intra = (block_e - deq) / 62.0                              # mean of the OTHER AC (excludes self)
    pad = np.pad(deq, ((1, 1), (1, 1), (0, 0), (0, 0)), mode="edge")
    inter = (pad[:-2, 1:-1] + pad[2:, 1:-1] +
             pad[1:-1, :-2] + pad[1:-1, 2:]) / 4.0               # same mode, 4 neighbours
    ctx = intra + inter                                          # context energy (WITHOUT self)
    ctx = np.where(ctx <= 0, 1e-6, ctx)
    rho = quant[None, None] / ctx                               # cheap where context is textured
    return np.where(ac[None, None], rho, 1e10)


def embed_ued(Y, quant, alpha, seed=_RNG_DEFAULT, spatial=None):
    """Simulated UED (ternary +/-1, adaptive on context energy)."""
    return _embed_ternary(Y, _ued_costs(Y, quant), alpha, seed)


# --------------------------------------------------------------------------- #
# J-MiPOD-style: MODEL-BASED cost from local spatial variance, Fisher-flavoured
# (~q^2 / var). A distinct family (variance model, not a distortion design).
# Simulator approximation of MiPOD for JPEG.
# --------------------------------------------------------------------------- #
from scipy.ndimage import uniform_filter


def _jmipod_costs(Y, quant, spatial):
    # J-MiPOD: MODEL-BASED cost from the local SPATIAL VARIANCE (pixel-domain texture
    # of the cover), NOT from the coefficient magnitude. Fisher-like cost ~ q^2 / sigma^2:
    # cheap (more changes) where texture is high. A different family from UERD/UED (which
    # use DCT energy), but likewise concentrated in textured regions.
    ac = _ac_mask(Y.shape)
    Hb, Wb = Y.shape[:2]
    sp = np.asarray(spatial, np.float64)
    mu = uniform_filter(sp, size=5, mode="nearest")
    var_s = np.clip(uniform_filter(sp * sp, size=5, mode="nearest") - mu * mu, 1.0, None)
    vb = var_s[:Hb * 8, :Wb * 8].reshape(Hb, 8, Wb, 8).mean(axis=(1, 3))  # mean variance per block
    rho = (quant[None, None] ** 2) / vb[:, :, None, None]                 # (Hb,Wb,8,8)
    return np.where(ac[None, None], rho, 1e10)


def embed_jmipod(Y, quant, alpha, seed=_RNG_DEFAULT, spatial=None):
    """Simulated J-MiPOD-style (ternary +/-1, local-variance/Fisher cost)."""
    if spatial is None:
        raise ValueError("J-MiPOD needs the spatial image (decoded cover)")
    return _embed_ternary(Y, _jmipod_costs(Y, quant, spatial), alpha, seed)


EMBEDDERS = {"nsF5": embed_nsf5, "UERD": embed_uerd, "JUNIWARD": embed_juniward,
             "UED": embed_ued, "JMIPOD": embed_jmipod}
