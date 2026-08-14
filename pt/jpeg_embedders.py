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
# UED (Uniform Embedding Distortion, Guo et al. 2014). COEFFICIENT-level cost:
# rho_ij = 1 / (neighbourhood magnitude), where the neighbourhood combines the
# FREQUENCY-ADJACENT modes in the same block (intra: (i+-1,j),(i,j+-1)) and the SAME
# mode in the 4 adjacent blocks (inter). The numerator is UNIFORM (=1), which is the
# defining "uniform embedding" property; a q_ij numerator would instead be the later
# UERD variant (Guo 2015, generated here via conseal). This makes UED a genuinely
# distinct DCT-energy family from UERD (per-block energy), not "UERD with a different
# denominator". DC is excluded from every neighbourhood and is not embeddable.
# --------------------------------------------------------------------------- #
UED_INTER_WEIGHT = 1.0     # relative weight of inter-block vs intra-block neighbours


def _ued_costs(Y, quant):
    ac = _ac_mask(Y.shape)
    deq = (np.abs(Y).astype(np.float64) * quant[None, None]) * ac[None, None]  # AC dequant magn (DC=0)
    # intra-block: sum of the frequency-adjacent modes within each block (self excluded)
    intra = np.zeros_like(deq)
    intra[:, :, 1:, :] += deq[:, :, :-1, :]     # neighbour (i-1, j)
    intra[:, :, :-1, :] += deq[:, :, 1:, :]     # neighbour (i+1, j)
    intra[:, :, :, 1:] += deq[:, :, :, :-1]     # neighbour (i, j-1)
    intra[:, :, :, :-1] += deq[:, :, :, 1:]     # neighbour (i, j+1)
    # inter-block: same mode (i,j) in the 4 spatially adjacent blocks
    pad = np.pad(deq, ((1, 1), (1, 1), (0, 0), (0, 0)), mode="edge")
    inter = pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
    D = intra + UED_INTER_WEIGHT * inter
    D = np.where(D <= 0, 1e-6, D)               # flat neighbourhood -> high cost
    rho = 1.0 / D                               # uniform numerator (Guo 2014)
    return np.where(ac[None, None], rho, 1e10)  # DC not embeddable


def embed_ued(Y, quant, alpha, seed=_RNG_DEFAULT, spatial=None):
    """Simulated UED (ternary +/-1) with the Guo-2014 coefficient-context cost."""
    return _embed_ternary(Y, _ued_costs(Y, quant), alpha, seed)


# --------------------------------------------------------------------------- #
# J-MiPOD (JPEG port of MiPOD, Cogranne et al. 2022). Statistical model, not a
# heuristic distortion: model each unquantised DCT coefficient as an independent
# Gaussian N(0, sigma_ij^2); choose change rates that MINIMISE THE DETECTABILITY of
# the optimal detector under the payload constraint, via the per-coefficient Fisher
# information. This reuses the reference MiPOD variance model and beta solver shipped
# in conseal (ported from the authors' Matlab), lifted from the pixel domain into the
# DCT domain here.
#
# Pipeline (faithful):
#   1. per-PIXEL variance sigma_p^2 from the MiPOD parametric model (wiener residual +
#      local 2D-DCT trigonometric-polynomial fit) -> conseal.mipod.estimate_variance.
#   2. propagate to a per-COEFFICIENT variance: for an orthonormal block DCT,
#      sigma_ij^2 = sum_{u,v} B_ij(u,v)^2 sigma_pixel(u,v)^2  (pixels approx independent).
#   3. Fisher information of a +/- q_ij dequantised change against the variance-based
#      optimal detector: FI_ij = q_ij^4 / sigma_ij^4 (generalises spatial MiPOD's
#      1/sigma^4, whose change step is 1; here the step is q_ij). DC is wet.
#   4. change rates beta from the MiPOD Lagrangian solver (conseal ternary_probs /
#      invxlnx3) at the bpnzAC payload, then simulate ternary +/-1 with prob 2*beta.
#
# FI_Q_EXPONENT sets the change step in the Fisher information: the deflection of a
# +/- q_ij dequantised change against a variance detector gives exponent 4.
# --------------------------------------------------------------------------- #
from conseal.mipod._costmap import estimate_variance
from conseal.mipod._defs import wiener2, ternary_probs

FI_Q_EXPONENT = 4             # FI_ij = q_ij^FI_Q_EXPONENT / sigma_ij^4
_VAR_FLOOR = 0.01             # minimum variance (conseal MiPOD, Eq. 27)
_SQ_DCT_BASIS = _UNI_B ** 2   # (8,8,8,8): squared orthonormal DCT synthesis patterns


def _coeff_variance(Y, spatial):
    """Per-coefficient DCT-domain variance sigma_ij^2, shape (Hb,Wb,8,8).

    Uses the reference MiPOD pixel-variance model (conseal) then propagates it through
    the orthonormal block DCT."""
    Hb, Wb = Y.shape[:2]
    x0 = np.asarray(spatial, np.float64)
    wres = x0 - wiener2(x0, kernel_size=(2, 2))
    var_pix = np.clip(estimate_variance(wres, block_size=9, degree=9), _VAR_FLOOR, None)
    # (H,W) -> per-block (Hb,Wb,8,8)
    vb = var_pix[:Hb * 8, :Wb * 8].reshape(Hb, 8, Wb, 8).transpose(0, 2, 1, 3)
    # sigma_ij^2 = sum_{u,v} B_ij(u,v)^2 var_pixel(u,v)
    var_dct = np.einsum("ijuv,hwuv->hwij", _SQ_DCT_BASIS, vb, optimize=True)
    return np.clip(var_dct, _VAR_FLOOR, None)


def _jmipod_fisher(Y, quant, spatial):
    """Per-coefficient Fisher information FI_ij = q_ij^e / sigma_ij^4 (DC wet)."""
    ac = _ac_mask(Y.shape)
    var_dct = _coeff_variance(Y, spatial)
    fi = (quant[None, None].astype(np.float64) ** FI_Q_EXPONENT) / (var_dct ** 2)
    return np.where(ac[None, None], fi, 1e30)      # DC wet -> beta ~ 0


def embed_jmipod(Y, quant, alpha, seed=_RNG_DEFAULT, spatial=None):
    """Simulated J-MiPOD (ternary +/-1) via the reference MiPOD detectability solver."""
    if spatial is None:
        raise ValueError("J-MiPOD needs the spatial image (decoded cover)")
    ac = _ac_mask(Y.shape)
    n_nz = int(((Y != 0) & ac[None, None]).sum())
    if n_nz == 0:
        return Y.copy(), 0
    fi = _jmipod_fisher(Y, quant, spatial)
    payload_nats = alpha * n_nz * np.log(2)                     # bpnzAC bits -> nats
    beta = ternary_probs(fi, payload_nats).reshape(fi.shape, order="F")   # prob(+1)=prob(-1)
    rng = np.random.default_rng(seed)
    change = rng.random(fi.shape) < (2.0 * beta)               # total change prob 2*beta
    sign = np.where(rng.random(fi.shape) < 0.5, 1, -1)
    delta = np.where(change, sign, 0).astype(Y.dtype)
    return Y + delta, int(change.sum())


EMBEDDERS = {"nsF5": embed_nsf5, "UERD": embed_uerd, "JUNIWARD": embed_juniward,
             "UED": embed_ued, "JMIPOD": embed_jmipod}
