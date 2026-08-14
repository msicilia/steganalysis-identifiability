# Steganalysis pipeline — code

Detection and algorithm identification of content-adaptive image steganography, across
two domains: **spatial** (HUGO, WOW, S-UNIWARD, MiPOD, HILL) and **JPEG** (J-UNIWARD,
UERD, UED, J-MiPOD, nsF5). All paths, payloads and methodological choices are centralised
in `config.py`; validity-critical points are flagged `[V1]…[V4]` there.

## Setup

A single virtual environment, managed with [uv](https://docs.astral.sh/uv/). From the
repository root:

```bash
uv sync                 # creates .venv with all dependencies
uv run python pt/extract_dctr.py --alpha 0.4   # run any script with `uv run`
```

`uv run` keeps the environment in sync with `pyproject.toml`/`uv.lock` and runs inside it,
so there is no need to activate or reference `.venv` by hand. (uv may print a one-line
re-install of a CUDA wheel on each run — a harmless upstream quirk; add `--no-sync`, i.e.
`uv run --no-sync python ...`, to skip the sync check.)

System prerequisite: the libjpeg development headers (`apt install libjpeg-dev`), needed to
build `jpeglib`. The environment includes PyTorch (CUDA 12.8 wheels), conseal, jpeglib,
NumPy, SciPy, scikit-learn, matplotlib, Pillow, and pycol-complexity (Lorena
data-complexity measures). For a CPU-only host, drop the `pytorch-cu128` index in
`pyproject.toml`.

## Configuration

Two sets of environment variables control a run; both are read by `config.py`:

- **`STEG_DOMAIN`** — `spatial` (default) or `jpeg`. It selects the dataset prefix, the
  class set and the model/feature folders. `config.py` derives these constants at *import
  time*, so the domain is chosen through the environment rather than a per-script CLI flag;
  set it once per shell (`export STEG_DOMAIN=jpeg`). Domain-specific scripts assert the
  expected domain, so a mismatch fails fast.
- **`STEG_DATA` / `STEG_MODELS` / `STEG_RESULTS`** — override the data, model and results
  directories (defaults: `data/`, `models/`, `results/` in the repository root). Use these
  to point at data that lives outside the repository; nothing else is hardcoded to a machine path.

## Pipeline

```
covers (manual)  →  stego generation  →  splits  →  DCTR features (JPEG only)  →  train / analyse
```

Covers are the public BOSSbase 1.01 set (10,000 grayscale 512×512), placed in
`data/covers/BOSSbase_1.01/`; all 10,000 covers are used. There is no download script.
Commands below are run from the repository root.

### Spatial
```bash
uv run python gen_spatial.py --alphas 0.1,0.2,0.4
uv run python make_splits.py --alpha 0.4
uv run python pt/train_binary.py     --alpha 0.4              # lightweight cover-vs-all detector
uv run python pt/train_gbras.py --algo HUGO --alpha 0.4      # GBRAS specialist (repeat per algorithm)
uv run python pt/train_gbras_wh.py --mode detect --alpha 0.4 # GBRAS cover-vs-all
uv run python pt/train_gbras_wh.py --mode family --alpha 0.4 # GBRAS 6-class family
uv run python pt/separability_spatial.py --alpha 0.4
uv run python pt/transfer_spatial.py --alpha 0.4
uv run python pt/spatial_bootstrap.py               # bootstrap CIs
```

### JPEG
```bash
export STEG_DOMAIN=jpeg
uv run python pt/gen_jpeg_conseal.py --alphas 0.1,0.2,0.4                 # reference: J-UNIWARD/UERD/nsF5
uv run python pt/gen_jpeg.py --embedders UED,JMIPOD --alphas 0.1,0.2,0.4  # home: UED/J-MiPOD
uv run python make_splits.py     --alpha 0.4
uv run python pt/extract_dctr.py --alpha 0.4
uv run python pt/dctr_ensemble.py --alpha 0.4    # detection + identification + monolithic vs modular
uv run python pt/dctr_bootstrap.py --alpha 0.4   # bootstrap CIs + classifier-robustness of the ceiling
uv run python pt/transfer_jpeg.py --alpha 0.4    # leave-one-algorithm-out detection transfer
uv run python pt/jpeg_kfold_cv.py --alpha 0.4 --folds 5   # cross-validated metrics + CIs
```

## Scripts

**Configuration & data**
| File | Role |
|---|---|
| `config.py` | Paths, payloads, class folders, hyperparameters, validity notes. Single config point. |
| `gen_spatial.py` | Spatial stego generation with conseal (PNG, lossless). |
| `pt/gen_jpeg_conseal.py` | JPEG stego with conseal reference costs (J-UNIWARD/UERD/nsF5), AC-only. |
| `pt/gen_jpeg.py` | JPEG stego with the home simulators (UED, J-MiPOD) via torchjpeg. |
| `pt/jpeg_embedders.py` | UED (Guo 2014 coefficient-context cost) and J-MiPOD (DCT-domain MiPOD variance model + detectability-optimal beta solver, reusing conseal's reference MiPOD) simulators, plus the shared ternary rate–distortion embedder. |
| `make_splits.py` | Train/fusion/test splits, **disjoint by cover ID** (no leakage). |
| `pt/extract_dctr.py` | DCTR feature extraction (Holub & Fridrich, 2015) for the JPEG domain. |

**Models & training**
| File | Role |
|---|---|
| `pt/models.py` | Detector architectures (lightweight SRM CNN; GBRAS-Net). |
| `pt/engine.py` | Shared PyTorch train/eval loop. |
| `pt/data.py` | PyTorch data loading (spatial). |
| `pt/gbras_io.py` | GBRAS-Net as a spatial detector (per-image inference helpers). |
| `pt/train_binary.py` | Cover-vs-all binary detector. |
| `pt/train_family.py` | Multiclass family identification (fixed payload). |
| `pt/train_gbras.py` | GBRAS-Net cover-vs-one specialist (one per algorithm, `--algo`). |
| `pt/train_gbras_wh.py` | GBRAS-Net cover-vs-all detector and 6-class family (`--mode detect|family`). |
| `pt/train_pair_specialist.py` | Per-algorithm specialist (cover vs one). |

**Analysis**
| File | Role |
|---|---|
| `pt/dctr_ensemble.py` | JPEG detection, identification, and monolithic-vs-modular ensemble. |
| `pt/dctr_bootstrap.py` | JPEG single-split bootstrap CIs and the monolithic-vs-modular ceiling across LogReg/LinearSVC/RandomForest. |
| `pt/jpeg_kfold_cv.py` | JPEG k-fold CV (by cover): detection, identification, transfer and Fisher separability, each with a 95% CI. |
| `pt/spatial_bootstrap.py` | Spatial bootstrap confidence intervals (detection sweep, GBRAS per-algo). |
| `pt/separability_spatial.py` | Spatial separability (Fisher + data-complexity measures). |
| `pt/build_ensemble.py`, `pt/build_gbras_ensemble.py` | Modular stacking ensemble (leak-free meta-features). |
| `pt/transfer_jpeg.py`, `pt/transfer_spatial.py` | Leave-one-algorithm-out detection transfer. |
| `pt/validate_embedders.py` | Fidelity check of the JPEG simulators (change rate, detectability, separability). |
| `pt/fig_fisher_heatmap.py` | Fisher separability heatmap (`--domain spatial|jpeg`). |
| `pt/aggregate.py` | Collect result logs into summary tables. |

## Data assets

- `SRM_Kernels1.npy` — the 30 fixed SRM high-pass kernels that form GBRAS-Net's first
  layer. GBRAS-Net starts with a **non-trainable** residual front end: these kernels
  convolve the input to amplify the tiny ±1 stego signal before the learnable layers.
  They are the published kernel bank of the SRM/GBRAS-Net design, not something learned
  during training, so they ship as a data file and are loaded verbatim in `pt/models.py`
  (`GBRASNet`). Required to build or load any GBRAS-Net model; without it `GBRASNet`
  cannot be instantiated. (The lightweight detector does not need it — it computes its 7
  SRM filters in code.)

## Validity notes (see `config.py`)

- **[V1]** Native 512×512 resolution; no resizing (it would erase the ±1 residual).
- **[V2]** Cover and stego from the same source (no cover-source mismatch).
- **[V3]** Splits disjoint by cover ID; ensemble meta-features computed out-of-fold.
- **[V4]** Fixed global seed; documented hyperparameters.
