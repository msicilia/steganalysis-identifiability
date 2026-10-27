"""Central configuration: paths, payloads, class folders and training hyperparameters for the
spatial and JPEG pipelines. The code refers to four validity notes:
  [V1] images are used at 512x512, without resizing or cropping;
  [V2] cover and stego come from the same source;
  [V3] splits are disjoint by cover id; FUSION is halved by cover into a checkpoint-selection
       half and a meta-learner half; TEST is never used for fitting or selection;
  [V4] a global seed, per-image embedding seeds and best-effort deterministic cuDNN.
"""

from pathlib import Path

# ----------------------------------------------------------------------------
# PAYLOAD SWEEP (bpp)
# ----------------------------------------------------------------------------
# Multiple payload values are used rather than a single one.
# Design rule:
#   - Binary detection (cover vs stego)   -> swept across ALL payloads (curve).
#   - Family identification (which algo)  -> FIXED payload = FAMILY_ID_PAYLOAD
#     (mixing payloads would confound "how much payload" with "which algorithm").
PAYLOADS          = [0.10, 0.20, 0.40]   # sweep payloads (bpp)
FAMILY_ID_PAYLOAD = 0.40                  # fixed payload for the multiclass part
# "Active" payload: the dataset/splits the scripts use by default.
# The training functions accept --alpha to override it.
ACTIVE_ALPHA      = 0.40

# ----------------------------------------------------------------------------
# Paths. One dataset (and its splits) PER payload, e.g.
# datasets/DATA_SPATIAL_a040 (alpha=0.40), _a020, _a010.
# ----------------------------------------------------------------------------
import os

PROJECT_DIR = Path(__file__).resolve().parent          # the code/ folder

def _envpath(var, default):
    return Path(os.environ[var]) if os.environ.get(var) else default

# Data / models / results directories. Self-contained by default (inside the project);
# override with the STEG_DATA / STEG_MODELS / STEG_RESULTS environment variables.
DATASETS   = _envpath("STEG_DATA", PROJECT_DIR / "data")

# Active domain via the STEG_DOMAIN environment variable: "spatial" (default) or "jpeg".
DOMAIN      = os.environ.get("STEG_DOMAIN", "spatial")
DATA_PREFIX = "DATA_JPEG_q75" if DOMAIN == "jpeg" else "DATA_SPATIAL"

def alpha_tag(alpha: float) -> str:
    """0.40 -> 'a040', 0.1 -> 'a010' (per-payload folder tag)."""
    return f"a{int(round(alpha * 100)):03d}"

def data_root(alpha: float) -> Path:
    """Dataset folder (spatial or JPEG, per DOMAIN) for a given payload."""
    return DATASETS / f"{DATA_PREFIX}_{alpha_tag(alpha)}"

def split_root(alpha: float) -> Path:
    """Folder of the train/fusion/test splits for a given payload. [V3]"""
    return DATASETS / f"{DATA_PREFIX}_{alpha_tag(alpha)}_SPLITS"

def dirs_for(alpha: float) -> dict:
    """train/fusion/test (disjoint by cover ID) for a given payload."""
    r = split_root(alpha)
    return {"train": r / "train", "fusion": r / "fusion", "test": r / "test"}

# Active paths (derived from ACTIVE_ALPHA). Scripts use these by default.
DATA_ROOT  = data_root(ACTIVE_ALPHA)       # output of gen_spatial.py
SPLIT_ROOT = split_root(ACTIVE_ALPHA)
_D         = dirs_for(ACTIVE_ALPHA)
TRAIN_DIR  = _D["train"]
FUSION_DIR = _D["fusion"]
VAL_DIR    = _D["fusion"]                   # alias for the validation/fusion split
# [V3] Disjoint by-cover halves of FUSION (see data.fusion_subset_stems): checkpoint
# selection on "select", stacking meta-learners on "meta".
FUSION_SUBSETS = ("select", "meta")
TEST_DIR   = _D["test"]

# Model folder (separated per domain so JPEG models do not overwrite spatial ones).
_MODELS    = _envpath("STEG_MODELS", PROJECT_DIR / "models")
MODEL_DIR  = _MODELS / "jpeg" if DOMAIN == "jpeg" else _MODELS

# Results directory (JSON outputs and figures).
RESULTS_DIR = _envpath("STEG_RESULTS", PROJECT_DIR / "results")

# ----------------------------------------------------------------------------
# Image parameters  [V1]
# ----------------------------------------------------------------------------
# BOSSbase is 512x512 grayscale.
# [V1] No resizing.
# No cropping either: content-adaptive embedding (HILL/WOW/S-UNIWARD/MiPOD) is
#   localised in textured regions. We train on the FULL 512x512 image (the embedding
#   was done on the whole image).
# Since images are already 512x512, image_size=(512,512) does not resample (no-op).
IMG_SIZE    = (512, 512)      # full image, no resizing, no cropping
LOAD_SIZE   = (512, 512)
USE_PATCHES = False           # False = full native image (no crop, no resize)
CHANNELS    = 1

# Smaller batch because 512x512 uses ~4x more memory than 256x256.
BATCH_SIZE  = 16

# ----------------------------------------------------------------------------
# Reproducibility [V4]
# ----------------------------------------------------------------------------
SEED = 42

# ----------------------------------------------------------------------------
# Label space
# ----------------------------------------------------------------------------
# Class folders (stable order). Spatial: 5 adaptive schemes; JPEG: 4 adaptive + 1 legacy.
if DOMAIN == "jpeg":
    CLASS_FOLDERS = ["COVER", "JUNIWARD", "UERD", "nsF5", "UED", "JMIPOD"]  # nsF5 = legacy; 4 adaptive
    LEGACY_SCHEMES = ["nsF5"]      # non-adaptive; excluded from the adaptive-only analyses
else:
    CLASS_FOLDERS = ["COVER", "HILL", "HUGO", "MiPOD", "S-UNIWARD", "WOW"]
    LEGACY_SCHEMES = []
COVER_NAME    = "COVER"

# ----------------------------------------------------------------------------
# Training hyperparameters
# ----------------------------------------------------------------------------
LR          = 1e-3
L2          = 1e-4
DROPOUT     = 0.5
EPOCHS_BIN  = 80     # binary detector (early stopping on val_auc); 40 cut runs still improving
EPOCHS_SPEC = 80     # specialist (early stopping)
ES_PATIENCE = 8      # engine.train default (trainers may pass their own)
# GBRAS-Net (train_gbras.py, train_gbras_wh.py). The original protocol trains a fixed number
# of epochs and keeps the best checkpoint; patience 8 on its noisy block accuracy stopped
# runs at epochs 17-51 while validation was still rising. None = no early stopping (the best
# checkpoint on the FUSION "select" half is still kept); an int restores patience-based
# stopping (e.g. 25) to cut GPU time. Override per run with --es-patience.
GBRAS_EPOCHS      = 100
GBRAS_ES_PATIENCE = None
LR_PATIENCE = 4      # ReduceLROnPlateau patience (factor 0.5)
# Early stopping may not stop a run before this many epochs (nor may the LR scheduler
# decay before it): runs at low payload / no-SRM previously stopped at epochs 9-12
# with best epoch 1-4 at chance, i.e. before learning had started.
ES_MIN_EPOCHS = 15
LABEL_SMOOTHING = 0.05

# Enable class re-balancing.
USE_CLASS_WEIGHTS = True
