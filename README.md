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
  to point at data that lives outside the repository; nothing else is hardcoded to a machine
  path. For a rerun, point `STEG_MODELS` and `STEG_RESULTS` at fresh directories so files
  from earlier code cannot be mixed into new results.

## Conventions every script follows

These exist because getting any of them wrong silently changes the published numbers.

- **Classifiers go through `util.fit_logreg`.** On the 20,480-dimensional DCTR features
  lbfgs needs many iterations, and an under-converged fit still produces a
  plausible-looking accuracy that depends on the cap rather than on the data. `fit_logreg`
  treats any sklearn `ConvergenceWarning` as a failure (the plain `n_iter < max_iter` test
  is not enough: scipy's L-BFGS-B also stops at 15,000 function evaluations), continues the
  fit from its previous coefficients until it converges, and raises at 160,000 total
  iterations. Every result file carries a `provenance` block with the number of fits, the
  iterations used and whether all converged. Do not construct `LogisticRegression` directly.
- **Confidence intervals resample COVERS, not images** (`util.group_boot`,
  `util.boot_ci(..., groups=...)`). A cover and its stego variants are the same image
  content, so treating them as independent draws understates every interval. An interval
  that does not contain its point estimate triggers a warning.
- **Neighbourhood measures use one variant per cover.** In a split where every cover
  appears once per class, a point's nearest neighbour is almost always its own cover in
  another class, which drives k-NN error above chance regardless of the features.
  `separability_spatial.py` therefore computes k-NN, neighbour confusion, silhouette and the
  pycol measures on a subsample with one randomly chosen class per cover; Fisher J uses all
  rows.
- **Every model and result name carries the payload tag and the seed**
  (`pt_<kind>_<aXXX>_s<seed>.pt`), and every trainer writes a sibling `.json` with its
  arguments, best epoch, epochs run and test metrics.
- **Early stopping cannot stop a run before `ES_MIN_EPOCHS` (15).** Runs that need several
  epochs before they start learning would otherwise be stopped at chance and reported as a
  detection limit. Patience values come from `config.py` (`ES_PATIENCE=8`, `LR_PATIENCE=4`).
  GBRAS-Net is the exception: it follows its original protocol and trains all
  `GBRAS_EPOCHS` (100) epochs with a constant Adam learning rate (no plateau scheduler, as in
  the original notebook), keeping the best checkpoint (`GBRAS_ES_PATIENCE=None`;
  `--es-patience N` restores early stopping).
- **Truncated runs are flagged.** Every model JSON records `epochs_cap` and
  `still_improving_at_cap` (all epochs used and the best checkpoint in the last
  `ES_PATIENCE` epochs); `aggregate.py` lists such runs, whose metrics depend on the budget.

## Running everything

```bash
export STEG_MODELS=../runs/<campaign>/models STEG_RESULTS=../runs/<campaign>/results
./run_all.sh spatial    # or: jpeg | all
```

`run_all.sh` refuses to start unless `STEG_MODELS` and `STEG_RESULTS` are set, and refuses a
non-empty directory that has no completion markers (files from another campaign;
`ALLOW_NONEMPTY=1` overrides).

`run_all.sh` runs every step below strictly in sequence (never two GPU jobs at once), logs
each step to `logs/<run-id>/<step>.log` with unbuffered output, and writes
`logs/<run-id>/provenance.txt` (git hash, uncommitted changes, package versions, GPU). A
finished step leaves a marker in `$STEG_RESULTS/.done/`, so rerunning the script resumes
after an interruption; delete a marker or set `FORCE=1` to redo steps. Commit the code
before launching so the results map to a version.

The JPEG phase refuses to start if JPEG data from an earlier code version is present (see
*Regenerating JPEG data* below). It pauses twice, exiting with instructions: after the
simulator fidelity check (`touch $STEG_RESULTS/.done/jpeg_fidelity_reviewed`) and after the
`time_logreg` runtime estimate (`touch $STEG_RESULTS/.done/jpeg_runtime_reviewed`), because the
JPEG k-fold CV fits 33 logistic regressions per fold on 20,480-dimensional features and can
take days. `jpeg_kfold_cv.py` checkpoints every finished fold in
`$STEG_RESULTS/jpeg_kfold_<aXXX>_folds/` and resumes from there (a checkpoint from a different
configuration or dataset is refused), and saves every out-of-fold prediction to
`jpeg_kfold_<aXXX>_oof.npz`, so later analyses need no refit. `dctr_bootstrap.py` writes its
CIs and the transfer table before the LinearSVC/RandomForest robustness rows; a robustness
row that fails to converge is recorded as an error instead of discarding the rest.

Multiclass model JSONs carry a collapse diagnosis (`collapsed`, `pred_class_counts`,
`max_posterior_std`): a network with input-independent or single-class predictions scores
chance by construction, which is a training failure, not an identification limit.
`aggregate.py` lists such seeds.

## Pipeline, step by step

```
covers (manual)  →  stego generation  →  splits  →  features  →  train / analyse
```

Covers are the public BOSSbase 1.01 set (10,000 grayscale 512×512), placed in
`data/covers/BOSSbase_1.01/`; all 10,000 covers are used. There is no download script.
Commands below are run from the repository root and are exactly what `run_all.sh` runs.

### Spatial
```bash
uv run python gen_spatial.py --alphas 0.1,0.2,0.4             # resumable; re-validates existing PNGs
uv run python make_splits.py --alphas 0.1,0.2,0.4
uv run python pt/separability_spatial.py --alpha 0.4 --features srm   # no model needed: run early, read its cover-vs-stego control
uv run python pt/train_binary.py --alpha 0.4                  # lightweight cover-vs-all detector, one per payload
uv run python pt/train_binary.py --alpha 0.2 --init <MODELS>/pt_binary_a040_s42.pt   # payload curriculum
uv run python pt/train_binary.py --alpha 0.1 --init <MODELS>/pt_binary_a020_s42.pt
uv run python pt/train_binary.py --alpha 0.4 --no-srm         # front-end ablation
uv run python pt/train_family.py --alpha 0.4 --seed 42        # lightweight 6-class monolith (run_all: seeds 42, 1, 2)
uv run python pt/train_pair_specialist.py --algo HUGO --alpha 0.4   # repeat per algorithm
uv run python pt/train_gbras.py --algo HUGO --alpha 0.4       # GBRAS specialist, 100 epochs, no early stopping (repeat per algorithm)
uv run python pt/train_gbras_wh.py --mode detect --alpha 0.4  # GBRAS cover-vs-all
uv run python pt/train_gbras_wh.py --mode family --alpha 0.4 --seed 42  # GBRAS 6-class family (+ recall_per_class; run_all: seeds 42, 1, 2)
uv run python pt/build_ensemble.py --alpha 0.4                # modular vs monolithic (lightweight CNN)
uv run python pt/build_gbras_ensemble.py --alpha 0.4          # modular vs monolithic (GBRAS) + learned-feature Fisher
uv run python pt/transfer_spatial.py --alpha 0.4              # trains and saves 5 leave-one-out detectors
uv run python pt/spatial_bootstrap.py                         # CIs for all payloads, ablation and GBRAS specialists
uv run python pt/diagnostic_vs_confusion.py --domain spatial --alpha 0.4   # does pairwise Fisher J predict pairwise confusion?
uv run python pt/fig_fisher_heatmap.py --domain spatial
uv run python pt/aggregate.py --alpha 0.4                     # summary tables; exits 1 if anything is missing
```

The ensemble, separability, transfer, bootstrap and figure steps consume the trained
models above, so run them last.

### JPEG
```bash
export STEG_DOMAIN=jpeg
uv run python pt/gen_jpeg.py --embedders UED,JMIPOD --alphas 0.1,0.2,0.4  # FIRST: creates COVER + UED/J-MiPOD
uv run python pt/gen_jpeg_conseal.py --alphas 0.1,0.2,0.4                 # then J-UNIWARD/UERD/nsF5 (conseal)
uv run python make_splits.py --alphas 0.1,0.2,0.4
uv run python pt/validate_embedders.py extract --n 400                    # simulator fidelity...
uv run python pt/validate_embedders.py analyze                            # ...and its report
uv run python pt/extract_dctr.py --alpha 0.4      # split feature cache (single-split analyses only)
uv run python pt/time_logreg.py --alpha 0.4      # runtime estimate for the analyses below
uv run python pt/dctr_bootstrap.py --alpha 0.4   # single split, every model fitted once: writes
                                                 #   dctr_ensemble_ (detection, specialists, confusion),
                                                 #   dctr_bootstrap_ (CIs, ceiling across classifiers),
                                                 #   transfer_jpeg_ (leave-one-algorithm-out transfer)
uv run python pt/jpeg_kfold_cv.py --alpha 0.1 --folds 5 --parts detection   # CV detection (also 0.2)
uv run python pt/jpeg_kfold_cv.py --alpha 0.4 --folds 5 --parts detection,identification,separability
uv run python pt/fisher_pca_sweep.py --alpha 0.4                     # pairwise Fisher J vs PCA dimension (20..500)
uv run python pt/fisher_pca_sweep.py --alpha 0.4 --protocol apriori  # same, train-row J vs test confusion
uv run python pt/fig_fisher_heatmap.py --domain jpeg --jpeg-k 200
uv run python pt/diagnostic_vs_confusion.py --domain jpeg --alpha 0.4
uv run python pt/train_binary.py --alpha 0.4                    # control: pixel-domain SRM CNN on JPEG
uv run python pt/gen_dct.py --alpha 0.4                         # control: quantised DCT cubes...
uv run python pt/train_dct.py --mode binary --alpha 0.4         # ...and the DCT-coefficient CNN
uv run python pt/aggregate_jpeg.py --alpha 0.4                  # summary tables; exits 1 if anything is missing
```

`dctr_ensemble.py` and `transfer_jpeg.py` still run on their own, but `run_all.sh` does not
call them: `dctr_bootstrap.py` reuses their code in one process, so the same 20,480-d
models are not fitted two or three times.

`gen_jpeg.py` is the only script that creates the JPEG covers, so it must run before
`gen_jpeg_conseal.py`, which exits with an error if it finds none. Both default to all
covers.

### Regenerating JPEG data

JPEG data generated by earlier versions of this code must be deleted and regenerated:

- **Seeds.** Every JPEG embedding is now seeded by `jpeg_embedders.embed_seed(stem, scheme,
  alpha)`. Earlier, all schemes on a cover shared one seed, so UED and J-MiPOD shared many
  change positions, which could make them look more alike than they are.
- **Container.** Covers and stego are now written by the same function
  (`jpeg_embedders.write_jpeg`), so their files differ only in the embedded coefficients.
  Earlier, covers were written by PIL and stego by jpeglib, with different headers.

The generators skip files that already exist, so delete `DATA_JPEG_q75_aXXX` together with
its `_SPLITS`, `_DCTR`, `_DCT` and `_KFOLD.npz` derivatives first. `gen_jpeg.py` marks each
folder it creates with a `.generator_v2` sentinel; the generators and `run_all.sh` refuse JPEG
data without it, while an interrupted generation simply resumes. Files are written under a
temporary name and renamed, so a killed run never leaves a truncated JPEG, and `gen_jpeg.py`
exits with an error if any cover failed. The `jpeg_kfold_cv.py`
feature cache records its inputs (DCTR settings, classes, a hash of the file list) and
refuses to load a cache that does not match the data; pass `--rebuild` to recreate it.
Spatial stego regenerates bit-exactly from its seeds and does not need to be rebuilt.

### Optional

Cross-dataset detection (BOSSbase → ALASKA#2; needs the ALASKA#2 JPEG set and the BOSSbase
DCTR cache):

```bash
uv run python pt/crossds_jpeg.py --alpha 0.4
```

Negative control (why JPEG uses DCTR rather than a CNN on the coefficients):

```bash
uv run python pt/gen_dct.py   --alpha 0.4                    # quantised DCT cubes
uv run python pt/train_dct.py --mode binary --alpha 0.4      # stays near chance
```

## Scripts

**Configuration & data**
| File | Role |
|---|---|
| `config.py` | Paths, payloads, class folders, hyperparameters, validity notes. Single config point. |
| `run_all.sh` | Sequential, resumable, logged run of the whole pipeline. |
| `gen_spatial.py` | Spatial stego generation with conseal (PNG, lossless), seeded per cover and scheme. |
| `pt/gen_jpeg.py` | JPEG covers (BOSSbase → Q75) and stego with the home simulators (UED, J-MiPOD) via jpeglib. |
| `pt/gen_jpeg_conseal.py` | JPEG stego with conseal reference costs (J-UNIWARD/UERD/nsF5), AC-only. |
| `pt/jpeg_embedders.py` | UED (JC-UED of Guo et al. 2014: Eq. 9 cost with α_ia=1.3, α_ir=1 on quantised coefficients; non-zero AC only, Eq. 15 changes that create no zeros; neighbours indexed on the whole-image coefficient grid, so at block edges the ±1 neighbour lies in the adjacent block, as Eq. 9 is written) and J-MiPOD (DCT-domain MiPOD variance model + detectability-optimal beta solver, reusing conseal's reference MiPOD) simulators, the shared ternary rate–distortion embedder, `embed_seed` and `write_jpeg`. |
| `make_splits.py` | Train/fusion/test splits, **disjoint by cover ID** (no leakage). |
| `pt/extract_dctr.py` | DCTR features for the JPEG domain: the unmerged-phase variant (64 filters × 64 phases × 5 bins = 20,480-d), not the 8,000-d phase-merged feature of Holub & Fridrich (2015). Like the reference `DCTR.m`, it filters the UNROUNDED decompressed image (block IDCT of the dequantised coefficients + 128), not uint8 pixels from a JPEG decoder; q = 4 is `DCTR.m`'s value at QF 75. Every DCTR consumer uses its `load_batch`. |
| `pt/gen_dct.py` | Quantised DCT cubes for the DCT-domain CNN control. |

**Models & training**
| File | Role |
|---|---|
| `pt/models.py` | Detector architectures (lightweight SRM CNN; GBRAS-Net; DCTNet). GBRAS-Net BatchNorm uses PyTorch `momentum=0.8`, the equivalent of the original Keras `momentum=0.2`. |
| `pt/engine.py` | Shared PyTorch train/eval loop (early stopping on FUSION, warm-up), best-effort determinism, metrics JSON writer. |
| `pt/data.py` | PyTorch data loading. |
| `pt/gbras_io.py` | GBRAS-Net as a spatial detector (per-image inference helpers). |
| `pt/train_binary.py` | Cover-vs-all binary detector (`--no-srm` for the ablation, `--init` for the payload curriculum). |
| `pt/train_family.py` | Multiclass family identification (fixed payload). |
| `pt/train_gbras.py` | GBRAS-Net cover-vs-one specialist (one per algorithm, `--algo`, 100 epochs max). |
| `pt/train_gbras_wh.py` | GBRAS-Net cover-vs-all detector and 6-class family (`--mode detect\|family`). In detect mode TRAIN draws a fresh seeded 1:1 stego subset every epoch (all stego seen over the run); the FUSION selection half uses a fixed 1:1 subset. |
| `pt/train_pair_specialist.py` | Per-algorithm specialist (cover vs one). |
| `pt/train_dct.py` | DCT-domain CNN control (near chance; motivates DCTR). |

**Analysis**
| File | Role |
|---|---|
| `pt/util.py` | Convergence-checked logistic regression, grouped bootstrap, Fisher J, feature/cover-id loading. |
| `pt/time_logreg.py` | Times one DCTR logistic-regression fit and extrapolates the runtime of each JPEG analysis. |
| `pt/dctr_ensemble.py` | JPEG detection, identification, and monolithic-vs-modular ensemble (single split). |
| `pt/dctr_bootstrap.py` | JPEG single-split bootstrap CIs and the monolithic-vs-modular ceiling across LogReg/LinearSVC/RandomForest (the RF cap applies only to the high-dimensional monolithic fit). |
| `pt/jpeg_kfold_cv.py` | JPEG k-fold CV (by cover): detection, identification, cross-fitted modular ensemble, transfer, each with a 95% CI; plus pairwise Fisher on a PCA-50 of all rows (descriptive, not cross-validated; superseded by `fisher_pca_sweep.py`). |
| `pt/fisher_pca_sweep.py` | Pairwise Fisher J on the leading k PCA components of the DCTR features (k = 20, 50, 100, 200, 500), with subsampling CIs and the Spearman/Mantel test against the pairwise confusion at each k. `--protocol all` uses the out-of-fold confusion; `--protocol apriori` fits PCA and J on the training rows of the single split and tests against its test confusion. The manuscript reports k = 200. |
| `pt/spatial_bootstrap.py` | Spatial CIs (detection sweep, ablation, GBRAS per-algo); flags collapsed detectors (constant scores), separately flags a CI that excludes its point estimate, and reports the within-cover P(stego score > cover score). |
| `pt/separability_spatial.py` | Spatial separability: Fisher J on all rows; neighbourhood and pycol measures on one variant per cover. |
| `pt/build_ensemble.py` | Lightweight-CNN modular ensemble. The primary detection ΔAUC uses a specialists-only meta-model; the variant that also includes the generalist is reported separately as nested. |
| `pt/build_gbras_ensemble.py` | GBRAS modular ensemble (specialists-only meta-model) and Fisher J on the learned 6-class features. |
| `pt/transfer_jpeg.py`, `pt/transfer_spatial.py` | Leave-one-algorithm-out detection transfer, with paired CIs for transfer − in-distribution. In JPEG, each held-out adaptive scheme is also scored by a detector trained on the other adaptive schemes only (`adaptive_only`; `config.LEGACY_SCHEMES` excluded), the protocol behind the adaptive-to-adaptive transfer claim. |
| `pt/diagnostic_vs_confusion.py` | Tests the diagnostic's prediction: Spearman correlation between pairwise Fisher J and pairwise confusion rate, with an exact Mantel permutation test over class labels. JPEG: train-row J vs TEST confusion (a priori) and CV; spatial: SRM statistics vs their linear probe, GBRAS detector features vs the GBRAS 6-class network. All classes, stego only, adaptive only. |
| `pt/crossds_jpeg.py` | Cross-dataset detection (BOSSbase → ALASKA#2). |
| `pt/validate_embedders.py` | Fidelity check of the JPEG simulators: change rate, detectability, adaptive-cluster membership, and a home-vs-conseal cross-check. Splits by cover. |
| `pt/fig_fisher_heatmap.py` | Fisher separability heatmap (`--domain spatial\|jpeg`), one panel on a logarithmic colour scale. Spatial: GBRAS detector features; JPEG: the PCA-k table of `fisher_pca_sweep.py` (`--jpeg-k`, default 200). `--domain spatial_all` adds the SRM-statistics and the circular 6-class panels. |
| `pt/aggregate.py` | Spatial summary tables from the model and result JSONs; lists anything missing. |
| `pt/aggregate_jpeg.py` | JPEG summary tables (fidelity, CV detection curve and identification, single split, transfer, diagnostic, controls); lists anything missing. |
| `pt/paper_numbers.py` | Collects every value the manuscript quotes into `$STEG_RESULTS/paper_numbers.json`, including the seed averages; no number in the paper comes from an ad-hoc command. |

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
- **[V2]** Cover and stego from the same source and, in JPEG, the same file writer (no
  cover-source or container mismatch).
- **[V3]** Splits disjoint by cover ID; TEST never used for fitting or selection. FUSION is
  split by cover into two halves (`data.fusion_subset_stems`): every spatial CNN selects its
  checkpoint on the "select" half and the stacking meta-learners train on the "meta" half. The
  JPEG single split uses all of FUSION for its meta-learner (no checkpoint selection); the JPEG
  k-fold analysis is strictly out-of-fold.
- **[V4]** Fixed global seed; every embedding seeded per image (spatial stego regenerates
  bit-exactly); best-effort deterministic cuDNN (`engine.set_seeds`); convergence-checked
  classifiers; every run logged with its git hash and environment (`run_all.sh`).
