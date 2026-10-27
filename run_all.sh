#!/usr/bin/env bash
# Full pipeline, strictly sequential (one GPU job at a time), resumable.
#
#   ./run_all.sh spatial     # spatial domain only
#   ./run_all.sh jpeg        # JPEG domain only
#   ./run_all.sh all         # both (default)
#
# Every step logs to logs/<run-id>/<step>.log and, on success, touches
# $STEG_RESULTS/.done/<step>. A rerun skips steps that already have a marker, so an
# interrupted campaign resumes where it stopped; delete a marker (or set FORCE=1) to redo
# a step. Output is unbuffered, so a log that stops growing means the process is really
# idle -- check `nvidia-smi` before concluding a long training step is stuck.
#
# Point STEG_MODELS / STEG_RESULTS at fresh directories for a rerun, so stale files from
# earlier code can never be mixed into the new results.
set -euo pipefail
cd "$(dirname "$0")"

PHASE=${1:-all}
PY=${PY:-.venv/bin/python}
ALPHAS="0.1,0.2,0.4"
ID_ALPHA=0.4                                    # identification / ensemble / transfer payload
SPATIAL_ALGOS="HILL HUGO MiPOD S-UNIWARD WOW"
# Training seeds of the identification networks (collapse vs learning is a central contrast,
# so it must not rest on one seed). The first must be config.SEED: the ensemble, separability
# and figure steps load that model.
ID_SEEDS="42 1 2"
DATA=${STEG_DATA:-data}
# Models and results must go to directories dedicated to this campaign: the defaults
# (models/, results/) hold files from earlier code, which aggregate.py would read and
# transfer_spatial.py would reuse.
if [[ -z "${STEG_MODELS:-}" || -z "${STEG_RESULTS:-}" ]]; then
  echo "ABORT: set STEG_MODELS and STEG_RESULTS to fresh directories for this campaign, e.g."
  echo "  STEG_MODELS=../runs/<campaign>/models STEG_RESULTS=../runs/<campaign>/results $0 $PHASE"
  exit 1
fi
MODELS=$STEG_MODELS
RESULTS=$STEG_RESULTS
DONE=$RESULTS/.done
# Resuming is fine (completion markers exist); a non-empty directory WITHOUT markers holds
# files from another campaign. ALLOW_NONEMPTY=1 overrides, e.g. after a step that failed
# after writing its outputs.
if [[ -z "$(ls -A "$DONE" 2>/dev/null)" && "${ALLOW_NONEMPTY:-0}" != 1 ]] \
   && [[ -n "$(ls -A "$MODELS" 2>/dev/null)" || -n "$(ls -A "$RESULTS" 2>/dev/null)" ]]; then
  echo "ABORT: $MODELS or $RESULTS is not empty but has no completion markers in $DONE."
  echo "Point STEG_MODELS/STEG_RESULTS at fresh directories (or set ALLOW_NONEMPTY=1)."
  exit 1
fi
RUN_ID=$(date +%Y%m%d_%H%M%S)
LOGDIR=${STEG_LOGS:-logs}/$RUN_ID
mkdir -p "$LOGDIR" "$DONE" "$MODELS"
export PYTHONUNBUFFERED=1

# ---- provenance: what code and environment produced this run ----
{
  echo "run_id: $RUN_ID"; echo "phase: $PHASE"; date -Iseconds
  echo "git: $(git rev-parse HEAD 2>/dev/null || echo 'not a git repo')"
  echo "uncommitted changes:"; git status --porcelain 2>/dev/null || true
  echo "STEG_DATA=$DATA STEG_MODELS=$MODELS STEG_RESULTS=$RESULTS"
  "$PY" -m pip freeze 2>/dev/null || uv pip freeze --python "$PY" 2>/dev/null || true
  nvidia-smi --query-gpu=name,driver_version --format=csv 2>/dev/null || true
} > "$LOGDIR/provenance.txt"
if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
  echo "WARNING: uncommitted changes -- commit first so results map to a code version."
fi

step() {                                         # step <name> <command...>
  local name=$1; shift
  if [[ -e "$DONE/$name" && "${FORCE:-0}" != 1 ]]; then
    echo "[skip] $name"; return
  fi
  echo "[$(date +%H:%M:%S)] $name  ->  $LOGDIR/$name.log"
  if "$@" > "$LOGDIR/$name.log" 2>&1; then
    touch "$DONE/$name"
  else
    echo "FAILED: $name (see $LOGDIR/$name.log)"; exit 1
  fi
}

run_spatial() {
  export STEG_DOMAIN=spatial
  step spatial_gen      "$PY" gen_spatial.py --alphas "$ALPHAS"
  step spatial_splits   "$PY" make_splits.py --alphas "$ALPHAS"
  # Model-free separability needs no trained model: run it first (CPU) and read its
  # control_cover_vs_stego block -- if the linear probe cannot tell cover from stego, the
  # 28-d SRM statistics are blind and stego-vs-stego overlap there means nothing.
  step separability_srm "$PY" pt/separability_spatial.py --alpha "$ID_ALPHA" --features srm
  # Detection curve with a payload curriculum: 0.4 bpp from scratch, then each lower
  # payload starts from the next higher one (training from scratch at 0.1 bpp collapsed).
  step "train_binary_a0.4" "$PY" pt/train_binary.py --alpha 0.4
  step "train_binary_a0.2" "$PY" pt/train_binary.py --alpha 0.2 --init "$MODELS/pt_binary_a040_s42.pt"
  step "train_binary_a0.1" "$PY" pt/train_binary.py --alpha 0.1 --init "$MODELS/pt_binary_a020_s42.pt"
  # no-SRM ablation: patience = EPOCHS_BIN (80) so it trains every epoch and the "did not
  # learn" claim rests on the full run, not on an early stop (an early-stopped run gave the
  # same AUC, 0.5002).
  step train_binary_nosrm "$PY" pt/train_binary.py --alpha "$ID_ALPHA" --no-srm --es-patience 80
  for s in $ID_SEEDS; do
    step "train_family_s$s" "$PY" pt/train_family.py --alpha "$ID_ALPHA" --seed "$s"
  done
  for g in $SPATIAL_ALGOS; do
    step "train_spec_$g"  "$PY" pt/train_pair_specialist.py --algo "$g" --alpha "$ID_ALPHA"
    step "train_gbras_$g" "$PY" pt/train_gbras.py --algo "$g" --alpha "$ID_ALPHA"
  done
  step train_gbraswh_detect "$PY" pt/train_gbras_wh.py --mode detect --alpha "$ID_ALPHA"
  for s in $ID_SEEDS; do
    step "train_gbraswh_family_s$s" "$PY" pt/train_gbras_wh.py --mode family --alpha "$ID_ALPHA" --seed "$s"
  done
  step build_ensemble       "$PY" pt/build_ensemble.py --alpha "$ID_ALPHA"
  step build_gbras_ensemble "$PY" pt/build_gbras_ensemble.py --alpha "$ID_ALPHA"
  step transfer_spatial     "$PY" pt/transfer_spatial.py --alpha "$ID_ALPHA"
  step spatial_bootstrap    "$PY" pt/spatial_bootstrap.py
  step diagnostic_spatial   "$PY" pt/diagnostic_vs_confusion.py --domain spatial --alpha "$ID_ALPHA"
  step fig_spatial          "$PY" pt/fig_fisher_heatmap.py --domain spatial
  step aggregate_spatial    "$PY" pt/aggregate.py --alpha "$ID_ALPHA"
}

run_jpeg() {
  export STEG_DOMAIN=jpeg
  # The generators skip files that already exist. JPEG data made before the per-scheme
  # seeds and the shared container writer would be silently mixed with new files. Current
  # data carries a sentinel (jpeg_embedders.GEN_SENTINEL) written when gen_jpeg.py creates
  # the folder, so an interrupted generation resumes while old data (or derivatives of it)
  # is refused.
  for p in "$DATA"/DATA_JPEG_q75_a0*; do
    [[ -e "$p" ]] || continue
    root=$(echo "$p" | sed -E 's/(_SPLITS|_DCTR|_DCT|_KFOLD\.npz)$//')
    if [[ ! -e "$root/.generator_v2" ]]; then
      echo "ABORT: $p comes from an earlier code version ($root has no .generator_v2)."
      echo "Delete DATA_JPEG_q75_aXXX and its _SPLITS/_DCTR/_DCT/_KFOLD.npz derivatives first."
      exit 1
    fi
  done
  # Simulator fidelity FIRST: the validator embeds its own 400 covers, so it needs no dataset,
  # and the full generation below is only worth its compute once the report has been read.
  step jpeg_validate_ext "$PY" pt/validate_embedders.py extract --n 400
  step jpeg_validate_ana "$PY" pt/validate_embedders.py analyze
  if [[ ! -e "$DONE/jpeg_fidelity_reviewed" ]]; then
    echo "PAUSED: review $RESULTS/embedder_fidelity.json (change rates, cover-vs-one AUCs,"
    echo "adaptive pairwise accuracies, home-vs-conseal |dAUC|) before generating the dataset."
    echo "To continue: touch $DONE/jpeg_fidelity_reviewed && $0 $PHASE"
    exit 0
  fi
  step jpeg_gen_home     "$PY" pt/gen_jpeg.py --embedders UED,JMIPOD --alphas "$ALPHAS"
  step jpeg_gen_conseal  "$PY" pt/gen_jpeg_conseal.py --alphas "$ALPHAS"
  step jpeg_splits       "$PY" make_splits.py --alphas "$ALPHAS"
  # The split feature cache is read only by the single-split analyses, which run at
  # ID_ALPHA; jpeg_kfold_cv extracts its own cache for every payload.
  step "extract_dctr_a$ID_ALPHA" "$PY" pt/extract_dctr.py --alpha "$ID_ALPHA"
  # Timing probe before the long logistic-regression campaigns; read its log to plan.
  step time_logreg     "$PY" pt/time_logreg.py --alpha "$ID_ALPHA"
  # Second gate: the steps below are the multi-day logistic-regression campaign, so the
  # extrapolated runtime is read before committing the CPU to it.
  if [[ ! -e "$DONE/jpeg_runtime_reviewed" ]]; then
    tl=$(ls -t "${STEG_LOGS:-logs}"/*/time_logreg.log 2>/dev/null | head -1)
    echo "PAUSED: read the runtime estimate in ${tl:-logs/<run-id>/time_logreg.log} before the"
    echo "logistic-regression campaign (dctr_single_split, jpeg_kfold_a*)."
    echo "To continue: touch $DONE/jpeg_runtime_reviewed && $0 $PHASE"
    exit 0
  fi
  # One process fits every single-split model once and writes dctr_ensemble_, dctr_bootstrap_
  # and transfer_jpeg_ JSONs (running dctr_ensemble.py / transfer_jpeg.py too would refit them).
  step dctr_single_split "$PY" pt/dctr_bootstrap.py --alpha "$ID_ALPHA"
  # CV: the low payloads feed only the detection curve. Transfer is not cross-validated
  # because the paper's transfer table comes from the single split above; add ",transfer"
  # to the ID_ALPHA parts if the CV transfer is wanted (+5 fits per fold).
  for a in ${ALPHAS//,/ }; do
    if [[ "$a" == "$ID_ALPHA" ]]; then parts=detection,identification,separability
    else parts=detection; fi
    step "jpeg_kfold_a$a" "$PY" pt/jpeg_kfold_cv.py --alpha "$a" --folds 5 --parts "$parts"
  done
  # Sensitivity of the pairwise Fisher J to the PCA dimension (20..500 components), with the
  # diagnostic-vs-confusion test at each k: all rows vs the out-of-fold confusion, and the
  # a priori protocol (TRAIN rows of the single split vs its TEST confusion). The manuscript
  # reports 200 components; fig_jpeg reads that table.
  step fisher_pca_sweep         "$PY" pt/fisher_pca_sweep.py --alpha "$ID_ALPHA"
  step fisher_pca_sweep_apriori "$PY" pt/fisher_pca_sweep.py --alpha "$ID_ALPHA" --protocol apriori
  step fig_jpeg        "$PY" pt/fig_fisher_heatmap.py --domain jpeg --jpeg-k 200
  step diagnostic_jpeg "$PY" pt/diagnostic_vs_confusion.py --domain jpeg --alpha "$ID_ALPHA"
  # Negative controls behind the paper's choice of DCTR: a pixel-domain SRM CNN and a CNN on
  # the quantised DCT coefficients, both cover-vs-any-stego. Their JSONs carry score_std, so
  # a collapsed (constant) detector is distinguishable from a blind one.
  step jpeg_ctrl_spatial_cnn "$PY" pt/train_binary.py --alpha "$ID_ALPHA"
  step jpeg_gen_dct          "$PY" pt/gen_dct.py --alpha "$ID_ALPHA"
  step jpeg_ctrl_dct_cnn     "$PY" pt/train_dct.py --mode binary --alpha "$ID_ALPHA"
  step aggregate_jpeg        "$PY" pt/aggregate_jpeg.py --alpha "$ID_ALPHA"
}

case "$PHASE" in
  spatial) run_spatial ;;
  jpeg)    run_jpeg ;;
  all)     run_spatial; run_jpeg ;;
  *) echo "usage: $0 [spatial|jpeg|all]"; exit 2 ;;
esac
echo "done: $PHASE (logs in $LOGDIR)"
