"""Shared PyTorch training and evaluation loop: early stopping on the FUSION selection half,
one evaluation on TEST, metrics written as JSON.
"""
import sys, copy, json, datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (roc_auc_score, classification_report,
                             confusion_matrix, f1_score)

sys.path.append(str(Path(__file__).resolve().parent.parent))   # config lives in code/
import config as C

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seeds(seed=42, deterministic=True):
    """Seed every RNG the pipeline uses and, by default, ask cuDNN for deterministic
    kernels. Without the cuDNN part a rerun on the same data and seed still drifts,
    because the default algorithm selection is free to use non-deterministic reductions --
    so 'fixed seed' alone does not make a training run reproducible. Deterministic kernels
    cost some throughput; pass deterministic=False to trade it back.

    Determinism is BEST-EFFORT: use_deterministic_algorithms(warn_only=True) lets ops
    that have no deterministic CUDA kernel (e.g. pooling backward) run
    non-deterministically with a warning instead of failing, so bit-exact reruns are
    not guaranteed."""
    import os, random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:                       # older torch without warn_only
            pass


def _binary_pos_weight(labels):
    """pos_weight = N_neg/N_pos to compensate for the imbalance in BCEWithLogits."""
    n_pos = float((labels == 1).sum()); n_neg = float((labels == 0).sum())
    return torch.tensor([n_neg / max(n_pos, 1.0)], device=DEVICE)


def _ce_weights(labels, n_classes):
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    w = counts.sum() / (n_classes * np.maximum(counts, 1.0))
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


@torch.no_grad()
def _predict(model, loader, task):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        out = model(x)
        if task == "binary":
            probs.append(torch.sigmoid(out).cpu().numpy())
        else:
            probs.append(F.softmax(out, dim=1).cpu().numpy())
        ys.append(np.asarray(y))
    return np.concatenate(probs), np.concatenate(ys)


def train(model, train_loader, val_loader, *, task, n_classes=2,
          class_weights=True, label_smoothing=0.0, epochs=40, lr=1e-3,
          es_patience=None, lr_patience=None, es_min_epochs=None, monitor=None,
          smoke=False, min_delta=1e-3, lr_schedule=True):
    """
    task: 'binary' (1 logit, BCEWithLogits) or 'multiclass' (CE).
    monitor: 'auc' (max) or 'loss' (min). Default auc for binary, loss for multiclass.
    es_patience / lr_patience / es_min_epochs: None -> config ES_PATIENCE / LR_PATIENCE /
    ES_MIN_EPOCHS. During the first es_min_epochs epochs neither early stopping nor the
    LR scheduler acts (warm-up); the best checkpoint is still tracked.
    lr_schedule=False keeps the learning rate fixed (no ReduceLROnPlateau), e.g. GBRAS-Net,
    whose original protocol trains with a constant Adam learning rate.
    Returns (model with best state, best) where best has score, epoch, epochs_run.
    """
    es_patience = C.ES_PATIENCE if es_patience is None else es_patience
    lr_patience = C.LR_PATIENCE if lr_patience is None else lr_patience
    es_min_epochs = C.ES_MIN_EPOCHS if es_min_epochs is None else es_min_epochs
    model = model.to(DEVICE)
    tr_labels = np.array([y for _, y in _iter_labels(train_loader)])
    monitor = monitor or ("auc" if task == "binary" else "loss")

    if task == "binary":
        pos_w = _binary_pos_weight(tr_labels) if class_weights else None
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    else:
        w = _ce_weights(tr_labels, n_classes) if class_weights else None
        loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # AUC and accuracy are maximise metrics, loss is a minimise metric; the plateau
    # scheduler must be told the same direction as the best-checkpoint tracker below,
    # otherwise a monitor="acc" run reduces the LR while accuracy is still improving.
    maximize = monitor in ("auc", "acc")
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode=("max" if maximize else "min"),
        factor=0.5, patience=lr_patience, min_lr=1e-6) if lr_schedule else None

    best = {"score": (-1e9 if maximize else 1e9), "state": None, "epoch": -1, "epochs_run": 0}
    bad = 0
    ep = 0
    for ep in range(1, epochs + 1):
        model.train()
        for bi, (x, y) in enumerate(train_loader):
            x = x.to(DEVICE, non_blocking=True)
            if task == "binary":
                y = y.float().to(DEVICE); out = model(x); loss = loss_fn(out, y)
            else:
                y = y.long().to(DEVICE); out = model(x); loss = loss_fn(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            if smoke and bi >= 4:
                break

        # --- validation ---
        probs, yv = _predict(model, val_loader, task)
        if task == "binary":
            vloss = float(F.binary_cross_entropy(torch.tensor(probs).clamp(1e-6, 1 - 1e-6),
                                                 torch.tensor(yv).float()))
            # No fallback: validation always holds both classes, so a ValueError means a
            # broken split/loader, and a silent 0.5 would just steer model selection.
            vauc = roc_auc_score(yv, probs)
            cur = vauc if monitor == "auc" else vloss
            print(f"  ep{ep:02d}  val_loss={vloss:.4f}  val_auc={vauc:.4f}", flush=True)
        else:
            yp = probs.argmax(1)
            # probs are already softmax outputs; use NLL on log-probs (not
            # cross_entropy, which would apply log_softmax a second time).
            vloss = float(F.nll_loss(torch.log(torch.tensor(probs).clamp_min(1e-9)),
                                     torch.tensor(yv).long()))
            vacc = float((yp == yv).mean())
            cur = vacc if monitor == "acc" else vloss
            print(f"  ep{ep:02d}  val_loss={vloss:.4f}  val_acc={vacc:.4f}", flush=True)

        # Warm-up: low-payload / no-SRM runs sit at chance for the first epochs before
        # they start learning; stopping or halving the LR there (as happened at epochs
        # 9-12 with best epoch 1-4) kills the run before it begins. The scheduler only
        # sees the metric once warm-up is over.
        if sched is not None and ep >= es_min_epochs:
            sched.step(cur)

        # min_delta: only counts as improvement if it exceeds the noise.
        better = (cur > best["score"] + min_delta) if maximize else (cur < best["score"] - min_delta)
        if better:
            best.update(score=cur, state=copy.deepcopy(model.state_dict()), epoch=ep)
            bad = 0
        else:
            bad += 1
            if bad >= es_patience and ep >= es_min_epochs:
                print(f"  early-stop @ep{ep} (best ep{best['epoch']})", flush=True)
                break
        if smoke:
            break

    best["epochs_run"] = ep
    best["epochs_cap"] = epochs
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, best


def write_metrics(path, args, best, res):
    """Metrics JSON next to the model (same stem). Test metrics stay at the top level
    (what the readers already expect); run metadata is added alongside."""
    out = dict(res)
    # A run that used every epoch and found its best checkpoint in the last ES_PATIENCE epochs
    # was cut by the cap while still improving: its metrics depend on the epoch budget.
    cap = best.get("epochs_cap")
    still_improving = bool(cap is not None and best["epochs_run"] == cap
                           and best["epoch"] > cap - C.ES_PATIENCE)
    out.update(args=vars(args), best_epoch=best["epoch"], best_score=best["score"],
               epochs_run=best["epochs_run"], epochs_cap=cap,
               still_improving_at_cap=still_improving,
               timestamp=datetime.datetime.now().isoformat(timespec="seconds"))
    Path(path).write_text(json.dumps(out, indent=2, default=float))


def _iter_labels(loader):
    ds = loader.dataset
    if hasattr(ds, "samples"):
        return [(0, ds._label(idx)) for _, idx in ds.samples]
    if hasattr(ds, "index"):
        return [(0, y) for _, y, _ in ds.index]
    raise ValueError("dataset without accessible labels")


# --------------------------- final reports -----------------------------
def report_binary(model, test_loader, tag=""):
    probs, y = _predict(model, test_loader, "binary")
    pred = (probs >= 0.5).astype(int)
    print(f"\n== TEST {tag} ==")
    print(classification_report(y, pred, target_names=["COVER", "STEGO"], digits=4, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y, probs)
    # score_std separates "blind" from "collapsed": a detector that outputs a constant also
    # scores AUC 0.5, but that is a training failure, not evidence the signal is absent.
    sd = float(np.std(probs))
    print(f"AUC={auc:.4f}  P_D={tp/(tp+fn):.4f}  P_FA={fp/(fp+tn):.4f}  score_std={sd:.2e}")
    return {"auc": auc, "P_D": tp / (tp + fn), "P_FA": fp / (fp + tn), "score_std": sd}


COLLAPSE_EPS = 1e-3          # posterior std below this for every class = input-independent output
COLLAPSE_DOMINANT = 0.95     # one class taking this share of all predictions = degenerate


def collapse_diagnosis(probs, pred, n_classes):
    """Flag a multiclass model that failed to train. A network whose posteriors do not depend
    on the input, or that sends almost every image to one class, scores the chance rate by
    construction; like a constant binary detector, that is a training failure and must not
    be read as evidence that the classes cannot be identified."""
    counts = np.bincount(np.asarray(pred), minlength=n_classes)
    dominant = float(counts.max() / max(counts.sum(), 1))
    post_std = float(np.asarray(probs).std(axis=0).max())
    reasons = [r for r, bad in (
        (f"max posterior std<{COLLAPSE_EPS}", post_std < COLLAPSE_EPS),
        (f"one class takes >={COLLAPSE_DOMINANT:.0%} of predictions", dominant >= COLLAPSE_DOMINANT))
        if bad]
    return {"pred_class_counts": counts.tolist(), "dominant_pred_fraction": round(dominant, 4),
            "max_posterior_std": post_std, "collapsed": bool(reasons),
            "collapsed_reasons": reasons}


def report_multiclass(model, test_loader, class_names, tag=""):
    probs, y = _predict(model, test_loader, "multiclass")
    pred = probs.argmax(1)
    print(f"\n== TEST {tag} ==")
    print(classification_report(y, pred, target_names=class_names, digits=4, zero_division=0))
    cm = confusion_matrix(y, pred, labels=np.arange(len(class_names)))
    print("Confusion matrix (rows=true):\n", cm)
    acc = float((pred == y).mean()); mf1 = f1_score(y, pred, average="macro")
    diag = collapse_diagnosis(probs, pred, len(class_names))
    print(f"acc={acc:.4f}  macro_F1={mf1:.4f}  predicted-class counts={diag['pred_class_counts']}")
    if diag["collapsed"]:
        print(f"WARNING: COLLAPSED ({'; '.join(diag['collapsed_reasons'])}): training failure, "
              f"not an identification result")
    return {"acc": acc, "macro_f1": mf1, "cm": cm.tolist(), **diag}
