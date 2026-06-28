"""
Common training/evaluation loop (PyTorch).

Model selection and early-stopping ONLY on the validation set (fusion); the test
set is evaluated ONCE at the end. Returns the best state.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (roc_auc_score, classification_report,
                             confusion_matrix, f1_score)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seeds(seed=42):
    import os, random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


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
          es_patience=8, lr_patience=4, monitor=None, smoke=False, min_delta=1e-3):
    """
    task: 'binary' (1 logit, BCEWithLogits) or 'multiclass' (CE).
    monitor: 'auc' (max) or 'loss' (min). Default auc for binary, loss for multiclass.
    """
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
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode=("max" if monitor == "auc" else "min"),
        factor=0.5, patience=lr_patience, min_lr=1e-6)

    maximize = monitor in ("auc", "acc")
    best = {"score": (-1e9 if maximize else 1e9), "state": None, "epoch": -1}
    bad = 0
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
            try:
                vauc = roc_auc_score(yv, probs)
            except ValueError:
                vauc = 0.5
            cur = vauc if monitor == "auc" else vloss
            print(f"  ep{ep:02d}  val_loss={vloss:.4f}  val_auc={vauc:.4f}")
        else:
            yp = probs.argmax(1)
            vloss = float(F.cross_entropy(torch.tensor(np.log(probs + 1e-9)),
                                          torch.tensor(yv).long()))
            vacc = float((yp == yv).mean())
            cur = vacc if monitor == "acc" else vloss
            print(f"  ep{ep:02d}  val_loss={vloss:.4f}  val_acc={vacc:.4f}")

        sched.step(cur)

        # min_delta: only counts as improvement if it exceeds the noise.
        better = (cur > best["score"] + min_delta) if maximize else (cur < best["score"] - min_delta)
        if better:
            best.update(score=cur, state=copy.deepcopy(model.state_dict()), epoch=ep)
            bad = 0
        else:
            bad += 1
            if bad >= es_patience:
                print(f"  early-stop @ep{ep} (best ep{best['epoch']})")
                break
        if smoke:
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, best


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
    print(f"AUC={auc:.4f}  P_D={tp/(tp+fn):.4f}  P_FA={fp/(fp+tn):.4f}")
    return {"auc": auc, "P_D": tp / (tp + fn), "P_FA": fp / (fp + tn)}


def report_multiclass(model, test_loader, class_names, tag=""):
    probs, y = _predict(model, test_loader, "multiclass")
    pred = probs.argmax(1)
    print(f"\n== TEST {tag} ==")
    print(classification_report(y, pred, target_names=class_names, digits=4, zero_division=0))
    cm = confusion_matrix(y, pred, labels=np.arange(len(class_names)))
    print("Confusion matrix (rows=true):\n", cm)
    acc = float((pred == y).mean()); mf1 = f1_score(y, pred, average="macro")
    print(f"acc={acc:.4f}  macro_F1={mf1:.4f}")
    return {"acc": acc, "macro_f1": mf1, "cm": cm.tolist()}
