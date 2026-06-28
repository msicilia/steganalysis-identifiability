"""Cross-dataset JPEG (cover-source mismatch).
Trains the DCTR detector (cover-vs-all) on BOSSbase and evaluates it on ALASKA#2,
comparing the cross-source AUC against the in-distribution AUC."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))          # code/pt (extract_dctr, data)
sys.path.append(str(Path(__file__).resolve().parent.parent))   # code/ (config)
import numpy as np, torch
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
import config as C
from extract_dctr import dct_basis_filters, dctr_features_batch
from data import list_images

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOSS = C.DATASETS / "DATA_JPEG_q75_a040_DCTR"
ALASKA = C.DATASETS / "DATA_JPEG_ALASKA2_q75_a040"
NAMES = ["COVER", "JUNIWARD", "UERD", "nsF5"]      # CLASS_FOLDERS order (jpeg)
q, T = 4.0, 4

@torch.no_grad()
def feats(folder, W):
    ps = list_images(folder); X = np.zeros((len(ps), 64*64*(T+1)), np.float32)
    for s in range(0, len(ps), 64):
        b = ps[s:s+64]
        imgs = torch.stack([torch.from_numpy(np.asarray(Image.open(p).convert("L"), np.float32))
                            for p in b])[:, None].to(DEVICE)
        X[s:s+len(b)] = dctr_features_batch(imgs, W, q, T)
    return X


def main():
    # detector trained on BOSSbase (cover-vs-all)
    Xtr = np.load(BOSS / "X_train.npy"); ytr = np.load(BOSS / "y_train.npy")
    Xte = np.load(BOSS / "X_test.npy");  yte = np.load(BOSS / "y_test.npy")
    sc = StandardScaler().fit(Xtr)
    det = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(
        sc.transform(Xtr), (ytr != 0).astype(int))
    sin = det.predict_proba(sc.transform(Xte))[:, 1]
    print(f"[in-dist BOSS] detection AUC = {roc_auc_score((yte != 0).astype(int), sin):.4f}")

    # DCTR of ALASKA
    W = dct_basis_filters().to(DEVICE)
    Xa, ya = [], []
    for i, nm in enumerate(NAMES):
        Xn = feats(ALASKA / nm, W); Xa.append(Xn); ya += [i]*len(Xn); print(f"  ALASKA {nm}: {len(Xn)} imgs")
    Xa = np.concatenate(Xa); ya = np.array(ya)

    # cross-eval
    sa = det.predict_proba(sc.transform(Xa))[:, 1]; ybin = (ya != 0).astype(int)
    pred = (sa >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(ybin, pred, labels=[0, 1]).ravel()
    print(f"[CROSS ALASKA] detection AUC = {roc_auc_score(ybin, sa):.4f}  "
          f"P_D={tp/(tp+fn):.4f}  P_FA={fp/(fp+tn):.4f}")
    cov = sa[ya == 0]
    for i, nm in enumerate(NAMES):
        if i == 0:
            continue
        s = np.r_[cov, sa[ya == i]]; y = np.r_[np.zeros(len(cov)), np.ones((ya == i).sum())]
        print(f"  cover-vs-{nm:9s}: AUC = {roc_auc_score(y, s):.4f}")


if __name__ == "__main__":
    main()
