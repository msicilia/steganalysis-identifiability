"""Fidelity check of the JPEG simulators, in two stages:
  extract  decodes covers, embeds, writes real JPGs, extracts DCTR -> npy
  analyze  reads the npy -> cover-vs-one AUC + pairwise separability
Run in the project venv. Usage:
  uv run python validate_embedders.py extract 200
  uv run python validate_embedders.py analyze
"""
import sys, glob, shutil
from pathlib import Path
import numpy as np

ALPHA, Q = 0.4, 75
SCHEMES = ["UERD", "UED", "JMIPOD", "nsF5"]   # J-UNIWARD omitted for slowness
ADAPT = ["UERD", "UED", "JMIPOD"]
TMP = Path("/tmp/embval")


def extract(N):
    import torch, torch.nn.functional as F
    from PIL import Image
    import jpeglib
    sys.path.append(str(Path(__file__).resolve().parent))         # code/pt (jpeg_embedders)
    sys.path.append(str(Path(__file__).resolve().parent.parent))  # code/ (config)
    import config as C
    from jpeg_embedders import EMBEDDERS
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    shutil.rmtree(TMP, ignore_errors=True); TMP.mkdir(parents=True)
    covers = sorted(glob.glob(str(C.DATASETS / "covers" / "BOSSbase_1.01" / "*.pgm")))[:N]
    print(f"[extract] N={len(covers)} alpha={ALPHA} Q={Q} dev={DEV}")
    rates = {s: [] for s in SCHEMES}
    for pgm in covers:
        stem = Path(pgm).stem
        cov = TMP / "COVER"; cov.mkdir(exist_ok=True)
        cjpg = cov / f"{stem}.jpg"
        Image.open(pgm).convert("L").save(cjpg, quality=Q)
        im = jpeglib.read_dct(str(cjpg))
        Yn = im.Y.astype(np.int32); q = im.qt[0].astype(np.float64)
        n_nz = int(((Yn != 0) & (np.arange(64).reshape(8, 8) != 0)[None, None]).sum())
        spatial = np.asarray(Image.open(cjpg).convert("L"))
        seed = int(stem) if stem.isdigit() else 12345
        for s in SCHEMES:
            Ys, nch = EMBEDDERS[s](Yn, q, ALPHA, seed=seed, spatial=spatial)
            d = TMP / s; d.mkdir(exist_ok=True)
            im.Y = Ys.astype(im.Y.dtype)
            im.write_dct(str(d / f"{stem}.jpg"))
            rates[s].append(nch / max(n_nz, 1))
    # DCTR (inline; numpy/torch only)
    x = np.arange(8)
    cc = lambda k: np.sqrt(1/8) if k == 0 else np.sqrt(2/8)
    B = np.zeros((64, 8, 8), np.float32)
    for i in range(8):
        for j in range(8):
            B[i*8+j] = np.outer(cc(i)*np.cos((2*x+1)*i*np.pi/16), cc(j)*np.cos((2*x+1)*j*np.pi/16))
    W = torch.from_numpy(B)[:, None].to(DEV)

    def dctr(folder):
        paths = sorted(glob.glob(str(TMP / folder / "*.jpg")))
        out = np.zeros((len(paths), 64*64*5), np.float32)
        for i in range(0, len(paths), 32):
            imgs = torch.stack([torch.from_numpy(np.asarray(Image.open(p).convert("L"), np.float32))
                                for p in paths[i:i+32]])[:, None].to(DEV)
            R = F.conv2d(imgs, W); Qm = torch.clamp(torch.round(R.abs()/4.0), 0, 4).to(torch.int8)
            feats = []
            for pr in range(8):
                for pc in range(8):
                    oneh = F.one_hot(Qm[:, :, pr::8, pc::8].long(), 5).float()
                    feats.append(oneh.sum(dim=(2, 3)))
            out[i:i+imgs.shape[0]] = torch.stack(feats, 1).reshape(imgs.shape[0], -1).cpu().numpy()
        return out

    for folder in ["COVER"] + SCHEMES:
        np.save(TMP / f"X_{folder}.npy", dctr(folder))
    np.save(TMP / "rates.npy", {s: float(np.mean(rates[s])) for s in SCHEMES}, allow_pickle=True)
    print("[extract] done ->", TMP)


def analyze():
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score
    # FULL DCTR features (no PCA) + regularized LogReg.
    X = {f: np.load(TMP / f"X_{f}.npy") for f in ["COVER"] + SCHEMES}
    rates = np.load(TMP / "rates.npy", allow_pickle=True).item()

    def fit(A, Bx):
        Xx = np.vstack([A, Bx]); yy = np.r_[np.zeros(len(A)), np.ones(len(Bx))]
        Xtr, Xte, ytr, yte = train_test_split(Xx, yy, test_size=0.5, random_state=0, stratify=yy)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=0.05, max_iter=3000).fit(sc.transform(Xtr), ytr)
        return clf, sc, Xte, yte

    print("\n-- change rate --")
    for s in SCHEMES:
        print(f"  {s:9s} rate={rates[s]:.4f}")
    print("\n-- cover-vs-one AUC --")
    for s in SCHEMES:
        clf, sc, Xte, yte = fit(X["COVER"], X[s])
        print(f"  COVER vs {s:9s} AUC={roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:,1]):.4f}")
    print("\n-- ADAPTIVE pairwise separability (acc) --")
    for i in range(len(ADAPT)):
        for j in range(i+1, len(ADAPT)):
            clf, sc, Xte, yte = fit(X[ADAPT[i]], X[ADAPT[j]])
            print(f"  {ADAPT[i]:9s} vs {ADAPT[j]:9s} acc={accuracy_score(yte, clf.predict(sc.transform(Xte))):.3f}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "extract"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    (extract(n) if mode == "extract" else analyze())


if __name__ == "__main__":
    main()
