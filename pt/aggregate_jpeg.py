"""Summary tables and completeness check of the JPEG pipeline from its result JSONs. Missing
files or keys are listed as MISSING and the script exits non-zero.

Usage:  STEG_DOMAIN=jpeg python aggregate_jpeg.py [--alpha 0.4]
"""
import sys, json, argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))   # code/ (config)
import config as C

MISSING = []


def load(path):
    path = Path(path)
    if not path.exists():
        MISSING.append(str(path))
        return None
    return json.loads(path.read_text())


def need(d, key, where):
    """d[key], or None with the absence recorded in MISSING."""
    if d is None:
        return None
    if key not in d:
        MISSING.append(f"{where}:{key}")
        return None
    return d[key]


def pci(v):
    """[point, [lo, hi]] -> 'point [lo, hi]'."""
    if v is None:
        return "MISSING"
    return f"{v[0]:.3f} [{v[1][0]:.3f}, {v[1][1]:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=C.FAMILY_ID_PAYLOAD,
                    help="payload of the identification / transfer / separability analyses")
    args = ap.parse_args()
    assert C.DOMAIN == "jpeg", "Run with STEG_DOMAIN=jpeg"
    at = C.alpha_tag(args.alpha)
    R = C.RESULTS_DIR

    # ---- simulator fidelity ----
    fid = load(R / "embedder_fidelity.json")
    if fid:
        print("## Simulator fidelity (400 covers, disjoint halves)\n")
        for s, v in need(fid, "home_vs_reference", "embedder_fidelity.json").items():
            print(f"- {s}: home AUC={v['home_auc']:.3f} reference AUC={v['reference_auc']:.3f} "
                  f"|Δ|={v['abs_delta']:.3f}")
        for pair, acc in need(fid, "adaptive_pairwise_acc", "embedder_fidelity.json").items():
            print(f"- adaptive pair {pair}: acc={acc:.3f}")

    # ---- CV detection curve ----
    print("\n## Detection, cover vs any stego (5-fold CV by cover, AUC [95% CI])\n")
    print("| payload (bpnzAC) | AUC |\n|--|--|")
    for a in C.PAYLOADS:
        kf = load(R / f"jpeg_kfold_{C.alpha_tag(a)}.json")
        print(f"| {a:.2f} | {pci(need(kf, 'detection_auc', f'jpeg_kfold_{C.alpha_tag(a)}.json'))} |")

    # ---- CV identification ----
    kfn = f"jpeg_kfold_{at}.json"
    kf = load(R / kfn)
    if kf:
        print(f"\n## Identification @{args.alpha} (CV, accuracy [95% CI])\n")
        print(f"- monolithic: {pci(need(kf, 'monolithic_acc', kfn))}")
        print(f"- modular:    {pci(need(kf, 'modular_acc', kfn))}")
        print(f"- Δ modular−monolithic: {pci(need(kf, 'delta_modular_minus_mono', kfn))}")
        for s, v in (need(kf, "specialist_auc", kfn) or {}).items():
            print(f"- specialist cover-vs-{s} AUC: {pci(v)}")
        need(kf, "fisher_pairs_pca50", kfn)
        need(kf, "monolithic_cm", kfn)

    # ---- single split ----
    en = load(R / f"dctr_ensemble_{at}.json")
    bs = load(R / f"dctr_bootstrap_{at}.json")
    if en and bs:
        print(f"\n## Single split @{args.alpha} (TEST)\n")
        b = need(bs, "bootstrap_CI95", f"dctr_bootstrap_{at}.json") or {}
        for k in ("detection_auc", "monolithic_acc", "modular_acc", "delta_modular_minus_mono"):
            print(f"- {k}: {pci(b.get(k))}")
        mono = en["monolithic"]
        cm = mono["cm"]
        recall = {n: cm[i][i] / max(sum(cm[i]), 1) for i, n in enumerate(en["classes"])}
        print("- monolithic recall: " + ", ".join(f"{n}={r:.3f}" for n, r in recall.items()))
        inv = need(bs, "ceiling_by_classifier", f"dctr_bootstrap_{at}.json") or {}
        for nm, row in inv.items():
            if "error" in row:
                print(f"- ceiling {nm}: FAILED ({row['error'][:80]})")
            else:
                print(f"- ceiling {nm}: mono={row['mono_acc']:.3f} modular={row['modular_acc']:.3f} "
                      f"Δ={row['delta']:+.3f}")
        # A classifier that cannot be fitted on this feature is a result about the feature,
        # not an absent value: LinearSVC does not converge on the 20,480-d DCTR even at
        # max_iter=80000, and that is reported rather than treated as a gap in the campaign.
        if bs.get("ceiling_by_classifier_failed"):
            print(f"- ceiling: no usable result for {bs['ceiling_by_classifier_failed']} "
                  f"(did not converge); the comparison rests on the remaining families")

    # ---- transfer ----
    tr = load(R / f"transfer_jpeg_{at}.json")
    if tr:
        print(f"\n## Leave-one-scheme-out transfer @{args.alpha} (single split, AUC)\n")
        print("| held out | transfer | in-dist (cover-vs-all) | adaptive-only transfer |\n|--|--|--|--|")
        for s, v in tr.items():
            if s == "provenance":
                continue
            ad = v.get("adaptive_only")
            ad_s = "—" if ad is None else f"{ad['transfer_auc']:.3f}"
            print(f"| {s} | {v['transfer_auc']:.3f} {v['transfer_ci95']} | {v['indist_auc']:.3f} | {ad_s} |")

    # ---- diagnostic ----
    dg = load(R / f"diagnostic_jpeg_{at}.json")
    if dg:
        print(f"\n## Does pairwise Fisher J predict pairwise confusion? @{args.alpha}\n")
        for src, block in dg["sources"].items():
            for label, res in block.items():
                if isinstance(res, dict):
                    print(f"- {src} / {label}: Spearman r={res['spearman_r']} "
                          f"Mantel p={res['mantel_p_one_sided_negative']} "
                          f"(min attainable {res['min_attainable_p']})")

    # ---- figure ----
    for ext in ("pdf", "png"):
        if not (R / f"fig_fisher_heatmap_jpeg.{ext}").exists():
            MISSING.append(str(R / f"fig_fisher_heatmap_jpeg.{ext}"))

    # ---- negative controls ----
    print(f"\n## Negative controls @{args.alpha} (cover vs any stego, TEST)\n")
    for label, stem in (("pixel-domain SRM CNN", f"pt_binary_{at}_s{C.SEED}"),
                        ("DCT-coefficient CNN", f"pt_dct_binary_{at}_s{C.SEED}")):
        j = load(C.MODEL_DIR / f"{stem}.json")
        if j:
            sd = j.get("score_std")
            flag = " (COLLAPSED: constant scores)" if sd is not None and sd < 1e-3 else ""
            print(f"- {label}: AUC={j['auc']:.3f} score_std={sd}{flag}")

    miss = sorted(set(MISSING))
    if miss:
        print(f"\nMISSING ({len(miss)}):")
        for m in miss:
            print("  -", m)
        sys.exit(1)
    print("\nall JPEG outputs present")


if __name__ == "__main__":
    main()
