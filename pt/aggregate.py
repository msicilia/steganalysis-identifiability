"""Summary tables of the spatial pipeline from the model and result JSONs. Missing files are
listed as MISSING and the script exits non-zero.

Usage:  python aggregate.py [--alpha 0.4]      # prints Markdown, writes results/summary.csv
"""
import sys, csv, json, argparse, re
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))   # code/ (config)
import numpy as np
import config as C

ALGOS = [c for c in C.CLASS_FOLDERS if c != C.COVER_NAME]
MISSING = []


def load(path, required=True):
    """JSON at `path`, or None. A missing required file is recorded in MISSING (exit 1);
    optional files are outputs run_all.sh does not produce."""
    path = Path(path)
    if not path.exists():
        if required:
            MISSING.append(str(path))
        return None
    return json.loads(path.read_text())


def f3(v):
    return "MISSING" if v is None else f"{v:.3f}"


def ci(c):
    return "" if not c else f" [{c[0]:.3f}, {c[1]:.3f}]"


def seeds_for(pattern):
    """Seed ids present for a model-JSON glob pattern ending in _s<seed>.json."""
    out = []
    for p in C.MODEL_DIR.glob(pattern):
        m = re.search(r"_s(\d+)\.json$", p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(set(out))


def over_seeds(stem, key):
    """(mean, std, n, value at C.SEED) of `key` across pt_<stem>_s*.json.
    The C.SEED file is required (reported MISSING if absent); other seeds are optional."""
    load(C.MODEL_DIR / f"{stem}_s{C.SEED}.json")          # records MISSING if absent
    vals = {s: json.loads((C.MODEL_DIR / f"{stem}_s{s}.json").read_text()).get(key)
            for s in seeds_for(f"{stem}_s*.json")}
    v = [x for x in vals.values() if x is not None]
    if not v:
        return None, None, 0, None
    return float(np.mean(v)), float(np.std(v)), len(v), vals.get(C.SEED)


def collapsed_seeds(stem):
    """Seeds of pt_<stem>_s*.json whose multiclass model is flagged collapsed (training
    failure: constant or single-class predictions; engine.collapse_diagnosis)."""
    return [s for s in seeds_for(f"{stem}_s*.json")
            if json.loads((C.MODEL_DIR / f"{stem}_s{s}.json").read_text()).get("collapsed")]


def print_collapsed(stem):
    bad = collapsed_seeds(stem)
    if bad:
        print(f"  WARNING: seeds {bad} COLLAPSED (constant/single-class predictions): training "
              f"failures, not identification limits; their accuracy is in the mean above")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=C.FAMILY_ID_PAYLOAD,
                    help="payload of the fixed-payload analyses (family, ensembles, transfer)")
    args = ap.parse_args()
    at = C.alpha_tag(args.alpha)
    rows = []

    # ---- models: binary detection sweep ----
    print(f"\n## Binary detection (TEST) — model JSONs in {C.MODEL_DIR}\n")
    print("| payload | front-end | AUC (mean±std over seeds) | AUC seed " f"{C.SEED} | P_D | P_FA | n seeds |")
    print("|--|--|--|--|--|--|--|")
    variants = [(a, "") for a in C.PAYLOADS] + [(C.ACTIVE_ALPHA, "_nosrm")]
    for a, sfx in variants:
        stem = f"pt_binary_{C.alpha_tag(a)}{sfx}"
        m, s, n, v = over_seeds(stem, "auc")
        j = load(C.MODEL_DIR / f"{stem}_s{C.SEED}.json") if n else None
        fe = "SRM" if not sfx else "no SRM (abl.)"
        ms = "MISSING" if m is None else f"{m:.3f}±{s:.3f}"
        print(f"| {a:.2f} | {fe} | {ms} | {f3(v)} | {f3(j and j.get('P_D'))} | "
              f"{f3(j and j.get('P_FA'))} | {n} |")
        rows.append(["binary", a, fe, m, s, n])

    # ---- models: family (multiclass CNN) ----
    print("\n## Monolithic multiclass CNN (TEST)\n")
    print("| payload | acc (mean±std) | macro-F1 (mean±std) | n seeds |\n|--|--|--|--|")
    stem = f"pt_family_{at}"
    ma, sa, n, _ = over_seeds(stem, "acc")
    mf, sf, _, _ = over_seeds(stem, "macro_f1")
    print(f"| {args.alpha:.2f} | {'MISSING' if ma is None else f'{ma:.3f}±{sa:.3f}'} | "
          f"{'MISSING' if mf is None else f'{mf:.3f}±{sf:.3f}'} | {n} |")
    print_collapsed(stem)
    rows.append(["family_cnn", args.alpha, "acc", ma, sa, n])

    # ---- models: pair specialists (BinaryCNN) and GBRAS per algorithm ----
    print(f"\n## Per-scheme detectors, seed {C.SEED} (TEST AUC)\n")
    print("| scheme | CNN pair specialist | GBRAS cover-vs-scheme (per image) | transfer (held out) |\n|--|--|--|--|")
    for algo in ALGOS:
        sp = load(C.MODEL_DIR / f"pt_spec_{algo}_{at}_s{C.SEED}.json")
        gb = load(C.MODEL_DIR / f"pt_gbras_{algo}_{at}_s{C.SEED}.json")
        tr = load(C.MODEL_DIR / f"pt_transfer_{algo}_{at}_s{C.SEED}.json")
        g = lambda j, k="auc": None if j is None else j.get(k)
        print(f"| {algo} | {f3(g(sp))} | {f3(g(gb, 'auc_image'))} | {f3(g(tr))} |")
        rows += [["spec", args.alpha, algo, g(sp), None, 1],
                 ["gbras_image", args.alpha, algo, g(gb, "auc_image"), None, 1]]

    # ---- models: GBRAS whole-image detect / family ----
    print(f"\n## GBRAS generalist and 6-class, seed {C.SEED} (TEST)\n")
    gd = load(C.MODEL_DIR / f"pt_gbraswh_detect_{at}_s{C.SEED}.json")
    gf = load(C.MODEL_DIR / f"pt_gbraswh_family_{at}_s{C.SEED}.json")
    print(f"- detect: AUC={f3(gd and gd.get('auc'))}  P_D={f3(gd and gd.get('P_D'))}  "
          f"P_FA={f3(gd and gd.get('P_FA'))}")
    fam_stem = f"pt_gbraswh_family_{at}"
    fa_m, fa_s, fa_n, _ = over_seeds(fam_stem, "acc")
    ff_m, ff_s, _, _ = over_seeds(fam_stem, "macro_f1")
    msd = lambda m, s: "MISSING" if m is None else f"{m:.3f}±{s:.3f}"
    print(f"- family: acc={msd(fa_m, fa_s)}  macro-F1={msd(ff_m, ff_s)}  (n seeds {fa_n}; "
          f"seed {C.SEED}: acc={f3(gf and gf.get('acc'))})")
    print_collapsed(fam_stem)
    rows.append(["gbras_family", args.alpha, "acc", fa_m, fa_s, fa_n])
    if gf and "recall_per_class" in gf:
        rpc = gf["recall_per_class"]
        items = rpc.items() if isinstance(rpc, dict) else zip(C.CLASS_FOLDERS, rpc)
        print("  recall per class: " + ", ".join(f"{k}={v:.3f}" for k, v in items))
    elif gf:
        print("  recall per class: MISSING key 'recall_per_class'")
        MISSING.append(f"pt_gbraswh_family_{at}_s{C.SEED}.json:recall_per_class")

    R = C.RESULTS_DIR
    # ---- results: ensemble ----
    print(f"\n## Results — {R}\n")
    e = load(R / f"ensemble_{at}.json")
    if e:
        det = e["detection"]
        print("### CNN modular ensemble (detection AUC)")
        for k in ("generalist", "ensemble_specialists_only", "ensemble_nested_gen_plus_specialists",
                  "fusion_mean"):
            print(f"- {k}: {f3(det.get(k, {}).get('auc'))}")
        for k in ("delta_specialists_only_minus_gen", "delta_nested_ens_minus_gen"):
            d = det.get(k)
            print(f"- {k}: " + ("MISSING" if d is None else f"{d[0]:+.4f}{ci(d[1])}"))
        mc = e.get("multiclass")
        if mc:
            print(f"- multiclass acc: monolithic={f3(mc['monolithic_cnn']['acc'])} "
                  f"modular={f3(mc['modular_ensemble']['acc'])}")
        else:
            print("- multiclass: MISSING (old file with 'fourclass' key?)")
    g = load(R / f"gbras_ensemble_{at}.json")
    if g:
        print("### GBRAS ensemble (detection AUC)")
        print(f"- generalist={f3(g['generalist_detect']['auc'])} ensemble={f3(g['ensemble_detect']['auc'])} "
              f"Δ={g['delta_auc']:+.4f}{ci(g.get('delta_auc_ci95'))}")
        print(f"- Fisher J: 6-class features={g['fisher_J_gbras_feats']} (trained on labels), "
              f"detector features={g.get('fisher_J_gbras_detect_feats', 'MISSING')} (no algorithm labels)")
        gi = g.get("identification")
        if gi is None:
            print("- identification: MISSING (old file; rerun build_gbras_ensemble.py)")
            MISSING.append(f"gbras_ensemble_{at}.json:identification")
        else:
            print(f"### GBRAS identification ({len(C.CLASS_FOLDERS)} classes, TEST acc)")
            print(f"- monolithic={f3(gi['monolithic_gbras_family']['acc'])}{ci(gi['mono_acc_ci95'])} "
                  f"modular={f3(gi['modular_ensemble']['acc'])}{ci(gi['modular_acc_ci95'])} "
                  f"Δ={gi['delta_acc_modular_minus_mono']:+.4f}{ci(gi['delta_acc_ci95'])}")

    # ---- results: transfer ----
    t = load(R / f"transfer_spatial_{at}.json")
    if t:
        per = t.get("per_scheme")
        if per is None:
            print("### Transfer: old format (no 'per_scheme'); rerun transfer_spatial.py")
        else:
            print("### Leave-one-scheme-out transfer (AUC)")
            print("| held out | transfer | in-dist | Δ transfer−indist |\n|--|--|--|--|")
            for k, v in per.items():
                print(f"| {k} | {v['transfer_auc']:.3f}{ci(v.get('transfer_ci95'))} | "
                      f"{v['indist_auc']:.3f} | {v['delta_transfer_minus_indist']:+.3f}{ci(v.get('delta_ci95'))} |")
            print(f"- mean transfer AUC = {t['mean_transfer_auc']:.3f} ± {t['std_transfer_auc']:.3f}")

    # ---- results: bootstrap ----
    b = load(R / "spatial_bootstrap.json")
    if b:
        print("### Spatial bootstrap (AUC, grouped CI)")
        for sec in ("detection_sweep", "gbras_per_algo"):
            for k, v in b.get(sec, {}).items():
                c = "collapsed (constant scores)" if v.get("collapsed") else ci(v.get("ci95")).strip()
                if v.get("ci_contains_point") is False:
                    c += " (CI excludes point)"
                pw = v.get("paired_within_cover", {}).get("p_stego_gt_cover")
                print(f"- {sec}/{k}: {v['auc']:.3f} {c}  P(stego>cover|cover)={pw}")

    # ---- results: separability ----
    # srm is a run_all step; detector-feature separability is an optional manual analysis.
    for feat in ("srm", "detector"):
        s = load(R / f"separability_{feat}_test_{at}.json", required=(feat == "srm"))
        if s:
            if "n_rows_neighbourhood" not in s:
                print(f"### Separability {feat}: old all-rows kNN protocol; rerun separability_spatial.py")
                continue
            print(f"### Separability ({feat}, test)")
            print(f"- Fisher J 6-class={s['fisher_J_6class']:.3f} 5-class stego={s['fisher_J_5class_stego_only']:.3f}")
            print(f"- 1-NN error={s['knn_1nn_error']:.3f} (chance {s['knn_1nn_error_chance']:.3f}), "
                  f"purity={s['neighborhood_purity_k10']:.3f} (chance {s['neighborhood_purity_chance']:.3f}), "
                  f"silhouette={s['silhouette_global']:.3f}, n={s['n_rows_neighbourhood']}")
            ctl = s.get("control_cover_vs_stego")
            if ctl is None:
                print("- control (cover vs stego): MISSING (old file; rerun separability_spatial.py)")
                if feat == "srm":
                    MISSING.append(f"separability_srm_test_{at}.json:control_cover_vs_stego")
            else:
                lp = ctl["linear_probe"]
                print(f"- control, can the features see the embedding? cover-vs-stego linear AUC="
                      f"{lp['cover_vs_stego_auc'][0]:.3f}{ci(lp['cover_vs_stego_auc'][1])}, "
                      f"1-NN error={ctl['knn_1nn_error']:.3f} (chance 0.5); stego 5-class linear acc="
                      f"{lp['stego_5class_acc'][0]:.3f}{ci(lp['stego_5class_acc'][1])} (chance 0.200)")

    # ---- training budget: runs cut by the epoch cap while still improving ----
    capped = []
    for p in sorted(C.MODEL_DIR.glob("pt_*.json")):
        j = json.loads(p.read_text())
        if j.get("still_improving_at_cap"):
            capped.append(f"{p.name} (best ep {j.get('best_epoch')} of {j.get('epochs_cap')})")
    if capped:
        print(f"\nWARNING: {len(capped)} run(s) used every epoch and were still improving; "
              f"their metrics depend on the epoch budget:")
        for c in capped:
            print("  -", c)

    R.mkdir(parents=True, exist_ok=True)
    with open(R / "summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["table", "alpha", "variant", "mean", "std", "n"]); w.writerows(rows)
    print(f"\nSaved: {R/'summary.csv'}")
    miss = sorted(set(MISSING))
    if miss:
        print(f"\nMISSING ({len(miss)}):")
        for m in miss:
            print("  -", m)
        sys.exit(1)


if __name__ == "__main__":
    main()
