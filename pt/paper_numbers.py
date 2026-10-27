"""Collects the values quoted in the paper into $STEG_RESULTS/paper_numbers.json from the
result files, including the averages over the identification seeds.

Usage:  python pt/paper_numbers.py
"""
import sys, json
from pathlib import Path
from statistics import mean, pstdev
sys.path.append(str(Path(__file__).resolve().parent.parent))   # repo root (config)
import config as C

A = "a040"
ID_SEEDS = [42, 1, 2]
SPATIAL = ["HILL", "HUGO", "MiPOD", "S-UNIWARD", "WOW"]


def load(p):
    f = Path(p)
    return json.load(open(f)) if f.exists() else None


def main():
    R, M = Path(C.RESULTS_DIR), Path(C.MODEL_DIR)
    out = {}

    # --- spatial detection sweep (bootstrap CIs, resampled by cover) ---
    boot = load(R / "spatial_bootstrap.json")
    if boot:
        out["detection_sweep"] = {
            k: {"auc": v["auc"], "ci95": v["ci95"], "collapsed": v["collapsed"],
                "collapsed_reasons": v["collapsed_reasons"], "score_std": v["score_std"]}
            for k, v in boot["detection_sweep"].items()}
        out["gbras_specialists_image"] = {
            k: {"auc": v["auc"], "ci95": v["ci95"]} for k, v in boot["gbras_per_algo"].items()}

    # --- identification: both architectures, averaged over seeds ---
    for tag, stem in (("family_lightweight", "pt_family"), ("family_gbras", "pt_gbraswh_family")):
        accs, recalls = {}, {}
        for s in ID_SEEDS:
            d = load(M / f"{stem}_{A}_s{s}.json")
            if not d:
                continue
            accs[s] = d["acc"]
            for cls, r in (d.get("recall_per_class") or {}).items():
                recalls.setdefault(cls, {})[s] = r
        if accs:
            out[tag] = {
                "acc_per_seed": accs,
                "acc_mean": round(mean(accs.values()), 4),
                "acc_sd": round(pstdev(list(accs.values())), 4) if len(accs) > 1 else None,
                "recall_mean_over_seeds": {c: round(mean(v.values()), 4) for c, v in recalls.items()},
                "recall_per_seed": recalls,
                "n_seeds": len(accs)}

    # --- monolithic vs modular, both domains ---
    ens, gens = load(R / f"ensemble_{A}.json"), load(R / f"gbras_ensemble_{A}.json")
    if ens:
        d = ens["detection"]
        out["spatial_detection_ensemble"] = {
            "generalist_auc": d["generalist"]["auc"],
            "specialists_only_auc": d["ensemble_specialists_only"]["auc"],
            "delta_specialists_only": d["delta_specialists_only_minus_gen"],
            "meta_features": "specialists only; the nested variant is reported separately"}
    if gens:
        out["gbras_detection_ensemble"] = {
            "generalist_auc": gens["generalist_detect"]["auc"],
            "ensemble_auc": gens["ensemble_detect"]["auc"],
            "delta": gens["delta_auc"], "delta_ci95": gens["delta_auc_ci95"]}
        idn = gens["identification"]
        out["gbras_identification"] = {
            "monolithic_acc": idn["monolithic_gbras_family"]["acc"],
            "modular_acc": idn["modular_ensemble"]["acc"]}
        # Learned-feature separability: the detector never saw algorithm labels, so it is
        # the non-circular panel; the 6-class network's own J is reported for contrast only.
        for key, name in (("fisher_pairs_detect", "learned_detector_features"),
                          ("fisher_pairs", "learned_6class_features_circular")):
            pairs = {k: v for k, v in gens[key].items() if "COVER" not in k}
            out[name] = {"stego_pair_min": min(pairs.values()), "stego_pair_max": max(pairs.values()),
                         "argmin": min(pairs, key=pairs.get), "argmax": max(pairs, key=pairs.get)}
        out["fisher_note"] = gens["fisher_note"]

    # --- model-free SRM control: do these features see the embedding at all? ---
    sep = load(R / f"separability_srm_test_{A}.json")
    if sep:
        ctrl = sep.get("control_cover_vs_stego", {})
        out["srm_control"] = {k: ctrl.get(k) for k in
                              ("linear_probe_auc", "linear_probe_auc_ci95",
                               "stego_5class_acc", "stego_5class_acc_ci95") if k in ctrl} or ctrl

    # --- transfer and diagnostic ---
    tr = load(R / f"transfer_spatial_{A}.json")
    if tr:
        out["transfer_spatial"] = {
            "per_scheme": {k: {"transfer_auc": v["transfer_auc"], "indist_auc": v["indist_auc"],
                               "delta": v["delta_transfer_minus_indist"]}
                           for k, v in tr["per_scheme"].items()},
            "mean_transfer_auc": tr["mean_transfer_auc"], "mean_indist_auc": tr["mean_indist_auc"],
            "mean_delta": tr["mean_delta_transfer_minus_indist"],
            "indist_protocol": tr["indist_protocol"]}
    diag = load(R / f"diagnostic_spatial_{A}.json")
    if diag:
        out["diagnostic_spatial"] = {
            src: {"spearman_r": v["stego_only"]["spearman_r"],
                  "mantel_p": v["stego_only"]["mantel_p_one_sided_negative"],
                  "n_pairs": v["stego_only"]["n_pairs"]}
            for src, v in diag["sources"].items() if "stego_only" in v}

    # --- JPEG (written only once the files exist) ---
    for stem, key in ((f"dctr_ensemble_{A}", "jpeg_single_split"),
                      (f"dctr_bootstrap_{A}", "jpeg_bootstrap"),
                      (f"transfer_jpeg_{A}", "jpeg_transfer")):
        d = load(R / f"{stem}.json")
        if d:
            out[key] = d
    for a in ("a010", "a020", "a040"):
        d = load(R / f"jpeg_kfold_{a}.json")
        if d:
            out.setdefault("jpeg_kfold", {})[a] = {k: d[k] for k in d if k != "feature_cache"}

    dst = R / "paper_numbers.json"
    json.dump(out, open(dst, "w"), indent=1, sort_keys=False)
    print(f"saved: {dst}")
    for k in out:
        print("  ", k)


if __name__ == "__main__":
    main()
