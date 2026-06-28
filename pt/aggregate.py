"""
Aggregates the results/*.log logs into tables (detection, family, baseline).
Prints Markdown and saves results/summary.csv. Robust to incomplete runs.

Usage:  python aggregate.py
"""
import re, sys, csv
from pathlib import Path
from collections import defaultdict
sys.path.append(str(Path(__file__).resolve().parent.parent))   # code/ (config)
import config as C

RES = C.RESULTS_DIR


def _f(pat, text):
    m = re.search(pat, text)
    return float(m.group(1)) if m else None


def parse_binary(text):
    return {"AUC": _f(r"AUC=([0-9.]+)", text),
            "P_D": _f(r"P_D=([0-9.]+)", text),
            "P_FA": _f(r"P_FA=([0-9.]+)", text)}


def parse_family(text):
    return {"acc": _f(r"acc=([0-9.]+)", text),
            "macro_F1": _f(r"macro_F1=([0-9.]+)", text)}


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return (None, None)
    m = sum(vals) / len(vals)
    s = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
    return (m, s)


def main():
    logs = {p.stem: p.read_text(errors="ignore") for p in RES.glob("*.log")}
    rows = []

    # ---- Exp1 detection: groups by (alpha, srm/nosrm) over seeds ----
    det = defaultdict(list)
    for name, txt in logs.items():
        m = re.match(r"exp1_a(\d+)_(srm|nosrm)_s(\d+)", name)
        if not m:
            continue
        alpha = int(m.group(1)) / 100
        key = (alpha, m.group(2))
        det[key].append(parse_binary(txt))

    print("\n## Exp1 — Cover/stego detection (disjoint test)\n")
    print("| payload | front-end | AUC | P_D | P_FA | n |")
    print("|--|--|--|--|--|--|")
    for (alpha, fe), runs in sorted(det.items()):
        a = mean_std([r["AUC"] for r in runs])
        pd_ = mean_std([r["P_D"] for r in runs])
        fa = mean_std([r["P_FA"] for r in runs])
        fe_lbl = "SRM" if fe == "srm" else "without SRM (abl.)"
        fmt = lambda ms: "—" if ms[0] is None else (f"{ms[0]:.3f}±{ms[1]:.3f}" if len(runs) > 1 else f"{ms[0]:.3f}")
        print(f"| {alpha:.2f} | {fe_lbl} | {fmt(a)} | {fmt(pd_)} | {fmt(fa)} | {len(runs)} |")
        rows.append(["exp1", alpha, fe, a[0], pd_[0], fa[0], len(runs)])

    # ---- Exp2 family ----
    fam = defaultdict(list)
    for name, txt in logs.items():
        m = re.match(r"exp2_family_a(\d+)_s(\d+)", name)
        if m:
            fam[int(m.group(1)) / 100].append(parse_family(txt))
    print("\n## Exp2 — Family identification (disjoint test)\n")
    print("| payload | accuracy | macro-F1 | n |")
    print("|--|--|--|--|")
    for alpha, runs in sorted(fam.items()):
        ac = mean_std([r["acc"] for r in runs]); f1 = mean_std([r["macro_F1"] for r in runs])
        fmt = lambda ms: "—" if ms[0] is None else (f"{ms[0]:.3f}±{ms[1]:.3f}" if len(runs) > 1 else f"{ms[0]:.3f}")
        print(f"| {alpha:.2f} | {fmt(ac)} | {fmt(f1)} | {len(runs)} |")
        rows.append(["exp2", alpha, "family", ac[0], f1[0], None, len(runs)])

    # ---- Baseline GBRAS ----
    print("\n## Baseline — GBRAS-Net, COVER vs algorithm @0.4 (block level)\n")
    print("| algorithm | AUC | P_D | P_FA |")
    print("|--|--|--|--|")
    for name, txt in sorted(logs.items()):
        m = re.match(r"base_gbras_(.+?)_a(\d+)_s(\d+)", name)
        if not m:
            continue
        b = parse_binary(txt)
        fmt = lambda v: "—" if v is None else f"{v:.3f}"
        print(f"| {m.group(1)} | {fmt(b['AUC'])} | {fmt(b['P_D'])} | {fmt(b['P_FA'])} |")
        rows.append(["gbras", int(m.group(2)) / 100, m.group(1), b["AUC"], b["P_D"], b["P_FA"], 1])

    with open(RES / "summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["exp", "alpha", "variant", "m1", "m2", "m3", "n"]); w.writerows(rows)
    print(f"\nSaved: {RES/'summary.csv'}")


if __name__ == "__main__":
    main()
