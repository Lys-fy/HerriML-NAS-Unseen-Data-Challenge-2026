#!/usr/bin/env python3
"""Would a short backbone bake-off pick the right backbone?

For each dataset, give conv / tcn1d / tcn2d a short wall-clock budget, look at
their best val-acc so far, and see if that early ranking matches who actually
wins at full budget. The "regret" is how much final score we lose by trusting
the early pick instead of an oracle that always picks the true best.
"""
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# dataset -> codename ; plus final TEST adj per backbone (conv, tcn1d, tcn2d)
DS = [
    ("Language",   "LaMelo",   (-1.03, +2.80, +2.05)),
    ("Cryptic",    "Cryptic",  (+7.03, +7.74, +7.12)),
    ("AddNIST",    "Adaline",  (+5.88, +4.24, +6.67)),
    ("CIFARTile",  "Caitie",   (+0.24, -0.95, +4.75)),
    ("Chesseract", "Chester",  (+0.30, +0.10, -0.18)),
    ("GameOfLife", "conway",   (+9.99, +10.00, +9.998)),
    ("GeoClassing","Sadie",    (+7.65, +7.18, +7.71)),
    ("Gutenberg",  "Gutenberg",(+1.03, -0.72, -0.06)),
    ("MultNIST",   "Mateo",    (+4.26, +2.21, +5.09)),
    ("Myofibre",   "Myopia",   (+0.34, -0.17, +3.34)),
    ("Sudoku",     "Sokoto",   (+9.45, +8.07, +8.81)),
    ("Voxel",      "Volga",    (+2.38, +3.57, +3.13)),
    ("Windspeed",  "windspeed",(+0.72, +0.59, +0.72)),
]
ARCHS = ["conv", "tcn1d", "tcn2d"]


def csv_path(arch, name, code):
    if arch == "conv":
        return REPO / f"dev/logs/tier2a_p1_v8/p1v8_s42/{code}.csv"
    return REPO / f"dev/logs/tcn_probe/{arch}_{name}_s42/{code}.csv"


def best_val_within(path, budget_s):
    """Best val_acc among epochs whose cumulative train time <= budget_s.
    budget_s=None -> full run."""
    best, t = -1.0, 0.0
    with open(path) as f:
        for r in csv.DictReader(f):
            t += float(r["epoch_time_s"])
            if budget_s is not None and t > budget_s:
                break
            best = max(best, float(r["val_acc"]),
                       float(r.get("val_acc_ema", 0) or 0))
    return best


for budget in (180, 300, 600):
    print(f"\n===== bake-off budget = {budget}s ({budget//60} min) per backbone =====")
    print(f"{'dataset':11s} {'early-pick':10s} {'true-best':10s} match  "
          f"{'pick_final':>10s} {'best_final':>10s} {'regret':>7s}")
    tot_pick, tot_oracle, n_match = 0.0, 0.0, 0
    for name, code, finals in DS:
        early = {a: best_val_within(csv_path(a, name, code), budget) for a in ARCHS}
        pick = max(ARCHS, key=lambda a: early[a])
        finals_d = dict(zip(ARCHS, finals))
        true_best = max(ARCHS, key=lambda a: finals_d[a])
        pick_final = finals_d[pick]
        best_final = finals_d[true_best]
        regret = best_final - pick_final
        match = "OK " if pick == true_best else "MISS"
        if pick == true_best:
            n_match += 1
        tot_pick += pick_final
        tot_oracle += best_final
        print(f"{name:11s} {pick:10s} {true_best:10s} {match:5s} "
              f"{pick_final:+10.2f} {best_final:+10.2f} {regret:7.2f}")
    print(f"  top-1 correct: {n_match}/13   "
          f"bake-off total {tot_pick:+.1f}   oracle {tot_oracle:+.1f}   "
          f"lost-to-misselect {tot_oracle - tot_pick:.1f}")
print("\nReference: conv(v8) +48.2 | tcn2d-fixed +59.2 | oracle +63.2")
