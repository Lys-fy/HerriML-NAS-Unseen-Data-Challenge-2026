#!/usr/bin/env python3
"""Offline validation of cheap ranking signals against the TCN-probe curves.

Question: at a small time budget (3/5/10 min per candidate), which signal picks
the right backbone (conv vs tcn1d vs tcn2d) for each dataset?

Signals tested (all computed ONLY from the curve prefix within the budget):
  earlyval : best val-acc so far (the naive bake-off; known weak)
  tse      : Training Speed Estimation, sum of the last E train losses,
             lower is better (Ru et al. 2021, [[2021 - Training Speed
             Estimation (TSE)]])
  sotl     : sum of ALL train losses so far, lower is better (same family)
  powlaw   : power-law extrapolation of the val curve to the arch's final
             epoch, y = a - b * x^(-c) (pow3, the DPL/Domhan family)
  gated    : earlyval leader, but ABSTAIN to the fixed default (tcn2d) when
             the top-2 separation is below the measured val-noise (the
             racing/abstention idea, [[2024 - AdaptSH Hypothesis-Test
             Successive Halving]])

Truth = final TEST adj of each backbone (from the completed full-budget runs).
Reference points: always-tcn2d = +59.2, oracle = +63.2, conv(v8) = +48.2.

Caveat: single seed (42), and conv curves come from the v8 pipeline runs while
TCN curves come from the probe (same trainer family, same aug-probe; train_loss
scales are comparable per dataset but not identical across pipelines).
"""
import csv
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# dataset, codename, final TEST adj per backbone (conv, tcn1d, tcn2d)
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
DEFAULT = "tcn2d"
TSE_E = 3  # window for TSE-E


def csv_path(arch, name, code):
    if arch == "conv":
        return REPO / f"dev/logs/tier2a_p1_v8/p1v8_s42/{code}.csv"
    return REPO / f"dev/logs/tcn_probe/{arch}_{name}_s42/{code}.csv"


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(dict(
                epoch=int(r["epoch"]),
                loss=float(r["train_loss"]),
                val=float(r["val_acc"]),
                ema=float(r.get("val_acc_ema", 0) or 0),
                dt=float(r["epoch_time_s"]),
            ))
    return rows


def prefix(rows, budget_s):
    out, t = [], 0.0
    for r in rows:
        t += r["dt"]
        if t > budget_s:
            break
        out.append(r)
    return out


def sig_earlyval(pre, rows):
    return max(max(r["val"], r["ema"]) for r in pre)


def sig_tse(pre, rows):
    last = pre[-TSE_E:]
    return -sum(r["loss"] for r in last) / len(last)   # negate: higher=better


def sig_sotl(pre, rows):
    return -sum(r["loss"] for r in pre) / len(pre)


def sig_powlaw(pre, rows):
    """Fit y = a - b * x^(-c) to (epoch+1, val) by grid over c with linear
    least squares for a, b. Extrapolate to this arch's own final epoch."""
    xs = [r["epoch"] + 1.0 for r in pre]
    ys = [r["val"] for r in pre]
    n = len(xs)
    if n < 4:
        return max(ys)
    x_final = rows[-1]["epoch"] + 1.0
    best_err, best_pred = None, max(ys)
    for c in [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        zs = [x ** (-c) for x in xs]
        zbar = sum(zs) / n
        ybar = sum(ys) / n
        szz = sum((z - zbar) ** 2 for z in zs)
        if szz < 1e-12:
            continue
        szy = sum((z - zbar) * (y - ybar) for z, y in zip(zs, ys))
        bneg = szy / szz          # slope on z; model says slope = -b
        a = ybar - bneg * zbar    # intercept = a
        b = -bneg
        if b < 0:                 # curve would be decreasing; skip
            continue
        err = sum((a - b * z - y) ** 2 for z, y in zip(zs, ys))
        if best_err is None or err < best_err:
            best_err = err
            best_pred = min(1.0, a - b * x_final ** (-c))
    return best_pred


def val_noise(pre):
    """Std of raw val over the last 10 epochs of the prefix."""
    tail = [r["val"] for r in pre[-10:]]
    if len(tail) < 3:
        return 0.05
    m = sum(tail) / len(tail)
    return math.sqrt(sum((v - m) ** 2 for v in tail) / len(tail))


def evaluate(budget_s):
    signals = {"earlyval": sig_earlyval, "tse": sig_tse,
               "sotl": sig_sotl, "powlaw": sig_powlaw}
    picks = {k: [] for k in list(signals) + ["gated"]}
    for name, code, finals in DS:
        data = {a: load(csv_path(a, name, code)) for a in ARCHS}
        pres = {a: prefix(data[a], budget_s) for a in ARCHS}
        if any(len(p) == 0 for p in pres.values()):
            # budget too small to see an epoch on some arch: default
            for k in picks:
                picks[k].append((name, DEFAULT, finals))
            continue
        scores = {k: {a: fn(pres[a], data[a]) for a in ARCHS}
                  for k, fn in signals.items()}
        for k in signals:
            lead = max(ARCHS, key=lambda a: scores[k][a])
            picks[k].append((name, lead, finals))
        # gated: earlyval leader unless top-2 gap < mean noise
        ev = scores["earlyval"]
        order = sorted(ARCHS, key=lambda a: ev[a], reverse=True)
        gap = ev[order[0]] - ev[order[1]]
        noise = sum(val_noise(pres[a]) for a in ARCHS) / len(ARCHS)
        pick = order[0] if gap > noise else DEFAULT
        picks["gated"].append((name, pick, finals))
    return picks


def report(picks):
    finals_d = {name: dict(zip(ARCHS, f)) for name, _, f in
                [(n, c, f) for n, c, f in DS]}
    truth = {name: max(ARCHS, key=lambda a: finals_d[name][a]) for name in finals_d}
    out = {}
    for k, plist in picks.items():
        total, correct = 0.0, 0
        for name, pick, finals in plist:
            fd = dict(zip(ARCHS, finals))
            total += fd[pick]
            if pick == truth[name]:
                correct += 1
        out[k] = (correct, total)
    return out


ORACLE = sum(max(f) for _, _, f in DS)
TCN2D = sum(f[2] for _, _, f in DS)
print(f"reference: always-tcn2d {TCN2D:+.1f} | oracle {ORACLE:+.1f} | conv +48.2")
print(f"{'budget':>7s}  {'signal':9s} {'top1':>5s} {'total':>7s}  vs tcn2d")
for budget in (180, 300, 600):
    picks = evaluate(budget)
    res = report(picks)
    for k in ["earlyval", "tse", "sotl", "powlaw", "gated"]:
        c, t = res[k]
        delta = t - TCN2D
        print(f"{budget:>6d}s  {k:9s} {c:>4d}/13 {t:>+7.1f}  {delta:+.1f}")
    print()

# per-dataset detail at the most interesting budget (300s)
print("=== per-dataset picks @300s (truth = final-adj winner) ===")
picks = evaluate(300)
finals_map = {n: dict(zip(ARCHS, f)) for n, _, f in DS}
for i, (name, _, finals) in enumerate(DS):
    fd = dict(zip(ARCHS, finals))
    truth = max(ARCHS, key=lambda a: fd[a])
    row = " ".join(f"{k}={picks[k][i][1]:6s}" for k in
                   ["earlyval", "tse", "powlaw", "gated"])
    print(f"  {name:11s} truth={truth:6s} {row}")
