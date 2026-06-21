#!/usr/bin/env python3
"""Recover per-epoch best-val curves for the 3 new datasets from the chain's
stdout logs (the runs did not set BASELINE_LOG_DIR, but every training epoch
line carries best=...). Writes CSVs in the same layout plot_newdata_grid.py
reads: dev/logs/v10_newdata_sweep/{v10_s42,v10_s1337,v10_s2024,resnet18_s42}/<code>.csv

v10new.log = 3 seeds x [Sedol, Sloan, Cauchy] in order; the n-th occurrence of a
code maps to SEEDS[n]. resnet.log = one seed (42), 3 datasets.
"""
import csv
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SWEEP = REPO / "dev/logs/v10_newdata_sweep"
SEEDS = ["42", "1337", "2024"]
HDR = re.compile(r"Dataset\s+(\S+)")
BEST = re.compile(r"\bepoch\s+\d+.*\bbest=([\d.]+)")


def sections(logpath):
    """Yield (code, [best_val fractions]) per dataset run, in file order."""
    code, best = None, []
    for line in Path(logpath).read_text(errors="ignore").splitlines():
        if "====" in line and "Dataset" in line:
            if code:
                yield code, best
            code, best = HDR.search(line).group(1), []
        else:
            m = BEST.search(line)
            if m and code:
                best.append(float(m.group(1)))
    if code:
        yield code, best


def write_csv(subdir, code, best):
    d = SWEEP / subdir
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{code}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "best_val_acc"])
        for i, b in enumerate(best):
            w.writerow([i, b])
    print(f"  {subdir}/{code}.csv  ({len(best)} epochs, final={best[-1]*100:.2f}%)")


def main():
    # v10: map n-th occurrence of each code to SEEDS[n]
    seen = {}
    for code, best in sections("/tmp/v10new.log"):
        if not best:
            continue
        k = seen.get(code, 0); seen[code] = k + 1
        if k < len(SEEDS):
            write_csv(f"v10_s{SEEDS[k]}", code, best)
    # resnet: all seed 42
    for code, best in sections("/tmp/resnet.log"):
        if best:
            write_csv("resnet18_s42", code, best)


if __name__ == "__main__":
    main()
