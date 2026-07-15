"""Compare paired equivalence runs: predictions, scores, decisions.

A cleaned submission is equivalent to its baseline if the (clean vs baseline)
difference is within the (baseline vs baseline) run-noise floor.
"""
import sys, pathlib, re
import numpy as np

OUT = pathlib.Path(sys.argv[1])
PAIRS = [
    ("v13a_chess", "v13b_chess", "NOISE FLOOR  (v13 vs v13, Chesseract)"),
    ("v13a_chess", "v2_chess",   "v2 EQUIVALENCE  (v13 vs clean-v2, Chesseract)"),
    ("v13_cryptic", "v2_cryptic", "v2 TOKEN EQUIVALENCE  (v13 vs clean-v2, Cryptic)"),
    ("v121_sudoku", "v1_sudoku", "v1 EQUIVALENCE  (v12_1 vs clean-v1, Sudoku)"),
]
DEC_KEYS = ("fingerprint", "backbone_gate", "backbone_skip", "final:", "token_gate")


def preds(label):
    fs = sorted((OUT / f"pred_{label}").glob("*.npy"))
    return np.load(fs[0]) if fs else None


def logtext(label):
    p = OUT / f"{label}.log"
    return p.read_text(errors="ignore") if p.exists() else ""


def score(label):
    m = re.findall(r"raw=\s*([\d.]+)%", logtext(label))
    return float(m[-1]) if m else None


def decisions(label):
    return [ln.strip() for ln in logtext(label).splitlines()
            if any(k in ln for k in DEC_KEYS)]


for a, b, desc in PAIRS:
    print(f"\n===== {desc}")
    pa, pb = preds(a), preds(b)
    if pa is not None and pb is not None and len(pa) == len(pb):
        print(f"  prediction agreement: {100.0*np.mean(pa==pb):.2f}%  ({len(pa)} preds)")
    else:
        print(f"  predictions: MISSING (a={pa is not None} b={pb is not None})")
    sa, sb = score(a), score(b)
    d = None if (sa is None or sb is None) else round(sb - sa, 3)
    print(f"  raw score: {a}={sa}  {b}={sb}   diff={d}")
    da, db = decisions(a), decisions(b)
    same = da == db
    print(f"  key decisions identical: {same}")
    if not same:
        for x in da: print(f"    A| {x}")
        for x in db: print(f"    B| {x}")
