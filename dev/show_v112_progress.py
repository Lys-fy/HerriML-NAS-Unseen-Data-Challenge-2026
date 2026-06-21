#!/usr/bin/env python3
"""Print the depth-test results + v11.2 decisions so far. Run on spark."""
import json
from pathlib import Path

H = Path.home()

print("=== DEPTH TEST (CIFARTile, uniform DilatedConv3, cosine 120ep) ===")
for D in (6, 12, 20):
    p = H / f"local/cifartile_depth/d{D}/Caitie.json"
    if not p.exists():
        print(f"  depth {D}: not done"); continue
    r = json.loads(p.read_text()); b = r["baseline"]; f = r["final"]
    cap = " [CAP]" if b.get("capped") else ""
    print(f"  depth {D:2d}: baseline val {b['best_val']:.4f} test {b['test_acc']:.4f} "
          f"({b['epochs']}ep{cap})  final test {f['test_acc']:.4f}")
print("  reference: v10 real CIFARTile = 73.65 ; PC-DARTS (depth~20) = 92.28")

print("\n=== v11.2 (decisions so far) ===")
for code in ("Chester", "Gutenberg", "Caitie"):
    p = H / f"local/v112/{code}.json"
    if not p.exists():
        print(f"  {code}: not done yet"); continue
    r = json.loads(p.read_text())
    de, sr, fa, v = r["decision"], r["searched"], r["family_alternatives"], r["v10_route"]
    print(f"  {code}: WINNER = {de['winner']}  val={de['winner_val']}  "
          f"adopt_over_v10={de['adopt_over_v10']}  margin={de['margin_val']:+}")
    print(f"     searched: {sr['depth']}L hetero={sr['heterogeneous']} val={sr['final_val']} "
          f"test={sr['test_acc']}")
    print(f"     alts: tcn1d {fa['tcn1d']['final_val']} | tcn2d {fa['tcn2d']['final_val']} "
          f"| v10_route {v['final_val']}")
    if sr.get("ops"):
        from collections import Counter
        print(f"     searched ops: {dict(Counter(sr['ops']))}")
