#!/usr/bin/env python3
"""v11.2: heterogeneous micro-macro NAS via multi-rung successive halving.

A real Hyperband-style search over the joint op x depth x width space (depth incl.
DEEP 20), evaluated multi-fidelity: cheap rungs PRUNE (never committed), the final
rung FULL-TRAINS the survivors AND the v10-route family-default arch with the
identical cosine recipe, and adopts the best heterogeneous arch ONLY if it beats
the v10 arch beyond noise (verify-before-adopt -> v11.2 can never lose to v10 on a
fair test).

Default schedule (3 rungs): 24 candidates @8ep -> keep 12 @40ep -> keep 4 @120ep
(final). Reuses the macro-search harness (build_cnn, cosine train_arm,
DataProcessor). Purely additive in dev/. Epoch-based => GPU-portable. Run one
dataset; dev/run_hetero_probe.sh loops the probe set.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from macro_search_indepth import (train_arm, _predict, _noise, Clock,        # noqa: E402
                                  load_dataset, OP_VOCAB, DataProcessor,
                                  STEM_EMBED_MIN_CH)

# Tier-S: SGD-stable, short-train-rankable ops. LSTMs (RowBiLSTM/CrossAxisBiLSTM)
# are excluded -- a short proxy cannot rank them fairly (v7 lesson) and they need a
# protected longer-warmup rung (cluster-era TODO). Attention is borderline-stable
# -> injected, capped at 2 per arch.
TIER_S = ["Conv3", "Conv5", "Conv7", "SepConv3", "GroupConv3", "ResNeXt",
          "DilatedConv3", "SEConv3"]
ATTN = "SpatialSelfAttn2D"
DEPTHS = [6, 10, 14, 20]
WIDTHS = [32, 48]


def v10_route_arch(encoded):
    op = "Conv3" if encoded else "DilatedConv3"   # encoded->conv, image-like->tcn2d
    return {"ops": [op] * 6, "width": 48, "depth": 6, "tag": "v10_route"}


def gen_candidates(n, rng):
    """Structured-random heterogeneous candidates + 2 deep anchors."""
    cands = [{"ops": ["Conv3"] * 20, "width": 48, "depth": 20, "tag": "anchor_deepconv"},
             {"ops": ["DilatedConv3"] * 14, "width": 48, "depth": 14, "tag": "anchor_deepdil"}]
    while len(cands) < n:
        D = int(rng.choice(DEPTHS)); W = int(rng.choice(WIDTHS))
        ops = [str(rng.choice(TIER_S)) for _ in range(D)]
        for _ in range(int(rng.integers(0, 3))):          # 0..2 attention, non-edge
            if D >= 4:
                ops[int(rng.integers(1, D - 1))] = ATTN
        cands.append({"ops": ops, "width": W, "depth": D, "tag": "rand"})
    return cands[:n]


def parse_rungs(s):
    """'8:12,40:4,120:0' -> [(8,12),(40,4),(120,0)]; keep<=0 marks the final rung."""
    out = []
    for part in s.split(","):
        e, k = part.split(":")
        out.append((int(e), int(k)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-candidates", type=int, default=24)
    ap.add_argument("--rungs", default="8:12,40:4,120:0",
                    help="epochs:keep per rung; keep<=0 = final full-train rung")
    ap.add_argument("--cap-per-epoch", type=float, default=60.0,
                    help="wall-clock safety cap = epochs * this (generous; should not truncate)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    rungs = parse_rungs(args.rungs)
    d = Path(args.dataset_dir); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, test_x, test_y, meta = load_dataset(d)
    code = meta.get("codename", d.name)

    clock = Clock(hours=999)
    dp = DataProcessor(tr[0], tr[1], va[0], va[1], test_x, meta, clock)
    train_loader, val_loader, test_loader = dp.process()
    loaders = (train_loader, val_loader)
    xb0, _ = next(iter(train_loader))
    in_ch, h, w = xb0.shape[1], xb0.shape[2], xb0.shape[3]
    nc = int(meta["num_classes"])
    stem = "channel_embed" if in_ch >= STEM_EMBED_MIN_CH else "conv3"
    encoded = bool(meta.get("fp_encoded", False))
    print(f"[{code}] device={device} fp_encoded={encoded} stem={stem} "
          f"N={args.n_candidates} rungs={rungs}", flush=True)

    def test_acc(model):
        if test_y is None or model is None:
            return None
        pred = _predict(model, test_loader, device)
        n = min(len(pred), len(test_y))
        return float((pred[:n] == test_y[:n]).mean()) if n else None

    def run_arm(c, epochs):
        cap = epochs * args.cap_per_epoch
        try:
            m, vals, et = train_arm(c["ops"], in_ch, nc, h, w, c["width"], stem,
                                    loaders, device, epochs, cap)
            return m, vals, et, max(vals), len(vals) < epochs, False
        except Exception as e:
            print(f"[{code}]   arm FAILED ({c['tag']}): {e}", flush=True)
            return None, [0.0], 0.0, 0.0, False, True

    rec = {"codename": code, "dataset": d.name, "device": str(device),
           "search_mode": "v11.2_successive_halving", "fp_corr1": meta.get("fp_corr1"),
           "fp_encoded": encoded, "in_ch": in_ch, "h": h, "w": w, "num_classes": nc,
           "stem": stem, "seed": args.seed,
           "space": {"depths": DEPTHS, "widths": WIDTHS, "tier_s": TIER_S, "attn": ATTN},
           "rungs_schedule": rungs, "n_candidates": args.n_candidates,
           "rungs": [], "v10_route": None, "decision": None, "timing": {}}
    curves = {}
    arm_seconds = 0.0
    n_arms = 0

    cands = gen_candidates(args.n_candidates, rng)
    survivors = list(range(len(cands)))
    best = (-1.0, None, None)   # (val, cand, test_acc)

    for r, (epochs, keep) in enumerate(rungs):
        is_final = keep <= 0 or r == len(rungs) - 1
        print(f"[{code}] rung{r}: {len(survivors)} arms x {epochs}ep "
              f"({'FINAL' if is_final else f'keep {keep}'})", flush=True)
        rung_log = {"rung": r, "epochs": epochs, "keep": keep, "final": is_final, "arms": []}
        results = []
        for idx in survivors:
            c = cands[idx]
            m, vals, et, v, capped, nan = run_arm(c, epochs)
            ta = test_acc(m) if is_final else None
            rung_log["arms"].append({"id": idx, "tag": c["tag"], "ops": c["ops"],
                                     "depth": c["depth"], "width": c["width"],
                                     "best_val": round(v, 4), "test_acc": ta,
                                     "epochs": len(vals), "seconds": round(et, 1),
                                     "capped": capped, "nan": nan})
            curves[f"r{r}_{idx}"] = np.array(vals, np.float32)
            results.append((v, idx, ta, c))
            arm_seconds += et; n_arms += 1
            if m is not None:
                del m
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print(f"[{code}] r{r} cand{idx:2d} {c['tag']:15s} D{c['depth']:<2d} W{c['width']} "
                  f"val={v:.4f}{' CAP' if capped else ''}", flush=True)
        rec["rungs"].append(rung_log)
        results.sort(reverse=True, key=lambda x: x[0])
        if is_final:
            best = (results[0][0], results[0][3], results[0][2])
            break
        survivors = [idx for _, idx, _, _ in results[:keep]]
        print(f"[{code}] r{r} -> survivors {survivors}", flush=True)

    # ---- v10-route reference (same recipe, final epochs) + adopt decision ----
    final_epochs = rungs[-1][0]
    vr = v10_route_arch(encoded)
    m, vvals, vt, v10_val, v10_cap, _ = run_arm(vr, final_epochs)
    v10_test = test_acc(m); v10_noise = _noise(vvals)
    rec["v10_route"] = {"ops": vr["ops"], "width": vr["width"], "depth": vr["depth"],
                        "best_val": round(v10_val, 4), "test_acc": v10_test,
                        "noise": round(v10_noise, 4), "epochs": len(vvals),
                        "seconds": round(vt, 1), "capped": v10_cap}
    curves["v10_route"] = np.array(vvals, np.float32)
    arm_seconds += vt; n_arms += 1
    if m is not None:
        del m
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    adopt = best[0] > v10_val + v10_noise
    rec["decision"] = {"adopted_hetero": bool(adopt),
                       "hetero_best_val": round(best[0], 4), "hetero_test_acc": best[2],
                       "hetero_ops": best[1]["ops"] if best[1] else None,
                       "hetero_depth": best[1]["depth"] if best[1] else None,
                       "hetero_width": best[1]["width"] if best[1] else None,
                       "v10_best_val": round(v10_val, 4), "v10_test_acc": v10_test,
                       "margin_val": round(best[0] - v10_val, 4),
                       "rule": "adopt iff hetero_best_val > v10_best_val + noise"}
    rec["timing"] = {"device": str(device), "search_seconds": round(arm_seconds, 1),
                     "n_arms": n_arms, "v10_route_seconds": round(vt, 1)}

    (out / f"{code}.json").write_text(json.dumps(rec, indent=2))
    np.savez_compressed(out / f"{code}_curves.npz", **curves)
    print(f"[{code}] DECISION adopt_hetero={adopt}  hetero_val={best[0]:.4f} "
          f"v10_val={v10_val:.4f} margin={best[0]-v10_val:+.4f}  "
          f"test hetero/v10 = {best[2]}/{v10_test}  search={arm_seconds/60:.0f}min "
          f"({n_arms} arms)", flush=True)


if __name__ == "__main__":
    main()
