#!/usr/bin/env python3
"""v11.3 evolutionary heterogeneous NAS -- a test implementation.

Regularised (aging) evolution over the SAME heterogeneous micro-macro space as
v11.2 (per-stage op from a cross-family vocab x depth x width), but population-
based instead of a greedy hill-climb, so it can escape local optima and explore
joint interactions the coordinate-descent cannot. Design = [[v11.3]]:

  - Population P, seeded from the v10-route arch + random heterogeneous archs
    (so we start no worse than what we already have).
  - Each cycle: TOURNAMENT-sample a subset, pick the best by a CHEAP-SCREEN
    (short-train val acc) -- selection pressure only, never a commit; MUTATE the
    parent (swap a stage's op / add a layer / remove a layer / change width);
    cheap-screen the child, add it, and remove the OLDEST member (age-based
    regularisation -> keeps exploring, robust to occasional bad selections).
  - COMMIT (reliability gate): take the top-K by cheap-screen and VERIFY them at
    a longer horizon; the discovered arch is the best verified one. The proxy
    wall is confined to SELECTION; the final full-budget judgement vs v10's real
    number is done by validate_winner.py (train-to-budget, real recipe).

Multi-fidelity + hardware-agnostic: cheap-screen and verify are epoch-based
(portable, fair across GPUs). Reuses v11.2's Search recipe + the op shape
pre-filter (so attention etc. are only in the vocabulary where they can run).
Bounded by a cycle count (and a wall-clock safety) so it fits a per-dataset
budget; the cluster era parallelises the population for the full version.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v112_search import (load_dataset, Clock, corr1_of, filter_ops_by_shape,   # noqa: E402
                         Search, OP_CANDS, ENCODED_CORR1)
V10 = Path(__file__).resolve().parent.parent / "baselines" / "v10"
sys.path.insert(0, str(V10))
from data_processor import DataProcessor             # noqa: E402

DEPTHS_INIT = [4, 6, 8, 10, 14]
WIDTHS = [32, 48, 64]
D_MIN, D_MAX = 3, 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--cycles", type=int, default=30)
    ap.add_argument("--tournament", type=int, default=4)
    ap.add_argument("--screen-epochs", type=int, default=10)
    ap.add_argument("--verify-epochs", type=int, default=40)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--search-hours", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    d = Path(args.dataset_dir); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va, test_x, test_y, meta = load_dataset(d)
    code = meta.get("codename", d.name)
    dp = DataProcessor(tr[0], tr[1], va[0], va[1], test_x, meta, Clock(999))
    train_loader, val_loader, _ = dp.process()
    xb0, _ = next(iter(train_loader))
    in_ch, h, w = xb0.shape[1], xb0.shape[2], xb0.shape[3]
    nc = int(meta["num_classes"])
    stem = "channel_embed" if in_ch >= 8 else "conv3"
    corr1 = corr1_of(tr[0]); encoded = corr1 < ENCODED_CORR1
    base_op = "Conv3" if encoded else "DilatedConv3"
    vocab, dropped = filter_ops_by_shape(OP_CANDS, in_ch, nc, h, w, stem, device)
    S = Search(in_ch, nc, h, w, stem, (train_loader, val_loader), device)
    clock = Clock(args.search_hours)
    SE, VE = args.screen_epochs, args.verify_epochs
    print(f"[{code}] EVO  device={device} corr1={corr1:+.4f} encoded={encoded} stem={stem} "
          f"base_op={base_op}  vocab={vocab} dropped={dropped}", flush=True)
    print(f"[{code}] pop={args.pop} cycles={args.cycles} tourn={args.tournament} "
          f"screen={SE} verify={VE} search<={args.search_hours}h", flush=True)

    def rand_genome():
        D = int(rng.choice(DEPTHS_INIT))
        return {"ops": [str(rng.choice(vocab)) for _ in range(D)],
                "width": int(rng.choice(WIDTHS))}

    def mutate(g):
        ops = list(g["ops"]); width = g["width"]; r = rng.random()
        if r < 0.40 and ops:                      # swap a stage's op
            ops[int(rng.integers(len(ops)))] = str(rng.choice(vocab))
        elif r < 0.60 and len(ops) < D_MAX:       # add a layer
            ops.insert(int(rng.integers(len(ops) + 1)), str(rng.choice(vocab)))
        elif r < 0.80 and len(ops) > D_MIN:       # remove a layer
            del ops[int(rng.integers(len(ops)))]
        else:                                     # change width
            width = int(rng.choice(WIDTHS))
        return {"ops": ops, "width": width}

    evaluated = {}     # key -> (genome, screen_val)

    def key_of(g):
        return f"{'-'.join(g['ops'])}|{g['width']}"

    def screen(g, tag):
        k = key_of(g)
        if k in evaluated:
            return evaluated[k][1]
        v, _ = S.score(g["ops"], g["width"], SE, tag)
        evaluated[k] = (g, v)
        return v

    # ---- seed the population: v10-route arch + random heterogeneous --------
    pop = [{"ops": [base_op] * 6, "width": 48}]
    while len(pop) < args.pop:
        pop.append(rand_genome())
    for i, g in enumerate(pop):
        screen(g, f"seed{i}")
    pop = [(g, evaluated[key_of(g)][1]) for g in pop]   # (genome, fitness)
    pop.sort(key=lambda x: x[1], reverse=True)
    print(f"[{code}] seeded pop best={pop[0][1]:.4f} ({key_of(pop[0][0])})", flush=True)

    # ---- aging-evolution cycles -------------------------------------------
    for c in range(args.cycles):
        if clock.check() <= 0:
            print(f"[{code}] clock hit at cycle {c}", flush=True); break
        idx = rng.choice(len(pop), size=min(args.tournament, len(pop)), replace=False)
        parent = max((pop[i] for i in idx), key=lambda x: x[1])[0]
        child = mutate(parent)
        cv = screen(child, f"cyc{c}")
        pop.append((child, cv))
        pop.pop(0)                                 # remove OLDEST (regularised)
        if (c + 1) % 10 == 0:
            best = max(pop, key=lambda x: x[1])
            print(f"[{code}]   cycle {c+1:3d}  child={cv:.4f}  pop_best={best[1]:.4f}  "
                  f"evals={len(evaluated)}", flush=True)

    # ---- commit: verify the top-K distinct by screen, pick best verified ---
    ranked = sorted(evaluated.values(), key=lambda x: x[1], reverse=True)
    seen, topk = set(), []
    for g, sv in ranked:
        k = key_of(g)
        if k in seen:
            continue
        seen.add(k); topk.append((g, sv))
        if len(topk) >= args.top_k:
            break
    best_g, best_verify = None, -1.0
    for g, sv in topk:
        vv, _ = S.score(g["ops"], g["width"], VE, f"verify_{key_of(g)}")
        print(f"[{code}]   verify screen={sv:.4f} -> verified={vv:.4f}  {key_of(g)}", flush=True)
        if vv > best_verify:
            best_verify, best_g = vv, g

    ops, width = best_g["ops"], best_g["width"]
    rec = {"codename": code, "dataset": d.name, "stem": stem, "device": str(device),
           "search_method": "evo_aging", "fp_corr1": round(corr1, 4),
           "fp_encoded": bool(encoded), "op_vocab": vocab, "ops_dropped_shape": dropped,
           "in_ch": in_ch, "h": h, "w": w, "num_classes": nc, "seed": args.seed,
           "evo": {"pop": args.pop, "cycles": args.cycles, "tournament": args.tournament,
                   "screen_epochs": SE, "verify_epochs": VE, "n_evaluated": len(evaluated),
                   "verified_best": round(best_verify, 4)},
           "searched": {"ops": ops, "width": width, "depth": len(ops),
                        "heterogeneous": len(set(ops)) > 1, "final_val": round(best_verify, 4)},
           "decision": {"winner": "searched"},
           "timing": {"n_arms": S.n_arms, "search_seconds": round(S.arm_seconds, 1)}}
    (out / f"{code}.json").write_text(json.dumps(rec, indent=2))
    print(f"[{code}] DONE  D={len(ops)} W{width} verified={best_verify:.4f} arms={S.n_arms}  "
          f"ops={ops}", flush=True)


if __name__ == "__main__":
    main()
