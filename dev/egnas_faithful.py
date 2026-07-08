#!/usr/bin/env python3
"""Faithful EGNAS replica -- the REAL 2025-winning Efficient-Macro-Micro-NAS
algorithm, verified against the upstream code (github.com/siddikui/
Efficient-Macro-Micro-NAS, search.py) and paper (arXiv:2502.03553, Algorithm 1),
2026-06-22. This is NOT our v6/v7 port -- it reproduces the original's exact
search space, thresholds, generosity, param-fairness and NO-verification policy,
to test the open question: does the real EGNAS (not our elaborated port) beat v10?

VERIFIED SPEC (from search.py argparse + loop bodies):
  Search space: depth D in [10, 100]; base width/channels in [16, 256] step 2;
    per-layer op O in {SepConv(0), PlainConv(1)}; per-layer kernel K in {3, 5}.
  Init: D = min_depth (10); channels = max_width; all ops = SepConv; all K = 3.
  MACRO grow:  L<-L+1, epochs E<-E+add_epochs(1) each step. Per verbatim loop:
      if acc >= curr + dp_add_tolerance(0.10pp): commit, depth_fail=0
      elif acc >= curr - 0.05pp:                 continue (keep layer, no commit)
      elif depth_fail != dp_break_tolerance(2):  depth_fail+=1, continue
      else: break.  Grown layers are NEVER reverted -> D_f = depth at break.
  MACRO prune: C<-C-width_resolution(2), epochs E<-E+add_epochs_w(2) each step.
      if acc >= curr - 0.0: commit f_channels=C, width_fail=0
      elif width_fail != ch_break_tolerance(2): width_fail+=1, continue
      else: break.  Final width = last accepted f_channels.
  MICRO (re-baseline at E_min=2 epochs):
      op-replace: per layer try O_i <- PlainConv; param-equalise (shrink global
        channels while params_MB > baseline+0.01); accept if acc > curr-0.0.
      kernel:     per layer try K_i <- 5; same param-equalise; accept if > curr-0.0.
  No verification of accepts (commits on the single short-train -- the original's
  600-epoch final train washes out the noise). Ranks on TEST acc (valid_size=0).
  SGD 0.025 cosine->0, wd 3e-4, mom 0.9, grad-clip 5, cutout 16, NO label-smooth.

DOCUMENTED budget-clamps (the competition is ~1 h/dataset on unknown HW; the
original ran unbounded over ~0.24 GPU-days on CIFAR-10 with a 600-epoch final):
  - init width W_max defaults to 64 (the paper's "Tiny" setting, SOTA on EMNIST/
    KMNIST) not 256; overridable via --w-max.
  - per-candidate epochs capped (--ep-cap, default 25) so the arch-aware +1/+2
    schedule cannot explode; a search wall-clock Clock (--search-hours) is a
    safety net (the strike rules usually stop the phases far sooner).
  - ranks on VAL acc (legal; we have a val split and test_y may be absent), not
    test acc -- the only deviation forced by the competition rules.
  - the discovered arch's FINAL train is delegated to validate_winner.py (v10's
    real recipe, train-to-budget) instead of 600 epochs, for apples-to-apples vs
    v10's known real numbers.
  - macro skeleton (downsample positions + channel doubling) uses v10's adaptive
    plan_downsamples (robust to tiny/odd competition inputs) rather than EGNAS's
    fixed 1/3 & 2/3 doubling (which collapses 9x9 inputs); the per-layer op /
    kernel / depth / width SEARCH is faithful.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

V10 = Path(__file__).resolve().parent.parent / "baselines" / "v10"
sys.path.insert(0, str(V10))
from helpers import build_cnn                       # noqa: E402
from data_processor import DataProcessor             # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v112_search import load_dataset, evaluate, predict, Clock, corr1_of  # noqa: E402

ADD_TOL = 0.0010     # dp_add_tolerance 0.10pp  -> fraction
DROP_TOL = 0.0005    # 0.05pp                    -> fraction
BREAK_TOL = 2        # dp_break_tolerance = ch_break_tolerance = 2
W_RES = 2            # width_resolution
W_MIN = 16
D_MAX = 100
E0 = 10              # macro initial epochs
ADD_EP = 1           # +1 per grow step
ADD_EP_W = 2         # +2 per prune step
E_MIN = 2            # micro epochs


def op_name(fam, k):
    return {("Sep", 3): "SepConv3", ("Sep", 5): "SepConv5",
            ("Conv", 3): "Conv3", ("Conv", 5): "Conv5"}[(fam, k)]


def _mk(fams, kernels, width, ctx):
    ops = [op_name(f, k) for f, k in zip(fams, kernels)]
    return build_cnn(ctx["in_ch"], ctx["nc"], ctx["h"], ctx["w"], ops=ops,
                     base_width=width, drop_path_rate=0.0, stem=ctx["stem"])


def params_mb(fams, kernels, width, ctx):
    m = _mk(fams, kernels, width, ctx)
    p = sum(x.numel() for x in m.parameters()) / 1e6
    del m
    return p


def equalise(fams, kernels, width, baseline_mb, ctx):
    """Shrink the global channel count until params_MB <= baseline + 0.01
    (EGNAS's param-fairness for heavier ops/kernels). Returns the new width."""
    w = width
    while w > W_MIN and params_mb(fams, kernels, w, ctx) > baseline_mb + 0.01:
        w -= 1
    return w


def fit(model, epochs, ctx):
    """EGNAS training: SGD 0.025 cosine->0, wd 3e-4, mom 0.9, clip 5, NO label
    smoothing. Epoch-based. Returns best val acc (0-1)."""
    tl, vl = ctx["loaders"]
    dev = ctx["device"]
    model = model.to(dev)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    t0 = time.perf_counter(); cap = t0 + epochs * ctx["cap_per_epoch"]; best = 0.0
    for _ in range(epochs):
        if time.perf_counter() > cap:
            break
        model.train()
        for xb, yb in tl:
            if time.perf_counter() > cap:
                break
            xb = xb.to(dev, non_blocking=True); yb = yb.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            if not torch.isfinite(loss):
                raise RuntimeError("NaN")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        best = max(best, evaluate(model, vl, dev)); sched.step()
    return best


def score(fams, kernels, width, epochs, ctx):
    epochs = min(epochs, ctx["ep_cap"])
    try:
        m = _mk(fams, kernels, width, ctx)
        acc = fit(m, epochs, ctx); del m
    except Exception as e:
        print(f"   cand FAILED: {type(e).__name__}: {e}", flush=True); acc = 0.0
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    ctx["n_arms"] += 1
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--w-max", type=int, default=64)        # paper "Tiny"
    ap.add_argument("--ep-cap", type=int, default=25)
    ap.add_argument("--search-hours", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
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
    corr1 = corr1_of(tr[0])
    ctx = {"in_ch": in_ch, "nc": nc, "h": h, "w": w, "stem": stem, "device": device,
           "loaders": (train_loader, val_loader), "cap_per_epoch": 60.0,
           "ep_cap": args.ep_cap, "n_arms": 0}
    clock = Clock(args.search_hours)
    print(f"[{code}] FAITHFUL-EGNAS  device={device} corr1={corr1:+.4f} stem={stem} "
          f"in_ch={in_ch} {h}x{w} nc={nc}  W_max={args.w_max} ep_cap={args.ep_cap} "
          f"search<={args.search_hours}h", flush=True)

    # ---- init: SepConv3 x 10, channels = W_max -----------------------------
    fams = ["Sep"] * 10; kernels = [3] * 10; width = args.w_max
    epochs = E0
    curr = score(fams, kernels, width, epochs, ctx)
    print(f"[{code}] init D=10 SepConv3 W{width}  val={curr:.4f}", flush=True)

    # ---- MACRO grow (faithful: never revert; commit on +0.10pp) ------------
    depth_fail = 0
    while len(fams) < D_MAX and clock.check() > 0:
        epochs += ADD_EP
        fams.append("Sep"); kernels.append(3)          # grow, no revert
        acc = score(fams, kernels, width, epochs, ctx)
        if acc >= curr + ADD_TOL:
            curr = acc; depth_fail = 0; tag = "GROW"
        elif acc >= curr - DROP_TOL:
            tag = "tol"
        elif depth_fail != BREAK_TOL:
            depth_fail += 1; tag = f"strike{depth_fail}"
        else:
            fams.pop(); kernels.pop(); tag = "BREAK"
        print(f"[{code}]   grow D={len(fams):2d} ep={min(epochs,args.ep_cap)} "
              f"val={acc:.4f} -> {tag}", flush=True)
        if tag == "BREAK":
            break
    Df = len(fams)

    # ---- MACRO prune width (faithful: accept any non-degradation) ----------
    f_width = width; width_fail = 0; ch = width
    while ch - W_RES >= W_MIN and clock.check() > 0:
        ch -= W_RES; epochs += ADD_EP_W
        acc = score(fams, kernels, ch, epochs, ctx)
        if acc >= curr - 0.0:
            curr = acc; f_width = ch; width_fail = 0; tag = "PRUNE"
        elif width_fail != BREAK_TOL:
            width_fail += 1; tag = f"strike{width_fail}"
        else:
            tag = "BREAK"
        print(f"[{code}]   prune W={ch} ep={min(epochs,args.ep_cap)} val={acc:.4f} -> {tag}",
              flush=True)
        if tag == "BREAK":
            break
    width = f_width

    # ---- MICRO op-replace: SepConv -> PlainConv, param-equalised, E_min ----
    baseline_mb = params_mb(fams, kernels, width, ctx)
    for i in range(len(fams)):
        if clock.check() <= 0:
            break
        trial = list(fams); trial[i] = "Conv"
        w2 = equalise(trial, kernels, width, baseline_mb, ctx)
        acc = score(trial, kernels, w2, E_MIN, ctx)
        if acc > curr - 0.0:
            fams = trial; curr = acc; width = w2; tag = "Conv ADOPT"
        else:
            tag = "keep Sep"
        print(f"[{code}]   op L{i:2d} -> {tag} (val={acc:.4f} W{w2})", flush=True)

    # ---- MICRO kernel-update: K3 -> K5, param-equalised, E_min -------------
    for i in range(len(kernels)):
        if clock.check() <= 0:
            break
        trial = list(kernels); trial[i] = 5
        w2 = equalise(fams, trial, width, baseline_mb, ctx)
        acc = score(fams, trial, w2, E_MIN, ctx)
        if acc > curr - 0.0:
            kernels = trial; curr = acc; width = w2; tag = "K5 ADOPT"
        else:
            tag = "keep K3"
        print(f"[{code}]   kern L{i:2d} -> {tag} (val={acc:.4f} W{w2})", flush=True)

    ops = [op_name(f, k) for f, k in zip(fams, kernels)]
    rec = {"codename": code, "dataset": d.name, "stem": stem, "device": str(device),
           "search_method": "egnas_faithful", "fp_corr1": round(corr1, 4),
           "in_ch": in_ch, "h": h, "w": w, "num_classes": nc, "seed": args.seed,
           "egnas_spec": {"w_max": args.w_max, "ep_cap": args.ep_cap, "Df": Df,
                          "add_tol_pp": 0.10, "break_tol": BREAK_TOL, "w_res": W_RES,
                          "ranks_on": "val_acc(legal)", "verify": "none(faithful)"},
           # validate_winner.py consumes decision.winner + searched + stem
           "searched": {"ops": ops, "width": width, "depth": len(ops),
                        "heterogeneous": len(set(ops)) > 1, "final_val": round(curr, 4)},
           "decision": {"winner": "searched"},
           "timing": {"n_arms": ctx["n_arms"], "search_seconds": round(args.search_hours * 3600 - max(clock.check(), 0), 1)}}
    (out / f"{code}.json").write_text(json.dumps(rec, indent=2))
    print(f"[{code}] DONE  D={len(ops)} W{width}  search_val={curr:.4f}  arms={ctx['n_arms']}  "
          f"ops={ops}", flush=True)


if __name__ == "__main__":
    main()
