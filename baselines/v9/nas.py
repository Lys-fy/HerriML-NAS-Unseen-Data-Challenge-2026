"""Tier-2b-full NAS: EGNAS-light **macro + micro** architecture search.

The COMPLETE EGNAS-light method. This dir is a copy of
`baselines/egnas_macro/` (frozen as the macro-only comparator) with
the micro phase added — so a full-vs-macro diff on the same datasets/seeds
cleanly attributes exactly what the micro op-search buys. Only `nas.py`
differs from egnas_macro (and from Tier 1).

**Why full EGNAS, not just macro (reasoning):** the macro-only diagnostic
proved architecture search matters (it cracked the CIFARTile ceiling no
Tier-1 config could move) but only *sipped* the headroom, and its
argmax-over-noisy-candidates selection was a Q8 arch-lottery (s2024 picked
D=5 for AddNIST → −2.81). Concluding "conv-search is tapped out / pivot to
E2" from *half* the proven 2025-winning method was premature. The micro
phase (a) searches *what each layer computes* (not just depth/width),
completing the method before we judge it, and (b) is an incumbent-kept
hill-climb — structurally robust to the noisy short-train val (a single
fluke cannot hijack it), which is the *principled* fix for the macro
lottery rather than a bolted-on denoiser.

  PHASE 1 depth-grow  (macro): append blocks @ W_MAX while val improves.
  PHASE 2 width-prune (macro): shrink width while val retained.
  PHASE 3 micro op-hill-climb: from the all-Conv3 macro winner, per layer
      try {Conv3,Conv5,SepConv3}; keep a swap only if it beats the
      incumbent by MICRO_MARGIN. Budget-bounded; failure keeps the macro
      winner (never the Tier-1 fallback).
  Return the single best arch fresh (random init, DropPath on) → the
  unchanged Tier-1 Trainer. Full evaluated pool logged for Tier-3.

Budget: macro gets MACRO_FRACTION of the 30 % search slice, micro the
rest. Clock hygiene inherited from Tier 1; whole-search failure → Tier-1
default arch (never a new -10). DataProcessor/Trainer/helpers
byte-identical to Tier 1.

Plan: docs/notes/09 - Implementation/plan/Tier 2b - EGNAS-light (macro).md
"""

from __future__ import annotations

import json
import os
import time
import traceback
from typing import Dict, List

import torch
import torch.nn as nn

from helpers import build_cnn, build_default_cnn, CHANNEL_EMBED_MIN_CHANNELS


# --- Search-space + budget knobs (env-overridable for diagnostics) -----------
SEARCH_BUDGET_FRACTION = 0.30        # same split as Tier 1: 30% NAS / rest Trainer
MACRO_OP = "Conv3"                   # macro-phase op; micro then searches per-layer ops
MACRO_FRACTION = 0.6                  # macro gets 60% of the search budget; micro 40%
MICRO_OPS = ("Conv3", "Conv5", "Conv7", "SepConv3")  # v7: conv-only (+Conv7); NO exotic ops
MICRO_MARGIN = 0.003                 # accept a per-layer op swap only if val improves > this
D_MIN = 3
D_MAX = int(os.environ.get("EGNAS_DMAX", "12"))
W_MAX = 48                           # = Tier-1 base_width (comparable baseline)
W_MIN = int(os.environ.get("EGNAS_WMIN", "16"))
W_STEP = 8
DEPTH_STRIKES = 3
WIDTH_STRIKES = 3
WIDTH_TOL = 0.01                     # accept a prune if val ≥ ref − 1%
BASE_SHORT_EPOCHS = 2                # arch-aware schedule: epochs grow with depth
MAX_SHORT_EPOCHS = 6

# v7 MACRO-PRUNE VERIFICATION — the headline fix for the v6 regressions.
# The 3-seed deep analysis showed the short-train ranker mis-judges macro
# decisions at BOTH boundaries:
#   - depth false-STRIKE: a deep net looks terrible at K epochs but solves at
#     full budget (GameOfLife — D=6 looked like 0.28 at K=12, trained to 99%).
#   - width false-ACCEPT: a narrow net looks within-tol at K epochs but craters
#     at full budget (Sudoku — W=16 accepted, dropped ~20 pp by 2 h).
# Fix: when the macro loop is about to make a "kill" decision (strike a deeper
# candidate, or accept a width prune), RE-TRAIN that candidate at VERIFY_K_MULT×
# the short epochs and re-check before committing. A net still climbing
# (positive slope) overtakes under verification; a net that only looked good at
# the short horizon falls back. Bounded by MAX_MACRO_VERIFY firings/phase so the
# cost stays a small fraction of the macro budget. Calibrate K_MULT/cap on GPU.
VERIFY_MACRO = os.environ.get("VERIFY_MACRO", "1") == "1"
VERIFY_K_MULT = int(os.environ.get("VERIFY_K_MULT", "3"))
MAX_MACRO_VERIFY = int(os.environ.get("MAX_MACRO_VERIFY", "5"))
MAX_VERIFY_EPOCHS = int(os.environ.get("MAX_VERIFY_EPOCHS", "18"))
# Rough candidate count for sizing the per-candidate clock slice
# (depth-grow ≤ D_MAX−D_MIN, width-prune ≤ (W_MAX−W_MIN)/W_STEP, + bootstrap).
_EXPECTED_CANDS = (D_MAX - D_MIN) + ((W_MAX - W_MIN) // W_STEP) + 2


class NAS:
    def __init__(self, train_loader, valid_loader, metadata, clock):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.metadata = metadata
        self.clock = clock

    # ----------------------------------------------------------------------- #
    # Tier-1 primitives, reused verbatim (proven across all 11 datasets).
    # ----------------------------------------------------------------------- #

    def _input_shape(self):
        shape = self.metadata["input_shape"]
        if len(shape) == 4:
            _, in_channels, h, w = shape
        elif len(shape) == 3:
            in_channels, h, w = shape
        else:
            raise ValueError(f"Unexpected input_shape: {shape}")
        return int(in_channels), int(h), int(w)

    @torch.no_grad()
    def _quick_eval(self, model: nn.Module, device) -> float:
        model.eval()
        correct = 0
        total = 0
        for xb, yb in self.valid_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += yb.numel()
        return correct / max(total, 1)

    def _short_train(self, model: nn.Module, device, deadline: float, max_epochs: int) -> float:
        """Train up to max_epochs OR until perf_counter() > deadline. Returns
        val_acc; raises on NaN. Identical to the Tier-1 short-train."""
        model = model.to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.025, momentum=0.9, weight_decay=3e-4, nesterov=True
        )
        epoch = 0
        while epoch < max_epochs and time.perf_counter() < deadline:
            model.train()
            for xb, yb in self.train_loader:
                if time.perf_counter() > deadline:
                    break
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                if not torch.isfinite(loss):
                    raise RuntimeError("NaN/inf loss in short-train")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            epoch += 1
        return self._quick_eval(model, device)

    # ----------------------------------------------------------------------- #
    # EGNAS macro.
    # ----------------------------------------------------------------------- #

    @staticmethod
    def _arch_epochs(n_blocks: int) -> int:
        """Architecture-aware schedule — bigger candidates train longer.
        EGNAS reports this lifts ranking Spearman 0.65→0.91; we VERIFY it
        holds under our cuDNN nondeterminism (Q8) via the logged pool, not
        assume it."""
        return min(MAX_SHORT_EPOCHS, BASE_SHORT_EPOCHS + (n_blocks - D_MIN) // 2 + 1)

    def _eval_ops(self, in_ch, nc, h, w, ops, base_width, device, cand_deadline):
        """Build the explicit `ops` arch @ base_width, short-train, return
        (val_acc, params). DropPath off for the proxy (matches Tier-1)."""
        model = build_cnn(in_ch, nc, h, w, ops=list(ops),
                          base_width=base_width, drop_path_rate=0.0)
        try:
            params = int(sum(p.numel() for p in model.parameters()))
            val = self._short_train(model, device, deadline=cand_deadline,
                                    max_epochs=self._arch_epochs(len(ops)))
            return val, params
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _eval_arch(self, in_ch, nc, h, w, n_blocks, base_width, device, cand_deadline):
        """Macro proxy: uniform [MACRO_OP]*n_blocks @ base_width."""
        return self._eval_ops(in_ch, nc, h, w, [MACRO_OP] * n_blocks,
                              base_width, device, cand_deadline)

    def _verify_arch(self, in_ch, nc, h, w, n_blocks, base_width, device,
                     verify_deadline):
        """Re-train [MACRO_OP]*n_blocks @ base_width at VERIFY_K_MULT× the
        normal short epochs (capped at MAX_VERIFY_EPOCHS) to see whether a
        macro short-train decision holds at a longer horizon. A still-climbing
        net overtakes here; a short-horizon mirage falls back. Returns val_acc
        (or -1.0 on failure, so a failed verify never rescues/blocks)."""
        ep = min(MAX_VERIFY_EPOCHS, VERIFY_K_MULT * self._arch_epochs(n_blocks))
        model = build_cnn(in_ch, nc, h, w, ops=[MACRO_OP] * n_blocks,
                          base_width=base_width, drop_path_rate=0.0)
        try:
            return self._short_train(model, device, deadline=verify_deadline,
                                     max_epochs=ep)
        except Exception as ex:
            print(f"    verify FAILED ({type(ex).__name__}: {ex})")
            return -1.0
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _log_pool(self, pool: List[Dict]) -> None:
        log_dir = os.environ.get("BASELINE_LOG_DIR")
        if not log_dir or not pool:
            return
        try:
            os.makedirs(log_dir, exist_ok=True)
            code = self.metadata.get("codename", "unknown")
            path = os.path.join(log_dir, f"{code}_egnas_pool.json")
            with open(path, "w") as f:
                json.dump(pool, f, indent=2)
            print(f"  EGNAS: logged {len(pool)} evaluated archs → {path}")
        except Exception as ex:  # logging must never break the submission
            print(f"  EGNAS: pool log failed ({type(ex).__name__}: {ex})")

    def search(self):
        in_ch, h, w = self._input_shape()
        nc = int(self.metadata["num_classes"])
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # v9 telemetry: the backbone is the dilated TCN-2d block (searched depth).
        print(f"  v9 backbone: dilated TCN-2d (in={in_ch}x{h}x{w}, classes={nc})")

        # Sanity hook: EGNAS_OFF=1 → return the Tier-1 default arch (6-block
        # all-Conv3), for A/B against the macro search.
        if os.environ.get("EGNAS_OFF") == "1":
            print("  EGNAS_OFF=1: returning Tier-1 default 6-block Conv3 arch")
            return build_default_cnn(in_channels=in_ch, num_classes=nc,
                                     input_h=h, input_w=w, base_width=W_MAX,
                                     n_blocks=6, drop_path_rate=0.1)

        # Experiment hook: FIXED_ARCH="Op1,Op2,..." skips search and builds the
        # named op sequence for the full trainer — used by the out-of-band
        # full-budget op probes (e.g. the recurrent-op Language A/B). No search.
        fixed = os.environ.get("FIXED_ARCH")
        if fixed:
            ops = [o.strip() for o in fixed.split(",") if o.strip()]
            fw = int(os.environ.get("FIXED_WIDTH", str(W_MAX)))
            print(f"  FIXED_ARCH: building {len(ops)}-block arch @ W={fw}: {ops}")
            return build_cnn(in_ch, nc, h, w, ops=ops, base_width=fw,
                             drop_path_rate=0.1)

        total_remaining = self.clock.check()
        search_budget = total_remaining * SEARCH_BUDGET_FRACTION
        per_cand = search_budget / max(_EXPECTED_CANDS, 1)
        search_deadline = time.perf_counter() + search_budget
        macro_deadline = time.perf_counter() + search_budget * MACRO_FRACTION
        print(f"  EGNAS full: search budget {search_budget:.0f}s "
              f"(macro {MACRO_FRACTION:.0%} / micro {1 - MACRO_FRACTION:.0%}, "
              f"~{per_cand:.0f}s/cand, D∈[{D_MIN},{D_MAX}] W∈[{W_MIN},{W_MAX}])")

        pool: List[Dict] = []
        best_D, best_W, best_val = D_MIN, W_MAX, -1.0

        try:
            # Bootstrap: the smallest arch.
            cd = min(time.perf_counter() + per_cand, macro_deadline)
            best_val, p0 = self._eval_arch(in_ch, nc, h, w, D_MIN, W_MAX, device, cd)
            pool.append({"phase": "bootstrap", "D": D_MIN, "W": W_MAX,
                         "op": MACRO_OP, "val": round(best_val, 4), "params": p0})
            print(f"  bootstrap   D={D_MIN:2d} W={W_MAX}  val={best_val:.4f}")

            # PHASE 1 — depth-grow (with v7 verify-of-rejects).
            D, strikes = D_MIN, 0
            macro_verify_used = 0
            while D < D_MAX and time.perf_counter() < macro_deadline \
                    and strikes < DEPTH_STRIKES:
                D += 1
                cd = min(time.perf_counter() + per_cand, macro_deadline)
                try:
                    val, params = self._eval_arch(in_ch, nc, h, w, D, W_MAX, device, cd)
                except Exception as ex:
                    print(f"  depth D={D:2d}  FAILED ({type(ex).__name__}: {ex})")
                    strikes += 1
                    continue
                pool.append({"phase": "depth", "D": D, "W": W_MAX,
                             "op": MACRO_OP, "val": round(val, 4), "params": params})
                if val > best_val:
                    print(f"  depth   D={D:2d} W={W_MAX}  val={val:.4f}  ↑ (best)")
                    best_val, best_D, strikes = val, D, 0
                    continue
                # v7: a deeper candidate that looks worse at K epochs may be a
                # FALSE STRIKE (still climbing — GameOfLife). Verify at
                # VERIFY_K_MULT× before striking, while budget/cap allow.
                rescued = False
                if VERIFY_MACRO and macro_verify_used < MAX_MACRO_VERIFY \
                        and time.perf_counter() < macro_deadline:
                    vd = min(time.perf_counter() + VERIFY_K_MULT * per_cand,
                             macro_deadline)
                    vval = self._verify_arch(in_ch, nc, h, w, D, W_MAX, device, vd)
                    macro_verify_used += 1
                    pool.append({"phase": "depth_verify", "D": D, "W": W_MAX,
                                 "op": MACRO_OP, "val": round(vval, 4),
                                 "short_val": round(val, 4)})
                    if vval > best_val:
                        print(f"  depth   D={D:2d} W={W_MAX}  val={val:.4f}→"
                              f"verify={vval:.4f}  ↑ (best, RESCUED false-strike)")
                        best_val, best_D, strikes = vval, D, 0
                        rescued = True
                if not rescued:
                    strikes += 1
                    print(f"  depth   D={D:2d} W={W_MAX}  val={val:.4f}  "
                          f"strike {strikes}/{DEPTH_STRIKES}")

            # PHASE 2 — width-prune at the winning depth (with v7 verify-of-accepts).
            # The Sudoku failure: a narrow net is within WIDTH_TOL of the wide
            # net at K epochs but craters ~20 pp by 2 h. To judge the prune
            # honestly we compare narrow-vs-wide at the SAME longer horizon:
            # verify the wide depth-winner once at VERIFY_K_MULT×, then accept a
            # prune only if the narrow net (also verified) stays within tol of
            # that long-horizon reference. Falls back to the short-horizon rule
            # if verification is disabled or budget/cap is exhausted.
            ref_val = best_val
            ref_verify = None
            W, strikes = W_MAX, 0
            while W - W_STEP >= W_MIN and time.perf_counter() < macro_deadline \
                    and strikes < WIDTH_STRIKES:
                W -= W_STEP
                cd = min(time.perf_counter() + per_cand, macro_deadline)
                try:
                    val, params = self._eval_arch(in_ch, nc, h, w, best_D, W, device, cd)
                except Exception as ex:
                    print(f"  width W={W:2d}  FAILED ({type(ex).__name__}: {ex})")
                    strikes += 1
                    continue
                pool.append({"phase": "width", "D": best_D, "W": W,
                             "op": MACRO_OP, "val": round(val, 4), "params": params})
                if val < ref_val - WIDTH_TOL:  # clearly worse even at K → strike
                    strikes += 1
                    print(f"  width   D={best_D:2d} W={W:2d}  val={val:.4f}  "
                          f"strike {strikes}/{WIDTH_STRIKES}")
                    continue
                # Short-horizon says "prune OK" — verify it's not a mirage.
                accept = True
                if VERIFY_MACRO and macro_verify_used < MAX_MACRO_VERIFY \
                        and time.perf_counter() < macro_deadline:
                    if ref_verify is None:  # verify the wide reference once
                        rd = min(time.perf_counter() + VERIFY_K_MULT * per_cand,
                                 macro_deadline)
                        ref_verify = self._verify_arch(in_ch, nc, h, w, best_D,
                                                       best_W, device, rd)
                        macro_verify_used += 1
                    vd = min(time.perf_counter() + VERIFY_K_MULT * per_cand,
                             macro_deadline)
                    vval = self._verify_arch(in_ch, nc, h, w, best_D, W, device, vd)
                    macro_verify_used += 1
                    pool.append({"phase": "width_verify", "D": best_D, "W": W,
                                 "op": MACRO_OP, "val": round(vval, 4),
                                 "short_val": round(val, 4),
                                 "ref_verify": round(ref_verify, 4)})
                    if ref_verify >= 0 and vval < ref_verify - WIDTH_TOL:
                        accept = False
                        strikes += 1
                        print(f"  width   D={best_D:2d} W={W:2d}  val={val:.4f}→"
                              f"verify={vval:.4f} (ref_verify={ref_verify:.4f})  "
                              f"✗ prune REJECTED (mirage)")
                if accept:
                    print(f"  width   D={best_D:2d} W={W:2d}  val={val:.4f}  "
                          f"✓ pruned (≥ ref−{WIDTH_TOL})")
                    best_W = W
                    best_val = max(best_val, val)
                    strikes = 0
        except Exception as ex:
            print(f"  EGNAS macro FAILED ({type(ex).__name__}: {ex}) — "
                  f"falling back to Tier-1 default arch")
            traceback.print_exc()
            self._log_pool(pool)
            return build_default_cnn(in_channels=in_ch, num_classes=nc,
                                     input_h=h, input_w=w, base_width=W_MAX,
                                     n_blocks=6, drop_path_rate=0.1)

        self._log_pool(pool)
        print(f"  EGNAS macro done: best D={best_D} W={best_W} "
              f"val={best_val:.4f} ({len(pool)} archs evaluated)")

        # ---- PHASE 3 — micro op hill-climb (completes full EGNAS) ----------
        # From the all-Conv3 macro winner, greedily try {Conv3,Conv5,
        # SepConv3} per layer; keep a swap only if it beats the incumbent by
        # MICRO_MARGIN. Incumbent-kept ⇒ a single noisy short-train cannot
        # hijack it — the principled fix for the macro argmax Q8 lottery.
        # Budget-bounded by search_deadline; any failure keeps the macro
        # winner (NOT the Tier-1 fallback).
        cur_ops = [MACRO_OP] * best_D
        # BUGFIX: the macro verify-of-rejects can set best_val to a *verified*
        # (long-horizon) value, but micro trials are short-train — comparing
        # short-train trials to a verified incumbent makes the micro accept
        # NOTHING (dead micro on ~80% of datasets, e.g. MultNIST never picks the
        # Conv5 it wants). Re-evaluate the macro winner at the SAME short-train
        # horizon the micro trials use, so the comparison is apples-to-apples.
        if VERIFY_MACRO and not os.environ.get("EGNAS_MICRO_OFF") == "1":
            try:
                cd = min(time.perf_counter() + per_cand, search_deadline)
                cur_val, _ = self._eval_ops(in_ch, nc, h, w, cur_ops, best_W,
                                            device, cd)
                print(f"  micro: incumbent re-eval @ short-train val={cur_val:.4f} "
                      f"(macro-verified was {best_val:.4f})")
            except Exception:
                cur_val = best_val   # fall back to the (inflated) macro value
        else:
            cur_val = best_val
        if os.environ.get("EGNAS_MICRO_OFF") == "1":
            print("  EGNAS_MICRO_OFF=1: skipping micro (≡ macro-only)")
        else:
            try:
                for i in range(best_D):
                    if time.perf_counter() >= search_deadline:
                        print("  micro: budget exhausted, stopping")
                        break
                    for op in MICRO_OPS:
                        if op == cur_ops[i] or \
                                time.perf_counter() >= search_deadline:
                            continue
                        trial = list(cur_ops)
                        trial[i] = op
                        cd = min(time.perf_counter() + per_cand,
                                 search_deadline)
                        try:
                            val, params = self._eval_ops(
                                in_ch, nc, h, w, trial, best_W, device, cd)
                        except Exception as ex:
                            print(f"  micro L{i:2d} {op:8s} FAILED "
                                  f"({type(ex).__name__})")
                            continue
                        pool.append({"phase": "micro", "D": best_D,
                                     "W": best_W, "layer": i, "op": op,
                                     "ops": list(trial),
                                     "val": round(val, 4), "params": params})
                        if val > cur_val + MICRO_MARGIN:
                            print(f"  micro L{i:2d} {cur_ops[i]}→{op}  "
                                  f"val={val:.4f}  ✓ accept")
                            cur_ops, cur_val = trial, val
                        else:
                            print(f"  micro L{i:2d} {op:8s} val={val:.4f}  "
                                  f"✗ keep {cur_ops[i]}")
            except Exception as ex:
                print(f"  micro FAILED ({type(ex).__name__}: {ex}) — "
                      f"keeping macro winner")

        self._log_pool(pool)
        n_chg = sum(1 for o in cur_ops if o != MACRO_OP)
        print(f"  EGNAS full done: D={best_D} W={best_W} "
              f"micro-changed {n_chg}/{best_D}  val={cur_val:.4f} "
              f"({len(pool)} evaluated)  ops={cur_ops}")

        # Build the winner fresh (random init), DropPath on for the full
        # retrain — handed to the unchanged Tier-1 Trainer.
        return build_cnn(in_ch, nc, h, w, ops=cur_ops,
                         base_width=best_W, drop_path_rate=0.1)
