"""Tier-2b NAS: EGNAS-light **macro** architecture search.

The first real NAS engine in the repo. Macro-only v1 (depth-grow then
width-prune); the micro op/kernel hill-climb is deferred to v1.1. Fixed op
= Conv3 (same primitive as the Tier-1 default, so this isolates "does
searching depth/width beat Tier-1's fixed 6-block arch?").

Plan: docs/notes/09 - Implementation/plan/Tier 2b - EGNAS-light (macro).md
Algorithm reference: docs/notes/09 - Implementation/plan/Tier 2 - EGNAS-Light.md

  PHASE 1 depth-grow:  from D_MIN blocks, append one block at a time at
      W_MAX; short-train each (architecture-aware: bigger → more epochs);
      track best depth; stop on D_MAX or DEPTH_STRIKES consecutive
      non-improvements.
  PHASE 2 width-prune: at the best depth, shrink base_width by W_STEP while
      val is *retained* (≥ ref − WIDTH_TOL); stop on W_MIN or strikes.
  Return the single best (D, W) arch fresh (random init, DropPath on) →
  the unchanged Tier-1 Trainer full-trains it.

Design decisions (locked 2026-05-18):
  - return SINGLE best; LOG the full evaluated pool to disk so Tier-3
    ensembling is free later (no extra train cost now).
  - clock hygiene inherited from Tier 1: per-candidate deadline, clock
    check between candidates, try/except per candidate, and if the WHOLE
    search fails → fall back to the Tier-1 default arch. Never a new -10.

Everything else (DataProcessor, Trainer, helpers) is byte-identical to
Tier 1 — only this file differs.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from typing import Dict, List

import torch
import torch.nn as nn

from helpers import build_cnn, build_default_cnn


# --- Search-space + budget knobs (env-overridable for diagnostics) -----------
SEARCH_BUDGET_FRACTION = 0.30        # same split as Tier 1: 30% NAS / rest Trainer
MACRO_OP = "Conv3"                   # fixed in macro (op search is micro = v1.1)
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

    def _eval_arch(self, in_ch, nc, h, w, n_blocks, base_width, device, cand_deadline):
        """Build [Conv3]*n_blocks @ base_width, short-train, return val_acc.
        DropPath off for the proxy (matches Tier-1 search)."""
        model = build_cnn(in_ch, nc, h, w, ops=[MACRO_OP] * n_blocks,
                          base_width=base_width, drop_path_rate=0.0)
        try:
            params = int(sum(p.numel() for p in model.parameters()))
            val = self._short_train(model, device, deadline=cand_deadline,
                                    max_epochs=self._arch_epochs(n_blocks))
            return val, params
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

        # Sanity hook: EGNAS_OFF=1 → return the Tier-1 default arch (6-block
        # all-Conv3), for A/B against the macro search.
        if os.environ.get("EGNAS_OFF") == "1":
            print("  EGNAS_OFF=1: returning Tier-1 default 6-block Conv3 arch")
            return build_default_cnn(in_channels=in_ch, num_classes=nc,
                                     input_h=h, input_w=w, base_width=W_MAX,
                                     n_blocks=6, drop_path_rate=0.1)

        total_remaining = self.clock.check()
        search_budget = total_remaining * SEARCH_BUDGET_FRACTION
        per_cand = search_budget / max(_EXPECTED_CANDS, 1)
        search_deadline = time.perf_counter() + search_budget
        print(f"  EGNAS macro: search budget {search_budget:.0f}s "
              f"(~{per_cand:.0f}s/cand, D∈[{D_MIN},{D_MAX}] "
              f"W∈[{W_MIN},{W_MAX}] op={MACRO_OP})")

        pool: List[Dict] = []
        best_D, best_W, best_val = D_MIN, W_MAX, -1.0

        try:
            # Bootstrap: the smallest arch.
            cd = min(time.perf_counter() + per_cand, search_deadline)
            best_val, p0 = self._eval_arch(in_ch, nc, h, w, D_MIN, W_MAX, device, cd)
            pool.append({"phase": "bootstrap", "D": D_MIN, "W": W_MAX,
                         "op": MACRO_OP, "val": round(best_val, 4), "params": p0})
            print(f"  bootstrap   D={D_MIN:2d} W={W_MAX}  val={best_val:.4f}")

            # PHASE 1 — depth-grow.
            D, strikes = D_MIN, 0
            while D < D_MAX and time.perf_counter() < search_deadline \
                    and strikes < DEPTH_STRIKES:
                D += 1
                cd = min(time.perf_counter() + per_cand, search_deadline)
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
                else:
                    strikes += 1
                    print(f"  depth   D={D:2d} W={W_MAX}  val={val:.4f}  "
                          f"strike {strikes}/{DEPTH_STRIKES}")

            # PHASE 2 — width-prune at the winning depth.
            ref_val = best_val
            W, strikes = W_MAX, 0
            while W - W_STEP >= W_MIN and time.perf_counter() < search_deadline \
                    and strikes < WIDTH_STRIKES:
                W -= W_STEP
                cd = min(time.perf_counter() + per_cand, search_deadline)
                try:
                    val, params = self._eval_arch(in_ch, nc, h, w, best_D, W, device, cd)
                except Exception as ex:
                    print(f"  width W={W:2d}  FAILED ({type(ex).__name__}: {ex})")
                    strikes += 1
                    continue
                pool.append({"phase": "width", "D": best_D, "W": W,
                             "op": MACRO_OP, "val": round(val, 4), "params": params})
                if val >= ref_val - WIDTH_TOL:  # accept the smaller, cheaper arch
                    print(f"  width   D={best_D:2d} W={W:2d}  val={val:.4f}  "
                          f"✓ pruned (≥ ref−{WIDTH_TOL})")
                    best_W = W
                    best_val = max(best_val, val)
                    strikes = 0
                else:
                    strikes += 1
                    print(f"  width   D={best_D:2d} W={W:2d}  val={val:.4f}  "
                          f"strike {strikes}/{WIDTH_STRIKES}")
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

        # Build the winner fresh (random init), DropPath on for the full
        # retrain — handed to the unchanged Tier-1 Trainer.
        return build_cnn(in_ch, nc, h, w, ops=[MACRO_OP] * best_D,
                         base_width=best_W, drop_path_rate=0.1)
