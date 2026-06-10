"""Tier-2b-full NAS: EGNAS-light **macro + micro** architecture search.

The COMPLETE EGNAS-light method. This dir is a copy of
`baselines/tier2b_egnas_macro/` (frozen as the macro-only comparator) with
the micro phase added — so a full-vs-macro diff on the same datasets/seeds
cleanly attributes exactly what the micro op-search buys. Only `nas.py`
differs from tier2b_egnas_macro (and from Tier 1).

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

from helpers import build_cnn, build_default_cnn, OP_NAMES

try:
    from zeroshot import compute_all_zeroshot
except Exception:
    compute_all_zeroshot = None


# --- Search-space + budget knobs (env-overridable for diagnostics) -----------
SEARCH_BUDGET_FRACTION = 0.30        # same split as Tier 1: 30% NAS / rest Trainer
MACRO_OP = "Conv3"                   # macro-phase op; micro then searches per-layer ops
MACRO_FRACTION = 0.6                  # macro gets 60% of the search budget; micro 40%
MICRO_OPS = ("Conv3", "Conv5", "Conv7", "SepConv3", "GroupConv3", "ResNeXt",
             "DilatedConv3", "SEConv3", "SpatialSelfAttn2D")  # v6: 9 swappable
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
MAX_SHORT_EPOCHS = 12                # v6: was 6 in v5; longer K reduces ranker noise

# v6 — clock-extend mode: NAS doesn't deduct from trainer budget.
# When ON, after NAS finishes we add the NAS duration back to clock.time_limit
# so the trainer always sees the full 2h budget regardless of NAS cost.
EXTEND_CLOCK_FOR_NAS = os.environ.get("EXTEND_CLOCK_FOR_NAS", "1") == "1"

# v6 — verification-of-accepts: re-train accepted micro swaps at 2× K and
# revert if the verification val_acc doesn't hold up. Catches the
# "noisy short-train picked the wrong op" failure mode.
VERIFY_MICRO_ACCEPTS = os.environ.get("VERIFY_MICRO_ACCEPTS", "1") == "1"
VERIFY_K_MULTIPLIER = 2

# v6 — zero-shot proxies as shadow signals (computed per candidate).
COMPUTE_ZEROSHOT = os.environ.get("COMPUTE_ZEROSHOT", "1") == "1"
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
        # Phase A / TSE: per-epoch mean training losses, stashed for _eval_ops
        # to log. Enables offline TSE-vs-valacc ranking comparison of search
        # candidates (within-family only; cross-family TSE is biased).
        self._last_epoch_losses = []
        while epoch < max_epochs and time.perf_counter() < deadline:
            model.train()
            ep_loss, ep_batches = 0.0, 0
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
                ep_loss += float(loss.item())
                ep_batches += 1
            if ep_batches:
                self._last_epoch_losses.append(round(ep_loss / ep_batches, 5))
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

    def _eval_ops(self, in_ch, nc, h, w, ops, base_width, device, cand_deadline,
                  max_epochs_override: int = None):
        """Build the explicit `ops` arch @ base_width, short-train, return
        (val_acc, params, zeroshot_scores, wall_clock_s).

        v6: also computes zero-shot proxies BEFORE training (untrained net).
        Returns rich tuple for pool logging.

        DropPath off for the proxy (matches Tier-1).
        max_epochs_override lets the verification-of-accepts caller request
        a longer training to confirm a swap actually wins."""
        t_start = time.perf_counter()
        model = build_cnn(in_ch, nc, h, w, ops=list(ops),
                          base_width=base_width, drop_path_rate=0.0)
        try:
            params = int(sum(p.numel() for p in model.parameters()))
            # v6: zero-shot proxies on the UNTRAINED random-init network.
            zs = self._zeroshot(model.to(device), device, in_ch, h, w)
            K = (max_epochs_override if max_epochs_override is not None
                 else self._arch_epochs(len(ops)))
            val = self._short_train(model, device, deadline=cand_deadline,
                                    max_epochs=K)
            wall_clock_s = time.perf_counter() - t_start
            # Phase A / TSE: one parseable line per candidate eval.
            print(f"  TSE_LOG D={len(ops)} W={base_width} val={val:.4f} "
                  f"losses={getattr(self, '_last_epoch_losses', [])}")
            return val, params, zs, wall_clock_s
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _eval_arch(self, in_ch, nc, h, w, n_blocks, base_width, device, cand_deadline):
        """Macro proxy: uniform [MACRO_OP]*n_blocks @ base_width.
        Returns the same 4-tuple as _eval_ops."""
        return self._eval_ops(in_ch, nc, h, w, [MACRO_OP] * n_blocks,
                              base_width, device, cand_deadline)

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

    def _refund_clock(self, nas_start: float) -> float:
        """v6 clock-extend: add the NAS wall-clock back to the harness clock so
        the Trainer always sees the full 2 h budget regardless of how long the
        search took. Returns the NAS duration in seconds."""
        dur = time.perf_counter() - nas_start
        if EXTEND_CLOCK_FOR_NAS:
            self.clock.time_limit += dur
            print(f"  [clock-extend] NAS used {dur:.0f}s ({dur/60:.1f} min); "
                  f"refunded to trainer clock — trainer now sees "
                  f"{self.clock.check()/60:.1f} min remaining")
        return dur

    def _zeroshot(self, model: nn.Module, device, in_ch: int, h: int, w: int):
        """Compute the 4 zero-shot proxies + param count for one candidate.
        Used as shadow signal: not consulted for decisions, but logged in
        the pool for post-sweep Spearman analysis."""
        if not COMPUTE_ZEROSHOT or compute_all_zeroshot is None:
            return {}
        try:
            # Get a small data batch from the valid loader for NASWOT/SNIP
            xb, yb = next(iter(self.valid_loader))
            return compute_all_zeroshot(
                model, xb, yb,
                input_shape=(in_ch, h, w),
                device=device,
            )
        except Exception as ex:
            return {"proxy_error": f"{type(ex).__name__}: {ex}"}

    def search(self):
        nas_start = time.perf_counter()      # v6: clock-extend reference
        in_ch, h, w = self._input_shape()
        nc = int(self.metadata["num_classes"])
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Sanity hook: EGNAS_OFF=1 → return the Tier-1 default arch (6-block
        # all-Conv3), for A/B against the macro search.
        if os.environ.get("EGNAS_OFF") == "1":
            print("  EGNAS_OFF=1: returning Tier-1 default 6-block Conv3 arch")
            model = build_default_cnn(in_channels=in_ch, num_classes=nc,
                                      input_h=h, input_w=w, base_width=W_MAX,
                                      n_blocks=6, drop_path_rate=0.1)
            self._refund_clock(nas_start)
            return model

        # Experiment hook: PROBE_ARCH=tcn1d|tcn2d builds a TCN probe (dilated-
        # conv net) instead of the conv search — full-budget numbers for the
        # TCN excursion across all datasets. FIXED_WIDTH sets base_width;
        # PROBE_TCN_BLOCKS / PROBE_TCN_KERNEL override the per-kind defaults.
        probe_arch = os.environ.get("PROBE_ARCH")
        if probe_arch in ("tcn1d", "tcn2d"):
            from helpers import build_tcn
            fw = int(os.environ.get("FIXED_WIDTH", str(W_MAX)))
            nb = int(os.environ.get("PROBE_TCN_BLOCKS", "0"))
            k = int(os.environ.get("PROBE_TCN_KERNEL", "0"))
            tstem = os.environ.get("PROBE_TCN_STEM", "conv3")
            tblock = os.environ.get("PROBE_TCN_BLOCK", "bn")
            print(f"  PROBE_ARCH={probe_arch}: building TCN @ W={fw} "
                  f"blocks={nb or 'default'} kernel={k or 'default'} "
                  f"stem={tstem} block={tblock} (in={in_ch}x{h}x{w}, classes={nc})")
            model = build_tcn(probe_arch, in_ch, nc, h, w,
                              base_width=fw, n_blocks=nb, kernel=k, stem=tstem,
                              block=tblock)
            self._refund_clock(nas_start)
            return model

        # Experiment hook: FIXED_ARCH="Op1,Op2,..." skips search entirely and
        # builds the named op sequence, then hands it to the real trainer for
        # the full budget. Used by the v7 LSTM probe to get faithful
        # full-budget numbers for fixed architectures (conv vs attn vs LSTM)
        # without any search noise. FIXED_WIDTH overrides base_width (default
        # W_MAX). No clock is consumed (search is skipped).
        fixed = os.environ.get("FIXED_ARCH")
        if fixed:
            ops = [o.strip() for o in fixed.split(",") if o.strip()]
            fw = int(os.environ.get("FIXED_WIDTH", str(W_MAX)))
            stem = os.environ.get("FIXED_STEM", "conv3")  # bake-off: conv3 | channel_embed
            print(f"  FIXED_ARCH: building {len(ops)}-block arch @ W={fw} stem={stem}: {ops}")
            model = build_cnn(in_ch, nc, h, w, ops=ops,
                              base_width=fw, drop_path_rate=0.1, stem=stem)
            self._refund_clock(nas_start)
            return model

        # Experiment hook: RAND_SEARCH=1 — random search over a WIDE op vocab.
        # The random-Bonsai idea: directed/greedy search overfits to normal-image
        # architectures, so on weird-shaped inputs (Chesseract, Sudoku, Gutenberg)
        # random sampling of a wider space can find odd-but-good designs the
        # greedy search never visits. Sample K archs (random depth/width/ops from
        # the wide vocab, minus the recurrent ops that crude short-train SGD
        # cannot fit), short-train each, keep the best on val, hand it to the full
        # trainer. No verification — the test is whether a budget-limited random
        # arm beats our best on the datasets where every fixed backbone is stuck.
        if os.environ.get("RAND_SEARCH") == "1":
            import random as _random
            rng = _random.Random(int(os.environ.get("BASELINE_SEED", "42")))
            wide = [o for o in OP_NAMES if o not in ("RowBiLSTM", "CrossAxisBiLSTM")]
            K = int(os.environ.get("RAND_K", "12"))
            rand_budget = self.clock.check() * float(os.environ.get("RAND_BUDGET_FRAC", "0.45"))
            per_cand = rand_budget / max(K, 1)
            rand_deadline = time.perf_counter() + rand_budget
            print(f"  RAND_SEARCH: {K} samples from {len(wide)} ops, "
                  f"budget {rand_budget:.0f}s (~{per_cand:.0f}s/cand), "
                  f"D in [{D_MIN},{D_MAX}] W in {{16,24,32,48}}")
            best_ops, best_w, best_val, pool = None, W_MAX, -1.0, []
            for k in range(K):
                if time.perf_counter() >= rand_deadline:
                    print("  RAND_SEARCH: budget exhausted"); break
                D = rng.randint(D_MIN, D_MAX)
                W = rng.choice([16, 24, 32, 48])
                ops = [rng.choice(wide) for _ in range(D)]
                cd = min(time.perf_counter() + per_cand, rand_deadline)
                try:
                    val, params, _zs, _wc = self._eval_ops(in_ch, nc, h, w, ops, W, device, cd)
                except Exception as ex:
                    print(f"  rand {k:2d} D={D:2d} W={W:2d} FAILED ({type(ex).__name__})"); continue
                pool.append({"phase": "rand", "D": D, "W": W, "ops": list(ops),
                             "val": round(val, 4), "params": params})
                mark = ""
                if val > best_val:
                    best_val, best_ops, best_w, mark = val, ops, W, "  <- best"
                print(f"  rand {k:2d} D={D:2d} W={W:2d} val={val:.4f}{mark}")
            if best_ops is None:
                best_ops, best_w = [MACRO_OP] * 6, W_MAX
                print("  RAND_SEARCH: nothing trained -> default 6-block Conv3")
            try:
                self._log_pool(pool)
            except Exception:
                pass
            print(f"  RAND_SEARCH best: D={len(best_ops)} W={best_w} "
                  f"val={best_val:.4f} ops={best_ops}")
            # used real search budget -> do NOT refund; trainer gets what is left
            return build_cnn(in_ch, nc, h, w, ops=best_ops, base_width=best_w,
                             drop_path_rate=0.1)

        # v6 clock-extend: in exploration mode, give the search a HUGE
        # budget so its internal deadlines never fire — let macro / micro
        # terminate via their natural state machines (depth / width strikes
        # and all-layers-tried). The wall-clock cost is refunded at exit.
        if EXTEND_CLOCK_FOR_NAS:
            search_budget = 6 * 3600.0        # 6 h "soft cap"
        else:
            total_remaining = self.clock.check()
            search_budget = total_remaining * SEARCH_BUDGET_FRACTION
        per_cand = search_budget / max(_EXPECTED_CANDS, 1)
        search_deadline = time.perf_counter() + search_budget
        macro_deadline = time.perf_counter() + search_budget * MACRO_FRACTION
        print(f"  EGNAS v6 full: search budget {search_budget:.0f}s "
              f"(macro {MACRO_FRACTION:.0%} / micro {1 - MACRO_FRACTION:.0%}, "
              f"~{per_cand:.0f}s/cand, D∈[{D_MIN},{D_MAX}] W∈[{W_MIN},{W_MAX}], "
              f"K_max={MAX_SHORT_EPOCHS}, verify={VERIFY_MICRO_ACCEPTS}, "
              f"clock-extend={EXTEND_CLOCK_FOR_NAS})")

        pool: List[Dict] = []
        best_D, best_W, best_val = D_MIN, W_MAX, -1.0

        try:
            # Bootstrap: the smallest arch.
            cd = min(time.perf_counter() + per_cand, macro_deadline)
            best_val, p0, zs0, wc0 = self._eval_arch(in_ch, nc, h, w, D_MIN, W_MAX, device, cd)
            pool.append({"phase": "bootstrap", "D": D_MIN, "W": W_MAX,
                         "op": MACRO_OP, "val": round(best_val, 4), "params": p0,
                         "zeroshot": zs0, "wall_clock_s": round(wc0, 2)})
            print(f"  bootstrap   D={D_MIN:2d} W={W_MAX}  val={best_val:.4f}")

            # PHASE 1 — depth-grow.
            D, strikes = D_MIN, 0
            while D < D_MAX and time.perf_counter() < macro_deadline \
                    and strikes < DEPTH_STRIKES:
                D += 1
                cd = min(time.perf_counter() + per_cand, macro_deadline)
                try:
                    val, params, zs, wc = self._eval_arch(in_ch, nc, h, w, D, W_MAX, device, cd)
                except Exception as ex:
                    print(f"  depth D={D:2d}  FAILED ({type(ex).__name__}: {ex})")
                    strikes += 1
                    continue
                pool.append({"phase": "depth", "D": D, "W": W_MAX,
                             "op": MACRO_OP, "val": round(val, 4), "params": params,
                             "zeroshot": zs, "wall_clock_s": round(wc, 2)})
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
            while W - W_STEP >= W_MIN and time.perf_counter() < macro_deadline \
                    and strikes < WIDTH_STRIKES:
                W -= W_STEP
                cd = min(time.perf_counter() + per_cand, macro_deadline)
                try:
                    val, params, zs, wc = self._eval_arch(in_ch, nc, h, w, best_D, W, device, cd)
                except Exception as ex:
                    print(f"  width W={W:2d}  FAILED ({type(ex).__name__}: {ex})")
                    strikes += 1
                    continue
                pool.append({"phase": "width", "D": best_D, "W": W,
                             "op": MACRO_OP, "val": round(val, 4), "params": params,
                             "zeroshot": zs, "wall_clock_s": round(wc, 2)})
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
            model = build_default_cnn(in_channels=in_ch, num_classes=nc,
                                      input_h=h, input_w=w, base_width=W_MAX,
                                      n_blocks=6, drop_path_rate=0.1)
            self._refund_clock(nas_start)
            return model

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
        cur_val = best_val
        n_accepted = 0
        n_verified_revert = 0
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
                            val, params, zs, wc = self._eval_ops(
                                in_ch, nc, h, w, trial, best_W, device, cd)
                        except Exception as ex:
                            print(f"  micro L{i:2d} {op:8s} FAILED "
                                  f"({type(ex).__name__})")
                            continue
                        entry = {"phase": "micro", "D": best_D,
                                 "W": best_W, "layer": i, "op": op,
                                 "ops": list(trial),
                                 "val": round(val, 4), "params": params,
                                 "incumbent_val_at_trial": round(cur_val, 4),
                                 "wall_clock_s": round(wc, 2),
                                 "zeroshot": zs,
                                 "accepted": False, "verified": None,
                                 "verify_val_acc": None}
                        if val > cur_val + MICRO_MARGIN:
                            # v6: verification-of-accepts — re-train this swap
                            # at 2× the budget. If it doesn't hold up, revert.
                            if VERIFY_MICRO_ACCEPTS:
                                try:
                                    verify_cd = min(
                                        time.perf_counter() + 2 * per_cand,
                                        search_deadline)
                                    K_verify = (VERIFY_K_MULTIPLIER
                                                * self._arch_epochs(len(trial)))
                                    vval, _, _, vwc = self._eval_ops(
                                        in_ch, nc, h, w, trial, best_W, device,
                                        verify_cd, max_epochs_override=K_verify)
                                except Exception as ex:
                                    print(f"  micro L{i:2d} {op:8s} verify "
                                          f"FAILED ({type(ex).__name__}) — "
                                          f"reverting swap conservatively")
                                    vval = cur_val - 1.0  # force revert
                                    vwc = 0.0
                                entry["verify_val_acc"] = round(vval, 4)
                                entry["verify_wall_clock_s"] = round(vwc, 2)
                                if vval > cur_val + MICRO_MARGIN:
                                    print(f"  micro L{i:2d} {cur_ops[i]}→{op}"
                                          f"  val={val:.4f} (verify "
                                          f"{vval:.4f})  ✓ accept+verified")
                                    cur_ops, cur_val = trial, vval
                                    entry["accepted"] = True
                                    entry["verified"] = True
                                    n_accepted += 1
                                else:
                                    print(f"  micro L{i:2d} {cur_ops[i]}→{op}"
                                          f"  val={val:.4f} (verify "
                                          f"{vval:.4f})  ✗ verify-revert")
                                    entry["accepted"] = False
                                    entry["verified"] = False
                                    n_verified_revert += 1
                            else:
                                print(f"  micro L{i:2d} {cur_ops[i]}→{op}  "
                                      f"val={val:.4f}  ✓ accept")
                                cur_ops, cur_val = trial, val
                                entry["accepted"] = True
                                n_accepted += 1
                        else:
                            print(f"  micro L{i:2d} {op:8s} val={val:.4f}  "
                                  f"✗ keep {cur_ops[i]}")
                        pool.append(entry)
            except Exception as ex:
                print(f"  micro FAILED ({type(ex).__name__}: {ex}) — "
                      f"keeping macro winner")

        self._log_pool(pool)
        n_chg = sum(1 for o in cur_ops if o != MACRO_OP)
        print(f"  EGNAS v6 full done: D={best_D} W={best_W} "
              f"micro-changed {n_chg}/{best_D}  val={cur_val:.4f} "
              f"({len(pool)} evaluated, {n_accepted} accepted, "
              f"{n_verified_revert} verify-reverted)  ops={cur_ops}")

        # Build the winner fresh (random init), DropPath on for the full
        # retrain — handed to the unchanged Tier-1 Trainer.
        model = build_cnn(in_ch, nc, h, w, ops=cur_ops,
                          base_width=best_W, drop_path_rate=0.1)
        self._refund_clock(nas_start)
        return model
