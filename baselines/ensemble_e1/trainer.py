"""Tier-1e1 Trainer — E1 budget-filling diverse-restart ensemble.

Strict superset of the Tier-1 Trainer. The single-model loop (label
smoothing, EMA, warmup+cosine projected after epoch 0, NaN recovery,
channels_last, patience early-stop) is unchanged and factored into
`_train_member`. On top of it:

  * **Member 0 = the chosen searched arch, trained with Tier-1's EXACT
    floor.** With no leftover clock this is bit-for-bit Tier-1 behaviour and
    Tier-1's score — E1 cannot regress (the design's load-bearing property).
  * **Extra members fill only otherwise-wasted slack.** Most datasets
    early-stop using a fraction of the budget; that slack trains additional
    *diverse* members: seed restarts of the chosen arch (the axis Step-0 /
    Q8 showed dominates AddNIST/MultNIST variance) interleaved with the
    decorrelated searched candidate archs handed over by NAS via `model._e1`.
  * **predict() averages softmax across members** (never select-by-val — no
    test labels → winner's curse). Peak VRAM stays at one model (members are
    held as CPU state-dicts, rebuilt on GPU one at a time). A hard clock
    guard drops the remaining members rather than ever overrunning — fewer
    members beats a -10 timeout ([[Reliability vs Peak]]).

Ablation hooks: `E1_OFF=1` → single member (≡ Tier 1). `E1_MAX_MEMBERS`
caps the ensemble (default 6). `TIER1_EMA_OFF=1` honoured as in Tier 1.
"""

from __future__ import annotations

import csv
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers import ModelEMA, build_cnn, lr_for_epoch


SAFETY_MARGIN_SECONDS = 60.0
MAX_EPOCHS = 200
WARMUP_EPOCHS = 20
PATIENCE = 30
LR_WARMUP_EPOCHS = 5
SEED = int(os.environ.get("BASELINE_SEED", "42"))

LABEL_SMOOTHING = 0.1
EMA_DECAY = 0.999

# E1 knobs.
E1_OFF = os.environ.get("E1_OFF") == "1"
E1_MAX_MEMBERS = int(os.environ.get("E1_MAX_MEMBERS", "6"))
# Multiply measured val-eval time to estimate one test-set forward pass for
# the predict reserve (test set is usually larger than val + build overhead).
# Deliberately generous — the -10 timeout cliff is asymmetric vs one fewer
# member ([[Reliability vs Peak]]).
E1_PREDICT_PASS_FACTOR = 4.0
# Don't start a new member unless at least this much TRAINING time remains
# after reserving predict (a member with <a few epochs adds noise).
E1_MIN_MEMBER_TRAIN_SECONDS = 90.0


def _csv_logger(metadata, suffix: str):
    log_dir = os.environ.get("BASELINE_LOG_DIR")
    if not log_dir:
        return None, None
    os.makedirs(log_dir, exist_ok=True)
    codename = metadata.get("codename", "unknown")
    f = open(os.path.join(log_dir, f"{codename}{suffix}.csv"), "w", newline="")
    w = csv.writer(f)
    w.writerow(["epoch", "lr", "train_loss", "val_acc", "val_acc_ema",
                "best_val_acc", "epoch_time_s", "clock_remaining_s"])
    return f, w


def _cpu_state(module: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


class Trainer:
    def __init__(self, model, device, train_dataloader, valid_dataloader, metadata, clock):
        self.device = device
        self.train_loader = train_dataloader
        self.valid_loader = valid_dataloader
        self.metadata = metadata
        self.clock = clock

        self.member0 = model.to(device, memory_format=torch.channels_last)
        self.use_ema = os.environ.get("TIER1_EMA_OFF") != "1"
        self.mixup_alpha = float(metadata.get("mixup_alpha", 0.0))

        # E1 rebuild recipe attached by NAS. None only on a legacy/contract
        # fallback → we run a single member (≡ Tier 1) via _fallback_model.
        self._recipe: Optional[Dict] = getattr(model, "_e1", None)

        # Each member: {"ops": list[str], "state": cpu state_dict,
        #               "val": float, "tag": str}
        self._members: List[Dict] = []
        self._fallback_model: Optional[nn.Module] = None  # used iff recipe None
        self._val_eval_s: float = 0.0  # one valid-pass time, measured on m0

    # ------------------------------------------------------------------ #
    # Per-member training — Tier-1's single-model loop, parameterised.
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _evaluate(self, model, ema, use_ema: bool) -> float:
        if use_ema and ema is None:
            return -1.0
        m = ema.ema if use_ema else model
        m.eval()
        correct = total = 0
        for xb, yb in self.valid_loader:
            xb = xb.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            yb = yb.to(self.device, non_blocking=True)
            correct += (m(xb).argmax(dim=1) == yb).sum().item()
            total += yb.numel()
        return correct / max(total, 1)

    def _run_epoch(self, model, ema, optimizer, criterion) -> float:
        model.train()
        total_loss = 0.0
        n = 0
        use_mixup = self.mixup_alpha > 0.0
        for xb, yb in self.train_loader:
            if self.clock.check() < SAFETY_MARGIN_SECONDS:
                return total_loss / max(n, 1)
            xb = xb.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            yb = yb.to(self.device, non_blocking=True)
            if use_mixup:
                lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
                lam = max(lam, 1.0 - lam)
                perm = torch.randperm(xb.size(0), device=self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(lam * xb + (1.0 - lam) * xb[perm])
                loss = lam * criterion(logits, yb) + (1.0 - lam) * criterion(logits, yb[perm])
            else:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xb), yb)
            if not torch.isfinite(loss):
                return float("nan")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            total_loss += float(loss.item())
            n += 1
        return total_loss / max(n, 1)

    def _train_member(self, model, seed: int, tag: str, clock_floor: float,
                      soft_deadline: Optional[float]) -> Tuple[dict, float]:
        """Train one member. Stops on patience, 3 NaN strikes, soft_deadline
        (perf_counter cap so an extra member can't eat all the slack), or
        clock.check() < clock_floor. Returns (best_state_cpu, best_val)."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        base_lr = 0.025
        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9,
                                    weight_decay=3e-4, nesterov=True)
        ema = ModelEMA(model, decay=EMA_DECAY) if self.use_ema else None

        total_epochs_target = MAX_EPOCHS
        nan_strikes = 0
        best_epoch = 0
        best_val = -1.0
        best_state = _cpu_state(model)

        def set_lr(ep):
            lr = lr_for_epoch(ep, total_epochs_target,
                              warmup_epochs=LR_WARMUP_EPOCHS, base_lr=base_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr
            return lr

        f, w = _csv_logger(self.metadata, "" if tag == "m0" else f".{tag}")
        try:
            epoch = 0
            while epoch < MAX_EPOCHS:
                if self.clock.check() < clock_floor:
                    break
                if soft_deadline is not None and time.perf_counter() > soft_deadline:
                    break
                lr = set_lr(epoch)
                t0 = time.perf_counter()
                train_loss = self._run_epoch(model, ema, optimizer, criterion)

                if not np.isfinite(train_loss):
                    nan_strikes += 1
                    if nan_strikes >= 3:
                        break
                    model.load_state_dict(
                        {k: v.to(self.device) for k, v in best_state.items()})
                    base_lr /= 5.0
                    epoch = 0
                    continue

                val = self._evaluate(model, ema, use_ema=False)
                tv = time.perf_counter()
                if self.use_ema:
                    val_ema = self._evaluate(model, ema, use_ema=True)
                    cand = max(val, val_ema)
                else:
                    val_ema = val
                    cand = val
                if tag == "m0" and self._val_eval_s == 0.0:
                    self._val_eval_s = max(time.perf_counter() - tv, 1e-3)

                if cand > best_val:
                    best_val = cand
                    src = ema.ema if (self.use_ema and val_ema > val) else model
                    best_state = _cpu_state(src)
                    best_epoch = epoch

                dt = time.perf_counter() - t0
                rem = self.clock.check()
                if epoch == 0:
                    avail = rem - clock_floor
                    more = max(1, int(avail / max(dt, 1e-3)))
                    total_epochs_target = min(
                        MAX_EPOCHS, max(LR_WARMUP_EPOCHS + 2, epoch + 1 + more))

                print(f"  [{tag}] epoch {epoch:3d}  loss={train_loss:.4f}  "
                      f"val={val:.4f} ema={val_ema:.4f}  best={best_val:.4f}  "
                      f"lr={lr:.4f}  dt={dt:.1f}s  clock={rem:.0f}s")
                if w is not None:
                    w.writerow([epoch, f"{lr:.6f}", f"{train_loss:.6f}",
                                f"{val:.6f}", f"{val_ema:.6f}", f"{best_val:.6f}",
                                f"{dt:.3f}", f"{rem:.1f}"])
                    f.flush()

                if epoch >= WARMUP_EPOCHS and epoch - best_epoch >= PATIENCE:
                    print(f"  [{tag}] early-stop (best ep {best_epoch}, "
                          f"val={best_val:.4f})")
                    break
                if rem < dt + clock_floor:
                    break
                epoch += 1
        finally:
            if f is not None:
                f.close()
        return best_state, best_val

    # ------------------------------------------------------------------ #

    def _predict_reserve(self, n_members: int) -> float:
        per_pass = max(self._val_eval_s, 1.0) * E1_PREDICT_PASS_FACTOR
        return n_members * per_pass + SAFETY_MARGIN_SECONDS

    def _build(self, ops) -> nn.Module:
        r = self._recipe
        return build_cnn(r["in_channels"], r["num_classes"], r["h"], r["w"],
                         ops=list(ops), base_width=r["base_width"],
                         drop_path_rate=r["drop_path_rate"]
                         ).to(self.device, memory_format=torch.channels_last)

    def _member_plan(self) -> List[Tuple[str, list, int]]:
        """(tag, ops, seed) for members AFTER member 0. Restarts of the
        chosen arch first (the dominant variance axis per Q8), interleaved
        with the decorrelated searched candidate archs."""
        chosen = self._recipe["chosen_ops"]
        others = self._recipe.get("ranked_other_ops", [])
        plan: List[Tuple[str, list, int]] = []
        oi = k = 0
        while len(plan) < E1_MAX_MEMBERS - 1:
            k += 1
            plan.append((f"r{k}", list(chosen), SEED + k))
            if oi < len(others) and len(plan) < E1_MAX_MEMBERS - 1:
                plan.append((f"a{oi + 1}", list(others[oi]), SEED + 100 + oi))
                oi += 1
            if k > E1_MAX_MEMBERS + len(others):  # hard safety bound
                break
        return plan

    def train(self):
        # --- Member 0: Tier-1 behaviour exactly (floor = SAFETY only). ----
        s0, v0 = self._train_member(self.member0, SEED, "m0",
                                    clock_floor=SAFETY_MARGIN_SECONDS,
                                    soft_deadline=None)

        if E1_OFF or self._recipe is None:
            # Pure Tier-1: keep the trained member0 object for predict (we
            # cannot rebuild it without the recipe).
            self.member0.load_state_dict(
                {k: v.to(self.device) for k, v in s0.items()})
            self._fallback_model = self.member0
            print(f"  E1: single member (E1_OFF={E1_OFF}, "
                  f"recipe={'yes' if self._recipe else 'none'}) — ≡ Tier 1")
            return

        # Recipe present → member0 is rebuildable from chosen_ops; free it.
        self._members.append({"ops": list(self._recipe["chosen_ops"]),
                              "state": s0, "val": v0, "tag": "m0"})
        self.member0.to("cpu")
        del self.member0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Extra members from leftover slack only. ----------------------
        plan = self._member_plan()
        print(f"  E1: member0 val={v0:.4f}; up to {len(plan)} extra members "
              f"queued; val_eval≈{self._val_eval_s:.2f}s")
        for tag, ops, seed in plan:
            reserve = self._predict_reserve(len(self._members) + 1)
            rem = self.clock.check()
            if rem < reserve + E1_MIN_MEMBER_TRAIN_SECONDS:
                print(f"  E1: stop adding members (clock {rem:.0f}s < reserve "
                      f"{reserve:.0f}s + min-train {E1_MIN_MEMBER_TRAIN_SECONDS:.0f}s)")
                break
            soft = time.perf_counter() + (rem - reserve)  # cap so it can't hog slack
            model = None
            try:
                model = self._build(ops)
                st, val = self._train_member(model, seed, tag,
                                             clock_floor=reserve, soft_deadline=soft)
                self._members.append({"ops": list(ops), "state": st,
                                      "val": val, "tag": tag})
                print(f"  E1: +member {tag} ({'/'.join(ops)}) val={val:.4f} "
                      f"[{len(self._members)} total]")
            except Exception as ex:  # an extra member must never -10 us
                print(f"  E1: member {tag} FAILED "
                      f"({type(ex).__name__}: {ex}) — skipped")
            finally:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        vs = ", ".join(f"{m['tag']}={m['val']:.4f}" for m in self._members)
        print(f"  E1: ensemble of {len(self._members)} members [{vs}]")
        return

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _forward_probs(self, model, test_loader) -> torch.Tensor:
        model.eval()
        parts = []
        for batch in test_loader:
            xb = batch[0] if isinstance(batch, (list, tuple)) else batch
            xb = xb.to(self.device, non_blocking=True,
                       memory_format=torch.channels_last)
            parts.append(F.softmax(model(xb), dim=1).float().cpu())
        return torch.cat(parts, dim=0)

    @torch.no_grad()
    def predict(self, test_loader) -> List[int]:
        # Contract fallback: no recipe → single trained model, ≡ Tier 1.
        if self._fallback_model is not None:
            probs = self._forward_probs(self._fallback_model, test_loader)
            return probs.argmax(dim=1).tolist()

        prob_sum: Optional[torch.Tensor] = None
        used = 0
        for m in self._members:
            # Member 0 always runs; afterwards bail before any pass we can't
            # afford rather than risk the -10 timeout.
            if used >= 1 and self.clock.check() < SAFETY_MARGIN_SECONDS:
                print(f"  E1 predict: clock low — stopping at {used} members")
                break
            model = None
            try:
                model = self._build(m["ops"])
                model.load_state_dict(
                    {k: v.to(self.device) for k, v in m["state"].items()})
                probs = self._forward_probs(model, test_loader)
                if prob_sum is None:
                    prob_sum = probs
                elif probs.shape == prob_sum.shape:
                    prob_sum = prob_sum + probs
                else:
                    print(f"  E1 predict: {m['tag']} shape "
                          f"{tuple(probs.shape)} ≠ {tuple(prob_sum.shape)} — skip")
                    continue
                used += 1
            except Exception as ex:
                print(f"  E1 predict: member {m['tag']} failed "
                      f"({type(ex).__name__}: {ex}) — skipped")
                continue
            finally:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if prob_sum is None:
            print("  E1 predict: no member produced output — degenerate")
            return []
        print(f"  E1 predict: averaged {used}/{len(self._members)} members")
        return prob_sum.argmax(dim=1).tolist()
