"""Tier-1 Trainer.

Tier 0's clock-aware loop + best-val checkpoint + CSV logging, plus:

  - Label smoothing ε=0.1 baked into the CrossEntropyLoss.
  - Mixup α from metadata['mixup_alpha'] (0 disables). Loss is the standard
    convex combination of CE against the two batches.
  - EMA-of-weights (decay=0.999); predict() uses the EMA model, not live.
  - Manual cosine LR schedule with linear warmup over 5 epochs, projected
    after epoch 0 from the observed wall-clock so the anneal length matches
    the budget actually available.
  - NaN-loss recovery: if the loss goes NaN/inf, restore from best_state,
    halve the base LR, and continue. Cheap insurance against bad inits or
    aggressive Mixup combos.
  - channels_last memory format for ~1.3-2× throughput on Ampere+.
  - Patience early-stop (warmup 20, patience 30) — same as Tier 0.
"""

from __future__ import annotations

import copy
import csv
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers import ModelEMA, lr_for_epoch


SAFETY_MARGIN_SECONDS = 60.0
MAX_EPOCHS = int(os.environ.get("TRAIN_MAX_EPOCHS", "600"))  # v10: train-to-budget
WARMUP_EPOCHS = 20
PATIENCE = int(os.environ.get("TRAIN_PATIENCE", "999"))  # v10: train-to-budget, no early stop (5 wins, 0 regressions)
LR_SCALE = float(os.environ.get("TRAIN_LR_SCALE", "1.0"))  # GameOfLife fix: <1 dampens the deep-net val oscillation
LR_WARMUP_EPOCHS = 5  # for the LR schedule (not to be confused with patience-warmup)
SEED = int(os.environ.get("BASELINE_SEED", "42"))
# Optional early stop once the val metric is effectively perfect: stop
# PERFECT_STOP_S budget-seconds after the val accuracy first reaches
# PERFECT_ACC. Default 0 = DISABLED (submission + main sweep train to budget
# as before); only the new-datasets sweep sets V10_PERFECT_STOP_S.
PERFECT_STOP_S = float(os.environ.get("V10_PERFECT_STOP_S", "0"))
PERFECT_ACC = float(os.environ.get("V10_PERFECT_ACC", "0.9995"))

LABEL_SMOOTHING = 0.1
EMA_DECAY = 0.999

# --------------------------------------------------------------------------
# Optional recipe experiments (second-wave A/B tests). ALL default OFF: with no
# env var set, the training path is byte-for-byte the submission recipe (the RNG
# stream is untouched, so default numerics are identical). Enable one at a time
# to measure its marginal accuracy under the SAME train-to-budget clock.
#   TRAIN_SAM=1          Sharpness-Aware Minimization (2 fwd/bwd per step; the
#                        budget buys fewer-but-flatter-minima epochs).
#   TRAIN_SELF_DISTILL=1 online self-distillation: KL(student || EMA-teacher).
#   TRAIN_TRIVIAL_AUG=1  TrivialAugment-style single-random-op augmentation.
# --------------------------------------------------------------------------
#   TRAIN_ADAMW=1        AdamW (lr 1e-3, wd 0.05) instead of SGD+momentum --
#                        the one optimizer axis never ablated at this scale.
USE_SAM = os.environ.get("TRAIN_SAM") == "1"
SAM_RHO = float(os.environ.get("TRAIN_SAM_RHO", "0.05"))
USE_ADAMW = os.environ.get("TRAIN_ADAMW") == "1"
ADAMW_LR = float(os.environ.get("TRAIN_ADAMW_LR", "1e-3"))
ADAMW_WD = float(os.environ.get("TRAIN_ADAMW_WD", "0.05"))
USE_SELF_DISTILL = os.environ.get("TRAIN_SELF_DISTILL") == "1"
SD_ALPHA = float(os.environ.get("TRAIN_SD_ALPHA", "1.0"))
SD_T = float(os.environ.get("TRAIN_SD_T", "2.0"))
USE_TRIVIAL_AUG = os.environ.get("TRAIN_TRIVIAL_AUG") == "1"
# ablation: drop flip_h from the TrivialAugment op set (the base aug policy
# bans flips on principle; TA won Sudoku +2.9 WITH them -- which is it?)
TA_NO_FLIP = os.environ.get("TRAIN_TA_NO_FLIP") == "1"


def _trivial_augment(xb: torch.Tensor) -> torch.Tensor:
    """TrivialAugment-style: pick ONE op at a random magnitude, apply to the
    batch. Op set is restricted to transforms valid for ARBITRARY C x H x W
    tensors, because the competition inputs are not all natural RGB images
    (12-channel boards, spectrograms, byte grids). Batch-level for GPU speed."""
    ops = ("identity", "flip_h", "translate", "brightness", "contrast",
           "noise", "cutout")
    if TA_NO_FLIP:
        ops = tuple(o for o in ops if o != "flip_h")
    op = ops[np.random.randint(len(ops))]
    if op == "flip_h":
        return torch.flip(xb, dims=[3])
    if op == "translate":
        return torch.roll(xb, shifts=int(np.random.randint(-3, 4)),
                          dims=int(np.random.choice([2, 3])))
    if op == "brightness":
        return xb + float(np.random.uniform(-0.3, 0.3))
    if op == "contrast":
        m = xb.mean(dim=(2, 3), keepdim=True)
        return (xb - m) * float(np.random.uniform(0.6, 1.4)) + m
    if op == "noise":
        return xb + torch.randn_like(xb) * float(np.random.uniform(0.0, 0.1))
    if op == "cutout":
        _, _, H, W = xb.shape
        ch, cw = H // 4, W // 4
        if ch > 0 and cw > 0:
            y0 = int(np.random.randint(0, H - ch + 1))
            x0 = int(np.random.randint(0, W - cw + 1))
            xb = xb.clone(); xb[:, :, y0:y0 + ch, x0:x0 + cw] = 0.0
    return xb


def _open_csv_logger(metadata) -> Tuple[Optional[object], Optional[csv.writer]]:
    log_dir = os.environ.get("BASELINE_LOG_DIR")
    if not log_dir:
        return None, None
    os.makedirs(log_dir, exist_ok=True)
    codename = metadata.get("codename", "unknown")
    path = os.path.join(log_dir, f"{codename}.csv")
    f = open(path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow([
        "epoch", "lr", "train_loss", "val_acc", "val_acc_ema",
        "best_val_acc", "epoch_time_s", "clock_remaining_s",
    ])
    return f, writer


class Trainer:
    def __init__(self, model, device, train_dataloader, valid_dataloader, metadata, clock):
        # channels_last format is free throughput on Ampere/Blackwell for
        # conv-heavy nets.
        model = model.to(device, memory_format=torch.channels_last)
        self.model = model
        self.device = device
        self.train_loader = train_dataloader
        self.valid_loader = valid_dataloader
        self.metadata = metadata
        self.clock = clock
        self.best_state = copy.deepcopy(self.model.state_dict())
        self.best_val_acc = -1.0
        # EMA shadow model — updated each step, used for prediction.
        # Ablation hook: TIER1_EMA_OFF=1 disables EMA entirely (live model only).
        self.use_ema = os.environ.get("TIER1_EMA_OFF") != "1"
        if self.use_ema:
            self.ema = ModelEMA(self.model, decay=EMA_DECAY)
        else:
            self.ema = None
            print("  TIER1_EMA_OFF=1: EMA disabled, live model only")
        # Mixup α from DataProcessor (0 = no mixup).
        self.mixup_alpha = float(metadata.get("mixup_alpha", 0.0))

    # -----------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self, use_ema: bool = False) -> float:
        if use_ema and not self.use_ema:
            return -1.0  # EMA disabled — signal "not available"
        model = self.ema.ema if use_ema else self.model
        model.eval()
        correct = 0
        total = 0
        for xb, yb in self.valid_loader:
            xb = xb.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            yb = yb.to(self.device, non_blocking=True)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.numel()
        return correct / max(total, 1)

    # -----------------------------------------------------------------------

    def _train_one_epoch(self, optimizer, criterion) -> float:
        """Returns mean per-batch loss. Updates EMA each step.

        If self.mixup_alpha > 0, applies Mixup: λ ~ Beta(α, α), mix inputs,
        loss = λ * CE(logits, y_a) + (1-λ) * CE(logits, y_b).
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        use_mixup = self.mixup_alpha > 0.0

        for xb, yb in self.train_loader:
            if self.clock.check() < SAFETY_MARGIN_SECONDS:
                return total_loss / max(n_batches, 1)
            xb = xb.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            yb = yb.to(self.device, non_blocking=True)

            # Optional TrivialAugment (default off — RNG untouched when disabled).
            if USE_TRIVIAL_AUG:
                xb = _trivial_augment(xb)

            if use_mixup:
                lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
                # Make λ at least 0.5 so the named "primary" sample dominates;
                # standard convention from the Mixup paper.
                lam = max(lam, 1.0 - lam)
                perm = torch.randperm(xb.size(0), device=self.device)
                xb_in = lam * xb + (1.0 - lam) * xb[perm]
                yb_b = yb[perm]
            else:
                lam, yb_b, xb_in = 1.0, yb, xb

            # Loss recomputable at the CURRENT weights (SAM needs a 2nd pass).
            def _compute_loss():
                logits = self.model(xb_in)
                if use_mixup:
                    loss = lam * criterion(logits, yb) + (1.0 - lam) * criterion(logits, yb_b)
                else:
                    loss = criterion(logits, yb)
                if USE_SELF_DISTILL and self.use_ema:
                    with torch.no_grad():
                        self.ema.ema.eval()
                        t_logits = self.ema.ema(xb_in)
                    kl = F.kl_div(F.log_softmax(logits / SD_T, dim=1),
                                  F.softmax(t_logits / SD_T, dim=1),
                                  reduction="batchmean") * (SD_T * SD_T)
                    loss = loss + SD_ALPHA * kl
                return loss

            optimizer.zero_grad(set_to_none=True)
            loss = _compute_loss()
            if not torch.isfinite(loss):
                return float("nan")
            loss.backward()

            if USE_SAM:
                # Ascend to w + e(w) (e = rho * g/||g||), recompute grad there,
                # restore w, then let the base optimizer step with the SAM grad.
                with torch.no_grad():
                    grads = [p.grad for p in self.model.parameters() if p.grad is not None]
                    norm = torch.norm(torch.stack([g.norm() for g in grads]))
                    scale = SAM_RHO / (norm + 1e-12)
                    eps = []
                    for p in self.model.parameters():
                        if p.grad is None:
                            continue
                        e = p.grad * scale
                        p.add_(e); eps.append((p, e))
                optimizer.zero_grad(set_to_none=True)
                loss2 = _compute_loss()
                if not torch.isfinite(loss2):
                    with torch.no_grad():
                        for p, e in eps:
                            p.sub_(e)                   # restore before bailing
                    return float("nan")
                loss2.backward()                        # grad at w+e (BEFORE restore)
                with torch.no_grad():
                    for p, e in eps:
                        p.sub_(e)                       # restore w, then step below

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            optimizer.step()
            if self.use_ema:
                self.ema.update(self.model)

            total_loss += float(loss.item())
            n_batches += 1
        return total_loss / max(n_batches, 1)

    # -----------------------------------------------------------------------

    def train(self):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = True

        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        if USE_ADAMW:
            base_lr = ADAMW_LR * LR_SCALE
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=base_lr, weight_decay=ADAMW_WD
            )
        else:
            base_lr = 0.025 * LR_SCALE
            optimizer = torch.optim.SGD(
                self.model.parameters(), lr=base_lr, momentum=0.9, weight_decay=3e-4, nesterov=True
            )

        # Deferred LR-schedule sizing: epoch 0 always runs at the warmup-step-1
        # LR (computed by lr_for_epoch with provisional total). After epoch 0
        # we re-anchor the schedule to the wall-clock budget that's actually
        # available — same trick as Tier 0 but with the warmup phase prepended.
        total_epochs_target = MAX_EPOCHS  # placeholder until epoch 0 measures
        nan_strikes = 0
        best_epoch = 0

        # Set the LR for epoch 0.
        def _set_lr_for(epoch: int):
            lr = lr_for_epoch(epoch, total_epochs_target,
                              warmup_epochs=LR_WARMUP_EPOCHS, base_lr=base_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr
            return lr

        csv_file, csv_writer = _open_csv_logger(self.metadata)
        try:
            epoch = 0
            perfect_remaining = None  # clock-remaining when val first hit PERFECT_ACC
            while epoch < MAX_EPOCHS:
                if self.clock.check() < SAFETY_MARGIN_SECONDS:
                    break
                current_lr = _set_lr_for(epoch)
                epoch_start = time.perf_counter()
                train_loss = self._train_one_epoch(optimizer, criterion)

                # NaN-loss recovery.
                if not np.isfinite(train_loss):
                    nan_strikes += 1
                    print(f"  WARN epoch {epoch}: train_loss=NaN, restoring best_state. "
                          f"strike {nan_strikes}/3.")
                    if nan_strikes >= 3:
                        print(f"  abort: 3 NaN strikes — returning best model so far.")
                        break
                    self.model.load_state_dict(self.best_state)
                    base_lr = base_lr / 5.0
                    # Restart from epoch 0 with smaller LR.
                    epoch = 0
                    continue

                val_acc = self._evaluate(use_ema=False)
                if self.use_ema:
                    val_acc_ema = self._evaluate(use_ema=True)
                    # Track the better of (live, EMA) — EMA usually wins after
                    # the first ~10 epochs, but during warmup the live model is
                    # tracking better.
                    cand_val = max(val_acc, val_acc_ema)
                else:
                    val_acc_ema = val_acc  # for CSV consistency only
                    cand_val = val_acc
                if cand_val > self.best_val_acc:
                    self.best_val_acc = cand_val
                    if self.use_ema and val_acc_ema > val_acc:
                        winning_state = self.ema.state_dict()
                    else:
                        winning_state = self.model.state_dict()
                    self.best_state = copy.deepcopy(winning_state)
                    best_epoch = epoch

                elapsed = time.perf_counter() - epoch_start
                remaining = self.clock.check()

                # After epoch 0, project total_epochs from observed wall time
                # so the warmup+cosine schedule sizes correctly.
                if epoch == 0:
                    remaining_for_train = remaining - SAFETY_MARGIN_SECONDS
                    projected_more = max(1, int(remaining_for_train / max(elapsed, 1e-3)))
                    total_epochs_target = min(MAX_EPOCHS, max(LR_WARMUP_EPOCHS + 2,
                                                              epoch + 1 + projected_more))
                    print(f"  schedule: warmup {LR_WARMUP_EPOCHS} + cosine over "
                          f"{total_epochs_target - LR_WARMUP_EPOCHS} epochs "
                          f"(epoch0_time={elapsed:.1f}s, remaining={remaining:.0f}s)")

                print(f"  epoch {epoch:3d}  loss={train_loss:.4f}  val={val_acc:.4f} "
                      f"ema={val_acc_ema:.4f}  best={self.best_val_acc:.4f}  "
                      f"lr={current_lr:.4f}  epoch_time={elapsed:.1f}s  clock={remaining:.0f}s")

                if csv_writer is not None:
                    csv_writer.writerow([
                        epoch, f"{current_lr:.6f}", f"{train_loss:.6f}",
                        f"{val_acc:.6f}", f"{val_acc_ema:.6f}",
                        f"{self.best_val_acc:.6f}",
                        f"{elapsed:.3f}", f"{remaining:.1f}",
                    ])
                    csv_file.flush()

                if epoch >= WARMUP_EPOCHS and epoch - best_epoch >= PATIENCE:
                    print(f"  early-stop: no val improvement for {PATIENCE} epochs "
                          f"(best at epoch {best_epoch}, val_acc={self.best_val_acc:.4f})")
                    break

                # Optional perfect-accuracy early stop (off unless V10_PERFECT_STOP_S
                # is set). Stop PERFECT_STOP_S budget-seconds after val first hits
                # PERFECT_ACC, to avoid burning the rest of the budget on a solved
                # dataset. Used only by the new-datasets sweep.
                if PERFECT_STOP_S > 0 and cand_val >= PERFECT_ACC:
                    if perfect_remaining is None:
                        perfect_remaining = remaining
                        print(f"  perfect-acc reached (val={cand_val:.4f}) at "
                              f"clock={remaining:.0f}s; will stop after {PERFECT_STOP_S:.0f}s")
                    elif perfect_remaining - remaining >= PERFECT_STOP_S:
                        print(f"  perfect-acc early-stop: {PERFECT_STOP_S:.0f}s elapsed "
                              f"since val hit {PERFECT_ACC:.4f} (best={self.best_val_acc:.4f})")
                        break

                if remaining < elapsed + SAFETY_MARGIN_SECONDS:
                    break

                epoch += 1
        finally:
            if csv_file is not None:
                csv_file.close()

        # Restore the best checkpoint (either live or EMA, whichever scored higher).
        self.model.load_state_dict(self.best_state)
        return self.model

    # -----------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, test_loader) -> List[int]:
        # Use whatever's currently loaded into self.model (this is best_state
        # post-train()). The EMA path is reflected in best_state via the
        # max(live, ema) tracking inside train().
        self.model.eval()
        predictions: List[int] = []
        for batch in test_loader:
            xb = batch[0] if isinstance(batch, (list, tuple)) else batch
            xb = xb.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            logits = self.model(xb)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
        return predictions
