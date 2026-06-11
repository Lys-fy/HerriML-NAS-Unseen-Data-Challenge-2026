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

from helpers import ModelEMA, lr_for_epoch


SAFETY_MARGIN_SECONDS = 60.0
MAX_EPOCHS = int(os.environ.get("TRAIN_MAX_EPOCHS", "600"))  # v10: train-to-budget
WARMUP_EPOCHS = 20
PATIENCE = int(os.environ.get("TRAIN_PATIENCE", "999"))  # v10: train-to-budget, no early stop (5 wins, 0 regressions)
LR_SCALE = float(os.environ.get("TRAIN_LR_SCALE", "1.0"))  # GameOfLife fix: <1 dampens the deep-net val oscillation
LR_WARMUP_EPOCHS = 5  # for the LR schedule (not to be confused with patience-warmup)
SEED = int(os.environ.get("BASELINE_SEED", "42"))

LABEL_SMOOTHING = 0.1
EMA_DECAY = 0.999


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

            if use_mixup:
                lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
                # Make λ at least 0.5 so the named "primary" sample dominates;
                # standard convention from the Mixup paper.
                lam = max(lam, 1.0 - lam)
                perm = torch.randperm(xb.size(0), device=self.device)
                xb_mixed = lam * xb + (1.0 - lam) * xb[perm]
                yb_b = yb[perm]
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(xb_mixed)
                loss = lam * criterion(logits, yb) + (1.0 - lam) * criterion(logits, yb_b)
            else:
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = criterion(logits, yb)

            if not torch.isfinite(loss):
                # NaN/inf — let outer loop handle recovery.
                return float("nan")

            loss.backward()
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
