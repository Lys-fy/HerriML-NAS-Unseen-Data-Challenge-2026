"""Tier-0 Trainer: SGD + cosine LR, clock-aware epoch loop, best-val checkpoint.

Stops as soon as clock.check() drops below a safety margin large enough to
cover predict() and grader overhead. Always retains the best-val state seen,
so a forced early stop still returns a usable model.
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


# Reserve this many seconds at the end of training for predict() + overhead.
# A skipped predict() is -10 (see docs/notes/02 - Framework/Failure Modes.md).
SAFETY_MARGIN_SECONDS = 60.0

# Hard ceiling on epochs even if the clock would allow more. Set high enough
# that the clock is the binding constraint on slow datasets and the cap only
# triggers on tiny ones where 200 epochs is already gratuitous overfit.
MAX_EPOCHS = 200

# Early-stop on val plateau. Don't activate before WARMUP epochs (model is
# still finding its feet); after that, stop if no val improvement for PATIENCE
# consecutive epochs. Saves a chunk of wall clock on small/easy datasets.
WARMUP_EPOCHS = 20
PATIENCE = 30

# Fixed seed for reproducible Tier-0 baseline runs. cudnn.benchmark stays True
# so we trade strict bitwise determinism for ~5% throughput — that's the right
# trade for a baseline run.
SEED = 42


def _open_csv_logger(metadata) -> Tuple[Optional[object], Optional[csv.writer]]:
    """Open a per-dataset CSV log when BASELINE_LOG_DIR is set; otherwise no-op.

    File is named <BASELINE_LOG_DIR>/<codename>.csv.
    """
    log_dir = os.environ.get("BASELINE_LOG_DIR")
    if not log_dir:
        return None, None
    os.makedirs(log_dir, exist_ok=True)
    codename = metadata.get("codename", "unknown")
    path = os.path.join(log_dir, f"{codename}.csv")
    f = open(path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow([
        "epoch", "lr", "train_loss", "val_acc", "best_val_acc",
        "epoch_time_s", "clock_remaining_s",
    ])
    return f, writer


class Trainer:
    def __init__(self, model, device, train_dataloader, valid_dataloader, metadata, clock):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_dataloader
        self.valid_loader = valid_dataloader
        self.metadata = metadata
        self.clock = clock
        self.best_state = copy.deepcopy(self.model.state_dict())
        self.best_val_acc = -1.0

    def _evaluate(self) -> float:
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in self.valid_loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                logits = self.model(xb)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.numel()
        return correct / max(total, 1)

    def _train_one_epoch(self, optimizer, criterion) -> float:
        """Run one training epoch. Returns mean per-batch cross-entropy loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for xb, yb in self.train_loader:
            if self.clock.check() < SAFETY_MARGIN_SECONDS:
                return total_loss / max(n_batches, 1)
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = self.model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        return total_loss / max(n_batches, 1)

    def train(self):
        # Deterministic seeds for reproducible Tier-0 baseline runs.
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = True

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.025, momentum=0.9, weight_decay=3e-4, nesterov=True
        )
        # Defer scheduler construction until after epoch 0. We don't know
        # how many epochs the clock will allow until we've measured one,
        # and a fixed T_max means a clock-bounded run never reaches the
        # low-LR fine-tuning phase (CosineAnnealingLR(T_max=100) only
        # anneals 12 % of the way in 12 epochs).
        scheduler = None
        best_epoch = 0

        csv_file, csv_writer = _open_csv_logger(self.metadata)
        try:
            for epoch in range(MAX_EPOCHS):
                if self.clock.check() < SAFETY_MARGIN_SECONDS:
                    break
                epoch_start = time.perf_counter()
                train_loss = self._train_one_epoch(optimizer, criterion)
                if scheduler is not None:
                    scheduler.step()

                val_acc = self._evaluate()
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_state = copy.deepcopy(self.model.state_dict())
                    best_epoch = epoch

                elapsed = time.perf_counter() - epoch_start
                remaining = self.clock.check()
                current_lr = optimizer.param_groups[0]["lr"]

                # After epoch 0, project total epochs and size the cosine schedule.
                if scheduler is None:
                    remaining_for_train = remaining - SAFETY_MARGIN_SECONDS
                    projected_more = max(1, int(remaining_for_train / max(elapsed, 1e-3)))
                    target_epochs = min(MAX_EPOCHS - epoch - 1, projected_more)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=max(target_epochs, 1)
                    )
                    print(f"  scheduler: cosine over next {target_epochs} epochs "
                          f"(epoch_time={elapsed:.1f}s, remaining={remaining:.0f}s)")

                print(f"  epoch {epoch:3d}  loss={train_loss:.4f}  val_acc={val_acc:.4f}  "
                      f"best={self.best_val_acc:.4f}  lr={current_lr:.4f}  "
                      f"epoch_time={elapsed:.1f}s  clock={remaining:.0f}s")

                if csv_writer is not None:
                    csv_writer.writerow([
                        epoch, f"{current_lr:.6f}", f"{train_loss:.6f}",
                        f"{val_acc:.6f}", f"{self.best_val_acc:.6f}",
                        f"{elapsed:.3f}", f"{remaining:.1f}",
                    ])
                    csv_file.flush()

                # Patience-based early stop (after warmup).
                if epoch >= WARMUP_EPOCHS and epoch - best_epoch >= PATIENCE:
                    print(f"  early-stop: no val improvement for {PATIENCE} epochs "
                          f"(best at epoch {best_epoch}, val_acc={self.best_val_acc:.4f})")
                    break

                # If the next epoch wouldn't finish before the safety margin, stop now.
                if remaining < elapsed + SAFETY_MARGIN_SECONDS:
                    break
        finally:
            if csv_file is not None:
                csv_file.close()

        # Restore the best checkpoint before predict().
        self.model.load_state_dict(self.best_state)
        return self.model

    def predict(self, test_loader) -> List[int]:
        self.model.eval()
        predictions: List[int] = []
        with torch.no_grad():
            for batch in test_loader:
                # test_loader yields just inputs (TensorDataset with one tensor
                # returns a 1-tuple, so unpack defensively).
                xb = batch[0] if isinstance(batch, (list, tuple)) else batch
                xb = xb.to(self.device, non_blocking=True)
                logits = self.model(xb)
                predictions.extend(logits.argmax(dim=1).cpu().tolist())
        return predictions
