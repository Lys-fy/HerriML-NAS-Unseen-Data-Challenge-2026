"""v12 Trainer -- the budget-adaptive controller.

v11.1's recipe (SGD warmup+cosine train-to-budget, EMA, label smoothing, NaN
recovery, channels_last) wrapped in a THROUGHPUT-KEYED plan that adapts every
component to the measured speed of THIS model on THIS data on THIS hardware.
Nothing is keyed on wall-clock hours or GPU names -- only on what a short
calibration actually measures, re-projected between stages, so a B300 at 2h
lands on the same rung a slower GPU needs 8h for and slow datasets demote
themselves mid-run instead of overrunning.

The evidence base is the v12 screening campaign (see the vault note
"Pipeline v12 - Screening Report"; runner dev/v12_integration.py):
  - TrivialAugment: +2.3 mean over 9 datasets, biggest single win -> DEFAULT ON
    (flip stays: removing it collapses GameOfLife 88->62).
  - K-seed logit-ensemble: +1.5 mean, 8/9 -> the surplus-budget conversion, but
    ONLY when each seed still gets a full training (at 2h total, 3-way splitting
    lost CIFARTile -5.6) -> ADAPTIVE K from measured throughput.
  - Uncapped longer training: CIFARTile plain-6h +2.8 over v10 (it was
    undertrained at 600 epochs) -> budget-aware epoch target up to CAP_EPOCHS.
  - flip-TTA: helps flip-tolerant grids, wrecks text -> never deployed blind,
    only via VAL-GATED SELECTION over {each seed, ensemble} x {no-TTA, TTA}
    (measure-then-decide; the proxy wall forbids predicting it).

Pipeline inside train():
  1. calibrate (~few batches): measured step time -> projected epoch time.
  2. plan: proj_epochs = usable_clock / epoch_time;
       K = 1                 while proj_epochs < 1.5 x FULL_EPOCHS (a split
                             would cut seeds below one full training)
       K = min(MAX_SEEDS, proj_epochs // FULL_EPOCHS) otherwise;
     per-seed schedule target = min(CAP_EPOCHS, proj_epochs / K).
  3. train K seeds from the SAME init (the validated shared-init regime),
     re-planning after each seed from its MEASURED duration (graceful
     demotion: a slow seed shrinks or drops the ones after it). A seed that
     reaches ~perfect val twice in a row stops early and donates its slice.
  4. val-gated selection, each extra val pass clock-guarded.
predict() applies the chosen config, degrading to single-best if the clock
would not fit the ensemble/TTA passes (predictions must always complete).

All legacy env toggles (TRAIN_SAM / TRAIN_SELF_DISTILL / TRAIN_ADAMW /
TIER1_EMA_OFF, default off) are preserved for ablations; V12_* knobs override
the adaptive defaults.
"""

from __future__ import annotations

import copy
import csv
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers import ModelEMA, lr_for_epoch


SAFETY_MARGIN_SECONDS = 60.0
# one "full training" = the schedule length every screening verdict was
# measured at; CAP is the validated ceiling for surplus single-seed budget.
FULL_EPOCHS = int(os.environ.get("V12_FULL_EPOCHS", "600"))
CAP_EPOCHS = int(os.environ.get("V12_CAP_EPOCHS", "2400"))
MAX_SEEDS = int(os.environ.get("V12_MAX_SEEDS", "3"))
SELECT_OFF = os.environ.get("V12_SELECT_OFF") == "1"     # ablation: skip selection
WARMUP_EPOCHS = 20
PATIENCE = int(os.environ.get("TRAIN_PATIENCE", "999"))
LR_SCALE = float(os.environ.get("TRAIN_LR_SCALE", "1.0"))
LR_WARMUP_EPOCHS = 5
SEED = int(os.environ.get("BASELINE_SEED", "42"))

LABEL_SMOOTHING = 0.1
EMA_DECAY = 0.999
# perfect-seed early stop: a seed stops after val >= PERFECT_ACC on two
# consecutive epochs, releasing its remaining slice to later seeds/selection.
PERFECT_ACC = float(os.environ.get("V12_PERFECT_ACC", "0.9995"))

# --------------------------------------------------------------------------
# Recipe toggles. TrivialAugment is ON by default in v12 (screen: +2.34 mean,
# 8/9 non-negative; flip kept -- the ablation showed it carries GameOfLife).
# The rejected-but-kept toggles (SAM / self-distill / AdamW) stay for future
# val-gated arms; all default OFF.
# --------------------------------------------------------------------------
USE_TRIVIAL_AUG = os.environ.get("TRAIN_TRIVIAL_AUG", "1") == "1"
TA_NO_FLIP = os.environ.get("TRAIN_TA_NO_FLIP") == "1"
USE_SAM = os.environ.get("TRAIN_SAM") == "1"
SAM_RHO = float(os.environ.get("TRAIN_SAM_RHO", "0.05"))
USE_SELF_DISTILL = os.environ.get("TRAIN_SELF_DISTILL") == "1"
SD_ALPHA = float(os.environ.get("TRAIN_SD_ALPHA", "1.0"))
SD_T = float(os.environ.get("TRAIN_SD_T", "2.0"))
USE_ADAMW = os.environ.get("TRAIN_ADAMW") == "1"
ADAMW_LR = float(os.environ.get("TRAIN_ADAMW_LR", "1e-3"))
ADAMW_WD = float(os.environ.get("TRAIN_ADAMW_WD", "0.05"))


def _trivial_augment(xb: torch.Tensor) -> torch.Tensor:
    """TrivialAugment-style: ONE op at a random magnitude per batch. Op set is
    restricted to transforms valid for ARBITRARY C x H x W tensors (the inputs
    are not all natural RGB: 12-channel boards, spectrograms, byte grids)."""
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
        "best_val_acc", "epoch_time_s", "clock_remaining_s", "seed",
    ])
    return f, writer


class Trainer:
    def __init__(self, model, device, train_dataloader, valid_dataloader, metadata, clock):
        model = model.to(device, memory_format=torch.channels_last)
        self.model = model
        self.device = device
        self.train_loader = train_dataloader
        self.valid_loader = valid_dataloader
        self.metadata = metadata
        self.clock = clock
        self.mixup_alpha = float(metadata.get("mixup_alpha", 0.0))
        self.use_ema = os.environ.get("TIER1_EMA_OFF") != "1"
        # SHARED init for all seeds -- the regime where the K-seed ensemble was
        # validated (independent inits collapse weight-space combos and add
        # nothing to logit ensembles at our scale).
        self.init_state = copy.deepcopy(model.state_dict())
        # controller outputs
        self.seed_states: List[Dict[str, torch.Tensor]] = []
        self.seed_vals: List[float] = []
        self.chosen: Tuple[str, Optional[int], bool] = ("single", 0, False)
        self.val_scores: Dict[str, float] = {}
        self._pass_s = 5.0            # measured duration of one val pass
        self.best_val_acc = -1.0      # API compat (max over seeds)

    # ------------------------------------------------------------- calibration

    def _calibrate_epoch_time(self) -> float:
        """Measure real step time on THIS hardware/data -> projected seconds per
        epoch (train pass + val pass estimate). Uses a handful of real steps on
        the actual model, then restores the init state so seed 0 is unaffected."""
        self.model.train()
        opt = torch.optim.SGD(self.model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        times, steps = [], 0
        for xb, yb in self.train_loader:
            if steps >= 4 or self.clock.check() < SAFETY_MARGIN_SECONDS:
                break
            xb = xb.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            yb = yb.to(self.device, non_blocking=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            loss = criterion(self.model(xb), yb)
            loss.backward()
            opt.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            steps += 1
        self.model.load_state_dict(self.init_state)   # undo calibration steps
        step_t = min(times[1:]) if len(times) > 1 else (times[0] if times else 0.1)
        n_train = max(1, len(self.train_loader))
        n_val = max(1, len(self.valid_loader))
        # val pass ~ forward-only ~ 1/3 of a train step; 1.1 fudge for logging etc.
        return 1.1 * (n_train * step_t + n_val * step_t / 3.0)

    def _plan(self, remaining_s: float, epoch_s: float) -> Tuple[int, int]:
        """Throughput-keyed rung selection: how many seeds, how long a schedule.
        K > 1 only when every seed still gets >= one FULL training (the 2h
        CIFARTile lesson); surplus beyond MAX_SEEDS full trainings goes into
        LONGER schedules (the uncapped-training lesson), up to CAP_EPOCHS."""
        proj = max(1.0, remaining_s / max(epoch_s, 1e-3))
        if proj < 1.5 * FULL_EPOCHS:
            return 1, int(min(proj, CAP_EPOCHS))
        k = int(min(MAX_SEEDS, proj // FULL_EPOCHS))
        return k, int(min(CAP_EPOCHS, proj / k))

    # ---------------------------------------------------------- one seed train

    def _train_one_seed(self, seed_idx: int, deadline: float, target_epochs: int,
                        csv_writer, csv_file) -> Tuple[Dict[str, torch.Tensor], float]:
        torch.manual_seed(SEED + seed_idx)
        np.random.seed(SEED + seed_idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED + seed_idx)

        self.model.load_state_dict(self.init_state)
        ema = ModelEMA(self.model, decay=EMA_DECAY) if self.use_ema else None
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        if USE_ADAMW:
            base_lr = ADAMW_LR * LR_SCALE
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=base_lr,
                                          weight_decay=ADAMW_WD)
        else:
            base_lr = 0.025 * LR_SCALE
            optimizer = torch.optim.SGD(self.model.parameters(), lr=base_lr,
                                        momentum=0.9, weight_decay=3e-4, nesterov=True)

        def remaining():
            return min(deadline - time.perf_counter(), self.clock.check())

        def set_lr(epoch, total):
            lr = lr_for_epoch(epoch, total, warmup_epochs=LR_WARMUP_EPOCHS, base_lr=base_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr
            return lr

        best_val, best_state, best_epoch = -1.0, copy.deepcopy(self.model.state_dict()), 0
        total_epochs = target_epochs
        nan_strikes = 0
        perfect_streak = 0
        epoch = 0
        while epoch < total_epochs:
            if remaining() < SAFETY_MARGIN_SECONDS:
                break
            lr = set_lr(epoch, total_epochs)
            t0 = time.perf_counter()
            train_loss = self._train_one_epoch(optimizer, criterion, ema, deadline)

            if not np.isfinite(train_loss):
                nan_strikes += 1
                print(f"  WARN seed {seed_idx} epoch {epoch}: NaN, restore best. "
                      f"strike {nan_strikes}/3.")
                if nan_strikes >= 3:
                    break
                self.model.load_state_dict(best_state)
                base_lr = base_lr / 5.0
                epoch = 0
                continue

            val = self._evaluate(self.model)
            val_ema = self._evaluate(ema.ema) if ema is not None else val
            cand = max(val, val_ema)
            if cand > best_val:
                best_val = cand
                src = ema.state_dict() if (ema is not None and val_ema > val) else self.model.state_dict()
                best_state = copy.deepcopy(src)
                best_epoch = epoch

            elapsed = time.perf_counter() - t0
            if epoch == 0:
                # re-anchor the schedule to what THIS seed's slice actually
                # affords (never above the planned target -> anneal completes)
                afford = max(LR_WARMUP_EPOCHS + 2,
                             int((remaining() - SAFETY_MARGIN_SECONDS) / max(elapsed, 1e-3)))
                total_epochs = min(target_epochs, afford)
                print(f"  seed {seed_idx}: schedule {total_epochs} epochs "
                      f"(epoch0 {elapsed:.1f}s, slice {remaining():.0f}s)")

            print(f"  s{seed_idx} e{epoch:3d} loss={train_loss:.4f} val={val:.4f} "
                  f"ema={val_ema:.4f} best={best_val:.4f} lr={lr:.4f} "
                  f"t={elapsed:.1f}s clock={self.clock.check():.0f}s", flush=True)
            if csv_writer is not None:
                csv_writer.writerow([epoch, f"{lr:.6f}", f"{train_loss:.6f}",
                                     f"{val:.6f}", f"{val_ema:.6f}", f"{best_val:.6f}",
                                     f"{elapsed:.3f}", f"{self.clock.check():.1f}", seed_idx])
                csv_file.flush()

            perfect_streak = perfect_streak + 1 if cand >= PERFECT_ACC else 0
            if perfect_streak >= 2:
                print(f"  seed {seed_idx}: ~perfect val twice, releasing slice")
                break
            if epoch >= WARMUP_EPOCHS and epoch - best_epoch >= PATIENCE:
                break
            if remaining() < elapsed + SAFETY_MARGIN_SECONDS:
                break
            epoch += 1

        cpu_state = {k: v.detach().cpu().clone() for k, v in best_state.items()}
        return cpu_state, best_val

    def _train_one_epoch(self, optimizer, criterion, ema, deadline) -> float:
        self.model.train()
        total_loss, n_batches = 0.0, 0
        use_mixup = self.mixup_alpha > 0.0
        for xb, yb in self.train_loader:
            if (min(deadline - time.perf_counter(), self.clock.check())
                    < SAFETY_MARGIN_SECONDS):
                return total_loss / max(n_batches, 1)
            xb = xb.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            yb = yb.to(self.device, non_blocking=True)
            if USE_TRIVIAL_AUG:
                xb = _trivial_augment(xb)
            if use_mixup:
                lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
                lam = max(lam, 1.0 - lam)
                perm = torch.randperm(xb.size(0), device=self.device)
                xb_in, yb_b = lam * xb + (1.0 - lam) * xb[perm], yb[perm]
            else:
                lam, yb_b, xb_in = 1.0, yb, xb

            def compute_loss():
                logits = self.model(xb_in)
                if use_mixup:
                    loss = lam * criterion(logits, yb) + (1.0 - lam) * criterion(logits, yb_b)
                else:
                    loss = criterion(logits, yb)
                if USE_SELF_DISTILL and ema is not None:
                    with torch.no_grad():
                        ema.ema.eval()
                        t_logits = ema.ema(xb_in)
                    kl = F.kl_div(F.log_softmax(logits / SD_T, dim=1),
                                  F.softmax(t_logits / SD_T, dim=1),
                                  reduction="batchmean") * (SD_T * SD_T)
                    loss = loss + SD_ALPHA * kl
                return loss

            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss()
            if not torch.isfinite(loss):
                return float("nan")
            loss.backward()

            if USE_SAM:
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
                loss2 = compute_loss()
                if not torch.isfinite(loss2):
                    with torch.no_grad():
                        for p, e in eps:
                            p.sub_(e)
                    return float("nan")
                loss2.backward()
                with torch.no_grad():
                    for p, e in eps:
                        p.sub_(e)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            optimizer.step()
            if ema is not None:
                ema.update(self.model)
            total_loss += float(loss.item())
            n_batches += 1
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _evaluate(self, model) -> float:
        model.eval()
        correct = total = 0
        for xb, yb in self.valid_loader:
            xb = xb.to(self.device, non_blocking=True, memory_format=torch.channels_last)
            yb = yb.to(self.device, non_blocking=True)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.numel()
        return correct / max(total, 1)

    # -------------------------------------------------------------- controller

    def train(self):
        torch.backends.cudnn.benchmark = True
        csv_file, csv_writer = _open_csv_logger(self.metadata)
        try:
            epoch_s = self._calibrate_epoch_time()
            usable = self.clock.check() - 2 * SAFETY_MARGIN_SECONDS
            k, target = self._plan(usable, epoch_s)
            print(f"  v12 plan: ~{epoch_s:.1f}s/epoch -> {usable/max(epoch_s,1e-3):.0f} "
                  f"projected epochs -> K={k} seeds x {target}-epoch schedule "
                  f"(TA={'on' if USE_TRIVIAL_AUG else 'off'})", flush=True)

            i = 0
            while i < k:
                remaining = self.clock.check() - SAFETY_MARGIN_SECONDS
                seeds_left = k - i
                slice_s = remaining / seeds_left
                if i > 0:
                    # re-plan from the MEASURED seed duration: drop seeds that
                    # no longer fit a meaningful training
                    if remaining < 0.35 * self._last_seed_s:
                        print(f"  v12 re-plan: {remaining:.0f}s left "
                              f"< 0.35 x last seed -> stop at {i} seed(s)")
                        break
                    target = int(min(target, max(LR_WARMUP_EPOCHS + 2,
                                                 slice_s / max(self._last_epoch_s, 1e-3))))
                t0 = time.perf_counter()
                deadline = t0 + slice_s
                state, val = self._train_one_seed(i, deadline, target, csv_writer, csv_file)
                self._last_seed_s = time.perf_counter() - t0
                self._last_epoch_s = max(self._last_seed_s / max(target, 1), 1e-3)
                self.seed_states.append(state)
                self.seed_vals.append(val)
                print(f"  seed {i} done: best_val={val:.4f} "
                      f"({self._last_seed_s/60:.1f}min)", flush=True)
                i += 1
        finally:
            if csv_file is not None:
                csv_file.close()

        if not self.seed_states:   # pathological tiny budget: keep init
            self.seed_states = [{k2: v.detach().cpu().clone()
                                 for k2, v in self.model.state_dict().items()}]
            self.seed_vals = [0.0]
        self.best_val_acc = max(self.seed_vals)
        self._select()
        # load the chosen (or best single) state for API compat
        kind, idx, _ = self.chosen
        self.model.load_state_dict(
            self.seed_states[idx if kind == "single" else int(np.argmax(self.seed_vals))])
        return self.model

    # --------------------------------------------------------------- selection

    @torch.no_grad()
    def _probs(self, state, loader, tta: bool) -> torch.Tensor:
        self.model.load_state_dict(state)
        model = self.model.to(self.device, memory_format=torch.channels_last)
        model.eval()
        outs = []
        for batch in loader:
            xb = batch[0] if isinstance(batch, (list, tuple)) else batch
            xb = xb.to(self.device, memory_format=torch.channels_last)
            p = model(xb).softmax(1)
            if tta:
                p = 0.5 * (p + model(torch.flip(xb, dims=[3])).softmax(1))
            outs.append(p.cpu())
        return torch.cat(outs)

    def _select(self):
        """Val-gated final selection: enumerate {each seed, logit-ensemble} x
        {no-TTA, flip-TTA}, score on validation, keep the winner. Every extra
        val pass is clock-guarded; on any shortfall we fall back to the
        best-val single seed (which costs nothing extra)."""
        best_i = int(np.argmax(self.seed_vals))
        self.chosen = ("single", best_i, False)
        if SELECT_OFF:
            return
        yv = torch.cat([yb for _, yb in self.valid_loader])
        n_seeds = len(self.seed_states)

        def acc(p):
            n = min(len(p), len(yv))
            return round(100.0 * float((p[:n].argmax(1) == yv[:n]).float().mean()), 3)

        val_p: Dict[Tuple[int, bool], torch.Tensor] = {}
        t0 = time.perf_counter()
        val_p[(best_i, False)] = self._probs(self.seed_states[best_i], self.valid_loader, False)
        self._pass_s = max(time.perf_counter() - t0, 0.2)
        # each further pass only if the clock affords it (+ margin)
        for tta in (False, True):
            for i in range(n_seeds):
                if (i, tta) in val_p:
                    continue
                if self.clock.check() < SAFETY_MARGIN_SECONDS + 2.5 * self._pass_s:
                    print("  v12 select: clock short, stopping extra val passes")
                    break
                val_p[(i, tta)] = self._probs(self.seed_states[i], self.valid_loader, tta)

        configs: Dict[str, Tuple[str, Optional[int], bool]] = {}
        scores: Dict[str, float] = {}
        for tta in (False, True):
            suf = "_tta" if tta else ""
            have = [i for i in range(n_seeds) if (i, tta) in val_p]
            for i in have:
                name = f"single{i}{suf}"
                configs[name] = ("single", i, tta)
                scores[name] = acc(val_p[(i, tta)])
            if len(have) == n_seeds and n_seeds > 1:
                name = f"ensemble{suf}"
                configs[name] = ("ensemble", None, tta)
                scores[name] = acc(torch.stack([val_p[(i, tta)] for i in have]).mean(0))
        # deterministic simple-first order (dict order: singles before ensemble,
        # no-TTA before TTA); strict argmax keeps the simplest val-winner
        chosen_name = max(configs, key=lambda nm: scores[nm])
        self.chosen = configs[chosen_name]
        self.val_scores = scores
        print(f"  v12 select: {scores} -> {chosen_name}", flush=True)

    # ---------------------------------------------------------------- predict

    @torch.no_grad()
    def predict(self, test_loader) -> List[int]:
        kind, idx, tta = self.chosen
        n_seeds = len(self.seed_states)
        # predictions must ALWAYS complete: degrade if the clock cannot fit the
        # chosen config's passes (test pass ~ val pass x |test|/|val| ~ pass_s)
        passes = (n_seeds if kind == "ensemble" else 1) * (2 if tta else 1)
        est = passes * self._pass_s * (len(test_loader) / max(len(self.valid_loader), 1))
        if self.clock.check() < est + 30.0:
            if kind == "ensemble" or tta:
                print("  v12 predict: clock short -> degrade to best single, no TTA")
            kind, idx, tta = "single", int(np.argmax(self.seed_vals)), False
        if kind == "ensemble":
            p = torch.stack([self._probs(s, test_loader, tta)
                             for s in self.seed_states]).mean(0)
        else:
            p = self._probs(self.seed_states[idx], test_loader, tta)
        return p.argmax(1).tolist()
