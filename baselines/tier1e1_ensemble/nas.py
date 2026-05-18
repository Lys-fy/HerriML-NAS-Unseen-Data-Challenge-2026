"""Tier-1e1 NAS: random op-type search + E1 ensemble candidate hand-off.

Identical search to Tier 1, plus: it records every short-trained candidate's
val_acc and attaches the top-ranked *other* architectures (and the rebuild
recipe) to the returned model as `model._e1`, so the E1 Trainer can spawn a
budget-filling ensemble (chosen-arch seed restarts + decorrelated candidate
archs). See docs/notes Probe-Driven Self-Configuration §E1.

Original Tier-1 docstring:
Tier-1 NAS: random op-type search over a linear stack.

Recipe (per docs/notes/09 - Implementation Tiers/Tier 1 - Random Op Search.md):

  1. Build the default all-Conv3 architecture as the bootstrap candidate.
  2. Sample K random architectures from the op vocabulary
     {Conv3, Conv5, SepConv3, MaxPool, Identity}, one op per block.
  3. Short-train each candidate for E epochs (or until per-candidate
     deadline) on the train_loader; rank by val accuracy.
  4. Return the best architecture as a fresh nn.Sequential — Trainer
     full-trains it from scratch.

Budget: 30% NAS / 65% Trainer / 5% buffer of the total metadata['time_remaining'].
The Trainer will use what's left on the clock when it runs, so under-spending
in NAS naturally flows to the Trainer (the strategy notes' implicit invariant).

Defensive: every candidate's short-train is wrapped in try/except. NaN losses
or OOMs drop that candidate silently and the loop continues. If the WHOLE
search fails, we fall back to the default Conv3 architecture so Trainer
always gets something working.
"""

from __future__ import annotations

import os
import random
import time
import traceback
from typing import List, Sequence

import torch
import torch.nn as nn

from helpers import (
    OP_NAMES,
    build_cnn,
    build_default_cnn,
)


# E1: how many *other* (non-chosen) searched candidate archs to hand the
# Trainer for the candidate-architecture ensemble axis. The Trainer also
# does seed restarts of the chosen arch (the axis Step-0 / Q8 showed is the
# real variance driver on AddNIST/MultNIST); this just adds cheap arch
# decorrelation on top.
E1_MAX_OTHER_ARCHS = 4


def _attach_e1(model, *, in_channels, num_classes, h, w,
               base_width, drop_path_rate, chosen_ops, ranked_other_ops):
    """Attach the E1 rebuild recipe to the returned nn.Module.

    The mock harness passes the *same* object NAS.search() returns straight
    into Trainer (verified against the contract), so a plain attribute is the
    clean, contract-safe channel for the Trainer to spawn ensemble members.
    Set on every return path (incl. fallbacks) — Trainer guards with getattr
    and degrades to a single member if it is missing.
    """
    model._e1 = {
        "in_channels": int(in_channels),
        "num_classes": int(num_classes),
        "h": int(h),
        "w": int(w),
        "base_width": int(base_width),
        "drop_path_rate": float(drop_path_rate),
        "chosen_ops": list(chosen_ops),
        "ranked_other_ops": [list(a) for a in ranked_other_ops],
    }
    return model


# How much of the per-dataset clock goes to NAS (search) vs final retrain.
SEARCH_BUDGET_FRACTION = 0.30

# Number of random architectures to sample per dataset.
K_CANDIDATES = 8

# Number of blocks in the linear stack. Fixed for Tier 1; Tier 2 (EGNAS) varies it.
N_BLOCKS = 6

# Per-candidate short-train max epochs (caps when budget is generous).
MAX_SHORT_EPOCHS = 3


class NAS:
    def __init__(self, train_loader, valid_loader, metadata, clock):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.metadata = metadata
        self.clock = clock

    # -----------------------------------------------------------------------

    def _input_shape(self):
        shape = self.metadata["input_shape"]
        if len(shape) == 4:
            _, in_channels, h, w = shape
        elif len(shape) == 3:
            in_channels, h, w = shape
        else:
            raise ValueError(f"Unexpected input_shape: {shape}")
        return int(in_channels), int(h), int(w)

    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------

    def _short_train(self, model: nn.Module, device, deadline: float, max_epochs: int) -> float:
        """Train model for up to max_epochs OR until time.perf_counter() > deadline.
        Returns val_acc; raises on NaN.
        """
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

    # -----------------------------------------------------------------------

    def _sample_arch(self, rng: random.Random) -> List[str]:
        """Pick an op per block uniformly from the op vocabulary."""
        return [rng.choice(OP_NAMES) for _ in range(N_BLOCKS)]

    # -----------------------------------------------------------------------

    def search(self):
        in_channels, h, w = self._input_shape()
        num_classes = int(self.metadata["num_classes"])
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Ablation hook: skip the search entirely, return the fixed default
        # all-Conv3 architecture. Used to isolate the contribution of the
        # random op-type search loop in tier-1 ablation runs.
        if os.environ.get("TIER1_SEARCH_OFF") == "1":
            print("  TIER1_SEARCH_OFF=1: skipping random op-type search, "
                  "returning default Conv3 architecture")
            default_ops = ["Conv3"] * N_BLOCKS
            m = build_default_cnn(
                in_channels=in_channels, num_classes=num_classes,
                input_h=h, input_w=w,
                base_width=48, n_blocks=N_BLOCKS, drop_path_rate=0.1,
            )
            # No searched pool here → E1 falls back to chosen-arch seed
            # restarts only (still the dominant variance axis per Q8).
            return _attach_e1(m, in_channels=in_channels, num_classes=num_classes,
                              h=h, w=w, base_width=48, drop_path_rate=0.1,
                              chosen_ops=default_ops, ranked_other_ops=[])

        total_remaining = self.clock.check()
        search_budget = total_remaining * SEARCH_BUDGET_FRACTION
        per_candidate_budget = search_budget / K_CANDIDATES
        search_deadline = time.perf_counter() + search_budget

        print(f"  search budget: {search_budget:.0f}s "
              f"(~{per_candidate_budget:.0f}s/candidate, K={K_CANDIDATES})")

        rng = random.Random(int(os.environ.get("BASELINE_SEED", "42")))  # seed-swept

        # Bootstrap with the default all-Conv3 arch as candidate 0 — guarantees
        # we always have at least one trained, working candidate even if every
        # random sample fails.
        candidates: List[List[str]] = [["Conv3"] * N_BLOCKS]
        for _ in range(K_CANDIDATES - 1):
            candidates.append(self._sample_arch(rng))

        best_arch: List[str] = candidates[0]
        best_acc = -1.0
        evaluated: List[tuple] = []  # (arch, val_acc) for every short-trained cand

        for i, arch in enumerate(candidates):
            if time.perf_counter() >= search_deadline:
                print(f"  candidate {i}: search budget exhausted, stopping early")
                break

            cand_deadline = time.perf_counter() + per_candidate_budget
            try:
                model = build_cnn(in_channels, num_classes, h, w, ops=arch,
                                  base_width=48, drop_path_rate=0.0)  # no droppath for short-train
                val_acc = self._short_train(model, device, deadline=cand_deadline,
                                            max_epochs=MAX_SHORT_EPOCHS)
                print(f"  candidate {i:2d}  ops={arch}  val_acc={val_acc:.4f}")
                evaluated.append((arch, val_acc))
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_arch = arch
            except Exception as ex:
                print(f"  candidate {i:2d}  ops={arch}  FAILED ({type(ex).__name__}: {ex})")
                # Keep going.
                continue
            finally:
                # Free the candidate's tensors so the next one starts clean.
                if "model" in locals():
                    del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print(f"  search done: best arch={best_arch}  val_acc={best_acc:.4f}")

        # E1: rank the *other* short-trained candidates best-first, dedupe,
        # drop the chosen one, cap. These give the Trainer decorrelated
        # architectures for the candidate-ensemble axis.
        ranked_other: List[List[str]] = []
        seen = {tuple(best_arch)}
        for arch, _ in sorted(evaluated, key=lambda av: -av[1]):
            key = tuple(arch)
            if key in seen:
                continue
            seen.add(key)
            ranked_other.append(list(arch))
            if len(ranked_other) >= E1_MAX_OTHER_ARCHS:
                break
        print(f"  E1: handing Trainer {len(ranked_other)} other candidate "
              f"arch(s) + chosen-arch seed restarts")

        # Build the winning architecture fresh (random init) so Trainer
        # full-trains it from scratch. DropPath enabled for the retrain.
        final_model = build_cnn(in_channels, num_classes, h, w, ops=best_arch,
                                base_width=48, drop_path_rate=0.1)
        return _attach_e1(final_model, in_channels=in_channels,
                          num_classes=num_classes, h=h, w=w, base_width=48,
                          drop_path_rate=0.1, chosen_ops=best_arch,
                          ranked_other_ops=ranked_other)
