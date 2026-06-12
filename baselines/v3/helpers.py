"""Tier 2a / P1 v3: aug-safety probe on top of full EGNAS.

Identical to baselines/egnas_full/helpers.py PLUS one new function:
`probe_aug_safety()`. Everything else in this file is byte-identical so
the P1-vs-EGNAS-full comparison is clean (the *only* delta in the
pipeline is the probe's decision flowing through DataProcessor).

The probe runs two short trainings on a stratified subset (default 30 %
of train, K=4 epochs each): one with the default per-metadata AugPolicy,
one with no input aug. It compares a **gap-penalised score**
(`val_acc − 0.3·max(0, train_acc − val_acc)`) — the gap term discounts
the arm that just over-fits faster on the short budget.

**v3 — asymmetric decision rule.** v2 used a symmetric ±0.5 pp margin and
coin-flipped on borderline datasets: the K=4 probe has a measured ~±6 pp
noise floor (AddNIST Δscore swung +7.66 → −4.13 across seeds), and a
false-OFF (~−9, the −10-floor crater) costs ~3× a false-ON (~−3, the
Language-class aug-on penalty). So v3 stays **ON unless the OFF signal
clears the noise band**: aug-off only when Δscore < −P1_OFF_THRESHOLD_PP
(default 6.0 pp). Language sits at −15..−31 — far past the threshold and
correctly OFF; AddNIST/MultNIST borderline cases now stay correctly ON.

Triggered by P1_PROBE=1 (default-on in this baseline); skipped when
metadata already implies no aug (e.g. C != 1, 3). Total probe cost
~30-60 s per dataset, well under 1 % of the 2 h time budget.
"""

from __future__ import annotations

import copy as _copy_module
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Downsample planner — same as Tier 0. Spread max_ds stride-2 ops evenly
# across n_blocks, never letting min spatial dim drop below min_spatial.
# ---------------------------------------------------------------------------


def plan_downsamples(h: int, w: int, n_blocks: int, min_spatial: int = 4) -> List[bool]:
    smaller = max(min(h, w), 1)
    max_ds = int(math.floor(math.log2(smaller / min_spatial))) if smaller >= min_spatial else 0
    if max_ds <= 0 or n_blocks <= 0:
        return [False] * n_blocks
    plan = [False] * n_blocks
    step = n_blocks / max_ds
    for i in range(max_ds):
        idx = min(n_blocks - 1, int(round((i + 1) * step)) - 1)
        plan[idx] = True
    return plan


# ---------------------------------------------------------------------------
# Per-channel normalisation stats — same as Tier 0.
# ---------------------------------------------------------------------------


def per_channel_stats(train_x: np.ndarray):
    arr = train_x.astype(np.float32)
    mean = arr.mean(axis=(0, 2, 3))
    std = arr.std(axis=(0, 2, 3))
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


# ---------------------------------------------------------------------------
# DropPath (stochastic depth, per-sample) — used inside ResConvBlock.
# Drops the conv branch with probability `drop_prob`, rescaling so the
# expected output magnitude is unchanged in train mode. In eval mode, no-op.
# ---------------------------------------------------------------------------


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep_prob + torch.rand(shape, device=x.device, dtype=x.dtype)
    mask.floor_()
    return x.div(keep_prob) * mask


# ---------------------------------------------------------------------------
# Op variants for the search space. Each is a small nn.Module mapping
# (in_ch, out_ch, stride) → tensor with stride applied. Used both by the
# fixed-arch builder (Tier 1 default starts with all Conv3) and the random
# op-type search.
# ---------------------------------------------------------------------------


def _conv_bn_act(in_ch: int, out_ch: int, kernel: int, stride: int) -> nn.Sequential:
    pad = kernel // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _sepconv_bn_act(in_ch: int, out_ch: int, kernel: int, stride: int) -> nn.Sequential:
    pad = kernel // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, kernel_size=kernel, stride=stride, padding=pad,
                  groups=in_ch, bias=False),
        nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _pool_op(in_ch: int, out_ch: int, kernel: int, stride: int) -> nn.Module:
    """Max pool followed by a 1x1 projection to out_ch (handles channel doubling)."""
    pad = kernel // 2
    pool = nn.MaxPool2d(kernel_size=kernel, stride=stride, padding=pad)
    if in_ch == out_ch:
        return pool
    return nn.Sequential(pool, nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))


def _identity_op(in_ch: int, out_ch: int, stride: int) -> nn.Module:
    """Identity (when shapes match), 1x1 projection, or stride-2 avgpool.

    ceil_mode=True on the AvgPool ensures odd input dims (e.g. Sudoku 9x9)
    round up to match the Conv2d(stride=2, padding=1) path's spatial output
    of ceil(H/2). Without ceil_mode, AvgPool floors and the residual add
    fails with a shape mismatch.
    """
    if stride == 1 and in_ch == out_ch:
        return nn.Identity()
    layers = []
    if stride != 1:
        layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True))
    if in_ch != out_ch:
        layers.append(nn.Conv2d(in_ch, out_ch, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))
    return nn.Sequential(*layers) if len(layers) > 1 else layers[0]


# The vocabulary used by the random op-type search. Each entry is a callable
# (in_ch, out_ch, stride) -> nn.Module.
OP_VOCAB = {
    "Conv3": lambda i, o, s: _conv_bn_act(i, o, 3, s),
    "Conv5": lambda i, o, s: _conv_bn_act(i, o, 5, s),
    "SepConv3": lambda i, o, s: _sepconv_bn_act(i, o, 3, s),
    "MaxPool": lambda i, o, s: _pool_op(i, o, 3, s),
    "Identity": lambda i, o, s: _identity_op(i, o, s),
}
OP_NAMES = tuple(OP_VOCAB.keys())


# ---------------------------------------------------------------------------
# ResConvBlock — residual wrapper around any of the OP_VOCAB ops. The
# residual path mirrors the op's spatial/channel transform with a cheap 1x1
# (or AvgPool + 1x1 when stride > 1). DropPath drops the op branch
# per-sample during training.
# ---------------------------------------------------------------------------


class ResConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, op_name: str = "Conv3",
                 drop_path_rate: float = 0.0):
        super().__init__()
        self.drop_path_rate = drop_path_rate
        self.op_name = op_name
        self.op = OP_VOCAB[op_name](in_ch, out_ch, stride)
        # Shortcut: identity when shapes match, else a 1x1 projection.
        # When stride > 1 we use a stride-2 1x1 conv directly — same spatial
        # output as the conv-3 branch (both compute ceil(H/2) on odd H, e.g.
        # Sudoku 9x9 → 5x5). The previous AvgPool-then-1x1 approach floored
        # and produced 4x4, which mismatched the conv branch's 5x5.
        if stride == 1 and in_ch == out_ch:
            self.shortcut: nn.Module = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        branch = self.op(x)
        branch = drop_path(branch, self.drop_path_rate, self.training)
        return branch + residual


# ---------------------------------------------------------------------------
# build_cnn: stem → N res-blocks → GAP → linear. Each block's op_name comes
# from `ops` (length n_blocks). Downsample positions and channel widths are
# determined by plan_downsamples + width-doubling on each downsample.
# ---------------------------------------------------------------------------


def build_cnn(in_channels: int, num_classes: int, input_h: int, input_w: int,
              ops: Sequence[str], base_width: int = 48, drop_path_rate: float = 0.1) -> nn.Sequential:
    n_blocks = len(ops)
    plan = plan_downsamples(input_h, input_w, n_blocks)

    layers: List[nn.Module] = [
        # Stem: plain Conv3 (always) — gives every architecture a consistent
        # starting representation.
        _conv_bn_act(in_channels, base_width, 3, 1),
    ]
    ch = base_width
    for i, downsample in enumerate(plan):
        op_name = ops[i]
        next_ch = ch * 2 if downsample else ch
        stride = 2 if downsample else 1
        layers.append(ResConvBlock(ch, next_ch, stride=stride, op_name=op_name,
                                   drop_path_rate=drop_path_rate))
        ch = next_ch

    layers.append(nn.AdaptiveAvgPool2d(1))
    layers.append(nn.Flatten())
    layers.append(nn.Linear(ch, num_classes))
    return nn.Sequential(*layers)


def build_default_cnn(in_channels: int, num_classes: int, input_h: int, input_w: int,
                      base_width: int = 48, n_blocks: int = 6,
                      drop_path_rate: float = 0.1) -> nn.Sequential:
    """All-Conv3 default architecture — used as the search bootstrap and as
    the final retrain fallback if search fails."""
    return build_cnn(in_channels, num_classes, input_h, input_w,
                     ops=["Conv3"] * n_blocks,
                     base_width=base_width, drop_path_rate=drop_path_rate)


# ---------------------------------------------------------------------------
# Augmentation policy — metadata-only, no codename gates.
#
# The competition is the Unseen Data Challenge: Phase-2/3 datasets are
# hidden and their codenames will be values we have never seen. So the
# policy has to be decided from metadata at runtime (input_shape, classes,
# benchmark) and default to conservative safe choices.
#
# Design principles:
#   1. Only LABEL-PRESERVING augmentations are enabled by default.
#      Cutout (zeros a patch) and small random crop (shifts by ≤ 2 px) are
#      essentially label-preserving on standard image classification.
#   2. Flip is NEVER enabled. Horizontal-flipping breaks orientation-
#      sensitive datasets (digits, encoded text, board games). We cannot
#      tell from metadata whether a dataset is flip-safe.
#   3. Mixup is NEVER enabled. Mixing two samples breaks any task whose
#      label is an arithmetic function of input identities (count, sum,
#      product, presence — and we cannot detect those from metadata).
#      Mixup may be reintroduced LATER via a runtime probe inside NAS
#      that measures whether it helps on this specific dataset.
#   4. Aug only when the input "looks image-like" (1 or 3 channels). Inputs
#      with many channels usually encode non-spatial structure (Chesseract
#      12-channel board, Voxel 20-channel 3D).
# ---------------------------------------------------------------------------


@dataclass
class AugPolicy:
    flip: bool = False
    crop_pad: int = 0           # 0 disables; otherwise pads by N then random-crops back
    cutout_size: int = 0        # 0 disables; otherwise length of cutout square
    mixup_alpha: float = 0.0    # 0 disables Mixup; otherwise Beta(alpha, alpha) λ

    def any_input_aug(self) -> bool:
        return self.flip or self.crop_pad > 0 or self.cutout_size > 0

    def describe(self) -> str:
        bits = []
        if self.flip:
            bits.append("flip")
        if self.crop_pad > 0:
            bits.append(f"crop_pad={self.crop_pad}")
        if self.cutout_size > 0:
            bits.append(f"cutout={self.cutout_size}")
        if self.mixup_alpha > 0:
            bits.append(f"mixup_a={self.mixup_alpha}")
        return ", ".join(bits) if bits else "none"


def aug_policy_for(metadata: dict) -> AugPolicy:
    """Pick the augmentation policy from metadata alone (no codename lookups).

    Conservative, intentionally — false positives (applying aug to an unseen
    dataset where it breaks the task) cost more than false negatives
    (skipping aug on a standard image dataset).

    Ablation hook: setting TIER1_AUG_OFF=1 in the environment forces the
    empty policy regardless of metadata, used to isolate the contribution
    of augmentation in tier-1 ablation runs.
    """
    if os.environ.get("TIER1_AUG_OFF") == "1":
        return AugPolicy()

    shape = metadata.get("input_shape", [])
    if len(shape) == 4:
        C = int(shape[1])
        H = int(shape[-2])
        W = int(shape[-1])
    elif len(shape) == 3:
        C = int(shape[0])
        H = int(shape[-2])
        W = int(shape[-1])
    else:
        return AugPolicy()

    # Aug only when the input is image-like (1-ch grayscale or 3-ch RGB).
    # Other channel counts almost always encode non-spatial structure on this
    # competition's datasets and shouldn't get spatial aug.
    if C not in (1, 3):
        return AugPolicy()

    p = AugPolicy()

    # Cutout — zeros a small patch. Length scales mildly with input size,
    # capped at 16 for big inputs. ≤ 16x16 in a 64x64 input is 6 % of area,
    # which is small enough to be safe on most reasonable tasks.
    if H >= 16 and W >= 16:
        p.cutout_size = min(16, max(4, H // 8))

    # Small random crop — pad by 2 with reflection, then random-crop back.
    # Only on square inputs ≥ 24 px (asymmetric shapes like 27×18 likely
    # encode positional information; square ≥ 24 px is the standard image
    # case). Pad=2 means at most a 2-pixel shift in any direction.
    if H >= 24 and W >= 24 and H == W:
        p.crop_pad = 2

    # Flip and Mixup remain disabled (see module docstring above).
    return p


# ---------------------------------------------------------------------------
# AugmentedTensorDataset — applies the configured per-sample transforms in
# __getitem__. Inputs must already be normalised (Float32 tensors).
# ---------------------------------------------------------------------------


class AugmentedTensorDataset(torch.utils.data.Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, policy: AugPolicy):
        self.x = x
        self.y = y
        self.policy = policy

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int):
        x = self.x[idx]
        p = self.policy

        if p.flip and torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[-1])

        if p.crop_pad > 0:
            x = self._random_crop(x, pad=p.crop_pad)

        if p.cutout_size > 0:
            x = self._cutout(x, size=p.cutout_size)

        return x, self.y[idx]

    @staticmethod
    def _random_crop(x: torch.Tensor, pad: int) -> torch.Tensor:
        C, H, W = x.shape
        x_pad = F.pad(x, [pad, pad, pad, pad], mode="reflect")
        top = int(torch.randint(0, 2 * pad + 1, (1,)).item())
        left = int(torch.randint(0, 2 * pad + 1, (1,)).item())
        return x_pad[..., top : top + H, left : left + W]

    @staticmethod
    def _cutout(x: torch.Tensor, size: int) -> torch.Tensor:
        C, H, W = x.shape
        cy = int(torch.randint(0, H, (1,)).item())
        cx = int(torch.randint(0, W, (1,)).item())
        y1 = max(0, cy - size // 2)
        y2 = min(H, cy + size // 2)
        x1 = max(0, cx - size // 2)
        x2 = min(W, cx + size // 2)
        x = x.clone()
        x[..., y1:y2, x1:x2] = 0.0
        return x


# ---------------------------------------------------------------------------
# EMA — exponential moving average of model weights, used for prediction.
# Almost-free upgrade per the Tier 1 ideas list.
# ---------------------------------------------------------------------------


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        import copy as _copy
        self.decay = decay
        self.ema = _copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        # Decay parameters; copy buffers (BN running stats) directly so EMA
        # tracks the live model's normalisation statistics.
        for ema_p, p in zip(self.ema.parameters(), model.parameters()):
            ema_p.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
        for ema_b, b in zip(self.ema.buffers(), model.buffers()):
            ema_b.copy_(b)

    def state_dict(self):
        return self.ema.state_dict()


# ---------------------------------------------------------------------------
# LR schedule helper — manual cosine + linear warmup. Used by trainer.
# (Avoids torch's SequentialLR which is finicky when we want to construct
# the schedule after epoch 0 from observed wall-clock.)
# ---------------------------------------------------------------------------


def lr_for_epoch(epoch: int, total_epochs: int, warmup_epochs: int = 5,
                 base_lr: float = 0.025, min_lr: float = 0.0) -> float:
    if total_epochs <= 1:
        return base_lr
    if epoch < warmup_epochs:
        # Linear warmup from min_lr to base_lr.
        return min_lr + (base_lr - min_lr) * (epoch + 1) / warmup_epochs
    # Cosine anneal from base_lr to min_lr over (total - warmup) epochs.
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    progress = min(1.0, progress)
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# P1 PROBE — aug-safety. The one thing that makes this baseline differ from
# egnas_full.
# ---------------------------------------------------------------------------


def _stratified_subset_indices(y: torch.Tensor, frac: float, generator: torch.Generator) -> torch.Tensor:
    """Per-class sample `frac` of the indices in `y`, concatenated.

    Stratified so the probe's two arms see the same class distribution as
    the full dataset (a non-stratified subset can bias short trainings on
    imbalanced datasets like Myofibre).
    """
    y_np = y.numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
    classes, counts = np.unique(y_np, return_counts=True)
    out_idx: List[int] = []
    for c in classes:
        idx_c = np.where(y_np == c)[0]
        k = max(1, int(round(len(idx_c) * frac)))
        # Use torch generator for reproducibility across the probe arms.
        perm = torch.randperm(len(idx_c), generator=generator).numpy()
        out_idx.extend(idx_c[perm[:k]].tolist())
    return torch.as_tensor(sorted(out_idx), dtype=torch.long)


def _probe_train_one_arm(train_x: torch.Tensor, train_y: torch.Tensor,
                         valid_x: torch.Tensor, valid_y: torch.Tensor,
                         metadata: dict, device: torch.device,
                         policy: "AugPolicy", batch_size: int,
                         epochs: int, base_lr: float = 0.025
                         ) -> Tuple[float, float]:
    """Train a default CNN for `epochs` epochs with the given AugPolicy,
    return (best_val_acc, final_train_acc).

    Minimal training loop — no EMA, no cosine schedule, no NaN recovery,
    no CSV. Just SGD + label-smoothed CE. The whole point is *cheap*
    signal on which aug arm wins, not a full Tier-1 training.

    Returns *both* val and train acc so the caller can build a
    gap-penalised score: a short-budget training where aug-off looks
    better on raw val often has a much larger train-val gap (the
    over-fitting signal that precedes the −10 floor we see at full
    budget on e.g. AddNIST). Picking on raw val alone is fooled by this;
    picking on (val − λ·gap) is not.
    """
    # Decode input shape (4D ship-form or 3D NCHW).
    shape = metadata["input_shape"]
    if len(shape) == 4:
        _, in_ch, h, w = shape
    elif len(shape) == 3:
        in_ch, h, w = shape
    else:
        raise ValueError(f"Unexpected input_shape: {shape}")
    in_ch, h, w = int(in_ch), int(h), int(w)
    num_classes = int(metadata.get("num_classes", int(train_y.max().item() + 1)))

    if policy.any_input_aug():
        train_ds = AugmentedTensorDataset(train_x, train_y, policy)
    else:
        train_ds = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=False, num_workers=0)
    valid_loader = DataLoader(TensorDataset(valid_x, valid_y),
                              batch_size=batch_size, shuffle=False,
                              drop_last=False, num_workers=0)

    model = build_default_cnn(in_ch, num_classes, h, w,
                              base_width=48, n_blocks=6,
                              drop_path_rate=0.0).to(device)
    model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val = -1.0
    last_train_acc = 0.0
    for ep in range(epochs):
        model.train()
        train_correct = train_total = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True,
                       memory_format=torch.channels_last)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if not torch.isfinite(loss):
                continue  # skip a NaN-loss batch; probe is best-effort
            loss.backward()
            opt.step()
            with torch.no_grad():
                train_correct += (logits.argmax(dim=1) == yb).sum().item()
                train_total += yb.size(0)
        # Final-epoch train acc is the cheapest no-extra-pass proxy we get.
        if ep == epochs - 1 and train_total > 0:
            last_train_acc = train_correct / train_total
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device, non_blocking=True,
                           memory_format=torch.channels_last)
                yb = yb.to(device, non_blocking=True)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        acc = correct / max(1, total)
        best_val = max(best_val, acc)
    return best_val, last_train_acc


def probe_aug_safety(train_x: torch.Tensor, train_y: torch.Tensor,
                     valid_x: torch.Tensor, valid_y: torch.Tensor,
                     metadata: dict, device: torch.device,
                     batch_size: int = 128,
                     subset_frac: float = 0.30,
                     epochs: int = 4,
                     off_threshold_pp: float = 6.0) -> Tuple[bool, Dict]:
    """P1 v3 — decide whether to use augmentation for this dataset.

    Runs two short trainings on a stratified `subset_frac` of train:
    arm A = default per-metadata `aug_policy_for(metadata)`,
    arm B = empty `AugPolicy()` (no input aug).

    Decision (v3 asymmetric rule): stay aug-ON unless the gap-penalised
    Δscore is strongly negative —
      - aug-OFF if Δscore < −off_threshold_pp
      - aug-ON  otherwise (the entire ±noise band defaults to ON)
    `off_threshold_pp` is env-overridable via P1_OFF_THRESHOLD_PP. The
    threshold sits at the measured K=4 noise floor (~6 pp) so noise-band
    borderline datasets cannot flip; only a Language-class signal
    (Δscore −15..−31) clears it.

    Returns (use_aug: bool, info: dict). `info` carries the per-arm
    val/train acc, gaps, score, the threshold, and the decision —
    serialised by DataProcessor for traceability.
    """
    base_policy = aug_policy_for(metadata)
    off_threshold_pp = float(os.environ.get("P1_OFF_THRESHOLD_PP",
                                            str(off_threshold_pp)))

    info: Dict = {
        "decision": None,
        "skipped": False,
        "subset_frac": subset_frac,
        "epochs": epochs,
        "off_threshold_pp": off_threshold_pp,
        "decision_rule": f"v3 asymmetric: OFF iff Δscore < −{off_threshold_pp:g}pp",
        "base_policy": base_policy.describe(),
    }

    # No aug applicable from metadata → nothing to decide; probe is a no-op.
    if not base_policy.any_input_aug():
        info["decision"] = "on"
        info["skipped"] = True
        info["reason"] = "no aug applicable (metadata)"
        return True, info

    # Stratified subset of train, fixed by the global torch seed so both arms
    # see the same data.
    gen = torch.Generator()
    gen.manual_seed(int(os.environ.get("BASELINE_SEED", "42")))
    idx = _stratified_subset_indices(train_y, subset_frac, gen)
    sub_train_x = train_x[idx].contiguous()
    sub_train_y = train_y[idx].contiguous()

    t0 = time.time()
    val_on, train_on = _probe_train_one_arm(
        sub_train_x, sub_train_y, valid_x, valid_y, metadata, device,
        policy=base_policy, batch_size=batch_size, epochs=epochs,
    )
    t1 = time.time()
    val_off, train_off = _probe_train_one_arm(
        sub_train_x, sub_train_y, valid_x, valid_y, metadata, device,
        policy=AugPolicy(), batch_size=batch_size, epochs=epochs,
    )
    t2 = time.time()

    # Gap-penalised score — fixes the "aug-off overfits faster at K=4"
    # failure mode: aug-off's higher raw val on short trainings is
    # discounted by its larger train-val gap. λ chosen so a 10 pp gap
    # difference offsets a 3 pp val advantage (a typical AddNIST signal).
    gap_lambda = 0.3
    gap_on = max(0.0, train_on - val_on)
    gap_off = max(0.0, train_off - val_off)
    score_on = val_on - gap_lambda * gap_on
    score_off = val_off - gap_lambda * gap_off
    delta_pp = (score_on - score_off) * 100.0  # *_on - *_off, sign-consistent

    # v3 asymmetric rule: stay ON unless the OFF signal clears the noise
    # band. False-OFF (~−9, the −10-floor crater) costs ~3× false-ON
    # (~−3), and the K=4 probe noise floor is ~±6 pp — so only act on a
    # signal past −off_threshold_pp. Language (Δscore −15..−31) clears it
    # comfortably; AddNIST/MultNIST borderline cases stay correctly ON.
    decision = not (delta_pp < -off_threshold_pp)

    info.update({
        "decision": "on" if decision else "off",
        "val_aug_on": val_on,
        "val_aug_off": val_off,
        "train_aug_on": train_on,
        "train_aug_off": train_off,
        "gap_aug_on": gap_on,
        "gap_aug_off": gap_off,
        "score_aug_on": score_on,
        "score_aug_off": score_off,
        "delta_pp_score": delta_pp,
        "delta_pp_raw_val": (val_on - val_off) * 100.0,
        "gap_lambda": gap_lambda,
        "t_aug_on_s": t1 - t0,
        "t_aug_off_s": t2 - t1,
        "n_train_subset": int(idx.numel()),
    })
    return decision, info
