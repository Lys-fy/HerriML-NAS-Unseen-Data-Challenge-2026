"""Shared utilities for Tier-1: augmentation, regularised model, search ops.

Tier 1 = Tier 0 + the regularization Tier 0 deliberately skipped + a first
search layer. Concretely:

  - Residual ConvBlock with DropPath (vs Tier 0's plain Conv→BN→ReLU)
  - Base width 48 (vs Tier 0's 32) — more capacity, slower memorisation
  - EMA-of-weights wrapper for prediction
  - Augmentation transforms (flip, crop+pad, cutout) gated by dataset codename
  - Op vocabulary for the random op-type search: {Conv3, Conv5, SepConv3,
    MaxPool, Identity}
  - build_cnn() that constructs a sequential stack from an op spec

See docs/notes/09 - Implementation Tiers/Tier 1 - Random Op Search.md.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
