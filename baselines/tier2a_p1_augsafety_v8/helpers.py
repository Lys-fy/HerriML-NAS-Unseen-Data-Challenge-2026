"""Tier 2a / P1 v4: 3-probe shadow comparison on top of full EGNAS.

v3 verdict was honest negative: the K=4 single-trial probe's noise
floor turned out to be ~±8 pp (wider than estimated), so threshold
−6 pp still let AddNIST/MultNIST flip OFF wrongly on bad short-budget
draws. v4 doesn't pick *one* fix — it runs THREE probes per dataset,
logs all three decisions, and uses the most-robust one to drive main
training. Post-sweep we get clean cross-probe agreement data.

Three configs run per dataset (when aug applies):

  A) MULTI-TRIAL K=4   (3 trials × K=4 each, unanimous-OFF rule, −6 pp)
     → DRIVES main training. Averages out single-trial noise via the
       √3-rule reduction; conservative even at the original threshold.
  B) SINGLE-TRIAL K=8  (1 trial × K=8, single −6 pp rule)
     → shadow. Tests: does longer K alone reduce noise enough?
  C) MULTI-TRIAL K=8   (3 trials × K=8, unanimous-OFF rule, −6 pp)
     → shadow. Tests: does longer K add anything beyond multi-trial?

Each config produces (decision, per-trial Δscores, votes). All three
get serialised to `<codename>_p1_probe.json` for later analysis. The
driver (A) is the only one whose decision flows through to the
DataProcessor's policy override.

Total probe cost per dataset:
  A: 3 × (2 × ~6s) ≈ 36 s
  B:      2 × ~12s ≈ 24 s
  C: 3 × (2 × ~12s) ≈ 72 s
  TOTAL ≈ 132 s / dataset, ~15 min across 7 aug-applicable datasets,
  still <1 % of the 2 h budget per dataset.
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


# ---------------------------------------------------------------------------
# Grouped-conv primitives — added in v5 to test the literature finding that
# ResNeXt-style cardinality is the architectural lever for character-frequency-
# like tasks (Language, possibly Cryptic). See Geada et al. arXiv:2404.02189
# Table 1 — ResNeXt-50 hits 93.97% on Language vs Conv-only NAS ≤ 90.12%.
#
# Two flavours added to the vocabulary so EGNAS micro hill-climb can A/B them
# per layer per dataset:
#   - GroupConv3 — single 3x3 grouped conv, mid-cardinality (~8 groups for
#       our small channel widths). Sits between Conv3 (no grouping, full
#       mixing) and SepConv3 (max grouping, no cross-channel mixing).
#   - ResNeXt    — true bottleneck-grouped-expand block, the published
#       ResNeXt-50 building block: 1x1 reduce → 3x3 grouped (cardinality 8)
#       → 1x1 expand. Higher fidelity to the literature recipe.
# ---------------------------------------------------------------------------


def _safe_groups(in_ch: int, out_ch: int, max_groups: int = 8) -> int:
    """Pick the largest valid `groups` value ≤ max_groups that divides both
    in_ch and out_ch. Required because Conv2d demands groups | in_ch and
    groups | out_ch. With our W ∈ {16,24,32,40,48} and downsampled doubles,
    not every (in,out) pair is divisible by 8 — fall back to the largest
    divisor available. Falls back to 1 (= ungrouped) as the worst case."""
    g = min(max_groups, in_ch, out_ch)
    while g > 1 and (in_ch % g != 0 or out_ch % g != 0):
        g -= 1
    return max(1, g)


# ---------------------------------------------------------------------------
# v6 NEW OPS — Conv7, DilatedConv3, SE-Conv3, SpatialSelfAttn2D
# ---------------------------------------------------------------------------


def _dilated_conv_bn_act(in_ch: int, out_ch: int, stride: int,
                         kernel: int = 3, dilation: int = 2) -> nn.Sequential:
    """Dilated 3×3 → BN → ReLU. Same param count as Conv3 but with larger
    effective receptive field (5×5 at d=2). Useful where a layer needs more
    global context without inflating params."""
    pad = ((kernel - 1) * dilation) // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride,
                  padding=pad, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _SEConv3(nn.Module):
    """Conv3 + Squeeze-and-Excitation channel attention. SE block applies
    a per-channel gain learned from a global-average-pooled summary.
    ~+5 % params vs Conv3, ~free FLOPs. Broadly useful regulariser."""

    def __init__(self, in_ch: int, out_ch: int, stride: int,
                 reduction: int = 16):
        super().__init__()
        self.conv = _conv_bn_act(in_ch, out_ch, 3, stride)
        squeeze = max(out_ch // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, squeeze, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze, out_ch, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        return y * self.se(y)


class _SpatialSelfAttn2D(nn.Module):
    """Multi-head self-attention over the flattened 2D spatial grid.

    Adds non-local feature mixing in a single layer (vs many conv layers
    needed to grow receptive field). Memory cost is O((H·W)²) per layer.

    For inputs where (H·W) > MAX_ATTN_TOKENS (256 = 16×16), the input is
    first adaptive-pooled down to 16×16, attention runs on the small grid,
    then the output is bilinearly upsampled back and added as a residual
    to the projected input. This keeps memory bounded for big inputs like
    CIFARTile (32×32 = 1024 tokens × batch 128 × 4 heads ≈ 2 GB without
    the cap) without changing the op's high-level "non-local mixing"
    semantics. Stride > 1 handled by a stride-2 depthwise downsample first.
    """

    MAX_ATTN_TOKENS = 256  # = 16×16 — caps memory for big inputs

    def __init__(self, in_ch: int, out_ch: int, stride: int,
                 num_heads: int = 4):
        super().__init__()
        self.down = (
            nn.Conv2d(in_ch, in_ch, kernel_size=2, stride=2,
                      groups=in_ch, bias=False)
            if stride > 1 else nn.Identity()
        )
        self.proj_in = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        # num_heads must divide out_ch — find the largest valid head count.
        nh = num_heads
        while out_ch % nh != 0 and nh > 1:
            nh -= 1
        self.attn = nn.MultiheadAttention(out_ch, nh, batch_first=True)
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(self.down(x))                     # [B, C, H, W]
        B, C, H, W = x.shape
        if H * W > self.MAX_ATTN_TOKENS:
            # Memory-bounded path: pool → attention → upsample → residual.
            target = max(1, int(self.MAX_ATTN_TOKENS ** 0.5))
            x_small = F.adaptive_avg_pool2d(x, (target, target))
            B_, C_, h, w = x_small.shape
            tokens = x_small.flatten(2).transpose(1, 2)    # [B, target², C]
            attn_out, _ = self.attn(tokens, tokens, tokens)
            out_small = self.norm(tokens + attn_out)
            out_small = out_small.transpose(1, 2).reshape(B_, C_, h, w)
            out_up = F.interpolate(out_small, size=(H, W),
                                   mode="bilinear", align_corners=False)
            return out_up + x                              # residual to original
        # Direct path: small enough for native attention.
        tokens = x.flatten(2).transpose(1, 2)              # [B, H·W, C]
        attn_out, _ = self.attn(tokens, tokens, tokens)
        out = self.norm(tokens + attn_out)
        return out.transpose(1, 2).reshape(B, C, H, W)


def _grpconv_bn_act(in_ch: int, out_ch: int, kernel: int, stride: int,
                    max_groups: int = 8) -> nn.Sequential:
    """3x3 grouped conv → BN → ReLU. Replaces full channel mixing with
    intra-group mixing only — the ResNeXt cardinality axis applied to a
    single layer (Option A in the v5 design)."""
    pad = kernel // 2
    g = _safe_groups(in_ch, out_ch, max_groups=max_groups)
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad,
                  groups=g, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _ResNeXtOp(nn.Module):
    """The true ResNeXt-50 bottleneck-grouped-expand block adapted to our
    small-channel pipeline (Option B in the v5 design).

    1x1 reduce → 3x3 grouped (cardinality 8) → 1x1 expand.
    Bottleneck channels D = out_ch (no expansion vs full Conv3, so FLOPs are
    comparable; cardinality 8 cuts the 3x3 conv FLOPs ~8x within the
    bottleneck while preserving multi-branch representational diversity).
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int,
                 max_cardinality: int = 8):
        super().__init__()
        D = out_ch  # bottleneck width — keep == out_ch so block FLOPs ≈ Conv3
        c = _safe_groups(D, D, max_groups=max_cardinality)
        self.body = nn.Sequential(
            # 1x1 reduce (in_ch → D) — channel mixing happens here
            nn.Conv2d(in_ch, D, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(D),
            nn.ReLU(inplace=True),
            # 3x3 grouped — the cardinality axis, applies stride
            nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1,
                      groups=c, bias=False),
            nn.BatchNorm2d(D),
            nn.ReLU(inplace=True),
            # 1x1 expand (D → out_ch) — re-mix across groups before residual
            nn.Conv2d(D, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# The vocabulary used by the random op-type search. v7 CUTS the vocab back to
# CONV-ONLY (4 ops) after the v6 3-seed verdict: the rich 9-op vocab was a
# LIABILITY — the exotic ops (attention/ResNeXt/SE/Dilated/Group) won the noisy
# K=12 short-train and CRATERED at full training (Sudoku −20 pp, same D=7 W=48).
# The micro search cannot be trusted with ops whose short-train rank doesn't
# survive full-budget training. Recurrent/attention ops are tested OUT-OF-BAND
# via the FIXED_ARCH full-budget probe (baselines/fixed_arch_probe), never in
# the live search. The unused exotic op modules above are kept (dead) for
# reference; only these 4 conv ops are searchable.
OP_VOCAB = {
    "Conv3":          lambda i, o, s: _conv_bn_act(i, o, 3, s),
    "Conv5":          lambda i, o, s: _conv_bn_act(i, o, 5, s),
    "Conv7":          lambda i, o, s: _conv_bn_act(i, o, 7, s),
    "SepConv3":       lambda i, o, s: _sepconv_bn_act(i, o, 3, s),
    "MaxPool":        lambda i, o, s: _pool_op(i, o, 3, s),
    "Identity":       lambda i, o, s: _identity_op(i, o, s),
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


# v8: adaptive stem. Inputs with many channels usually encode non-spatial
# feature planes (Voxel 20ch 3D slices, Chesseract 12ch piece-types) rather
# than an RGB/grayscale image. For those, a 1x1 channel-mix BEFORE the 3x3
# spatial conv ("interpret as feature planes") beats the plain conv3 stem
# ("interpret as image"). Blind/metadata-driven (channel count only, no
# codename gate) -> Unseen Data Challenge legal. Bake-off: clear win on Voxel
# (+0.5), wash on 12ch (Chesseract/Windspeed), HURTS low-channel (MultNIST 3ch
# -0.8) -> hence the gate keeps 1/3-channel image inputs on the plain stem.
CHANNEL_EMBED_MIN_CHANNELS = int(os.environ.get("STEM_EMBED_MIN_CH", "8"))


def build_cnn(in_channels: int, num_classes: int, input_h: int, input_w: int,
              ops: Sequence[str], base_width: int = 48, drop_path_rate: float = 0.1) -> nn.Sequential:
    n_blocks = len(ops)
    plan = plan_downsamples(input_h, input_w, n_blocks)

    # Stem: gives every architecture a consistent starting representation.
    # Adaptive (see CHANNEL_EMBED_MIN_CHANNELS): high-channel feature-plane
    # inputs get a 1x1 channel-mix -> 3x3 spatial; image inputs get plain conv3.
    if in_channels >= CHANNEL_EMBED_MIN_CHANNELS:
        layers: List[nn.Module] = [
            _conv_bn_act(in_channels, base_width, 1, 1),   # 1x1 channel-embed
            _conv_bn_act(base_width, base_width, 3, 1),     # 3x3 spatial
        ]
    else:
        layers = [_conv_bn_act(in_channels, base_width, 3, 1)]
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
# tier2b_egnas_full.
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


def _run_one_trial(train_x: torch.Tensor, train_y: torch.Tensor,
                   valid_x: torch.Tensor, valid_y: torch.Tensor,
                   metadata: dict, device: torch.device,
                   base_policy: "AugPolicy", batch_size: int,
                   subset_frac: float, epochs: int,
                   sub_seed: int) -> Dict:
    """One probe trial = (stratified subset, train aug-on K epochs,
    train aug-off K epochs). Returns the trial's metrics + gap-penalised
    Δscore. Each trial uses a different `sub_seed` so trials draw
    independent subsets + model inits → multi-trial averaging actually
    reduces noise (√N rule) rather than re-running the same draw."""
    gen = torch.Generator()
    gen.manual_seed(sub_seed)
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

    gap_lambda = 0.3
    gap_on = max(0.0, train_on - val_on)
    gap_off = max(0.0, train_off - val_off)
    score_on = val_on - gap_lambda * gap_on
    score_off = val_off - gap_lambda * gap_off
    delta_pp = (score_on - score_off) * 100.0

    return {
        "sub_seed": sub_seed,
        "epochs": epochs,
        "n_train_subset": int(idx.numel()),
        "val_aug_on": val_on,
        "val_aug_off": val_off,
        "train_aug_on": train_on,
        "train_aug_off": train_off,
        "gap_aug_on": gap_on,
        "gap_aug_off": gap_off,
        "delta_pp_score": delta_pp,
        "delta_pp_raw_val": (val_on - val_off) * 100.0,
        "t_aug_on_s": t1 - t0,
        "t_aug_off_s": t2 - t1,
    }


def _run_probe_config(name: str, n_trials: int, epochs: int,
                      off_threshold_pp: float,
                      train_x: torch.Tensor, train_y: torch.Tensor,
                      valid_x: torch.Tensor, valid_y: torch.Tensor,
                      metadata: dict, device: torch.device,
                      base_policy: "AugPolicy", batch_size: int,
                      subset_frac: float, base_seed: int) -> Tuple[bool, Dict]:
    """Run `n_trials` trials with K=`epochs` each, apply the decision
    rule. Single-trial → asymmetric threshold. Multi-trial → unanimous
    OFF for OFF (any trial says ON → ON). Returns (decision, config-info).
    """
    trials: List[Dict] = []
    for t in range(n_trials):
        sub_seed = base_seed * 1000 + t
        trials.append(_run_one_trial(
            train_x, train_y, valid_x, valid_y, metadata, device,
            base_policy, batch_size, subset_frac, epochs, sub_seed,
        ))

    deltas = [t["delta_pp_score"] for t in trials]
    off_votes = sum(1 for d in deltas if d < -off_threshold_pp)
    if n_trials == 1:
        decision = not (deltas[0] < -off_threshold_pp)
        rule = f"single-trial: OFF iff Δscore < −{off_threshold_pp:g}pp"
    else:
        decision = not (off_votes == n_trials)
        rule = (f"multi-trial({n_trials}): OFF iff all trials' "
                f"Δscore < −{off_threshold_pp:g}pp "
                f"({off_votes}/{n_trials} OFF votes)")

    return decision, {
        "name": name,
        "n_trials": n_trials,
        "epochs": epochs,
        "off_threshold_pp": off_threshold_pp,
        "decision": "on" if decision else "off",
        "decision_rule": rule,
        "off_votes": off_votes,
        "deltas_pp_score": deltas,
        "median_delta_pp": sorted(deltas)[len(deltas) // 2],
        "trials": trials,
    }


def probe_aug_safety(train_x: torch.Tensor, train_y: torch.Tensor,
                     valid_x: torch.Tensor, valid_y: torch.Tensor,
                     metadata: dict, device: torch.device,
                     batch_size: int = 128,
                     subset_frac: float = 0.30,
                     off_threshold_pp: float = 6.0) -> Tuple[bool, Dict]:
    """P1 v4 — 3-probe shadow comparison. Returns the *driver*'s
    decision (A: multi-trial K=4 with the v3 asymmetric −6 pp threshold
    + unanimous-OFF rule), but logs all three configs for analysis:
      - A_multi_K4  (driver)
      - B_single_K8 (shadow)
      - C_multi_K8  (shadow)
    See module docstring for full design. The whole `info` dict —
    including each probe's per-trial Δscores — is serialised by
    DataProcessor to `<codename>_p1_probe.json`.
    """
    base_policy = aug_policy_for(metadata)
    off_threshold_pp = float(os.environ.get("P1_OFF_THRESHOLD_PP",
                                            str(off_threshold_pp)))
    base_seed = int(os.environ.get("BASELINE_SEED", "42"))

    info: Dict = {
        "version": "v6",
        "driver": "C_multi_K8",   # v4-shadow validated — keep only C in v6
        "skipped": False,
        "subset_frac": subset_frac,
        "off_threshold_pp": off_threshold_pp,
        "base_policy": base_policy.describe(),
    }

    if not base_policy.any_input_aug():
        info["decision"] = "on"
        info["skipped"] = True
        info["reason"] = "no aug applicable (metadata)"
        return True, info

    # v6: just C. A and B were the shadow probes for the v4 design
    # experiment that proved K=8 is the lever, not multi-trial. With
    # K=8 validated, the K=4 (A) and single-trial K=8 (B) shadows add
    # no decision value — drop for cleaner code + ~60 s saved per dataset.
    _, info_C = _run_probe_config(
        "C_multi_K8", n_trials=3, epochs=8,
        off_threshold_pp=off_threshold_pp,
        train_x=train_x, train_y=train_y,
        valid_x=valid_x, valid_y=valid_y,
        metadata=metadata, device=device,
        base_policy=base_policy, batch_size=batch_size,
        subset_frac=subset_frac, base_seed=base_seed,
    )
    info["C_multi_K8"] = info_C
    info["decision"] = info_C["decision"]
    decision = (info_C["decision"] == "on")
    return decision, info
