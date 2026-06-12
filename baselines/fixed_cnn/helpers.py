"""Shared utilities for the Tier-0 fixed-CNN baseline.

Tier 0 ships zero search: NAS.search() builds a small fixed CNN whose depth
is constant but whose downsampling positions are picked at runtime so the
same code works on Sudoku 9x9, AddNIST 28x28, CIFARTile 64x64, Myofibre
128x128, and Voxel 20x20 (20-channel) without spatial collapse.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn


def plan_downsamples(h: int, w: int, n_blocks: int, min_spatial: int = 4) -> List[bool]:
    """Decide per-block whether to apply stride-2 downsampling.

    Returns a list of length n_blocks. Downsamples are spread evenly across
    the depth but never applied if they would shrink min(H, W) below
    min_spatial. Small inputs like Sudoku 9x9 therefore get an all-False
    plan instead of collapsing to <1x1.
    """
    smaller = max(min(h, w), 1)
    # How many halvings keep us at or above min_spatial.
    max_ds = int(math.floor(math.log2(smaller / min_spatial))) if smaller >= min_spatial else 0
    if max_ds <= 0 or n_blocks <= 0:
        return [False] * n_blocks

    # Spread max_ds True flags evenly across n_blocks (offset so the first
    # downsample isn't the very first block — keeps an initial stem at full res).
    plan = [False] * n_blocks
    step = n_blocks / max_ds
    for i in range(max_ds):
        idx = min(n_blocks - 1, int(round((i + 1) * step)) - 1)
        plan[idx] = True
    return plan


class ConvBlock(nn.Module):
    """Conv → BN → ReLU, optionally with stride-2 downsampling."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


def build_fixed_cnn(in_channels: int, num_classes: int, input_h: int, input_w: int,
                    base_width: int = 32, n_blocks: int = 6) -> nn.Sequential:
    """Build a small sequential CNN sized for the given input shape.

    Architecture: stem (Conv3x3 → BN → ReLU) → n_blocks ConvBlocks with
    adaptive downsampling at evenly spaced positions (channels doubled on
    each downsample) → GAP → Linear(num_classes).
    """
    plan = plan_downsamples(input_h, input_w, n_blocks)

    layers: List[nn.Module] = [ConvBlock(in_channels, base_width, stride=1)]
    ch = base_width
    for downsample in plan:
        next_ch = ch * 2 if downsample else ch
        layers.append(ConvBlock(ch, next_ch, stride=2 if downsample else 1))
        ch = next_ch

    layers.append(nn.AdaptiveAvgPool2d(1))
    layers.append(nn.Flatten())
    layers.append(nn.Linear(ch, num_classes))
    return nn.Sequential(*layers)


def per_channel_stats(train_x: "np.ndarray"):
    """Compute per-channel mean and std from a (N, C, H, W) float array.

    Used by DataProcessor to normalise inputs without depending on torchvision
    transform pipelines (which can be version-flaky across PyTorch releases).
    """
    import numpy as np
    arr = train_x.astype(np.float32)
    # Reduce over N, H, W -> shape (C,)
    mean = arr.mean(axis=(0, 2, 3))
    std = arr.std(axis=(0, 2, 3))
    std = np.where(std < 1e-6, 1.0, std)  # avoid divide-by-zero on constant channels
    return mean, std
