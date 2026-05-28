"""Tier 2a / Language Excursion NAS — no-search Tier-1 fallback.

For the Language-only excursion we DISABLE EGNAS entirely. Architecture
is fixed to a default conv stack with env-overridable depth/width so the
'larger model' variation can be tested without involving search. This
keeps the experiment clean: only the variations under test (patience,
LR schedule, mixup, model width) move; everything else is held constant.

Env knobs:
  - LANG_DEPTH: int, default 6 — number of residual ConvBlocks in the stack.
  - LANG_WIDTH: int, default 48 — base channel width.
  - LANG_DROP_PATH: float, default 0.1 — DropPath rate inside ResConvBlock.
"""

from __future__ import annotations

import os

import torch

from helpers import build_cnn, build_default_cnn


class NAS:
    def __init__(self, train_loader, valid_loader, metadata, clock):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.metadata = metadata
        self.clock = clock

    def _input_shape(self):
        shape = self.metadata["input_shape"]
        if len(shape) == 4:
            _, in_channels, h, w = shape
        elif len(shape) == 3:
            in_channels, h, w = shape
        else:
            raise ValueError(f"Unexpected input_shape: {shape}")
        return int(in_channels), int(h), int(w)

    def search(self):
        in_channels, h, w = self._input_shape()
        num_classes = int(self.metadata.get("num_classes", 10))
        depth = int(os.environ.get("LANG_DEPTH", "6"))
        width = int(os.environ.get("LANG_WIDTH", "48"))
        drop_path = float(os.environ.get("LANG_DROP_PATH", "0.1"))
        print(f"  Language excursion NAS: fixed default arch D={depth} W={width} "
              f"drop_path={drop_path} (no EGNAS search)")
        return build_cnn(in_channels, num_classes, h, w,
                         ops=["Conv3"] * depth,
                         base_width=width,
                         drop_path_rate=drop_path)
