"""Tier-0 NAS: no search. Returns a single hand-sized CNN.

Reads in_channels and spatial dims from metadata['input_shape'], num_classes
from metadata['num_classes']. helpers.build_fixed_cnn picks downsample
positions adaptively, so this same code produces working models for inputs
as small as Sudoku 1x9x9 and as large as Myofibre 3x128x128.
"""

from __future__ import annotations

from helpers import build_fixed_cnn


class NAS:
    def __init__(self, train_loader, valid_loader, metadata, clock):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.metadata = metadata
        self.clock = clock

    def search(self):
        shape = self.metadata["input_shape"]
        # input_shape is (N, C, H, W); some 2024 datasets may publish it
        # without a leading N — accept either form.
        if len(shape) == 4:
            _, in_channels, h, w = shape
        elif len(shape) == 3:
            in_channels, h, w = shape
        else:
            raise ValueError(f"Unexpected input_shape: {shape}")

        return build_fixed_cnn(
            in_channels=int(in_channels),
            num_classes=int(self.metadata["num_classes"]),
            input_h=int(h),
            input_w=int(w),
            base_width=32,
            n_blocks=6,
        )
