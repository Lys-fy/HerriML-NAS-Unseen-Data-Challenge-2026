"""Tier-0 DataProcessor: numpy arrays → normalised tensors → DataLoaders.

Kept deliberately simple — no augmentation, no per-dataset specialisation.
The only adaptive behaviour is per-channel normalisation computed from
train_x at runtime (handles 1ch, 3ch, 12ch, 20ch inputs uniformly).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from helpers import per_channel_stats


class DataProcessor:
    def __init__(self, train_x, train_y, valid_x, valid_y, test_x, metadata, clock):
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.test_x = test_x
        self.metadata = metadata
        self.clock = clock
        # Conservative default — small enough to fit on most GPUs across all
        # known input shapes (Myofibre 3x128x128 at 128 batch ≈ 25 MB activations).
        self.batch_size = 128

    @staticmethod
    def _to_nchw(x: np.ndarray) -> np.ndarray:
        # Some datasets (e.g. Sudoku) ship 3D arrays despite advertising a
        # 4D shape in metadata. Force everything to (N, C, H, W).
        if x.ndim == 3:
            x = x[:, None, :, :]
        return x

    def _normalise(self, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
        # (N, C, H, W) float32, channel-wise z-score.
        arr = self._to_nchw(x).astype(np.float32)
        arr = (arr - mean[None, :, None, None]) / std[None, :, None, None]
        return torch.from_numpy(arr)

    def process(self):
        train_x_nchw = self._to_nchw(self.train_x)
        mean, std = per_channel_stats(train_x_nchw)

        train_x = self._normalise(self.train_x, mean, std)
        valid_x = self._normalise(self.valid_x, mean, std)
        test_x = self._normalise(self.test_x, mean, std)

        train_y = torch.as_tensor(np.asarray(self.train_y), dtype=torch.long)
        valid_y = torch.as_tensor(np.asarray(self.valid_y), dtype=torch.long)

        train_loader = DataLoader(
            TensorDataset(train_x, train_y),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
        valid_loader = DataLoader(
            TensorDataset(valid_x, valid_y),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        # Test loader must be deterministic and complete — the grader asserts
        # both. SequentialSampler is explicit insurance against accidental
        # shuffling if a future refactor changes defaults.
        test_dataset = TensorDataset(test_x)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            sampler=SequentialSampler(test_dataset),
            drop_last=False,
            num_workers=0,
        )
        return train_loader, valid_loader, test_loader
