"""Tier-2a / Language Excursion DataProcessor.

Same shape-robustness and per-channel z-score normalisation as Tier 0,
plus the existing AugPolicy plumbing — but with one additional knob for
the excursion experiments: `LANG_MIXUP_ALPHA` env var overrides the
metadata-derived mixup α *after* the policy is picked, so we can test
'aug-off but with Mixup' as a variation independently of spatial aug.

Existing env vars still respected:
  - `TIER1_AUG_OFF=1` forces the empty spatial AugPolicy (helpers.py logic).
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from helpers import (
    AugmentedTensorDataset,
    aug_policy_for,
    per_channel_stats,
)


class DataProcessor:
    def __init__(self, train_x, train_y, valid_x, valid_y, test_x, metadata, clock):
        self.train_x = train_x
        self.train_y = train_y
        self.valid_x = valid_x
        self.valid_y = valid_y
        self.test_x = test_x
        self.metadata = metadata
        self.clock = clock
        self.batch_size = 128

    @staticmethod
    def _to_nchw(x: np.ndarray) -> np.ndarray:
        # Some datasets (Sudoku) ship 3D arrays even though metadata claims 4D.
        if x.ndim == 3:
            x = x[:, None, :, :]
        return x

    def _normalise(self, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
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

        policy = aug_policy_for(self.metadata)
        # Excursion override: LANG_MIXUP_ALPHA env var (default 0.0) replaces
        # the metadata mixup α so we can test 'aug-off + Mixup' variations
        # without changing helpers.py.
        mixup_alpha = float(os.environ.get("LANG_MIXUP_ALPHA",
                                           str(policy.mixup_alpha)))
        print(f"  aug policy: {policy.describe()}  |  mixup_α={mixup_alpha}")

        self.metadata["mixup_alpha"] = mixup_alpha

        # Train dataset: apply input-side augmentation per __getitem__ if any
        # aug is enabled; otherwise fall back to plain TensorDataset (no-op
        # __getitem__ overhead).
        if policy.any_input_aug():
            train_dataset = AugmentedTensorDataset(train_x, train_y, policy)
        else:
            train_dataset = TensorDataset(train_x, train_y)

        train_loader = DataLoader(
            train_dataset,
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
        test_dataset = TensorDataset(test_x)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            sampler=SequentialSampler(test_dataset),
            drop_last=False,
            num_workers=0,
        )
        return train_loader, valid_loader, test_loader
