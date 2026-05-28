"""Tier 2a / P1 DataProcessor.

Same as baselines/tier2b_egnas_full/data_processor.py PLUS the P1
aug-safety probe: between normalisation and DataLoader construction the
probe runs (when applicable) and the probe's decision overrides the
metadata-derived AugPolicy. Probe info is dumped as
`<BASELINE_LOG_DIR>/<codename>_p1_probe.json` for traceability.

Probe is skipped when the base policy already implies no aug (C != 1, 3)
or when `P1_PROBE=0` (ablation switch — turns the baseline into
byte-equivalent EGNAS-full).
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from helpers import (
    AugPolicy,
    AugmentedTensorDataset,
    aug_policy_for,
    per_channel_stats,
    probe_aug_safety,
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

        base_policy = aug_policy_for(self.metadata)
        policy = base_policy

        # ----- P1 aug-safety probe ---------------------------------------
        # Only runs when there's actually an aug decision to make and the
        # probe isn't disabled. The probe is deterministic across arms
        # (same subset, same seed), uses fewer ops than the main training,
        # and writes its verdict to <log_dir>/<codename>_p1_probe.json for
        # later analysis.
        if (base_policy.any_input_aug()
                and os.environ.get("P1_PROBE", "1") == "1"):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            use_aug, probe_info = probe_aug_safety(
                train_x, train_y, valid_x, valid_y,
                metadata=self.metadata, device=device,
                batch_size=self.batch_size,
            )
            probe_info["base_policy"] = base_policy.describe()
            if not use_aug:
                policy = AugPolicy()  # override to empty
            log_dir = os.environ.get("BASELINE_LOG_DIR")
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                codename = self.metadata.get("codename", "unknown")
                with open(os.path.join(log_dir, f"{codename}_p1_probe.json"),
                          "w") as f:
                    json.dump(probe_info, f, indent=2)
            print(f"  P1 probe: AUG {'ON' if use_aug else 'OFF'}  "
                  f"(val_on={probe_info.get('val_aug_on', float('nan')):.3f}, "
                  f"val_off={probe_info.get('val_aug_off', float('nan')):.3f}; "
                  f"gap_on={probe_info.get('gap_aug_on', 0.0):.3f}, "
                  f"gap_off={probe_info.get('gap_aug_off', 0.0):.3f}; "
                  f"Δscore={probe_info.get('delta_pp_score', 0.0):+.2f}pp, "
                  f"margin={probe_info['margin_pp']:.2f}pp)")
        print(f"  aug policy: {policy.describe()}")

        # Stash the mixup α on metadata so Trainer can pick it up. Setting to 0
        # disables mixup entirely (no extra cost in the train loop).
        self.metadata["mixup_alpha"] = float(policy.mixup_alpha)

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
