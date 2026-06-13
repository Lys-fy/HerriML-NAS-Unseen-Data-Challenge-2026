"""ResNet-18 baseline (no search).

The competition's official datasets use ResNet-18 accuracy as the benchmark
zero-point in the adjusted-score formula. Our custom datasets ship
`benchmark: 0.0` (placeholder), so this baseline produces the missing
zero-point: a standard ResNet-18 trained to the same budget as v10.

Built from scratch (torchvision is not installed on the runner). The stem
adapts to the input size so the network is not degenerate on the small
encoded inputs: a CIFAR-style 3x3 stem (no max-pool) for max(H,W) <= 48,
the standard 7x7 stride-2 + max-pool stem otherwise. The final
AdaptiveAvgPool handles any remaining spatial size.

Interface matches the harness: NAS(...).search() returns the model; the
shared Trainer (copied from v10) trains it to budget.
"""
import os
import json
from typing import Dict

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + idt)


class ResNet18(nn.Module):
    def __init__(self, in_ch, num_classes, small_stem):
        super().__init__()
        if small_stem:
            # CIFAR-style stem: 3x3 stride-1, no max-pool.
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, 64, 3, 1, 1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1),
            )
        self.layer1 = self._make(64, 64, 2, 1)
        self.layer2 = self._make(64, 128, 2, 2)
        self.layer3 = self._make(128, 256, 2, 2)
        self.layer4 = self._make(256, 512, 2, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make(self, in_ch, out_ch, blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class NAS:
    def __init__(self, train_loader, valid_loader, metadata, clock):
        self.metadata = metadata
        self.clock = clock

    def _input_shape(self):
        shape = self.metadata["input_shape"]
        c, h, w = (shape[1:] if len(shape) == 4 else shape)
        return int(c), int(h), int(w)

    def search(self):
        c, h, w = self._input_shape()
        nc = int(self.metadata["num_classes"])
        small_stem = max(h, w) <= 48
        model = ResNet18(c, nc, small_stem)
        # Record the (trivial) decision for parity with the v10 logs.
        log_dir = os.environ.get("BASELINE_LOG_DIR")
        if log_dir:
            try:
                os.makedirs(log_dir, exist_ok=True)
                code = self.metadata.get("codename", "unknown")
                rec = {"stages": [{"stage": "final", "arch": "resnet18",
                                   "stem": "cifar" if small_stem else "standard",
                                   "in_shape": [c, h, w], "classes": nc}],
                       "codename": code}
                with open(os.path.join(log_dir, f"{code}_decisions.json"), "w") as f:
                    json.dump(rec, f, indent=2)
            except Exception:
                pass
        print(f"  resnet18: in={c}x{h}x{w} classes={nc} "
              f"stem={'cifar' if small_stem else 'standard'}")
        return model
