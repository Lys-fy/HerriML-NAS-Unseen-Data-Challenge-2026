"""Multi-branch ensemble probe (long-budget ensemble test).

Three backbone families (conv / 2-D TCN / 1-D TCN) in parallel, their LOGITS
averaged, trained jointly to budget by the shared Trainer. Tests whether a
cross-family ensemble beats the single best family v10 would pick, at the same
2h budget (the "what to do with surplus time at long budgets" question; E1
showed ensembles are conditionally, not universally, useful).

Logit-average (not softmax) so it composes with CrossEntropyLoss unchanged.
NAS.search() returns the EnsembleNet; the shared Trainer trains it.
"""
import torch
import torch.nn as nn

from helpers import build_default_cnn, TCN1D, TCN2D

STEM_EMBED_MIN_CH = 8


class EnsembleNet(nn.Module):
    def __init__(self, c, nc, h, w):
        super().__init__()
        stem = "channel_embed" if c >= STEM_EMBED_MIN_CH else "conv3"
        self.conv = build_default_cnn(c, nc, h, w, base_width=48, n_blocks=6,
                                      drop_path_rate=0.1)
        self.tcn2d = TCN2D(c, nc, h, w, base_width=48, n_blocks=6, stem=stem)
        self.tcn1d = TCN1D(c, nc, seq_len=h * w, base_width=48, n_blocks=8)

    def forward(self, x):
        return (self.conv(x) + self.tcn2d(x) + self.tcn1d(x)) / 3.0


class NAS:
    def __init__(self, train_loader, valid_loader, metadata, clock):
        self.metadata = metadata

    def search(self):
        s = self.metadata["input_shape"]
        c, h, w = (s[1:] if len(s) == 4 else s)
        nc = int(self.metadata["num_classes"])
        print(f"  ensemble_probe: conv+tcn2d+tcn1d logit-avg, in={c}x{h}x{w} classes={nc}")
        return EnsembleNet(int(c), nc, int(h), int(w))
