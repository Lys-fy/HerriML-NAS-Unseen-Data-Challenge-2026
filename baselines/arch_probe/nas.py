"""Family-admission gauntlet: fixed-arch probes for candidate architecture
families, selected by the PROBE_ARCH env var. No search - NAS.search() returns
the fixed arch and the shared Trainer trains it to budget, so we can see where
each family beats the current defaults (conv / tcn2d) and decide whether to
admit it as a gated candidate at large budgets.

Families (all from ops already in the vocab, so no new untested blocks):
  attn   - SpatialSelfAttn2D (the board-game lead: Sudoku s42, Chesseract s1337)
  recur  - RowBiLSTM (the recurrent/GRU family; cracked Language in v7)
  modern - Conv7 + SEConv3 + ResNeXt (a ConvNeXt-ish large-kernel/SE/grouped block)
All depth-6, width-48, channel-embed stem for >=8-channel inputs (v10 rule).
"""
import os
import json

from helpers import build_cnn

STEM_EMBED_MIN_CH = 8
ARCHS = {
    "attn":   ["SpatialSelfAttn2D", "Conv3", "Conv3", "SpatialSelfAttn2D", "Conv3", "Conv3"],
    "recur":  ["RowBiLSTM", "Conv3", "Conv3", "RowBiLSTM", "Conv3", "Conv3"],
    "modern": ["Conv7", "SEConv3", "ResNeXt", "Conv7", "SEConv3", "ResNeXt"],
}
PROBE = os.environ.get("PROBE_ARCH", "attn")
OPS = ARCHS[PROBE]


class NAS:
    def __init__(self, train_loader, valid_loader, metadata, clock):
        self.metadata = metadata

    def _shape(self):
        s = self.metadata["input_shape"]
        c, h, w = (s[1:] if len(s) == 4 else s)
        return int(c), int(h), int(w)

    def search(self):
        c, h, w = self._shape()
        nc = int(self.metadata["num_classes"])
        stem = "channel_embed" if c >= STEM_EMBED_MIN_CH else "conv3"
        model = build_cnn(c, nc, h, w, ops=OPS, base_width=48,
                          drop_path_rate=0.1, stem=stem)
        log_dir = os.environ.get("BASELINE_LOG_DIR")
        if log_dir:
            try:
                os.makedirs(log_dir, exist_ok=True)
                code = self.metadata.get("codename", "unknown")
                rec = {"stages": [{"stage": "final",
                                   "arch": f"{PROBE}[" + ",".join(OPS) + "] W=48",
                                   "stem": stem, "in_shape": [c, h, w], "classes": nc}],
                       "codename": code, "family": PROBE}
                with open(os.path.join(log_dir, f"{code}_decisions.json"), "w") as f:
                    json.dump(rec, f, indent=2)
            except Exception:
                pass
        print(f"  arch_probe[{PROBE}]: in={c}x{h}x{w} classes={nc} stem={stem} ops={OPS}")
        return model
