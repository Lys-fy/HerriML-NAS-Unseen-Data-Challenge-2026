"""Macro-ONLY baseline: isolate the per-stage heterogeneous (macro) search and
ALWAYS train its result, so we measure the macro search's own standalone
performance on every dataset -- no falling back to "the family default was
better, train that instead".

Pipeline: fingerprint (for the stem only) -> per-stage greedy verified macro
search (uniform MACRO_DEFAULT_OP incumbent vs per-stage swaps to MACRO_OPS,
default-incumbent + verify) -> train the resulting arch to budget. If no swap
verifies, the arch stays uniform MACRO_DEFAULT_OP (still a macro-search result,
NOT the v10 family pick). Fixed depth/width (no depth/width/family search).
"""
import os
import json
import math
import time

import torch
import torch.nn as nn

from helpers import build_cnn

STEM_EMBED_MIN_CH = 8
MACRO_FRAC = float(os.environ.get("MACRO_FRAC", "0.30"))
MACRO_STAGES = int(os.environ.get("MACRO_STAGES", "3"))
MACRO_DEPTH = int(os.environ.get("MACRO_DEPTH", "6"))
MACRO_WIDTH = int(os.environ.get("MACRO_WIDTH", "48"))
MACRO_DEFAULT_OP = os.environ.get("MACRO_DEFAULT_OP", "DilatedConv3")
MACRO_OPS = [o for o in os.environ.get("MACRO_OPS", "SpatialSelfAttn2D,SEConv3,Conv5").split(",") if o]


class NAS:
    def __init__(self, train_loader, valid_loader, metadata, clock):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.metadata = metadata
        self.clock = clock

    def _shape(self):
        s = self.metadata["input_shape"]
        c, h, w = (s[1:] if len(s) == 4 else s)
        return int(c), int(h), int(w)

    @torch.no_grad()
    def _eval(self, model, device):
        model.eval()
        correct = total = 0
        for xb, yb in self.valid_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += yb.numel()
        return correct / max(total, 1)

    def _timed_train(self, model, device, budget_s, max_epochs=200):
        deadline = time.perf_counter() + budget_s
        model = model.to(device)
        crit = nn.CrossEntropyLoss(label_smoothing=0.1)
        opt = torch.optim.SGD(model.parameters(), lr=0.025, momentum=0.9,
                              weight_decay=3e-4, nesterov=True)
        vals, ep = [], 0
        while ep < max_epochs and time.perf_counter() < deadline:
            model.train()
            for xb, yb in self.train_loader:
                if time.perf_counter() > deadline:
                    break
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = crit(model(xb), yb)
                if not torch.isfinite(loss):
                    raise RuntimeError("NaN in macro probe")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            vals.append(self._eval(model, device))
            ep += 1
        return vals or [0.0]

    @staticmethod
    def _noise(vals):
        tail = vals[-10:]
        if len(tail) < 3:
            return 0.02
        m = sum(tail) / len(tail)
        return math.sqrt(sum((v - m) ** 2 for v in tail) / len(tail))

    def search(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        c, h, w = self._shape()
        nc = int(self.metadata["num_classes"])
        stem = "channel_embed" if c >= STEM_EMBED_MIN_CH else "conv3"
        D = MACRO_DEPTH
        n_st = max(1, min(MACRO_STAGES, D))
        bounds = [round(i * D / n_st) for i in range(n_st + 1)]
        stage_ops = [MACRO_DEFAULT_OP] * D

        t0 = self.clock.check()
        slice_s = t0 * MACRO_FRAC
        n_short = n_st * len(MACRO_OPS) + n_st * 1.5
        cand_s = slice_s / max(n_short, 1.0)

        m = build_cnn(c, nc, h, w, ops=stage_ops, base_width=MACRO_WIDTH,
                      drop_path_rate=0.0, stem=stem)
        inc_vals = self._timed_train(m, device, cand_s)
        inc = max(inc_vals)
        mnoise = self._noise(inc_vals)
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        stages_log = []
        for gi in range(n_st):
            a, b = bounds[gi], bounds[gi + 1]
            cur = stage_ops[a]
            best_op, best_v = cur, inc
            for cop in MACRO_OPS:
                if cop == cur:
                    continue
                trial = list(stage_ops)
                for k in range(a, b):
                    trial[k] = cop
                m = build_cnn(c, nc, h, w, ops=trial, base_width=MACRO_WIDTH,
                              drop_path_rate=0.0, stem=stem)
                try:
                    v = max(self._timed_train(m, device, cand_s))
                except Exception:
                    v = 0.0
                del m
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if v > best_v + mnoise:
                    best_op, best_v = cop, v
            accepted = False
            if best_op != cur and best_v > inc + mnoise:
                trial = list(stage_ops)
                for k in range(a, b):
                    trial[k] = best_op
                m = build_cnn(c, nc, h, w, ops=trial, base_width=MACRO_WIDTH,
                              drop_path_rate=0.0, stem=stem)
                try:
                    vv = max(self._timed_train(m, device, cand_s * 1.5))
                except Exception:
                    vv = 0.0
                del m
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if vv > inc + mnoise:
                    for k in range(a, b):
                        stage_ops[k] = best_op
                    inc = vv
                    accepted = True
            stages_log.append({"stage": gi, "picked": stage_ops[a],
                               "best_short": round(best_v, 4), "accepted": accepted})

        # ALWAYS train the macro arch (uniform if nothing verified) -- macro-only.
        final = build_cnn(c, nc, h, w, ops=stage_ops, base_width=MACRO_WIDTH,
                          drop_path_rate=0.1, stem=stem)
        log_dir = os.environ.get("BASELINE_LOG_DIR")
        if log_dir:
            try:
                os.makedirs(log_dir, exist_ok=True)
                code = self.metadata.get("codename", "unknown")
                rec = {"stages": [
                    {"stage": "macro_search", "default_op": MACRO_DEFAULT_OP,
                     "n_stages": n_st, "cand_ops": MACRO_OPS, "stages": stages_log,
                     "final_ops": stage_ops, "heterogeneous": any(o != MACRO_DEFAULT_OP for o in stage_ops)},
                    {"stage": "final", "arch": f"macro[{'/'.join(stage_ops)}] W={MACRO_WIDTH} D={D}"}],
                    "codename": code}
                with open(os.path.join(log_dir, f"{code}_decisions.json"), "w") as f:
                    json.dump(rec, f, indent=2)
            except Exception:
                pass
        print(f"  macro_probe: in={c}x{h}x{w} nc={nc} final_ops={stage_ops}")
        return final
