"""v10 NAS: the measure-gate-train pipeline.

Phase-A-validated routing (see notes: v10 - Design + Idea Ledger):

  1. FINGERPRINT (computed in DataProcessor, flags in metadata):
     corr1 < 0.2  -> encoded data, the image prior is wrong.
  2. BACKBONE:
     image-like -> TCN-2d, NO bake-off (the skip is load-bearing: gating
     every dataset falls into the Myofibre/GeoClassing early-leader traps;
     A1 totals: fixed +59.2, gate-everywhere +58.5, fingerprint+gate +61.2).
     encoded    -> 3-arm gate (conv / tcn1d / tcn2d short-train at matched
     budget); pick the leader only if it beats the CONV arm by more than
     the measured noise, else abstain to conv (the robust encoded default).
  3. AUG PROBE moved here, AFTER the backbone choice, with arms of the
     chosen family (A5b: the conv-derived decision does NOT transfer:
     Language tcn1d aug-on 89.3 vs probe's aug-off 86.6).
  4. DEPTH: default-incumbent. The family default (D=6) is the incumbent;
     candidates {4, 8} must beat it by more than noise at short-train AND
     survive verification at a longer horizon (v9: 13 verification fires,
     real rescues; v9's grow-from-minimum mis-picked depth on encoded data
     in both directions, the incumbent rule blocks that).
  5. RANDOM ARM: encoded data only, only if budget remains: K wide samples
     (validated: Chesseract +0.90 vs conv +0.30, all seeds; loses on
     normal data, hence gated).
  6. Trainer trains the winner to the full remaining budget (LR -> 0).

Every stage appends to a decision record:
  <BASELINE_LOG_DIR>/<codename>_decisions.json

Budget fractions of the clock remaining at search start; at low budgets
the pipeline collapses to fingerprint + default backbone + train-to-budget
(the metadata time_limit default is 30 MINUTES; Phase 3 budget is unknown).
"""

from __future__ import annotations

import json
import math
import os
import random
import time
import traceback
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from helpers import (
    OP_NAMES,
    TCN1D,
    TCN2D,
    build_cnn,
    build_default_cnn,
)

# ---- budget fractions (of clock remaining at search start) -----------------
GATE_FRAC = 0.18          # 3-arm bake-off, encoded data only
AUG_FRAC = 0.08           # two-arm aug probe, family-matched
DEPTH_FRAC = 0.14         # 2 candidate shorts + 1 verification
RAND_FRAC = 0.12          # random arm, encoded only

# low-budget collapse thresholds (seconds remaining at search start)
COLLAPSE_FULL_S = 35 * 60     # below this: no gate, no depth search, no random
COLLAPSE_AUG_S = 15 * 60      # below this: not even the aug probe

DEFAULT_DEPTH = 6             # the validated incumbent (fixed tcn2d D=6 = +59.2)
DEPTH_CANDS = (4, 8)
STEM_EMBED_MIN_CH = 8         # A5a: channel-embed wins on the dilated family
ENCODED_CORR1 = 0.2           # fingerprint gap is 0.07 vs 0.38 (5x)
RAND_K = 6
WIDE_VOCAB = tuple(o for o in OP_NAMES
                   if o not in ("RowBiLSTM", "CrossAxisBiLSTM"))


class NAS:
    def __init__(self, train_loader, valid_loader, metadata, clock):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.metadata = metadata
        self.clock = clock
        self.record: Dict = {"stages": []}

    # ---- plumbing -----------------------------------------------------------

    def _input_shape(self):
        shape = self.metadata["input_shape"]
        if len(shape) == 4:
            _, c, h, w = shape
        else:
            c, h, w = shape
        return int(c), int(h), int(w)

    def _log(self, stage: str, **kw):
        entry = {"stage": stage, "t_remaining_s": round(self.clock.check(), 1)}
        entry.update(kw)
        self.record["stages"].append(entry)
        brief = {k: v for k, v in kw.items() if not isinstance(v, (list, dict))}
        print(f"  v10 {stage}: {brief}")

    def _dump_record(self):
        log_dir = os.environ.get("BASELINE_LOG_DIR")
        if not log_dir:
            return
        try:
            os.makedirs(log_dir, exist_ok=True)
            code = self.metadata.get("codename", "unknown")
            with open(os.path.join(log_dir, f"{code}_decisions.json"), "w") as f:
                json.dump(self.record, f, indent=2)
        except Exception:
            pass

    @torch.no_grad()
    def _eval(self, model, device) -> float:
        model.eval()
        correct = total = 0
        for xb, yb in self.valid_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            correct += (model(xb).argmax(dim=1) == yb).sum().item()
            total += yb.numel()
        return correct / max(total, 1)

    def _timed_train(self, model, device, budget_s: float,
                     max_epochs: int = 200) -> List[float]:
        """Train for up to budget_s wall-clock; eval after each epoch.
        Returns the per-epoch val series (for leader + noise estimates)."""
        deadline = time.perf_counter() + budget_s
        model = model.to(device)
        crit = nn.CrossEntropyLoss(label_smoothing=0.1)
        opt = torch.optim.SGD(model.parameters(), lr=0.025, momentum=0.9,
                              weight_decay=3e-4, nesterov=True)
        vals: List[float] = []
        epoch = 0
        while epoch < max_epochs and time.perf_counter() < deadline:
            model.train()
            for xb, yb in self.train_loader:
                if time.perf_counter() > deadline:
                    break
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = crit(model(xb), yb)
                if not torch.isfinite(loss):
                    raise RuntimeError("NaN in v10 probe train")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            vals.append(self._eval(model, device))
            epoch += 1
        return vals or [0.0]

    @staticmethod
    def _noise(vals: List[float]) -> float:
        tail = vals[-10:]
        if len(tail) < 3:
            return 0.02
        m = sum(tail) / len(tail)
        return math.sqrt(sum((v - m) ** 2 for v in tail) / len(tail))

    # ---- model builders -----------------------------------------------------

    def _build(self, family: str, in_ch, nc, h, w, depth: int = DEFAULT_DEPTH,
               width: int = 48):
        stem = "channel_embed" if in_ch >= STEM_EMBED_MIN_CH else "conv3"
        if family == "conv":
            return build_default_cnn(in_ch, nc, h, w, base_width=width,
                                     n_blocks=depth, drop_path_rate=0.1)
        if family == "tcn1d":
            return TCN1D(in_ch, nc, seq_len=h * w, base_width=width, n_blocks=8)
        if family == "tcn2d":
            return TCN2D(in_ch, nc, h, w, base_width=width, n_blocks=depth,
                         stem=stem)
        raise ValueError(family)

    def _build_small(self, family: str, in_ch, nc, h, w):
        """Aug-probe arm: a small net OF THE CHOSEN FAMILY (A5b)."""
        if family == "conv":
            return build_default_cnn(in_ch, nc, h, w, base_width=16,
                                     n_blocks=3, drop_path_rate=0.0)
        if family == "tcn1d":
            return TCN1D(in_ch, nc, seq_len=h * w, base_width=16, n_blocks=4)
        stem = "channel_embed" if in_ch >= STEM_EMBED_MIN_CH else "conv3"
        return TCN2D(in_ch, nc, h, w, base_width=16, n_blocks=3, stem=stem)

    # ---- the pipeline -------------------------------------------------------

    def search(self):
        try:
            return self._search_inner()
        except Exception as ex:
            traceback.print_exc()
            self._log("FALLBACK", error=f"{type(ex).__name__}: {ex}")
            self._dump_record()
            in_ch, h, w = self._input_shape()
            nc = int(self.metadata["num_classes"])
            return self._build("tcn2d", in_ch, nc, h, w)

    def _search_inner(self):
        in_ch, h, w = self._input_shape()
        nc = int(self.metadata["num_classes"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t0 = self.clock.check()
        corr1 = float(self.metadata.get("fp_corr1", 1.0))
        encoded = bool(self.metadata.get("fp_encoded", corr1 < ENCODED_CORR1))
        collapse = t0 < COLLAPSE_FULL_S
        self.record.update(dict(codename=self.metadata.get("codename"),
                                in_shape=[in_ch, h, w], classes=nc,
                                t_at_search_s=round(t0, 1)))
        self._log("fingerprint", corr1=corr1, encoded=encoded,
                  collapse=collapse,
                  stem="channel_embed" if in_ch >= STEM_EMBED_MIN_CH else "conv3")

        # ---- stage 2: backbone ------------------------------------------------
        # defaults: image-like -> tcn2d; encoded under collapse -> conv
        # (the robust encoded default, same fallback the gate abstains to)
        family = "conv" if (encoded and collapse) else "tcn2d"
        gate_best: Dict[str, float] = {}
        noise = 0.02
        if encoded and not collapse:
            arm_budget = t0 * GATE_FRAC / 3.0
            noises = []
            for fam in ("conv", "tcn1d", "tcn2d"):
                model = self._build(fam, in_ch, nc, h, w)
                try:
                    vals = self._timed_train(model, device, arm_budget)
                except Exception as ex:
                    vals = [0.0]
                    print(f"  v10 gate arm {fam} FAILED ({type(ex).__name__})")
                gate_best[fam] = max(vals)
                noises.append(self._noise(vals))
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            noise = sum(noises) / len(noises)
            leader = max(gate_best, key=gate_best.get)
            # abstain-to-default: deviate from conv only when clearly better
            family = leader if gate_best[leader] - gate_best["conv"] > noise \
                else "conv"
            self._log("backbone_gate", arms={k: round(v, 4) for k, v in
                                             gate_best.items()},
                      noise=round(noise, 4), leader=leader, picked=family)
        else:
            self._log("backbone_skip", picked=family,
                      reason="image-like" if not encoded else "low budget")

        # ---- stage 3: aug probe, family-matched arms --------------------------
        ds = self.train_loader.dataset
        policy = getattr(ds, "policy", None)
        if policy is not None and policy.any_input_aug() \
                and self.clock.check() > COLLAPSE_AUG_S:
            saved = (policy.flip, policy.crop_pad, policy.cutout_size)
            arm_budget = t0 * AUG_FRAC / 2.0
            results = {}
            for arm in ("on", "off"):
                if arm == "off":
                    policy.flip, policy.crop_pad, policy.cutout_size = \
                        False, 0, 0
                else:
                    (policy.flip, policy.crop_pad,
                     policy.cutout_size) = saved
                model = self._build_small(family, in_ch, nc, h, w)
                try:
                    vals = self._timed_train(model, device, arm_budget)
                    results[arm] = max(vals)
                except Exception:
                    results[arm] = 0.0
                del model
            use_aug = results["on"] >= results["off"]
            if use_aug:
                (policy.flip, policy.crop_pad, policy.cutout_size) = saved
                self.metadata["mixup_alpha"] = 0.0
            else:
                policy.flip, policy.crop_pad, policy.cutout_size = False, 0, 0
                self.metadata["mixup_alpha"] = float(
                    os.environ.get("P1_MIXUP_ALPHA", "0.2"))
            self._log("aug_probe", family=family,
                      on=round(results["on"], 4), off=round(results["off"], 4),
                      decision="on" if use_aug else "off+mixup")
        else:
            self._log("aug_probe", skipped=True,
                      reason="no aug applicable" if policy is None or
                      not policy.any_input_aug() else "low budget")

        # ---- stage 4: depth, default-incumbent --------------------------------
        depth = DEFAULT_DEPTH
        incumbent_short: Optional[float] = gate_best.get(family)
        if not collapse and family != "tcn1d":
            slice_s = t0 * DEPTH_FRAC
            cand_s = slice_s / (len(DEPTH_CANDS) + 1.5)  # shorts + verify
            if incumbent_short is None:
                model = self._build(family, in_ch, nc, h, w, depth=DEFAULT_DEPTH)
                vals = self._timed_train(model, device, cand_s)
                incumbent_short = max(vals)
                noise = self._noise(vals)
                del model
            best_cand, best_val = None, incumbent_short
            cand_results = {}
            for d in DEPTH_CANDS:
                model = self._build(family, in_ch, nc, h, w, depth=d)
                try:
                    vals = self._timed_train(model, device, cand_s)
                    v = max(vals)
                except Exception:
                    v = 0.0
                cand_results[d] = round(v, 4)
                if v > best_val + noise:
                    best_cand, best_val = d, v
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            verified = None
            if best_cand is not None:
                # verification: re-train the challenger fresh, longer horizon
                # (v9 evidence: short-train alone mis-picks depth; 13 fires)
                model = self._build(family, in_ch, nc, h, w, depth=best_cand)
                try:
                    vvals = self._timed_train(model, device, cand_s * 1.5)
                    verified = round(max(vvals), 4)
                except Exception:
                    verified = 0.0
                del model
                if verified > incumbent_short + noise:
                    depth = best_cand
            self._log("depth_search", incumbent=round(incumbent_short or 0, 4),
                      candidates=cand_results, noise=round(noise, 4),
                      challenger=best_cand, verified=verified, picked=depth)
        else:
            self._log("depth_search", skipped=True, picked=depth)

        # ---- stage 5: random arm, encoded only --------------------------------
        rand_pick = None
        if encoded and not collapse and self.clock.check() > 0.45 * t0:
            slice_s = t0 * RAND_FRAC
            cand_s = slice_s / RAND_K
            rng = random.Random(int(os.environ.get("BASELINE_SEED", "42")))
            base = incumbent_short or 0.0
            best_r, best_cfg = base, None
            samples = []
            for _ in range(RAND_K):
                D = rng.randint(3, 12)
                W = rng.choice([16, 24, 32, 48])
                ops = [rng.choice(WIDE_VOCAB) for _ in range(D)]
                model = build_cnn(in_ch, nc, h, w, ops=ops, base_width=W,
                                  drop_path_rate=0.0)
                try:
                    vals = self._timed_train(model, device, cand_s)
                    v = max(vals)
                except Exception:
                    v = 0.0
                samples.append(round(v, 4))
                if v > best_r + 2 * noise:
                    best_r, best_cfg = v, (ops, W)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if best_cfg is not None:
                rand_pick = best_cfg
            self._log("random_arm", samples=samples,
                      incumbent=round(base, 4), adopted=bool(rand_pick),
                      ops=list(rand_pick[0]) if rand_pick else None)
        else:
            self._log("random_arm", skipped=True)

        # ---- final build -------------------------------------------------------
        if rand_pick is not None:
            ops, W = rand_pick
            final = build_cnn(in_ch, nc, h, w, ops=ops, base_width=W,
                              drop_path_rate=0.1)
            self._log("final", arch=f"random[{len(ops)} ops W={W}]")
        else:
            final = self._build(family, in_ch, nc, h, w, depth=depth)
            self._log("final", arch=f"{family} D={depth} W=48")
        self._dump_record()
        return final
