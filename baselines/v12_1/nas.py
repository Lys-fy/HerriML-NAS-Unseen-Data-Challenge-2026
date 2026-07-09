"""v11.1 NAS: the measure-gate-train pipeline (v10 + verified attention probe).

v11.1 = v10 with one change: stage 5's fragile random arm is replaced by a
NARROW, verified spatial-self-attention probe for board/encoded data -- the one
architectural lever confirmed three independent ways (v10 random arm, v11.2
hill-climb, v11.3 evolution all find early attention worth ~+4 on Chesseract).
Everything else (fingerprint routing, family gate, family-matched aug probe,
default-incumbent depth, train-to-budget with cosine-to-0) is v10 unchanged.

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
RAND_FRAC = float(os.environ.get("RAND_FRAC", "0.12"))  # random arm, encoded only

# v12.1 probe floors/caps: pure clock FRACTIONS made probes indecisive at
# small budgets (Chesseract attn verify missed adoption by 0.0006 on the V100)
# and wasteful at big ones (8h -> 19min per probe leg). The floor applies only
# when the search budget affords it; the cap binds only above ~2h, so the
# validated <=2h behavior is unchanged.
PROBE_LO_S = float(os.environ.get("V12_PROBE_LO_S", "120"))
PROBE_HI_S = float(os.environ.get("V12_PROBE_HI_S", "420"))
PROBE_FLOOR_BUDGET_S = float(os.environ.get("V12_PROBE_FLOOR_BUDGET_S", str(4 * 3600)))

# low-budget collapse thresholds (seconds remaining at search start)
# env-overridable for testing the non-collapse path at small budgets
COLLAPSE_FULL_S = int(os.environ.get("V11_COLLAPSE_FULL_S", 35 * 60))  # below: no gate/depth/attn
COLLAPSE_AUG_S = int(os.environ.get("V11_COLLAPSE_AUG_S", 15 * 60))    # below: not even the aug probe

DEFAULT_DEPTH = 6             # the validated incumbent (fixed tcn2d D=6 = +59.2)
DEPTH_CANDS = (4, 8)
DEFAULT_WIDTH = 48           # the validated incumbent width
WIDTH_CANDS = (32, 64)
# Verified width-search stage (default-incumbent W=48 vs candidates, short-train
# + re-train verification before adopting) -- the systematic, protected version
# of "search width again" (width-prune cratered in v9 only because it had no
# verification). Env-gated; default OFF so the submission/main v10 is unchanged.
WIDTH_SEARCH = os.environ.get("WIDTH_SEARCH", "0") == "1"
WIDTH_FRAC = float(os.environ.get("WIDTH_FRAC", "0.12"))
# Real "macro" search (BLOX sense): per-stage HETEROGENEOUS ops. Split the net
# into MACRO_STAGES contiguous stages; greedily swap each stage's op to a
# candidate, default-incumbent + verify before adopting (never below the
# uniform arch). Env-gated, default OFF; a long-budget v11 experiment. The
# candidate ops are a first guess, to be refined from the arch gauntlet.
MACRO_SEARCH = os.environ.get("MACRO_SEARCH", "0") == "1"
MACRO_FRAC = float(os.environ.get("MACRO_FRAC", "0.25"))
MACRO_STAGES = int(os.environ.get("MACRO_STAGES", "3"))
MACRO_DEFAULT_OP = os.environ.get("MACRO_DEFAULT_OP", "DilatedConv3")
MACRO_OPS = [o for o in os.environ.get("MACRO_OPS", "SpatialSelfAttn2D,SEConv3").split(",") if o]
STEM_EMBED_MIN_CH = 8         # A5a: channel-embed wins on the dilated family
ENCODED_CORR1 = 0.2           # fingerprint gap is 0.07 vs 0.38 (5x)
RAND_K = int(os.environ.get("RAND_K", "6"))
# random-arm width pool (env-configurable for the micro-search-expansion study;
# default unchanged). Verification (added 2026-06-15) makes a bigger search
# safe to revisit -- a short-train mirage no longer gets adopted.
RAND_WIDTHS = [int(x) for x in os.environ.get("RAND_WIDTHS", "16,24,32,48").split(",")]
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

    @staticmethod
    def _slice(t0: float, frac: float, parts: float) -> float:
        """Per-leg probe budget: clock fraction with a decisiveness FLOOR (only
        when the total budget affords it) and a waste CAP (binds only >~2h)."""
        s = t0 * frac / parts
        if t0 > PROBE_FLOOR_BUDGET_S:
            s = max(s, PROBE_LO_S)
        return min(s, PROBE_HI_S)

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
        alt_family = None          # v12.1 heterogeneous-seed candidate
        gate_margin = None
        if encoded and not collapse:
            arm_budget = self._slice(t0, GATE_FRAC, 3.0)
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
            # v12.1: when the family race is INSIDE the noise band, remember the
            # best non-picked family -- the trainer can spend one ensemble seed
            # on it (heterogeneous seed) and let validation settle the coin flip.
            # Evidence: 4 GPUs picked 3 different families on Chesseract, worth
            # ~5pp; the gate margins there (0.007-0.027) were all sub-noise.
            others = {f: v for f, v in gate_best.items() if f != family}
            if others:
                alt_family = max(others, key=others.get)
                gate_margin = abs(gate_best[family] - others[alt_family])
                if gate_margin >= noise:
                    alt_family = None
            self._log("backbone_gate", arms={k: round(v, 4) for k, v in
                                             gate_best.items()},
                      noise=round(noise, 4), leader=leader, picked=family,
                      alt_family=alt_family,
                      gate_margin=round(gate_margin, 4) if alt_family else None)
        else:
            self._log("backbone_skip", picked=family,
                      reason="image-like" if not encoded else "low budget")

        # ---- stage 3: aug probe, family-matched arms --------------------------
        ds = self.train_loader.dataset
        policy = getattr(ds, "policy", None)
        if policy is not None and policy.any_input_aug() \
                and self.clock.check() > COLLAPSE_AUG_S:
            saved = (policy.flip, policy.crop_pad, policy.cutout_size)
            arm_budget = self._slice(t0, AUG_FRAC, 2.0)
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
            # Asymmetric, conservative rule (the validated P1 shape): aug is
            # regularization, so at probe horizons OFF leads even where ON
            # pays at full budget (A6: full-size tcn2d arms at 0.3h voted OFF
            # broadly, contradicting known-good 2h ON results). Deviate from
            # the metadata default (ON) only on a CLEAR off-margin.
            off_margin = float(os.environ.get("AUG_OFF_MARGIN", "0.02"))
            use_aug = results["off"] - results["on"] < off_margin
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
            cand_s = self._slice(t0, DEPTH_FRAC, len(DEPTH_CANDS) + 1.5)  # shorts + verify
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

        # ---- stage 4b: width, default-incumbent (optional, env-gated) ----------
        width = DEFAULT_WIDTH
        if WIDTH_SEARCH and not collapse and family != "tcn1d":
            slice_s = t0 * WIDTH_FRAC
            cand_s = slice_s / (len(WIDTH_CANDS) + 1.5)  # shorts + verify
            model = self._build(family, in_ch, nc, h, w, depth=depth,
                                width=DEFAULT_WIDTH)
            vals = self._timed_train(model, device, cand_s)
            inc_w = max(vals)
            wnoise = self._noise(vals)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            best_wc, best_wv = None, inc_w
            wres = {}
            for wd in WIDTH_CANDS:
                model = self._build(family, in_ch, nc, h, w, depth=depth, width=wd)
                try:
                    vals = self._timed_train(model, device, cand_s)
                    v = max(vals)
                except Exception:
                    v = 0.0
                wres[wd] = round(v, 4)
                if v > best_wv + wnoise:
                    best_wc, best_wv = wd, v
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            wverified = None
            if best_wc is not None:
                # verification: re-train the challenger fresh, longer horizon
                model = self._build(family, in_ch, nc, h, w, depth=depth, width=best_wc)
                try:
                    vv = self._timed_train(model, device, cand_s * 1.5)
                    wverified = round(max(vv), 4)
                except Exception:
                    wverified = 0.0
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if wverified > inc_w + wnoise:
                    width = best_wc
            self._log("width_search", incumbent=round(inc_w, 4), candidates=wres,
                      noise=round(wnoise, 4), challenger=best_wc,
                      verified=wverified, picked=width)
        else:
            self._log("width_search", skipped=True, picked=width)

        # ---- stage 4c: per-stage macro (heterogeneous) search (env-gated) ------
        # The real "macro" search: let stages use DIFFERENT ops, found greedily
        # with default-incumbent + verify (worst case = the uniform arch, no
        # regression). Long-budget v11 experiment; default OFF.
        macro_pick = None
        if MACRO_SEARCH and not collapse and family != "tcn1d":
            mstem = "channel_embed" if in_ch >= STEM_EMBED_MIN_CH else "conv3"
            D = depth
            n_st = max(1, min(MACRO_STAGES, D))
            bounds = [round(i * D / n_st) for i in range(n_st + 1)]
            stage_ops = [MACRO_DEFAULT_OP] * D
            slice_s = t0 * MACRO_FRAC
            n_short = n_st * len(MACRO_OPS) + n_st * 1.5
            cand_s = slice_s / max(n_short, 1.0)
            model = build_cnn(in_ch, nc, h, w, ops=stage_ops, base_width=width,
                              drop_path_rate=0.0, stem=mstem)
            vals = self._timed_train(model, device, cand_s)
            inc = max(vals)
            mnoise = self._noise(vals)
            del model
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
                    model = build_cnn(in_ch, nc, h, w, ops=trial, base_width=width,
                                      drop_path_rate=0.0, stem=mstem)
                    try:
                        v = max(self._timed_train(model, device, cand_s))
                    except Exception:
                        v = 0.0
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if v > best_v + mnoise:
                        best_op, best_v = cop, v
                accepted = False
                if best_op != cur and best_v > inc + mnoise:
                    # verification: re-train the swapped arch fresh, longer horizon
                    trial = list(stage_ops)
                    for k in range(a, b):
                        trial[k] = best_op
                    model = build_cnn(in_ch, nc, h, w, ops=trial, base_width=width,
                                      drop_path_rate=0.0, stem=mstem)
                    try:
                        vv = max(self._timed_train(model, device, cand_s * 1.5))
                    except Exception:
                        vv = 0.0
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if vv > inc + mnoise:
                        for k in range(a, b):
                            stage_ops[k] = best_op
                        inc = vv
                        accepted = True
                stages_log.append({"stage": gi, "picked": stage_ops[a],
                                   "best_short": round(best_v, 4), "accepted": accepted})
            if any(o != MACRO_DEFAULT_OP for o in stage_ops):
                macro_pick = stage_ops
            self._log("macro_search", default_op=MACRO_DEFAULT_OP, n_stages=n_st,
                      cand_ops=list(MACRO_OPS), stages=stages_log,
                      final_ops=stage_ops, adopted=bool(macro_pick))
        elif MACRO_SEARCH:
            self._log("macro_search", skipped=True)

        # ---- stage 5: verified attention probe, encoded/board data only -------
        # v11.1: replace v10's fragile random arm with a NARROW, verified probe
        # for the one confirmed architectural lever -- early spatial self-attention
        # on board-like (encoded) data. Chesseract +4.3, confirmed THREE ways
        # (v10 random arm, v11.2 hill-climb, v11.3 evolution). Build the family
        # default as a uniform build_cnn stack, and a challenger with its first
        # ATTN_STAGES blocks replaced by SpatialSelfAttn2D; short-train both, then
        # VERIFY the challenger at a longer horizon, and adopt only if it beats the
        # default beyond noise -- so it can only help, never regress. The attention
        # op crashes on odd grids (e.g. Sedol 19x19) -> the try/except scores it
        # 0.0 and the family default is kept.
        attn_pick = None
        astem = "channel_embed" if in_ch >= STEM_EMBED_MIN_CH else "conv3"
        ATTN_STAGES = int(os.environ.get("V11_ATTN_STAGES", "2"))
        if encoded and not collapse and self.clock.check() > 0.45 * t0:
            base_op = "DilatedConv3" if family == "tcn2d" else "Conv3"
            cand_s = self._slice(t0, RAND_FRAC, 3.0)   # default-short + attn-short + verify
            # incumbent = the pipeline's ACTUAL chosen family default (tcn1d /
            # tcn2d / conv), trained at the SAME horizon as the challenger, so the
            # attention arm can only be adopted if it beats whatever the gate
            # picked -- never a regression regardless of the (noisy) family choice.
            try:
                inc_vals = self._timed_train(
                    self._build(family, in_ch, nc, h, w, depth=depth, width=width),
                    device, cand_s)
                inc_short, anoise = max(inc_vals), self._noise(inc_vals)
            except Exception:
                inc_short, anoise = (incumbent_short or 0.0), noise
            k = max(1, min(ATTN_STAGES, depth))
            # v12.1 probe hardening: a swallowed exception here cost 4.9pp on
            # Chesseract/H100 (verified=None -> silent reject). Retry once, and
            # on repeated failure substitute the shape-robust LinearAttn2D
            # (cannot crash on odd grids) instead of giving up. Every failure
            # is logged in the decision record, never silent.
            probe_err = None
            attn_ops = None
            attn_short = 0.0
            attn_op_used = None
            for op_name in ("SpatialSelfAttn2D", "SpatialSelfAttn2D", "LinearAttn2D"):
                cand_ops = [op_name] * k + [base_op] * (depth - k)
                try:
                    attn_short = max(self._timed_train(
                        build_cnn(in_ch, nc, h, w, ops=cand_ops, base_width=width,
                                  drop_path_rate=0.0, stem=astem), device, cand_s))
                    attn_ops, attn_op_used, probe_err = cand_ops, op_name, None
                    break
                except Exception as ex:
                    probe_err = type(ex).__name__
                    print(f"  v11.1 attn arm FAILED ({probe_err}), op={op_name}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            verified = None
            if attn_ops is not None and attn_short > inc_short + anoise:
                for _attempt in range(2):    # verification also gets one retry
                    try:
                        verified = round(max(self._timed_train(
                            build_cnn(in_ch, nc, h, w, ops=attn_ops, base_width=width,
                                      drop_path_rate=0.0, stem=astem),
                            device, cand_s * 1.5)), 4)
                        break
                    except Exception as ex:
                        probe_err = type(ex).__name__
                        verified = None
                        print(f"  v11.1 attn verify FAILED ({probe_err}), retrying")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if verified is not None and verified > inc_short + anoise:
                    attn_pick = (attn_ops, width, astem)
            self._log("attn_probe", base_op=base_op, attn_stages=k,
                      attn_op=attn_op_used, probe_error=probe_err,
                      incumbent=round(inc_short, 4), attn_short=round(attn_short, 4),
                      verified=verified, noise=round(anoise, 4),
                      adopted=bool(attn_pick))
        else:
            self._log("attn_probe", skipped=True)

        # ---- final build -------------------------------------------------------
        if attn_pick is not None:
            ops, W, astem = attn_pick
            final = build_cnn(in_ch, nc, h, w, ops=ops, base_width=W,
                              drop_path_rate=0.1, stem=astem)
            self._log("final", arch=f"attn[{ops.count('SpatialSelfAttn2D')}xAttn+{ops[-1]}] W={W}")
        elif macro_pick is not None:
            mstem = "channel_embed" if in_ch >= STEM_EMBED_MIN_CH else "conv3"
            final = build_cnn(in_ch, nc, h, w, ops=macro_pick, base_width=width,
                              drop_path_rate=0.1, stem=mstem)
            self._log("final", arch=f"macro[{'/'.join(macro_pick)}] W={width}")
        else:
            final = self._build(family, in_ch, nc, h, w, depth=depth, width=width)
            self._log("final", arch=f"{family} D={depth} W={width}")
        # v12.1 side-channel to the trainer (4-file contract untouched): when the
        # family gate was a sub-noise coin flip, hand over a BUILT runner-up-family
        # model so one ensemble seed can be spent on it, val-gated. Kept on CPU.
        alt_model = None
        if alt_family is not None and not collapse:
            try:
                alt_model = self._build(alt_family, in_ch, nc, h, w,
                                        depth=depth, width=width)
                self._log("alt_seed", family=alt_family,
                          gate_margin=gate_margin)
            except Exception as ex:
                self._log("alt_seed", family=alt_family,
                          probe_error=type(ex).__name__)
                alt_model = None
        final._v12_meta = {"family": family, "alt_family": alt_family,
                           "gate_margin": gate_margin, "gate_noise": noise,
                           "alt_model": alt_model}
        self._dump_record()
        return final
