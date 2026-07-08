# BDH-harvest build plan (persisted handoff — 2026-07-01)

Building 5 improvements harvested from the Dragon Hatchling paper (see
`docs/notes/08 - References/papers/2025 - Dragon Hatchling (BDH).md`), then ONE
test queue, then decide 3-seed × multi-interval on the winners. **Current queue
INTERRUPTED (GPU free).** BDH results so far kept: Sudoku 96.21% (T=81),
Language stopped mid-run at val 0.82 (superseded — will redo with B1/B2).

## Build list

### C1 — `LinearAttn2D` op  [add to `baselines/v10/helpers.py` OP_VOCAB + v11_1 copy]
Bidirectional linear-attention spatial mixer, O(HW) (vs `SpatialSelfAttn2D` O((HW)²), which also crashed on odd grids). Module signature `(in_ch, out_ch, stride)` like other ops (match `_SpatialSelfAttn2D` for stride/channel handling):
```
q=relu(to_q(x)); k=relu(to_k(x)); v=to_v(x)         # 1x1 convs; to_q/to_k -> r*C (r=2)
Q,K,V flatten to (B,HW,rC),(B,HW,rC),(B,HW,C)
S = K.transpose(1,2) @ V                             # (B, rC, C) global context, ONCE
ctx = (Q @ S) / (Q @ K.sum(1,keepdim=True).transpose(1,2) + eps)   # (B,HW,C)
out = proj(unflatten(ctx)); return bn(residual + out)
```
Register `"LinearAttn2D"`. **Shares its kernel with B1.**

### C2 — `SparseGated` op  [helpers.py OP_VOCAB]
ReLU·ReLU-gated inverted bottleneck + learned negative bias (denoiser):
```
y = down( relu(up1(x)) * relu(up2(x)) )   # up1 has bias (learned neg threshold), r=3
return relu(bn(residual + y))
```
Register `"SparseGated"`.

### C4 — weight-shared depth  [macro knob in `build_cnn`]
Apply each op-block ×k reusing the SAME module instance (shared weights → many effective layers, few params). Add `build_cnn(..., shared_repeat=k)`: build block once, apply k times in forward where in_ch==out_ch (no downsample within the repeat). Env `V11_SHARED_REPEAT`.

### B1 + B2 — `dev/bdh_bidir.py` (subclass of vendored BDH; DO NOT edit `baselines/bdh/bdh.py`)
- **B1:** bidirectional linear attention — replace `Attention.forward`'s `(QR@KR.mT).tril(-1)@V` with associative `QR @ (KR.mT @ V)` (no causal mask) → O(T), and lets medium-T datasets (Cryptic 4608, AddNIST/MultNIST 2352) become viable.
- **B2:** all-layer readout — accumulate `logits += x_l @ lm_head` across all L layers (not just last).
- Wire into `dev/bdh_excursion.py` via a `--model bidir` flag (import BidirBDH instead of bdh_mod.BDH). Keep mean-pool + aux-LM readout.

## Test queue (one pass, then decide)
`dev/queue_harvest.sh`:
1. **C1/C2/C4 fixed-arch A/B** — for each representative dataset {Chesseract(encoded/board), Sudoku(encoded), Language(seq), Cryptic(seq), GeoClassing(image)}: train `default` vs `default+C1@stage0` vs `default+C2` vs `default+C4(shared×k)` at a fixed budget (~1.5h), full recipe (reuse `validate_winner`-style runner or a small fixed-arch trainer), report test acc vs v10. This tells us if the new ops HELP before any search.
2. **BDH-bidir (B1+B2)** run on the viable + now-medium-T datasets at long budget (viability-gated).
Then: pick winners → 3-seed × multi-interval (0.5/1/2/4h) on those only.

## BUILD DONE + QUEUE LAUNCHED (2026-07-01 21:11 CEST)
All 5 built, smoke-tested green on Spark (GB10), queue running detached.
- **C1** `_LinearAttn2D`, **C2** `_SparseGated`, **C4** `_SharedRepeat`+`build_cnn(shared_repeat=)` — in `baselines/v10/helpers.py` (OP_VOCAB entries `"LinearAttn2D"`,`"SparseGated"`; env `V11_SHARED_REPEAT`), mirrored byte-identical to `baselines/v11_1/helpers.py`.
- **B1+B2** `dev/bdh_bidir.py` (`BidirBDH`/`BidirAttention`), wired into `dev/bdh_excursion.py` via `--model bidir` (outputs suffixed `_bidir`). Smoke: B1 associative == full unmasked attn (err 3e-4); BidirBDH 1.70M params.
- **A/B trainer** `dev/harvest_ab.py` (default vs +C1@stage / +C2 / +C4×3, v10 recipe, out `{code}_ab_{tag}.json`, curves in `harvest/{tag}/{code}.csv`).
- **Queue** `dev/queue_harvest.sh`: smoke→PhaseA(C1/C2/C4 A/B, HOURS=1.5, sets Chesseract Sudoku Language Cryptic GeoClassing)→PhaseB(BDH-bidir, HOURS_BDH=2, sets Sudoku Chesseract Sedol Language Cryptic AddNIST MultNIST). Skip-if-done via json existence. Launched: `setsid bash dev/queue_harvest.sh > ~/local/harvest.log`. Outputs in spark `~/local/harvest/`.
- **Est runtime:** Phase A 5×4×1.5h=30h + Phase B ~7×2h=14h ≈ 1.8 days. Primary baseline = queue's OWN `default` arm (same budget); v10 real is secondary ref.
- **Next after results:** compare each op's margin vs default → pick winners → 3-seed × multi-interval (0.5/1/2/4h) on winners only.

## State to resume
- GPU free. Runs: none. Datasets on spark `~/local/nas-comp-datasets` + local `~/local/nas-comp-datasets`.
- Key files: `baselines/v10/helpers.py` (OP_VOCAB ~line 408, build_cnn ~line 468; op module patterns e.g. `_SpatialSelfAttn2D` ~line 226), `baselines/v11_1/` (submission copy — mirror op additions), `dev/bdh_excursion.py`, `baselines/bdh/bdh.py` (vendored, untouched).
- v11.1 s42 bake-off DONE (v11.1 best of 3-way, mean −0.63 vs v10). 3-seed sweep + EGNAS/evo reruns SKIPPED for now (verdict already clear).
- Submission-hardening TODO (separate): VRAM-adaptive batch in v11.1, starter-kit stress-test (5/30/120/1440min), Phase-3 = 24h pooled / ~8h per dataset.
