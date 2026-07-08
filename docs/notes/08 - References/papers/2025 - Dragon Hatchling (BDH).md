# 2025 - Dragon Hatchling (BDH)

> **Status:** read + mined (2026-07-01) | **Topic:** "The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain" (Pathway; arXiv:2509.26507). Read for (a) whether BDH is a viable architecture for the competition and (b) portable primitives for our NAS search space / pipeline. | **PDF:** `docs/papers/Kosowski-2025_Dragon-Hatchling-BDH_arxiv-preprint.pdf` | **Code:** github.com/pathwaycom/bdh (vendored unmodified at `baselines/bdh/bdh.py`) | **Related:** [[EGNAS]], [[Zero-Cost Proxies]], [[SOTA Scan - Novel Directions]], [[Key Learnings]], [[v11.1]], [[Pipeline v10 - Design + Idea Ledger]]

## What BDH is

BDH reframes a Transformer as a large **interacting-particle / graph** system of `n` "neurons." The trainable model, **BDH-GPU** (Def. 4, Eq. 8), is a **linear-attention state-space model**: each block lifts a narrow residual stream `D` into a **wide, positive, sparse ReLU** vector of size `n` ("neurons"), runs **linear attention** using that wide positive vector as query AND key with the narrow stream as value, applies a second **self-gated** wide ReLU expansion, and projects back to `D` with a residual. Layers **share weights** (Universal-Transformer style). It matches GPT-2 scaling laws (10M-1B params, Fig. 7), claims to **learn faster per token / prefer scarce-data regimes** (Sec. 4.2), and shows emergent ~5% sparse monosemantic activations + a scale-free modular weight graph (Secs. 5-6). It is a **causal, 1-D, next-token LM**; classification, pooling, and 2-D inputs are entirely our adaptation (see `dev/bdh_excursion.py`).

## Our empirical results (the BDH excursion)

We vendored `bdh.py` unmodified and adapted the I/O only: byte-tokenize input (flatten C*H*W to 0-255 tokens) -> BDH -> **mean-pool the vocab head over positions + auxiliary next-byte LM loss** for dense supervision. Downscaled `mlp_internal_dim_multiplier` 128->8 (~1.7M params, comparable to our v10/v11 nets).

- **Sudoku (T=81): 96.2% vs v10 97.7%** -- competitive on its showcase domain.
- **The T-tier reality (cost is O(T^2) in the vendored code):** BDH is only affordable where the flattened sequence is short. SMALL (T<=800): Sloan 50, Sudoku 81, GameOfLife 100, Quincy 441, Gutenberg 486, Cauchy 512, **Language 576**, Sedol 722, Chesseract 768. MED: AddNIST/MultNIST 2352. LARGE (hopeless): Cryptic 4608, images 12288+, Myofibre 49k.
- **Striding a flattened grid destroys structure** (Chesseract capped-to-128 -> 38.8 < 53.7 uncapped). So we keep FULL resolution and gate on a runtime **viability check** (calibrate step time -> project epochs at full res; VIABLE if >= target, else fall back to v11.1). `dev/bdh_excursion.py` + `dev/queue_bdh.sh`.

## BDH-specific findings (if we keep the model)

- **B1 -- the O(T) formulation is native; our T^2 wall is a vendored-code artifact. [Eq. 8 p19; Sec. 4.2 p24; Appendix E p62].** Eq. 8 is a linear-attention recurrence (state `rho in R^{n x d}`, rank-1 update per token, O(T*n*d) = linear in T). The shipped `LinearAttention` uses the naive `(Q@K^T).tril @ V` = O(T^2*n) instead. **For CLASSIFICATION we drop the causal mask -> bidirectional linear attention collapses to `Q @ (K^T V)` -> O(T) for free, no scan** (this is C1 below).
- **B2 -- read out from ALL layers. [Sec. B.2 p54].** The variant that matches Transformers at small scale ("BDH-GPU'") merges next-token predictions across all L shared-weight layers. Our readout uses only the last layer; pooling logits across layers is the single change the paper credits for closing the small-model gap.
- **Honest caveats:** ALL results are language/translation next-token loss. **No image, classification, OOD, or length-generalization experiments** -- the "generalizes to unseen data" story is aspirational, not measured. The emergent sparsity/monosemanticity/modularity are shown only at **>= 24M params, d=256** (theory needs `d = Omega(log n)`); our 1.7M downscale is ~10x below that, so those properties likely do not survive. The one competition-relevant claim NOT gated on scale: **sample efficiency** ("learns faster per token").

## Portable primitives for our NAS search + pipeline (the real payoff)

BDH decomposed into drop-in modules -- **all bidirectional, classification-ready, usable WITHOUT BDH**. Ranked usefulness/effort. The four the team flagged as "interesting to try" are marked ⭐.

### C1 -- LinearAttn2D: bidirectional linear-attention spatial mixer  ★ TOP PICK (new search op)
A global spatial mixer **linear in HW**, vs our `SpatialSelfAttn2D` which is O((HW)^2) AND crashed on odd grids:
```python
q = F.relu(self.to_q(x)); k = F.relu(self.to_k(x)); v = self.to_v(x)   # 1x1 convs; ReLU keys/queries
Q,K,V = flatten_HW(q,k,v)                    # (B, HW, rC), (B, HW, rC), (B, HW, C)
S = K.transpose(1,2) @ V                     # (B, rC, C) global context, computed ONCE
ctx = (Q @ S) / (Q @ K.sum(1,keepdim=True).transpose(1,2) + eps)   # (B, HW, C)
x = x + self.proj(unflatten(ctx))
```
Cheaper + shape-robust replacement for our attention op. Positive ReLU keys/queries make linear attention a valid high-dim kernel (Sec. 6.1; FAVOR+ lineage). Slots into the next v11.2/v11.3 search.

### C2 ⭐ -- Sparse-positive gated expansion (denoising op motif)  (new search op / FFN block)
`y = down( ReLU(up1(x)) * ReLU(up2(x)) )` + residual -- an inverted bottleneck with a **positive (ReLU*ReLU) gate** instead of SwiGLU's SiLU. Paper: ReLU + learned **negative bias** is a denoiser that recovers sparse in-cluster signal from low-rank noise (Claim 5, Eq. 11) and yields monosemantic units (Sec. 6.3) -- a robustness prior for small/unseen data. Trivial (3 pointwise convs). **Add the learned negative bias before ReLU** as the paper prescribes.

### C3 ⭐ -- Activation-sparsity fingerprint (training-free proxy)  (pipeline routing signal)
Fig. 14 / Sec. 6.4: ReLU activation sparsity **rises with input complexity, falls with predictability**. Run a few forward passes of an *untrained* wide-ReLU block over a batch, record mean sparsity + variance -> a **cheap dataset-difficulty fingerprint** to complement our corr1 signal for routing (bigger vs smaller default). A few forwards + `count_nonzero`; no training. Caveat: strong regime is >= 24M params -- validate the correlation on our own tasks first (methodology: freeze the threshold on held-out data, never tune on eval).

### C4 ⭐ -- Universal-Transformer weight sharing (recurrence-as-depth)  (macro-construction knob)
BDH reuses one weight set across all L layers, and argues L should track the problem's working-memory / syntax-tree DEPTH, not its size (Appendix D). Add a macro knob "**apply the searched block x k with shared weights**" -- many effective layers at one block's params. Tailor-made for our ~1.7M-param / 1h regime. `for _ in range(k): x = block(x)`.

### C5 ⭐ -- Model composition by tensor concatenation  (cheap ensemble primitive)
Sec. 7.1 / Table 2 / Fig. 15: BDH models trained on disjoint data merge by **concatenating parameter tensors along the neuron dimension** (+ averaging shared params), yielding a working combined model **with no retraining**. A cheap cross-family **composition/ensemble** primitive for our surplus-budget tier (cf. the v11.1 gated ensemble) -- worth testing whether the concat-merge idea transfers to our conv nets or inspires a param-space (vs logit-space) ensemble.

### C6-C8 (lower priority)
- **C6** Grouped channels + per-group affine-free LayerNorm (cheap regularizer / capacity knob for small data).
- **C7** Full composite "BDH block" as one selectable per-stage op (C1+C2+residual+LN) -- moderate effort, emergent benefits unlikely at our scale; one candidate op, not a backbone.
- **C8** Heavy-tailed / scale-free init for hub structure -- speculative, cheap to try, lowest confidence.

## Verdict

- **BDH-the-model:** a cautiously-worthwhile **short-sequence, scarce-data specialist**, only after B1 (recurrent/bidirectional O(T) attention) + B2 (all-layer readout), and only gated on short T. Its "unseen-data generalization" is unproven relative to our needs.
- **BDH-the-primitives:** the higher-value harvest. **C1-C5 are low-risk, low-effort additions to our proven pipeline** and stand alone without adopting BDH. This fits our "informed-defaults + verified search" ethos better than betting the architecture on BDH. The four flagged for trial: **C2 (sparse-positive gated op), C3 (sparsity fingerprint), C4 (weight-shared depth), C5 (param concat ensemble)**, plus C1 as the top new search op.
