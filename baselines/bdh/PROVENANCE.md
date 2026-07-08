# BDH (Dragon Hatchling) — vendored, UNMODIFIED

- Source: https://github.com/pathwaycom/bdh  (`bdh.py`), MIT licence (see `LICENSE.md`).
- `bdh.py` is a **byte-identical copy** of upstream `bdh.py` — no edits. Verified with `diff`.
- Used by `dev/bdh_excursion.py` as a **fixed-arch family excursion** (like the TCN excursion), NOT a search op: BDH, like recurrence, loses the short-train screen and only performs under full training, so it is tested as a full-budget A/B, never selected by the proxy.
- The model is autoregressive byte-level (vocab 256, causal `tril(-1)`). The excursion's ONLY adaptation is task I/O (LM → classifier), done OUTSIDE this file: byte-tokenize the input, and read `logits[:, -1, :num_classes]` from the unmodified vocab head at the last (full-context) position. No layer of BDH is changed.
- Original training recipe kept faithful: `BDHConfig()` defaults (n_layer 6, n_embd 256, n_head 4, vocab 256, mlp_mult 128, dropout 0.1), AdamW lr 1e-3 / wd 0.1, bf16 autocast, seed 1337.
