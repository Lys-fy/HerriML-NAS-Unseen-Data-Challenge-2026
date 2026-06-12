# Quincy (QR codes)

> **Status:** built 2026-06-12, spec frozen | **Topic:** version-1 QR codes encoding a class token; 20-way classification with 10% ECC-recoverable corruption. First of our own additional datasets (built blind to the pipeline, for the RQ2 generalisation test).
> **Related:** [[Datasets Overview]], [[Pipeline v10 - Design + Idea Ledger]], [[Sudoku]], [[GameOfLife]]

![[Quincy.png]]
> *Row 1: six clean codes of the same class (the visible variation comes from the random payload suffix and the random mask pattern). Row 2: six different classes. Row 3: the two corruption types. Regenerate: `/tmp/qrenv/bin/python dev/plot_quincy_examples.py` (any env with the `qrcode` package works).*

The network sees a raw 21x21 binary module grid and must recover which of 20 class tokens was encoded into it. Unlike the official datasets this one is ours: the spec below was fixed before the pipeline ever saw a sample, so whatever v10 does on it is an honest out-of-design-distribution result.

| Field                  | Value                                                                         |
| ---------------------- | ----------------------------------------------------------------------------- |
| Codename               | Quincy                                                                        |
| Shape                  | 1x21x21 (channels-first, values {0,1}, float32)                               |
| Train / val / test     | 50,000 / 10,000 / 10,000                                                      |
| Classes                | 20 (the class token encoded in the payload)                                   |
| Corruption             | 10.0% of samples (uniform across splits)                                      |
| Benchmark (`metadata`) | **0.0%** placeholder, ResNet-18 run still owed                                |
| Generator              | `dev/make_qr_dataset.py` (seed 7, fully regenerable)                          |
| Location               | `dev/datasets/Quincy` (gitignored) + spark `~/local/nas-comp-datasets/Quincy` |

## How it was generated

Generator: `dev/make_qr_dataset.py` (committed; arrays are not in git because 13 seconds of CPU regenerates them bit-identically from the seed).

1. **Payload.** Each sample draws a class `k` in 0..19 and a random 8-character alphanumeric suffix, then encodes the string `"C<k:02d>-<suffix>"`. The class lives only in the first three characters; the suffix forces intra-class variation through the data codewords and the error-correction codewords.
2. **QR encoding.** `qrcode.QRCode(version=1, error_correction=ERROR_CORRECT_Q, border=0, mask_pattern=rng.integers(0,8))`. Version 1 gives exactly 21x21 modules. ECC level Q adds ~25% recoverable redundancy. The mask pattern (one of 8, normally chosen for print quality) is drawn at random, so the same payload can produce 8 different module layouts; the net has to read the format information to undo the mask before the data modules mean anything.
3. **Corruption (10% of samples).** A coin flip picks one of two damage types: every module flips independently with p = 0.06, or a 5x5 patch is overwritten with a constant. Both stay within what ECC-Q can correct in principle, so the label is always recoverable and corruption is a robustness test, not label noise.
4. **Packaging.** float32 `train/valid/test_x.npy`, int64 `*_y.npy`, JSON `metadata` with codename/input_shape/num_classes/benchmark, exactly the official format (`dev/mock_harness.py` loads it unchanged).

```python
# the core of make_qr_dataset.py
payload = f"C{cls:02d}-{suffix}"            # class token + 8 random alnum
qr = qrcode.QRCode(version=1, error_correction=ERROR_CORRECT_Q,
                   border=0, mask_pattern=int(rng.integers(0, 8)))
qr.add_data(payload); qr.make(fit=False)
m = np.array(qr.get_matrix(), dtype=np.float32)   # 21x21, {0,1}
if rng.random() < 0.10:
    m = corrupt(rng, m)                     # flips p=0.06 OR 5x5 occlusion
```

## Measured properties (before any pipeline run)

- **corr1 = 0.054**: neighbouring-module correlation is near zero because data modules are pseudo-random; the v10 fingerprint will route Quincy ENCODED (gate + random arm active). This was the prediction for QR and is why it is a fingerprint edge case: the finder/timing patterns are strongly local, the payload is not.
- **No template shortcut**: two same-class codes differ in ~114 of 441 modules.
- **Learnable, not linearly solvable**: logistic regression on raw modules reaches 24.9% test accuracy vs 5% chance (a foothold from mask-specific module correlations); full decoding is possible in principle, so the ceiling is near 100%.

## Open items

- ResNet-18 benchmark run to replace the 0.0 placeholder.
- First v10 run, queued behind the 3-seed sweep (see [[Pipeline v10 - Design + Idea Ledger]]). Watch the backbone gate: QR mixes strongly local landmarks with non-local payload structure, so the encoded-route arms may genuinely disagree here.
- The corruption rate is the difficulty dial (10% now); a harder variant just changes `CORRUPT_FRAC`/`FLIP_P` and reruns the generator.
