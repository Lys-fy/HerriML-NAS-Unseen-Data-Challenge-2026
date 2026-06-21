# Cauchy (convergence prediction in complex space)

> **Status:** built 2026-06-17 (framing 2 prototype), spec frozen | **Topic:** predict whether a fixed-step iterative solver converges on a convex quadratic over the complex field, from the raw Hermitian matrix entries. A synthetic ENCODED task, built blind to the pipeline for the RQ2 generalisation test.
> **Related:** [[Datasets Overview]], [[Dataset Ideas - Backlog]], [[Sedol]], [[Sloan]], [[Quincy]], [[Pipeline v10 - Design + Idea Ledger]]

![[Cauchy.png]]
> *Two converging and two diverging samples: the real and imaginary parts of the Hermitian matrix A. The titles print the hidden lambda_max that decides the label; note the planes look alike to the eye. Regenerate: `python3 dev/plot_cauchy_examples.py`.*

A convex quadratic `f(x) = x^H A x` on C^n with A Hermitian positive-definite. Fixed-step gradient descent `x <- x - eta0 * A x` has iteration matrix `(I - eta0 A)` and converges iff `lambda_max(A) < 2/eta0`. The label is exactly that convergence test. The network sees only the matrix entries and must in effect estimate the top eigenvalue, a nonlinear spectral quantity with no spatial-image prior, so the task is encoded and hard.

| Field                  | Value                                                                         |
| ---------------------- | ----------------------------------------------------------------------------- |
| Codename               | Cauchy (complex analysis + convergence)                                       |
| Shape                  | 2x16x16 (real and imaginary parts of Hermitian A, float32)                    |
| corr1 / route          | **-0.03 / encoded** (decorrelated matrix entries; the clean encoded probe)    |
| Train / val / test     | 30,000 / 7,000 / 7,000 (balanced binary)                                       |
| Classes                | 2 (1 = converges, 0 = diverges); chance 50%                                   |
| Generation             | random unitary basis, gamma spectrum, trace fixed to n=16                      |
| Benchmark (`metadata`) | **92.21%** (ResNet-18 zero-point, 2026-06-18)                                 |
| Generator              | `dev/make_cauchy_dataset.py` (seed 17, fully synthetic)                        |
| Location               | `dev/datasets/Cauchy` (gitignored) + spark `~/local/nas-comp-datasets/Cauchy`  |

## How it was generated

1. **Random Hermitian PD A.** A random unitary basis (QR of a complex Gaussian) times eigenvalues drawn from a gamma distribution whose shape varies per sample, so the spectral concentration varies widely. `A = U diag(lambda) U^H`.
2. **Defeat the trivial shortcut.** Each spectrum is rescaled so the trace is fixed to n. The answer is therefore NOT readable from the trace; it depends on the spectral SHAPE (how concentrated the mass is in the top eigenvalue). Frobenius norm remains a weak residual cue, but it is itself a nonlinear function of the entries the net has to compute.
3. **Balanced label.** Over a pool of 50,000 matrices, the threshold `eta0 = 2 / median(lambda_max)` splits converge vs diverge exactly in half (median lambda_max = 2.231, so eta0 = 0.896).
4. **Encoding + package.** Two 16x16 planes (real symmetric, imag antisymmetric). float32 `*_x.npy`, int64 `*_y.npy`, JSON `metadata`.

## Results (2026-06-18)

![[newdata_curves_grid_4x5.png]]
> *Custom-dataset grid (bottom three rows are the new sets): v10 best-val (green, 3 seeds + aggregate) over the ResNet-18 zero-point (blue).*

ResNet-18 benchmark **92.21%**. v10 3-seed test accuracy **93.91%** (s42 93.89, s1337 93.86, s2024 93.99), so **adjusted +2.18** (the strongest of the three new sets). Encoded and learnable: v10 clearly beats ResNet. Caveat: the curve plateaus fast at ~94%, because the label correlates with a simple Frobenius-norm cue, so the net shortcuts rather than doing full spectral reasoning. A harder variant (normalise the Frobenius norm away) would force genuine spectral estimation and make it a sharper architecture discriminator.

## Measured properties (before any pipeline run)

- **Routes encoded (corr1 = -0.03), confirmed.** The matrix entries are decorrelated, so the fingerprint routes it encoded (gate + random arm active) -- the only one of the three new sets to do so. The label is a nonlinear function of the eigenvalues with no local-pixel template, exactly the regime where the spatial prior misleads.
- **Honest difficulty.** The fixed-trace trick removes the obvious linear shortcut; the residual Frobenius-norm correlation makes it learnable but not trivial. Expect a clear-but-imperfect benchmark above the 50% floor.
- **Framing 2, on purpose.** The rejected framing 1 (render the loss landscape as a heatmap) is smooth and image-like; this coefficient-tensor framing is the encoded sibling we want. See [[Dataset Ideas - Backlog]].

## Open items

- ResNet-18 benchmark (92.21%) + v10 3-seed (93.91%): DONE (see Results).
- First v10 run: does the encoded route plus the random/macro arm find anything above a plain conv here, or is spectral estimation flat across architectures? Either answer is informative.
- Difficulty / richness dials: matrix dimension n, the gamma shape range, and a multi-way iteration-count variant instead of the binary converge/diverge.
