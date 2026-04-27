> **Status:** drafting | **Topic:** Why our development decisions may not transfer to the final scoring run.
> **Related:** [[Phases and Timeline]], [[Reliability vs Peak]], [[Failure Modes]], [[Mock Harness]], [[Search vs Train Split]], [[Score-Efficient Targeting]], [[Predictor-Guided Evolutionary]]

The submission we tune locally is **not** the submission the organisers score. Several gaps separate the two; each is a distinct risk. Listed in rough order of expected impact.

## 1. Eval environment drift

We don't know the evaluation hardware. Quote from the organisers (see [[Rules]]): *"we have decided to withhold the GPU that we will use to evaluate final submission, though each submission will be tested on machines of equivalent specs"*. Concretely:

- **GPU model and VRAM**: unknown. Our local box could be more or less constrained.
- **CPU and RAM**: not specified. Dataloader worker counts and memory-mapped tensors behave differently across configurations.
- **PyTorch / CUDA**: starter kit pins `torch==1.10.1+cu113` (2021-era) and the deprecated `sklearn` package name — the actual eval pin is presumably newer but unspecified.

**Mitigation**: design for adaptivity, not for a fixed target. Read GPU memory at startup, scale batch size accordingly, never assume fixed VRAM. Don't import anything beyond what's in the starter kit's `requirements.txt` — surprise dependencies break silently. See [[Submission Packaging]].

## 2. Dev / eval dataset divergence

We develop on the [[Datasets Overview|public 13]] (minus Isabella). The Phase 2 / Phase 3 datasets are **different** by design — the entire competition framing is "unseen data" (see Geada CVPR 2024 abstract in [[Papers]]). Hidden datasets share only the file layout, not the distribution.

Specifically:

- A method tuned for AddNIST's `(3, 28, 28)` MNIST channels may not generalise to a hidden dataset's `(1, 64, 128)` spectrogram-shaped tensor.
- A predictor trained on the public 13 (e.g. [[Predictor-Guided Evolutionary]] approaches) may have learned domain priors that don't transfer. The 2025 Cryptic gap (76 % vs 73 %, see [[2025 - Sichuan teams]]) suggests this is what cost the Sichuan groups the competition.
- The Geada paper explicitly warns: *"none of the Type-2 datasets have been correctly identified"* by participants in prior years. Our intuitions about input semantics will be wrong on hidden data.

**Mitigation**: prefer search algorithms that don't carry strong dataset priors — [[EGNAS]]-style hill-climbing inspects the dataset directly, while predictor-based approaches encode priors from their training set. Validate any priors via a **leave-one-out** style check: train a predictor on 12 of the public 13 and predict the 13th, repeated for each.

## 3. Overfit to public 13

The reverse risk: our ablations and architecture choices are *only* validated on datasets we've seen. Even a method that doesn't carry dataset priors may have its hyperparameters tuned to match public-13 quirks (e.g. cutout=16 from [[EGNAS]]'s public defaults is a CIFAR convention).

**Mitigation**:

- Pick reasonable defaults from the *literature* (cosine LR, label smoothing, mixup), don't tune them on the public 13.
- Run cross-validation across the public 13 with held-out subsets. See [[Testing Strategy]].
- When tempted to add a heuristic that "fixes" performance on a specific public dataset (say, a special branch for 1-channel inputs), ask: would this help or hurt on a hidden dataset of the same input shape? If unclear, leave it out.

## 4. Randomness

The score is computed from a single test-set run. NAS and training are stochastic. A submission scoring +9 on one run might score +6 on a re-run with a different seed.

**Mitigation**:

- Set seeds for `torch`, `numpy`, and CUDA at the top of every class.
- Run the [[Mock Harness]] at least 3 times per submission and look at variance, not just mean. See [[Testing Strategy]].
- For very high-leverage datasets (high slope, see [[Score-Efficient Targeting]]), an ensemble across 2–3 trained models can reduce variance — only worth it if budget allows. See [[Budget Allocation]].

## 5. Transfer assumptions in our budget allocator

The [[Search vs Train Split]] regime classifier reads `metadata['benchmark']`. We *assume* benchmark correlates with achievable ceiling. This is true on the public 13 (high benchmark → small headroom, low benchmark → small absolute gap) but the organisers could engineer a hidden dataset that breaks the correlation: low benchmark + high achievable ceiling, or vice versa.

**Mitigation**: log per-dataset progress in the [[Submission Contract]] (validation accuracy at each epoch). If a Regime-A dataset is plateauing at 60 % validation accuracy when benchmark is 85 %, something is wrong — but we can't replan inside the submission. Across submissions, this signal informs the next iteration of the regime classifier.

## 6. Pipeline contract changes

`evaluation/main.py` is the grader's authoritative shape. We mirror it in [[Mock Harness]], but only as of Phase 1. The organisers have changed details year-over-year (e.g. dataset list rotated, time limits adjusted).

**Mitigation**: re-read the official starter kit just before each [[Phase 2 Submission Strategy|Phase 2 attempt]]. Diff the eval `main.py` against our `dev/mock_harness.py`. Any drift is a candidate for −10.

## 7. Reproducibility of the winning submission

Our best Phase 2 submission is auto-promoted to Phase 3 unchanged ([[Phases and Timeline]]). If we've been "lucky" at Phase 2, we may not be lucky at Phase 3 on the same hidden datasets — same code, same datasets, but a different random run.

**Mitigation**: don't optimise for a peak Phase 2 number that we can't reproduce. Submit the most *consistent* high-score variant, not the highest-ever observed score. See [[Phase 2 Submission Strategy]].
