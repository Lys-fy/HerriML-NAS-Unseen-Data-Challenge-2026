# Environment notes

## Open question: actual evaluation environment

The starter kit ships:
```
torch==1.10.1+cu113
torchvision==0.11.2+cu113
sklearn          # deprecated package name since 2023
```

This looks stale (2021-era). **Before pinning anything, email the organisers and confirm the real Phase 2/3 environment** (Python version, PyTorch version, CUDA version).

## Architecture split

| Machine | Arch | Role |
|---------|------|------|
| DGX Spark | ARM64 + Grace Blackwell | Primary dev / training. Needs CUDA 12.8+ / PyTorch ≥ 2.5 — cannot run the starter-kit pins. |
| M75q (Lenovo) | x86 + AMD iGPU | **Cross-verification.** Run the eval-pinned env here before every submission. |
| Eval server | unknown (likely x86) | Where the scored run executes. |

Implication: code must be portable across PyTorch versions and architectures. Avoid:
- `torch.compile` (PyTorch 2.0+ only)
- `torchvision.transforms.v2` (recent only)
- Anything in `torch.func` newer than the eval target
- Fancy AMP API quirks specific to recent versions

## Plan once confirmed

- `requirements-dev.txt` — modern PyTorch for Spark
- `requirements-eval.txt` — pinned to match eval server, used on M75q for final validation
- Submission must pass on M75q with `requirements-eval.txt` before each push.
