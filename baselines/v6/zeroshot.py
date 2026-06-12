"""Zero-shot NAS proxies — shadow signals computed per EGNAS candidate.

These are mathematical functions of the *untrained* (randomly-initialised)
architecture. They don't require any training and are essentially free
to compute (~ms per candidate). v6 logs all four alongside each
candidate's short-train val_acc so we can compute Spearman τ post-sweep
and see whether any of them transfer to the Unseen suite — if so, v7
could replace the noisy short-train ranker entirely.

References:
- Abdelfattah et al. "Zero-Cost Proxies for Lightweight NAS" ICLR 2021
- Mellor et al. "Neural Architecture Search without Training" ICML 2021
- Tanaka et al. "Pruning neural networks without any data" NeurIPS 2020
- Lee et al. "SNIP: Single-shot Network Pruning" ICLR 2019
- Lin et al. "Zen-NAS" ICCV 2021

All four are legal in the NAS Unseen Data Challenge — they're standard
architecture-analysis math, no pre-trained model, no external data.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def _maybe_iter_relus(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.ReLU,)):
            yield m


def naswot_score(model: nn.Module, x: torch.Tensor) -> float:
    """NASWOT (Mellor et al. 2021) — count distinct ReLU activation patterns
    across a batch. The intuition: an architecture is "expressive" at init
    if it maps different inputs to different ReLU activation patterns. A
    network that hits the same pattern for everything can't learn to
    distinguish inputs.

    Returns log|det(K @ K.T)| where K[i, j] = 1 if the j-th ReLU output on
    sample i was positive. We sum across ReLUs in the network."""
    model.eval()
    hooks = []
    K_blocks = []

    def hook_fn(_module, _inp, out):
        # out: any shape; flatten to (B, F), record post-ReLU sign
        flat = (out > 0).int().flatten(1)
        K_blocks.append(flat)

    for relu in _maybe_iter_relus(model):
        hooks.append(relu.register_forward_hook(hook_fn))

    try:
        model(x)
    except Exception:
        for h in hooks:
            h.remove()
        return float("nan")
    finally:
        for h in hooks:
            h.remove()

    if not K_blocks:
        return float("nan")

    K = torch.cat(K_blocks, dim=1).float()                 # (B, total_relus)
    M = K @ K.T + (1 - K) @ (1 - K).T                       # (B, B) sim matrix
    # Add tiny diag for numerical stability of logdet
    M = M + 1e-3 * torch.eye(M.size(0), device=M.device)
    sign, logabsdet = torch.slogdet(M)
    if sign.item() <= 0 or not torch.isfinite(logabsdet):
        return float("nan")
    return float(logabsdet.item())


def synflow_score(model: nn.Module, input_shape: Tuple[int, ...],
                  device: torch.device) -> float:
    """Synflow (Tanaka et al. 2020) — sum of |weight × gradient| after
    pushing an all-ones input through a sign-stripped network. Data-free —
    needs no training data at all. Captures structural trainability."""
    model.eval()

    # 1. Save signs + abs-ify all weights (the "sign-strip" trick).
    signs = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            signs[name] = torch.sign(p.data).clone()
            p.data.abs_()

    # 2. Forward all-ones input.
    x = torch.ones(1, *input_shape, device=device)
    try:
        model.zero_grad()
        out = model(x)
        loss = out.sum()
        loss.backward()
    except Exception:
        # Restore signs even on failure
        with torch.no_grad():
            for name, p in model.named_parameters():
                p.data.mul_(signs[name])
        return float("nan")

    # 3. Score = sum over all params of |w * grad|.
    score = 0.0
    for p in model.parameters():
        if p.grad is not None:
            score += (p.detach() * p.grad).abs().sum().item()

    # 4. Restore original signs.
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.data.mul_(signs[name])
    model.zero_grad()
    return float(math.log(score + 1e-12))


def snip_score(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """SNIP (Lee et al. 2019) — connection sensitivity score: sum of
    |grad × weight| per parameter using a real data batch. How much would
    removing each connection hurt the loss? Sum = total trainability."""
    model.eval()
    try:
        model.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
    except Exception:
        return float("nan")

    score = 0.0
    for p in model.parameters():
        if p.grad is not None:
            score += (p.detach() * p.grad).abs().sum().item()
    model.zero_grad()
    return float(math.log(score + 1e-12))


def zen_score(model: nn.Module, input_shape: Tuple[int, ...],
              device: torch.device, n_samples: int = 32) -> float:
    """Zen-NAS-style (Lin et al. 2021) — log variance of network output
    under random Gaussian input. Cheap proxy for "is this network using
    its capacity"."""
    model.eval()
    x = torch.randn(n_samples, *input_shape, device=device)
    try:
        with torch.no_grad():
            out = model(x)
    except Exception:
        return float("nan")
    v = out.var().item()
    return float(math.log(v + 1e-12))


def compute_all_zeroshot(model: nn.Module,
                         x_batch: torch.Tensor, y_batch: torch.Tensor,
                         input_shape: Tuple[int, ...],
                         device: torch.device) -> Dict[str, float]:
    """Orchestrator — runs all 4 proxies + plain param count and returns
    them as a dict to attach to the EGNAS pool entry. Each is robust to
    weird shapes / dead networks (returns NaN on failure rather than
    raising).

    NOTE: forced to CPU because NVRTC on the Blackwell GB10 fails with
    "invalid value for --gpu-architecture" on the fused CUDA kernels the
    proxies trigger (e.g. NASWOT's bool→int conversion, slogdet). Proxies
    are cheap enough on CPU (~ms-seconds per candidate, no batch loop).
    """
    # Move a copy of the model to CPU for proxy work; the original on-GPU
    # model is left untouched for the search's actual training.
    import copy
    proxy_model = copy.deepcopy(model).to("cpu")
    proxy_device = torch.device("cpu")

    bs_for_nawsot = min(16, x_batch.size(0))
    sub_x = x_batch[:bs_for_nawsot].cpu()
    sub_y = y_batch[:bs_for_nawsot].cpu()

    scores = {
        "naswot": naswot_score(proxy_model, sub_x),
        "synflow": synflow_score(proxy_model, input_shape, proxy_device),
        "snip": snip_score(proxy_model, sub_x, sub_y),
        "zen": zen_score(proxy_model, input_shape, proxy_device),
        "params": int(sum(p.numel() for p in proxy_model.parameters())),
    }
    # Free the CPU copy
    del proxy_model
    return scores
