#!/usr/bin/env python3
"""Fast correctness smoke test for the BDH-harvested build (run on Spark before
the queue). Exercises forward+backward of each new op inside build_cnn, the C4
weight-shared knob, and the BidirBDH B1+B2 model. Tiny tensors, CPU-or-CUDA,
a few seconds. Also checks that B1's associative attention matches an unmasked
full-attention reference (numerical equivalence of QR@(KR^T V) vs (QR@KR^T)@V)."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "baselines" / "v10"))
sys.path.insert(0, str(ROOT / "dev"))
from helpers import build_cnn, OP_NAMES          # noqa: E402
import bdh_bidir                                   # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={dev}  ops present: LinearAttn2D={'LinearAttn2D' in OP_NAMES} "
      f"SparseGated={'SparseGated' in OP_NAMES}")

fails = []


def check_cnn(tag, ops, **kw):
    try:
        # odd grid (9x9) on purpose -- LinearAttn2D must survive odd H/W.
        m = build_cnn(3, 5, 9, 9, ops=ops, base_width=16, **kw).to(dev)
        x = torch.randn(4, 3, 9, 9, device=dev, requires_grad=True)
        y = m(x)
        assert y.shape == (4, 5), f"bad out shape {y.shape}"
        y.sum().backward()
        g = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
        assert g > 0, "no gradient flowed"
        n = sum(p.numel() for p in m.parameters())
        print(f"  OK  {tag:26s} out={tuple(y.shape)} params={n/1e3:.1f}K grad_sum={g:.2f}")
    except Exception as e:
        print(f"  FAIL {tag:26s} {type(e).__name__}: {e}")
        fails.append(tag)


print("\n[1] build_cnn with each harvested op / knob (odd 9x9 grid):")
check_cnn("default", ["Conv3"] * 6)
check_cnn("C1 LinearAttn2D@3", ["Conv3", "Conv3", "Conv3", "LinearAttn2D", "Conv3", "Conv3"])
check_cnn("C1 LinearAttn2D@0", ["LinearAttn2D"] + ["Conv3"] * 5)   # stage 0, full res
check_cnn("C2 SparseGated@3", ["Conv3", "Conv3", "Conv3", "SparseGated", "Conv3", "Conv3"])
check_cnn("C4 shared_repeat=3", ["Conv3"] * 6, shared_repeat=3)
check_cnn("C1+C2 combined", ["Conv3", "LinearAttn2D", "Conv3", "SparseGated", "Conv3", "Conv3"])
check_cnn("RepConv3@3", ["Conv3", "Conv3", "Conv3", "RepConv3", "Conv3", "Conv3"])
check_cnn("RepConv3@0 (stride/downsample)", ["RepConv3"] + ["Conv3"] * 5)

print("\n[1b] RepConv3 reparameterize() == multi-branch (numeric fold check):")
try:
    from helpers import _RepConv3
    for (ic, oc, st) in [(16, 16, 1), (16, 32, 2)]:
        m = _RepConv3(ic, oc, st).to(dev).eval()   # eval -> BN uses running stats
        x = torch.randn(3, ic, 9, 9, device=dev)
        y_branch = m(x)
        m.reparameterize()
        y_fused = m(x)
        err = (y_branch - y_fused).abs().max().item()
        ok = err < 1e-4
        print(f"  {'OK' if ok else 'FAIL'}  in{ic}->out{oc} s{st}  fold_err={err:.2e}")
        if not ok:
            fails.append(f"RepConv-fold-{ic}-{oc}-{st}")
except Exception as e:
    print(f"  FAIL RepConv-fold {type(e).__name__}: {e}")
    fails.append("RepConv-fold")

print("\n[2] BidirBDH (B1+B2) forward/backward:")
try:
    import bdh as bdh_mod
    cfg = bdh_mod.BDHConfig(mlp_internal_dim_multiplier=8)
    m = bdh_bidir.BidirBDH(cfg).to(dev)
    idx = torch.randint(0, 256, (2, 40), device=dev)
    tgt = torch.randint(0, 256, (2, 40), device=dev)
    logits, loss = m(idx, tgt)
    assert logits.shape == (2, 40, 256), f"bad logits {logits.shape}"
    loss.backward()
    g = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
    assert g > 0
    n = sum(p.numel() for p in m.parameters())
    print(f"  OK  BidirBDH logits={tuple(logits.shape)} loss={loss.item():.3f} "
          f"params={n/1e6:.2f}M grad_sum={g:.1f}")
except Exception as e:
    print(f"  FAIL BidirBDH {type(e).__name__}: {e}")
    fails.append("BidirBDH")

print("\n[3] B1 associative == unmasked full attention (numeric):")
try:
    import bdh as bdh_mod
    cfg = bdh_mod.BDHConfig(mlp_internal_dim_multiplier=4)
    att = bdh_bidir.BidirAttention(cfg).to(dev)
    B, nh, T, N, D = 2, cfg.n_head, 16, cfg.mlp_internal_dim_multiplier * cfg.n_embd // cfg.n_head, cfg.n_embd
    Q = torch.randn(B, nh, T, N, device=dev)
    V = torch.randn(B, 1, T, D, device=dev)
    out_assoc = att(Q, Q, V)                                   # QR @ (KR^T V)
    # reference: full unmasked scores @ V, using the SAME roped Q
    r_phases = (torch.arange(0, T, device=dev, dtype=att.freqs.dtype)
                .view(1, 1, -1, 1)) * att.freqs
    QR = att.rope(r_phases, Q)
    out_ref = (QR @ QR.mT) @ V
    err = (out_assoc - out_ref).abs().max().item()
    ok = err < 1e-3
    print(f"  {'OK' if ok else 'FAIL'}  max_abs_err={err:.2e} (assoc vs full unmasked)")
    if not ok:
        fails.append("B1-numeric")
except Exception as e:
    print(f"  FAIL B1-numeric {type(e).__name__}: {e}")
    fails.append("B1-numeric")

print("\n" + ("SMOKE FAIL: " + ", ".join(fails) if fails else "SMOKE PASS: all green"))
sys.exit(1 if fails else 0)
