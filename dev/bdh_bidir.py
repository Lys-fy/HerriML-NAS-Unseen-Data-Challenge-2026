#!/usr/bin/env python3
"""BDH with the two paper-credited improvements, as a THIN subclass over the
vendored, byte-identical upstream model (baselines/bdh/bdh.py is never touched).

  B1 -- bidirectional linear attention [Eq. 8 p19; Sec. 4.2 p24].
        The upstream Attention forms the full T x T score matrix and masks it
        causally: scores = (QR @ KR^T).tril(-1); out = scores @ V  -> O(T^2 * N).
        Classification needs no autoregressive mask, so we DROP the .tril and use
        the associative form out = QR @ (KR^T @ V): the (N x D) global context
        KR^T @ V is formed once, giving O(T * N * D) -- LINEAR in T, the native
        cost of the recurrence. This removes the T^2 wall so medium-T datasets
        (Cryptic 4608, AddNIST/MultNIST 2352) can become viable. RoPE is kept
        (relative position still informative under a bidirectional readout); the
        result is left unnormalised exactly as upstream (the following self.ln
        rescales it per token).

  B2 -- all-layer readout [Sec. B.2 p54, "BDH-GPU'"].
        Upstream reads logits from the LAST layer only. The small-scale-matching
        variant merges next-token predictions across all L (weight-shared)
        layers; we accumulate lm_head applied to every layer's post-block state
        and average -- the single change the paper credits for closing the
        small-model gap.

Used by dev/bdh_excursion.py via --model bidir. The task I/O (byte-tokenise,
mean-pool + aux-LM readout, viability gate) all live in the excursion runner and
are unchanged; only the model internals differ.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_BDH = Path(__file__).resolve().parent.parent / "baselines" / "bdh"
if str(_BDH) not in sys.path:
    sys.path.insert(0, str(_BDH))
import bdh as bdh_mod  # noqa: E402  -- the UNMODIFIED upstream module


class BidirAttention(bdh_mod.Attention):
    """B1: drop the causal mask and use the associative (linear) form so the
    cost is O(T) instead of O(T^2). Same RoPE, same lack of normalisation."""

    def forward(self, Q, K, V):
        assert self.freqs.dtype == torch.float32
        assert K is Q
        _, _, T, _ = Q.size()
        r_phases = (
            torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype)
            .view(1, 1, -1, 1)
        ) * self.freqs
        QR = self.rope(r_phases, Q)
        KR = QR
        # associative order: form the (N x D) global context ONCE, then project.
        # (QR @ KR^T) @ V == QR @ (KR^T @ V), but the RHS never materialises the
        # T x T matrix -> linear in T. Bidirectional: every position sees all.
        S = KR.mT @ V                  # [B, nh, N, D] global context
        return QR @ S                  # [B, nh, T, D]


class BidirBDH(bdh_mod.BDH):
    """Upstream BDH with B1 attention + B2 all-layer readout. Everything else
    (encoder/decoder/lm_head params, LayerNorm, dropout, weight sharing across
    layers) is inherited unchanged."""

    def __init__(self, config: bdh_mod.BDHConfig):
        super().__init__(config)
        self.attn = BidirAttention(config)   # B1 -- swap in the bidirectional op

    def forward(self, idx, targets=None):
        C = self.config
        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = self.embed(idx).unsqueeze(1)
        x = self.ln(x)  # B, 1, T, D

        logits_acc = None
        for _level in range(C.n_layer):
            x_latent = x @ self.encoder
            x_sparse = F.relu(x_latent)          # B, nh, T, N

            yKV = self.attn(Q=x_sparse, K=x_sparse, V=x)
            yKV = self.ln(yKV)

            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse       # B, nh, T, N
            xy_sparse = self.drop(xy_sparse)

            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            y = self.ln(yMLP)
            x = self.ln(x + y)

            # B2 -- accumulate a readout from THIS layer (not just the last).
            layer_logits = x.view(B, T, D) @ self.lm_head
            logits_acc = layer_logits if logits_acc is None else logits_acc + layer_logits

        logits = logits_acc / C.n_layer
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
