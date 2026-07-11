"""Cryptic-class ARCHITECTURE specialist excursion (v12.3 T3).

Cryptic is the one podium-decisive gap left: the conv pipeline plateaus at
72.0-73.8 at ANY budget/hardware, BDH is closed (flat across 15x params),
capacity and compute are ruled out. Its input is [N, 1, 6, 768] -- six
768-dim rows that look like word/clue EMBEDDINGS, not an image. The conv
family can only see them through a spatial lens; this excursion treats them
as TOKENS instead.

Three arms, sequential, val-gated, each on a slice of --hours:
  tok256   6 tokens x 768d -> Linear proj 256 + pos emb, 4-layer Transformer
           encoder, mean-pool head.
  tok384   deeper/wider twin (384d, 6 layers) -- does token capacity scale?
  mixer    MLP-Mixer contrast (token-mix over 6, channel-mix over 768) --
           is it attention that matters, or just cross-token mixing?

Each arm: AdamW + cosine, label smoothing, token dropout; best-val state is
kept in RAM and scored on test at the end. If >=25 min remain after all arms,
the best arm is retrained on a second seed and the 2-seed logit ensemble is
scored too (the pipeline's own ensemble lesson).

Success bar: beat the pipeline plateau (v12.1 GB10 8h = 72.84; all-time
external best = 73.82). Anything >= 74 changes the Phase-3 podium math.

Usage (spark):
  .venv/bin/python dev/cryptic_specialist.py \
      --dataset-dir ~/local/nas-comp-datasets/Cryptic \
      --out-dir ~/local/v123_gb10/cryptic_spec --hours 5
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--hours", type=float, default=5.0)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--max-epochs", type=int, default=400)
    p.add_argument("--patience", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# --------------------------------------------------------------------------- data
def load_split(d: Path, split: str):
    x = np.load(d / f"{split}_x.npy").astype(np.float32)
    y = np.load(d / f"{split}_y.npy").astype(np.int64)
    return torch.from_numpy(x).squeeze(1), torch.from_numpy(y)  # [N, 6, 768]


def make_loaders(d: Path, batch: int):
    xs, ys = {}, {}
    for split in ("train", "valid", "test"):
        xs[split], ys[split] = load_split(d, split)
    mu = xs["train"].mean(dim=0, keepdim=True)          # [1, 6, 768] per-position
    sd = xs["train"].std(dim=0, keepdim=True).clamp_min(1e-5)
    loaders = {}
    for split in ("train", "valid", "test"):
        xn = (xs[split] - mu) / sd
        ds = torch.utils.data.TensorDataset(xn, ys[split])
        loaders[split] = torch.utils.data.DataLoader(
            ds, batch_size=batch, shuffle=(split == "train"),
            num_workers=2, pin_memory=True, drop_last=(split == "train"))
    n_cls = int(ys["train"].max().item()) + 1
    return loaders, n_cls


# --------------------------------------------------------------------------- models
class TokenTransformer(nn.Module):
    """6 tokens x 768d treated as a sequence, not an image."""

    def __init__(self, n_cls: int, dim: int, depth: int, heads: int = 8):
        super().__init__()
        self.proj = nn.Linear(768, dim)
        self.pos = nn.Parameter(torch.zeros(1, 6, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_cls))

    def forward(self, x):                    # x: [B, 6, 768]
        h = self.enc(self.proj(x) + self.pos)
        return self.head(h.mean(dim=1))


class MixerBlock(nn.Module):
    def __init__(self, n_tok: int, dim: int, tok_hidden: int, ch_hidden: int):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.tok = nn.Sequential(nn.Linear(n_tok, tok_hidden), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(tok_hidden, n_tok))
        self.n2 = nn.LayerNorm(dim)
        self.ch = nn.Sequential(nn.Linear(dim, ch_hidden), nn.GELU(),
                                nn.Dropout(0.1), nn.Linear(ch_hidden, dim))

    def forward(self, x):                    # x: [B, 6, dim]
        x = x + self.tok(self.n1(x).transpose(1, 2)).transpose(1, 2)
        return x + self.ch(self.n2(x))


class TokenMixer(nn.Module):
    """Cross-token mixing WITHOUT attention -- the contrast arm."""

    def __init__(self, n_cls: int, dim: int = 256, depth: int = 6):
        super().__init__()
        self.proj = nn.Linear(768, dim)
        self.blocks = nn.Sequential(*[MixerBlock(6, dim, 64, dim * 4)
                                      for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_cls))

    def forward(self, x):
        return self.head(self.blocks(self.proj(x)).mean(dim=1))


ARMS = [
    ("tok256", lambda n: TokenTransformer(n, dim=256, depth=4)),
    ("tok384", lambda n: TokenTransformer(n, dim=384, depth=6)),
    ("mixer",  lambda n: TokenMixer(n)),
]


# --------------------------------------------------------------------------- train
@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        correct += (model(xb).argmax(1) == yb).sum().item()
        total += yb.numel()
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def logits_of(model, loader, device) -> torch.Tensor:
    model.eval()
    outs = [model(xb.to(device)).float().cpu() for xb, _ in loader]
    return torch.cat(outs)


def train_arm(name, build, loaders, n_cls, device, deadline, args, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build(n_cls).to(device)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    warm = 5
    sched = lambda e: (e + 1) / warm if e < warm else \
        0.5 * (1 + math.cos(math.pi * (e - warm) / max(args.max_epochs - warm, 1)))
    best_val, best_state, best_epoch = -1.0, None, -1
    hist = []
    print(f"[{name}] {n_par:.2f}M params, budget "
          f"{(deadline - time.time()) / 60:.0f}min", flush=True)
    for epoch in range(args.max_epochs):
        if time.time() > deadline:
            print(f"[{name}] arm deadline at epoch {epoch}", flush=True)
            break
        for g in opt.param_groups:
            g["lr"] = 3e-4 * sched(epoch)
        model.train()
        tot_loss = n_b = 0
        for xb, yb in loaders["train"]:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            if model.training:  # token dropout: zero one random token row per sample
                drop = torch.rand(xb.size(0), 6, 1, device=xb.device) < (0.1 / 6)
                xb = xb.masked_fill(drop, 0.0)
            loss = crit(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += loss.item(); n_b += 1
        val = evaluate(model, loaders["valid"], device)
        hist.append(val)
        if val > best_val:
            best_val, best_epoch = val, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch % 10 == 0 or val == best_val:
            print(f"[{name}] e{epoch} loss={tot_loss / max(n_b, 1):.4f} "
                  f"val={val:.2f} best={best_val:.2f}", flush=True)
        if epoch - best_epoch > args.patience:
            print(f"[{name}] patience out at e{epoch} (best e{best_epoch})", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(model, loaders["test"], device)
    print(f"[{name}] DONE best_val={best_val:.2f} (e{best_epoch}) "
          f"TEST={test:.2f}", flush=True)
    return {"arm": name, "params_m": n_par, "best_val": best_val,
            "best_epoch": best_epoch, "test": test, "val_hist": hist,
            "model": model}


def main():
    args = get_args()
    out = Path(args.out_dir).expanduser(); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_end = time.time() + args.hours * 3600 - 120
    loaders, n_cls = make_loaders(Path(args.dataset_dir).expanduser(), args.batch)
    print(f"device={device} n_cls={n_cls} "
          f"train={len(loaders['train'].dataset)} "
          f"valid={len(loaders['valid'].dataset)} "
          f"test={len(loaders['test'].dataset)}", flush=True)

    results = []
    for i, (name, build) in enumerate(ARMS):
        arm_deadline = min(t_end, time.time() +
                           (t_end - time.time()) / (len(ARMS) - i))
        results.append(train_arm(name, build, loaders, n_cls, device,
                                 arm_deadline, args, args.seed))

    # surplus: second seed of the best arm -> 2-seed logit ensemble
    best = max(results, key=lambda r: r["best_val"])
    ens = None
    if t_end - time.time() > 25 * 60:
        print(f"[ens] surplus {(t_end - time.time()) / 60:.0f}min -> "
              f"seed-2 of {best['arm']}", flush=True)
        build = dict((n, b) for n, b in ARMS)[best["arm"]]
        r2 = train_arm(best["arm"] + "_s2", build, loaders, n_cls, device,
                       t_end, args, args.seed + 1)
        lg = (logits_of(best["model"], loaders["test"], device).softmax(1)
              + logits_of(r2["model"], loaders["test"], device).softmax(1))
        yt = torch.cat([yb for _, yb in loaders["test"]])
        ens = 100.0 * (lg.argmax(1) == yt).float().mean().item()
        print(f"[ens] 2-seed TEST={ens:.2f}", flush=True)
        results.append({k: v for k, v in r2.items() if k != "model"})

    summary = {
        "arms": [{k: v for k, v in r.items() if k != "model"} for r in results],
        "best_arm": best["arm"], "best_test": best["test"],
        "ensemble_test": ens,
        "pipeline_plateau": {"v12_1_gb10_8h": 72.84, "all_time_external": 73.82},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[cryptic_spec] SUMMARY: " +
          ", ".join(f"{r['arm']}={r['test']:.2f}" for r in results) +
          (f", ens={ens:.2f}" if ens else "") +
          f"  (bar: 72.84 pipeline / 73.82 all-time)", flush=True)


if __name__ == "__main__":
    main()
