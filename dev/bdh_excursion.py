#!/usr/bin/env python3
"""BDH excursion: the Dragon Hatchling architecture (pathwaycom/bdh) as a FIXED-
arch, full-budget classifier on the competition datasets -- a family excursion
like the TCN one, NOT a search op (BDH, like recurrence, loses the short-train
screen and only performs under full training; it is tested as a full-budget A/B).

FAITHFUL to the original (baselines/bdh/bdh.py is byte-identical to upstream):
  - the UNMODIFIED BDH model + BDHConfig() defaults (n_layer 6, n_embd 256,
    n_head 4, vocab 256, mlp_mult 128, dropout 0.1);
  - the original recipe: AdamW lr 1e-3 / wd 0.1, bf16 autocast, torch.compile
    (best-effort), seed 1337.
The ONLY adaptation is task I/O (necessary: LM -> classifier), done here, never
in the model:
  - input tensor -> BYTE tokens: min-max quantize to 0..255, flatten C*H*W, and
    stride to <= --max-seq. This is exactly the byte-level sequence the original
    eats. (Caveat: continuous-feature sets like Cryptic's BERT vectors are a poor
    fit for byte-level modelling -- quantization is lossy; logged, not worked
    around, since the point is to test the architecture as-is.)
  - readout (DENSE-supervision variant, "option 2"): last-token-only readout did
    not learn (one label per ~80 tokens = ~80x sparser gradient than the LM's
    next-token training; loss stuck at ln(num_classes)). So classify by MEAN-
    POOLING the unmodified vocab head over all positions (logits.mean(1)[:nc] ->
    gradient to every position) AND add an auxiliary next-byte LM loss
    (LM_AUX * CE on the model's own next-token prediction) for dense self-
    supervision. The MODEL is still unmodified; only the loss/readout around it
    changed. drop_last + eager keep shapes static (the compiled path recompile-
    thrashed on ragged batches -> ~770s/epoch; eager is predictable).
Train-to-budget (wall-clock), like the other excursions, on the UNKNOWN-hardware
competition assumption. Writes a Trainer-style CSV for the curve plots.
"""
import argparse
import csv
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BDH = Path(__file__).resolve().parent.parent / "baselines" / "bdh"
sys.path.insert(0, str(BDH))
import bdh as bdh_mod  # noqa: E402  -- the UNMODIFIED upstream module

V10_REAL = {"Chesseract": 57.66, "Gutenberg": 47.28, "CIFARTile": 73.65,
            "Sudoku": 97.70, "Language": 90.03, "Cauchy": 93.91, "Sedol": 56.55,
            "Sloan": 84.60, "Cryptic": 72.08, "AddNIST": 96.33, "MultNIST": 95.52,
            "GeoClassing": 95.36, "GameOfLife": 100.0, "Myofibre": 91.09,
            "Voxel": 81.56, "Windspeed": 19.36}
SEED = 1337  # the original's seed
LM_AUX = 0.3  # weight of the auxiliary next-byte LM loss (dense self-supervision)


def tokenize(x, max_seq):
    """(N,C,H,W)|(N,H,W) -> (N,T) int64 byte tokens, strided to <= max_seq."""
    x = np.asarray(x)
    if x.ndim == 3:
        x = x[:, None]
    n = x.shape[0]
    flat = x.reshape(n, -1).astype(np.float64)
    lo, hi = float(flat.min()), float(flat.max())
    if hi > lo:
        flat = (flat - lo) / (hi - lo) * 255.0
    tok = np.clip(np.rint(flat), 0, 255).astype(np.int64)
    t = tok.shape[1]
    step = 1
    if t > max_seq:
        step = int(np.ceil(t / max_seq))
        tok = tok[:, ::step][:, :max_seq]
    return tok, tok.shape[1], step


@torch.no_grad()
def evaluate(model, X, Y, nc, bs, device, ctx):
    model.eval()
    correct = total = 0
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(device)
        with ctx:
            logits, _ = model(xb)
        pred = logits.float().mean(1)[:, :nc].argmax(1).cpu().numpy()
        yb = Y[i:i + bs]
        correct += int((pred == yb).sum()); total += len(yb)
    return correct / max(total, 1)


@torch.no_grad()
def predict(model, X, nc, bs, device, ctx):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).to(device)
        with ctx:
            logits, _ = model(xb)
        out.append(logits.float().mean(1)[:, :nc].argmax(1).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hours", type=float, default=1.5)
    # KEEP FULL RESOLUTION by default: striding a flattened grid destroys the 2-D
    # encoding (Chesseract capped-to-128 scored 38.8 vs 53.7 uncapped). max_seq is
    # now just a high OOM ceiling; memory is handled by the VRAM-adaptive BATCH,
    # not by mangling the input. The budget drives the epoch count (train-to-budget)
    # -- BDH is viable on large grids only with a large budget, which is variable
    # (read from the clock), so we do not trade away input structure for epochs.
    ap.add_argument("--max-seq", type=int, default=4608)   # OOM ceiling only (covers the reasoning sets)
    ap.add_argument("--target-epochs", type=int, default=40)
    # opt-in: stride T down to hit --target-epochs (structure-destroying; only for
    # tiny budgets where full-res would get near-zero epochs). Default OFF.
    ap.add_argument("--cap-epochs", action="store_true", default=False)
    ap.add_argument("--batch", type=int, default=128)  # big batches (few steps)
    # faithful default is 128 -> internal N=8192, ~25M params, ~13min/epoch on the
    # GB10 = untrainable in a 1.5h budget. Downscale for the small competition
    # datasets (paper spans 10M-1B; a smaller internal dim is a legit config).
    ap.add_argument("--mlp-mult", type=int, default=8)  # ~1.7M params, comparable to the v10/v11 nets
    # "faithful" = byte-identical upstream BDH (causal O(T^2), last-layer readout).
    # "bidir" = paper-credited B1 (bidirectional O(T) attention) + B2 (all-layer
    # readout) via dev/bdh_bidir.py; unlocks medium-T datasets and small-scale gap.
    ap.add_argument("--model", choices=("faithful", "bidir"), default="faithful")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    d = Path(args.dataset_dir); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_x = np.load(d / "train_x.npy"); tr_y = np.load(d / "train_y.npy")
    va_x = np.load(d / "valid_x.npy"); va_y = np.load(d / "valid_y.npy")
    te_x = np.load(d / "test_x.npy")
    te_y = np.load(d / "test_y.npy") if (d / "test_y.npy").exists() else None
    meta = json.loads((d / "metadata").read_text())
    code = meta.get("codename", d.name); nc = int(meta["num_classes"])
    ref = V10_REAL.get(code) or V10_REAL.get(d.name)

    Xtr, T, step = tokenize(tr_x, args.max_seq)
    Xva, _, _ = tokenize(va_x, args.max_seq)
    Xte, _, _ = tokenize(te_x, args.max_seq)
    Ytr = tr_y.astype(np.int64); Yva = va_y.astype(np.int64)
    # VRAM- and T-adaptive batch (hardware-agnostic): BDH attention is O(T^2),
    # so the dominant activation per example ~= n_head * T^2 * 2 bytes (bf16),
    # times a fudge for the other layer activations + grads. Size the batch to a
    # fraction of the GPU's ACTUAL total VRAM so it self-fits on unknown hardware
    # (GB10 128GB -> big batch; an 8GB card -> small) instead of a hardcoded cap.
    cfg0 = bdh_mod.BDHConfig(mlp_internal_dim_multiplier=args.mlp_mult)
    if torch.cuda.is_available():
        total_vram = torch.cuda.mem_get_info()[1]
    else:
        total_vram = 8 << 30
    per_example = cfg0.n_head * T * T * 2 * 8 + 1   # 8x fudge: layers + grads
    bs = int(max(4, min(args.batch, 0.45 * total_vram / per_example)))
    print(f"[{code}] BDH excursion seed={args.seed} {args.hours}h  T={T} (raw "
          f"{int(np.prod(tr_x.shape[1:]))}, stride {step})  nc={nc}  batch={bs}  v10={ref}", flush=True)

    # bf16 autocast (the original's default on CUDA), no GradScaler needed for bf16
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ctx = (torch.amp.autocast(device_type=device.type,
                              dtype=torch.bfloat16 if use_bf16 else torch.float16)
           if device.type == "cuda" else nullcontext())

    if args.model == "bidir":
        import bdh_bidir  # noqa: E402  -- B1+B2 subclass over the vendored model
        model = bdh_bidir.BidirBDH(cfg0).to(device)
    else:
        model = bdh_mod.BDH(cfg0).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    # torch.compile is the original's throughput optimisation (not architectural);
    # off by default because Spark's inductor lacks Python.h to JIT its CUDA helper.
    # BDH_COMPILE=1 to enable where the toolchain supports it.
    compiled = False
    if os.environ.get("BDH_COMPILE") == "1":
        try:
            model = torch.compile(model); compiled = True
        except Exception as e:
            print(f"[{code}] torch.compile off ({type(e).__name__})", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    print(f"[{code}] BDH params={n_params/1e6:.2f}M compiled={compiled} bf16={use_bf16}", flush=True)

    # --- DYNAMIC sequence cap ------------------------------------------------
    # Measure the real per-step time on THIS hardware, project the epoch count in
    # the budget, and if it is below --target-epochs, stride T down until it isn't.
    # Attention is O(T^2) and the VRAM-adaptive batch scales ~1/T^2, so epoch time
    # ~ T^2: new_T = T * sqrt(proj_epochs / target). Input-adaptive (per-dataset T)
    # AND hardware-adaptive (measured, not assumed) -- fast GPU/short grid keeps
    # full resolution; slow GPU/long grid shrinks so it still trains enough epochs.
    def _step_time(k=4):
        model.train(); ts = []
        for s in range(k):
            xb = torch.from_numpy(Xtr[s * bs:(s + 1) * bs]).to(device)
            yb = torch.from_numpy(Ytr[s * bs:(s + 1) * bs]).to(device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with ctx:
                lo, _ = model(xb); lf = lo.float()
                ls = F.cross_entropy(lf.mean(1)[:, :nc], yb) + LM_AUX * F.cross_entropy(
                    lf[:, :-1, :].reshape(-1, lf.size(-1)), xb[:, 1:].reshape(-1))
            ls.backward(); model.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return min(ts[1:]) if len(ts) > 1 else ts[0]   # skip step 0 (compile warmup)

    # VIABILITY GATE: calibrate the real step time on THIS hardware, project how
    # many epochs full-resolution training gets in the budget. This is the runtime
    # deploy gate for BDH-as-specialist: >= target -> BDH worth running; else it is
    # MARGINAL and a deployment would fall back to the v11.1 pipeline. Logged +
    # recorded always. Only strides the input if --cap-epochs (structure-destroying,
    # off by default -- we keep full resolution and let the budget drive epochs).
    proj_epochs = None; viable = None
    if len(Xtr) > bs * 5 and T > 64:
        st = _step_time()
        proj_epochs = round((args.hours * 3600) / max((len(Xtr) // bs) * st, 1.0), 1)
        viable = bool(proj_epochs >= args.target_epochs)
        print(f"[{code}] viability: T={T} step={st:.2f}s -> ~{proj_epochs:.0f} epochs in "
              f"{args.hours}h (target {args.target_epochs}: {'VIABLE' if viable else 'MARGINAL->would fall back'})",
              flush=True)
        if args.cap_epochs and not viable:
            new_t = max(64, int(T * (proj_epochs / args.target_epochs) ** 0.5))
            st2 = int(np.ceil(T / new_t))
            if st2 > 1:
                Xtr = Xtr[:, ::st2][:, :new_t]; Xva = Xva[:, ::st2][:, :new_t]
                Xte = Xte[:, ::st2][:, :new_t]
                T = Xtr.shape[1]; step *= st2
                per_example = cfg0.n_head * T * T * 2 * 8 + 1
                bs = int(max(4, min(args.batch, 0.45 * total_vram / per_example)))
                print(f"[{code}] --cap-epochs: strided to T={T} bs={bs}", flush=True)

    tag = "" if args.model == "faithful" else f"_{args.model}"   # keep runs from clobbering
    csv_f = open(out / f"{code}{tag}.csv", "w", newline=""); cw = csv.writer(csv_f)
    cw.writerow(["epoch", "train_loss", "val_acc", "best_val_acc", "epoch_time_s", "clock_remaining_s"])

    deadline = time.perf_counter() + args.hours * 3600
    rng = np.random.default_rng(args.seed)
    best_val, best_state, best_ep = -1.0, None, 0
    epoch = 0
    while time.perf_counter() < deadline - 60:
        model.train(); t0 = time.perf_counter(); losses = []
        idx = rng.permutation(len(Xtr))
        for i in range(0, len(idx) - bs + 1, bs):   # drop_last: only full batches (static shape)
            if time.perf_counter() > deadline - 60:
                break
            j = idx[i:i + bs]
            xb = torch.from_numpy(Xtr[j]).to(device)
            yb = torch.from_numpy(Ytr[j]).to(device)
            with ctx:
                logits, _ = model(xb)
                lf = logits.float()
                # DENSE supervision (the fix for last-token-only not learning):
                # mean-pool the vocab head over all positions for classification,
                # plus an auxiliary next-byte LM loss so every position gets gradient.
                cls_loss = F.cross_entropy(lf.mean(1)[:, :nc], yb)
                lm_loss = F.cross_entropy(lf[:, :-1, :].reshape(-1, lf.size(-1)),
                                          xb[:, 1:].reshape(-1))
                loss = cls_loss + LM_AUX * lm_loss
            if not torch.isfinite(loss):
                raise RuntimeError("NaN")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            losses.append(float(loss.item()))
        val = evaluate(model, Xva, Yva, nc, bs, device, ctx)
        dt = time.perf_counter() - t0; rem = deadline - time.perf_counter()
        if val > best_val:
            best_val = val; best_ep = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        tl = sum(losses) / max(len(losses), 1)
        cw.writerow([epoch, f"{tl:.6f}", f"{val:.6f}", f"{best_val:.6f}", f"{dt:.3f}", f"{rem:.1f}"]); csv_f.flush()
        print(f"  epoch {epoch:3d}  loss={tl:.4f}  val={val:.4f}  best={best_val:.4f}  "
              f"epoch_time={dt:.1f}s  clock={rem:.0f}s", flush=True)
        epoch += 1
    csv_f.close()
    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc = None
    if te_y is not None:
        pred = predict(model, Xte, nc, bs, device, ctx)
        n = min(len(pred), len(te_y))
        test_acc = round(100.0 * float((pred[:n] == te_y[:n].astype(np.int64)).mean()), 3)
    rec = {"codename": code, "dataset": d.name, "pipeline": "bdh_excursion",
           "model": args.model, "seed": args.seed,
           "hours": args.hours, "params_M": round(n_params / 1e6, 3), "seq_len": T,
           "raw_seq": int(np.prod(tr_x.shape[1:])), "stride": step, "compiled": compiled,
           "readout": "meanpool+auxLM", "lm_aux": LM_AUX, "mlp_mult": args.mlp_mult,
           "proj_epochs": proj_epochs, "viable": viable, "target_epochs": args.target_epochs,
           "epochs": epoch, "best_val_acc": round(best_val, 4),
           "full_budget_test_acc": test_acc, "v10_real_test_acc": ref,
           "beats_v10_real": (test_acc > ref) if (test_acc is not None and ref) else None,
           "margin_vs_v10": round(test_acc - ref, 3) if (test_acc is not None and ref) else None}
    (out / f"{code}_bdh{tag}.json").write_text(json.dumps(rec, indent=2))
    print(f"[{code}] BDH DONE: test {test_acc}%  vs v10 {ref}%  margin {rec['margin_vs_v10']}  "
          f"(best_val {best_val:.4f}, {epoch} epochs, {n_params/1e6:.2f}M params)", flush=True)


if __name__ == "__main__":
    main()
