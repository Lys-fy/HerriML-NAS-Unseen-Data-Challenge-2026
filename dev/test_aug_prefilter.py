#!/usr/bin/env python3
"""Phase A3: can zero-cost proxies replicate the aug-safety probe's decision?

The EG-ENAS idea ([[2025 - EG-ENAS]]): score augmentations with zero-cost
proxies on a minibatch instead of training probe arms. Our trusted reference:
the two-arm aug-safety probe's logged decisions (3 seeds of v7).

Test: for each dataset, compute naswot + snip on a CLEAN minibatch and on an
AUGMENTED minibatch (the exact transforms our DataProcessor would apply),
through a few small random conv nets. Proxy decision: aug ON if the augmented
batch scores >= clean. Compare against the probe's per-dataset decision.

Pass criterion (predeclared): agreement on the datasets where the probe was
unanimous across seeds. Disagreement on split-decision datasets is tolerated.
Runs on spark, a few seconds per dataset, CPU or GPU.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "baselines" / "fixed_arch_probe"))

from helpers import build_default_cnn, aug_policy_for  # noqa: E402
from zeroshot import naswot_score, snip_score          # noqa: E402

DATA = Path.home() / "local" / "nas-comp-datasets"
DATASETS = ["AddNIST", "CIFARTile", "Chesseract", "Cryptic", "GameOfLife",
            "GeoClassing", "Gutenberg", "Language", "MultNIST", "Myofibre",
            "Sudoku", "Voxel", "Windspeed"]
# dataset name -> codename used in the probe logs (authoritative mapping)
CODENAME = {"AddNIST": "Adaline", "CIFARTile": "Caitie", "Chesseract": "Chester",
            "Cryptic": "Cryptic", "GameOfLife": "conway", "GeoClassing": "Sadie",
            "Gutenberg": "Gutenberg", "Language": "LaMelo", "MultNIST": "Mateo",
            "Myofibre": "Myopia", "Sudoku": "Sokoto", "Voxel": "Volga",
            "Windspeed": "windspeed"}
BATCH = 64
N_MODELS = 5
RNG = np.random.default_rng(42)


def probe_decisions():
    """Collect the logged two-arm probe decisions (v7, 3 seeds)."""
    out = {}
    for seed in (42, 1337, 2024):
        d = REPO / f"dev/logs/tier2a_p1_v7/p1v7_s{seed}"
        if not d.exists():
            continue
        for f in d.glob("*_p1_probe.json"):
            code = f.name.replace("_p1_probe.json", "")
            try:
                j = json.loads(f.read_text())
            except Exception:
                continue
            dec = None
            def find_dec(o):
                nonlocal dec
                if isinstance(o, dict):
                    if "decision" in o and isinstance(o["decision"], str):
                        dec = o["decision"]
                    for v in o.values():
                        find_dec(v)
                elif isinstance(o, list):
                    for v in o:
                        find_dec(v)
            find_dec(j)
            if dec:
                out.setdefault(code, []).append(dec)
    return out


def augment_batch(x, policy):
    """Apply the policy's label-preserving transforms (crop shift + cutout),
    mirroring AugmentedTensorDataset's ops on a whole batch."""
    n, c, h, w = x.shape
    out = x.clone()
    if getattr(policy, "crop_pad", 0) > 0:
        p = policy.crop_pad
        padded = torch.nn.functional.pad(out, (p, p, p, p), mode="reflect")
        for i in range(n):
            dy, dx = RNG.integers(0, 2 * p + 1, size=2)
            out[i] = padded[i, :, dy:dy + h, dx:dx + w]
    if getattr(policy, "cutout_frac", 0.0) > 0:
        ch_pix = max(1, int(h * policy.cutout_frac))
        cw_pix = max(1, int(w * policy.cutout_frac))
        for i in range(n):
            cy = int(RNG.integers(0, h)); cx = int(RNG.integers(0, w))
            y0, y1 = max(0, cy - ch_pix // 2), min(h, cy + ch_pix // 2 + 1)
            x0, x1 = max(0, cx - cw_pix // 2), min(w, cx + cw_pix // 2 + 1)
            out[i, :, y0:y1, x0:x1] = 0.0
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probes = probe_decisions()
    print(f"{'dataset':12s} {'proxy':6s} {'probe(seeds)':14s} verdict")
    agree_unanimous, n_unanimous = 0, 0
    for name in DATASETS:
        try:
            xs = np.load(DATA / name / "train_x.npy", mmap_mode="r")
            ys = np.load(DATA / name / "train_y.npy", mmap_mode="r")
            meta_path = DATA / name / "metadata"
            meta = {}
            if meta_path.exists():
                for line in meta_path.read_text().splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        meta[k.strip()] = v.strip()
            idx = np.sort(RNG.choice(xs.shape[0], size=min(BATCH, xs.shape[0]),
                                     replace=False))
            x = torch.tensor(np.asarray(xs[idx], dtype=np.float32))
            y = torch.tensor(np.asarray(ys[idx]).astype(np.int64))
            if x.ndim == 3:
                x = x[:, None]
            mu = x.mean(dim=(0, 2, 3), keepdim=True)
            sd = x.std(dim=(0, 2, 3), keepdim=True) + 1e-6
            x = (x - mu) / sd
            n, c, h, w = x.shape
            nc = int(y.max().item()) + 1
            policy = aug_policy_for({"input_shape": [n, c, h, w],
                                     "num_classes": nc, "codename": name})
            x_aug = augment_batch(x, policy)

            score_clean, score_aug = 0.0, 0.0
            for m in range(N_MODELS):
                torch.manual_seed(1000 + m)
                net = build_default_cnn(c, nc, h, w, base_width=16,
                                        n_blocks=3, drop_path_rate=0.0).to(device)
                xc, xa, yy = x.to(device), x_aug.to(device), y.to(device)
                for which, batch in (("clean", xc), ("aug", xa)):
                    s = 0.0
                    try:
                        s += float(naswot_score(net, batch))
                    except Exception:
                        pass
                    try:
                        s += float(snip_score(net, batch, yy))
                    except Exception:
                        pass
                    if which == "clean":
                        score_clean += s
                    else:
                        score_aug += s
            proxy = "on" if score_aug >= score_clean else "off"
            pr = probes.get(CODENAME.get(name, name), [])
            pstr = ",".join(pr) if pr else "?"
            unanimous = len(pr) == 3 and len(set(pr)) == 1
            verdict = ""
            if unanimous:
                n_unanimous += 1
                if pr[0] == proxy:
                    agree_unanimous += 1
                    verdict = "AGREE"
                else:
                    verdict = "DISAGREE"
            print(f"{name:12s} {proxy:6s} {pstr:14s} {verdict}", flush=True)
        except Exception as ex:
            print(f"{name:12s} FAILED {type(ex).__name__}: {ex}", flush=True)
    if n_unanimous:
        print(f"\nagreement on unanimous-probe datasets: "
              f"{agree_unanimous}/{n_unanimous}")
    else:
        print("\nWARNING: no unanimous probe decisions matched by name; "
              "codename mapping needed for the comparison")


if __name__ == "__main__":
    main()
