"""
dev/mock_harness.py — local mirror of evaluation/main.py.

Mimics how the organisers' grader invokes our submission, so that anything
passing here has a high chance of passing on their servers.

Differences from the real grader:
  - Adds submission/ to sys.path before importing (so this script can live
    outside the package).
  - Datasets read from $DATASETS_DIR (default <repo>/datasets/).
  - Predictions written to <repo>/dev/predictions/.
  - --score flag computes the per-dataset adjusted score against local
    test_y.npy (the real grader strips test_y.npy from the package before
    running, then scores separately in score.py).

The Clock class, the call sequence, the test_loader assertions, and the
try/except handling around run_submission are kept identical to the grader.

Usage:
    python dev/mock_harness.py                          # run all datasets in datasets/
    python dev/mock_harness.py Sudoku GameOfLife        # filter to specific datasets
    python dev/mock_harness.py --time-limit 0.05        # override per-dataset budget (3 min)
    python dev/mock_harness.py --score                  # also compute score after running
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle as pkl
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMISSION_DIR = REPO_ROOT / "submission"
DEFAULT_DATASETS_DIR = REPO_ROOT / "datasets"
DEFAULT_PREDICTIONS_DIR = REPO_ROOT / "dev" / "predictions"

# Submission imports must succeed exactly the way the grader's main.py imports.
sys.path.insert(0, str(SUBMISSION_DIR))

import torch  # noqa: E402
from torch.utils.data import RandomSampler  # noqa: E402

from nas import NAS  # noqa: E402
from data_processor import DataProcessor  # noqa: E402
from trainer import Trainer  # noqa: E402


# --- Clock and helpers (kept identical to evaluation/main.py) -----------------


class Clock:
    def __init__(self, time_limit_in_hours: float):
        self.start_time = time.perf_counter()
        self.time_limit = self.start_time + (time_limit_in_hours * 60 * 60)

    def check(self) -> float:
        return self.time_limit - time.perf_counter()


def show_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m,{s}s"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h,{m}m,{s}s"


def general_num_params(model) -> int:
    return int(sum(np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())))


# --- Dataset loading ----------------------------------------------------------


def load_dataset(dataset_path: Path):
    train_x = np.load(dataset_path / "train_x.npy")
    train_y = np.load(dataset_path / "train_y.npy")
    valid_x = np.load(dataset_path / "valid_x.npy")
    valid_y = np.load(dataset_path / "valid_y.npy")
    test_x = np.load(dataset_path / "test_x.npy")
    metadata = json.loads((dataset_path / "metadata").read_text())
    return (train_x, train_y), (valid_x, valid_y), test_x, metadata


# --- Failure handling (kept identical in shape) -------------------------------


def fail_dataset(metadata: dict, predictions_dir: Path) -> None:
    print(f"Dataset {metadata['codename']} failed.")
    stats = {"Failed": True, "Runtime": -1, "Params": None}
    (predictions_dir / f"{metadata['codename']}_stats.pkl").write_bytes(pkl.dumps(stats))


def is_out_of_time(clock: Clock, metadata: dict, predictions_dir: Path, grace: bool = False) -> bool:
    if clock.check() < 0:
        if grace and clock.check() > -60:
            print("Out of time but within 1-minute grace window — predictions still ran (organisers may apply penalty).")
        print("\n=== Skipping dataset — out of time ===\n")
        fail_dataset(metadata, predictions_dir)
        return True
    return False


# --- The actual pipeline (mirrors run_submission in evaluation/main.py) -------


def run_submission(runclock: Clock, dataset_path: Path, predictions_dir: Path) -> None:
    immutable = copy.deepcopy(runclock)
    (train_x, train_y), (valid_x, valid_y), test_x, metadata = load_dataset(dataset_path)
    metadata["time_remaining"] = immutable.check()
    start = time.perf_counter()

    print(f"\n========== Dataset {metadata['codename']:^10} ==========")
    print("  Metadata:")
    for k, v in metadata.items():
        print(f"   - {k:<20}: {v}")

    print("\n=== Processing Data ===")
    print(f"  Time remaining: ~{show_time(immutable.check())}")
    if is_out_of_time(immutable, metadata, predictions_dir):
        return
    data_processor = DataProcessor(train_x, train_y, valid_x, valid_y, test_x, metadata, runclock)
    train_loader, valid_loader, test_loader = data_processor.process()
    metadata["time_remaining"] = immutable.check()

    # Grader assertions — these crash the whole submission if violated.
    assert_msg = "Test Dataloader is {}, this will break evaluation. Please fix this in your DataProcessor init."
    assert not isinstance(test_loader.sampler, RandomSampler), assert_msg.format("shuffling")
    assert not test_loader.drop_last, assert_msg.format("dropping last batch")

    print("\n=== Performing NAS ===")
    print(f"  Time remaining: ~{show_time(immutable.check())}")
    if is_out_of_time(immutable, metadata, predictions_dir):
        return
    model = NAS(train_loader, valid_loader, metadata, runclock).search()
    n_params = general_num_params(model)
    metadata["time_remaining"] = immutable.check()

    print("\n=== Training ===")
    print(f"  Time remaining: ~{show_time(immutable.check())}")
    if is_out_of_time(immutable, metadata, predictions_dir):
        return
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    trainer = Trainer(model, device, train_loader, valid_loader, metadata, runclock)
    trainer.train()
    if is_out_of_time(immutable, metadata, predictions_dir, grace=True):
        return

    print("\n=== Predicting ===")
    print(f"  Time remaining: ~{show_time(runclock.check())}")
    predictions = trainer.predict(test_loader)
    runtime = time.perf_counter() - start

    print(f"  Run time for {metadata['codename']}: {runtime:.2f}s")
    stats = {"Failed": False, "Runtime": round(runtime, 2), "Params": n_params}
    (predictions_dir / f"{metadata['codename']}_stats.pkl").write_bytes(pkl.dumps(stats))
    np.save(predictions_dir / f"{metadata['codename']}.npy", np.asarray(predictions))
    print(f"  Wrote {len(predictions)} predictions for {metadata['codename']}.")


# --- Optional scoring (uses local test_y.npy; the real grader scores in score.py)


def score_predictions(datasets_dir: Path, predictions_dir: Path) -> None:
    from sklearn.metrics import accuracy_score

    print("\n========== Scoring ==========")
    total = 0.0
    for ds in sorted(datasets_dir.iterdir()):
        if not ds.is_dir() or not (ds / "metadata").exists():
            continue
        metadata = json.loads((ds / "metadata").read_text())
        codename = metadata["codename"]
        stats_path = predictions_dir / f"{codename}_stats.pkl"
        if not stats_path.exists():
            continue
        stats = pkl.loads(stats_path.read_bytes())
        if stats["Failed"]:
            print(f"  {codename:>15}: FAILED   adj_score = -10.00")
            total += -10
            continue
        labels = np.load(ds / "test_y.npy")
        preds = np.load(predictions_dir / f"{codename}.npy")
        labels = labels[: len(preds)]
        raw = 100.0 * accuracy_score(labels, preds)
        benchmark = float(metadata.get("benchmark", 0.0))
        scaling = 10.0 / max(100.0 - benchmark, 1e-6)
        adj = max(-10.0, (raw - benchmark) * scaling)
        print(f"  {codename:>15}: raw={raw:6.3f}%  benchmark={benchmark:6.3f}%  adj={adj:+7.3f}  runtime={stats['Runtime']:.1f}s")
        total += adj
    print(f"\n  Total score: {total:+.3f}")


# --- CLI ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("datasets", nargs="*", help="Dataset directory names (default: all under datasets/).")
    parser.add_argument("--datasets-dir", default=os.environ.get("DATASETS_DIR", str(DEFAULT_DATASETS_DIR)))
    parser.add_argument("--predictions-dir", default=str(DEFAULT_PREDICTIONS_DIR))
    parser.add_argument("--time-limit", type=float, default=None,
                        help="Override per-dataset time limit in hours (default: from metadata, fallback 0.5).")
    parser.add_argument("--score", action="store_true",
                        help="Compute scores after running (requires local test_y.npy and sklearn).")
    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir).resolve()
    predictions_dir = Path(args.predictions_dir).resolve()
    predictions_dir.mkdir(parents=True, exist_ok=True)

    if args.datasets:
        targets = [datasets_dir / name for name in args.datasets]
        for t in targets:
            if not (t / "metadata").exists():
                print(f"  WARN: {t} has no metadata file, skipping", file=sys.stderr)
        targets = [t for t in targets if (t / "metadata").exists()]
    else:
        targets = [p for p in sorted(datasets_dir.iterdir()) if p.is_dir() and (p / "metadata").exists()]

    if not targets:
        print(f"No datasets found in {datasets_dir}", file=sys.stderr)
        sys.exit(1)

    print("=" * 78)
    print("=" * 13 + "    Local mock harness — NAS Unseen Data Challenge      " + "=" * 13)
    print("=" * 78)
    print(f"  datasets_dir:    {datasets_dir}")
    print(f"  predictions_dir: {predictions_dir}")
    print(f"  targets:         {[t.name for t in targets]}")
    print(f"  cuda available:  {torch.cuda.is_available()}")

    for ds_path in targets:
        metadata = json.loads((ds_path / "metadata").read_text())
        time_limit = args.time_limit if args.time_limit is not None else float(metadata.get("time_limit", 0.5))
        runclock = Clock(time_limit)
        try:
            run_submission(runclock, ds_path, predictions_dir)
        except Exception as ex:
            print(f"Exception while running {ds_path.name}: {ex}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            fail_dataset(metadata, predictions_dir)
        print(f"Finished dataset: {metadata['codename']}")

    if args.score:
        score_predictions(datasets_dir, predictions_dir)


if __name__ == "__main__":
    main()
