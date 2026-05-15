"""Internal benchmark overrides for tier-to-tier comparisons.

The shipped Sudoku metadata has `benchmark: 0.0` — flagged in the
docs/notes Datasets Overview as a placeholder, not a real ResNet-18 number.
This makes Sudoku's adj score uninformative for tracking progress across
tiers (every tier's Sudoku score is just raw_acc / 10, with a tiny slope
that drowns real improvements in numerical precision).

We override Sudoku's benchmark with Tier 0's raw test accuracy, giving
Tier 0 an internal adj of 0 on Sudoku and Tier 1+ a meaningful slope
(slope = 10 / (100 - 70.95) ≈ 0.344 per pp instead of 0.1 per pp).

Used by analysis / comparison scripts ONLY. mock_harness.py must remain
faithful to the shipped metadata, because that's what the real grader
will use on the eval server.
"""

from __future__ import annotations

# Map of codename -> (override_benchmark_pct, reason)
INTERNAL_BENCHMARK_OVERRIDES = {
    "Sokoto": (
        70.95,
        "Tier 0 raw test acc (sweep 2026-05-12). "
        "Shipped metadata benchmark=0.0 is a placeholder per docs/notes/03 - Datasets/Datasets Overview.md.",
    ),
}


def resolve_benchmark(codename: str, shipped_value: float) -> float:
    """Return the internal-comparison benchmark for codename.

    Falls back to the shipped value if no override exists.
    """
    if codename in INTERNAL_BENCHMARK_OVERRIDES:
        return INTERNAL_BENCHMARK_OVERRIDES[codename][0]
    return shipped_value


def adj_score(raw_pct: float, benchmark_pct: float) -> float:
    """Standard scoring formula. Clipped to [-10, +10]."""
    if benchmark_pct >= 100.0:
        return 0.0
    adj = (raw_pct - benchmark_pct) * 10.0 / (100.0 - benchmark_pct)
    return max(-10.0, min(10.0, adj))


def adj_score_internal(codename: str, raw_pct: float, shipped_benchmark_pct: float) -> float:
    """Compute adj using the internal benchmark when one exists, else shipped."""
    return adj_score(raw_pct, resolve_benchmark(codename, shipped_benchmark_pct))


if __name__ == "__main__":
    # Quick sanity check / demo when run directly.
    print("Internal benchmark overrides:")
    for codename, (value, reason) in INTERNAL_BENCHMARK_OVERRIDES.items():
        print(f"  {codename}: {value}%  ({reason})")
    print()
    print("Demo — Sudoku scoring under both lenses:")
    for raw in (70.95, 75.0, 80.0):
        shipped = adj_score(raw, 0.0)
        internal = adj_score_internal("Sokoto", raw, 0.0)
        print(f"  raw={raw:6.2f}%   shipped adj={shipped:+.3f}   internal adj={internal:+.3f}")
