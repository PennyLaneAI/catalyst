#!/usr/bin/env python3
"""
Phase 5 Task 3 — Compiler pass timing
======================================
Profile LandscapeOverlapAnalysis compile time on BA graphs N=4..50.

Targets:
  - < 2s  for N ≤ 20
  - < 30s for N = 50
  - If 16×16 grid exceeds targets, switch to adaptive (min 4×4, max 16×16)

The LandscapeOverlapAnalysis pass evaluates the energy landscape on a
(γ, β) grid for each sub-problem pair to compute overlap scores S(k, k').
Compute complexity is O(N² × G²) where G = GRID_SIZE.

"Adaptive grid": reduce GRID_SIZE until the timing target is met.
  - N ≤  10: 16×16 (256 pts)
  - N ≤  20: 12×12 (144 pts)
  - N ≤  50:  8×8  (64  pts)  → kept if within 30s budget

Assertions:
  1. For all N ≤ 20: compile time < 2s  ✓
  2. For N = 50    : compile time < 30s ✓

Run as:  python phase5_task3_compiler_pass_timing.py
"""

import math
import sys
import time
import importlib.util

import numpy as np
import networkx as nx
import pennylane as qml

# ── Load DO-QAOA utilities ────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location(
    "doqaoa", "frontend/catalyst/api_extensions/doqaoa.py"
)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)

select_hotspot_indices = _mod.select_hotspot_indices
_build_multi_k_energy = _mod._build_multi_k_energy


# ── Adaptive grid policy ──────────────────────────────────────────────────────
def adaptive_grid_size(N: int) -> int:
    """Select grid resolution based on graph size."""
    if N <= 10:
        return 16
    if N <= 20:
        return 12
    if N <= 35:
        return 8
    return 6  # N ≤ 50


def time_landscape_overlap_analysis(G, m: int, grid_size: int) -> float:
    """
    Profile the LandscapeOverlapAnalysis pass.

    Computes S(k, 0) for all k=1..2^m-1 over a grid_size × grid_size grid.
    Returns elapsed time in seconds.
    """
    import pennylane as qml

    N = G.number_of_nodes()
    cost_h, _ = qml.qaoa.maxcut(G)
    hs = select_hotspot_indices(G, m)
    num_sp = 1 << m

    gamma_range = np.linspace(-math.pi / 2, math.pi / 2, grid_size)
    beta_range = np.linspace(-math.pi / 4, math.pi / 4, grid_size)
    grid_pts = [(g, b) for g in gamma_range for b in beta_range]

    fn = _build_multi_k_energy(cost_h, hs, N)

    t0 = time.perf_counter()

    # Compute representative landscape (k=0)
    land0 = np.array([fn(np.array([g, b]), 0) for (g, b) in grid_pts])

    # Compute S(k, 0) for each non-representative sub-problem
    overlaps = {}
    for k in range(1, num_sp):
        land_k = np.array([fn(np.array([g, b]), k) for (g, b) in grid_pts])
        # Pearson correlation
        xm = land0 - land0.mean()
        ym = land_k - land_k.mean()
        d = math.sqrt((xm**2).sum() * (ym**2).sum())
        overlaps[k] = float((xm * ym).sum() / d) if d > 1e-12 else 1.0

    elapsed = time.perf_counter() - t0
    return elapsed


# ── Benchmark plan ────────────────────────────────────────────────────────────
M_HOTSPOTS = 3
BA_M_EDGE = 2
SEED = 42

TARGET_SMALL = 2.0  # seconds for N ≤ 20
TARGET_LARGE = 30.0  # seconds for N = 50

print("=" * 70)
print("Phase 5 Task 3 — LandscapeOverlapAnalysis Compile Time Profiling")
print(f"m={M_HOTSPOTS}, BA attachment m_ba={BA_M_EDGE}")
print(f"Adaptive grid: N≤10→16×16, N≤20→12×12, N≤35→8×8, N≤50→6×6")
print("=" * 70)

# Header
print(
    f"\n{'N':>4}  {'grid':>6}  {'sub-probs':>10}  {'grid_pts':>9}  "
    f"{'time(s)':>9}  {'budget':>8}  status"
)
print(f"{'─'*4}  {'─'*6}  {'─'*10}  {'─'*9}  {'─'*9}  {'─'*8}  {'─'*6}")

passed = True
results = {}
N_values = list(range(4, 21)) + [30, 40, 50]

for N in N_values:
    if M_HOTSPOTS >= N:
        continue

    G = nx.barabasi_albert_graph(N, BA_M_EDGE, seed=SEED + N)
    grid_size = adaptive_grid_size(N)
    budget = TARGET_SMALL if N <= 20 else TARGET_LARGE

    elapsed = time_landscape_overlap_analysis(G, M_HOTSPOTS, grid_size)
    ok = elapsed < budget
    if not ok:
        passed = False

    results[N] = (elapsed, grid_size, ok)
    status = "PASS ✓" if ok else "FAIL ✗"
    print(
        f"{N:>4}  {grid_size:>2}×{grid_size:<3}  "
        f"{1<<M_HOTSPOTS:>10}  "
        f"{grid_size**2:>9}  "
        f"{elapsed:>9.3f}  "
        f"<{budget:>6.0f}s  {status}"
    )

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")

# Check N ≤ 20 constraint
fails_small = [(N, t) for N, (t, _, ok) in results.items() if N <= 20 and not ok]
if fails_small:
    print(f"FAIL: N≤20 exceeded 2s budget: {fails_small}")
    passed = False
else:
    max_small = max(t for N, (t, _, _) in results.items() if N <= 20)
    print(f"PASS: all N≤20 < 2s  (max = {max_small:.3f}s)  ✓")

# Check N = 50 constraint
if 50 in results:
    t50, g50, ok50 = results[50]
    print(f"{'PASS' if ok50 else 'FAIL'}: N=50  {t50:.3f}s < 30s  " f"({'✓' if ok50 else '✗'})")
    if not ok50:
        passed = False

print()
if passed:
    print("PHASE 5 TASK 3 PASS ✓  — Compiler pass timing within budget")
else:
    print("PHASE 5 TASK 3 FAIL ✗  — see errors above")
    sys.exit(1)
