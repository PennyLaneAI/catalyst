#!/usr/bin/env python3
"""
Phase 5 Task 1 — Power-law graph benchmark
==========================================
Run DO-QAOA on Barabási-Albert graphs N=4..20, m=1,2,3.

Paper targets (Sang et al., Table IV — Power-law graphs):
  ARG (Approximation Ratio Gap) = 100 × |E_min − ⟨H_C⟩| / |E_min|
    m=1 → ARG ≈ 52%   (median across N=4..20)
    m=2 → ARG ≈ 37%
    m=3 → ARG ≈ 26%   ← primary target

  Shots (aggregate over all N=4..20 runs):
    Total shots ≤ 0.17 × 10⁶ = 170,000

E_min is computed by brute-force enumeration of all 2^(N−m) free-qubit
bitstrings for the best sub-problem k*, giving the classical minimum of the
sub-problem Ising Hamiltonian.  ⟨H_C⟩ is the p=1 QAOA expectation from
DO-QAOA's optimised parameters.

Assertions:
  1. Median ARG ≤ 55% for m=1   (paper ≈ 52%)
  2. Median ARG ≤ 40% for m=2   (paper ≈ 37%)
  3. Median ARG ≤ 30% for m=3   (paper ≈ 26%)
  4. Aggregate total shots ≤ 170,000 across all (N, m=3) runs

Run as:  python phase5_task1_power_law_benchmark.py
"""

import importlib.util
import math
import statistics
import sys
import time

import networkx as nx
import numpy as np
import pennylane as qml

# ── Load DO-QAOA module ───────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location(
    "doqaoa", "frontend/catalyst/api_extensions/doqaoa.py"
)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)

do_qaoa = _mod.do_qaoa
DOQAOAResult = _mod.DOQAOAResult
select_hotspot_indices = _mod.select_hotspot_indices
extract_coupling_matrix = _mod.extract_coupling_matrix

# ── Parameters ────────────────────────────────────────────────────────────────
N_VALUES = list(range(4, 21))  # N = 4 … 20
M_VALUES = [1, 2, 3]
BA_M_EDGE = 2
SEED = 42

# Paper Table IV approximate median ARG targets (%) and assertion thresholds
ARG_TARGET = {1: 52, 2: 37, 3: 26}
ARG_ASSERT = {1: 55, 2: 40, 3: 30}  # slightly relaxed for graph-to-graph variance
SHOTS_BUDGET = 170_000  # aggregate shots across all N=4..20 for m=3


# ── Brute-force sub-problem E_min ─────────────────────────────────────────────
def emin_subproblem(cost_h, hotspot_indices, N, k_idx):
    """Classical minimum energy of sub-problem k_idx by full enumeration."""
    J_dict, h_dict = extract_coupling_matrix(cost_h)
    J_mat = np.zeros((N, N))
    for (i, j), v in J_dict.items():
        J_mat[i, j] = v
        J_mat[j, i] = v
    h_vec = np.array([h_dict.get(i, 0.0) for i in range(N)])

    hs = list(hotspot_indices)
    m = len(hs)
    frozen = {hs[i]: (-1 if (k_idx >> i) & 1 else 1) for i in range(m)}
    free = [q for q in range(N) if q not in frozen]
    nf = len(free)

    h_eff = np.array([h_vec[q] + sum(J_mat[q, fq] * frozen[fq] for fq in frozen) for q in free])
    J_free = np.array([[J_mat[free[i], free[j]] for j in range(nf)] for i in range(nf)])

    best = float("inf")
    for bits in range(1 << nf):
        spins = np.array([1 - 2 * ((bits >> i) & 1) for i in range(nf)], dtype=float)
        e = sum(J_free[i, j] * spins[i] * spins[j] for i in range(nf) for j in range(i + 1, nf))
        e += float(h_eff @ spins)
        if e < best:
            best = e
    return best


print("=" * 70)
print("Phase 5 Task 1 — Power-law (Barabási-Albert) Graph Benchmark")
print(f"N ∈ {{{N_VALUES[0]}..{N_VALUES[-1]}}},  m ∈ {M_VALUES},  BA attachment m_ba={BA_M_EDGE}")
print(f"ARG = 100 × |E_min − ⟨H_C⟩| / |E_min|  (Eq. 4.1, Sang et al. 2026)")
print("=" * 70)

# ── Results storage ───────────────────────────────────────────────────────────
args_by_m = {1: [], 2: [], 3: []}
shots_by_m = {1: [], 2: [], 3: []}
passed = True
t_global = time.perf_counter()

# ── Sweep ─────────────────────────────────────────────────────────────────────
for m in M_VALUES:
    print(f"\n{'─'*70}")
    print(f"  m = {m}  (target median ARG ≈ {ARG_TARGET[m]}%,  assert ≤ {ARG_ASSERT[m]}%)")
    print(
        f"  {'N':>3}  {'edges':>5}  {'shots':>6}  {'⟨H_C⟩':>8}  {'E_min':>8}  {'ARG%':>6}  status"
    )
    print(f"  {'─'*3}  {'─'*5}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*6}")

    for N in N_VALUES:
        if m >= N:
            print(f"  {N:>3}  {'n/a':>5}  {'─':>6}  {'─':>8}  {'─':>8}  {'─':>6}  SKIP (m≥N)")
            continue

        G = nx.barabasi_albert_graph(N, BA_M_EDGE, seed=SEED + N)
        cost_h, mixer_h = qml.qaoa.maxcut(G)
        dev = qml.device("default.qubit", wires=N)

        @qml.qnode(dev)
        def circuit(params):
            for w in range(N):
                qml.Hadamard(wires=w)
            qml.qaoa.cost_layer(params[0], cost_h)
            qml.qaoa.mixer_layer(params[1], mixer_h)
            return qml.expval(cost_h)

        res = do_qaoa(
            circuit,
            cost_h,
            m=m,
            full_epochs=100,
            warmstart_epochs=10,
            learning_rate=0.01,
            grad_norm_tol=1e-4,
            seed=SEED,
            max_warmstarts=1,
            bias_threshold=0.3,
        )(G)

        hs = list(select_hotspot_indices(G, m))
        emin = emin_subproblem(cost_h, hs, N, res.best_k)
        arg = 100 * abs(emin - res.best_energy) / abs(emin) if abs(emin) > 1e-9 else 0.0

        args_by_m[m].append(arg)
        shots_by_m[m].append(res.total_shots)

        ok = res.best_energy < 0
        status = "ok" if ok else "FAIL"
        print(
            f"  {N:>3}  {G.number_of_edges():>5}  {res.total_shots:>6,}  "
            f"{res.best_energy:>8.4f}  {emin:>8.4f}  {arg:>6.1f}  {status}"
        )

    if args_by_m[m]:
        med = statistics.median(args_by_m[m])
        mean = sum(args_by_m[m]) / len(args_by_m[m])
        agg_s = sum(shots_by_m[m])
        print(
            f"\n  Median ARG = {med:.1f}%   Mean = {mean:.1f}%   "
            f"Paper target ≈ {ARG_TARGET[m]}%   Assert ≤ {ARG_ASSERT[m]}%"
        )
        print(f"  Aggregate shots (m={m}): {agg_s:,}")

# ── Global assertions ─────────────────────────────────────────────────────────
elapsed = time.perf_counter() - t_global
print(f"\n{'='*70}")
print(f"  Total wall-clock time: {elapsed:.1f}s")
print()

for m in M_VALUES:
    if not args_by_m[m]:
        continue
    med = statistics.median(args_by_m[m])
    thr = ARG_ASSERT[m]
    if med <= thr:
        print(f"PASS: m={m} median ARG {med:.1f}% ≤ {thr}%  (paper ≈ {ARG_TARGET[m]}%)  ✓")
    else:
        print(f"FAIL: m={m} median ARG {med:.1f}% > {thr}%  (paper ≈ {ARG_TARGET[m]}%)  ✗")
        passed = False

# Aggregate shots for m=3
agg_m3 = sum(shots_by_m[3])
if agg_m3 <= SHOTS_BUDGET:
    print(f"PASS: aggregate shots m=3 = {agg_m3:,} ≤ {SHOTS_BUDGET:,}  ✓")
else:
    print(f"FAIL: aggregate shots m=3 = {agg_m3:,} > {SHOTS_BUDGET:,}  ✗")
    passed = False

print()
if passed:
    print("PHASE 5 TASK 1 PASS ✓  — Power-law benchmark complete")
    print(
        f"  ARG medians: m=1={statistics.median(args_by_m[1]):.1f}%  "
        f"m=2={statistics.median(args_by_m[2]):.1f}%  "
        f"m=3={statistics.median(args_by_m[3]):.1f}%"
    )
    print(f"  Aggregate shots (m=3): {agg_m3:,}")
else:
    print("PHASE 5 TASK 1 FAIL ✗  — see errors above")
    sys.exit(1)
