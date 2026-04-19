#!/usr/bin/env python3
"""
Acceptance Criterion 5 — ARG on Erdős-Rényi (random) graphs, m=3 ≤ 40%
=======================================================================
Mirrors the power-law benchmark (phase5_task1) but on ER G(N, 0.3) graphs.

Paper target (Sang et al., Table IV — ER graphs):
  ARG (Approximation Ratio Gap) = 100 × |E_min − ⟨H_C⟩| / |E_min|
    m=3 → ARG ≈ 26% median across N=4..20  (Table IV "random graph" row)

Assertion: median ARG ≤ 40% for m=3 on ER G(N, 0.3) graphs N=4..20.

Run as:  python acceptance_c5_er_arg_m3.py
"""

import math
import sys
import time
import statistics
import importlib.util

import numpy as np
import pennylane as qml
import networkx as nx

spec = importlib.util.spec_from_file_location(
    "doqaoa", "catalyst/frontend/catalyst/api_extensions/doqaoa.py"
)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)

do_qaoa                  = _mod.do_qaoa
DOQAOAResult             = _mod.DOQAOAResult
select_hotspot_indices   = _mod.select_hotspot_indices
extract_coupling_matrix  = _mod.extract_coupling_matrix

N_VALUES     = list(range(4, 21))
M            = 3
P_EDGE       = 0.3
SEED         = 42
ARG_ASSERT   = 40          # %
SHOTS_BUDGET = 170_000     # aggregate for m=3 across N=4..20


def emin_subproblem(cost_h, hotspot_indices, N, k_idx):
    J_dict, h_dict = extract_coupling_matrix(cost_h)
    J_mat = np.zeros((N, N))
    for (i, j), v in J_dict.items():
        J_mat[i, j] = v; J_mat[j, i] = v
    h_vec = np.array([h_dict.get(i, 0.0) for i in range(N)])

    hs = list(hotspot_indices)
    m  = len(hs)
    frozen = {hs[i]: (-1 if (k_idx >> i) & 1 else 1) for i in range(m)}
    free   = [q for q in range(N) if q not in frozen]
    nf     = len(free)

    h_eff  = np.array([h_vec[q] + sum(J_mat[q, fq] * frozen[fq] for fq in frozen)
                       for q in free])
    J_free = np.array([[J_mat[free[i], free[j]] for j in range(nf)]
                       for i in range(nf)])
    best = float("inf")
    for bits in range(1 << nf):
        spins = np.array([1 - 2 * ((bits >> i) & 1) for i in range(nf)], dtype=float)
        e  = sum(J_free[i, j] * spins[i] * spins[j]
                 for i in range(nf) for j in range(i + 1, nf))
        e += float(h_eff @ spins)
        if e < best:
            best = e
    return best


def _connected_er_graph(N, p, seed):
    """Return a connected ER graph, bumping p if needed."""
    for attempt in range(20):
        G = nx.erdos_renyi_graph(N, p, seed=seed + attempt)
        if nx.is_connected(G):
            return G
    return nx.erdos_renyi_graph(N, max(p + 0.3, 0.9), seed=seed)


print("=" * 70)
print("Acceptance Criterion 5 — ARG on Erdős-Rényi G(N, 0.3) graphs, m=3")
print(f"N ∈ {{{N_VALUES[0]}..{N_VALUES[-1]}}},  p={P_EDGE},  target median ARG ≤ {ARG_ASSERT}%")
print("=" * 70)

args        = []
total_shots = 0
t0          = time.perf_counter()

print(f"\n  {'N':>3}  {'edges':>5}  {'shots':>6}  {'⟨H_C⟩':>8}  {'E_min':>8}  {'ARG%':>6}  status")
print(f"  {'─'*3}  {'─'*5}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*6}")

for N in N_VALUES:
    if M >= N:
        print(f"  {N:>3}  {'n/a':>5}  {'─':>6}  {'─':>8}  {'─':>8}  {'─':>6}  SKIP (m≥N)")
        continue

    G       = _connected_er_graph(N, P_EDGE, SEED + N)
    cost_h, mixer_h = qml.qaoa.maxcut(G)
    dev     = qml.device("default.qubit", wires=N)

    @qml.qnode(dev)
    def circuit(params):
        for w in range(N):
            qml.Hadamard(wires=w)
        qml.qaoa.cost_layer(params[0], cost_h)
        qml.qaoa.mixer_layer(params[1], mixer_h)
        return qml.expval(cost_h)

    res = do_qaoa(
        circuit, cost_h,
        m=M,
        full_epochs=100,
        warmstart_epochs=10,
        learning_rate=0.01,
        grad_norm_tol=1e-4,
        seed=SEED,
        max_warmstarts=1,
        bias_threshold=0.3,
    )(G)

    hs   = list(select_hotspot_indices(G, M))
    emin = emin_subproblem(cost_h, hs, N, res.best_k)
    arg  = 100 * abs(emin - res.best_energy) / abs(emin) if abs(emin) > 1e-9 else 0.0

    args.append(arg)
    total_shots += res.total_shots

    ok     = res.best_energy < 0
    status = "ok" if ok else "FAIL"
    print(
        f"  {N:>3}  {G.number_of_edges():>5}  {res.total_shots:>6,}  "
        f"{res.best_energy:>8.4f}  {emin:>8.4f}  {arg:>6.1f}  {status}"
    )

elapsed = time.perf_counter() - t0
med  = statistics.median(args)
mean = sum(args) / len(args)

print(f"\n{'='*70}")
print(f"  Wall-clock: {elapsed:.1f}s")
print(f"  Median ARG = {med:.1f}%   Mean = {mean:.1f}%")
print(f"  Aggregate shots (m=3): {total_shots:,}")
print()

passed = True

if med <= ARG_ASSERT:
    print(f"PASS: median ARG {med:.1f}% ≤ {ARG_ASSERT}%  ✓")
else:
    print(f"FAIL: median ARG {med:.1f}% > {ARG_ASSERT}%  ✗")
    passed = False

if total_shots <= SHOTS_BUDGET:
    print(f"PASS: aggregate shots {total_shots:,} ≤ {SHOTS_BUDGET:,}  ✓")
else:
    print(f"FAIL: aggregate shots {total_shots:,} > {SHOTS_BUDGET:,}  ✗")
    passed = False

print()
if passed:
    print("ACCEPTANCE CRITERION 5 PASS ✓  — ER ARG m=3 within budget")
    print(f"  Median ARG: {med:.1f}%  (target ≤ {ARG_ASSERT}%)")
    print(f"  Total shots: {total_shots:,}")
else:
    print("ACCEPTANCE CRITERION 5 FAIL ✗  — see errors above")
    sys.exit(1)
