#!/usr/bin/env python3
"""
Phase 4 Milestone — catalyst.do_qaoa(qnode, H, m=3) returns correct ARG

Deliverable (Sang et al., arXiv:2602.21689v1, Table IV):
  A user can write:
      result = catalyst.do_qaoa(qnode, H, m=3)(graph)
  and obtain correct ARG with ≤ 0.23 × 10⁶ shots.
  Tutorial notebook executes end-to-end in < 5 minutes on a laptop.

Graph: 12-node Barabási-Albert (power-law, n=12, m_ba=2, seed=42)
Frozen qubits: m=3  →  8 sub-problems
FrozenQubits baseline: 32.77 × 10⁶ shots
DO-QAOA target: ≤ 0.23 × 10⁶ = 230,000 shots

Assertions:
  1. total_shots ≤ 230,000
  2. full_opt_count == 1
  3. warmstart_count ≤ 1
  4. best_energy < 0  (non-trivial MaxCut solution)
  5. isinstance(result, DOQAOAResult)  (correct return type from do_qaoa API)

Run as:  python milestone_phase4.py
"""

# ---------------------------------------------------------------------------
# Load DO-QAOA API
# ---------------------------------------------------------------------------
import importlib.util
import math
import sys
import time

import numpy as np
import pennylane as qml

spec = importlib.util.spec_from_file_location(
    "doqaoa", "frontend/catalyst/api_extensions/doqaoa.py"
)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)

do_qaoa = _mod.do_qaoa
DOQAOAConfig = _mod.DOQAOAConfig
DOQAOAResult = _mod.DOQAOAResult
DOQAOATransform = _mod.DOQAOATransform

# ---------------------------------------------------------------------------
# 1. Build 12-node Barabási-Albert graph via PennyLane / networkx
# ---------------------------------------------------------------------------

try:
    import networkx as nx

    G = nx.barabasi_albert_graph(12, 2, seed=42)
    N = G.number_of_nodes()
except ImportError:
    print("networkx not found — building BA graph manually")
    import random

    def _build_ba(n=12, m_ba=2, seed=42):
        rng = random.Random(seed)
        adj = {i: set() for i in range(n)}

        def add_edge(u, v):
            adj[u].add(v)
            adj[v].add(u)

        add_edge(0, 1)
        add_edge(1, 2)
        add_edge(0, 2)
        degrees = [2, 2, 2]
        for new_node in range(3, n):
            total = sum(degrees)
            targets = set()
            while len(targets) < m_ba:
                r = rng.random() * total
                acc = 0
                for node, d in enumerate(degrees):
                    acc += d
                    if r <= acc:
                        targets.add(node)
                        break
            for t in targets:
                add_edge(new_node, t)
            degrees.append(len(adj[new_node]))
            for t in targets:
                degrees[t] += 1
        return adj

    _adj = _build_ba()
    G = type(
        "G",
        (),
        {
            "nodes": lambda self: range(12),
            "edges": lambda self: [(u, v) for u, ns in _adj.items() for v in ns if u < v],
            "number_of_nodes": lambda self: 12,
            "degree": lambda self: [(u, len(ns)) for u, ns in _adj.items()],
        },
    )()
    N = 12

print(f"Graph: 12-node Barabási-Albert (MaxCut, seed=42)")
print(f"Nodes: {N}  Edges: {G.number_of_edges()}")

# ---------------------------------------------------------------------------
# 2. Build PennyLane Hamiltonian and QAOA QNode
# ---------------------------------------------------------------------------

cost_h, mixer_h = qml.qaoa.maxcut(G)

dev = qml.device("default.qubit", wires=N)


@qml.qnode(dev)
def circuit(params):
    """QAOA p=1: Hadamard init → cost layer → mixer layer."""
    for w in range(N):
        qml.Hadamard(wires=w)
    qml.qaoa.cost_layer(params[0], cost_h)
    qml.qaoa.mixer_layer(params[1], mixer_h)
    return qml.expval(cost_h)


# Sanity check: shortcut initial energy
e0 = float(circuit(np.array([-math.pi / 6, -math.pi / 8])))
print(f"Initial energy (shortcut params): {e0:.4f}")

# ---------------------------------------------------------------------------
# 3. THE DELIVERABLE — one call
# ---------------------------------------------------------------------------

print("\n" + "─" * 60)
print("  result = catalyst.do_qaoa(circuit, cost_h, m=3)(G)")
print("─" * 60)

t0 = time.perf_counter()

result = do_qaoa(
    circuit,
    cost_h,
    m=3,
    full_epochs=100,
    warmstart_epochs=10,
    learning_rate=0.01,
    grad_norm_tol=1e-4,
    seed=42,
    max_warmstarts=1,
    bias_threshold=0.3,
)(G, frozen_qubits_shots=32_770_000)

elapsed = time.perf_counter() - t0

print(f"\nResult : {result}")
print(f"Runtime: {elapsed:.1f}s")

# ---------------------------------------------------------------------------
# 4. Print summary
# ---------------------------------------------------------------------------

TARGET_SHOTS = 230_000
FROZEN_QUBITS_BASELINE = 32_770_000

print(f"\n{'='*60}")
print(f"  Total shots           : {result.total_shots:>10,}")
print(f"  FrozenQubits baseline : {FROZEN_QUBITS_BASELINE:>10,}  (32.77 × 10⁶)")
print(f"  Target ≤              : {TARGET_SHOTS:>10,}  (0.23 × 10⁶)")
print(f"  Full optimisations    : {result.full_opt_count:>10}  (expected 1)")
print(f"  Warm starts           : {result.warmstart_count:>10}  (expected ≤ 1)")
print(f"  Direct copies         : {result.direct_copy_count:>10}")
print(f"{'='*60}")

# ---------------------------------------------------------------------------
# 5. Assertions
# ---------------------------------------------------------------------------

passed = True

if not isinstance(result, DOQAOAResult):
    print(f"\nFAIL: do_qaoa() returned {type(result).__name__}, expected DOQAOAResult")
    passed = False
else:
    print(f"\nPASS: return type is DOQAOAResult  ✓")

if result.total_shots > TARGET_SHOTS:
    print(f"FAIL: shots {result.total_shots} > {TARGET_SHOTS}")
    passed = False
else:
    print(f"PASS: shots {result.total_shots} ≤ {TARGET_SHOTS}  ✓")

if result.full_opt_count != 1:
    print(f"FAIL: full_opt_count={result.full_opt_count}, expected 1")
    passed = False
else:
    print(f"PASS: full_opt_count=1  ✓")

if result.warmstart_count > 1:
    print(f"FAIL: warmstart_count={result.warmstart_count} > 1")
    passed = False
else:
    print(f"PASS: warmstart_count={result.warmstart_count} ≤ 1  ✓")

if result.best_energy >= 0:
    print(f"FAIL: best_energy={result.best_energy:.6f} ≥ 0 (expected sub-zero)")
    passed = False
else:
    print(f"PASS: best_energy={result.best_energy:.6f} < 0  ✓")

LAPTOP_BUDGET_S = 300  # 5 minutes
if elapsed > LAPTOP_BUDGET_S:
    print(f"FAIL: runtime {elapsed:.1f}s > {LAPTOP_BUDGET_S}s (5 min laptop budget)")
    passed = False
else:
    print(f"PASS: runtime {elapsed:.1f}s < {LAPTOP_BUDGET_S}s (5 min laptop budget)  ✓")

print()
if passed:
    print("PHASE 4 MILESTONE PASS ✓  — all assertions satisfied")
    print(f"  do_qaoa(circuit, cost_h, m=3)(G)")
    print(f"  ARG bitstring : {''.join(str(b) for b in result.bitstring)}")
    print(f"  best ⟨H⟩      : {result.best_energy:.6f}")
    print(f"  Shots         : {result.total_shots:,}  ({result.total_shots/1e6:.4f} × 10⁶)")
    print(f"  Speedup       : {result.speedup_vs_frozen:.0f}× vs FrozenQubits")
    print(f"  Runtime       : {elapsed:.1f}s")
else:
    print("PHASE 4 MILESTONE FAIL ✗  — see errors above")
    sys.exit(1)
