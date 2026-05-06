#!/usr/bin/env python3
"""
Phase 5 Task 4 — Shot count regression tests (CI pipeline)
===========================================================
Automated test asserting:
  - DO-QAOA (m=3) total_shots ≤ 0.25 × 10⁶ = 250,000
  - FrozenQubits baseline = 65.54 × 10⁶ shots  (2^m × baseline_per_run)
  - Minimum speedup ≥ 262× (= 65,540,000 / 250,000)

Regression matrix (Table IV):
  Graph             | DO-QAOA  | FrozenQubits | Speedup
  12-node BA m=3    |  ≤ 230k  | 32.77M       | ≥ 142×
  12-node BA m=3    |  actual  | -            | -

This test is designed to be added to the Catalyst CI pipeline as a
pytest module. It also runs standalone.

Exit codes: 0 = all pass, 1 = any failure (CI-compatible).

Run as:
  python phase5_task4_shot_regression.py
  pytest phase5_task4_shot_regression.py -v
"""

import sys
import math
import importlib.util
import time

import numpy as np
import pennylane as qml
import networkx as nx

# ── Load DO-QAOA API ──────────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location(
    "doqaoa", "frontend/catalyst/api_extensions/doqaoa.py"
)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)

do_qaoa = _mod.do_qaoa
DOQAOAResult = _mod.DOQAOAResult

# ── Regression constants ──────────────────────────────────────────────────────
# These are the HARD LIMITS that must not regress in CI.
MAX_SHOTS_M1 = 170_000  # ≤ 0.17 × 10⁶  (Table IV)
MAX_SHOTS_M2 = 170_000  # ≤ 0.17 × 10⁶
MAX_SHOTS_M3 = 250_000  # ≤ 0.25 × 10⁶  (conservative ≥ 0.23M target)
FROZEN_BASELINE_M3 = 65_540_000  # 65.54 × 10⁶ (from Table IV)
MIN_SPEEDUP_M3 = FROZEN_BASELINE_M3 // MAX_SHOTS_M3  # 262×

# ── Test helpers ──────────────────────────────────────────────────────────────


def make_ba_circuit_and_hamiltonian(N: int, m_ba: int = 2, seed: int = 42):
    """Build BA graph, cost Hamiltonian, and QAOA QNode."""
    G = nx.barabasi_albert_graph(N, m_ba, seed=seed)
    cost_h, mixer_h = qml.qaoa.maxcut(G)
    dev = qml.device("default.qubit", wires=N)

    @qml.qnode(dev)
    def circuit(params):
        for w in range(N):
            qml.Hadamard(wires=w)
        qml.qaoa.cost_layer(params[0], cost_h)
        qml.qaoa.mixer_layer(params[1], mixer_h)
        return qml.expval(cost_h)

    return G, cost_h, circuit


def run_do_qaoa(circuit, cost_h, m: int, G):
    """Run do_qaoa and return DOQAOAResult."""
    return do_qaoa(
        circuit,
        cost_h,
        m=m,
        full_epochs=100,
        warmstart_epochs=10,
        learning_rate=0.01,
        grad_norm_tol=1e-4,
        seed=42,
        max_warmstarts=1,
        bias_threshold=0.3,
    )(G)


# ── pytest-compatible test functions ─────────────────────────────────────────


def test_shot_budget_m1_12node():
    """DO-QAOA m=1 on 12-node BA graph: shots ≤ 170,000."""
    G, cost_h, circuit = make_ba_circuit_and_hamiltonian(12, seed=42)
    res = run_do_qaoa(circuit, cost_h, m=1, G=G)
    assert isinstance(res, DOQAOAResult), f"Expected DOQAOAResult, got {type(res)}"
    assert (
        res.total_shots <= MAX_SHOTS_M1
    ), f"m=1 shots {res.total_shots:,} > limit {MAX_SHOTS_M1:,}"
    assert res.best_energy < 0, f"Energy {res.best_energy:.4f} ≥ 0"
    assert res.full_opt_count == 1, f"full_opt_count = {res.full_opt_count} ≠ 1"


def test_shot_budget_m2_12node():
    """DO-QAOA m=2 on 12-node BA graph: shots ≤ 170,000."""
    G, cost_h, circuit = make_ba_circuit_and_hamiltonian(12, seed=42)
    res = run_do_qaoa(circuit, cost_h, m=2, G=G)
    assert isinstance(res, DOQAOAResult)
    assert (
        res.total_shots <= MAX_SHOTS_M2
    ), f"m=2 shots {res.total_shots:,} > limit {MAX_SHOTS_M2:,}"
    assert res.best_energy < 0
    assert res.full_opt_count == 1


def test_shot_budget_m3_12node():
    """DO-QAOA m=3 on 12-node BA graph: shots ≤ 250,000. (Primary regression)"""
    G, cost_h, circuit = make_ba_circuit_and_hamiltonian(12, seed=42)
    res = run_do_qaoa(circuit, cost_h, m=3, G=G)
    assert isinstance(res, DOQAOAResult)
    assert (
        res.total_shots <= MAX_SHOTS_M3
    ), f"m=3 shots {res.total_shots:,} > limit {MAX_SHOTS_M3:,}"
    assert res.best_energy < 0
    assert res.full_opt_count == 1
    assert res.warmstart_count <= 1


def test_speedup_vs_frozen_m3():
    """DO-QAOA m=3 speedup vs FrozenQubits ≥ 262×."""
    G, cost_h, circuit = make_ba_circuit_and_hamiltonian(12, seed=42)
    res = run_do_qaoa(circuit, cost_h, m=3, G=G)
    actual_speedup = FROZEN_BASELINE_M3 / res.total_shots
    assert (
        actual_speedup >= MIN_SPEEDUP_M3
    ), f"Speedup {actual_speedup:.0f}× < minimum {MIN_SPEEDUP_M3}×"


def test_return_type():
    """do_qaoa() must return DOQAOAResult."""
    G, cost_h, circuit = make_ba_circuit_and_hamiltonian(8, seed=42)
    res = run_do_qaoa(circuit, cost_h, m=2, G=G)
    assert isinstance(res, DOQAOAResult), f"Wrong type: {type(res)}"


def test_warmstart_cap():
    """warm_start_count ≤ max_warmstarts=1 always."""
    G, cost_h, circuit = make_ba_circuit_and_hamiltonian(12, seed=42)
    res = run_do_qaoa(circuit, cost_h, m=3, G=G)
    assert res.warmstart_count <= 1, f"warmstart_count={res.warmstart_count} > 1"


def test_sweep_ba_graphs_m3():
    """DO-QAOA m=3 on BA graphs N=6..16: all shots ≤ 250,000."""
    for N in range(6, 17):
        G, cost_h, circuit = make_ba_circuit_and_hamiltonian(N, seed=42 + N)
        res = run_do_qaoa(circuit, cost_h, m=3, G=G)
        assert (
            res.total_shots <= MAX_SHOTS_M3
        ), f"N={N}: shots {res.total_shots:,} > {MAX_SHOTS_M3:,}"
        assert res.best_energy < 0, f"N={N}: energy {res.best_energy:.4f} ≥ 0"


# ── Standalone runner ─────────────────────────────────────────────────────────

TESTS = [
    ("test_return_type", test_return_type),
    ("test_shot_budget_m1_12node", test_shot_budget_m1_12node),
    ("test_shot_budget_m2_12node", test_shot_budget_m2_12node),
    ("test_shot_budget_m3_12node", test_shot_budget_m3_12node),
    ("test_speedup_vs_frozen_m3", test_speedup_vs_frozen_m3),
    ("test_warmstart_cap", test_warmstart_cap),
    ("test_sweep_ba_graphs_m3", test_sweep_ba_graphs_m3),
]

if __name__ == "__main__":
    print("=" * 70)
    print("Phase 5 Task 4 — Shot Count Regression Tests (CI pipeline)")
    print(f"Limits: m=1/2 ≤ {MAX_SHOTS_M1:,}, m=3 ≤ {MAX_SHOTS_M3:,} shots")
    print(f"FrozenQubits baseline: {FROZEN_BASELINE_M3:,} | Min speedup: {MIN_SPEEDUP_M3}×")
    print("=" * 70)

    passed_all = True

    for name, fn in TESTS:
        t0 = time.perf_counter()
        try:
            fn()
            elapsed = time.perf_counter() - t0
            print(f"  PASS ✓  {name}  ({elapsed:.2f}s)")
        except AssertionError as e:
            elapsed = time.perf_counter() - t0
            print(f"  FAIL ✗  {name}  ({elapsed:.2f}s)")
            print(f"         → {e}")
            passed_all = False
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"  ERROR   {name}  ({elapsed:.2f}s)")
            print(f"         → {e}")
            passed_all = False

    print(f"\n{'='*70}")
    if passed_all:
        print("PHASE 5 TASK 4 PASS ✓  — All regression tests satisfied")
    else:
        print("PHASE 5 TASK 4 FAIL ✗  — see failures above")
        sys.exit(1)
