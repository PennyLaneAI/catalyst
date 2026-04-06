#!/usr/bin/env python3
"""
Phase 1 Milestone: 4-qubit MaxCut DO-QAOA round-trip.

Uses doqaoa_partition() + hamiltonian_to_graph_attrs() to generate a
compilable MLIR module containing quantum.freeze_partition on a toy
4-qubit MaxCut circuit, then verifies it round-trips through quantum-opt.

Circuit: 4-node cycle graph  (0-1-2-3-0)
Hamiltonian: H = -0.5*(Z0Z1 + Z1Z2 + Z2Z3 + Z3Z0)
Hotspot qubits: m=2 (nodes 0, 1 by degree centrality)
"""

import importlib.util
import pathlib
import subprocess
import sys
import types
import tempfile
import os

# ── Minimal catalyst stub so doqaoa.py loads standalone ──────────────────────
cat_stub = types.ModuleType("catalyst")
cat_stub.api_extensions = types.ModuleType("catalyst.api_extensions")
sys.modules["catalyst"] = cat_stub
sys.modules["catalyst.api_extensions"] = cat_stub.api_extensions

spec = importlib.util.spec_from_file_location(
    "doqaoa",
    pathlib.Path(__file__).parent /
    "catalyst/frontend/catalyst/api_extensions/doqaoa.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DOQAOAConfig           = mod.DOQAOAConfig
doqaoa_partition       = mod.doqaoa_partition
hamiltonian_to_graph_attrs = mod.hamiltonian_to_graph_attrs
select_hotspot_indices = mod.select_hotspot_indices
compute_bias           = mod.compute_bias

import networkx as nx
import pennylane as qml
import numpy as np

# ── 1. Graph and Hamiltonian ──────────────────────────────────────────────────
G = nx.cycle_graph(4)          # 4-node cycle: edges (0,1),(1,2),(2,3),(3,0)
NUM_QUBITS = 4

cost_h = qml.Hamiltonian(
    [-0.5, -0.5, -0.5, -0.5],
    [
        qml.PauliZ(0) @ qml.PauliZ(1),
        qml.PauliZ(1) @ qml.PauliZ(2),
        qml.PauliZ(2) @ qml.PauliZ(3),
        qml.PauliZ(3) @ qml.PauliZ(0),
    ]
)

# ── 2. DO-QAOA config and decorator ─────────────────────────────────────────
config = DOQAOAConfig(m=2, bias_threshold=0.3, init_strategy="shortcut")

dev = qml.device("default.qubit", wires=NUM_QUBITS)

@doqaoa_partition(graph=G, config=config)
@qml.qnode(dev)
def maxcut_circuit(params):
    # QAOA p=1 ansatz
    for i, j in G.edges():
        qml.CNOT(wires=[i, j])
        qml.RZ(params[0], wires=j)
        qml.CNOT(wires=[i, j])
    for i in range(NUM_QUBITS):
        qml.RX(params[1], wires=i)
    return qml.expval(cost_h)

hotspots   = maxcut_circuit.hotspot_indices   # [0, 1]
B_rep      = compute_bias(cost_h, NUM_QUBITS) # 0.0 (pure ZZ)
B_target   = B_rep                            # same sub-problem → direct copy

# ── 3. Hamiltonian → MLIR attribute strings ──────────────────────────────────
h_quad_attr, h_lin_attr = hamiltonian_to_graph_attrs(cost_h, NUM_QUBITS)

# ── 4. Generate MLIR module text ─────────────────────────────────────────────
hotspot_arr = ", ".join(str(i) for i in hotspots)
m = config.m

mlir_module = f"""\
// Phase 1 Milestone — 4-qubit MaxCut DO-QAOA circuit
// Graph : 4-node cycle  (0-1-2-3-0)
// H     : -0.5*(Z0Z1 + Z1Z2 + Z2Z3 + Z3Z0)
// m     : {m} frozen hotspot qubits  →  2^{m} = {2**m} sub-problems
// K     : 1 landscape cluster  (sparse cycle graph, s > sc ≈ 0.6)
// B_rep : {B_rep:.4f}  (pure-ZZ Hamiltonian, no linear bias)
//
// Python decorator selected hotspot qubits: {hotspots}
// H_quad: {h_quad_attr[:70]}...
// H_lin : {h_lin_attr}

module {{

  // ----------------------------------------------------------------
  // Full DO-QAOA pipeline on the 4-qubit MaxCut circuit.
  // Input : %params_rep — optimised parameters for the representative
  //         sub-circuit (theta = [gamma_1, beta_1] at p=1).
  // Output: (implicit) %bitstring — best node assignment found.
  // ----------------------------------------------------------------
  func.func @maxcut_4qubit_doqaoa(%params_rep: !quantum.params) {{

    // Step 1: Annotate frozen qubits and embed graph metadata.
    //   hotspot_count = m = {m}
    //   hotspot_indices = {hotspots}  (highest degree-centrality nodes)
    //   h_quad = J_ij coupling matrix (MaxCut weights)
    //   h_lin  = h_i bias vector (all zero for pure-ZZ)
    %partition = quantum.freeze_partition {{
        hotspot_count    = {m} : i32,
        hotspot_indices  = array<i32: {hotspot_arr}>,
        h_quad           = {h_quad_attr},
        h_lin            = {h_lin_attr}
    }} : !quantum.partition<{NUM_QUBITS}, {m}>

    // Step 2: Cluster 2^m={2**m} sub-problem landscapes.
    //   K=1 because the 4-cycle is sparse (s > sc ≈ 0.6) — all
    //   sub-problems share the same landscape shape.
    %cluster_map = quantum.landscape_cluster(
        %partition : !quantum.partition<{NUM_QUBITS}, {m}>)
        {{k = 1 : i32}} : !quantum.cluster_map<1>

    // Step 3: Pick the representative sub-circuit for the single cluster.
    %circuit_ref = quantum.select_representative(
        %cluster_map : !quantum.cluster_map<1>)
        : !quantum.circuit_ref

    // Step 4: Apply Bias-Aware Transfer Rule.
    //   |B_target - B_rep| = |{B_target:.4f} - {B_rep:.4f}| = 0.0 < threshold=0.3
    //   → direct parameter copy, zero extra training sessions.
    %params_out = quantum.bias_transfer(
        %params_rep : !quantum.params)
        {{B_rep = {B_rep:.6e} : f64,
          B_target = {B_target:.6e} : f64,
          threshold = 3.000000e-01 : f64}}
        : !quantum.params

    // Step 5: Select the sub-circuit bitstring with minimum <H>.
    %bitstring = quantum.aggregate_min(
        %params_out : !quantum.params)
        : !quantum.bitstring

    func.return
  }}

}}
"""

# ── 5. Write to temp file and round-trip through quantum-opt ─────────────────
QUANTUM_OPT = (
    pathlib.Path(__file__).parent /
    "catalyst/mlir/build/bin/quantum-opt"
)

out_path = pathlib.Path(__file__).parent / "MaxCut4QubitMilestone.mlir"
out_path.write_text(mlir_module)
print(f"Generated: {out_path}")
print()

result = subprocess.run(
    [str(QUANTUM_OPT), str(out_path)],
    capture_output=True, text=True
)

if result.returncode != 0:
    print("=== ROUND-TRIP FAILED ===")
    print(result.stderr)
    sys.exit(1)

print("=== quantum-opt round-trip output ===")
print(result.stdout)
print("=== MILESTONE PASS ===")
